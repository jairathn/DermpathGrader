"""Melanocytic grading call (protocol v2.0).

Changes from v1
---------------
1. **Melanoma is a possible answer.** v1 could only emit mild / moderate /
   severe dysplasia, so a melanoma in the input had nowhere to go and was
   forced onto the top of the dysplasia ladder. The label space is now
   four-way (mild, moderate, severe, melanoma), and melanoma carries its
   own subtype and Breslow estimate rather than being "severe plus".

2. **MPATH-Dx v2.0 replaces the old two-tier field.** v1 emitted
   `mpath_grade` of "Low-Grade" / "High-Grade Dysplasia" and told the
   model low-grade meant mild and high-grade meant moderate-to-severe.
   That is not MPATH-Dx v1.0 (which had five classes) and it is not
   v2.0. The field is now `mpath_dx_v2_class`, one of 0/I/II/III/IV,
   defined from the consensus statement in mpath_dx.py.

3. **Four images instead of one**, each captioned with its magnification,
   plus a `magnification_evidence` field so the response records which
   power a finding came from. A grade attributed to 40x that only appears
   at 4x is a detectable inconsistency; v1 could not detect anything.

4. **Structured outputs replace the keyword fallback.** v1's parser had a
   three-layer fallback ending in keyword inference that returned
   "Moderate Dysplasia" / "High-Grade" when everything failed - a silent
   default indistinguishable from a real answer in the logs (CLAUDE.md
   landmine 4). The API now enforces the schema, so a malformed response
   is an error rather than a fabricated grade. `parse_strategy` is still
   logged, and `schema_enforced` records that the guarantee was in play.

5. **max_tokens raised to config.MAX_TOKENS** from 1500, against a larger
   schema (landmine 3), and the call is streamed.

6. **temperature removed.** This model family rejects sampling
   parameters; v1's temperature=0.1 would now be a 400.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from anthropic import Anthropic

import config
import image_utils
import mpath_dx


NEVUS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "lesion_category": {
            "type": "string",
            "enum": ["dysplastic_nevus", "melanoma",
                     "benign_nevus_no_atypia", "nondiagnostic"],
        },
        "dysplasia_grade": {
            "type": "string",
            "enum": ["mild", "moderate", "severe", "not_applicable"],
        },
        "melanoma_subtype": {
            "type": "string",
            "enum": ["in_situ", "invasive", "not_applicable"],
        },
        "breslow_estimate_mm": {"type": ["number", "null"]},
        "stratum_label": {
            "type": "string",
            "enum": ["mild", "moderate", "severe", "melanoma"],
        },
        "mpath_dx_v2_class": {
            "type": "string",
            "enum": list(mpath_dx.CLASSES),
        },
        "confidence_level": {"type": "string",
                             "enum": ["High", "Medium", "Low"]},
        "architectural_features": {"type": "array", "items": {"type": "string"}},
        "cytological_features": {"type": "array", "items": {"type": "string"}},
        "magnification_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "magnification": {"type": "string",
                                      "enum": list(config.MAGNIFICATIONS)},
                    "finding": {"type": "string"},
                },
                "required": ["magnification", "finding"],
                "additionalProperties": False,
            },
        },
        "grading_rationale": {"type": "string"},
        "clinical_significance": {"type": "string"},
    },
    "required": [
        "lesion_category", "dysplasia_grade", "melanoma_subtype",
        "breslow_estimate_mm", "stratum_label", "mpath_dx_v2_class",
        "confidence_level", "architectural_features", "cytological_features",
        "magnification_evidence", "grading_rationale", "clinical_significance",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA_FIELDS = list(NEVUS_OUTPUT_SCHEMA["properties"].keys())


class NeviAnalyzer:
    """Melanocytic lesion grader: dysplasia grade plus melanoma."""

    def __init__(self, nevi_rag_system):
        self.nevi_rag_system = nevi_rag_system
        self.setup_claude()

    def setup_claude(self) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)
        self.model = config.MODEL_ID

    # ── prompt ───────────────────────────────────────────────────────

    def create_analysis_prompt(self, context: str) -> str:
        mags = ", ".join(config.MAGNIFICATIONS)
        return f"""You are an expert dermatopathologist grading a melanocytic lesion.

You have been given {len(config.MAGNIFICATIONS)} images of the SAME lesion at
different magnifications, in this order: {mags}. Read them together the way you
would read a slide: establish architecture and symmetry at low power, then
confirm cytology at high power. Do not grade from one image alone.

Retrieved literature context:
{context}

### WHAT YOU ARE DECIDING

First decide what the lesion IS, then grade it:

- If it is a dysplastic (atypical) nevus, assign a dysplasia grade of mild,
  moderate or severe.
- If it is melanoma, say so. Do not report melanoma as "severe dysplasia".
  Melanoma is a different diagnosis, not the top of the dysplasia scale.
  State whether it is in situ or invasive, and for invasive melanoma give
  your best Breslow thickness estimate in millimetres.
- If it is a banal nevus with no meaningful atypia, use
  benign_nevus_no_atypia and set the dysplasia grade to mild.
- If the material will not support a diagnosis, use nondiagnostic.

### DYSPLASIA GRADING CRITERIA (for dysplastic nevi)

- Mild: nuclei about the size of resting basal keratinocyte nuclei, minimal
  atypia, symmetric, orderly maturation with descent.
- Moderate: nuclei about 1.5x basal keratinocyte nuclei, some pleomorphism
  and chromatin clumping, well-developed nests, bridging may be present.
- Severe: nuclei at or above 2x basal keratinocyte nuclei, three or more
  nuclear abnormalities, or any high-risk architectural feature (confluent
  junctional hyperplasia, pagetoid spread, epidermal mitoses).

### FEATURES THAT SHOULD MAKE YOU CONSIDER MELANOMA RATHER THAN DYSPLASIA

Asymmetry and poor circumscription at low power; confluent sheets of
melanocytes; prominent pagetoid spread above the basal layer, especially at
the periphery; absent maturation with descent; dermal mitoses; single-cell
predominance over nests; necrosis; and ulceration. Weigh these at the
magnification where they are actually assessable rather than assuming them.

### {mpath_dx.SCHEMA_VERSION}

{mpath_dx.prompt_block()}

Assign the MPATH-Dx class from the criteria above, independently of the word
you chose for the dysplasia grade. If you grade the lesion as moderate
dysplasia, decide Class I versus Class II on nuclear size relative to resting
basal keratinocytes and the other cytologic criteria.

### RULES

1. `stratum_label` is the single four-way study label: mild, moderate,
   severe, or melanoma. For any melanoma, in situ or invasive, it is
   "melanoma".
2. `dysplasia_grade` is "not_applicable" when lesion_category is melanoma.
3. `melanoma_subtype` is "not_applicable" when the lesion is not melanoma,
   and `breslow_estimate_mm` is null unless the melanoma is invasive.
4. In `magnification_evidence`, cite at least one finding per magnification,
   naming only what that power can actually show.
5. Grade what is in front of you. Do not hedge to the middle category to
   avoid committing."""

    # ── call ─────────────────────────────────────────────────────────

    def analyze_images(self, prepared_images: list, case_logger=None
                       ) -> dict[str, Any]:
        """Grade one case from its prepared multi-magnification images."""
        context = self.nevi_rag_system.get_grading_criteria()
        prompt = self.create_analysis_prompt(context)

        content = image_utils.build_image_content_blocks(prepared_images)
        content.append({"type": "text", "text": prompt})

        started = time.time()
        with self.client.messages.stream(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            thinking=config.THINKING,
            output_config={
                "effort": config.EFFORT,
                "format": {"type": "json_schema",
                           "schema": NEVUS_OUTPUT_SCHEMA},
            },
            messages=[{"role": "user", "content": content}],
        ) as stream:
            response = stream.get_final_message()
        latency_ms = int((time.time() - started) * 1000)

        analysis_text = next(
            (b.text for b in response.content if b.type == "text"), "")
        result = self.parse_analysis_response(analysis_text)
        result["raw_analysis"] = analysis_text
        result["context_used"] = bool(context.strip())

        if case_logger is not None:
            self._log(case_logger, prepared_images, prompt, context,
                      response, analysis_text, latency_ms, result)
        return result

    def _log(self, case_logger, prepared_images, prompt, context,
             response, analysis_text, latency_ms, result) -> None:
        case_logger.set_request(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            thinking=config.THINKING,
            effort=config.EFFORT,
            system_text="",
            user_text=prompt,
            message_structure=image_utils.message_structure(
                prepared_images,
                hashlib.sha256(prompt.encode()).hexdigest()),
            schema_enforced=True,
            output_schema_sha256=hashlib.sha256(
                json.dumps(NEVUS_OUTPUT_SCHEMA, sort_keys=True).encode()
            ).hexdigest(),
        )
        case_logger.set_response(
            api_request_id=getattr(response, "id", ""),
            model_returned=getattr(response, "model", ""),
            stop_reason=getattr(response, "stop_reason", "") or "",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
            raw_text=analysis_text,
        )
        case_logger.set_parsing(
            strategy_used=self._last_parse_strategy,
            fallback_invoked=(self._last_parse_strategy != "structured_output"),
            parse_errors=self._last_parse_errors,
            parsed={k: result.get(k) for k in OUTPUT_SCHEMA_FIELDS},
        )
        if hasattr(self.nevi_rag_system, "_retrieval_log"):
            rlog = self.nevi_rag_system._retrieval_log
            case_logger.set_retrieval(
                subqueries_issued=list(dict.fromkeys(r["query"] for r in rlog)),
                retrieved_chunk_ids_in_order=[r["chunk_id"] for r in rlog],
                context_block_sha256=hashlib.sha256(
                    context.encode()).hexdigest(),
                manifest_context_sha256=getattr(
                    case_logger, "_manifest_context_sha256", ""),
            )

    # ── parsing ──────────────────────────────────────────────────────

    def parse_analysis_response(self, response_text: str) -> dict[str, Any]:
        """Parse the response.

        `output_config.format` guarantees valid JSON matching the schema,
        so the happy path is a plain json.loads. When that fails, this
        raises instead of inventing a grade: a fabricated answer that
        looks like a real one in the logs is worse than a missing case.
        """
        self._last_parse_strategy = "structured_output"
        self._last_parse_errors = []
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as exc:
            self._last_parse_strategy = "failed"
            self._last_parse_errors = [str(exc)]
            raise ValueError(
                "structured output did not return valid JSON; refusing to "
                "infer a grade from free text. Raw response preserved in the "
                f"case log. Error: {exc}"
            ) from exc

        result = {k: data.get(k) for k in OUTPUT_SCHEMA_FIELDS}
        result.update(self._derive_consistency_flags(result))
        return result

    @staticmethod
    def _derive_consistency_flags(result: dict[str, Any]) -> dict[str, Any]:
        """Cross-field checks recorded alongside the grade, not corrections.

        These do not alter the model's answer. They mark cases where the
        answer is internally inconsistent so the analysis can report the
        rate rather than silently normalising it away.
        """
        flags: list[str] = []
        category = result.get("lesion_category")
        stratum = result.get("stratum_label")
        grade = result.get("dysplasia_grade")
        subtype = result.get("melanoma_subtype")
        mclass = result.get("mpath_dx_v2_class")

        if category == "melanoma" and stratum != "melanoma":
            flags.append("melanoma_category_but_nonmelanoma_stratum")
        if category != "melanoma" and stratum == "melanoma":
            flags.append("melanoma_stratum_but_nonmelanoma_category")
        if category == "melanoma" and grade != "not_applicable":
            flags.append("melanoma_with_dysplasia_grade")
        if category == "dysplastic_nevus" and grade == "not_applicable":
            flags.append("dysplastic_nevus_without_grade")
        if subtype == "invasive" and result.get("breslow_estimate_mm") is None:
            flags.append("invasive_melanoma_without_breslow")

        if mclass and stratum:
            try:
                expected = mpath_dx.expected_classes(
                    stratum, subtype, result.get("breslow_estimate_mm"))
                if mclass not in expected:
                    flags.append(
                        f"mpath_class_{mclass}_outside_expected_"
                        f"{'_'.join(sorted(expected))}_for_{stratum}")
            except ValueError:
                flags.append("mpath_class_uncheckable")

        return {"consistency_flags": flags,
                "internally_consistent": not flags}
