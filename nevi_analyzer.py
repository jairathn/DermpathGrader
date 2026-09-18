"""Melanocytic grading call (protocol v2.0).

Changes from v1
---------------
1. **The MPATH-Dx v2.0 class is the primary output.** v1 emitted
   `mpath_grade` of "Low-Grade" / "High-Grade Dysplasia" and told the
   model low-grade meant mild and high-grade meant moderate-to-severe.
   That is not MPATH-Dx v1.0 (which had five classes) and it is not v2.0.
   `mpath_dx_v2_class` is now one of 0/I/II/III/IV, defined from the
   consensus statement in mpath_dx.py, and it is what concordance is
   scored on.

2. **Melanoma is a possible answer, and is subtyped.** v1 could only emit
   three dysplasia tiers, so a melanoma in the input had nowhere to go and
   was forced onto the top of the dysplasia ladder. Melanoma is now a
   lesion category carrying `melanoma_subtype` (in situ vs invasive),
   `melanoma_histologic_subtype`, a Breslow estimate, ulceration and
   mitotic rate. In situ versus invasive is what separates Class II from
   Classes III and IV, so it is asked for explicitly rather than
   inferred. The three-tier dysplasia grade survives as a secondary
   descriptive field.

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
        # PRIMARY LABEL. The study strata are the v2.0 classes, so this
        # is what concordance is scored on.
        "mpath_dx_v2_class": {
            "type": "string",
            "enum": list(mpath_dx.CLASSES),
        },
        # What the lesion is. Needed because Class II holds both
        # high-grade dysplasia and melanoma in situ.
        "lesion_category": {
            "type": "string",
            "enum": ["benign_nevus_no_atypia", "dysplastic_nevus",
                     "melanoma", "nondiagnostic"],
        },
        # Secondary descriptive field, cross-tabulated against class but
        # no longer scored as a stratum.
        "dysplasia_grade": {
            "type": "string",
            "enum": list(config.DYSPLASIA_GRADES),
        },
        "melanoma_subtype": {
            "type": "string",
            "enum": list(config.MELANOMA_SUBTYPES),
        },
        "melanoma_histologic_subtype": {
            "type": "string",
            "enum": list(config.MELANOMA_HISTOLOGIC_SUBTYPES),
        },
        "breslow_estimate_mm": {"type": ["number", "null"]},
        "ulceration_present": {"type": ["boolean", "null"]},
        "mitoses_per_mm2": {"type": ["number", "null"]},
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
        "mpath_dx_v2_class", "lesion_category", "dysplasia_grade",
        "melanoma_subtype", "melanoma_histologic_subtype",
        "breslow_estimate_mm", "ulceration_present", "mitoses_per_mm2",
        "confidence_level", "architectural_features",
        "cytological_features", "magnification_evidence",
        "grading_rationale", "clinical_significance",
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

### YOUR PRIMARY OUTPUT IS AN MPATH-Dx v2.0 CLASS

{mpath_dx.prompt_block()}

Assign exactly one class. Everything else you report should be consistent
with it.

### HOW TO GET THERE

First decide what the lesion is, then place it:

- Benign nevus with no meaningful atypia, or a dysplastic nevus with
  low-grade atypia: Class I. Melanocyte nuclei are smaller than 1.5x
  resting basal keratinocyte nuclei.
- Dysplastic nevus with high-grade atypia: Class II. Nuclei from 1.5x up
  to and beyond 2x resting basal keratinocyte nuclei.
- Melanoma in situ, including lentigo maligna: also Class II. Class II
  covers both high-grade dysplasia and in situ melanoma, so use
  lesion_category and melanoma_subtype to say which one you mean. Do not
  push an in situ melanoma to Class III to signal that it is melanoma.
- Invasive melanoma under 0.8 mm Breslow: Class III.
- Invasive melanoma 0.8 mm or greater: Class IV.
- Material that will not support a diagnosis: Class 0.

Report the three-tier dysplasia grade (mild / moderate / severe) as well,
in dysplasia_grade, for any dysplastic nevus. It is descriptive here, not
the label you are being scored on, so do not let it drive the class: v2.0
deliberately removed the standalone moderate category, and a lesion you
would call moderate can be either Class I or Class II depending on
nuclear size and the rest of the cytologic criteria.

### MELANOMA REPORTING

When lesion_category is melanoma:

- melanoma_subtype: in_situ or invasive. This is the distinction that
  separates Class II from Classes III and IV, so make it explicitly.
- melanoma_histologic_subtype: lentigo_maligna, superficial_spreading,
  nodular, acral_lentiginous, desmoplastic, or other.
- breslow_estimate_mm: for invasive melanoma, your best estimate in
  millimetres from the granular layer to the deepest invasive cell. This
  is what separates Class III from Class IV at 0.8 mm, so if the images
  do not let you judge depth, say so in grading_rationale and give your
  best estimate rather than leaving it null.
- ulceration_present and mitoses_per_mm2 where assessable, otherwise null.

### FEATURES FAVOURING MELANOMA OVER DYSPLASIA

Asymmetry and poor circumscription at low power; confluent sheets of
melanocytes; prominent pagetoid spread above the basal layer, especially
peripherally; absent maturation with descent; dermal mitoses; single-cell
predominance over nests; necrosis; ulceration. Weigh each at the
magnification where it is actually assessable.

### RULES

1. Exactly one mpath_dx_v2_class.
2. dysplasia_grade is "not_applicable" when the lesion is melanoma or
   nondiagnostic.
3. melanoma_subtype and melanoma_histologic_subtype are "not_applicable"
   when the lesion is not melanoma, and breslow_estimate_mm is null
   unless the melanoma is invasive.
4. In magnification_evidence, cite at least one finding per
   magnification, naming only what that power can actually show.
5. Grade what is in front of you. Do not hedge toward a middle class to
   avoid committing.
"""

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
        """Cross-field checks recorded alongside the class, not corrections.

        These never alter the model's answer. They mark outputs that
        contradict themselves, so the analysis can report the rate
        instead of silently normalising it away.
        """
        flags: list[str] = []
        category = result.get("lesion_category")
        grade = result.get("dysplasia_grade")
        subtype = result.get("melanoma_subtype")
        histology = result.get("melanoma_histologic_subtype")
        breslow = result.get("breslow_estimate_mm")
        mclass = str(result.get("mpath_dx_v2_class") or "")

        is_melanoma = category == "melanoma"

        if is_melanoma and grade != "not_applicable":
            flags.append("melanoma_with_dysplasia_grade")
        if category == "dysplastic_nevus" and grade in (
                None, "", "not_applicable"):
            flags.append("dysplastic_nevus_without_grade")
        if not is_melanoma and subtype not in (None, "", "not_applicable"):
            flags.append("melanoma_subtype_on_nonmelanoma")
        if is_melanoma and subtype in (None, "", "not_applicable"):
            flags.append("melanoma_without_subtype")
        if is_melanoma and histology in (None, "", "not_applicable"):
            flags.append("melanoma_without_histologic_subtype")
        if subtype == "invasive" and breslow is None:
            flags.append("invasive_melanoma_without_breslow")
        if subtype == "in_situ" and breslow is not None:
            flags.append("in_situ_melanoma_with_breslow")

        # Does the class agree with the diagnosis the model itself gave?
        if mclass and mpath_dx.is_valid_class(mclass):
            normalised = mpath_dx._normalise(mclass)

            if is_melanoma and subtype in ("in_situ", "invasive"):
                try:
                    implied = mpath_dx.class_for_melanoma(subtype, breslow)
                    if implied != normalised:
                        flags.append(
                            f"class_{normalised}_contradicts_"
                            f"{subtype}_melanoma_implying_{implied}")
                except ValueError:
                    # Invasive without Breslow: already flagged above, and
                    # III vs IV genuinely cannot be checked.
                    if normalised not in ("III", "IV"):
                        flags.append(
                            f"class_{normalised}_contradicts_"
                            f"invasive_melanoma")

            if not is_melanoma and mpath_dx.is_melanoma_class(normalised):
                flags.append(
                    f"class_{normalised}_is_invasive_melanoma_but_"
                    f"category_is_{category}")

            if category == "nondiagnostic" and normalised != "0":
                flags.append("nondiagnostic_with_nonzero_class")
            if normalised == "0" and category != "nondiagnostic":
                flags.append("class_0_without_nondiagnostic_category")

            if (category == "benign_nevus_no_atypia"
                    and normalised not in ("I",)):
                flags.append(f"benign_nevus_in_class_{normalised}")

        return {"consistency_flags": flags,
                "internally_consistent": not flags}
