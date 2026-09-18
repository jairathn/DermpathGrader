"""Melanocytic grading call (protocol v2.1).

Output
------
The primary label is the MPATH-Dx v2.0 class (0/I/II/III/IV), which is
what the study is scored on. Alongside it: what the lesion is, the
three-tier dysplasia grade as a descriptive field, full melanoma
subtyping (in situ vs invasive, histologic subtype, Breslow, ulceration,
mitoses), a differential, the ancillary studies a pathologist would
order, and an adequacy call. Every field is enforced server-side by the
JSON schema below, so a response either matches it or is an error.

Request shape
-------------
The prompt scaffold - criteria, MPATH-Dx definitions, retrieved
literature - is identical for every case in the pathway. It goes in the
`system` block with a cache breakpoint, so a 1,000-call batch pays for
those ~12k tokens once per cache window rather than every call. The
user turn carries only the four captioned images and a one-line
instruction. `claude_transport` owns retries, refusal and truncation.

What changed from v1 (2026-09-18)
---------------------------------
- Melanoma is a first-class, subtyped answer; v1 could emit only three
  dysplasia tiers.
- `mpath_dx_v2_class` replaced a two-tier Low/High-Grade field that
  matched neither MPATH-Dx v1.0 nor v2.0.
- Four captioned images per case instead of one.
- Structured outputs replaced a three-layer parser whose keyword
  fallback silently returned "Moderate Dysplasia" on failure.
- max_tokens 1500 -> config.MAX_TOKENS; temperature removed (rejected by
  this model family); the call is streamed and retried.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from anthropic import Anthropic

import claude_transport
import config
import image_utils
import mpath_dx


ANCILLARY_STUDIES = (
    "PRAME", "Melan-A/MART-1", "SOX10", "HMB-45", "Ki-67", "p16",
    "FISH", "NGS", "deeper_levels", "none",
)

NEVUS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # PRIMARY LABEL - the strata are the classes.
        "mpath_dx_v2_class": {"type": "string", "enum": list(mpath_dx.CLASSES)},
        "lesion_category": {
            "type": "string",
            "enum": ["benign_nevus_no_atypia", "dysplastic_nevus",
                     "melanoma", "nondiagnostic"],
        },
        "specimen_adequacy": {
            "type": "string",
            "enum": ["adequate", "limited", "nondiagnostic"],
        },
        # Descriptive; cross-tabulated against class, not scored as a stratum.
        "dysplasia_grade": {"type": "string",
                            "enum": list(config.DYSPLASIA_GRADES)},
        "melanoma_subtype": {"type": "string",
                             "enum": list(config.MELANOMA_SUBTYPES)},
        "melanoma_histologic_subtype": {
            "type": "string",
            "enum": list(config.MELANOMA_HISTOLOGIC_SUBTYPES)},
        "breslow_estimate_mm": {"type": ["number", "null"]},
        "ulceration_present": {"type": ["boolean", "null"]},
        "mitoses_per_mm2": {"type": ["number", "null"]},
        "confidence_level": {"type": "string",
                             "enum": ["High", "Medium", "Low"]},
        "differential_diagnosis": {"type": "array",
                                   "items": {"type": "string"}},
        "recommended_ancillary_studies": {
            "type": "array",
            "items": {"type": "string", "enum": list(ANCILLARY_STUDIES)}},
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
        "mpath_dx_v2_class", "lesion_category", "specimen_adequacy",
        "dysplasia_grade", "melanoma_subtype", "melanoma_histologic_subtype",
        "breslow_estimate_mm", "ulceration_present", "mitoses_per_mm2",
        "confidence_level", "differential_diagnosis",
        "recommended_ancillary_studies", "architectural_features",
        "cytological_features", "magnification_evidence",
        "grading_rationale", "clinical_significance",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA_FIELDS = list(NEVUS_OUTPUT_SCHEMA["properties"].keys())
OUTPUT_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(NEVUS_OUTPUT_SCHEMA, sort_keys=True).encode()).hexdigest()

# The user turn. Short on purpose: everything stable lives in the cached
# system scaffold, and this is the only text that follows the images.
USER_INSTRUCTION = (
    "Grade the melanocytic lesion shown in the images above according to "
    "the instructions you were given. Return only the JSON object."
)

# {context} is filled with the retrieved literature block at call time.
SYSTEM_TEMPLATE = """You are an expert dermatopathologist grading a melanocytic lesion for a research study.

You will be given {n_images} images of the SAME lesion at different magnifications, in this order: {magnifications}. Read them together the way you would read a slide: establish architecture and symmetry at low power, then confirm cytology at high power. Do not grade from one image alone.

Retrieved literature context:
{context}

### YOUR PRIMARY OUTPUT IS AN MPATH-Dx v2.0 CLASS

{mpath_block}

Assign exactly one class. Everything else you report must be consistent with it.

### HOW TO GET THERE

First decide whether the material is adequate. If it is not, use lesion_category "nondiagnostic", specimen_adequacy "nondiagnostic" and Class 0. Use "limited" when you can grade but something material is missing (no deep margin, tangential section, crush artefact) and say what in grading_rationale.

Then decide what the lesion is, and place it:

- Benign nevus with no meaningful atypia, or a dysplastic nevus with low-grade atypia: Class I. Melanocyte nuclei are smaller than 1.5x resting basal keratinocyte nuclei.
- Dysplastic nevus with high-grade atypia: Class II. Nuclei from 1.5x up to and beyond 2x resting basal keratinocyte nuclei.
- Melanoma in situ, including lentigo maligna: also Class II. Class II covers both high-grade dysplasia and in situ melanoma, so use lesion_category and melanoma_subtype to say which one you mean. Do not push an in situ melanoma to Class III to signal that it is melanoma.
- Invasive melanoma under 0.8 mm Breslow: Class III.
- Invasive melanoma 0.8 mm or greater: Class IV.

Report the three-tier dysplasia grade (mild / moderate / severe) in dysplasia_grade for any dysplastic nevus. It is descriptive, not the label you are scored on, so do not let it drive the class: v2.0 removed the standalone moderate category, and a lesion you would call moderate can be Class I or Class II depending on nuclear size and the other cytologic criteria.

### MELANOMA REPORTING

When lesion_category is melanoma:
- melanoma_subtype: in_situ or invasive. This separates Class II from Classes III and IV, so make it explicitly.
- melanoma_histologic_subtype: lentigo_maligna, superficial_spreading, nodular, acral_lentiginous, desmoplastic, or other.
- breslow_estimate_mm: for invasive melanoma, your best estimate in millimetres from the granular layer to the deepest invasive cell. This separates Class III from Class IV at 0.8 mm. If the images do not let you judge depth, say so in grading_rationale and still give your best estimate.
- ulceration_present and mitoses_per_mm2 where assessable, otherwise null.

### FEATURES FAVOURING MELANOMA OVER DYSPLASIA

Asymmetry and poor circumscription at low power; confluent sheets of melanocytes; prominent pagetoid spread above the basal layer, especially peripherally; absent maturation with descent; dermal mitoses; single-cell predominance over nests; necrosis; ulceration. Weigh each at the magnification where it is actually assessable.

### DIFFERENTIAL AND ANCILLARY STUDIES

List the two to four diagnoses you actively considered in differential_diagnosis, most likely first, even when confident. In recommended_ancillary_studies list what you would order to resolve the case if this were sign-out: PRAME and Melan-A/MART-1 are the usual adjuncts for a borderline junctional proliferation; SOX10 or HMB-45 for a dermal component; Ki-67 for proliferation; p16, FISH or NGS for a spitzoid or ambiguous lesion; deeper_levels when the section is unrepresentative. Use "none" only when you would sign out on H&E alone.

### RULES

1. Exactly one mpath_dx_v2_class.
2. dysplasia_grade is "not_applicable" when the lesion is melanoma or nondiagnostic.
3. melanoma_subtype and melanoma_histologic_subtype are "not_applicable" when the lesion is not melanoma, and breslow_estimate_mm is null unless the melanoma is invasive.
4. In magnification_evidence, cite at least one finding per magnification, naming only what that power can actually show.
5. Grade what is in front of you. Do not hedge toward a middle class to avoid committing. Confidence goes in confidence_level, not in the class."""


class NeviAnalyzer:
    """Melanocytic lesion grader."""

    def __init__(self, nevi_rag_system, client: Anthropic | None = None):
        self.nevi_rag_system = nevi_rag_system
        self.model = config.MODEL_ID
        self.client = client if client is not None else self._make_client()

    @staticmethod
    def _make_client() -> Anthropic:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        return Anthropic()

    # ── prompt ───────────────────────────────────────────────────────

    @staticmethod
    def build_system_prompt(context: str) -> str:
        return SYSTEM_TEMPLATE.format(
            n_images=len(config.MAGNIFICATIONS),
            magnifications=", ".join(config.MAGNIFICATIONS),
            context=context,
            mpath_block=mpath_dx.prompt_block(),
        )

    # Kept for callers and make_manifest that still use the old name.
    create_analysis_prompt = build_system_prompt

    # ── call ─────────────────────────────────────────────────────────

    def build_request(self, prepared_images: list, context: str
                      ) -> dict[str, Any]:
        content = image_utils.build_image_content_blocks(prepared_images)
        content.append({"type": "text", "text": USER_INSTRUCTION})
        return claude_transport.build_request(
            model=self.model,
            system_text=self.build_system_prompt(context),
            user_content=content,
            output_schema=NEVUS_OUTPUT_SCHEMA,
            max_tokens=config.MAX_TOKENS,
            thinking=config.THINKING,
            effort=config.EFFORT,
        )

    def analyze_images(self, prepared_images: list, case_logger=None
                       ) -> dict[str, Any]:
        """Grade one case. Raises claude_transport.GradingError subclasses."""
        context = self.nevi_rag_system.get_grading_criteria()
        request = self.build_request(prepared_images, context)
        system_text = request["system"][0]["text"]

        if case_logger is not None:
            # Record the request before the call so a failed case still
            # has a log that says exactly what was attempted.
            case_logger.set_request(
                model=self.model, max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE, thinking=config.THINKING,
                effort=config.EFFORT,
                system_text=system_text, user_text=USER_INSTRUCTION,
                message_structure=image_utils.message_structure(
                    prepared_images,
                    system_sha256=hashlib.sha256(system_text.encode()).hexdigest(),
                    user_sha256=hashlib.sha256(USER_INSTRUCTION.encode()).hexdigest()),
                schema_enforced=True,
                output_schema_sha256=OUTPUT_SCHEMA_SHA256,
            )
            self._log_retrieval(case_logger, context)

        try:
            response = claude_transport.grade(self.client, request)
        except claude_transport.GradingError as exc:
            if case_logger is not None:
                case_logger.set_failure(exc)
            raise

        result = {k: response.parsed.get(k) for k in OUTPUT_SCHEMA_FIELDS}
        result.update(self._derive_consistency_flags(result))
        result["raw_analysis"] = response.text
        result["context_used"] = bool(context.strip())

        if case_logger is not None:
            case_logger.set_attempt(response.attempt, response.prior_attempt_errors)
            case_logger.set_response(
                api_request_id=getattr(response.message, "id", ""),
                model_returned=getattr(response.message, "model", ""),
                stop_reason=response.stop_reason,
                stop_details=response.stop_details,
                usage=response.usage,
                latency_ms=response.latency_ms,
                raw_text=response.text,
            )
            case_logger.set_parsing(
                strategy_used="structured_output", fallback_invoked=False,
                parse_errors=[],
                parsed={k: result.get(k) for k in OUTPUT_SCHEMA_FIELDS
                        + ["consistency_flags", "internally_consistent"]},
            )
        return result

    def _log_retrieval(self, case_logger, context: str) -> None:
        if not hasattr(self.nevi_rag_system, "_retrieval_log"):
            return
        rlog = self.nevi_rag_system._retrieval_log
        case_logger.set_retrieval(
            subqueries_issued=list(dict.fromkeys(r["query"] for r in rlog)),
            retrieved_chunk_ids_in_order=[r["chunk_id"] for r in rlog],
            context_block_sha256=hashlib.sha256(context.encode()).hexdigest(),
            manifest_context_sha256=getattr(
                case_logger, "_manifest_context_sha256", ""),
        )

    # ── consistency ──────────────────────────────────────────────────

    @staticmethod
    def _derive_consistency_flags(result: dict[str, Any]) -> dict[str, Any]:
        """Cross-field checks recorded alongside the class, never corrections."""
        flags: list[str] = []
        category = result.get("lesion_category")
        adequacy = result.get("specimen_adequacy")
        grade = result.get("dysplasia_grade")
        subtype = result.get("melanoma_subtype")
        histology = result.get("melanoma_histologic_subtype")
        breslow = result.get("breslow_estimate_mm")
        mclass = str(result.get("mpath_dx_v2_class") or "")
        is_melanoma = category == "melanoma"

        if is_melanoma and grade != "not_applicable":
            flags.append("melanoma_with_dysplasia_grade")
        if category == "dysplastic_nevus" and grade in (None, "", "not_applicable"):
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
        if (category == "nondiagnostic") != (adequacy == "nondiagnostic"):
            flags.append("adequacy_and_category_disagree_on_nondiagnostic")

        if mclass and mpath_dx.is_valid_class(mclass):
            normalised = mpath_dx._normalise(mclass)
            if is_melanoma and subtype in ("in_situ", "invasive"):
                try:
                    implied = mpath_dx.class_for_melanoma(subtype, breslow)
                    if implied != normalised:
                        flags.append(f"class_{normalised}_contradicts_"
                                     f"{subtype}_melanoma_implying_{implied}")
                except ValueError:
                    if normalised not in ("III", "IV"):
                        flags.append(f"class_{normalised}_contradicts_"
                                     f"invasive_melanoma")
            if not is_melanoma and mpath_dx.is_melanoma_class(normalised):
                flags.append(f"class_{normalised}_is_invasive_melanoma_but_"
                             f"category_is_{category}")
            if category == "nondiagnostic" and normalised != "0":
                flags.append("nondiagnostic_with_nonzero_class")
            if normalised == "0" and category != "nondiagnostic":
                flags.append("class_0_without_nondiagnostic_category")
            if category == "benign_nevus_no_atypia" and normalised != "I":
                flags.append(f"benign_nevus_in_class_{normalised}")

        return {"consistency_flags": flags, "internally_consistent": not flags}
