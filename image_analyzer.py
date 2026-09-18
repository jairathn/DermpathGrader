"""CSCC differentiation grading call (protocol v2.1).

Output
------
Differentiation grade (well / moderately / poorly) is the study label.
Alongside it, the fields a sign-out report would carry and that drive
staging: histologic subtype, Broders grade, depth of invasion, the
high-risk features that move a tumour between AJCC / BWH stages,
a differential, ancillary studies, and an adequacy call.

Request shape mirrors nevi_analyzer: cached system scaffold, four
captioned images plus a one-line instruction in the user turn,
claude_transport for retries, refusal and truncation.
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


HISTOLOGIC_SUBTYPES = (
    "conventional", "acantholytic", "adenosquamous", "desmoplastic",
    "spindle_cell", "verrucous", "keratoacanthoma_like", "clear_cell",
    "other",
)

HIGH_RISK_FEATURES = (
    "perineural_invasion", "lymphovascular_invasion",
    "invasion_beyond_subcutaneous_fat", "tumor_thickness_over_6mm",
    "desmoplastic_subtype", "poorly_differentiated", "bone_invasion",
    "none_identified",
)

DEPTH_OF_INVASION = (
    "in_situ", "papillary_dermis", "reticular_dermis", "subcutis",
    "beyond_subcutis", "not_assessable",
)

ANCILLARY_STUDIES = (
    "p63", "p40", "CK5/6", "AE1/AE3", "EMA", "S100", "SOX10", "CD10",
    "Ber-EP4", "deeper_levels", "none",
)

CSCC_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_grade": {
            "type": "string",
            "enum": ["Well Differentiated", "Moderately Differentiated",
                     "Poorly Differentiated"],
        },
        "broders_grade": {"type": "integer", "enum": [1, 2, 3, 4]},
        "specimen_adequacy": {
            "type": "string",
            "enum": ["adequate", "limited", "nondiagnostic"],
        },
        "histologic_subtype": {"type": "string",
                               "enum": list(HISTOLOGIC_SUBTYPES)},
        "depth_of_invasion": {"type": "string",
                              "enum": list(DEPTH_OF_INVASION)},
        "high_risk_features": {
            "type": "array",
            "items": {"type": "string", "enum": list(HIGH_RISK_FEATURES)}},
        "confidence_level": {"type": "string",
                             "enum": ["High", "Medium", "Low"]},
        "keratinization_present": {"type": "boolean"},
        "atypia_level": {"type": "string",
                         "enum": ["minimal", "moderate", "high"]},
        "differential_diagnosis": {"type": "array",
                                   "items": {"type": "string"}},
        "recommended_ancillary_studies": {
            "type": "array",
            "items": {"type": "string", "enum": list(ANCILLARY_STUDIES)}},
        "key_features": {"type": "array", "items": {"type": "string"}},
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
        "additional_observations": {"type": "string"},
    },
    "required": [
        "primary_grade", "broders_grade", "specimen_adequacy",
        "histologic_subtype", "depth_of_invasion", "high_risk_features",
        "confidence_level", "keratinization_present", "atypia_level",
        "differential_diagnosis", "recommended_ancillary_studies",
        "key_features", "magnification_evidence", "additional_observations",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA_FIELDS = list(CSCC_OUTPUT_SCHEMA["properties"].keys())
OUTPUT_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(CSCC_OUTPUT_SCHEMA, sort_keys=True).encode()).hexdigest()

USER_INSTRUCTION = (
    "Grade the cutaneous squamous cell carcinoma shown in the images above "
    "according to the instructions you were given. Return only the JSON "
    "object."
)

SYSTEM_TEMPLATE = """You are an expert dermatopathologist grading the differentiation of a cutaneous squamous cell carcinoma (CSCC) for a research study.

You will be given {n_images} images of the SAME lesion at different magnifications, in this order: {magnifications}. Read them together: judge tumour architecture, depth and invasive pattern at low power, then keratinization and cytologic atypia at high power. Do not grade from one image alone.

Retrieved literature context:
{context}

### ADEQUACY FIRST

If the material will not support a grade, use specimen_adequacy "nondiagnostic" and explain in additional_observations; still return your best grade rather than leaving fields empty. Use "limited" when you can grade but something material is missing (no deep margin, tangential section, crush artefact) and say what.

### ASSESS

- Keratinization: keratin pearls, horn cysts, intracellular keratin. More keratinization means better differentiated.
- Cellular atypia: pleomorphism, nuclear abnormalities. More atypia means more poorly differentiated.
- Architecture: organised and pushing versus infiltrative and disorganised.
- Squamous maturation: the gradient from basal to keratinised cells.
- Mitotic activity: low, moderate or high.

### GRADING CRITERIA

- Well differentiated (Broders 1): abundant keratinization, keratin pearls present, minimal atypia, organised architecture.
- Moderately differentiated (Broders 2-3): some keratinization, moderate atypia.
- Poorly differentiated (Broders 4): minimal or absent keratinization, high atypia, infiltrative, basaloid.

Report broders_grade as your best placement on the 1-4 scale, consistent with primary_grade.

### SUBTYPE, DEPTH AND HIGH-RISK FEATURES

Name the histologic subtype. Report depth_of_invasion as the deepest level you can see. In high_risk_features list every feature present from: perineural invasion, lymphovascular invasion, invasion beyond subcutaneous fat, tumour thickness over 6 mm, desmoplastic subtype, poorly differentiated, bone invasion; or none_identified. These are what move a tumour between stages, so say what you can and cannot assess at these magnifications in additional_observations rather than guessing.

### DIFFERENTIAL AND ANCILLARY STUDIES

List the two to four diagnoses you actively considered in differential_diagnosis, most likely first: for a keratinizing tumour that usually includes keratoacanthoma and pseudoepitheliomatous hyperplasia; for a poorly differentiated one, spindle-cell or basaloid mimics including melanoma, atypical fibroxanthoma and basal cell carcinoma. In recommended_ancillary_studies list what you would order to resolve it at sign-out; use "none" only when you would sign out on H&E alone.

### RULES

1. Choose one of the three grades. Do not return unknown.
2. Minimal or absent keratinization AND high atypia means poorly differentiated.
3. Abundant keratinization AND minimal atypia means well differentiated.
4. For a mixed-grade tumour, report the worst (least differentiated) component.
5. In magnification_evidence, cite at least one finding per magnification, naming only what that power can actually show. Keratin pearls are assessable at 4x and 10x; nuclear detail and mitoses need 40x; perineural invasion needs 10x-40x and a nerve in the field.
6. Grade what is in front of you. Do not hedge to the middle category to avoid committing. Confidence goes in confidence_level, not in the grade."""


class ImageAnalyzer:
    """Cutaneous squamous cell carcinoma differentiation grader."""

    def __init__(self, rag_system, client: Anthropic | None = None):
        self.rag_system = rag_system
        self.model = config.MODEL_ID
        self.client = client if client is not None else self._make_client()

    @staticmethod
    def _make_client() -> Anthropic:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        return Anthropic()

    @staticmethod
    def build_system_prompt(context: str) -> str:
        return SYSTEM_TEMPLATE.format(
            n_images=len(config.MAGNIFICATIONS),
            magnifications=", ".join(config.MAGNIFICATIONS),
            context=context,
        )

    create_analysis_prompt = build_system_prompt

    def build_request(self, prepared_images: list, context: str
                      ) -> dict[str, Any]:
        content = image_utils.build_image_content_blocks(prepared_images)
        content.append({"type": "text", "text": USER_INSTRUCTION})
        return claude_transport.build_request(
            model=self.model,
            system_text=self.build_system_prompt(context),
            user_content=content,
            output_schema=CSCC_OUTPUT_SCHEMA,
            max_tokens=config.MAX_TOKENS,
            thinking=config.THINKING,
            effort=config.EFFORT,
        )

    def analyze_images(self, prepared_images: list, case_logger=None
                       ) -> dict[str, Any]:
        """Grade one case. Raises claude_transport.GradingError subclasses."""
        context = self.rag_system.get_grading_criteria()
        request = self.build_request(prepared_images, context)
        system_text = request["system"][0]["text"]

        if case_logger is not None:
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
        if not hasattr(self.rag_system, "_retrieval_log"):
            return
        rlog = self.rag_system._retrieval_log
        case_logger.set_retrieval(
            subqueries_issued=list(dict.fromkeys(r["query"] for r in rlog)),
            retrieved_chunk_ids_in_order=[r["chunk_id"] for r in rlog],
            context_block_sha256=hashlib.sha256(context.encode()).hexdigest(),
            manifest_context_sha256=getattr(
                case_logger, "_manifest_context_sha256", ""),
        )

    @staticmethod
    def _derive_consistency_flags(result: dict[str, Any]) -> dict[str, Any]:
        """Cross-field checks recorded alongside the grade, never corrections."""
        flags: list[str] = []
        grade = result.get("primary_grade") or ""
        broders = result.get("broders_grade")
        risks = result.get("high_risk_features") or []
        adequacy = result.get("specimen_adequacy")
        subtype = result.get("histologic_subtype")
        depth = result.get("depth_of_invasion")

        expected_broders = {"Well Differentiated": {1},
                            "Moderately Differentiated": {2, 3},
                            "Poorly Differentiated": {4}}.get(grade)
        if expected_broders and broders not in expected_broders:
            flags.append(f"broders_{broders}_inconsistent_with_{grade.split()[0].lower()}")

        if "none_identified" in risks and len(risks) > 1:
            flags.append("none_identified_alongside_named_high_risk_features")
        if grade == "Poorly Differentiated" and "poorly_differentiated" not in risks:
            flags.append("poorly_differentiated_not_listed_as_high_risk_feature")
        if grade != "Poorly Differentiated" and "poorly_differentiated" in risks:
            flags.append("poorly_differentiated_listed_but_grade_is_not")
        if subtype == "desmoplastic" and "desmoplastic_subtype" not in risks:
            flags.append("desmoplastic_subtype_not_listed_as_high_risk_feature")
        if "invasion_beyond_subcutaneous_fat" in risks and depth not in (
                "beyond_subcutis", "not_assessable"):
            flags.append("beyond_fat_listed_but_depth_disagrees")
        if depth == "in_situ" and any(r in risks for r in (
                "perineural_invasion", "lymphovascular_invasion",
                "invasion_beyond_subcutaneous_fat", "bone_invasion")):
            flags.append("in_situ_with_invasive_high_risk_feature")
        if adequacy == "nondiagnostic" and result.get("confidence_level") == "High":
            flags.append("nondiagnostic_specimen_with_high_confidence")

        return {"consistency_flags": flags, "internally_consistent": not flags}
