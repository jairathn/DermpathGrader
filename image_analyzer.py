"""CSCC differentiation grading call (protocol v2.0).

Changes from v1
---------------
1. Four captioned images per case instead of one, plus a
   `magnification_evidence` field recording which power each finding
   came from.
2. Structured outputs replace the three-layer parser. v1 fell through
   JSON -> regex -> keyword inference, and the keyword layer defaulted to
   "Moderately Differentiated" when it could not decide - a fabricated
   grade that looked identical to a real one in the logs (CLAUDE.md
   landmine 4). The schema is now enforced by the API and a malformed
   response raises.
3. Model is config.MODEL_ID; max_tokens is config.MAX_TOKENS, up from
   1500; the call is streamed; `temperature` is gone because this model
   family rejects sampling parameters.

The label space is unchanged: well / moderately / poorly differentiated.
Only the melanocytic arm gained a category.
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


CSCC_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_grade": {
            "type": "string",
            "enum": ["Well Differentiated", "Moderately Differentiated",
                     "Poorly Differentiated"],
        },
        "confidence_level": {"type": "string",
                             "enum": ["High", "Medium", "Low"]},
        "keratinization_present": {"type": "boolean"},
        "atypia_level": {"type": "string",
                         "enum": ["minimal", "moderate", "high"]},
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
        "primary_grade", "confidence_level", "keratinization_present",
        "atypia_level", "key_features", "magnification_evidence",
        "additional_observations",
    ],
    "additionalProperties": False,
}

OUTPUT_SCHEMA_FIELDS = list(CSCC_OUTPUT_SCHEMA["properties"].keys())


class ImageAnalyzer:
    """Cutaneous squamous cell carcinoma differentiation grader."""

    def __init__(self, rag_system):
        self.rag_system = rag_system
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
        return f"""You are an expert dermatopathologist grading the differentiation
of a cutaneous squamous cell carcinoma (CSCC).

You have been given {len(config.MAGNIFICATIONS)} images of the SAME lesion at
different magnifications, in this order: {mags}. Read them together: judge
tumour architecture and invasive pattern at low power, then keratinization and
cytologic atypia at high power. Do not grade from one image alone.

Retrieved literature context:
{context}

### ASSESS

- Keratinization: keratin pearls, horn cysts, intracellular keratin.
  More keratinization means better differentiated.
- Cellular atypia: pleomorphism, nuclear abnormalities. More atypia means
  more poorly differentiated.
- Architecture: organised and pushing versus infiltrative and disorganised.
- Squamous maturation: the gradient from basal to keratinised cells.
- Mitotic activity: low, moderate or high.

### GRADING CRITERIA

- Well differentiated: abundant keratinization, keratin pearls present,
  minimal atypia, organised architecture.
- Moderately differentiated: some keratinization, moderate atypia.
- Poorly differentiated: minimal or absent keratinization, high atypia,
  infiltrative, basaloid.

### RULES

1. Choose one of the three grades. Do not return unknown.
2. Minimal or absent keratinization AND high atypia means poorly
   differentiated.
3. Abundant keratinization AND minimal atypia means well differentiated.
4. For a mixed-grade tumour, report the worst (least differentiated)
   component.
5. In `magnification_evidence`, cite at least one finding per magnification,
   naming only what that power can actually show. Keratin pearls are
   assessable at 4x and 10x; nuclear detail and mitoses need 40x.
6. Grade what is in front of you. Do not hedge to the middle category to
   avoid committing."""

    # ── call ─────────────────────────────────────────────────────────

    def analyze_images(self, prepared_images: list, case_logger=None
                       ) -> dict[str, Any]:
        """Grade one case from its prepared multi-magnification images."""
        context = self.rag_system.get_grading_criteria()
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
                "format": {"type": "json_schema", "schema": CSCC_OUTPUT_SCHEMA},
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
                json.dumps(CSCC_OUTPUT_SCHEMA, sort_keys=True).encode()
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
        if hasattr(self.rag_system, "_retrieval_log"):
            rlog = self.rag_system._retrieval_log
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

        The schema is enforced server-side, so this is a plain json.loads.
        On failure it raises rather than inferring a grade from keywords:
        the v1 fallback's silent default to "Moderately Differentiated"
        was indistinguishable from a genuine answer once logged.
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
        return {k: data.get(k) for k in OUTPUT_SCHEMA_FIELDS}
