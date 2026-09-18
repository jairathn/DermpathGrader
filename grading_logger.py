"""
grading_logger.py
-----------------
Core logging infrastructure for CSCC and Nevus grading runs.

Provides:
  CaseLogger   – writes one analysis_logs/{pathway}/{case_id}__rep{n}.json per call
  sha256_str   – SHA-256 of a UTF-8 string
  sha256_bytes – SHA-256 of raw bytes

Design rules (from spec):
  • Never overwrite a log: rep numbers increment automatically.
  • raw_text is stored byte-for-byte before parsing.
  • This module never imports ground truth files.
"""

import hashlib
import json
import pathlib
import datetime
from typing import Any, Dict, List, Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utcnow() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# ── CaseLogger ───────────────────────────────────────────────────────────────

class CaseLogger:
    """
    Accumulates fields for one case-replicate, then writes the JSON log.

    Usage:
        logger = CaseLogger("CSCC", "CSCC-001", replicate=1, session_id="…")
        logger.set_image_meta(…)
        logger.set_retrieval(…)
        logger.set_request(…)
        logger.set_response(…)
        logger.set_parsing(…)
        logger.save()          # raises FileExistsError if file already exists
    """

    LOG_ROOT = pathlib.Path("analysis_logs")

    def __init__(self, pathway: str, case_id: str, replicate: int,
                 session_id: str):
        self.pathway    = pathway
        self.case_id    = case_id
        self.replicate  = replicate
        self.session_id = session_id

        self._record: Dict[str, Any] = {
            "log_version":        "1.0",
            "manifest_session_id": session_id,
            "pathway":            pathway,
            "case_id":            case_id,
            "replicate":          replicate,
            "timestamp_utc":      _utcnow(),

            "image": {
                "source_registry":    "",
                "source_filename":    "",
                "source_sha256":      "",
                "source_dimensions_px": [0, 0],
                "sent_media_type":    "image/jpeg",
                "sent_dimensions_px": [0, 0],
                "sent_bytes":         0,
                "sent_sha256":        "",
                "resize_applied":     False,
                "compression_quality": None,
            },

            "retrieval": {
                "subqueries_issued":          [],
                "retrieved_chunk_ids_in_order": [],
                "context_block_sha256":       "",
                "matches_manifest_context":   True,
            },

            "request": {
                "model":               "",
                "temperature":         0.1,
                "max_tokens":          0,
                "system_text":         "",
                "system_sha256":       "",
                "user_text":           "",
                "user_sha256":         "",
                "message_structure":   [],
                "attempt":             1,
                "prior_attempt_errors": [],
            },

            "response": {
                "api_request_id": "",
                "model_returned": "",
                "stop_reason":    "",
                "usage":          {"input_tokens": 0, "output_tokens": 0},
                "latency_ms":     0,
                "raw_text":       "",
            },

            "parsing": {
                "strategy_used":   "direct_json",
                "fallback_invoked": False,
                "parse_errors":    [],
                "parsed":          {},
            },

            "errors": [],
        }

    # ── setters ──────────────────────────────────────────────────────────────

    def set_image_meta(self, *,
                       source_registry: str,
                       source_filename: str,
                       source_sha256: str,
                       source_dimensions_px: List[int],
                       sent_media_type: str,
                       sent_dimensions_px: List[int],
                       sent_bytes: int,
                       sent_sha256: str,
                       resize_applied: bool,
                       compression_quality: Optional[int]):
        self._record["image"].update({
            "source_registry":      source_registry,
            "source_filename":      source_filename,
            "source_sha256":        source_sha256,
            "source_dimensions_px": source_dimensions_px,
            "sent_media_type":      sent_media_type,
            "sent_dimensions_px":   sent_dimensions_px,
            "sent_bytes":           sent_bytes,
            "sent_sha256":          sent_sha256,
            "resize_applied":       resize_applied,
            "compression_quality":  compression_quality,
        })

    def set_retrieval(self, *,
                      subqueries_issued: List[str],
                      retrieved_chunk_ids_in_order: List[str],
                      context_block_sha256: str,
                      manifest_context_sha256: str):
        matches = (context_block_sha256 == manifest_context_sha256)
        self._record["retrieval"].update({
            "subqueries_issued":            subqueries_issued,
            "retrieved_chunk_ids_in_order": retrieved_chunk_ids_in_order,
            "context_block_sha256":         context_block_sha256,
            "matches_manifest_context":     matches,
        })

    def set_request(self, *,
                    model: str,
                    temperature: float,
                    max_tokens: int,
                    system_text: str,
                    user_text: str,
                    image_sha256: str,
                    attempt: int = 1,
                    prior_attempt_errors: Optional[List[str]] = None,
                    sent_media_type: str = "image/jpeg"):
        # Block order matches the actual API call: text first, image second
        self._record["request"].update({
            "model":             model,
            "temperature":       temperature,
            "max_tokens":        max_tokens,
            "system_text":       system_text,
            "system_sha256":     sha256_str(system_text),   # always compute, even for ""
            "user_text":         user_text,
            "user_sha256":       sha256_str(user_text),
            "message_structure": [
                {"role": "user", "blocks": [
                    {"type": "text",  "sha256": sha256_str(user_text)},
                    {"type": "image", "media_type": sent_media_type,
                     "sha256": image_sha256},
                ]}
            ],
            "attempt":               attempt,
            "prior_attempt_errors":  prior_attempt_errors or [],
        })

    def set_response(self, *,
                     api_request_id: str,
                     model_returned: str,
                     stop_reason: str,
                     input_tokens: int,
                     output_tokens: int,
                     latency_ms: int,
                     raw_text: str):
        self._record["response"].update({
            "api_request_id": api_request_id,
            "model_returned": model_returned,
            "stop_reason":    stop_reason,
            "usage":          {"input_tokens": input_tokens,
                               "output_tokens": output_tokens},
            "latency_ms":     latency_ms,
            "raw_text":       raw_text,   # byte-for-byte, before parsing
        })

    def set_parsing(self, *,
                    strategy_used: str,
                    fallback_invoked: bool,
                    parse_errors: List[str],
                    parsed: Dict[str, Any]):
        self._record["parsing"].update({
            "strategy_used":    strategy_used,
            "fallback_invoked": fallback_invoked,
            "parse_errors":     parse_errors,
            "parsed":           parsed,
        })

    def add_error(self, error: str):
        self._record["errors"].append(error)

    # ── persistence ──────────────────────────────────────────────────────────

    def _filepath(self) -> pathlib.Path:
        dir_ = self.LOG_ROOT / self.pathway
        dir_.mkdir(parents=True, exist_ok=True)
        return dir_ / f"{self.case_id}__rep{self.replicate}.json"

    def save(self) -> pathlib.Path:
        """Write the log. Raises FileExistsError if file already exists."""
        fp = self._filepath()
        if fp.exists():
            raise FileExistsError(
                f"Log already exists: {fp}  "
                f"(increment replicate number or check for duplicate run)"
            )
        fp.write_text(json.dumps(self._record, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        return fp

    def next_free_replicate(self) -> int:
        """Return lowest replicate number whose log file does not yet exist."""
        n = self.replicate
        while (self.LOG_ROOT / self.pathway /
               f"{self.case_id}__rep{n}.json").exists():
            n += 1
        return n

    @property
    def record(self) -> Dict[str, Any]:
        return self._record
