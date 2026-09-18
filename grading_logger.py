"""Per-case structured logging (protocol v2.0).

Writes one analysis_logs/{pathway}/{case_id}__rep{n}.json per grading
call.

Changes from log_version 1.0
----------------------------
- `image` (a single object) became `images` (an ordered list, one entry
  per magnification) plus `image_set` summary fields. A v1.0 log has one
  image and no magnification; a v2.0 log has four. The verifier checks
  the version and applies the right rules, so old logs stay readable and
  are never silently pooled with new ones.
- `request` gained `thinking`, `effort`, `schema_enforced` and
  `output_schema_sha256`, and `temperature` may now be null because this
  model family rejects sampling parameters.
- `protocol_version` is recorded so a mixed-protocol batch is detectable.
- `parsing.strategy_used` of "structured_output" means the API enforced
  the schema. Anything else means a fallback ran, which under v2.0 should
  not happen and is worth investigating rather than ignoring.

Design rules carried over unchanged:
  - Never overwrite a log; replicate numbers increment.
  - raw_text is stored byte-for-byte before parsing.
  - This module never imports ground truth.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
from typing import Any, Optional

import config


def sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


class CaseLogger:
    """Accumulates one case-replicate, then writes its JSON log.

        logger = CaseLogger("Nevus", "NEV-001", replicate=1, session_id=...)
        logger.set_image_set(prepared_images, source_registry="...")
        logger.set_retrieval(...)
        logger.set_request(...)
        logger.set_response(...)
        logger.set_parsing(...)
        logger.save()      # FileExistsError if the file already exists
    """

    LOG_ROOT = pathlib.Path(config.LOG_ROOT)

    def __init__(self, pathway: str, case_id: str, replicate: int,
                 session_id: str):
        self.pathway = pathway
        self.case_id = case_id
        self.replicate = replicate
        self.session_id = session_id

        self._record: dict[str, Any] = {
            "log_version": config.LOG_VERSION,
            "protocol_version": config.PROTOCOL_VERSION,
            "manifest_session_id": session_id,
            "pathway": pathway,
            "case_id": case_id,
            "replicate": replicate,
            "timestamp_utc": _utcnow(),

            "image_set": {
                "source_registry": "",
                "magnifications_expected": list(
                    config.REQUIRED_MAGNIFICATIONS),
                "magnifications_sent": [],
                "image_count": 0,
                "total_sent_bytes": 0,
                "set_sha256": "",
                "wsi_source_filename": "",
                "wsi_sha256": "",
                "tile_selection_method": "",
            },
            "images": [],

            "retrieval": {
                "subqueries_issued": [],
                "retrieved_chunk_ids_in_order": [],
                "context_block_sha256": "",
                "matches_manifest_context": True,
            },

            "request": {
                "model": "",
                "temperature": None,
                "thinking": None,
                "effort": "",
                "max_tokens": 0,
                "schema_enforced": False,
                "output_schema_sha256": "",
                "system_text": "",
                "system_sha256": "",
                "user_text": "",
                "user_sha256": "",
                "message_structure": [],
                "attempt": 1,
                "prior_attempt_errors": [],
            },

            "response": {
                "api_request_id": "",
                "model_returned": "",
                "stop_reason": "",
                "stop_details": None,
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0},
                "latency_ms": 0,
                "raw_text": "",
            },

            # Populated only when the call did not yield a grade. A failed
            # case still gets a log: the request that was attempted, the
            # error class, and for a refusal the API's stated category.
            "failure": None,

            "parsing": {
                "strategy_used": "structured_output",
                "fallback_invoked": False,
                "parse_errors": [],
                "parsed": {},
            },

            "errors": [],
        }

    # ── setters ──────────────────────────────────────────────────────

    def set_image_set(self, prepared_images, *,
                      source_registry: str = "",
                      wsi_source_filename: str = "",
                      wsi_sha256: str = "",
                      tile_selection_method: str = "") -> None:
        """Record every magnification sent, in presentation order.

        `set_sha256` is the hash of the concatenated per-image sent
        hashes: one value that identifies the whole four-image payload,
        so two runs of the same case can be compared without walking the
        list.
        """
        images = [img.to_log_dict() for img in prepared_images]
        concatenated = "".join(i["sent_sha256"] for i in images)

        self._record["images"] = images
        self._record["image_set"].update({
            "source_registry": source_registry,
            "magnifications_sent": [i["magnification"] for i in images],
            "image_count": len(images),
            "total_sent_bytes": sum(i["sent_bytes"] for i in images),
            "set_sha256": sha256_str(concatenated),
            "wsi_source_filename": wsi_source_filename,
            "wsi_sha256": wsi_sha256,
            "tile_selection_method": tile_selection_method,
        })

    def set_retrieval(self, *, subqueries_issued: list[str],
                      retrieved_chunk_ids_in_order: list[str],
                      context_block_sha256: str,
                      manifest_context_sha256: str) -> None:
        self._record["retrieval"].update({
            "subqueries_issued": subqueries_issued,
            "retrieved_chunk_ids_in_order": retrieved_chunk_ids_in_order,
            "context_block_sha256": context_block_sha256,
            "matches_manifest_context":
                context_block_sha256 == manifest_context_sha256,
        })

    def set_request(self, *, model: str, max_tokens: int,
                    system_text: str, user_text: str,
                    message_structure: list,
                    temperature: Optional[float] = None,
                    thinking: Optional[dict] = None,
                    effort: str = "",
                    schema_enforced: bool = False,
                    output_schema_sha256: str = "",
                    attempt: int = 1,
                    prior_attempt_errors: Optional[list[str]] = None) -> None:
        self._record["request"].update({
            "model": model,
            "temperature": temperature,
            "thinking": thinking,
            "effort": effort,
            "max_tokens": max_tokens,
            "schema_enforced": schema_enforced,
            "output_schema_sha256": output_schema_sha256,
            "system_text": system_text,
            "system_sha256": sha256_str(system_text),
            "user_text": user_text,
            "user_sha256": sha256_str(user_text),
            "message_structure": message_structure,
            "attempt": attempt,
            "prior_attempt_errors": prior_attempt_errors or [],
        })

    def set_response(self, *, api_request_id: str, model_returned: str,
                     stop_reason: str, latency_ms: int, raw_text: str,
                     usage: Optional[dict] = None,
                     stop_details: Optional[dict] = None,
                     input_tokens: Optional[int] = None,
                     output_tokens: Optional[int] = None) -> None:
        if usage is None:
            usage = {"input_tokens": input_tokens or 0,
                     "output_tokens": output_tokens or 0}
        merged = dict(self._record["response"]["usage"])
        merged.update(usage)
        self._record["response"].update({
            "api_request_id": api_request_id,
            "model_returned": model_returned,
            "stop_reason": stop_reason,
            "stop_details": stop_details,
            "usage": merged,
            "latency_ms": latency_ms,
            "raw_text": raw_text,
        })

    def set_attempt(self, attempt: int,
                    prior_attempt_errors: Optional[list[str]] = None) -> None:
        """How many tries the transport needed, and what the failures were."""
        self._record["request"]["attempt"] = attempt
        self._record["request"]["prior_attempt_errors"] = list(
            prior_attempt_errors or [])

    def set_failure(self, exc: BaseException) -> None:
        """Record why no grade was produced. The log is still saved."""
        record: dict[str, Any] = {
            "error_class": type(exc).__name__,
            "message": str(exc),
        }
        for attr in ("category", "explanation"):
            if hasattr(exc, attr):
                record[attr] = getattr(exc, attr)
        self._record["failure"] = record
        self._record["parsing"].update({
            "strategy_used": "failed",
            "fallback_invoked": False,
            "parse_errors": [str(exc)],
            "parsed": {},
        })
        self.add_error(f"{type(exc).__name__}: {exc}")

    def set_parsing(self, *, strategy_used: str, fallback_invoked: bool,
                    parse_errors: list[str], parsed: dict[str, Any]) -> None:
        self._record["parsing"].update({
            "strategy_used": strategy_used,
            "fallback_invoked": fallback_invoked,
            "parse_errors": parse_errors,
            "parsed": parsed,
        })

    def add_error(self, error: str) -> None:
        self._record["errors"].append(error)

    # ── persistence ──────────────────────────────────────────────────

    def _filepath(self) -> pathlib.Path:
        directory = self.LOG_ROOT / self.pathway
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{self.case_id}__rep{self.replicate}.json"

    def save(self) -> pathlib.Path:
        """Write the log. Raises FileExistsError if it already exists."""
        path = self._filepath()
        if path.exists():
            raise FileExistsError(
                f"Log already exists: {path} (increment the replicate number "
                f"or check for a duplicate run)")
        path.write_text(
            json.dumps(self._record, indent=2, ensure_ascii=False),
            encoding="utf-8")
        return path

    def next_free_replicate(self) -> int:
        """Lowest replicate number whose log file does not yet exist."""
        n = self.replicate
        while (self.LOG_ROOT / self.pathway /
               f"{self.case_id}__rep{n}.json").exists():
            n += 1
        return n

    @property
    def record(self) -> dict[str, Any]:
        return self._record
