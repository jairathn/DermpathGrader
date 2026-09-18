"""The one place a grading request is sent to the API.

Both analyzers build their own prompt, schema and image payload, then
hand them here. This module owns everything about the *call* that the
pathways should not each reimplement: retries, refusal handling,
truncation detection, timing, and the attempt record the case log needs.
It carries no grading logic, which is the line the codebase convention
draws: prompts, schemas and retrieval panels stay per pathway.

Behaviour worth stating explicitly
----------------------------------
- **No model fallbacks.** The SDK offers server-side fallback to another
  model on refusal. It is deliberately not enabled: a batch where some
  cases were graded by a different model is not one experiment, and the
  substitution would be invisible unless someone read every log. A
  refusal is raised, logged as such, and counted.

- **Truncation is an error, not a partial result.** `stop_reason ==
  "max_tokens"` means the JSON is incomplete. Parsing it would recover a
  plausible-looking grade from a response the model did not finish.
  It raises; the fix is config.MAX_TOKENS, not a retry.

- **Retries are bounded and recorded.** Transient failures (connection,
  timeout, rate limit, 5xx, overload) retry with exponential backoff and
  jitter, honouring Retry-After. Every failed attempt's error goes into
  `prior_attempt_errors` in the case log, so a case that needed three
  tries is distinguishable from one that needed one.

- **The scaffold is cached.** The prompt is identical for every case in
  a pathway, so it goes in `system` with a cache breakpoint and the
  per-case images go in the user turn. The cache hit shows up in
  `usage.cache_read_input_tokens`, which is logged; a batch with zero
  cache reads after the first case means something in the prefix is
  varying and should be found.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import anthropic

import config


# ── errors ───────────────────────────────────────────────────────────

class GradingError(RuntimeError):
    """Base for failures that leave a case without a usable grade."""


class GradingRefused(GradingError):
    """The model declined to grade. Carries the API's stop_details."""

    def __init__(self, message: str, category: str | None,
                 explanation: str | None):
        super().__init__(message)
        self.category = category
        self.explanation = explanation


class GradingTruncated(GradingError):
    """Output hit max_tokens; the JSON is incomplete and must not be parsed."""


class GradingSchemaError(GradingError):
    """The response was not valid JSON despite server-side enforcement."""


class GradingTransportError(GradingError):
    """Retries exhausted on transient failures."""


# ── result ───────────────────────────────────────────────────────────

@dataclass
class GradingResponse:
    text: str
    parsed: dict[str, Any]
    message: Any                       # anthropic.types.Message
    attempt: int
    prior_attempt_errors: list[str] = field(default_factory=list)
    latency_ms: int = 0

    @property
    def usage(self) -> dict[str, int]:
        u = self.message.usage
        return {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }

    @property
    def stop_reason(self) -> str:
        return getattr(self.message, "stop_reason", "") or ""

    @property
    def stop_details(self) -> dict[str, Any] | None:
        details = getattr(self.message, "stop_details", None)
        if details is None:
            return None
        return {
            "type": getattr(details, "type", None),
            "category": getattr(details, "category", None),
            "explanation": getattr(details, "explanation", None),
        }


# ── retry policy ─────────────────────────────────────────────────────

MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (anthropic.APIConnectionError,
                        anthropic.APITimeoutError,
                        anthropic.RateLimitError,
                        anthropic.InternalServerError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        # 529 overloaded is not an InternalServerError subclass.
        return exc.status_code in (408, 409, 429, 529) or exc.status_code >= 500
    return False


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after")
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _backoff(attempt: int, exc: BaseException) -> float:
    hinted = _retry_after_seconds(exc)
    if hinted is not None:
        return min(hinted, BACKOFF_CAP_S)
    base = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return base * (0.5 + random.random())   # full jitter, [0.5x, 1.5x]


# ── the call ─────────────────────────────────────────────────────────

def build_request(*, model: str, system_text: str,
                  user_content: list[dict[str, Any]],
                  output_schema: dict[str, Any],
                  max_tokens: int = config.MAX_TOKENS,
                  thinking: dict[str, Any] | None = None,
                  effort: str = config.EFFORT) -> dict[str, Any]:
    """The exact kwargs sent to the API, so logs and tests can see them."""
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{
            "type": "text",
            "text": system_text,
            # Identical across every case in a pathway; the cache hit is
            # what makes a 1,000-call batch affordable.
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{"role": "user", "content": user_content}],
        "output_config": {
            "effort": effort,
            "format": {"type": "json_schema", "schema": output_schema},
        },
    }
    if thinking is not None:
        request["thinking"] = thinking
    return request


def grade(client: anthropic.Anthropic, request: dict[str, Any],
          *, max_attempts: int = MAX_ATTEMPTS,
          sleep=time.sleep) -> GradingResponse:
    """Send one grading request, with bounded retries on transient failure.

    Raises GradingRefused, GradingTruncated, GradingSchemaError or
    GradingTransportError. Non-transient API errors (400, 401, 403, 404)
    propagate unchanged: they are configuration problems, and hiding
    them behind a retry would only delay the diagnosis.
    """
    prior_errors: list[str] = []

    for attempt in range(1, max_attempts + 1):
        started = time.time()
        try:
            with client.messages.stream(**request) as stream:
                message = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 - classified below
            if _is_transient(exc) and attempt < max_attempts:
                prior_errors.append(f"attempt {attempt}: "
                                    f"{type(exc).__name__}: {exc}")
                sleep(_backoff(attempt, exc))
                continue
            if _is_transient(exc):
                raise GradingTransportError(
                    f"{max_attempts} attempts failed; last: "
                    f"{type(exc).__name__}: {exc}") from exc
            raise

        latency_ms = int((time.time() - started) * 1000)
        stop_reason = getattr(message, "stop_reason", "") or ""

        if stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise GradingRefused(
                "model declined to grade this case",
                getattr(details, "category", None),
                getattr(details, "explanation", None))

        if stop_reason == "max_tokens":
            raise GradingTruncated(
                f"output truncated at max_tokens={request['max_tokens']}; "
                f"the JSON is incomplete. Raise config.MAX_TOKENS rather "
                f"than parsing a partial response.")

        text = next((b.text for b in message.content
                     if getattr(b, "type", "") == "text"), "")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            # Should not happen with server-side schema enforcement. One
            # retry is reasonable; a second failure is a real problem.
            if attempt < max_attempts:
                prior_errors.append(
                    f"attempt {attempt}: invalid JSON despite schema: {exc}")
                sleep(_backoff(attempt, exc))
                continue
            raise GradingSchemaError(
                f"invalid JSON after {attempt} attempts: {exc}") from exc

        return GradingResponse(text=text, parsed=parsed, message=message,
                               attempt=attempt,
                               prior_attempt_errors=prior_errors,
                               latency_ms=latency_ms)

    raise GradingTransportError("unreachable")   # pragma: no cover
