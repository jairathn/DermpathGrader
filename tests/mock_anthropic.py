"""A stand-in for anthropic.Anthropic that never touches the network.

Scripted per call: each entry in `script` is either a Message-like
object to return, or an exception to raise. Records every request so a
test can assert on exactly what would have been sent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeUsage:
    input_tokens: int = 15000
    output_tokens: int = 900
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeStopDetails:
    type: str = "refusal"
    category: str | None = None
    explanation: str | None = None


@dataclass
class FakeMessage:
    content: list
    stop_reason: str = "end_turn"
    stop_details: Any = None
    usage: FakeUsage = field(default_factory=FakeUsage)
    id: str = "msg_fake_000001"
    model: str = "claude-opus-5"


def message_with_json(payload: dict[str, Any], **kw) -> FakeMessage:
    return FakeMessage(content=[FakeTextBlock(json.dumps(payload))], **kw)


class _Stream:
    def __init__(self, outcome):
        self._outcome = outcome

    def __enter__(self):
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._outcome


class _Messages:
    def __init__(self, script: list):
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        if not self.script:
            raise AssertionError("mock client script exhausted")
        return _Stream(self.script.pop(0))

    def count_tokens(self, **kwargs):
        # Rough, deterministic: enough for a dry-run estimate to be tested.
        n = 0
        for block in kwargs.get("system", []) or []:
            n += len(block.get("text", "")) // 4
        for msg in kwargs.get("messages", []):
            for block in msg.get("content", []):
                if block.get("type") == "text":
                    n += len(block["text"]) // 4
                elif block.get("type") == "image":
                    n += 1500
        return type("Count", (), {"input_tokens": n})()


class FakeAnthropic:
    """Drop-in for anthropic.Anthropic in tests."""

    def __init__(self, script: list | None = None, **_ignored):
        self.messages = _Messages(script or [])

    @property
    def requests(self) -> list[dict[str, Any]]:
        return self.messages.requests
