"""
Transcripts of agent-model interactions.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from typing_extensions import Optional

from resym.llm.client import UsageSource


@dataclass(frozen=True)
class LanguageModelExchange:
    """
    One prompt/response round between an agent and the model.
    """

    agent_name: str
    """
    Name of the agent that issued the prompt.
    """

    model_description: str
    """
    Identity of the backing model that answered.
    """

    prompt: str
    """
    The full prompt that was sent.
    """

    response: str
    """
    The raw text the model returned.
    """

    attempt: int
    """
    1-based attempt index within one structured query (retries increment it).
    """

    parse_error: Optional[str] = None
    """
    Validation error of this response, if structured parsing failed.
    """

    parse_recovery: Optional[str] = None
    """
    Conservative syntax recovery applied before validation, if any.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    usage_source: UsageSource = UsageSource.UNREPORTED


@dataclass
class TranscriptRecorder:
    """
    Collects exchanges in memory and, when given a path, appends each one as a JSON line
    for post-hoc debugging.
    """

    path: Optional[Path] = None
    """
    JSONL file the transcript is appended to; in-memory only when ``None``.
    """

    exchanges: list[LanguageModelExchange] = field(default_factory=list)
    """
    All recorded exchanges, in order.
    """

    context: dict = field(default_factory=dict)
    """
    Fields stamped onto every recorded entry (see :meth:`scoped`).
    """

    @contextmanager
    def scoped(self, **fields):
        """
        Stamp every exchange recorded inside the block with ``fields`` (``None`` values
        are dropped); restores the previous context on exit, so scopes nest.
        """
        previous = self.context
        self.context = {
            **previous,
            **{key: value for key, value in fields.items() if value is not None},
        }
        try:
            yield self
        finally:
            self.context = previous

    def record(self, exchange: LanguageModelExchange) -> None:
        """
        Store one exchange and append it to the transcript file if configured.
        """
        self.exchanges.append(exchange)
        if self.path is None:
            return
        entry = {
            **self.context,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
            "agent_name": exchange.agent_name,
            "model": exchange.model_description,
            "attempt": exchange.attempt,
            "prompt": exchange.prompt,
            "response": exchange.response,
            "parse_error": exchange.parse_error,
            "parse_recovery": exchange.parse_recovery,
            "input_tokens": exchange.input_tokens,
            "output_tokens": exchange.output_tokens,
            "usage_source": exchange.usage_source,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def responses_of(self, agent_name: str) -> list[str]:
        """
        The raw responses a given agent received, for replay scripting.
        """
        return [
            exchange.response
            for exchange in self.exchanges
            if exchange.agent_name == agent_name
        ]
