"""
Provider-agnostic seam to a language model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum


class UsageSource(StrEnum):
    """
    Where a usage measurement came from.
    """

    PROVIDER = "provider"
    ESTIMATED = "estimated"
    UNREPORTED = "unreported"


def estimate_tokens(text: str) -> int:
    """
    Deterministic fallback when a provider does not report usage.
    """
    return max(1, len(text) // 4)


@dataclass(frozen=True)
class CompletionUsage:
    """
    Usage of exactly one provider call (including a format retry).
    """

    input_tokens: int
    output_tokens: int
    source: UsageSource = UsageSource.PROVIDER

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CompletionResult:
    text: str
    usage: CompletionUsage
    output_token_limit: int | None = None
    """
    Configured output allowance, when reported by the adapter.
    """

    @property
    def output_limit_reached(self) -> bool:
        """
        Whether measured usage reached the configured output allowance.
        """
        return (
            self.output_token_limit is not None
            and self.usage.source is UsageSource.PROVIDER
            and self.usage.output_tokens >= self.output_token_limit
        )


class TransientInfrastructureError(RuntimeError):
    """
    A provider outage after its retry policy was exhausted.

    Only this explicit exception is eligible for exclusion as a lost infrastructure
    episode.
    """


class CompletionClient(ABC):
    """
    Single-shot text completion, the only capability agents may rely on.
    """

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """
        Send one prompt and return the model's raw text reply.
        """

    def complete_with_usage(self, prompt: str) -> CompletionResult:
        """
        Complete and return usage; estimate it for simple clients.
        """
        response = self.complete(prompt)
        return CompletionResult(
            text=response,
            usage=CompletionUsage(
                input_tokens=estimate_tokens(prompt),
                output_tokens=estimate_tokens(response),
                source=UsageSource.ESTIMATED,
            ),
        )

    def complete_with_output_limit(
        self, prompt: str, output_token_limit: int | None
    ) -> CompletionResult:
        """
        Complete with an optional per-call output allowance.
        """
        if output_token_limit is not None:
            raise ValueError("This completion backend cannot override output limits")
        return self.complete_with_usage(prompt)

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Identity of the backing model, recorded into transcripts.
        """


@dataclass
class ScriptedCompletionClient(CompletionClient):
    """
    Replays a fixed list of canned responses and records every prompt it was given.
    """

    responses: list[str]
    """
    Replies handed out in order, one per :meth:`complete` call.
    """

    received_prompts: list[str] = field(default_factory=list)
    """
    Every prompt this client was asked to complete, in call order.
    """

    def complete(self, prompt: str) -> str:
        self.received_prompts.append(prompt)
        if len(self.received_prompts) > len(self.responses):
            raise ScriptExhaustedError(len(self.responses))
        return self.responses[len(self.received_prompts) - 1]

    @property
    def description(self) -> str:
        return "scripted"


class ScriptExhaustedError(Exception):
    """
    Raised when a scripted client receives more calls than it has responses.
    """

    def __init__(self, response_count: int):
        super().__init__(
            f"Scripted client ran out of responses after {response_count} calls."
        )
