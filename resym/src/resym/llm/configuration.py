"""
Experiment configuration for the language-model layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from typing_extensions import Any, Self

from resym.llm.client import CompletionClient


@dataclass(frozen=True)
class ExperimentConfiguration:
    """
    All tunable knobs of one language-model experiment.
    """

    agent: dict[str, Any] = field(default_factory=dict)

    tasks: dict[str, Any] = field(default_factory=dict)

    structured_maximum_attempts: int = 3
    """
    Attempt budget of one structured query (extract + validate + re-ask).
    """

    @classmethod
    def load(cls, path: Path) -> Self:
        """
        Read a configuration from a JSON file (see config/llm.example.json).
        """
        return cls(**json.loads(path.read_text()))

    def to_metadata(self) -> dict[str, Any]:
        """
        Return the non-secret model settings to preserve with a run.
        """
        return {
            "agent": dict(self.agent),
            "tasks": dict(self.tasks),
            "structured_maximum_attempts": self.structured_maximum_attempts,
        }


def build_completion_client(
    configuration: ExperimentConfiguration,
) -> CompletionClient:
    """
    The real client for a configuration.

    .. note:: The adapter module is imported here, not at the top of the
       module: llm-agent-kit requires the ``openai`` package, which is not
       installed inside the cram container. Everything that only ever uses
       the scripted client stays importable there.
    """
    from resym.llm.llm_agent_kit_client import (
        LLMAgentKitCompletionClient,
    )

    return LLMAgentKitCompletionClient(configuration)
