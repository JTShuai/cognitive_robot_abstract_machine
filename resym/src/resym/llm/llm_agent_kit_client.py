"""
The real completion backend: llm-agent-kit.

llm-agent-kit owns provider selection (``CLIENT_TYPE``/``API_KEY``/``BASE_URL``
from the environment or a ``.env`` file), API-level retries, and usage
accounting. This adapter reduces it to the
:class:`~resym.llm.client.CompletionClient` seam. The
configuration sections are validated by llm-agent-kit's own Pydantic models,
so its parameter schema is reused instead of mirrored.

.. warning:: Importing this module requires the ``openai`` package (an
   llm-agent-kit dependency), which the cram container does not have. Go
   through :func:`resym.llm.configuration.build_completion_client`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import TYPE_CHECKING

from llm_agent_kit import (
    APICallError,
    DefaultRecorder,
    LLMAgent,
    LLMAgentConfig,
    LLMTasksConfig,
    MaxRetriesExceededError,
)

from resym.llm.client import (
    CompletionClient,
    CompletionResult,
    CompletionUsage,
    TransientInfrastructureError,
    UsageSource,
)

if TYPE_CHECKING:
    from resym.llm.configuration import ExperimentConfiguration


class _PerCallUsageRecorder(DefaultRecorder):
    """
    Captures llm-agent-kit's provider usage for the most recent call.
    """

    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.usage: CompletionUsage | None = None

    def on_call_start(self, model: str, messages: list) -> None:
        super().on_call_start(model, messages)
        self.usage = None

    def on_call_end(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_seconds: float,
        response: str,
    ) -> None:
        super().on_call_end(
            model, input_tokens, output_tokens, latency_seconds, response
        )
        self.usage = CompletionUsage(input_tokens, output_tokens, UsageSource.PROVIDER)


@dataclass
class LLMAgentKitCompletionClient(CompletionClient):
    """
    One-shot completions through a configured :class:`llm_agent_kit.LLMAgent`.
    """

    configuration: ExperimentConfiguration
    """
    The experiment's raw ``agent``/``tasks`` sections.
    """

    agent: LLMAgent = field(init=False)
    """
    The underlying llm-agent-kit agent, built from the configuration.
    """

    usage_recorder: _PerCallUsageRecorder = field(init=False)

    def __post_init__(self):
        model_configuration = LLMAgentConfig.model_validate(self.configuration.agent)
        self.usage_recorder = _PerCallUsageRecorder(model_configuration.model)
        self.agent = LLMAgent(
            llm_model_config=model_configuration,
            task_config=LLMTasksConfig.model_validate(self.configuration.tasks),
            recorder=self.usage_recorder,
        )

    def complete(self, prompt: str) -> str:
        return self.complete_with_usage(prompt).text

    def complete_with_usage(self, prompt: str) -> CompletionResult:
        try:
            response = self.agent.call(prompt)
        except MaxRetriesExceededError as error:
            raise TransientInfrastructureError(str(error)) from error
        # APICallError means a non-retryable request/config/program error. It
        # intentionally propagates and must not disappear from denominators.
        except APICallError:
            raise
        usage = self.usage_recorder.usage
        if usage is None:  # Defensive: a provider adapter violated recorder API.
            raise RuntimeError("llm-agent-kit returned without token usage")
        if not isinstance(response, str):
            raise TypeError("single-completion client received multiple responses")
        return CompletionResult(response, usage)

    @property
    def description(self) -> str:
        return f"llm-agent-kit:{self.agent.model}"
