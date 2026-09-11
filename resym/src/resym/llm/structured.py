"""
Structured output over the completion seam.

An agent asks for a Pydantic-typed answer; the completer sends the prompt, extracts the
JSON payload from the raw reply, and validates it. A malformed reply is not fatal: the
validation error is appended to the prompt and the model is asked again, up to a bounded
number of attempts. Every attempt is recorded in the transcript.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError
from typing_extensions import Callable, Optional, Type, TypeVar

from resym.llm.client import CompletionClient, CompletionUsage
from resym.llm.transcript import (
    LanguageModelExchange,
    TranscriptRecorder,
)

OutputT = TypeVar("OutputT", bound=BaseModel)


class MalformedResponseError(Exception):
    """
    Raised when a reply contains no safely recoverable JSON payload.
    """

    def __init__(self, response: str, reason: str = "no JSON object or array found"):
        super().__init__(f"Malformed JSON reply ({reason}): {response[:200]!r}")


class StructuredOutputRetriesExceededError(Exception):
    """
    Raised when the model keeps producing unparsable or invalid replies.
    """

    def __init__(self, agent_name: str, attempts: int, last_error: str):
        super().__init__(
            f"Agent '{agent_name}' got no valid structured reply in {attempts} "
            f"attempts; last error: {last_error}"
        )


@dataclass(frozen=True)
class _JsonExtraction:
    payload: str
    recovery: Optional[str] = None


def _extract_json(text: str) -> _JsonExtraction:
    """
    Extract one object/array and conservatively close its outer container.

    The recovery deliberately handles only the failure: a complete nested
    value followed by one missing outer ``}`` or ``]``.
    """
    matching = {"{": "}", "[": "]"}
    search_from = 0
    saw_opening = False
    while search_from < len(text):
        starts = [
            index
            for index in (
                text.find("{", search_from),
                text.find("[", search_from),
            )
            if index >= 0
        ]
        if not starts:
            break
        saw_opening = True
        start = min(starts)
        expected_closers: list[str] = []
        in_string = False
        escaped = False
        end: Optional[int] = None
        mismatch = False

        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in matching:
                expected_closers.append(matching[character])
            elif character in ("}", "]"):
                if not expected_closers or character != expected_closers[-1]:
                    mismatch = True
                    break
                expected_closers.pop()
                if not expected_closers:
                    end = index + 1
                    break

        if end is not None:
            candidate = text[start:end]
            try:
                json.loads(candidate)
            except json.JSONDecodeError:
                search_from = end
                continue
            return _JsonExtraction(candidate)

        if mismatch:
            search_from = start + 1
            continue
        if in_string:
            raise MalformedResponseError(text, "unterminated string")

        candidate = text[start:].rstrip()
        if len(expected_closers) == 1 and candidate and candidate[-1] in ("}", "]"):
            repaired = candidate + expected_closers[0]
            try:
                json.loads(repaired)
            except json.JSONDecodeError:
                pass
            else:
                return _JsonExtraction(
                    repaired, recovery="closed_one_trailing_container"
                )
        raise MalformedResponseError(text, "incomplete or ambiguous payload")

    reason = "invalid JSON syntax" if saw_opening else "no JSON object or array found"
    raise MalformedResponseError(text, reason)


def extract_json_block(text: str) -> str:
    """
    Return the outermost safely extractable JSON object or array.
    """
    return _extract_json(text).payload


@dataclass
class StructuredCompleter:
    """
    Asks the model for answers that validate against a Pydantic type, feeding validation
    errors back to the model as retry context.
    """

    client: CompletionClient
    """
    The completion backend, real or scripted.
    """

    transcript: TranscriptRecorder
    """
    Where every attempt is recorded.
    """

    maximum_attempts: int = 3
    """
    How many replies to accept-or-reject before giving up.
    """

    def complete(
        self,
        agent_name: str,
        prompt: str,
        output_type: Type[OutputT],
        charge_usage: Optional[Callable[[CompletionUsage], None]] = None,
        exchange_sink: Optional[Callable[[LanguageModelExchange], None]] = None,
    ) -> OutputT:
        """
        One structured query; returns the validated output object.
        """
        current_prompt = prompt
        last_error = ""
        for attempt in range(1, self.maximum_attempts + 1):
            completion = self.client.complete_with_usage(current_prompt)
            response = completion.text
            parsed, error, recovery = self._parse(response, output_type)
            exchange = LanguageModelExchange(
                agent_name=agent_name,
                model_description=self.client.description,
                prompt=current_prompt,
                response=response,
                attempt=attempt,
                parse_error=error,
                parse_recovery=recovery,
                input_tokens=completion.usage.input_tokens,
                output_tokens=completion.usage.output_tokens,
                usage_source=completion.usage.source,
            )
            self.transcript.record(exchange)
            if exchange_sink is not None:
                exchange_sink(exchange)
            # Charge every provider call, including malformed replies. Do this
            # after recording so a budget crossing remains auditable.
            if charge_usage is not None:
                charge_usage(completion.usage)
            if parsed is not None:
                return parsed
            last_error = error
            current_prompt = (
                f"{prompt}\n\n"
                f"Your previous reply could not be used:\n{error}\n"
                f"Reply again with ONLY a JSON document matching the requested schema."
            )
        raise StructuredOutputRetriesExceededError(
            agent_name, self.maximum_attempts, last_error
        )

    def _parse(
        self, response: str, output_type: Type[OutputT]
    ) -> tuple[OutputT | None, str | None, str | None]:
        """
        Extract-and-validate; also reports any safe syntax recovery.
        """
        try:
            extraction = _extract_json(response)
        except MalformedResponseError as error:
            return None, str(error), None
        try:
            return (
                output_type.model_validate_json(extraction.payload),
                None,
                extraction.recovery,
            )
        except ValidationError as error:
            return None, str(error), extraction.recovery
        except json.JSONDecodeError as error:
            return None, str(error), extraction.recovery
