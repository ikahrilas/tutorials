"""A thin, stateful wrapper around the OpenAI chat completions API.

`ConversationManager` owns three things the prompting patterns in this repo
depend on: the running message history, a token budget that trims that history
before it overflows the context window, and the system message ("persona") that
sits at the head of the history.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_HISTORY_FILE = "conversation_history.json"
FALLBACK_ENCODING = "cl100k_base"

Message = dict[str, Any]


class ChatCompletionError(RuntimeError):
    """Raised when the API call fails and no response can be returned."""


def _timestamped_history_path() -> Path:
    """Build a unique history filename so runs don't overwrite each other."""
    path = Path(DEFAULT_HISTORY_FILE)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


class ConversationManager:
    """Tracks a conversation and sends it to the model within a token budget."""

    #: Named system messages. `set_persona` selects one by key.
    SYSTEM_MESSAGES: dict[str, str] = {
        "sassy_assistant": "A sassy assistant who is fed up with answering questions.",
        "angry_assistant": "An angry assistant that likes yelling in all caps.",
        "thoughtful_assistant": (
            "A thoughtful assistant, always ready to dig deeper. This assistant asks "
            "clarifying questions to ensure understanding and approaches problems "
            "with a step-by-step methodology."
        ),
        "analytical_assistant": (
            "An analytical assistant that reasons carefully and methodically. Before "
            "answering, think through the problem step by step: break it into its "
            "component parts, reason about each part explicitly, and only then state "
            "a conclusion. Always show this chain-of-thought reasoning process to the "
            "user rather than jumping straight to an answer."
        ),
        "custom": "A placeholder for your custom system message.",
    }
    DEFAULT_PERSONA = "thoughtful_assistant"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        system_message: str | None = None,
        history_file: str | Path | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.reasoning_effort = reasoning_effort

        # Per-instance copy so one manager's custom message can't leak into
        # every other manager via the class-level dict.
        self.system_messages = dict(self.SYSTEM_MESSAGES)

        self.history_file = (
            Path(history_file) if history_file else _timestamped_history_path()
        )

        if system_message:
            # An explicit system_message overrides the default persona; store it
            # under "custom" so it stays reachable via set_persona("custom").
            self.system_messages["custom"] = system_message
            self.system_message = system_message
        else:
            self.system_message = self.system_messages[self.DEFAULT_PERSONA]

        self.conversation_history: list[Message] = [
            {"role": "system", "content": self.system_message}
        ]

        self._encoding: tiktoken.Encoding | None = None
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # --- History persistence ---------------------------------------------

    def _fresh_history(self) -> list[Message]:
        return [{"role": "system", "content": self.system_message}]

    def load_conversation_history(self) -> None:
        """Load history from disk, falling back to a fresh history on failure."""
        try:
            with open(self.history_file, encoding="utf-8") as file:
                self.conversation_history = json.load(file)
            return
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            logger.error(
                "Invalid JSON in %s. Starting with empty history.", self.history_file
            )
        except OSError:
            logger.error(
                "Could not read %s due to a filesystem or permissions issue. "
                "Starting with empty history.",
                self.history_file,
            )
        self.conversation_history = self._fresh_history()

    def save_conversation_history(self) -> None:
        """Write history to disk. Failures are logged, never raised.

        Persistence is a convenience here, so a write failure should not take
        down an otherwise successful conversation.
        """
        try:
            with open(self.history_file, "w", encoding="utf-8") as file:
                json.dump(self.conversation_history, file, indent=4)
        except OSError:
            logger.error(
                "Could not save conversation history to %s; this conversation "
                "will not persist between sessions.",
                self.history_file,
            )
        except TypeError:
            logger.error(
                "Conversation history contains data that could not be serialized "
                "to JSON; the history was not saved."
            )

    # --- Persona ----------------------------------------------------------

    def set_persona(self, persona: str) -> None:
        """Switch to one of the named system messages.

        Raises:
            ValueError: if `persona` is not a known key.
        """
        if persona not in self.system_messages:
            raise ValueError(
                f"Invalid persona {persona!r}. Choose one of "
                f"{sorted(self.system_messages)}"
            )
        self.system_message = self.system_messages[persona]
        self._update_system_message_in_history()

    def set_custom_system_message(self, message: str) -> None:
        """Set an arbitrary system message and make it the active one.

        Raises:
            ValueError: if `message` is empty.
        """
        if not message.strip():
            raise ValueError("Custom system message cannot be empty.")
        self.system_messages["custom"] = message
        self.system_message = message
        self._update_system_message_in_history()

    def _update_system_message_in_history(self) -> None:
        """Keep the system message at index 0 of the history in sync."""
        history = self.conversation_history
        if history and history[0]["role"] == "system":
            history[0]["content"] = self.system_message
        else:
            history.insert(0, {"role": "system", "content": self.system_message})

    # --- Token accounting -------------------------------------------------

    @property
    def encoding(self) -> tiktoken.Encoding:
        """The tokenizer for `self.model`, resolved once and reused."""
        if self._encoding is None:
            try:
                self._encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                logger.debug(
                    "No tiktoken encoding for %s; falling back to %s.",
                    self.model,
                    FALLBACK_ENCODING,
                )
                self._encoding = tiktoken.get_encoding(FALLBACK_ENCODING)
        return self._encoding

    def count_tokens(self, text: str) -> int:
        """Approximate the token count of a string."""
        return len(self.encoding.encode(text))

    def total_tokens_used(self) -> int:
        """Approximate the token count of the entire conversation history."""
        return sum(
            self.count_tokens(message.get("content") or "")
            for message in self.conversation_history
        )

    def enforce_token_budget(self) -> None:
        """Drop the oldest non-system messages until we're under budget."""
        while (
            len(self.conversation_history) > 1
            and self.total_tokens_used() >= self.token_budget
        ):
            self.conversation_history.pop(1)

    # --- Completion -------------------------------------------------------

    def chat_completion(self, prompt: str, temperature: float | None = None) -> str:
        """Send `prompt` in the context of the conversation and return the reply.

        On failure the history is rolled back to its pre-call state, so a failed
        turn leaves no dangling user message behind.

        Raises:
            ChatCompletionError: if the API call fails or returns no content.
        """
        history_snapshot = list(self.conversation_history)
        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                # The current model only supports the default temperature (1).
                # Preserve that constraint by ignoring any custom temperature value.
                temperature=self.temperature if temperature is None or self.model == DEFAULT_MODEL else temperature,
                max_completion_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
        except OpenAIError as error:
            self.conversation_history = history_snapshot
            logger.error("Chat completion failed: %s", error)
            raise ChatCompletionError(str(error)) from error

        choice = response.choices[0]
        content = choice.message.content

        if choice.finish_reason == "length" and not content:
            self.conversation_history = history_snapshot
            raise ChatCompletionError(
                f"Response was truncated before producing any output "
                f"(max_tokens={self.max_tokens} was consumed by reasoning tokens). "
                f"Try raising max_tokens."
            )
        if not content:
            self.conversation_history = history_snapshot
            raise ChatCompletionError(
                f"Model returned no content (finish_reason={choice.finish_reason})."
            )

        self.conversation_history.append({"role": "assistant", "content": content})
        self.enforce_token_budget()
        self.save_conversation_history()
        return content
