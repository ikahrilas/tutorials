"""Tests for conversation state, token budgeting, and error handling.

The OpenAI client is replaced with a stub, so no network calls are made.
"""

from types import SimpleNamespace

import pytest
from openai import APIError

from conversation_manager import ChatCompletionError, ConversationManager


class StubClient:
    """Stands in for `OpenAI`, returning canned replies or raising."""

    def __init__(self, replies=None, error=None, finish_reason="stop"):
        self.replies = list(replies or [])
        self.error = error
        self.finish_reason = finish_reason
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        content = self.replies.pop(0) if self.replies else None
        choice = SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=self.finish_reason
        )
        return SimpleNamespace(choices=[choice])


def make_manager(tmp_path, client, **kwargs):
    manager = ConversationManager(
        api_key="test-key", history_file=tmp_path / "history.json", **kwargs
    )
    manager.client = client
    return manager


def api_error():
    return APIError("boom", request=SimpleNamespace(), body=None)


class TestInitialization:
    def test_zero_temperature_is_respected(self, tmp_path):
        """A falsy-but-valid value must not fall back to the default."""
        manager = make_manager(tmp_path, StubClient(), temperature=0)

        assert manager.temperature == 0

    def test_history_starts_with_system_message(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())

        assert manager.conversation_history == [
            {"role": "system", "content": manager.system_message}
        ]

    def test_custom_system_message_is_reachable_as_persona(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(), system_message="Be terse.")

        assert manager.system_message == "Be terse."
        manager.set_persona("custom")
        assert manager.system_message == "Be terse."

    def test_custom_message_does_not_leak_between_instances(self, tmp_path):
        make_manager(tmp_path, StubClient(), system_message="Be terse.")
        other = make_manager(tmp_path, StubClient())

        assert other.system_messages["custom"] != "Be terse."


class TestPersona:
    def test_unknown_persona_raises_and_leaves_state_untouched(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())
        before = manager.system_message

        with pytest.raises(ValueError):
            manager.set_persona("pirate")

        assert manager.system_message == before

    def test_set_persona_updates_history_in_place(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())

        manager.set_persona("angry_assistant")

        assert manager.conversation_history[0]["content"] == manager.system_message
        assert len(manager.conversation_history) == 1

    def test_empty_custom_message_is_rejected(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())

        with pytest.raises(ValueError):
            manager.set_custom_system_message("   ")


class TestTokenBudget:
    def test_trims_oldest_messages_but_keeps_system_message(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(), token_budget=20)
        manager.conversation_history += [
            {"role": "user", "content": "word " * 50},
            {"role": "assistant", "content": "word " * 50},
        ]

        manager.enforce_token_budget()

        assert manager.conversation_history[0]["role"] == "system"
        assert len(manager.conversation_history) == 1

    def test_handles_null_content(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())
        manager.conversation_history.append({"role": "assistant", "content": None})

        assert manager.total_tokens_used() >= 0


class TestChatCompletion:
    def test_returns_content_and_records_both_turns(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(replies=["hello"]))

        assert manager.chat_completion("hi") == "hello"
        assert manager.conversation_history[-2:] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_persists_history_to_disk(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(replies=["hello"]))

        manager.chat_completion("hi")

        assert manager.history_file.exists()

    def test_per_call_temperature_overrides_default(self, tmp_path):
        client = StubClient(replies=["hello"])
        manager = make_manager(tmp_path, client, temperature=1)

        manager.chat_completion("hi")

        assert client.calls[0]["temperature"] == 1

    def test_api_failure_raises_and_rolls_back_history(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(error=api_error()))
        before = list(manager.conversation_history)

        with pytest.raises(ChatCompletionError):
            manager.chat_completion("hi")

        assert manager.conversation_history == before

    def test_empty_response_raises_rather_than_returning_none(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(replies=[None]))

        with pytest.raises(ChatCompletionError):
            manager.chat_completion("hi")

    def test_truncated_response_explains_max_tokens(self, tmp_path):
        manager = make_manager(
            tmp_path, StubClient(replies=[None], finish_reason="length")
        )

        with pytest.raises(ChatCompletionError, match="max_tokens"):
            manager.chat_completion("hi")


class TestHistoryPersistence:
    def test_missing_file_yields_fresh_history(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())
        manager.history_file = tmp_path / "does_not_exist.json"

        manager.load_conversation_history()

        assert manager.conversation_history[0]["role"] == "system"

    def test_corrupt_file_yields_fresh_history(self, tmp_path):
        manager = make_manager(tmp_path, StubClient())
        manager.history_file.write_text("{not json")

        manager.load_conversation_history()

        assert manager.conversation_history[0]["role"] == "system"

    def test_round_trip(self, tmp_path):
        manager = make_manager(tmp_path, StubClient(replies=["hello"]))
        manager.chat_completion("hi")
        saved = list(manager.conversation_history)

        manager.conversation_history = []
        manager.load_conversation_history()

        assert manager.conversation_history == saved
