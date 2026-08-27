import logging
import os
import json
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_TEMPERATURE = 1
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_HISTORY_FILE = "conversation_history.json"

class ConversationManager:
    def __init__(
        self,
        api_key=None,
        model=None,
        temperature=None,
        max_tokens=None,
        token_budget=None,
        reasoning_effort=None,
        system_message=None,
        history_file=None
    ):
        self.api_key = api_key if api_key else DEFAULT_API_KEY
        self.model = model if model else DEFAULT_MODEL
        self.temperature = temperature if temperature else DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens else DEFAULT_MAX_TOKENS
        self.token_budget = token_budget if token_budget else DEFAULT_TOKEN_BUDGET
        self.reasoning_effort = reasoning_effort if reasoning_effort else DEFAULT_REASONING_EFFORT
        self.system_messages = {
            "sassy_assistant": "A sassy assistant who is fed up with answering questions.",
            "angry_assistant": "An angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "A thoughtful assistant, always ready to dig deeper. This assistant asks clarifying questions to ensure understanding and approaches problems with a step-by-step methodology.",
            "analytical_assistant": "An analytical assistant that reasons carefully and methodically. Before answering, think through the problem step by step: break it into its component parts, reason about each part explicitly, and only then state a conclusion. Always show this chain-of-thought reasoning process to the user rather than jumping straight to an answer.",
            "custom": "A placeholder for your custom system message.",
        }
        if history_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(DEFAULT_HISTORY_FILE)
            self.history_file = f"{base}_{timestamp}{ext}"
        else:
            self.history_file = history_file
        if system_message:
            # An explicit system_message overrides the default persona; store
            # it under "custom" so it stays reachable via set_persona("custom").
            self.system_messages["custom"] = system_message
            self.system_message = system_message
        else:
            self.system_message = self.system_messages["thoughtful_assistant"]  # default persona
        self.conversation_history = [{"role": "system", "content": self.system_message}] 
        self.client = OpenAI(api_key=self.api_key)
    
    def load_conversation_history(self):
        try:
            with open(self.history_file, "r") as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            logger.error("Invalid JSON format in history file. Starting with empty history.")
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except (OSError, PermissionError):
            logger.error(
                "Could not read the conversation history file due to a "
                "filesystem or permissions issue. Starting with empty history."
            )
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except Exception:
            logger.error(
                "An unexpected error occurred while loading conversation "
                "history. Starting with empty history."
            )
            self.conversation_history = [{"role": "system", "content": self.system_message}]

    def save_conversation_history(self):
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent=4)
        except (OSError, PermissionError):
            logger.error(
                "Could not save the conversation history due to a "
                "filesystem or permissions issue. Your conversation will "
                "not persist between sessions."
            )
        except TypeError:
            logger.error(
                "Conversation history contains data that could not be "
                "serialized to JSON; the history was not saved."
            )
        except Exception:
            logger.error(
                "An unexpected error occurred while saving conversation "
                "history."
            )

    def set_persona(self, persona):
        if persona not in self.system_messages:
            # Not an error condition to log/handle -- it's caller-supplied
            # invalid input, so it should keep surfacing as a normal
            # exception with an unchanged state.
            raise ValueError(f"Invalid persona {persona}. Please select from one of {list(self.system_messages.keys())}")
        previous_message = self.system_message
        try:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        except Exception as e:
            logger.error(f"Failed to switch persona to '{persona}': {e}")
            self.system_message = previous_message
            raise

    def set_custom_system_message(self, message):
        previous_custom = self.system_messages.get("custom")
        previous_message = self.system_message
        try:
            self.system_messages["custom"] = message
            self.system_message = self.system_messages["custom"]
            self.update_system_message_in_history()
        except Exception as e:
            logger.error(f"Failed to set custom system message: {e}")
            self.system_messages["custom"] = previous_custom
            self.system_message = previous_message
            raise

    def update_system_message_in_history(self):
        try:
            if (
                self.conversation_history
                and self.conversation_history[0].get("role") == "system"
            ):
                self.conversation_history[0]["content"] = self.system_message
            else:
                self.conversation_history.insert(
                    0, {"role": "system", "content": self.system_message}
                )
        except Exception as e:
            logger.error(f"Failed to update system message in history: {e}")
            raise

    def count_tokens(self, text):
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoding.encode(text)
        return len(tokens)
    
    def total_tokens_used(self):
        return sum(
            self.count_tokens(message["content"] or "")
            for message in self.conversation_history
        )

    def enforce_token_budget(self):
        try:
            while (
                self.total_tokens_used() >= self.token_budget
                and len(self.conversation_history) > 1
            ):
                self.conversation_history.pop(1)
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while trimming conversation "
                f"history to stay within the token budget: {e}"
            )

    def chat_completion(self, prompt):
        history_snapshot = list(self.conversation_history)
        self.conversation_history.append({"role": "user", "content": prompt})
        self.enforce_token_budget()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort
            )
            choice = response.choices[0]
            if choice.finish_reason == "length":
                logger.warning(
                    "Response truncated before any visible output was "
                    "produced (max_tokens=%s consumed by reasoning/output "
                    "tokens); consider raising max_tokens.",
                    self.max_tokens,
                )
            self.conversation_history.append(
                {"role": "assistant", "content": choice.message.content}
            )
            self.enforce_token_budget()
            total_tokens_used = self.total_tokens_used()
            return choice.message.content
            self.save_conversation_history()
            return choice.message.content, total_tokens_used
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            print(
                "Sorry, I couldn't generate a response right now due to an "
                "unexpected error. Please try again in a moment."
            )
            # Restore the pre-call state, undoing the user append and any
            # token-budget evictions (and any assistant append that may have
            # happened before the failure), rather than assuming the last
            # element is the unanswered user prompt.
            self.conversation_history = history_snapshot
            friendly_message = (
                "I'm unable to generate a response at this time. Please try again shortly."
            )
            return friendly_message, self.total_tokens_used()