"""Tests for the generate-validate-repair loop.

`ConversationManager` is replaced with a fake that replays scripted responses,
so the retry behavior can be tested without calling the API.
"""

import json

import pytest

from campaign_brief import GenerationError, generate_validated
from models import CampaignBrief

VALID_JSON = json.dumps(
    {
        "campaign_name": "Bloom in Every Cup",
        "target_audience": "Specialty coffee enthusiasts.",
        "key_message": "Floral complexity from a single estate.",
        "campaign_goal": "awareness",
        "call_to_action": "Order your bag today.",
        "channel_recommendations": ["Instagram"],
    }
)

INVALID_GOAL_JSON = VALID_JSON.replace('"awareness"', '"sales"')


class FakeConversation:
    """Replays scripted responses and records the prompts it received."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def chat_completion(self, prompt, temperature=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_valid_first_response_makes_no_repair_call():
    conversation = FakeConversation([VALID_JSON])

    brief = generate_validated("prompt", CampaignBrief, conversation)

    assert isinstance(brief, CampaignBrief)
    assert len(conversation.prompts) == 1


def test_invalid_response_is_repaired():
    conversation = FakeConversation([INVALID_GOAL_JSON, VALID_JSON])

    brief = generate_validated("prompt", CampaignBrief, conversation)

    assert brief.campaign_name == "Bloom in Every Cup"
    assert len(conversation.prompts) == 2


def test_repair_prompt_reports_the_error_and_reminders():
    conversation = FakeConversation([INVALID_GOAL_JSON, VALID_JSON])

    generate_validated("prompt", CampaignBrief, conversation, reminders="Be careful.")

    repair_prompt = conversation.prompts[1]
    assert "campaign_goal" in repair_prompt
    assert "Be careful." in repair_prompt


def test_markdown_fenced_response_is_accepted():
    conversation = FakeConversation([f"```json\n{VALID_JSON}\n```"])

    brief = generate_validated("prompt", CampaignBrief, conversation)

    assert isinstance(brief, CampaignBrief)


def test_exhausted_retries_raise_with_last_error():
    conversation = FakeConversation([INVALID_GOAL_JSON] * 3)

    with pytest.raises(GenerationError, match="CampaignBrief"):
        generate_validated("prompt", CampaignBrief, conversation, max_retries=2)

    assert len(conversation.prompts) == 3


def test_max_retries_zero_makes_a_single_attempt():
    conversation = FakeConversation([INVALID_GOAL_JSON])

    with pytest.raises(GenerationError):
        generate_validated("prompt", CampaignBrief, conversation, max_retries=0)

    assert len(conversation.prompts) == 1
