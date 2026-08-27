"""Tests for JSON extraction and schema validation.

These run entirely offline; nothing here touches the OpenAI API.
"""

import json

import pytest

from models import CampaignBrief, CampaignGoal
from utils import extract_json, validate_json_output

VALID_BRIEF = {
    "campaign_name": "Bloom in Every Cup",
    "target_audience": "Specialty coffee enthusiasts.",
    "key_message": "Floral complexity from a single Ethiopian estate.",
    "campaign_goal": "awareness",
    "call_to_action": "Order your bag today.",
    "channel_recommendations": ["Instagram", "Email"],
}


class TestExtractJson:
    def test_parses_bare_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fence(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_strips_unlabeled_fence(self):
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_ignores_surrounding_prose(self):
        assert extract_json('Here you go:\n{"a": 1}\nHope that helps!') == {"a": 1}

    def test_handles_nested_objects(self):
        assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}

    def test_raises_when_no_object_present(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("no json here")

    def test_raises_on_top_level_array(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("[1, 2, 3]")


class TestValidateJsonOutput:
    def test_valid_output_returns_model(self):
        success, result = validate_json_output(json.dumps(VALID_BRIEF), CampaignBrief)

        assert success
        assert isinstance(result, CampaignBrief)
        assert result.campaign_goal is CampaignGoal.AWARENESS

    def test_missing_field_reports_validation_error(self):
        payload = {k: v for k, v in VALID_BRIEF.items() if k != "key_message"}

        success, result = validate_json_output(json.dumps(payload), CampaignBrief)

        assert not success
        assert "key_message" in result

    def test_invalid_enum_value_reports_validation_error(self):
        payload = VALID_BRIEF | {"campaign_goal": "sales"}

        success, result = validate_json_output(json.dumps(payload), CampaignBrief)

        assert not success
        assert "Validation errors" in result

    def test_extra_field_is_rejected(self):
        payload = VALID_BRIEF | {"budget": "$10k"}

        success, result = validate_json_output(json.dumps(payload), CampaignBrief)

        assert not success

    def test_malformed_json_reports_parsing_error(self):
        success, result = validate_json_output("{not json", CampaignBrief)

        assert not success
        assert "JSON parsing error" in result
