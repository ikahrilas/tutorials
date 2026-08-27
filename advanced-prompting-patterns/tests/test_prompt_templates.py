"""Tests that composed prompts contain what the schemas require."""

from models import ExtractedProductInfo, PricePoint
from prompt_templates import (
    build_campaign_prompt,
    build_draft_prompt,
    build_extract_prompt,
    build_repair_prompt,
)

FACTSHEET = "Product: House Blend\nPrice: $12.99/bag"

EXTRACTED = ExtractedProductInfo(
    product_name="House Blend",
    origin_story="A blend of Colombian and Brazilian beans.",
    unique_features=["Balanced body"],
    flavor_highlights=["Chocolate", "Nuts"],
    certifications=[],
    price_point=PricePoint.BUDGET,
    scarcity_factors=[],
)


def test_campaign_prompt_includes_factsheet_and_schema():
    prompt = build_campaign_prompt(FACTSHEET)

    assert "House Blend" in prompt
    for field in ("campaign_name", "campaign_goal", "channel_recommendations"):
        assert field in prompt


def test_prompts_use_single_braces():
    """The JSON examples are plain strings, so braces must not be escaped."""
    for prompt in (build_campaign_prompt(FACTSHEET), build_extract_prompt(FACTSHEET)):
        assert "{{" not in prompt
        assert "}}" not in prompt


def test_extract_prompt_lists_every_schema_field():
    prompt = build_extract_prompt(FACTSHEET)

    for field in ExtractedProductInfo.model_fields:
        assert field in prompt, f"{field} missing from extraction prompt"


def test_draft_prompt_embeds_extracted_facts():
    prompt = build_draft_prompt(EXTRACTED)

    assert "House Blend" in prompt
    assert "budget" in prompt


def test_repair_prompt_includes_errors_and_reminders():
    prompt = build_repair_prompt("campaign_goal: invalid value", "Use lowercase.")

    assert "campaign_goal: invalid value" in prompt
    assert "Use lowercase." in prompt


def test_blocks_are_separated_by_blank_lines():
    prompt = build_extract_prompt(FACTSHEET)

    assert "\n\n" in prompt
    assert not prompt.startswith("\n")
