"""Reusable prompt building blocks.

Prompts are composed from named blocks (persona, task, constraints, reference,
output format) rather than written as one long string. Keeping the blocks
separate makes it obvious which part to change when output quality drifts, and
lets several pipeline steps share the same wording.
"""

from models import ExtractedProductInfo

# --- Campaign brief -------------------------------------------------------
# Version: 1.0
# Last Updated: 2026-08-27
# Changelog:
# v1.0 (2026-08-27): Initial version

CAMPAIGN_SYSTEM = """
You are a marketing campaign strategist for GlobalJava Roasters,
a premium coffee company focused on quality and sustainability.
"""

CAMPAIGN_TASK = """
Create a marketing campaign brief for the product described in the
reference section below.
"""

CAMPAIGN_CONSTRAINTS = """
Follow these rules strictly:
- Base all claims on information in the reference section
- Use professional but engaging language
- Focus on the product's unique characteristics
- Recommend channels appropriate for premium coffee consumers
"""

CAMPAIGN_OUTPUT = """
Output your response as valid JSON with this exact structure:

{
  "campaign_name": "string",
  "target_audience": "string",
  "key_message": "string",
  "campaign_goal": "awareness" | "engagement" | "conversion",
  "call_to_action": "string",
  "channel_recommendations": ["string", "string", ...]
}

Respond with JSON only. No explanations or markdown formatting.
Keep each string value to 1-2 sentences max.
"""

# --- Product extraction ---------------------------------------------------
# Version: 1.0
# Last Updated: 2026-08-27
# Changelog:
# v1.0 (2026-08-27): Initial version

EXTRACT_SYSTEM = "You are a data extraction specialist."

EXTRACT_TASK = """
Extract key marketing-relevant information from the product factsheet below.
Focus on facts that would matter for a marketing campaign.
"""

EXTRACT_OUTPUT = """
Output JSON with this structure:

{
  "product_name": "string",
  "origin_story": "string",
  "unique_features": ["string", "string", ...],
  "flavor_highlights": ["string", "string", ...],
  "certifications": ["string", "string", ...],
  "price_point": "budget" | "mid-range" | "premium" | "luxury",
  "scarcity_factors": ["string", ...]
}

Use [] for scarcity_factors if there are none.
Respond with JSON only. No explanations or markdown formatting.
"""


def _compose(*blocks: str) -> str:
    """Join prompt blocks into a single prompt, one blank line between each."""
    return "\n\n".join(block.strip() for block in blocks if block.strip())


def build_campaign_prompt(factsheet: str) -> str:
    """Build the single-shot prompt that turns a raw factsheet into a brief."""
    reference = f"Product Information:\n{factsheet.strip()}"
    return _compose(
        CAMPAIGN_SYSTEM,
        CAMPAIGN_TASK,
        CAMPAIGN_CONSTRAINTS,
        reference,
        CAMPAIGN_OUTPUT,
    )


def build_extract_prompt(factsheet: str) -> str:
    """Build the prompt for step one of the pipeline: fact extraction."""
    reference = f"Product Factsheet:\n{factsheet.strip()}"
    return _compose(EXTRACT_SYSTEM, EXTRACT_TASK, reference, EXTRACT_OUTPUT)


def build_draft_prompt(extracted: ExtractedProductInfo) -> str:
    """Build the prompt for step two: draft a brief from extracted facts.

    Passing the validated extraction back as JSON keeps the drafting step
    grounded in facts that have already been checked, instead of re-reading
    the raw factsheet.
    """
    reference = f"Extracted Product Information:\n{extracted.model_dump_json(indent=2)}"
    return _compose(
        CAMPAIGN_SYSTEM,
        CAMPAIGN_TASK,
        CAMPAIGN_CONSTRAINTS,
        reference,
        CAMPAIGN_OUTPUT,
    )


def build_repair_prompt(errors: str, reminders: str = "") -> str:
    """Build a follow-up prompt asking the model to fix its own invalid JSON."""
    return _compose(
        "The JSON you provided had validation errors:",
        str(errors),
        "Please provide corrected JSON that fixes these errors.",
        reminders,
        "Respond with valid JSON only. Keep each string value to 1-2 sentences max.",
    )