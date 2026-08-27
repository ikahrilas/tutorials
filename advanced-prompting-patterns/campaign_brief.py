"""Generate validated marketing campaign briefs from product factsheets.

Two strategies are implemented:

* `generate_campaign_brief` - one prompt, factsheet straight to brief.
* `generate_campaign_brief_pipeline` - extract facts first, then draft from the
  validated extraction. Slower and more expensive, but more reliable on long or
  messy factsheets, where a single prompt tends to drop details.

Both funnel through `generate_validated`, which validates the response against a
schema and feeds any errors back to the model as a repair prompt.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

from pydantic import BaseModel

from conversation_manager import ChatCompletionError, ConversationManager
from models import CampaignBrief, ExtractedProductInfo
from prompt_templates import (
    build_campaign_prompt,
    build_draft_prompt,
    build_extract_prompt,
    build_repair_prompt,
)
from utils import validate_json_output

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3

BRIEF_REMINDERS = """
Remember:
- campaign_goal must be exactly "awareness", "engagement", or "conversion"
- All required fields must be present: campaign_name, target_audience,
  key_message, campaign_goal, call_to_action, channel_recommendations
- channel_recommendations must be a list of strings
"""

EXTRACT_REMINDERS = """
Remember:
- price_point must be exactly "budget", "mid-range", "premium", or "luxury"
- Every list field must be a list of strings; use [] for scarcity_factors if none
"""

SIMPLE_FACTSHEET = """
Product: Ethiopian Yirgacheffe Single-Origin Coffee
Origin: Yirgacheffe region, Ethiopia
Flavor Profile: Bright citrus notes, floral aroma, light body
Processing: Washed
Certifications: Fair Trade, Organic
Price Point: Premium ($18.99/bag)
"""

DETAILED_FACTSHEET = """
Product: Limited Edition Geisha Reserve
Origin: Hacienda La Esmeralda, Panama
Altitude: 1,600-1,800 meters
Processing: Natural, 72-hour fermentation
Flavor Profile: Jasmine, bergamot, white peach, honey sweetness,
silky body, complex finish with hints of tropical fruit
Certifications: Single Estate, Competition Grade
Limited Production: Only 500 bags produced this season
Story: This micro-lot scored 94.1 points in the 2024 Cup of Excellence
competition. The beans come from 30-year-old Geisha trees grown in
volcanic soil. The extended fermentation process was developed
specifically for this lot to enhance the floral characteristics.
Price: $89.99/bag
Previous Customer Feedback: "Best coffee I've ever tasted" - Coffee
Review Magazine. Sold out in 3 days last year.
"""


class GenerationError(RuntimeError):
    """Raised when the model never produced output matching the schema."""


def generate_validated[M: BaseModel](
    prompt: str,
    schema: type[M],
    conversation: ConversationManager | None = None,
    reminders: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> M:
    """Prompt the model until its JSON validates against `schema`.

    Each failed attempt is fed back to the model as a repair prompt describing
    the validation errors. Because the conversation carries its own history, the
    model sees its previous attempt alongside the correction.

    Args:
        prompt: The initial prompt.
        schema: The Pydantic model the response must satisfy.
        conversation: Conversation to use. A fresh one is created if omitted.
        reminders: Extra schema-specific guidance included in repair prompts.
        max_retries: Number of repair attempts after the first response.

    Raises:
        GenerationError: if no attempt produced valid output.
        ChatCompletionError: if the API call itself fails.
    """
    conversation = conversation or ConversationManager()

    response = conversation.chat_completion(prompt)
    success, result = validate_json_output(response, schema)
    if success:
        logger.info("%s valid on first attempt.", schema.__name__)
        return cast(M, result)

    last_error = result
    for attempt in range(1, max_retries + 1):
        logger.info("Attempting repair %d/%d: %s", attempt, max_retries, last_error)

        response = conversation.chat_completion(
            build_repair_prompt(last_error, reminders)
        )
        success, result = validate_json_output(response, schema)
        if success:
            logger.info("Repair succeeded on attempt %d.", attempt)
            return cast(M, result)

        last_error = result

    raise GenerationError(
        f"Could not produce a valid {schema.__name__} after {max_retries} "
        f"repair attempts. Last error: {last_error}"
    )


def extract_product_info(
    factsheet: str, max_retries: int = DEFAULT_MAX_RETRIES
) -> ExtractedProductInfo:
    """Pull structured, marketing-relevant facts out of a raw factsheet."""
    return generate_validated(
        build_extract_prompt(factsheet),
        ExtractedProductInfo,
        reminders=EXTRACT_REMINDERS,
        max_retries=max_retries,
    )


def generate_campaign_brief(
    factsheet: str, max_retries: int = DEFAULT_MAX_RETRIES
) -> CampaignBrief:
    """Generate a brief directly from a factsheet in a single step."""
    return generate_validated(
        build_campaign_prompt(factsheet),
        CampaignBrief,
        reminders=BRIEF_REMINDERS,
        max_retries=max_retries,
    )


def generate_campaign_brief_pipeline(
    factsheet: str, max_retries: int = DEFAULT_MAX_RETRIES
) -> CampaignBrief:
    """Generate a brief in two steps: extract facts, then draft from them."""
    logger.info("Step 1: extracting product information.")
    extracted = extract_product_info(factsheet, max_retries)
    logger.info("Extracted info for: %s", extracted.product_name)

    logger.info("Step 2: drafting campaign brief.")
    return generate_validated(
        build_draft_prompt(extracted),
        CampaignBrief,
        reminders=BRIEF_REMINDERS,
        max_retries=max_retries,
    )


def format_brief(brief: CampaignBrief) -> str:
    """Render a brief as readable console output."""
    return "\n".join(
        [
            f"Campaign: {brief.campaign_name}",
            f"Target:   {brief.target_audience}",
            f"Goal:     {brief.campaign_goal.value}",
            f"Message:  {brief.key_message}",
            f"CTA:      {brief.call_to_action}",
            f"Channels: {', '.join(brief.channel_recommendations)}",
        ]
    )


def main() -> int:
    """Run the pipeline on the sample factsheet. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        brief = generate_campaign_brief_pipeline(DETAILED_FACTSHEET)
    except (GenerationError, ChatCompletionError) as error:
        logger.error("Failed to generate brief: %s", error)
        return 1

    print(format_brief(brief))
    return 0


if __name__ == "__main__":
    sys.exit(main())
