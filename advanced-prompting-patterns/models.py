"""Pydantic schemas describing the structured output we ask the model for.

Each schema is the single source of truth for a step in the pipeline: it drives
both the JSON shape described in the prompt and the validation that decides
whether a response needs repairing.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class CampaignGoal(StrEnum):
    """The primary objective a campaign is optimizing for."""

    AWARENESS = "awareness"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"


class PricePoint(StrEnum):
    """Where a product sits in the market, used to steer audience targeting."""

    BUDGET = "budget"
    MID_RANGE = "mid-range"
    PREMIUM = "premium"
    LUXURY = "luxury"


class CampaignBrief(BaseModel):
    """A complete marketing campaign brief."""

    model_config = ConfigDict(extra="forbid")

    campaign_name: str
    target_audience: str
    key_message: str
    campaign_goal: CampaignGoal
    call_to_action: str
    channel_recommendations: list[str]


class ExtractedProductInfo(BaseModel):
    """Marketing-relevant facts pulled out of a raw product factsheet."""

    model_config = ConfigDict(extra="forbid")

    product_name: str
    origin_story: str
    unique_features: list[str]
    flavor_highlights: list[str]
    certifications: list[str]
    price_point: PricePoint
    scarcity_factors: list[str] = Field(default_factory=list)
