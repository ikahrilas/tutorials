from pydantic import BaseModel, ConfigDict
from typing import List
from enum import Enum

class CampaignGoal(str, Enum):
    AWARENESS = "awareness"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"

class CampaignBrief(BaseModel):
    model_config = ConfigDict(extra="forbid")
    campaign_name: str
    target_audience: str
    key_message: str
    campaign_goal: CampaignGoal
    call_to_action: str
    channel_recommendations: List[str]