from conversation_manager import ConversationManager
from utils import validate_json_output
from models import CampaignGoal, CampaignBrief

conversation = ConversationManager(max_tokens=5000)

factsheet = """
Product: Ethiopian Yirgacheffe Single-Origin Coffee
Origin: Yirgacheffe region, Ethiopia
Flavor Profile: Bright citrus notes, floral aroma, light body
Processing: Washed
Certifications: Fair Trade, Organic
Price Point: Premium ($18.99/bag)
"""

SYSTEM_PROMPT = """
You are a marketing campaign strategist for GlobalJava Roasters,
a premium coffee company focused on quality and sustainability.
"""

TASK = """
Create a marketing campaign brief for the product described in the
reference section below.
"""

CONSTRAINTS = """
Follow these rules strictly:
- Base all claims on information in the reference section
- Use professional but engaging language
- Focus on the product's unique characteristics
- Recommend channels appropriate for premium coffee consumers
"""

REFERENCE = f"""
Product Information:
{factsheet}
"""

OUTPUT_FORMAT = """
Output your response as valid JSON with this exact structure:

{{
  "campaign_name": "string",
  "target_audience": "string",
  "key_message": "string",
  "campaign_goal": "awareness" | "engagement" | "conversion",
  "call_to_action": "string",
  "channel_recommendations": ["string", "string", ...]
}}

Respond with JSON only. No explanations or markdown formatting. Keep each string value to 1–2 sentences max.
"""

def build_campaign_prompt(factsheet):
    """Compose a structured prompt from reusable blocks."""
    return f"""
{SYSTEM_PROMPT}

{TASK}

{CONSTRAINTS}

{REFERENCE}

{OUTPUT_FORMAT}
""".strip()

def generate_campaign_brief(factsheet, max_retries=3):
    """Generate a validated campaign brief with automatic repair."""

    conversation = ConversationManager()

    initial_prompt = build_campaign_prompt(factsheet=factsheet)

    response = conversation.chat_completion(initial_prompt)
    print("Initial response:")
    print(response)
    print()

    success, result = validate_json_output(response, CampaignBrief)

    # If valid on first try, return it
    if success:
        print("✓ Valid on first attempt!")
        return result

    # Otherwise, try to repair
    print(f"✗ Validation failed: {result}")
    print()

    retries = 0
    while retries < max_retries:
        print(f"Attempting repair {retries + 1}/{max_retries}...")

        repair_prompt = f"""
             The JSON you provided had validation errors:
             
             {result}
             
             Please provide corrected JSON that fixes these errors. Remember:
             - campaign_goal must be exactly "awareness", "engagement", or "conversion"
             - All required fields must be present: campaign_name, target_audience, key_message, campaign_goal, call_to_action, channel_recommendations
             - channel_recommendations must be a list of strings
             
             Respond with valid JSON only. Keep each string value to 1–2 sentences max.
        """

        response = conversation.chat_completion(repair_prompt)
        print("Repair response:")
        print(response)
        print()

        success, result = validate_json_output(response, CampaignBrief)

        if success:
            print("✓ Repair successful!")
            return result

        print(f"✗ Still invalid: {result}")
        print()
        retries += 1

    # If we exhausted retries, raise an error
    raise ValueError(f"Could not generate valid campaign brief after {max_retries} attempts. Last error: {result}")

try:
    brief = generate_campaign_brief(factsheet)
    print("✓ Generated valid campaign brief!")
    print(f"Campaign: {brief.campaign_name}")
    print(f"Target: {brief.target_audience}")
    print(f"Goal: {brief.campaign_goal.value}")
    print(f"Channels: {', '.join(brief.channel_recommendations)}")
except ValueError as e:
    print(f"✗ Failed to generate brief: {e}")