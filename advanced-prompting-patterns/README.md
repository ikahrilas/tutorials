# Advanced Prompting Patterns

Patterns for getting **reliable structured output** out of an LLM, built around a
running example: turning a coffee product factsheet into a marketing campaign
brief.

Three ideas do the work:

1. **A schema is the contract.** A Pydantic model defines the expected output,
   and the same model both shapes the prompt and validates the response.
2. **Repair instead of retry.** When validation fails, the errors are fed back to
   the model in a follow-up prompt. The model sees its own bad output alongside
   the correction, which usually fixes it in one turn.
3. **Decompose when a single prompt strains.** Long factsheets lose detail in a
   one-shot prompt. Extracting facts first, then drafting from the validated
   extraction, keeps the output grounded.

## Setup

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
```

## Usage

```bash
uv run python campaign_brief.py
```

Or from your own code:

```python
from campaign_brief import generate_campaign_brief, generate_campaign_brief_pipeline

brief = generate_campaign_brief(factsheet)            # one-shot
brief = generate_campaign_brief_pipeline(factsheet)   # extract, then draft
print(brief.campaign_goal.value)
```

## Layout

| File | Responsibility |
| --- | --- |
| `models.py` | Pydantic schemas for the structured output. |
| `prompt_templates.py` | Named prompt blocks and the functions that compose them. |
| `utils.py` | Extract JSON from a response and validate it against a schema. |
| `conversation_manager.py` | API client wrapper: message history, token budget, personas. |
| `campaign_brief.py` | The generate-validate-repair loop and the two pipelines. |
| `tests/` | Offline tests; the API client is stubbed out. |
| `solution-files/` | Reference implementations from the tutorial. Not used at runtime. |

## Development

```bash
uv run pytest         # no API key needed; nothing hits the network
uv run ruff check .
uv run ruff format .
```

## Configuration

Defaults live at the top of `conversation_manager.py` and can be overridden per
instance:

```python
ConversationManager(model="gpt-5-nano-2025-08-07", max_tokens=2048, token_budget=8192)
```

`token_budget` caps the size of the conversation history. When it is exceeded,
the oldest non-system messages are dropped; the system message is always kept.
