"""Helpers for turning a raw model response into a validated object."""

import json
import re

from pydantic import BaseModel, ValidationError

_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n|\n```$")


def extract_json(text: str) -> dict:
    """Extract the first JSON object from a model response.

    Handles the two things models routinely do even when told not to: wrapping
    the JSON in a markdown code fence, and adding a sentence of preamble.

    Raises:
        json.JSONDecodeError: if no JSON object can be parsed from the text.
    """
    text = _FENCE_RE.sub("", text.strip()).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise json.JSONDecodeError("Expected a JSON object", text, start)
    return data


def validate_json_output[M: BaseModel](
    response_text: str, model_class: type[M]
) -> tuple[bool, M | str]:
    """Parse JSON from a model response and validate it against a schema.

    Returns:
        ``(True, instance)`` on success, or ``(False, error_message)`` on
        failure. The error message is written to be fed straight back to the
        model in a repair prompt.
    """
    try:
        data = extract_json(response_text)
        return True, model_class.model_validate(data)
    except json.JSONDecodeError as error:
        return False, f"JSON parsing error: {error}"
    except ValidationError as error:
        return False, f"Validation errors: {error.errors()}"
