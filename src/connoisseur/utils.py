"""Small helpers shared across modules."""
from __future__ import annotations

import json
import re

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def strip_code_fences(text: str) -> str:
    """Remove a wrapping ```json ... ``` fence if the model added one."""
    text = (text or "").strip()
    match = _FENCE.match(text)
    return match.group(1).strip() if match else text


def parse_json_response(text: str) -> dict:
    """Parse an LLM reply into a dict.

    Tolerates markdown fences (models often add them despite instructions).
    Raises json.JSONDecodeError (a ValueError) on bad JSON, ValueError if not an object.
    """
    parsed = json.loads(strip_code_fences(text))
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed
