from __future__ import annotations

import json
import re
from typing import Any


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class JsonExtractionError(ValueError):
    """Raised when a model response does not contain valid JSON."""


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()

    parsed = _try_parse_object(candidate)
    if parsed is not None:
        return parsed

    for block_match in JSON_BLOCK_PATTERN.finditer(candidate):
        parsed = _try_parse_object(block_match.group(1).strip())
        if parsed is not None:
            return parsed

    for sliced in _iter_json_object_candidates(candidate):
        parsed = _try_parse_object(sliced)
        if parsed is not None:
            return parsed

    if "{" not in candidate or "}" not in candidate:
        raise JsonExtractionError("No JSON object found in model response.")

    raise JsonExtractionError("Could not parse a valid JSON object from model response.")


def _try_parse_object(candidate: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _iter_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start : index + 1])
                start = None

    return candidates
