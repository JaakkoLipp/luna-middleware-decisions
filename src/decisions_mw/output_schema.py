"""Strict JSON schema for Luna's structured output, built per request.

Output shape (aliases from `prompt.py`):
    {"q1": 87, "q2": {"o1": 95, "o2": 5}, "q3": {"l0": 10, "l1": 70, "l2": 20},
     "q4": {"top": [{"o": "o17", "w": 80}, {"o": "o3", "w": 15}]}}
"""

from typing import Any

from .prompt import QuestionSpec

WEIGHT = {"type": "integer", "minimum": 0, "maximum": 100}


def _obj(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def question_schema(spec: QuestionSpec) -> dict[str, Any]:
    if spec.kind == "noul":
        return dict(WEIGHT)
    if spec.topk is not None:
        entry = _obj({"o": {"type": "string", "enum": spec.slots}, "w": dict(WEIGHT)})
        return _obj(
            {"top": {"type": "array", "items": entry, "minItems": 1, "maxItems": spec.topk}}
        )
    return _obj({slot: dict(WEIGHT) for slot in spec.slots})


def build_schema(specs: list[QuestionSpec]) -> dict[str, Any]:
    return _obj({s.alias: question_schema(s) for s in specs})


def build_response_format(specs: list[QuestionSpec]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": "decisions", "strict": True, "schema": build_schema(specs)},
    }


def estimate_output_tokens(specs: list[QuestionSpec]) -> int:
    """Rough upper bound on output tokens for compact JSON, used to size max_tokens."""
    n = 4
    for s in specs:
        n += 4
        if s.kind == "noul":
            n += 3
        elif s.topk is not None:
            n += 6 + 12 * s.topk
        else:
            n += 4 + 7 * len(s.slots)
    return n
