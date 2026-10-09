"""From Luna's integer weights to Jev answers.

Every question becomes a probability vector: `[p_false, p_true]` for noul, one entry per option
for choice, one per level for score. Samples are averaged, calibrated, then shaped into answers.
"""

import json
import math
from collections.abc import Sequence
from typing import Any

from .prompt import QuestionSpec
from .schemas import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer

DECIMALS = 6


class OutputParseError(ValueError):
    pass


def normalize(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def ensemble(distributions: Sequence[Sequence[float]]) -> list[float]:
    n = len(distributions)
    return [sum(column) / n for column in zip(*distributions, strict=True)]


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise OutputParseError(f"weight must be a number, got {value!r}")
    return min(100.0, max(0.0, float(value)))


def _topk_distribution(spec: QuestionSpec, top: Any) -> list[float]:
    if not isinstance(top, list) or not top:
        raise OutputParseError(f"{spec.alias}.top must be a non-empty list")
    index = {slot: i for i, slot in enumerate(spec.slots)}
    weights = [0.0] * len(spec.slots)
    listed: set[int] = set()
    for entry in top:
        if not isinstance(entry, dict):
            raise OutputParseError(f"{spec.alias}.top entries must be objects")
        i = index.get(entry.get("o"))
        if i is None:
            continue
        weights[i] += _weight(entry.get("w"))
        listed.add(i)
    if not listed:
        raise OutputParseError(f"{spec.alias}.top names no known option")
    # Weight the model didn't assign to its top list is shared evenly by the unlisted options.
    rest = [i for i in range(len(weights)) if i not in listed]
    remaining = max(0.0, 100.0 - sum(weights))
    if rest and remaining > 0:
        for i in rest:
            weights[i] = remaining / len(rest)
    return normalize(weights)


def distribution(spec: QuestionSpec, value: Any) -> list[float]:
    if spec.kind == "noul":
        p = _weight(value) / 100.0
        return [1.0 - p, p]
    if not isinstance(value, dict):
        raise OutputParseError(f"{spec.alias} must be an object")
    if spec.topk is not None:
        return _topk_distribution(spec, value.get("top"))
    return normalize([_weight(value[s]) if s in value else 0.0 for s in spec.slots])


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.removesuffix("```").strip()
    return text


def parse_output(content: str | None, specs: Sequence[QuestionSpec]) -> dict[str, list[float]]:
    """Parse one completion into `{qid: distribution}`; raises OutputParseError if unusable."""
    if not content:
        raise OutputParseError("empty completion")
    try:
        data = json.loads(_strip_fences(content))
    except json.JSONDecodeError as e:
        raise OutputParseError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise OutputParseError("completion is not a JSON object")
    out = {}
    for spec in specs:
        if spec.alias not in data:
            raise OutputParseError(f"missing {spec.alias}")
        out[spec.qid] = distribution(spec, data[spec.alias])
    return out


def _r(x: float) -> float:
    return round(x, DECIMALS)


def build_answer(spec: QuestionSpec, probs: Sequence[float]) -> Answer:
    if spec.kind == "noul":
        return NoulAnswer(noul=_r(probs[1]))
    best = max(range(len(probs)), key=probs.__getitem__)  # ties go to the earliest option
    if spec.kind == "choice":
        return ChoiceAnswer(
            choice=spec.labels[best],
            confidence=_r(probs[best]),
            probabilities={label: _r(p) for label, p in zip(spec.labels, probs, strict=True)},
        )
    return ScoreAnswer(
        score=_r(sum(i * p for i, p in enumerate(probs))),
        confidence=_r(probs[best]),
        legend={str(i): level for i, level in enumerate(spec.labels)},
        probabilities={str(i): _r(p) for i, p in enumerate(probs)},
    )
