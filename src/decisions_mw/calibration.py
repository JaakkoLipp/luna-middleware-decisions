"""Post-hoc calibration of Luna's stated probabilities, keyed by model slug.

calibration.json:
    {"openai/gpt-6-luna": {"noul": {"a": 0.8, "b": 0.0},
                           "choice": {"a": 0.6, "b": -0.1},
                           "score": {"a": 0.7, "b": 0.0}}}

Every type uses Platt scaling, sigmoid(a * logit(p) + b). For noul, p is the probability of
true. For choice and score it's the top answer's probability ("top-label" calibration): the top
answer gets the calibrated value and the other options are rescaled to share the rest, so the
result depends only on the top probability, never on how many options there are. Missing models
or fields fall back to the identity. Fit with `eval/fit_calibration.py`.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EPS = 1e-4
_KINDS = ("noul", "choice", "score")


def logit(p: float) -> float:
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    z = math.exp(x)
    return z / (1 + z)


@dataclass(frozen=True)
class Platt:
    a: float = 1.0
    b: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.a == 1.0 and self.b == 0.0

    def __call__(self, p: float) -> float:
        return sigmoid(self.a * logit(p) + self.b)


def calibrate_top(probs: Sequence[float], f: Platt) -> list[float]:
    """Calibrate the top answer's probability and rescale the others to fill the remainder."""
    if len(probs) < 2:
        return list(probs)
    top = max(range(len(probs)), key=probs.__getitem__)
    p = probs[top]
    rest = 1.0 - p
    new_top = f(p)
    if rest > 0:
        # The rescaled runner-up must not overtake the top answer.
        runner_up = max(q for i, q in enumerate(probs) if i != top)
        new_top = max(new_top, runner_up / (rest + runner_up) + 1e-9)
    new_top = min(1.0, new_top)
    if rest > 0:
        out = [q * (1.0 - new_top) / rest for q in probs]
    else:  # the model put everything on one answer: spread the remainder evenly
        out = [(1.0 - new_top) / (len(probs) - 1)] * len(probs)
    out[top] = new_top
    return out


@dataclass(frozen=True)
class Calibration:
    noul: Platt = Platt()
    choice: Platt = Platt()
    score: Platt = Platt()

    def apply(self, kind: str, probs: Sequence[float]) -> list[float]:
        f: Platt = getattr(self, kind)
        if f.is_identity:
            return list(probs)
        if kind == "noul":
            p = f(probs[1])
            return [1.0 - p, p]
        return calibrate_top(probs, f)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Calibration":
        params = {}
        for kind in _KINDS:
            entry = d.get(kind, {})
            params[kind] = Platt(a=float(entry.get("a", 1.0)), b=float(entry.get("b", 0.0)))
        return cls(**params)

    def to_dict(self) -> dict[str, Any]:
        return {kind: {"a": getattr(self, kind).a, "b": getattr(self, kind).b} for kind in _KINDS}


IDENTITY = Calibration()


class CalibrationStore:
    def __init__(self, table: dict[str, Calibration] | None = None):
        self.table = table or {}

    @classmethod
    def load(cls, path: Path | None) -> "CalibrationStore":
        if path is None:
            return cls()
        raw = json.loads(Path(path).read_text())
        return cls({model: Calibration.from_dict(v) for model, v in raw.items()})

    def for_model(self, model: str) -> Calibration:
        return self.table.get(model, IDENTITY)
