"""Metrics and calibration fitting used by `eval/run_eval.py` and `eval/fit_calibration.py`.

A prediction is a probability vector plus the index of the correct answer; noul vectors are
`[p_false, p_true]`, so label 1 means "true".
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .calibration import Calibration, Platt, logit, sigmoid

NLL_FLOOR = 1e-6


@dataclass(frozen=True)
class Prediction:
    kind: str  # noul | choice | score
    probs: list[float]
    label: int


def _argmax(probs: Sequence[float]) -> int:
    return max(range(len(probs)), key=probs.__getitem__)


def accuracy(preds: Sequence[Prediction]) -> float:
    return sum(_argmax(p.probs) == p.label for p in preds) / len(preds)


def brier(preds: Sequence[Prediction]) -> float:
    """Binary Brier on p_true for noul, multi-class Brier otherwise."""
    total = 0.0
    for p in preds:
        if p.kind == "noul":
            total += (p.probs[1] - p.label) ** 2
        else:
            total += sum((q - (i == p.label)) ** 2 for i, q in enumerate(p.probs))
    return total / len(preds)


def nll(preds: Sequence[Prediction]) -> float:
    return -sum(math.log(max(p.probs[p.label], NLL_FLOOR)) for p in preds) / len(preds)


def ece(preds: Sequence[Prediction], bins: int = 10) -> float:
    """Expected calibration error of the top answer's confidence, equal-width bins."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p in preds:
        best = _argmax(p.probs)
        conf = p.probs[best]
        buckets[min(bins - 1, int(conf * bins))].append((conf, best == p.label))
    err = 0.0
    for b in buckets:
        if b:
            avg_conf = sum(c for c, _ in b) / len(b)
            acc = sum(ok for _, ok in b) / len(b)
            err += abs(acc - avg_conf) * len(b) / len(preds)
    return err


def score_mae(preds: Sequence[Prediction]) -> float:
    return sum(abs(sum(i * q for i, q in enumerate(p.probs)) - p.label) for p in preds) / len(preds)


def summarize(preds: Sequence[Prediction]) -> dict[str, float]:
    if not preds:
        return {"n": 0}
    out = {
        "n": len(preds),
        "accuracy": accuracy(preds),
        "brier": brier(preds),
        "nll": nll(preds),
        "ece": ece(preds),
    }
    if preds[0].kind == "score":
        out["mae"] = score_mae(preds)
    return out


def _softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def fit_platt(
    preds: Sequence[Prediction], l2: float = 1.0, iters: int = 100
) -> tuple[float, float]:
    """Damped Newton for sigmoid(a*logit(p) + b), with an L2 pull toward the identity (1, 0)."""
    xs = [logit(p.probs[1]) for p in preds]
    ys = [float(p.label) for p in preds]

    def objective(a: float, b: float) -> float:
        reg = l2 * ((a - 1) ** 2 + b**2)
        return reg + sum(
            _softplus(a * x + b) - y * (a * x + b) for x, y in zip(xs, ys, strict=True)
        )

    a, b = 1.0, 0.0
    for _ in range(iters):
        ga, gb = 2 * l2 * (a - 1), 2 * l2 * b
        haa = hbb = 2 * l2
        hab = 0.0
        for x, y in zip(xs, ys, strict=True):
            s = sigmoid(a * x + b)
            w = s * (1 - s)
            ga += (s - y) * x
            gb += s - y
            haa += w * x * x
            hab += w * x
            hbb += w
        det = haa * hbb - hab * hab
        if det <= 0:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        current, step = objective(a, b), 1.0
        while step > 1e-8 and objective(a - step * da, b - step * db) > current:
            step /= 2
        a, b = a - step * da, b - step * db
        if abs(step * da) < 1e-10 and abs(step * db) < 1e-10:
            break
    if a < 0:
        # Confidence is anti-informative on this data. Never invert answers: use the base rate.
        return 0.0, logit((sum(ys) + 0.5) / (len(ys) + 1))
    return a, b


def top_label(p: Prediction) -> Prediction:
    """The binary event "the top answer is right", as a noul-shaped prediction."""
    best = _argmax(p.probs)
    return Prediction("noul", [1 - p.probs[best], p.probs[best]], int(best == p.label))


def fit_calibration(preds: Sequence[Prediction], min_n: int = 10) -> tuple[Calibration, list[str]]:
    """Fit every question type that has at least `min_n` examples; others stay at identity.

    noul is fitted on p(true); choice and score on their top answer's confidence (top-label).
    """
    notes = []
    params: dict[str, Platt] = {}
    for kind in ("noul", "choice", "score"):
        items = [p for p in preds if p.kind == kind]
        if len(items) < min_n:
            notes.append(f"{kind}: {len(items)} examples (< {min_n}), left at identity")
            continue
        binary = items if kind == "noul" else [top_label(p) for p in items]
        params[kind] = Platt(*fit_platt(binary))
    return Calibration(**params), notes
