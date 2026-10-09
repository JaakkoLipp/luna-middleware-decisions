import math
import random

import pytest

from decisions_mw.evaluation import (
    Prediction,
    accuracy,
    brier,
    ece,
    fit_calibration,
    fit_platt,
    nll,
    score_mae,
    summarize,
)


def test_metrics_on_a_hand_computed_example():
    preds = [
        Prediction("choice", [0.8, 0.2], 0),
        Prediction("choice", [0.6, 0.4], 1),
    ]
    assert accuracy(preds) == 0.5
    assert brier(preds) == pytest.approx(((0.2**2 + 0.2**2) + (0.6**2 + 0.6**2)) / 2)
    assert nll(preds) == pytest.approx(-(math.log(0.8) + math.log(0.4)) / 2)
    # Bins: 0.8 (right) and 0.6 (wrong) -> |1-0.8|/2 + |0-0.6|/2
    assert ece(preds) == pytest.approx(0.4)


def test_noul_brier_is_binary():
    assert brier([Prediction("noul", [0.1, 0.9], 1)]) == pytest.approx(0.01)


def test_score_mae_uses_expected_level():
    assert score_mae([Prediction("score", [0.0, 0.5, 0.5], 0)]) == pytest.approx(1.5)
    assert "mae" in summarize([Prediction("score", [0.0, 1.0], 1)])


def overconfident(kind: str, n: int = 400, seed: int = 1) -> list[Prediction]:
    """The top answer always gets 0.95 but is right only ~70% of the time."""
    rng = random.Random(seed)
    preds = []
    for _ in range(n):
        right = rng.random() < 0.7
        if kind == "noul":
            preds.append(Prediction("noul", [0.05, 0.95], 1 if right else 0))
        else:
            preds.append(Prediction(kind, [0.95, 0.03, 0.02], 0 if right else 1))
    return preds


@pytest.mark.parametrize("kind", ["choice", "score"])
def test_top_label_fit_softens_overconfidence(kind):
    preds = overconfident(kind)
    cal, _ = fit_calibration(preds)
    calibrated = [Prediction(kind, cal.apply(kind, p.probs), p.label) for p in preds]
    assert calibrated[0].probs[0] == pytest.approx(0.7, abs=0.03)
    assert ece(calibrated) < ece(preds)


def test_platt_fit_softens_overconfidence():
    a, b = fit_platt(overconfident("noul"))
    assert 0 < a < 1


def test_platt_never_inverts_answers():
    # Confident answers are always wrong: the unconstrained fit would flip them.
    preds = [Prediction("noul", [0.05, 0.95], 0) for _ in range(30)]
    preds += [Prediction("noul", [0.95, 0.05], 1) for _ in range(30)]
    a, b = fit_platt(preds)
    assert a == 0.0
    assert b == pytest.approx(0.0)


def test_well_calibrated_data_stays_near_identity():
    rng = random.Random(3)
    preds = []
    for _ in range(2000):
        label = 0 if rng.random() < 0.7 else rng.choice([1, 2])
        preds.append(Prediction("choice", [0.7, 0.15, 0.15], label))
    cal, _ = fit_calibration(preds)
    assert cal.apply("choice", [0.7, 0.15, 0.15])[0] == pytest.approx(0.7, abs=0.03)


def test_small_samples_stay_identity():
    cal, notes = fit_calibration(overconfident("score", n=5))
    assert cal.score.is_identity
    assert any("score" in n for n in notes)
