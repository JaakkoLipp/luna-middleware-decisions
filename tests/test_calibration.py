import json
import math

import pytest

from decisions_mw.calibration import Calibration, CalibrationStore, Platt, calibrate_top

SOFTEN = Platt(a=0.5)  # pulls confident answers toward 50%


def top(probs: list[float]) -> float:
    return max(probs)


def test_identity_calibration_is_exact():
    cal = Calibration()
    assert cal.apply("choice", [0.7, 0.3]) == [0.7, 0.3]
    assert cal.apply("noul", [0.2, 0.8]) == [0.2, 0.8]


def test_platt_noul():
    p = Calibration(noul=SOFTEN).apply("noul", [0.01, 0.99])[1]
    assert 0.5 < p < 0.99


@pytest.mark.parametrize("n", [2, 4, 30, 255])
def test_top_label_calibration_does_not_depend_on_option_count(n):
    """The old temperature scaling turned 0.90 into 0.75 with 2 options but 0.25 with 255."""
    probs = [0.9, 0.1] + [0.0] * (n - 2)
    out = calibrate_top(probs, SOFTEN)
    assert out[0] == pytest.approx(SOFTEN(0.9))
    assert math.isclose(sum(out), 1.0)
    assert out[1] == pytest.approx(1 - SOFTEN(0.9))  # the remainder keeps its proportions
    assert sum(out[2:]) == 0.0


def test_certain_answer_spreads_remainder_evenly():
    out = calibrate_top([0.0, 1.0, 0.0], SOFTEN)
    assert out[1] == pytest.approx(SOFTEN(1.0))
    assert out[0] == out[2] == pytest.approx((1 - SOFTEN(1.0)) / 2)


def test_runner_up_never_overtakes_the_top_answer():
    out = calibrate_top([0.6, 0.4], Platt(a=0.0, b=-2.0))  # would push the top down to 0.12
    assert out[0] > out[1]
    assert math.isclose(sum(out), 1.0)


def test_sharpening_is_capped_at_one():
    out = calibrate_top([0.2, 0.8], Platt(a=5.0, b=5.0))
    assert out[1] <= 1.0 and math.isclose(sum(out), 1.0)


def test_calibration_store_roundtrip(tmp_path):
    cal = Calibration(noul=Platt(0.8, 0.1), choice=Platt(0.6, -0.1), score=Platt(0.7, 0.0))
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"openai/gpt-6-luna": cal.to_dict()}))
    store = CalibrationStore.load(path)
    assert store.for_model("openai/gpt-6-luna") == cal
    assert store.for_model("other") == Calibration()
