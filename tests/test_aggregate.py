import json
import math

import pytest

from decisions_mw.aggregate import (
    OutputParseError,
    build_answer,
    ensemble,
    normalize,
    parse_output,
)
from decisions_mw.prompt import build_specs
from decisions_mw.schemas import DecisionRequest

from .conftest import JEV_EXAMPLE


@pytest.fixture
def specs():
    req = DecisionRequest.model_validate(JEV_EXAMPLE)
    return build_specs(req.questions, topk_threshold=20, topk=5)


def topk_spec(n: int = 30, k: int = 5):
    body = {
        "model": "m",
        "state": "s",
        "questions": {
            "c": {
                "type": "choice",
                "instructions": "x",
                "criteria": {f"label{i}": None for i in range(n)},
            }
        },
    }
    req = DecisionRequest.model_validate(body)
    return build_specs(req.questions, topk_threshold=20, topk=k)[0]


def test_normalize():
    assert normalize([1, 3]) == [0.25, 0.75]
    assert normalize([0, 0, 0, 0]) == [0.25] * 4


def test_ensemble_is_elementwise_mean():
    assert ensemble([[1.0, 0.0], [0.5, 0.5]]) == [0.75, 0.25]


def test_parse_jev_like_output(specs):
    out = parse_output(
        json.dumps({"q1": 99, "q2": {"o1": 100, "o2": 0}, "q3": {"l0": 0, "l1": 0, "l2": 100}}),
        specs,
    )
    assert out["refund"] == pytest.approx([0.01, 0.99])
    assert out["department"] == [1.0, 0.0]
    assert out["urgency"] == [0.0, 0.0, 1.0]


def test_weights_are_clamped_and_renormalized(specs):
    out = parse_output(
        json.dumps({"q1": 150, "q2": {"o1": 30, "o2": -5}, "q3": {"l0": 1, "l1": 1, "l2": 2}}),
        specs,
    )
    assert out["refund"] == [0.0, 1.0]
    assert out["department"] == [1.0, 0.0]
    assert out["urgency"] == [0.25, 0.25, 0.5]


def test_markdown_fences_are_tolerated(specs):
    payload = '{"q1": 50, "q2": {"o1": 1, "o2": 1}, "q3": {"l0": 1, "l1": 1, "l2": 1}}'
    content = f"```json\n{payload}\n```"
    assert parse_output(content, specs)["refund"] == [0.5, 0.5]


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "not json",
        "[1, 2]",
        '{"q1": 50, "q2": {"o1": 1, "o2": 1}}',
        '{"q1": "high", "q2": {"o1": 1, "o2": 1}, "q3": {"l0": 1, "l1": 1, "l2": 1}}',
        '{"q1": true, "q2": {"o1": 1, "o2": 1}, "q3": {"l0": 1, "l1": 1, "l2": 1}}',
        '{"q1": 50, "q2": [1, 2], "q3": {"l0": 1, "l1": 1, "l2": 1}}',
    ],
    ids=[
        "none",
        "empty",
        "garbage",
        "array",
        "missing-q3",
        "string-weight",
        "bool-weight",
        "choice-not-object",
    ],
)
def test_unusable_output_raises(specs, content):
    with pytest.raises(OutputParseError):
        parse_output(content, specs)


def test_topk_spreads_leftover_weight_over_unlisted_options():
    spec = topk_spec()
    out = parse_output(
        json.dumps({"q1": {"top": [{"o": "o3", "w": 60}, {"o": "o7", "w": 10}]}}), [spec]
    )
    probs = out["c"]
    assert math.isclose(sum(probs), 1.0)
    assert probs[2] == pytest.approx(0.6)
    assert probs[6] == pytest.approx(0.1)
    assert probs[0] == pytest.approx(0.3 / 28)


def test_topk_with_full_mass_leaves_rest_at_zero():
    spec = topk_spec()
    probs = parse_output(json.dumps({"q1": {"top": [{"o": "o1", "w": 100}]}}), [spec])["c"]
    assert probs[0] == 1.0 and sum(probs[1:]) == 0.0


def test_topk_ignores_unknown_options_but_needs_one_known():
    spec = topk_spec()
    out = parse_output(
        json.dumps({"q1": {"top": [{"o": "o99", "w": 50}, {"o": "o2", "w": 50}]}}), [spec]
    )
    assert out["c"][1] == pytest.approx(0.5)
    with pytest.raises(OutputParseError):
        parse_output(json.dumps({"q1": {"top": [{"o": "o99", "w": 50}]}}), [spec])


def test_answers_match_jev_shapes(specs):
    refund, department, urgency = specs
    assert build_answer(refund, [0.01, 0.99]).model_dump() == {"type": "noul", "noul": 0.99}
    choice = build_answer(department, [1.0, 0.0]).model_dump()
    assert choice == {
        "type": "choice",
        "choice": "billing",
        "confidence": 1.0,
        "probabilities": {"billing": 1.0, "technical": 0.0},
    }
    score = build_answer(urgency, [0.0, 0.0, 1.0]).model_dump()
    assert score == {
        "type": "score",
        "score": 2.0,
        "confidence": 1.0,
        "legend": {
            "0": "Low urgency",
            "1": "Needs attention today",
            "2": "Requires immediate action",
        },
        "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
    }


def test_score_is_probability_weighted_and_choice_ties_go_first(specs):
    _, department, urgency = specs
    assert build_answer(urgency, [0.2, 0.5, 0.3]).score == pytest.approx(1.1)
    assert build_answer(department, [0.5, 0.5]).choice == "billing"
