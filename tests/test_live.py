"""Phase 0 spike as a test: the real upstream and Luna. Skipped unless an upstream key is set.

    OPENROUTER_API_KEY=... uv run pytest -m live -s
    DMW_UPSTREAM_BASE_URL=http://litellm:4000/v1 DMW_UPSTREAM_API_KEY=sk-... \
        DMW_LUNA_MODEL=luna-decisions uv run pytest -m live -s
"""

import os

import pytest
from fastapi.testclient import TestClient

from decisions_mw.config import Settings
from decisions_mw.main import create_app

from .conftest import JEV_EXAMPLE

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("DMW_UPSTREAM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")),
        reason="no upstream key (DMW_UPSTREAM_API_KEY / OPENROUTER_API_KEY)",
    ),
]


@pytest.mark.parametrize("samples", [1, 3])
def test_jev_example_against_real_luna(samples):
    settings = Settings(
        _env_file=None, api_keys="", forward_auth=False, allow_anonymous=True, calibration_path=None
    )
    with TestClient(create_app(settings)) as client:
        resp = client.post("/v1/systemone", json=JEV_EXAMPLE, headers={"X-Samples": str(samples)})
    assert resp.status_code == 200, resp.text
    answers = resp.json()["answers"]
    print(
        f"\nsamples={samples} model={resp.headers['X-Upstream-Model']} "
        f"provider={resp.headers.get('X-Upstream-Provider', '-')} "
        f"latency={resp.headers['X-Latency-Ms']}ms "
        f"cost=${resp.headers.get('X-Cost-USD', '?')}\n{answers}"
    )
    assert answers["refund"]["noul"] > 0.5
    assert answers["department"]["choice"] == "billing"
    assert 0.0 <= answers["urgency"]["score"] <= 2.0
