import pytest
from pydantic import ValidationError

from decisions_mw.config import Settings

BASE = {"upstream_api_key": "k", "api_keys": "inbound"}


def make(**overrides) -> Settings:
    return Settings(_env_file=None, **{**BASE, **overrides})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"api_keys": ""}, "no inbound auth"),
        ({"upstream_api_key": None}, "no upstream key"),
        ({"forward_auth": True}, "mutually exclusive"),
        ({"samples": 9, "max_samples": 8}, "DMW_SAMPLES"),
    ],
    ids=["no-auth", "no-upstream-key", "forward-and-keys", "samples-over-max"],
)
def test_unsafe_or_inconsistent_config_fails_at_startup(overrides, message):
    with pytest.raises(ValidationError, match=message):
        make(**overrides)


def test_valid_auth_modes():
    make()  # own keys
    make(api_keys="", allow_anonymous=True)
    make(api_keys="", forward_auth=True, upstream_api_key=None)


def test_env_names_and_empty_values(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("DMW_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("DMW_API_KEYS", "a, b")
    monkeypatch.setenv("DMW_SEED", "")
    monkeypatch.setenv("DMW_CALIBRATION_PATH", "")
    s = Settings(_env_file=None)
    assert s.upstream_api_key.get_secret_value() == "sk-or"
    assert s.api_key_set == {"a", "b"}
    assert s.seed is None
    assert s.calibration_path is None


def test_generic_upstream_env_names_win(monkeypatch):
    monkeypatch.setenv("DMW_UPSTREAM_API_KEY", "sk-litellm")
    monkeypatch.setenv("DMW_UPSTREAM_BASE_URL", "http://litellm:4000/v1")
    monkeypatch.setenv("DMW_ALLOW_ANONYMOUS", "true")
    s = Settings(_env_file=None)
    assert s.upstream_api_key.get_secret_value() == "sk-litellm"
    assert s.flavor == "openai"


@pytest.mark.parametrize(
    ("url", "flavor", "expected"),
    [
        ("https://openrouter.ai/api/v1", "auto", "openrouter"),
        ("http://litellm:4000/v1", "auto", "openai"),
        ("https://my-proxy.example/v1", "openrouter", "openrouter"),
    ],
)
def test_flavor(url, flavor, expected):
    assert make(upstream_base_url=url, upstream_flavor=flavor).flavor == expected


def test_allowed_models_always_include_the_default():
    s = make(luna_model="luna-decisions", allowed_models="openai/gpt-5.6-luna")
    assert s.allowed_model_set == {"luna-decisions", "openai/gpt-5.6-luna"}
