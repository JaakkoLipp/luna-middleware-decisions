from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv(value: str) -> frozenset[str]:
    return frozenset(v.strip() for v in value.split(",") if v.strip())


class Settings(BaseSettings):
    """Runtime configuration from `DMW_`-prefixed env vars (and `OPENROUTER_API_KEY`).

    The upstream is OpenRouter or any OpenAI-compatible endpoint, e.g. a LiteLLM proxy in front
    of an Azure deployment. Invalid or unsafe combinations fail at startup, not on first request.
    """

    model_config = SettingsConfigDict(
        env_prefix="DMW_", env_file=".env", extra="ignore", populate_by_name=True
    )

    # Upstream
    upstream_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("DMW_UPSTREAM_BASE_URL", "DMW_OPENROUTER_BASE_URL"),
    )
    upstream_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "DMW_UPSTREAM_API_KEY", "OPENROUTER_API_KEY", "DMW_OPENROUTER_API_KEY"
        ),
    )
    # auto: "openrouter" for openrouter.ai, otherwise "openai" (OpenAI, LiteLLM, Azure, ...)
    upstream_flavor: Literal["auto", "openrouter", "openai"] = "auto"
    luna_model: str = "openai/gpt-6-luna"
    allowed_models: str = ""  # comma-separated extra models a request may name explicitly
    # Luna accepts none | low | medium | high | xhigh (not minimal); empty omits the parameter.
    reasoning_effort: str = "none"
    app_title: str = "decisions-middleware"
    app_referer: str | None = None

    # Decision behaviour
    samples: int = Field(default=1, ge=1)
    max_samples: int = Field(default=8, ge=1)
    seed: int | None = 0
    topk_threshold: int = Field(default=20, ge=1)
    topk: int = Field(default=5, ge=1)
    max_questions_per_call: int = Field(default=8, ge=1)
    calibration_path: Path | None = None

    # Request limits
    max_questions: int = Field(default=64, ge=1)
    max_state_chars: int = Field(default=200_000, ge=1)
    request_timeout_s: float = Field(default=30.0, gt=0)

    # Transport
    timeout_s: float = Field(default=20.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    retry_backoff_s: float = Field(default=0.25, ge=0)
    max_concurrency: int = Field(default=16, ge=1)

    # Inbound
    api_keys: str = ""  # comma-separated bearer keys checked by the middleware itself
    forward_auth: bool = False  # forward the caller's bearer key upstream (LiteLLM virtual keys)
    allow_anonymous: bool = False  # run with no auth at all (local development only)
    rate_limit_per_minute: int = Field(default=0, ge=0)  # per key; 0 disables

    log_level: str = "INFO"

    @field_validator("upstream_api_key", "seed", "calibration_path", "app_referer", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: Any) -> Any:
        return None if value == "" else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.samples > self.max_samples:
            raise ValueError("DMW_SAMPLES must not exceed DMW_MAX_SAMPLES")
        if self.forward_auth and self.api_key_set:
            raise ValueError(
                "DMW_FORWARD_AUTH and DMW_API_KEYS are mutually exclusive: when forwarding, "
                "the upstream validates the caller's key"
            )
        if not self.forward_auth and self.upstream_api_key is None:
            raise ValueError(
                "no upstream key: set DMW_UPSTREAM_API_KEY (or OPENROUTER_API_KEY), "
                "or DMW_FORWARD_AUTH=true to use each caller's key"
            )
        if not (self.api_key_set or self.forward_auth or self.allow_anonymous):
            raise ValueError(
                "no inbound auth: set DMW_API_KEYS or DMW_FORWARD_AUTH=true "
                "(DMW_ALLOW_ANONYMOUS=true disables auth, for local development only)"
            )
        return self

    @property
    def api_key_set(self) -> frozenset[str]:
        return _csv(self.api_keys)

    @property
    def allowed_model_set(self) -> frozenset[str]:
        return _csv(self.allowed_models) | {self.luna_model}

    @property
    def flavor(self) -> Literal["openrouter", "openai"]:
        if self.upstream_flavor != "auto":
            return self.upstream_flavor
        return "openrouter" if "openrouter.ai" in self.upstream_base_url else "openai"


@lru_cache
def get_settings() -> Settings:
    return Settings()
