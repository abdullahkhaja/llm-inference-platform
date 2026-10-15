"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the Phase 1 mock inference server."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server
    app_name: str = "llm-inference-platform"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Backend selection — Phase 1 is mock-only
    backend: Literal["mock", "vllm"] = "mock"

    # Future real-model paths (unused in Phase 1)
    model_path: str = Field(
        default="",
        description="Base model id/path for Phase 2+ (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    adapter_path: str = Field(
        default="",
        description="Path to QLoRA adapter from training/finetune_spider.py (Phase 2+)",
    )

    # Input / output limits
    max_question_chars: int = 2_000
    max_schema_chars: int = 20_000
    max_new_tokens: int = 256
    default_max_new_tokens: int = 128

    # Mock timing model (serialized single-worker baseline)
    # Tuned to resemble a small GPU serving a short SQL completion — simulated only.
    mock_prefill_ms: float = 40.0
    mock_per_token_ms: float = 12.0
    mock_jitter_ms: float = 2.0
    mock_seed: int = 42

    # Logging
    request_log_path: Path = Path("logs/requests.jsonl")
    log_prompts: bool = False

    @field_validator("backend", mode="before")
    @classmethod
    def normalize_backend(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
