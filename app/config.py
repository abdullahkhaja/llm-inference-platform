"""Application configuration via environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for mock and remote-vLLM inference serving."""

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

    # Backend selection
    backend: Literal["mock", "vllm"] = "mock"

    # Base model + adapter (used by RunPod/vLLM startup; FastAPI uses VLLM_* for remote)
    model_path: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Base model id for Phase 2+ GPU serving",
    )
    adapter_path: str = Field(
        default="",
        description="HF repo id or local path containing the QLoRA adapter",
    )
    adapter_subfolder: str = Field(
        default="sft/final",
        description="Subfolder inside ADAPTER_PATH with adapter_config.json",
    )
    hf_token: Optional[str] = Field(
        default=None,
        description="Hugging Face token for private adapters / gated base model",
    )

    # Remote vLLM OpenAI-compatible HTTP API (FastAPI → RunPod)
    vllm_base_url: str = Field(
        default="http://127.0.0.1:8001/v1",
        description="OpenAI-compatible base URL, e.g. https://<pod>-8001.proxy.runpod.net/v1",
    )
    vllm_model_name: str = Field(
        default="spider-sft",
        description="Model id exposed by vLLM (LoRA module name or base model)",
    )
    vllm_api_key: str = Field(
        default="EMPTY",
        description="Bearer token for vLLM (often unused; RunPod may not require it)",
    )
    vllm_timeout_s: float = Field(default=120.0, ge=1.0)
    vllm_connect_timeout_s: float = Field(default=10.0, ge=1.0)
    vllm_max_retries: int = Field(default=2, ge=0)
    vllm_temperature: float = Field(default=0.0, ge=0.0)
    vllm_ready_timeout_s: float = Field(
        default=30.0,
        ge=1.0,
        description="Max seconds to wait for /models readiness at startup",
    )

    # vLLM process on the GPU host (not used to load weights in this API).
    # V1 enables prefix caching unless --no-enable-prefix-caching is passed.
    enable_prefix_caching: bool = True

    # API admission. Off preserves Phase 2 forwarding with no extra queue.
    admission_enabled: bool = False
    max_concurrent_requests: int = Field(default=4, ge=1)
    max_waiting_requests: int = Field(default=8, ge=0)
    max_queue_wait_ms: int = Field(default=2000, ge=0)

    # Optional tracing. Off by default; spans add overhead when enabled.
    tracing_enabled: bool = False
    otlp_endpoint: Optional[str] = None
    trace_log_path: Optional[Path] = None

    # Input / output limits
    max_question_chars: int = 2_000
    max_schema_chars: int = 20_000
    max_new_tokens: int = 256
    default_max_new_tokens: int = 128

    # Mock timing model (serialized single-worker baseline)
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

    @field_validator("hf_token", mode="before")
    @classmethod
    def empty_token_to_none(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def measurements_mode(self) -> str:
        return "simulated" if self.backend == "mock" else "real"


@lru_cache
def get_settings() -> Settings:
    return Settings()
