"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GenerateRequest(BaseModel):
    """Natural-language question + schema → SQL generation request."""

    model_config = ConfigDict(populate_by_name=True)

    question: str = Field(..., min_length=1, description="Natural language question")
    # JSON field remains "schema" (per API contract); Python attr avoids BaseModel shadowing.
    db_schema: str = Field(
        ...,
        alias="schema",
        min_length=1,
        description="Database schema DDL / description",
    )
    max_new_tokens: int = Field(
        default=128,
        ge=1,
        description="Maximum output tokens to generate",
    )

    @field_validator("question", "db_schema")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be empty or whitespace-only")
        return cleaned


class GenerateResponse(BaseModel):
    """Generation response. ``simulated`` reflects the active backend."""

    request_id: str
    sql: str
    backend: str
    simulated: bool = True
    input_tokens: int
    output_tokens: int
    queue_wait_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    latency_ms: float
    ttft_ms: Optional[float] = None
    # Backward-compatible name from Phase 1; equals output_tokens_per_sec.
    simulated_output_tokens_per_sec: Optional[float] = None
    output_tokens_per_sec: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    backend: str
    ready: bool
    simulated: bool = True
    queue_depth: int = 0
    inflight: int = 0
    admission_enabled: bool = False
    admission_active: int = 0
    admission_waiting: int = 0
    version: str
    vllm_base_url: Optional[str] = None
    vllm_model_name: Optional[str] = None


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    error: str
    detail: Optional[str] = None
    category: str = "client_error"
