"""Inference backend implementations."""

from app.backends.base import Backend, GenerationRequest, GenerationResult, StreamEvent
from app.backends.mock import MockBackend
from app.backends.vllm import VLLMBackend

__all__ = [
    "Backend",
    "GenerationRequest",
    "GenerationResult",
    "StreamEvent",
    "MockBackend",
    "VLLMBackend",
    "build_backend",
]


def build_backend(settings: "object") -> Backend:
    """Factory for backends: mock (CPU) or vllm (remote HTTP)."""
    from app.config import Settings

    assert isinstance(settings, Settings)
    if settings.backend == "mock":
        return MockBackend(settings)
    if settings.backend == "vllm":
        return VLLMBackend(settings)
    raise ValueError(f"Unknown BACKEND={settings.backend!r}. Expected 'mock' or 'vllm'.")
