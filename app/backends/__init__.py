"""Inference backend implementations."""

from app.backends.base import Backend, GenerationRequest, GenerationResult
from app.backends.mock import MockBackend

__all__ = ["Backend", "GenerationRequest", "GenerationResult", "MockBackend", "build_backend"]


def build_backend(settings: "object") -> Backend:
    """Factory for backends. Phase 1 supports mock only."""
    from app.config import Settings

    assert isinstance(settings, Settings)
    if settings.backend == "mock":
        return MockBackend(settings)
    if settings.backend == "vllm":
        raise NotImplementedError(
            "VLLMBackend is planned for Phase 2. Use BACKEND=mock for Phase 1."
        )
    raise ValueError(f"Unknown BACKEND={settings.backend!r}. Expected 'mock' (Phase 1).")
