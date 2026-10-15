"""Backend abstraction shared by mock and future real inference engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized generation request handed to a backend."""

    request_id: str
    question: str
    schema: str
    max_new_tokens: int
    prompt_text: str


@dataclass
class GenerationResult:
    """Backend output with timing breakdown (monotonic seconds)."""

    text: str
    input_tokens: int
    output_tokens: int
    queue_wait_s: float
    inference_s: float
    prefill_s: float
    decode_s: float
    simulated: bool = True
    backend: str = "mock"
    metadata: dict = field(default_factory=dict)


class Backend(ABC):
    """Inference backend interface.

    Implementations must serialize GPU-like work. Concurrent callers may await
    the same lock/worker; they must not independently sleep in a way that
    implies unlimited parallel inference capacity.
    """

    name: str

    @abstractmethod
    async def startup(self) -> None:
        """Load resources once at process start."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Run (or simulate) inference for one request."""

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Number of requests waiting for the inference worker."""

    @property
    def ready(self) -> bool:
        return True
