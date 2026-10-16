"""Backend abstraction shared by mock and remote vLLM inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional


@dataclass(frozen=True)
class GenerationRequest:
    """Normalized generation request handed to a backend."""

    request_id: str
    question: str
    schema: str
    max_new_tokens: int
    prompt_text: str
    messages: list[dict[str, str]] = field(default_factory=list)


@dataclass
class GenerationResult:
    """Backend output with timing breakdown (monotonic seconds).

    Timing fields that the backend cannot observe are left as None rather than
    fabricated zeros. Mock backends fill all fields; remote vLLM leaves queue /
    prefill / decode unknown unless the server exposes them.
    """

    text: str
    input_tokens: int
    output_tokens: int
    queue_wait_s: Optional[float]
    inference_s: Optional[float]
    prefill_s: Optional[float]
    decode_s: Optional[float]
    simulated: bool = True
    backend: str = "mock"
    ttft_s: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """One SSE-oriented streaming event from a backend."""

    event: str  # token | usage | error | done
    data: dict[str, Any] = field(default_factory=dict)


class Backend(ABC):
    """Inference backend interface.

    MockBackend serializes work behind a lock to emulate a single worker.
    VLLMBackend relies on the remote vLLM server for scheduling / batching and
    issues concurrent HTTP requests without a local batcher.
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

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[StreamEvent]:
        """Stream tokens. Default: not supported."""
        raise NotImplementedError(f"{self.name} does not support streaming")
        yield  # pragma: no cover — makes this an async generator for type checkers

    @property
    @abstractmethod
    def queue_depth(self) -> int:
        """Number of requests waiting / in-flight at the backend layer."""

    @property
    def ready(self) -> bool:
        return True
