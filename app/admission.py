"""API-level admission control for the remote inference backend.

This is backpressure in front of vLLM, not a replacement for vLLM's scheduler.
When disabled, acquire() is a no-op and does not serialize requests.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from app.metrics import (
    ADMISSION_ACTIVE,
    ADMISSION_REJECTED,
    ADMISSION_TIMEOUTS,
    ADMISSION_WAIT,
    ADMISSION_WAITING,
)


class AdmissionRejected(Exception):
    """Raised when a request is refused because the API is at capacity."""

    def __init__(self, reason: str, retry_after_s: int = 1) -> None:
        self.reason = reason
        self.retry_after_s = retry_after_s
        super().__init__(reason)


@dataclass
class AdmissionLease:
    """Holds one active slot until release(). Safe to release more than once."""

    controller: "AdmissionController"
    held: bool = False

    async def release(self) -> None:
        if not self.held:
            return
        self.held = False
        await self.controller.release_active()


class AdmissionController:
    def __init__(
        self,
        *,
        enabled: bool,
        max_active: int,
        max_waiting: int,
        max_wait_s: float,
        retry_after_s: int = 1,
    ) -> None:
        if max_active < 1:
            raise ValueError("max_active must be >= 1")
        if max_waiting < 0:
            raise ValueError("max_waiting must be >= 0")
        self.enabled = enabled
        self.max_active = max_active
        self.max_waiting = max_waiting
        self.max_wait_s = max_wait_s
        self.retry_after_s = retry_after_s
        self._active = 0
        self._waiting = 0
        self._cond = asyncio.Condition()

    @classmethod
    def from_settings(cls, settings: object) -> "AdmissionController":
        return cls(
            enabled=bool(getattr(settings, "admission_enabled")),
            max_active=int(getattr(settings, "max_concurrent_requests")),
            max_waiting=int(getattr(settings, "max_waiting_requests")),
            max_wait_s=float(getattr(settings, "max_queue_wait_ms")) / 1000.0,
        )

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    def _publish(self) -> None:
        ADMISSION_ACTIVE.set(self._active)
        ADMISSION_WAITING.set(self._waiting)

    async def acquire(self) -> AdmissionLease:
        if not self.enabled:
            return AdmissionLease(self, held=False)

        started = time.perf_counter()
        async with self._cond:
            if self._active < self.max_active:
                self._active += 1
                self._publish()
            else:
                if self._waiting >= self.max_waiting:
                    ADMISSION_REJECTED.labels(reason="full").inc()
                    raise AdmissionRejected("admission_full", self.retry_after_s)
                self._waiting += 1
                self._publish()
                queued = True
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.max_wait_s
                try:
                    while self._active >= self.max_active:
                        remaining = deadline - loop.time()
                        if remaining <= 0:
                            ADMISSION_TIMEOUTS.inc()
                            ADMISSION_REJECTED.labels(reason="wait_timeout").inc()
                            raise AdmissionRejected(
                                "admission_timeout", self.retry_after_s
                            )
                        try:
                            await asyncio.wait_for(self._cond.wait(), remaining)
                        except TimeoutError:
                            ADMISSION_TIMEOUTS.inc()
                            ADMISSION_REJECTED.labels(reason="wait_timeout").inc()
                            raise AdmissionRejected(
                                "admission_timeout", self.retry_after_s
                            ) from None
                    self._waiting -= 1
                    self._active += 1
                    queued = False
                    self._publish()
                except AdmissionRejected:
                    if queued:
                        self._waiting -= 1
                        self._publish()
                        self._cond.notify(1)
                    raise
                except asyncio.CancelledError:
                    if queued:
                        self._waiting -= 1
                        self._publish()
                        self._cond.notify(1)
                    raise

        ADMISSION_WAIT.observe(time.perf_counter() - started)
        return AdmissionLease(self, held=True)

    async def release_active(self) -> None:
        async with self._cond:
            if self._active > 0:
                self._active -= 1
            self._publish()
            self._cond.notify(1)
