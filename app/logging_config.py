"""Structured JSONL request logging and application logging setup."""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


class RequestLogger:
    """Append-only JSONL request log for benchmarks and troubleshooting.

    By default does NOT log full prompts/schemas.
    Durations should be passed in seconds (from monotonic clocks).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(
        self,
        *,
        request_id: str,
        backend: str,
        input_tokens: int,
        output_tokens: int,
        queue_wait_s: Optional[float],
        inference_s: Optional[float],
        latency_s: float,
        http_status: int,
        error_category: Optional[str] = None,
        simulated: bool = True,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        record: dict[str, Any] = {
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "simulated": simulated,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "queue_wait_ms": None
            if queue_wait_s is None
            else round(queue_wait_s * 1000.0, 3),
            "inference_ms": None
            if inference_s is None
            else round(inference_s * 1000.0, 3),
            "latency_ms": round(latency_s * 1000.0, 3),
            "http_status": http_status,
            "error_category": error_category,
        }
        if extra:
            record.update(extra)
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
