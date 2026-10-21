"""Pure helpers for Phase 4. No GPU and no network."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Optional


def window_cost_usd(hourly_usd: float, window_s: float) -> float:
    if window_s < 0:
        raise ValueError("window_s must be >= 0")
    return hourly_usd * (window_s / 3600.0)


def cost_per_successful_request(window_cost: float, successes: int) -> Optional[float]:
    if successes <= 0:
        return None
    return window_cost / successes


def cost_per_million_output_tokens(window_cost: float, output_tokens: int) -> Optional[float]:
    if output_tokens <= 0:
        return None
    return window_cost / output_tokens * 1_000_000.0


def grafana_range_ms(start: datetime, end: datetime) -> tuple[int, int]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    if end_ms <= start_ms:
        raise ValueError("end must be after start")
    return start_ms, end_ms


def is_png(blob: bytes) -> bool:
    return len(blob) > 32 and blob.startswith(b"\x89PNG\r\n\x1a\n")


def redact_secrets(text: str) -> str:
    text = re.sub(r"hf_[A-Za-z0-9]+", "hf_REDACTED", text)
    text = re.sub(r"rpa_[A-Za-z0-9]+", "rpa_REDACTED", text)
    return text


def json_safe(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def percent_change(baseline: Optional[float], candidate: Optional[float]) -> Optional[float]:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return (candidate - baseline) / baseline * 100.0
