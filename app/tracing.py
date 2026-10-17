"""Optional OpenTelemetry spans for the API request path.

Disabled unless TRACING_ENABLED=true. Spans cover API stages only. They are
not GPU prefill or decode times. Do not put prompts, schemas, or tokens on
span attributes.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

_ENABLED = False
_EXPORTER: Any = None

_FORBIDDEN = {
    "prompt",
    "schema",
    "question",
    "sql",
    "authorization",
    "token",
    "hf_token",
    "api_key",
}


class _JsonlSpanExporter:
    """Write finished spans as JSON lines. Attributes already passed the forbid list."""

    def __init__(self, path: str) -> None:
        self.path = path

    def export(self, spans: Any) -> Any:
        import json
        from opentelemetry.sdk.trace.export import SpanExportResult

        lines = []
        for finished in spans:
            ctx = finished.context
            parent = finished.parent
            start = finished.start_time
            end = finished.end_time
            duration_ms = None
            if start is not None and end is not None:
                duration_ms = round((end - start) / 1_000_000.0, 3)
            attrs = {
                key: value
                for key, value in dict(finished.attributes or {}).items()
                if key.lower() not in _FORBIDDEN
            }
            lines.append(
                json.dumps(
                    {
                        "name": finished.name,
                        "trace_id": format(ctx.trace_id, "032x"),
                        "span_id": format(ctx.span_id, "016x"),
                        "parent_span_id": format(parent.span_id, "016x") if parent else None,
                        "duration_ms": duration_ms,
                        "attributes": attrs,
                    },
                    ensure_ascii=True,
                )
            )
        if lines:
            from pathlib import Path

            target = Path(self.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def configure(
    *,
    enabled: bool,
    otlp_endpoint: Optional[str] = None,
    trace_log_path: Optional[str] = None,
) -> None:
    global _ENABLED, _EXPORTER
    _ENABLED = enabled
    _EXPORTER = None
    if not enabled:
        return
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    class _MemoryExporter(SpanExporter):
        def __init__(self) -> None:
            self.spans: list[Any] = []

        def export(self, spans: Any) -> SpanExportResult:
            self.spans.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None

    from opentelemetry import trace as trace_api

    # Tests and a restarted API process both call configure(). The SDK
    # otherwise ignores every provider after the first.
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False
    provider = TracerProvider()
    memory = _MemoryExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    if trace_log_path:
        provider.add_span_processor(
            SimpleSpanProcessor(_JsonlSpanExporter(trace_log_path))
        )
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            OTLPSpanExporter = None  # type: ignore[misc, assignment]
        if OTLPSpanExporter is not None:
            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
            )
    trace.set_tracer_provider(provider)
    _EXPORTER = memory


def exported_spans() -> list[Any]:
    if _EXPORTER is None:
        return []
    return list(_EXPORTER.spans)


def reset() -> None:
    global _ENABLED, _EXPORTER
    _ENABLED = False
    _EXPORTER = None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    if not _ENABLED:
        yield
        return
    from opentelemetry import trace

    tracer = trace.get_tracer("llm-inference-platform")
    with tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if key.lower() in _FORBIDDEN or value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                current.set_attribute(key, value)
        try:
            yield
        except Exception as exc:
            current.record_exception(exc)
            raise
