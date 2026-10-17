"""FastAPI entrypoint for mock (Phase 1) and remote vLLM (Phase 2) serving."""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.admission import AdmissionRejected
from app.backends import build_backend
from app.backends.vllm import VLLMRemoteError
from app.config import Settings, get_settings
from app.logging_config import RequestLogger, setup_logging
from app.metrics import (
    GENERATED_TOKENS,
    IN_FLIGHT,
    INFLIGHT,
    INFERENCE_DURATION,
    INPUT_TOKENS,
    QUEUE_DEPTH,
    QUEUE_WAIT,
    REAL_OUTPUT_TOKENS_PER_SEC,
    REMOTE_ERRORS,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    SIM_OUTPUT_TOKENS_PER_SEC,
    STREAM_REQUESTS,
    TTFT,
    mode_label,
    set_app_info,
)
from app.schemas import ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse
from app.serving import InferenceService
from app.tracing import configure as configure_tracing
from app.tracing import span

# Module-level handles populated during lifespan
_settings: Optional[Settings] = None
_service: Optional[InferenceService] = None
_request_logger: Optional[RequestLogger] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _settings, _service, _request_logger
    settings = get_settings()
    setup_logging(settings.log_level)
    configure_tracing(
        enabled=settings.tracing_enabled,
        otlp_endpoint=settings.otlp_endpoint,
        trace_log_path=str(settings.trace_log_path) if settings.trace_log_path else None,
    )
    backend = build_backend(settings)
    await backend.startup()
    service = InferenceService(backend, settings)
    request_logger = RequestLogger(settings.request_log_path)
    set_app_info(
        version=__version__,
        backend=backend.name,
        mode=settings.measurements_mode,
    )

    _settings = settings
    _service = service
    _request_logger = request_logger
    app.state.settings = settings
    app.state.service = service
    app.state.request_logger = request_logger

    yield

    await backend.shutdown()
    _settings = None
    _service = None
    _request_logger = None


app = FastAPI(
    title="LLM Inference Platform",
    description=(
        "Text-to-SQL inference serving. "
        "BACKEND=mock uses simulated CPU timings; "
        "BACKEND=vllm proxies to a remote OpenAI-compatible vLLM server (e.g. RunPod)."
    ),
    version=__version__,
    lifespan=lifespan,
)


def _get_service(request: Request) -> InferenceService:
    service: InferenceService = request.app.state.service
    return service


def _get_settings(request: Request) -> Settings:
    return request.app.state.settings


def _get_logger(request: Request) -> RequestLogger:
    return request.app.state.request_logger


def _ms(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(value * 1000.0, 3)


def _tokens_per_sec(output_tokens: int, inference_s: Optional[float]) -> Optional[float]:
    if inference_s is None or inference_s <= 0:
        return None
    return round(output_tokens / inference_s, 3)


def _validate_generate(body: GenerateRequest, settings: Settings) -> None:
    if len(body.question) > settings.max_question_chars:
        raise HTTPException(
            status_code=413,
            detail=f"question exceeds max_question_chars={settings.max_question_chars}",
        )
    if len(body.db_schema) > settings.max_schema_chars:
        raise HTTPException(
            status_code=413,
            detail=f"schema exceeds max_schema_chars={settings.max_schema_chars}",
        )
    if body.max_new_tokens > settings.max_new_tokens:
        raise HTTPException(
            status_code=400,
            detail=f"max_new_tokens exceeds configured max ({settings.max_new_tokens})",
        )


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    service = _get_service(request)
    settings = _get_settings(request)
    simulated = service.backend.name == "mock"
    return HealthResponse(
        status="ok" if service.backend.ready else "starting",
        backend=service.backend.name,
        ready=service.backend.ready,
        simulated=simulated,
        queue_depth=service.queue_depth,
        inflight=service.backend.inflight,
        admission_enabled=service.admission.enabled,
        admission_active=service.admission.active,
        admission_waiting=service.admission.waiting,
        version=__version__,
        vllm_base_url=settings.vllm_base_url if settings.backend == "vllm" else None,
        vllm_model_name=settings.vllm_model_name if settings.backend == "vllm" else None,
    )


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    service = _get_service(request)
    QUEUE_DEPTH.labels(backend=service.backend.name).set(service.queue_depth)
    INFLIGHT.labels(backend=service.backend.name).set(service.backend.inflight)
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
)
async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    settings = _get_settings(request)
    service = _get_service(request)
    logger = _get_logger(request)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    with span("request_validation", backend=service.backend.name):
        _validate_generate(body, settings)

    backend_name = service.backend.name
    mode = settings.measurements_mode
    IN_FLIGHT.labels(backend=backend_name, mode=mode).inc()
    t0 = time.perf_counter()
    error_category: Optional[str] = None
    http_status = 200

    try:
        result, latency_s = await service.generate(
            request_id=request_id,
            question=body.question,
            schema=body.db_schema,
            max_new_tokens=body.max_new_tokens,
        )
        mode = mode_label(result.simulated)
        if result.queue_wait_s is not None:
            QUEUE_WAIT.labels(backend=result.backend, mode=mode).observe(result.queue_wait_s)
        if result.inference_s is not None:
            INFERENCE_DURATION.labels(backend=result.backend, mode=mode).observe(
                result.inference_s
            )
        REQUEST_LATENCY.labels(backend=result.backend, mode=mode).observe(latency_s)
        GENERATED_TOKENS.labels(backend=result.backend, mode=mode).inc(result.output_tokens)
        INPUT_TOKENS.labels(backend=result.backend, mode=mode).inc(result.input_tokens)
        tok_per_s = _tokens_per_sec(result.output_tokens, result.inference_s)
        if tok_per_s is not None:
            if result.simulated:
                SIM_OUTPUT_TOKENS_PER_SEC.set(tok_per_s)
            else:
                REAL_OUTPUT_TOKENS_PER_SEC.set(tok_per_s)
        REQUESTS_TOTAL.labels(status="success", backend=result.backend, mode=mode).inc()
        QUEUE_DEPTH.labels(backend=result.backend).set(service.queue_depth)

        logger.log(
            request_id=request_id,
            backend=result.backend,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            queue_wait_s=result.queue_wait_s,
            inference_s=result.inference_s,
            latency_s=latency_s,
            http_status=200,
            simulated=result.simulated,
            extra={
                "ttft_ms": _ms(result.ttft_s),
                "queue_wait_known": result.queue_wait_s is not None,
                "inference_known": result.inference_s is not None,
            },
        )

        return GenerateResponse(
            request_id=request_id,
            sql=result.text,
            backend=result.backend,
            simulated=result.simulated,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            queue_wait_ms=_ms(result.queue_wait_s),
            inference_ms=_ms(result.inference_s),
            latency_ms=round(latency_s * 1000.0, 3),
            ttft_ms=_ms(result.ttft_s),
            simulated_output_tokens_per_sec=tok_per_s if result.simulated else None,
            output_tokens_per_sec=tok_per_s,
            metadata={
                "phase": 2 if not result.simulated else 1,
                "measurements": "simulated" if result.simulated else "real",
                **result.metadata,
            },
        )
    except HTTPException:
        raise
    except AdmissionRejected as exc:
        REQUESTS_TOTAL.labels(status="rejected", backend=backend_name, mode=mode).inc()
        latency_s = time.perf_counter() - t0
        logger.log(
            request_id=request_id,
            backend=backend_name,
            input_tokens=0,
            output_tokens=0,
            queue_wait_s=None,
            inference_s=None,
            latency_s=latency_s,
            http_status=503,
            error_category=exc.reason,
            simulated=backend_name == "mock",
        )
        raise HTTPException(
            status_code=503,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_s)},
        ) from exc
    except VLLMRemoteError as exc:
        mode = "real"
        category = exc.category
        REMOTE_ERRORS.labels(category=category).inc()
        REQUESTS_TOTAL.labels(status="error", backend=backend_name, mode=mode).inc()
        http_status = 504 if category == "timeout" else 502
        error_category = category
        latency_s = time.perf_counter() - t0
        logger.log(
            request_id=request_id,
            backend=backend_name,
            input_tokens=0,
            output_tokens=0,
            queue_wait_s=None,
            inference_s=None,
            latency_s=latency_s,
            http_status=http_status,
            error_category=error_category,
            simulated=False,
        )
        raise HTTPException(status_code=http_status, detail=str(exc)) from exc
    except Exception as exc:
        http_status = 500
        error_category = "server_error"
        REQUESTS_TOTAL.labels(status="error", backend=backend_name, mode=mode).inc()
        latency_s = time.perf_counter() - t0
        logger.log(
            request_id=request_id,
            backend=backend_name,
            input_tokens=0,
            output_tokens=0,
            queue_wait_s=None,
            inference_s=None,
            latency_s=latency_s,
            http_status=http_status,
            error_category=error_category,
            simulated=backend_name == "mock",
        )
        raise HTTPException(status_code=500, detail="internal inference error") from exc
    finally:
        IN_FLIGHT.labels(backend=backend_name, mode=mode).dec()


def _sse_pack(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/generate/stream")
async def generate_stream(body: GenerateRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events stream of generated SQL tokens."""
    settings = _get_settings(request)
    service = _get_service(request)
    logger = _get_logger(request)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    with span("request_validation", backend=service.backend.name):
        _validate_generate(body, settings)
    try:
        with span("admission_wait", request_id=request_id):
            lease = await service.admission.acquire()
    except AdmissionRejected as exc:
        REQUESTS_TOTAL.labels(
            status="rejected",
            backend=service.backend.name,
            mode=settings.measurements_mode,
        ).inc()
        raise HTTPException(
            status_code=503,
            detail=exc.reason,
            headers={"Retry-After": str(exc.retry_after_s)},
        ) from exc

    backend_name = service.backend.name
    mode = settings.measurements_mode

    async def event_source() -> AsyncIterator[str]:
        IN_FLIGHT.labels(backend=backend_name, mode=mode).inc()
        t0 = time.perf_counter()
        status = "success"
        usage: dict[str, Any] = {}
        try:
            yield _sse_pack("meta", {"request_id": request_id, "backend": backend_name})
            async for event in service.generate_stream(
                request_id=request_id,
                question=body.question,
                schema=body.db_schema,
                max_new_tokens=body.max_new_tokens,
                lease=lease,
            ):
                if await request.is_disconnected():
                    status = "cancelled"
                    REMOTE_ERRORS.labels(category="cancelled").inc()
                    yield _sse_pack(
                        "error",
                        {"category": "cancelled", "message": "client disconnected"},
                    )
                    break

                if event.event == "usage":
                    usage = event.data
                    ttft = usage.get("ttft_s")
                    if ttft is not None:
                        TTFT.labels(
                            backend=usage.get("backend", backend_name),
                            mode=mode_label(bool(usage.get("simulated", True))),
                        ).observe(float(ttft))
                    inf = usage.get("inference_s")
                    if inf is not None:
                        INFERENCE_DURATION.labels(
                            backend=usage.get("backend", backend_name),
                            mode=mode_label(bool(usage.get("simulated", True))),
                        ).observe(float(inf))
                    out_tok = int(usage.get("output_tokens") or 0)
                    in_tok = int(usage.get("input_tokens") or 0)
                    m = mode_label(bool(usage.get("simulated", True)))
                    b = str(usage.get("backend", backend_name))
                    GENERATED_TOKENS.labels(backend=b, mode=m).inc(out_tok)
                    INPUT_TOKENS.labels(backend=b, mode=m).inc(in_tok)
                    tok_per_s = _tokens_per_sec(out_tok, usage.get("inference_s"))
                    if tok_per_s is not None:
                        if usage.get("simulated", True):
                            SIM_OUTPUT_TOKENS_PER_SEC.set(tok_per_s)
                        else:
                            REAL_OUTPUT_TOKENS_PER_SEC.set(tok_per_s)

                if event.event == "error":
                    status = "error"
                    category = str(event.data.get("category") or "remote_error")
                    REMOTE_ERRORS.labels(category=category).inc()

                yield _sse_pack(event.event, event.data)
        except Exception as exc:  # noqa: BLE001
            status = "error"
            REMOTE_ERRORS.labels(category="server_error").inc()
            yield _sse_pack(
                "error",
                {"category": "server_error", "message": str(exc)},
            )
        finally:
            await lease.release()
            latency_s = time.perf_counter() - t0
            REQUEST_LATENCY.labels(backend=backend_name, mode=mode).observe(latency_s)
            STREAM_REQUESTS.labels(
                status=status, backend=backend_name, mode=mode
            ).inc()
            REQUESTS_TOTAL.labels(
                status=status if status != "cancelled" else "error",
                backend=backend_name,
                mode=mode,
            ).inc()
            logger.log(
                request_id=request_id,
                backend=backend_name,
                input_tokens=int(usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                queue_wait_s=usage.get("queue_wait_s"),
                inference_s=usage.get("inference_s"),
                latency_s=latency_s,
                http_status=200 if status == "success" else 499 if status == "cancelled" else 502,
                error_category=None if status == "success" else status,
                simulated=bool(usage.get("simulated", backend_name == "mock")),
                extra={"stream": True, "ttft_ms": _ms(usage.get("ttft_s"))},
            )
            IN_FLIGHT.labels(backend=backend_name, mode=mode).dec()

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Request-Id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = request.headers.get("x-request-id")
    category = "client_error" if exc.status_code < 500 else "server_error"
    if exc.status_code == 503 and str(exc.detail) in {"admission_full", "admission_timeout"}:
        category = str(exc.detail)
    elif exc.status_code == 501:
        category = "not_implemented"
    elif exc.status_code == 413:
        category = "payload_too_large"
    elif exc.status_code == 502:
        category = "remote_error"
    elif exc.status_code == 504:
        category = "timeout"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            request_id=request_id,
            error=str(exc.detail),
            category=category,
        ).model_dump(),
        headers=dict(exc.headers or {}),
    )


@app.get("/", response_class=PlainTextResponse)
async def root(request: Request) -> str:
    settings = _get_settings(request)
    mode = "SIMULATED mock" if settings.backend == "mock" else "REAL remote vLLM"
    return (
        f"LLM Inference Platform — {mode}\n"
        "Endpoints: GET /healthz, GET /metrics, POST /generate, POST /generate/stream\n"
    )
