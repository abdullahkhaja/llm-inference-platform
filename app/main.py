"""FastAPI entrypoint for Phase 1 mock inference serving."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import __version__
from app.backends import build_backend
from app.config import Settings, get_settings
from app.logging_config import RequestLogger, setup_logging
from app.metrics import (
    GENERATED_TOKENS,
    IN_FLIGHT,
    INFERENCE_DURATION,
    INPUT_TOKENS,
    QUEUE_DEPTH,
    QUEUE_WAIT,
    REQUEST_LATENCY,
    REQUESTS_TOTAL,
    SIM_OUTPUT_TOKENS_PER_SEC,
    set_app_info,
)
from app.schemas import ErrorResponse, GenerateRequest, GenerateResponse, HealthResponse
from app.serving import InferenceService

# Module-level handles populated during lifespan
_settings: Optional[Settings] = None
_service: Optional[InferenceService] = None
_request_logger: Optional[RequestLogger] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _settings, _service, _request_logger
    settings = get_settings()
    setup_logging(settings.log_level)
    backend = build_backend(settings)
    await backend.startup()
    service = InferenceService(backend, settings)
    request_logger = RequestLogger(settings.request_log_path)
    set_app_info(version=__version__, backend=backend.name)

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
        "Phase 1: CPU mock inference serving for text-to-SQL. "
        "All latency and throughput numbers are SIMULATED."
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


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    service = _get_service(request)
    settings = _get_settings(request)
    return HealthResponse(
        status="ok" if service.backend.ready else "starting",
        backend=service.backend.name,
        ready=service.backend.ready,
        simulated=True,
        queue_depth=service.queue_depth,
        version=__version__,
    )


@app.get("/metrics")
async def metrics(request: Request) -> Response:
    service = _get_service(request)
    QUEUE_DEPTH.set(service.queue_depth)
    payload = generate_latest()
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        501: {"model": ErrorResponse},
    },
)
async def generate(body: GenerateRequest, request: Request) -> GenerateResponse:
    settings = _get_settings(request)
    service = _get_service(request)
    logger = _get_logger(request)
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    # Input size limits
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

    IN_FLIGHT.inc()
    t0 = time.perf_counter()
    error_category: Optional[str] = None
    http_status = 200
    result = None
    latency_s = 0.0

    try:
        result, latency_s = await service.generate(
            request_id=request_id,
            question=body.question,
            schema=body.db_schema,
            max_new_tokens=body.max_new_tokens,
        )
        QUEUE_WAIT.observe(result.queue_wait_s)
        INFERENCE_DURATION.observe(result.inference_s)
        REQUEST_LATENCY.observe(latency_s)
        GENERATED_TOKENS.inc(result.output_tokens)
        INPUT_TOKENS.inc(result.input_tokens)
        tok_per_s = (
            result.output_tokens / result.inference_s if result.inference_s > 0 else 0.0
        )
        SIM_OUTPUT_TOKENS_PER_SEC.set(tok_per_s)
        REQUESTS_TOTAL.labels(status="success").inc()
        QUEUE_DEPTH.set(service.queue_depth)

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
        )

        return GenerateResponse(
            request_id=request_id,
            sql=result.text,
            backend=result.backend,
            simulated=True,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            queue_wait_ms=round(result.queue_wait_s * 1000.0, 3),
            inference_ms=round(result.inference_s * 1000.0, 3),
            latency_ms=round(latency_s * 1000.0, 3),
            simulated_output_tokens_per_sec=round(tok_per_s, 3),
            metadata={
                "phase": 1,
                "measurements": "simulated",
                **result.metadata,
            },
        )
    except HTTPException:
        raise
    except NotImplementedError as exc:
        http_status = 501
        error_category = "not_implemented"
        REQUESTS_TOTAL.labels(status="error").inc()
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        http_status = 500
        error_category = "server_error"
        REQUESTS_TOTAL.labels(status="error").inc()
        latency_s = time.perf_counter() - t0
        logger.log(
            request_id=request_id,
            backend=service.backend.name,
            input_tokens=0,
            output_tokens=0,
            queue_wait_s=0.0,
            inference_s=0.0,
            latency_s=latency_s,
            http_status=http_status,
            error_category=error_category,
            simulated=True,
        )
        raise HTTPException(status_code=500, detail="internal inference error") from exc
    finally:
        IN_FLIGHT.dec()
        if error_category and http_status in (400, 413) and result is None:
            # validation errors are raised before try body completes logging
            pass


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = request.headers.get("x-request-id")
    category = "client_error" if exc.status_code < 500 else "server_error"
    if exc.status_code == 501:
        category = "not_implemented"
    elif exc.status_code == 413:
        category = "payload_too_large"
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            request_id=request_id,
            error=str(exc.detail),
            category=category,
        ).model_dump(),
    )


@app.get("/", response_class=PlainTextResponse)
async def root() -> str:
    return (
        "LLM Inference Platform — Phase 1 (SIMULATED mock backend)\n"
        "Endpoints: GET /healthz, GET /metrics, POST /generate\n"
    )
