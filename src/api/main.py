"""The FastAPI application.

    uv run python -m src.api          # then open http://127.0.0.1:8000/docs

Endpoints (all but /health need the X-API-Key header):
    POST /ask                 natural-language question -> the LangGraph agent (5 tools)
    POST /forecast            demand forecast for one outlet x SKU
    GET  /recommend/{outlet}  next week's order recommendation
    GET  /health              readiness (no auth, for load balancers)

Every request writes ONE structured log line with latency, tokens and estimated cost.
Endpoints are plain `def` functions on purpose: the tools block on the database and the LLM, so
FastAPI runs them in a thread pool and the server stays responsive.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

import openai
from fastapi import APIRouter, Depends, FastAPI, Path, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.agent import AgentError, AgentResult, run_agent
from src.api.auth import configured_keys, require_api_key
from src.api.health import readiness
from src.api.logging_setup import clean_request_id, configure_logging
from src.api.schemas import (
    ID_PATTERN,
    AskRequest,
    AskResponse,
    ErrorResponse,
    ForecastRequest,
    HealthResponse,
    ToolCallInfo,
    UsageInfo,
)
from src.errors import NotFoundError, NotReadyError, ToolError
from src.tools import forecast_demand, recommend_order
from src.tools.schemas import ForecastResult, OrderRecommendation

__version__ = "0.1.0"
logger = logging.getLogger("fmcg.api")

AgentRunner = Callable[[str], AgentResult]


def get_agent_runner() -> AgentRunner:
    """Dependency that provides the function answering a question (replaced in tests)."""
    return run_agent


def warm_up(app: FastAPI) -> None:
    """Load the model and build the policy index now, so the first request is not slow."""
    started = time.perf_counter()
    app.state.policy_index = None
    try:
        from src.forecast.predict import load_model
        from src.tools.search_policies import build_index

        load_model()
        build_index()
        app.state.policy_index = True
    except Exception:  # noqa: BLE001 - the API still starts; /health shows what is missing
        app.state.policy_index = False
        logger.exception("warm-up failed")
    logger.info(
        "startup",
        extra={"fields": {"event": "startup", "warmup_s": round(time.perf_counter() - started, 2)}},
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if not configured_keys():
        logger.warning("API_KEY is not set: every protected endpoint will answer 503")
    elif any(len(key) < 16 for key in configured_keys()):
        logger.warning("API_KEY is shorter than 16 characters: use a long random value")
    if os.getenv("API_WARMUP", "true").lower() != "false":
        await run_in_threadpool(warm_up, app)
    yield


ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "Missing or invalid API key"},
    404: {"model": ErrorResponse, "description": "Unknown outlet, SKU or promotion"},
    422: {"model": ErrorResponse, "description": "Invalid input"},
    503: {"model": ErrorResponse, "description": "Service not ready (model, data or LLM key)"},
}

router = APIRouter(dependencies=[Depends(require_api_key)], responses=ERROR_RESPONSES)


@router.post("/ask", response_model=AskResponse, tags=["agent"], summary="Ask the copilot")
def ask(
    body: AskRequest,
    request: Request,
    run: AgentRunner = Depends(get_agent_runner),
) -> AskResponse:
    """Answer a question in plain English. The agent decides which of the 5 tools to call.

    The response lists the tools it used and what the request cost (tokens and an estimated USD).
    """
    result = run(body.question)
    tool_calls = [
        ToolCallInfo(
            name=call.name,
            args=call.args,
            ok=call.ok,
            seconds=call.seconds,
            sql=_sql_of(call) if call.name == "query_sales" else None,
        )
        for call in result.tool_calls
    ]
    usage = UsageInfo(
        llm_calls=result.llm_calls,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=round(result.cost_usd, 6),
    )
    # The question text is NOT logged (it may be sensitive); its length and the tools used are.
    request.state.log_fields.update(
        usage.model_dump()
        | {
            "question_chars": len(body.question),
            "tools": [c.name for c in result.tool_calls],
            "hit_step_limit": result.hit_step_limit,
        }
    )
    return AskResponse(
        request_id=request.state.request_id,
        answer=result.answer,
        tool_calls=tool_calls,
        usage=usage,
        seconds=result.seconds,
        hit_step_limit=result.hit_step_limit,
    )


def _sql_of(call) -> str | None:
    import json

    try:
        return json.loads(call.output).get("sql")
    except (json.JSONDecodeError, AttributeError):
        return None


@router.post("/forecast", response_model=ForecastResult, tags=["tools"], summary="Forecast demand")
def forecast(body: ForecastRequest) -> ForecastResult:
    """Expected weekly units of one SKU at one outlet for the next 1-4 weeks."""
    as_of = body.as_of.isoformat() if body.as_of else None
    return forecast_demand(body.outlet_id, body.sku_id, body.weeks, as_of)


@router.get(
    "/recommend/{outlet_id}",
    response_model=OrderRecommendation,
    tags=["tools"],
    summary="Recommend next week's order",
)
def recommend(
    outlet_id: str = Path(pattern=ID_PATTERN, examples=["O010"]),  # pattern must be on Path()
) -> OrderRecommendation:
    """Per-SKU order suggestion = forecast + safety stock - estimated stock, largest first."""
    return recommend_order(outlet_id)


def create_app() -> FastAPI:
    app = FastAPI(
        title="FMCG Sales Copilot API",
        version=__version__,
        description="AI assistant for beverage sales reps: sales questions, forecasts, "
        "order recommendations, promotion impact and policy search. All data is synthetic.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.get("/health", response_model=HealthResponse, tags=["ops"], summary="Readiness")
    def health(response: Response) -> dict:
        """No authentication. 200 when the service can work, 503 when data or model is missing."""
        info = readiness(__version__, getattr(app.state, "policy_index", None))
        if info["status"] == "unavailable":
            response.status_code = 503
        return info

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Time every request, give it an id, and write one structured log line for it."""
        request_id = clean_request_id(request.headers.get("x-request-id")) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        request.state.log_fields = {}
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error", extra={"fields": {"request_id": request_id}})
            response = JSONResponse(
                {"detail": "Internal server error.", "request_id": request_id}, status_code=500
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = str(latency_ms)

        route = request.scope.get("route")
        fields = {
            "event": "request",
            "request_id": request_id,
            "method": request.method,
            "path": route.path if route else request.url.path,  # template: low cardinality
            "status": response.status_code,
            "latency_ms": latency_ms,
            "llm_calls": 0,  # endpoints that use the LLM overwrite these
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            **request.state.log_fields,
        }
        quiet = request.url.path == "/health" and response.status_code == 200
        level = (
            logging.DEBUG
            if quiet
            else logging.WARNING
            if response.status_code >= 500
            else logging.INFO
        )
        logger.log(level, "request", extra={"fields": fields})
        return response

    @app.exception_handler(ToolError)
    async def tool_error(request: Request, error: ToolError) -> JSONResponse:
        if isinstance(error, NotFoundError):
            status = 404
        elif isinstance(error, NotReadyError):
            status = 503
        else:
            status = 422
        return _error(request, status, str(error))

    @app.exception_handler(AgentError)
    async def agent_error(request: Request, error: AgentError) -> JSONResponse:
        if isinstance(error.__cause__, openai.RateLimitError):
            return _error(request, 429, str(error), headers={"Retry-After": "30"})
        return _error(request, 502, str(error))

    return app


def _error(request: Request, status: int, detail: str, headers: dict | None = None) -> JSONResponse:
    body = {"detail": detail, "request_id": getattr(request.state, "request_id", None)}
    return JSONResponse(body, status_code=status, headers=headers)


app = create_app()
