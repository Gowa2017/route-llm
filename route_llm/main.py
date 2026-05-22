"""FastAPI application — OpenAI-compatible proxy endpoint."""

import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from route_llm.config import load_config
from route_llm.middleware import RoutingService
from route_llm.provider.anthropic import AnthropicProvider
from route_llm.tracker import UsageTracker

_log = logging.getLogger("route_llm")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.handlers[0].setFormatter(logging.Formatter("%(levelname)s %(message)s"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    app.state.service = RoutingService(config=config)
    app.state.tracker = UsageTracker()
    yield


app = FastAPI(
    title="route_llm",
    description="Multi-provider LLM proxy with time-based routing",
    version="0.1.0",
    lifespan=lifespan,
)


async def _check_auth(request: Request):
    """Validate x-api-key header against configured api_key."""
    cfg = request.app.state.config
    if not cfg.api_key:
        return  # auth disabled
    key = request.headers.get("x-api-key") or ""
    # Also check Authorization: Bearer <key>
    if not key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            key = auth_header[7:]
    if key != cfg.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models(request: Request):
    await _check_auth(request)
    return {
        "object": "list",
        "data": [
            {"id": "routed", "object": "model", "created": 0, "owned_by": "route_llm"}
        ],
    }


@app.get("/v1/usage")
async def get_usage(
    request: Request,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
    model: str | None = None,
):
    """Return aggregated usage per provider+model.

    Query params (all optional):
      start_date  - ISO date (default: today)
      end_date    - ISO date (default: today)
      provider    - substring filter on provider name
      model       - substring filter on model name
    """
    await _check_auth(request)
    tracker: UsageTracker = request.app.state.tracker
    result = tracker.query(
        start_date=start_date, end_date=end_date, provider=provider, model=model
    )
    return {"usage": result}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _check_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    _log.info("chat_completions body model=%s", body.get("model"))
    service: RoutingService = request.app.state.service
    try:
        resp = await service.chat_completion(body)
        return JSONResponse(content=resp.model_dump())
    except ValueError as e:
        _log.error("chat_completions ValueError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        detail = e.response.text or str(e)
        _log.error("chat_completions HTTPStatusError: %s", detail)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        _log.error("chat_completions error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


async def _track_stream_usage(stream, tracker, provider_name, model):
    """Wrap SSE stream, extract usage from events, record after stream ends."""
    usage = {}
    async for chunk in stream:
        yield chunk
        # Parse SSE data lines to accumulate usage
        for line in chunk.split(b"\n"):
            line = line.strip()
            if line.startswith(b"data: "):
                try:
                    data = json.loads(line[6:])
                    evt = data.get("type")
                    if evt == "message_start":
                        msg = data.get("message", {})
                        u = msg.get("usage", {})
                        usage["input_tokens"] = u.get("input_tokens", 0)
                        usage["cache_read_input_tokens"] = u.get(
                            "cache_read_input_tokens", 0
                        )
                        usage["cache_creation_input_tokens"] = u.get(
                            "cache_creation_input_tokens", 0
                        )
                    elif evt == "message_delta":
                        u = data.get("usage", {})
                        usage["output_tokens"] = u.get("output_tokens", 0)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

    if usage.get("input_tokens") is not None:
        tracker.record(
            provider_name,
            model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
        )


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic-format endpoint (compatible with Claude Code CLI)."""
    await _check_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    _log.info(
        "anthropic_messages body model=%s stream=%s",
        body.get("model"),
        body.get("stream"),
    )
    service: RoutingService = request.app.state.service
    stream = body.get("stream", False)
    try:
        # Streaming → 单次路由, 不支持重试 (已开始响应)
        if stream:
            provider_name, model, provider = service.route(body.get("model"))
            _log.info("routed → provider=%s model=%s", provider_name, model)
            body["model"] = model

            if isinstance(provider, AnthropicProvider):
                return StreamingResponse(
                    _track_stream_usage(
                        provider.proxy_request_stream(body),
                        service._tracker,
                        provider_name,
                        model,
                    ),
                    media_type="text/event-stream",
                )
            raise HTTPException(
                status_code=501,
                detail=f"Anthropic→{type(provider).__name__} routing not supported",
            )

        # 非流式 → 带自动故障转移
        exclude: set[str] = set()
        last_error: Exception | None = None

        for _attempt in range(3):
            provider_name, model = service._router.select(body.get("model"), exclude)
            _log.info("routed → provider=%s model=%s", provider_name, model)
            provider = service._providers.get(provider_name)
            if not provider:
                raise ValueError(f"Unknown provider: {provider_name}")
            if not isinstance(provider, AnthropicProvider):
                raise HTTPException(
                    status_code=501,
                    detail=f"Anthropic→{type(provider).__name__} routing not supported",
                )

            body["model"] = model
            try:
                resp = await provider.proxy_request(body)
            except httpx.HTTPStatusError as e:
                last_error = e
                detail = e.response.text or str(e)
                if e.response.status_code in {429, 500, 502, 503}:
                    _log.warning(
                        "Provider %s %s failed (HTTP %d), retrying…",
                        provider_name,
                        model,
                        e.response.status_code,
                    )
                    service._router.failure_tracker.mark(provider_name)
                    exclude.add(provider_name)
                    continue
                _log.error("anthropic_messages HTTPStatusError: %s", detail)
                raise HTTPException(
                    status_code=e.response.status_code, detail=detail
                ) from e

            usage = resp.get("usage", {})
            if usage:
                service._tracker.record(
                    provider_name,
                    model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cache_read_input_tokens", 0),
                    usage.get("cache_creation_input_tokens", 0),
                )
            return JSONResponse(content=resp)

        # 所有重试用尽
        raise last_error or RuntimeError("All providers exhausted")
    except ValueError as e:
        _log.error("anthropic_messages ValueError: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        detail = e.response.text or str(e)
        _log.error("anthropic_messages HTTPStatusError: %s", detail)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except HTTPException:
        _log.error("anthropic_messages HTTPException", exc_info=True)
        raise
    except Exception as e:
        _log.error("anthropic_messages error: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
