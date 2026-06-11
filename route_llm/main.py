"""FastAPI application — OpenAI-compatible proxy endpoint."""

import json
import logging
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from route_llm.config import load_config
from route_llm.middleware import RoutingService
from route_llm.provider.anthropic import AnthropicProvider
from route_llm.tracker import UsageTracker

import os

_DEBUG_BODY = os.environ.get("ROUTE_LLM_DEBUG_BODY", "").lower() in ("1", "true", "yes")

_log = logging.getLogger("route_llm")
_log.setLevel(logging.DEBUG)
if not _log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    _log.addHandler(handler)

# 让 uvicorn.access 也用 asctime 格式
access_log = logging.getLogger("uvicorn.access")
if not access_log.handlers:
    access_log.addHandler(logging.StreamHandler())
access_log.handlers[0].setFormatter(
    logging.Formatter("%(asctime)s %(levelname)s %(message)s")
)


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


def _calc_cost(usage_items: list[dict], config) -> list[dict]:
    """Enrich usage records with cost if pricing is configured."""
    for item in usage_items:
        pm = item.get("provider_model", "")
        if "/" not in pm:
            continue
        provider_key, model_name = pm.split("/", 1)
        if "." not in provider_key:
            continue
        proto, vendor = provider_key.split(".", 1)
        vendor_cfg = config.providers.get(proto, {}).get(vendor)
        if not vendor_cfg:
            continue
        mc = vendor_cfg.models.get(model_name)
        if not mc:
            continue
        has_price = False
        cost = {}
        m = 1_000_000
        if mc.input_price is not None:
            cost["input_cost"] = round(item["input_tokens"] / m * mc.input_price, 6)
            has_price = True
        if mc.cache_read_price is not None:
            cost["cache_read_cost"] = round(item["cache_read_tokens"] / m * mc.cache_read_price, 6)
            has_price = True
        if mc.output_price is not None:
            cost["output_cost"] = round(item["output_tokens"] / m * mc.output_price, 6)
            has_price = True
        if has_price:
            cost["total_cost"] = round(sum(cost.values()), 6)
            item.update(cost)
    return usage_items


def _display_width(s: str) -> int:
    """CJK chars are 2 columns wide; others 1."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _pad(s: str, width: int) -> str:
    """Right-pad to *display* width."""
    return s + " " * (width - _display_width(s))


def _fmt_table(items: list[dict]) -> str:
    """Format usage data as box-drawing character table."""
    headers = ["模型", "调用次数", "输入 tokens", "输出 tokens", "缓存读取", "费用(元)", "平均首字", "最大首字", "平均 tok/s", "峰值 tok/s"]
    rows = []
    for item in items:
        cost = item.get("total_cost")
        cost_str = f"¥{cost:.3f}" if cost is not None else "—"
        avg_ttft = item.get("avg_ttft_ms")
        max_ttft = item.get("max_ttft_ms")
        avg_tps = item.get("avg_tokens_per_sec")
        peak_tps = item.get("peak_tokens_per_sec")
        rows.append([
            item["provider_model"],
            str(item["calls"]),
            f"{item['input_tokens']:,}",
            f"{item['output_tokens']:,}",
            f"{item.get('cache_read_tokens', 0):,}",
            cost_str,
            f"{avg_ttft:.0f}ms" if avg_ttft is not None else "—",
            f"{max_ttft:.0f}ms" if max_ttft is not None else "—",
            f"{avg_tps:.1f}" if avg_tps is not None else "—",
            f"{peak_tps:.1f}" if peak_tps is not None else "—",
        ])

    all_rows = [headers] + rows
    widths = [max(_display_width(c) for c in col) for col in zip(*all_rows)]

    def hline(left, mid, right, fill="─"):
        return left + mid.join(fill * (w + 2) for w in widths) + right

    def fmt_row(cells):
        return "│ " + " │ ".join(_pad(c, w) for c, w in zip(cells, widths)) + " │"

    lines = [
        hline("┌", "┬", "┐"),
        fmt_row(headers),
        hline("├", "┼", "┤"),
    ]
    for row in rows:
        lines.append(fmt_row(row))
        lines.append(hline("├", "┼", "┤"))
    lines[-1] = hline("└", "┴", "┘")
    return "\n".join(lines)


async def _get_usage_data(request: Request, start_date, end_date, provider, model):
    await _check_auth(request)
    tracker: UsageTracker = request.app.state.tracker
    result = tracker.query(
        start_date=start_date, end_date=end_date, provider=provider, model=model
    )
    return _calc_cost(result, request.app.state.config)


@app.get("/v1/usage")
async def get_usage(
    request: Request,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
    model: str | None = None,
):
    """Return aggregated usage per provider+model (JSON)."""
    result = await _get_usage_data(request, start_date, end_date, provider, model)
    return {"usage": result}


@app.get("/v1/usage/table")
async def get_usage_table(
    request: Request,
    start_date: str | None = None,
    end_date: str | None = None,
    provider: str | None = None,
    model: str | None = None,
):
    """Return aggregated usage as ASCII table."""
    result = await _get_usage_data(request, start_date, end_date, provider, model)
    return PlainTextResponse(_fmt_table(result) + "\n")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    await _check_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    _log.info("chat_completions body model=%s", body.get("model"))
    if _DEBUG_BODY:
        _log.debug("chat_completions body: %s", json.dumps(body, ensure_ascii=False))
    _log.info(
        "chat_completions extra fields: thinking=%s reasoning_effort=%s",
        body.get("thinking"),
        body.get("reasoning_effort"),
    )
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
    start = time.monotonic()
    usage = {}
    upstream_model = None
    ttft_ms = None
    buf = b""
    try:
        async for chunk in stream:
            yield chunk
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if line.startswith(b"data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        evt = data.get("type")
                        if evt == "message_start":
                            if ttft_ms is None:
                                ttft_ms = int((time.monotonic() - start) * 1000)
                            msg = data.get("message", {})
                            upstream_model = msg.get("model")
                            u = msg.get("usage", {})
                            _log.debug("message_start usage: %s", u)
                            usage["input_tokens"] = u.get("input_tokens", 0)
                            usage["cache_read_input_tokens"] = u.get(
                                "cache_read_input_tokens", 0
                            )
                            usage["cache_creation_input_tokens"] = u.get(
                                "cache_creation_input_tokens", 0
                            )
                        elif evt == "message_delta":
                            u = data.get("usage", {})
                            _log.debug("message_delta usage: %s", u)
                            if u:
                                usage["output_tokens"] = u.get("output_tokens", usage.get("output_tokens", 0))
                                # Some providers (xiaomi/mimo, zhipu/glm-5.1) report
                                # input_tokens=0 in message_start and the real values
                                # here in message_delta.
                                for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
                                    if key in u:
                                        usage[key] = u[key]
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
    except httpx.HTTPStatusError:
        raise
    except Exception as e:
        _log.error("_track_stream_usage error", exc_info=True)
        yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n".encode()
        return
    finally:
        duration_ms = int((time.monotonic() - start) * 1000)
        if usage.get("input_tokens") is not None:
            if upstream_model and upstream_model != model:
                _log.info(
                    "upstream: provider=%s requested=%s upstream_model=%s",
                    provider_name, model, upstream_model,
                )
            try:
                tracker.record(
                    provider_name,
                    model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cache_read_input_tokens", 0),
                    usage.get("cache_creation_input_tokens", 0),
                    upstream_model=upstream_model,
                    duration_ms=duration_ms,
                    ttft_ms=ttft_ms,
                )
            except Exception as rec_err:
                _log.error("_track_stream_usage record error", exc_info=True)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic-format endpoint (compatible with Claude Code CLI)."""
    await _check_auth(request)
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    _log.info(
        "anthropic_messages model=%s stream=%s max_tokens=%s "
        "thinking=%s reasoning_effort=%s context_management=%s output_config=%s",
        body.get("model"),
        body.get("stream"),
        body.get("max_tokens"),
        body.get("thinking"),
        body.get("reasoning_effort"),
        body.get("context_management"),
        body.get("output_config"),
    )
    if _DEBUG_BODY:
        _log.debug("anthropic_messages body: %s", json.dumps(body, ensure_ascii=False))
    service: RoutingService = request.app.state.service
    stream = body.get("stream", False)
    try:
        # Streaming → 单次路由, 不支持重试 (已开始响应)
        if stream:
            provider_name, model, provider = service.route(body.get("model"))
            _log.info("routed → provider=%s model=%s", provider_name, model)
            body["model"] = model

            if isinstance(provider, AnthropicProvider):
                stream = _track_stream_usage(
                    provider.proxy_request_stream(body),
                    service._tracker,
                    provider_name,
                    model,
                )
                try:
                    first_chunk = await stream.__anext__()
                except httpx.HTTPStatusError as e:
                    detail = e.response.text or str(e)
                    _log.error(
                        "anthropic_messages stream HTTPStatusError: %s", detail
                    )
                    raise HTTPException(
                        status_code=e.response.status_code, detail=detail
                    ) from e
                except Exception as e:
                    _log.error("anthropic_messages stream error: %s", e)
                    raise HTTPException(status_code=502, detail=str(e)) from e

                async def _wrapped():
                    yield first_chunk
                    async for chunk in stream:
                        yield chunk

                return StreamingResponse(
                    _wrapped(), media_type="text/event-stream"
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
                if last_error:
                    raise last_error
                raise ValueError(f"Unknown provider: {provider_name}")
            if not isinstance(provider, AnthropicProvider):
                raise HTTPException(
                    status_code=501,
                    detail=f"Anthropic→{type(provider).__name__} routing not supported",
                )

            body["model"] = model
            try:
                resp = await provider.proxy_request(body)
                service._router.failure_tracker.reset(provider_name)
            except httpx.HTTPStatusError as e:
                last_error = e
                detail = e.response.text or str(e)
                if e.response.status_code in {429, 500, 502, 503}:
                    _log.warning(
                        "Provider %s %s failed (HTTP %d): %s",
                        provider_name, model, e.response.status_code, detail[:200],
                    )
                    service._router.failure_tracker.mark(
                        provider_name, f"HTTP {e.response.status_code}: {detail[:100]}"
                    )
                    exclude.add(provider_name)
                    continue
                _log.error("anthropic_messages HTTPStatusError: %s", detail)
                raise HTTPException(
                    status_code=e.response.status_code, detail=detail
                ) from e

            upstream_model = resp.get("model")
            if upstream_model and upstream_model != model:
                _log.info(
                    "upstream: provider=%s requested=%s upstream_model=%s",
                    provider_name, model, upstream_model,
                )
            usage = resp.get("usage", {})
            if usage:
                service._tracker.record(
                    provider_name,
                    model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cache_read_input_tokens", 0),
                    usage.get("cache_creation_input_tokens", 0),
                    upstream_model=upstream_model,
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
