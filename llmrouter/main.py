"""FastAPI application — OpenAI-compatible proxy endpoint."""

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from llmrouter.middleware import RoutingService
from llmrouter.provider.anthropic import AnthropicProvider
from llmrouter.tracker import UsageTracker

_log = logging.getLogger("llmrouter")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _log.addHandler(logging.StreamHandler())
    _log.handlers[0].setFormatter(logging.Formatter("%(levelname)s %(message)s"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = RoutingService()
    app.state.tracker = UsageTracker()
    yield


app = FastAPI(
    title="llmrouter",
    description="Multi-provider LLM proxy with time-based routing",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "routed", "object": "model", "created": 0, "owned_by": "llmrouter"}
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    service: RoutingService = request.app.state.service
    try:
        resp = await service.chat_completion(body)
        return JSONResponse(content=resp.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        detail = e.response.text or str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic-format endpoint (compatible with Claude Code CLI)."""
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    service: RoutingService = request.app.state.service
    try:
        provider_name, model, provider = service.route(body.get("model"))
        body["model"] = model

        if isinstance(provider, AnthropicProvider):
            resp = await provider.proxy_request(body)
            usage = resp.get("usage", {})
            if usage:
                service._tracker.record(
                    provider_name, model,
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                )
            return JSONResponse(content=resp)
        else:
            # Anthropic → OpenAI not yet supported
            raise HTTPException(
                status_code=501,
                detail=f"Anthropic→{type(provider).__name__} routing not supported",
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPStatusError as e:
        detail = e.response.text or str(e)
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
