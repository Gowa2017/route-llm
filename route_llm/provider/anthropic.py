"""Anthropic provider — translates between OpenAI and Anthropic formats."""

import json
import logging
import time
import uuid

import httpx

_log = logging.getLogger("route_llm")

from route_llm.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
    ProviderConfig,
)
from route_llm.provider.base import BaseProvider

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


async def _consume_sse(resp) -> dict:
    """Read SSE stream from an Anthropic response, aggregate into final JSON."""
    message = None
    stop_reason = None
    stop_sequence = None
    usage = {}
    blocks: dict[int, dict] = {}
    block_texts: dict[int, list[str]] = {}

    async for line in resp.aiter_lines():
        line = line.strip()
        if not line or line.startswith("event:"):
            continue
        if line.startswith("data: "):
            data = json.loads(line[6:])
            evt = data.get("type")
            if evt == "message_start":
                message = data.get("message", {})
                usage = message.get("usage", {})
                _log.debug("_consume_sse message_start: message=%s usage=%s", message.get("id"), usage)
            elif evt == "content_block_start":
                idx = data.get("index", 0)
                blocks[idx] = data.get("content_block", {})
                block_texts.setdefault(idx, [])
            elif evt == "content_block_delta":
                idx = data.get("index", 0)
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    block_texts.setdefault(idx, []).append(delta.get("text", ""))
            elif evt == "content_block_stop":
                idx = data.get("index", 0)
                if idx in blocks and idx in block_texts:
                    blocks[idx]["text"] = "".join(block_texts[idx])
            elif evt == "message_delta":
                delta = data.get("delta", {})
                stop_reason = delta.get("stop_reason", stop_reason)
                stop_sequence = delta.get("stop_sequence", stop_sequence)
                delta_usage = data.get("usage", {})
                if delta_usage:
                    _log.debug("_consume_sse message_delta: usage before=%s delta=%s", usage, delta_usage)
                    usage.update(delta_usage)
                    _log.debug("_consume_sse message_delta: usage after=%s", usage)

    if message is None:
        _log.warning("_consume_sse: message is None, returning empty dict")
        return {}

    if stop_reason is not None:
        message["stop_reason"] = stop_reason
    if stop_sequence is not None:
        message["stop_sequence"] = stop_sequence
    if usage:
        message["usage"] = usage
        _log.debug("_consume_sse final: usage=%s", usage)

    sorted_blocks = [blocks[i] for i in sorted(blocks)]
    if any(not b.get("text") for b in sorted_blocks):
        # fill in partial content from blocks
        pass
    if sorted_blocks:
        message["content"] = sorted_blocks

    return message


def _to_anthropic_messages(
    req: ChatCompletionRequest,
) -> tuple[str | None, list[dict]]:
    """Convert OpenAI messages to Anthropic format.

    Returns (system_prompt, messages_list).
    Anthropic uses a separate `system` field instead of a role="system" message.
    """
    system = None
    messages = []
    for msg in req.messages:
        if msg.role == "system":
            system = msg.content
        else:
            role = "assistant" if msg.role == "assistant" else "user"
            messages.append({"role": role, "content": msg.content})
    return system, messages


def _from_anthropic_response(data: dict, model: str) -> ChatCompletionResponse:
    """Convert Anthropic response to OpenAI format."""
    content_parts = data.get("content", [])
    text = "".join(
        block["text"] for block in content_parts if block["type"] == "text"
    )

    stop_reason = data.get("stop_reason", "end_turn")
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_reason_map.get(stop_reason, "stop")

    usage_data = data.get("usage", {})
    usage = Usage(
        prompt_tokens=usage_data.get("input_tokens", 0),
        completion_tokens=usage_data.get("output_tokens", 0),
        total_tokens=usage_data.get("input_tokens", 0)
        + usage_data.get("output_tokens", 0),
        cache_read_input_tokens=usage_data.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", 0),
    )

    return ChatCompletionResponse(
        id=data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
        created=int(time.time()),
        model=model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


class AnthropicProvider(BaseProvider):
    """Proxy requests to Anthropic API, translating format."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
            timeout=120,
        )

    async def proxy_request(self, body: dict) -> dict:
        """Forward Anthropic-format request, return JSON dict.

        Handles both JSON and SSE responses — if upstream returns SSE,
        aggregates events into a single JSON dict internally.
        """
        resp = await self._client.post("/v1/messages", json=body)
        resp.raise_for_status()
        ct = resp.headers.get("content-type", "")
        if "text/event-stream" in ct:
            return await _consume_sse(resp)
        return resp.json()

    async def proxy_request_stream(self, body: dict):
        """Forward Anthropic-format request, yield SSE byte chunks."""
        async with self._client.stream("POST", "/v1/messages", json=body) as resp:
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "text/event-stream" not in ct:
                yield b"data: " + (await resp.aread()) + b"\n\n"
                return
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        system, messages = _to_anthropic_messages(request)
        body = {
            "model": request.model,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if system:
            body["system"] = system

        data = await self.proxy_request(body)
        return _from_anthropic_response(data, request.model or "")
