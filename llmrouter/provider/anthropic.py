"""Anthropic provider — translates between OpenAI and Anthropic formats."""

import time
import uuid

import httpx

from llmrouter.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
    ProviderConfig,
)
from llmrouter.provider.base import BaseProvider

_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096


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
        """Forward a raw Anthropic-format request and return raw response."""
        resp = await self._client.post("/v1/messages", json=body)
        resp.raise_for_status()
        return resp.json()

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
