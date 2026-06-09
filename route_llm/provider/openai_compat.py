"""OpenAI-compatible API provider (OpenAI, DeepSeek, 通义千问, etc.)."""

import httpx

from route_llm.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    Usage,
    ProviderConfig,
)
from route_llm.provider.base import BaseProvider


class OpenAICompatProvider(BaseProvider):
    """Proxy requests to any OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=360,
        )

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        body = request.model_dump(exclude_none=True, exclude={"stream"})
        resp = await self._client.post("/v1/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        return ChatCompletionResponse(**data)
