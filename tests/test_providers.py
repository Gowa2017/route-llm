"""Tests for provider adapters with mocked HTTP."""

import json

import pytest
import respx
from httpx import Response

from llmrouter.models import ChatCompletionRequest, ChatMessage, ProviderConfig
from llmrouter.provider.anthropic import AnthropicProvider
from llmrouter.provider.openai_compat import OpenAICompatProvider


@pytest.fixture
def openai_cfg():
    return ProviderConfig(
        api_key="sk-test", base_url="https://api.openai.com"
    )


@pytest.fixture
def anthropic_cfg():
    return ProviderConfig(
        api_key="sk-ant-test", base_url="https://api.anthropic.com"
    )


pytestmark = pytest.mark.asyncio


class TestOpenAICompatProvider:
    @respx.mock
    async def test_chat_completion_basic(self, openai_cfg):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "id": "chatcmpl-xxx",
                    "object": "chat.completion",
                    "created": 1000000,
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello from OpenAI!",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
            )
        )

        provider = OpenAICompatProvider(openai_cfg)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="gpt-4o",
        )
        resp = await provider.chat_completion(req)

        assert resp.id == "chatcmpl-xxx"
        assert resp.choices[0].message.content == "Hello from OpenAI!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.total_tokens == 30

        # Verify upstream request format
        sent = route.calls.last.request.content
        import json

        body = json.loads(sent)
        assert body["model"] == "gpt-4o"
        assert body["messages"][0]["role"] == "user"
        assert "Authorization" in route.calls.last.request.headers
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"


class TestAnthropicProvider:
    @respx.mock
    async def test_chat_completion_basic(self, anthropic_cfg):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(
                200,
                json={
                    "id": "msg_abc123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello from Claude!"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 25},
                },
            )
        )

        provider = AnthropicProvider(anthropic_cfg)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")],
            model="claude-sonnet-4-20250514",
        )
        resp = await provider.chat_completion(req)

        assert resp.choices[0].message.content == "Hello from Claude!"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 25

        sent = route.calls.last.request.content
        import json

        body = json.loads(sent)
        assert body["model"] == "claude-sonnet-4-20250514"
        assert "x-api-key" in route.calls.last.request.headers
        assert route.calls.last.request.headers["x-api-key"] == "sk-ant-test"

    @respx.mock
    async def test_system_prompt_extraction(self, anthropic_cfg):
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(
                200,
                json={
                    "id": "msg_xyz",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 1},
                },
            )
        )

        provider = AnthropicProvider(anthropic_cfg)
        req = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="Be concise."),
                ChatMessage(role="user", content="Tell me a joke"),
            ],
            model="claude-sonnet-4-20250514",
        )
        await provider.chat_completion(req)

        body = json.loads(route.calls.last.request.content)
        assert body["system"] == "Be concise."
        assert body["messages"] == [{"role": "user", "content": "Tell me a joke"}]

    @respx.mock
    async def test_stop_reason_mapping(self, anthropic_cfg):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=Response(
                200,
                json={
                    "id": "msg_max",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "..."}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 5, "output_tokens": 100},
                },
            )
        )

        provider = AnthropicProvider(anthropic_cfg)
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Write a long essay")],
            model="claude-sonnet-4-20250514",
        )
        resp = await provider.chat_completion(req)
        assert resp.choices[0].finish_reason == "length"
