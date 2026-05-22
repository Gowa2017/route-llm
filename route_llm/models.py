"""Pydantic models for route_llm."""

from datetime import time
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool | None = False


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None


class TimeRange(BaseModel):
    start: str  # "HH:MM"
    end: str    # "HH:MM"


class RoutingRule(BaseModel):
    name: str
    time_range: TimeRange | None = None
    provider: str | None = None   # None = 同厂商重定向, 有值 = 跨厂商路由
    match_model: str | None = None  # incoming model to match
    model: str | None = None        # override target model
    priority: int = 0


class ModelConfig(BaseModel):
    level: str = "small"      # "small" | "medium" | "large"
    max_tokens: int = 4096


class ProviderConfig(BaseModel):
    api_key: str
    base_url: str
    weight: int = 1
    models: dict[str, ModelConfig] = {}
    rules: list[RoutingRule] = []


class AppConfig(BaseModel):
    api_key: str = ""
    providers: dict[str, dict[str, ProviderConfig]] = {}
