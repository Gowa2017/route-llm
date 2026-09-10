"""Pydantic models for route_llm."""

from datetime import time
from pydantic import BaseModel, field_validator


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


_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


class PeakWindow(TimeRange):
    """高峰期窗口: 时段 + 可选星期过滤。days 省略 = 每天生效。

    start > end 表示跨午夜, 午夜之后那段归属窗口起始的那一天。
    """

    days: set[int] | None = None  # Mon=0 .. Sun=6

    @field_validator("days", mode="before")
    @classmethod
    def _parse_days(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            v = [v]
        days = set()
        for name in v:
            key = str(name).strip().lower()[:3]
            if key not in _WEEKDAYS:
                raise ValueError(f"invalid weekday {name!r}, expected one of mon..sun")
            days.add(_WEEKDAYS[key])
        return days


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
    input_price: float | None = None          # per million tokens, 非高峰基准价
    cache_read_price: float | None = None     # per million tokens, 非高峰基准价
    output_price: float | None = None         # per million tokens, 非高峰基准价
    peak_input_price: float | None = None          # 高峰期价, 未配则回落基准价
    peak_cache_read_price: float | None = None     # 高峰期价, 未配则回落基准价
    peak_output_price: float | None = None         # 高峰期价, 未配则回落基准价
    peak_windows: list[PeakWindow] | None = None   # 覆盖厂商级; [] = 该模型不分峰谷


class ProviderConfig(BaseModel):
    api_key: str
    base_url: str
    weight: int = 1
    models: dict[str, ModelConfig] = {}
    rules: list[RoutingRule] = []
    peak_windows: list[PeakWindow] | None = None


class AppConfig(BaseModel):
    api_key: str = ""
    providers: dict[str, dict[str, ProviderConfig]] = {}
