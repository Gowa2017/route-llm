"""Abstract provider interface."""

from abc import ABC, abstractmethod

from llmrouter.models import ChatCompletionRequest, ChatCompletionResponse, ProviderConfig


class BaseProvider(ABC):
    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        ...
