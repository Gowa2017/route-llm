"""Request routing orchestration."""

import logging

import httpx

from llmrouter.config import load_config
from llmrouter.models import AppConfig, ChatCompletionRequest, ChatCompletionResponse
from llmrouter.provider.anthropic import AnthropicProvider
from llmrouter.provider.base import BaseProvider
from llmrouter.provider.openai_compat import OpenAICompatProvider
from llmrouter.router import Router
from llmrouter.tracker import UsageTracker

_log = logging.getLogger("llmrouter")

_RETRYABLE_STATUSES = {429, 500, 502, 503}
_MAX_RETRIES = 3

_PLACEHOLDER_PREFIXES = ("sk-your-", "sk-ant-your-")


def _key_configured(api_key: str) -> bool:
    if not api_key:
        return False
    if api_key.startswith(_PLACEHOLDER_PREFIXES):
        return False
    return True


class RoutingService:
    """Orchestrates routing: pick provider → forward request → return response."""

    def __init__(self, config: AppConfig | None = None, tracker: UsageTracker | None = None):
        if config is None:
            config = load_config()
        self._config = config
        self._providers: dict[str, BaseProvider] = {}
        self._tracker = tracker or UsageTracker()

        for proto_type, vendors in config.providers.items():
            for vendor_name, pconfig in vendors.items():
                if not _key_configured(pconfig.api_key):
                    continue
                key = f"{proto_type}.{vendor_name}"
                if proto_type == "anthropic":
                    self._providers[key] = AnthropicProvider(pconfig)
                else:
                    self._providers[key] = OpenAICompatProvider(pconfig)

        self._router = Router(config, set(self._providers.keys()))
        self._log_loaded()

    def _log_loaded(self):
        log = logging.getLogger("llmrouter")
        log.info("Loaded %d providers:", len(self._providers))
        for key in sorted(self._providers):
            models = self._provider_models(key)
            if models:
                log.info("  └ %-25s %d models: %s", key, len(models), ", ".join(models))
            else:
                log.info("  └ %s", key)

    def _provider_models(self, key: str) -> list[str]:
        proto, vendor = key.split(".", 1)
        cfg = self._config.providers.get(proto, {}).get(vendor)
        return list(cfg.models.keys()) if cfg and cfg.models else []

    def route(self, model_hint: str | None = None) -> tuple[str, str, BaseProvider]:
        """Single-shot route (no retry)."""
        provider_name, model = self._router.select(model_hint)
        provider = self._providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider_name, model, provider

    async def _call_with_fallback(self, model_hint: str | None, call_fn):
        """Route then call, retrying fallback providers on HTTP 429/5xx.

        Failed providers are added to the failure tracker so subsequent
        requests skip them for the cooldown period (default 30 min).
        """
        exclude: set[str] = set()
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            provider_name, model = self._router.select(model_hint, exclude)
            provider = self._providers.get(provider_name)
            if not provider:
                raise ValueError(f"Unknown provider: {provider_name}")

            try:
                return await call_fn(provider, provider_name, model)
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in _RETRYABLE_STATUSES:
                    _log.warning(
                        "Provider %s %s failed (HTTP %d), retrying…",
                        provider_name, model, e.response.status_code,
                    )
                    self._router.failure_tracker.mark(provider_name)
                    exclude.add(provider_name)
                    continue
                raise

        raise last_error or RuntimeError("All providers exhausted")

    async def chat_completion(self, body: dict) -> ChatCompletionResponse:
        """Route a chat completion request and return the upstream response.

        Automatically retries with fallback providers on HTTP 429/5xx.
        """
        request = ChatCompletionRequest(**body)
        model_hint = request.model

        async def _call(provider, provider_name, model):
            request.model = model
            resp = await provider.chat_completion(request)
            if resp.usage:
                self._tracker.record(
                    provider_name, model,
                    resp.usage.prompt_tokens, resp.usage.completion_tokens,
                    resp.usage.cache_read_input_tokens, resp.usage.cache_creation_input_tokens,
                )
            return resp

        return await self._call_with_fallback(model_hint, _call)
