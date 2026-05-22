"""Tests for routing service / middleware."""

from route_llm.middleware import _key_configured
from route_llm.models import AppConfig, ProviderConfig
from route_llm.middleware import RoutingService


class TestKeyConfigured:
    def test_empty_key(self):
        assert not _key_configured("")

    def test_placeholder_openai(self):
        assert not _key_configured("sk-your-openai-key")

    def test_placeholder_anthropic(self):
        assert not _key_configured("sk-ant-your-anthropic-key")

    def test_real_key(self):
        assert _key_configured("sk-real-key-abc123")

    def test_real_anthropic_key(self):
        assert _key_configured("sk-ant-real-key-xyz")


class TestRoutingServiceInit:
    def test_skip_unconfigured_providers(self):
        """Providers with placeholder keys should not be loaded."""
        cfg = AppConfig(
            providers={
                "openai": {
                    "siliconflow": ProviderConfig(
                        api_key="sk-your-placeholder", base_url="https://a.com"
                    ),
                    "dmxapi": ProviderConfig(
                        api_key="sk-real-key", base_url="https://b.com"
                    ),
                }
            }
        )
        svc = RoutingService(cfg)
        assert "openai.siliconflow" not in svc._providers
        assert "openai.dmxapi" in svc._providers

    def test_all_placeholder_returns_empty(self):
        cfg = AppConfig(
            providers={
                "openai": {
                    "siliconflow": ProviderConfig(
                        api_key="sk-your-xxx", base_url="https://a.com"
                    ),
                }
            }
        )
        svc = RoutingService(cfg)
        assert len(svc._providers) == 0
