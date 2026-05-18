"""Tests for routing engine."""

from datetime import time

import pytest

from llmrouter.models import AppConfig, RoutingRule, TimeRange, ProviderConfig
from llmrouter.router import Router, _in_time_range


class TestInTimeRange:
    def test_normal_range_inside(self):
        assert _in_time_range("08:00", "23:00", time(12, 0))

    def test_normal_range_before(self):
        assert not _in_time_range("08:00", "23:00", time(6, 0))

    def test_normal_range_after(self):
        assert not _in_time_range("08:00", "23:00", time(23, 30))

    def test_normal_range_edge_start(self):
        assert _in_time_range("08:00", "23:00", time(8, 0))

    def test_normal_range_edge_end(self):
        assert _in_time_range("08:00", "23:00", time(23, 0))

    def test_cross_day_inside_night(self):
        assert _in_time_range("23:00", "08:00", time(2, 0))

    def test_cross_day_inside_evening(self):
        assert _in_time_range("23:00", "08:00", time(23, 30))

    def test_cross_day_outside(self):
        assert not _in_time_range("23:00", "08:00", time(12, 0))

    def test_cross_day_edge_start(self):
        assert _in_time_range("23:00", "08:00", time(23, 0))

    def test_cross_day_edge_end(self):
        assert _in_time_range("23:00", "08:00", time(8, 0))

    def test_cross_day_exact_midnight(self):
        assert _in_time_range("23:00", "08:00", time(0, 0))


class TestRouterSelect:
    def make_config(
        self,
        rules: list | None = None,
        default: str = "openai.deepseek",
    ) -> AppConfig:
        return AppConfig(
            default_provider=default,
            rules=[RoutingRule(**r) for r in (rules or [])],
            providers={
                "openai": {
                    "deepseek": ProviderConfig(
                        api_key="sk-test", base_url="https://test.com"
                    ),
                },
                "anthropic": {
                    "default": ProviderConfig(
                        api_key="sk-ant-test", base_url="https://anthropic.com"
                    ),
                },
            },
        )

    def test_time_based_selection(self, monkeypatch):
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(12, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "day",
                    "time_range": {"start": "08:00", "end": "20:00"},
                    "provider": "openai.deepseek",
                    "model": "gpt-4o",
                    "priority": 10,
                },
                {
                    "name": "night",
                    "time_range": {"start": "20:00", "end": "08:00"},
                    "provider": "anthropic.default",
                    "model": "claude-opus",
                    "priority": 10,
                },
            ]
        )
        router = Router(cfg)
        provider, model = router.select()
        assert provider == "openai.deepseek"
        assert model == "gpt-4o"

    def test_night_time_selection(self, monkeypatch):
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(2, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "day",
                    "time_range": {"start": "08:00", "end": "20:00"},
                    "provider": "openai.deepseek",
                    "model": "gpt-4o",
                    "priority": 10,
                },
                {
                    "name": "night",
                    "time_range": {"start": "20:00", "end": "08:00"},
                    "provider": "anthropic.default",
                    "model": "claude-opus",
                    "priority": 10,
                },
            ]
        )
        router = Router(cfg)
        provider, model = router.select()
        assert provider == "anthropic.default"
        assert model == "claude-opus"

    def test_fallback_to_default(self, monkeypatch):
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(12, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "night",
                    "time_range": {"start": "20:00", "end": "08:00"},
                    "provider": "anthropic.default",
                    "model": "claude-opus",
                    "priority": 10,
                },
            ],
            default="openai.deepseek",
        )
        router = Router(cfg)
        provider, model = router.select(model_hint="gpt-4o-mini")
        assert provider == "openai.deepseek"
        assert model == "gpt-4o-mini"

    def test_priority_wins(self, monkeypatch):
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(12, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "low",
                    "time_range": {"start": "08:00", "end": "20:00"},
                    "provider": "openai.deepseek",
                    "model": "gpt-4o-mini",
                    "priority": 1,
                },
                {
                    "name": "high",
                    "time_range": {"start": "08:00", "end": "20:00"},
                    "provider": "anthropic.default",
                    "model": "claude-haiku",
                    "priority": 100,
                },
            ]
        )
        router = Router(cfg)
        provider, model = router.select()
        # High priority should win
        assert provider == "anthropic.default"

    def test_model_hint_matches_rule(self, monkeypatch):
        """Rule with match_model only matches when request model equals match_model."""
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(12, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "for-gpt4",
                    "time_range": {"start": "08:00", "end": "20:00"},
                    "provider": "openai.deepseek",
                    "match_model": "gpt-4o",
                    "model": "gpt-4o",
                    "priority": 10,
                },
                {
                    "name": "catch-all",
                    "time_range": {"start": "08:00", "end": "20:00"},
                    "provider": "openai.deepseek",
                    "model": "gpt-4o-mini",
                    "priority": 5,
                },
            ]
        )
        router = Router(cfg)
        provider, model = router.select(model_hint="gpt-4o")
        assert model == "gpt-4o"
        assert provider == "openai.deepseek"

    def test_model_redirect(self, monkeypatch):
        """Redirect glm-5.1 → glm-4.7 during 14:00-18:00."""
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(16, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "glm-redirect",
                    "time_range": {"start": "14:00", "end": "18:00"},
                    "provider": "openai.deepseek",
                    "match_model": "glm-5.1",
                    "model": "glm-4.7",
                    "priority": 10,
                },
            ]
        )
        router = Router(cfg)
        provider, model = router.select(model_hint="glm-5.1")
        assert provider == "openai.deepseek"
        assert model == "glm-4.7"

    def test_model_redirect_no_match_wrong_time(self, monkeypatch):
        """Outside redirect window, match_model should not match."""
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(10, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "glm-redirect",
                    "time_range": {"start": "14:00", "end": "18:00"},
                    "provider": "openai.deepseek",
                    "match_model": "glm-5.1",
                    "model": "glm-4.7",
                    "priority": 10,
                },
            ],
            default="openai.deepseek",
        )
        router = Router(cfg)
        provider, model = router.select(model_hint="glm-5.1")
        assert provider == "openai.deepseek"
        assert model == "glm-5.1"  # no override, original model preserved

    def test_model_redirect_no_match_different_model(self, monkeypatch):
        """Request for a different model should not trigger the redirect."""
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(16, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "glm-redirect",
                    "time_range": {"start": "14:00", "end": "18:00"},
                    "provider": "openai.deepseek",
                    "match_model": "glm-5.1",
                    "model": "glm-4.7",
                    "priority": 10,
                },
            ],
            default="openai.deepseek",
        )
        router = Router(cfg)
        provider, model = router.select(model_hint="glm-4-flash")
        assert provider == "openai.deepseek"
        assert model == "glm-4-flash"

    def test_match_model_comma_separated(self, monkeypatch):
        """match_model supports comma-separated list."""
        monkeypatch.setattr("llmrouter.router.datetime", _fake_datetime(time(16, 0)))
        cfg = self.make_config(
            rules=[
                {
                    "name": "redirect-multi",
                    "time_range": {"start": "14:00", "end": "18:00"},
                    "provider": "openai.deepseek",
                    "match_model": "glm-5.1, glm-5-turbo",
                    "model": "glm-4.7",
                    "priority": 10,
                },
            ],
            default="openai.deepseek",
        )
        router = Router(cfg)

        # Both models in the list should match
        p1, m1 = router.select(model_hint="glm-5.1")
        assert m1 == "glm-4.7"

        p2, m2 = router.select(model_hint="glm-5-turbo")
        assert m2 == "glm-4.7"

        # Model not in list should NOT match
        p3, m3 = router.select(model_hint="glm-4.7")
        assert m3 == "glm-4.7"  # fallback, no override


def _fake_datetime(fixed_time: time):
    """Return a module-like object whose datetime.now() returns a fixed time."""

    class FakeDatetime:
        @classmethod
        def now(cls):
            return FakeDateTime(fixed_time)

    class FakeDateTime:
        def __init__(self, t):
            self._t = t

        def time(self):
            return self._t

    return FakeDatetime
