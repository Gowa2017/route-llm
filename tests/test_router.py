"""Tests for routing engine — FailureTracker + weighted selection + rules."""

import time
from datetime import time as time_type

import pytest

from route_llm.models import AppConfig, ProviderConfig, RoutingRule, TimeRange
from route_llm.router import FailureTracker, Router, _in_time_range, FailureType


# ── FailureTracker ──────────────────────────────────────────────────────────

class TestFailureTracker:
    def test_mark_and_is_blocked(self):
        ft = FailureTracker(temporary_cooldown=3600)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)
        assert ft.is_blocked("openai.deepseek")

    def test_not_marked(self):
        ft = FailureTracker(temporary_cooldown=3600)
        assert not ft.is_blocked("openai.deepseek")

    def test_temporary_cooldown_expires(self):
        ft = FailureTracker(temporary_cooldown=0.01)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)
        time.sleep(0.02)
        assert not ft.is_blocked("openai.deepseek")

    def test_permanent_cooldown_expires(self):
        ft = FailureTracker(permanent_cooldown=0.01, permanent_threshold=1)
        ft.mark("openai.deepseek", FailureType.PERMANENT)
        time.sleep(0.02)
        assert not ft.is_blocked("openai.deepseek")

    def test_clear(self):
        ft = FailureTracker(temporary_cooldown=3600)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)
        ft.mark("anthropic.zhipu", FailureType.PERMANENT)
        ft.clear()
        assert not ft.is_blocked("openai.deepseek")
        assert not ft.is_blocked("anthropic.zhipu")
        assert ft.failed_providers == set()

    def test_failed_providers_property(self):
        ft = FailureTracker(temporary_cooldown=3600, permanent_threshold=1)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)
        ft.mark("anthropic.zhipu", FailureType.PERMANENT)
        assert ft.failed_providers == {"openai.deepseek", "anthropic.zhipu"}

    def test_multiple_marks_same_provider(self):
        ft = FailureTracker(temporary_cooldown=3600)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)  # no crash
        assert ft.is_blocked("openai.deepseek")

    def test_temporary_vs_permanent_cooldown(self):
        """Verify different failure types use different cooldown times."""
        ft = FailureTracker(temporary_cooldown=1, permanent_threshold=1)
        # Temporary error should expire quickly
        ft.mark("openai.deepseek", FailureType.TEMPORARY)
        time.sleep(1.1)
        assert not ft.is_blocked("openai.deepseek")

    def test_is_blocked_when_any_type_blocks(self):
        """Provider is blocked if ANY failure type reaches threshold."""
        ft = FailureTracker(permanent_threshold=3)
        ft.mark("openai.deepseek", FailureType.TEMPORARY)  # threshold=1, blocked
        ft.mark("openai.deepseek", FailureType.PERMANENT)  # count=1, not blocked
        assert ft.is_blocked("openai.deepseek")

    def test_only_provider_not_blocked_on_temporary_error(self):
        """Unique provider should still be used even if blocked."""
        cfg = AppConfig(providers={
            "openai": {
                "deepseek": ProviderConfig(
                    api_key="sk-test", base_url="https://test.com",
                    models={"gpt-4o": {"level": "large"}},
                ),
            },
        })
        router = Router(cfg, available_providers={"openai.deepseek"})
        router.failure_tracker.mark("openai.deepseek", FailureType.TEMPORARY)
        # Even though blocked, should still return the only provider
        provider, model = router.select("gpt-4o")
        assert provider == "openai.deepseek"


# ── InTimeRange (unchanged from original) ───────────────────────────────────

class TestInTimeRange:
    def test_normal_range_inside(self):
        assert _in_time_range("08:00", "23:00", time_type(12, 0))

    def test_normal_range_before(self):
        assert not _in_time_range("08:00", "23:00", time_type(6, 0))

    def test_normal_range_after(self):
        assert not _in_time_range("08:00", "23:00", time_type(23, 30))

    def test_cross_day_inside_night(self):
        assert _in_time_range("23:00", "08:00", time_type(2, 0))

    def test_cross_day_outside(self):
        assert not _in_time_range("23:00", "08:00", time_type(12, 0))

    def test_cross_day_edge_start(self):
        assert _in_time_range("23:00", "08:00", time_type(23, 0))

    def test_cross_day_edge_end(self):
        assert _in_time_range("23:00", "08:00", time_type(8, 0))


# ── Router – weighted selection ─────────────────────────────────────────────

class TestRouterWeightedSelect:
    def make_config(self, providers: dict | None = None) -> AppConfig:
        return AppConfig(
            providers=providers or {
                "openai": {
                    "deepseek": ProviderConfig(
                        api_key="sk-test", base_url="https://test.com",
                        weight=5,
                        models={"gpt-4o": {"level": "large"}},
                    ),
                },
                "anthropic": {
                    "zhipu": ProviderConfig(
                        api_key="sk-ant-test", base_url="https://zhipu.com",
                        weight=10,
                        models={"gpt-4o": {"level": "large"}},
                    ),
                    "baidu": ProviderConfig(
                        api_key="sk-baidu-test", base_url="https://baidu.com",
                        weight=3,
                        models={"gpt-4o": {"level": "large"}},
                    ),
                },
            }
        )

    def test_highest_weight_wins(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu", "anthropic.baidu"})
        provider, model = router.select("gpt-4o")
        # anthropic.zhipu has weight 10 > openai.deepseek weight 5
        assert provider == "anthropic.zhipu"
        assert model == "gpt-4o"

    def test_filters_by_model_availability(self):
        """Only providers with the model in their models dict are candidates."""
        cfg = self.make_config()
        # Only openai.deepseek has "gpt-4o-mini" (not configured above)
        # Actually all have "gpt-4o" in this setup, so add a model only one has
        cfg.providers["openai"]["deepseek"].models["gpt-4o-mini"] = {"level": "small"}
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu"})
        provider, model = router.select("gpt-4o-mini")
        assert provider == "openai.deepseek"

    def test_model_not_in_any_provider_returns_empty(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu"})
        provider, model = router.select("nonexistent-model")
        assert provider == ""
        assert model == "nonexistent-model"

    def test_exclude_providers(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu"})
        provider, model = router.select("gpt-4o", exclude_providers={"anthropic.zhipu"})
        # Next highest weight is openai.deepseek (weight 5)
        assert provider == "openai.deepseek"

    def test_exclude_all_providers_returns_empty(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu"})
        provider, model = router.select("gpt-4o", exclude_providers={"openai.deepseek", "anthropic.zhipu"})
        assert provider == ""

    def test_failure_tracker_blocks_provider(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu"})
        router.failure_tracker.mark("anthropic.zhipu", FailureType.TEMPORARY)
        provider, model = router.select("gpt-4o")
        # anthropic.zhipu is blocked, falls back to openai.deepseek
        assert provider == "openai.deepseek"

    def test_no_model_hint_picks_highest_weight(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek", "anthropic.zhipu"})
        provider, model = router.select()
        assert provider == "anthropic.zhipu"
        assert model == ""

    def test_available_providers_filter(self):
        cfg = self.make_config()
        router = Router(cfg, available_providers={"openai.deepseek"})
        provider, model = router.select("gpt-4o")
        assert provider == "openai.deepseek"


# ── Router – provider-level rules ───────────────────────────────────────────

class TestRouterRules:
    def make_config(self) -> AppConfig:
        return AppConfig(
            providers={
                "openai": {
                    "deepseek": ProviderConfig(
                        api_key="sk-test", base_url="https://test.com",
                        weight=5,
                        models={"gpt-4o": {"level": "large"}},
                    ),
                },
                "anthropic": {
                    "zhipu": ProviderConfig(
                        api_key="sk-ant-test", base_url="https://zhipu.com",
                        weight=10,
                        models={"glm-5.1": {"level": "large"}, "glm-4.7": {"level": "large"}},
                        rules=[
                            RoutingRule(
                                name="peak-downgrade",
                                time_range=TimeRange(start="14:00", end="18:00"),
                                match_model="glm-5.1",
                                model="glm-4.7",
                                priority=10,
                            ),
                        ],
                    ),
                },
            }
        )

    def test_rule_model_override_during_window(self, monkeypatch):
        monkeypatch.setattr("route_llm.router.datetime", _fake_datetime(time_type(16, 0)))
        cfg = self.make_config()
        router = Router(cfg, available_providers={"anthropic.zhipu", "openai.deepseek"})
        provider, model = router.select("glm-5.1")
        assert provider == "anthropic.zhipu"
        assert model == "glm-4.7"

    def test_rule_not_applied_outside_window(self, monkeypatch):
        monkeypatch.setattr("route_llm.router.datetime", _fake_datetime(time_type(10, 0)))
        cfg = self.make_config()
        router = Router(cfg, available_providers={"anthropic.zhipu", "openai.deepseek"})
        provider, model = router.select("glm-5.1")
        assert provider == "anthropic.zhipu"
        assert model == "glm-5.1"

    def test_rule_match_model_comma_separated(self, monkeypatch):
        monkeypatch.setattr("route_llm.router.datetime", _fake_datetime(time_type(16, 0)))
        cfg = self.make_config()
        # Add turbo to zhipu's models and a rule for it
        cfg.providers["anthropic"]["zhipu"].models["glm-5-turbo"] = {"level": "large"}
        cfg.providers["anthropic"]["zhipu"].rules.append(
            RoutingRule(
                name="turbo-downgrade",
                time_range=TimeRange(start="14:00", end="18:00"),
                match_model="glm-5.1, glm-5-turbo",
                model="glm-4.7",
                priority=10,
            ),
        )
        router = Router(cfg, available_providers={"anthropic.zhipu", "openai.deepseek"})
        p1, m1 = router.select("glm-5.1")
        assert m1 == "glm-4.7"
        p2, m2 = router.select("glm-5-turbo")
        assert m2 == "glm-4.7"

    def test_rule_without_match_model_applies_to_all(self, monkeypatch):
        monkeypatch.setattr("route_llm.router.datetime", _fake_datetime(time_type(16, 0)))
        cfg = self.make_config()
        cfg.providers["anthropic"]["zhipu"].models["any-model"] = {"level": "large"}
        cfg.providers["anthropic"]["zhipu"].rules.append(
            RoutingRule(
                name="catch-all",
                time_range=TimeRange(start="14:00", end="18:00"),
                model="glm-4.7",
                priority=5,
            ),
        )
        router = Router(cfg, available_providers={"anthropic.zhipu", "openai.deepseek"})
        provider, model = router.select("any-model")
        assert model == "glm-4.7"

    def test_cross_provider_redirect(self, monkeypatch):
        """Rule with 'provider' field redirects to another provider."""
        monkeypatch.setattr("route_llm.router.datetime", _fake_datetime(time_type(16, 0)))
        cfg = self.make_config()
        # Add cross-provider redirect rule to zhipu
        cfg.providers["anthropic"]["zhipu"].rules.append(
            RoutingRule(
                name="redirect-to-deepseek",
                time_range=TimeRange(start="14:00", end="18:00"),
                match_model="glm-5.1",
                provider="openai.deepseek",
                model="deepseek-chat",
                priority=20,
            ),
        )
        router = Router(cfg, available_providers={"anthropic.zhipu", "openai.deepseek"})
        provider, model = router.select("glm-5.1")
        assert provider == "openai.deepseek"
        assert model == "deepseek-chat"

    def test_highest_priority_rule_wins(self, monkeypatch):
        """Among multiple matching rules, highest priority wins."""
        monkeypatch.setattr("route_llm.router.datetime", _fake_datetime(time_type(16, 0)))
        cfg = self.make_config()
        cfg.providers["anthropic"]["zhipu"].rules.append(
            RoutingRule(
                name="low-priority",
                time_range=TimeRange(start="14:00", end="18:00"),
                match_model="glm-5.1",
                model="glm-4.5-air",
                priority=1,
            ),
        )
        # Original rule has priority 10 → "glm-4.7"
        # New rule has priority 1 → lower
        router = Router(cfg, available_providers={"anthropic.zhipu", "openai.deepseek"})
        provider, model = router.select("glm-5.1")
        assert model == "glm-4.7"


# ── helpers ─────────────────────────────────────────────────────────────────

def _fake_datetime(fixed_time: time_type):
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
