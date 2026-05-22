"""Routing engine — weighted provider selection with per-provider rules."""

import time
from datetime import datetime, time as time_type

from route_llm.models import AppConfig, ProviderConfig, RoutingRule


def _parse_time(t_str: str) -> time_type:
    """Parse 'HH:MM' string to time object."""
    parts = t_str.split(":")
    return time_type(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def _in_time_range(start: str, end: str, now: time_type | None = None) -> bool:
    """Check if *now* falls within [start, end].

    Supports cross-day ranges: start > end means the range crosses midnight.
    """
    if now is None:
        now = datetime.now().time()
    start_t = _parse_time(start)
    end_t = _parse_time(end)

    if start_t <= end_t:
        return start_t <= now <= end_t
    return now >= start_t or now <= end_t


class FailureTracker:
    """In-memory failure cooldown tracker.

    Providers marked as failed are blocked for *cooldown* seconds (default 30 min).
    Cleanup is lazy — entries expire on the next ``is_blocked()`` check.
    """

    def __init__(self, cooldown: int = 1800):
        self._cooldown = cooldown
        self._failures: dict[str, float] = {}

    def mark(self, provider: str):
        self._failures[provider] = time.time()

    def is_blocked(self, provider: str) -> bool:
        ts = self._failures.get(provider)
        if ts is None:
            return False
        if time.time() - ts >= self._cooldown:
            del self._failures[provider]
            return False
        return True

    def clear(self):
        self._failures.clear()

    @property
    def failed_providers(self) -> set[str]:
        return set(self._failures.keys())


class Router:
    """Select provider by weight, then apply provider-level rules.

    Flow
    ----
    1. Find candidate providers that have the requested model.
    2. Pick the highest-weight candidate (skip blocked / excluded).
    3. Apply the provider's rules — model override or cross-provider redirect.
    """

    def __init__(self, config: AppConfig, available_providers: set[str] | None = None):
        self.config = config
        self.available_providers = available_providers
        self.failure_tracker = FailureTracker()

    def select(self, model_hint: str | None = None,
               exclude_providers: set[str] | None = None) -> tuple[str, str]:
        """Return (provider_name, model_name) for the current context."""
        exclude = exclude_providers or set()
        now = datetime.now().time()

        # Explicit provider.model — bind directly, skip routing
        if model_hint and "." in model_hint:
            explicit_provider, explicit_model = model_hint.split(".", 1)
            for proto_type, vendors in self.config.providers.items():
                for vendor_name, pconfig in vendors.items():
                    key = f"{proto_type}.{vendor_name}"
                    if vendor_name == explicit_provider and key not in exclude:
                        if not self.failure_tracker.is_blocked(key):
                            return key, explicit_model

        candidates = self._find_candidates(model_hint, exclude)
        if not candidates:
            return "", model_hint or ""

        provider_key = self._pick_by_weight(candidates)
        proto, vendor = provider_key.split(".", 1)
        pconfig = self.config.providers[proto][vendor]

        # Check provider-level rules
        matched = self._match_rules(pconfig.rules, model_hint, now)
        if matched:
            best = matched[0]
            resolved_model = best.model or model_hint or ""
            if best.provider:
                return best.provider, resolved_model
            return provider_key, resolved_model

        # No rule matched — use original model
        return provider_key, model_hint or ""

    # ── internals ──

    def _find_candidates(
        self, model_hint: str | None, exclude: set[str],
    ) -> list[tuple[str, ProviderConfig]]:
        """Return (provider_key, config) pairs that can serve *model_hint*."""
        candidates: list[tuple[str, ProviderConfig]] = []

        for proto_type, vendors in self.config.providers.items():
            for vendor_name, pconfig in vendors.items():
                key = f"{proto_type}.{vendor_name}"
                if key in exclude:
                    continue
                if self.available_providers and key not in self.available_providers:
                    continue
                if self.failure_tracker.is_blocked(key):
                    continue
                if model_hint and model_hint not in pconfig.models:
                    continue
                candidates.append((key, pconfig))

        return candidates

    def _pick_by_weight(self, candidates: list[tuple[str, ProviderConfig]]) -> str:
        """Pick provider with highest weight (stable sort for ties)."""
        candidates.sort(key=lambda t: t[1].weight, reverse=True)
        return candidates[0][0]

    def _match_rules(self, rules: list[RoutingRule], model_hint: str | None,
                     now: time_type) -> list[RoutingRule]:
        """Filter and sort rules by time/match_model, return highest-priority first."""
        matched: list[RoutingRule] = []
        for rule in rules:
            if rule.time_range and not _in_time_range(
                rule.time_range.start, rule.time_range.end, now
            ):
                continue
            if rule.match_model:
                models = [m.strip() for m in rule.match_model.split(",")]
                if model_hint not in models:
                    continue
            matched.append(rule)

        matched.sort(key=lambda r: r.priority, reverse=True)
        return matched
