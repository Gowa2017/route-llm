"""Routing engine — weighted provider selection with per-provider rules."""

import logging
import time
from datetime import datetime, time as time_type
from enum import Enum

from route_llm.models import AppConfig, ProviderConfig, RoutingRule

_log = logging.getLogger("route_llm")


class FailureType(str, Enum):
    TEMPORARY = "temporary"  # 429 限速, 503 服务临时不可用
    PERMANENT = "permanent"  # 500 内部错误, 502 网关错误


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
    """In-memory failure cooldown tracker with tiered handling.

    Temporary failures (429/503): short cooldown, low threshold.
    Permanent failures (500/502): long cooldown, high threshold.
    A single success resets counters for both types.
    """

    def __init__(
        self,
        temporary_cooldown: int = 30,
        temporary_threshold: int = 1,
        permanent_cooldown: int = 1800,
        permanent_threshold: int = 3,
    ):
        self._temporary_cooldown = temporary_cooldown
        self._temporary_threshold = temporary_threshold
        self._permanent_cooldown = permanent_cooldown
        self._permanent_threshold = permanent_threshold
        self._failures: dict[str, dict[FailureType, float]] = {}   # provider → {type → blocked_since}
        self._counts: dict[str, dict[FailureType, int]] = {}        # provider → {type → count}

    def mark(self, provider: str, failure_type: FailureType, reason: str = ""):
        """Record a failure. Block provider once consecutive failures reach threshold."""
        if provider not in self._failures:
            self._failures[provider] = {}
            self._counts[provider] = {}

        counts = self._counts[provider]
        count = counts.get(failure_type, 0) + 1
        counts[failure_type] = count

        threshold = (
            self._temporary_threshold if failure_type == FailureType.TEMPORARY
            else self._permanent_threshold
        )

        _log.warning(
            "Provider %s %s failure %d/%d%s%s",
            provider, failure_type, count, threshold,
            f": {reason}" if reason else "",
            " → BLOCKED" if count >= threshold else "",
        )

        if count >= threshold:
            self._failures[provider][failure_type] = time.time()

    def reset(self, provider: str):
        """Reset consecutive failure count for a provider (call on success)."""
        self._failures.pop(provider, None)
        self._counts.pop(provider, None)

    def is_blocked(self, provider: str) -> bool:
        """Check if provider is blocked by ANY failure type."""
        failures = self._failures.get(provider)
        if not failures:
            return False

        now = time.time()
        for ftype, ts in list(failures.items()):
            cooldown = (
                self._temporary_cooldown if ftype == FailureType.TEMPORARY
                else self._permanent_cooldown
            )
            if now - ts >= cooldown:
                # Cooldown expired for this type
                del failures[ftype]
                self._counts[provider].pop(ftype, None)

        # Clean up if no failures remain
        if not failures:
            self._failures.pop(provider, None)
            self._counts.pop(provider, None)
            return False

        return True

    def clear(self):
        self._failures.clear()
        self._counts.clear()

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

    def _count_providers_for_model(self, model_hint: str | None) -> int:
        """Count providers that can serve the given model."""
        if not model_hint:
            return len(self._providers_dict())

        count = 0
        for proto_type, vendors in self.config.providers.items():
            for vendor_name, pconfig in vendors.items():
                if model_hint in pconfig.models:
                    count += 1
        return count

    def _providers_dict(self) -> dict[str, ProviderConfig]:
        """Flatten providers dict into {key: config}."""
        result = {}
        for proto_type, vendors in self.config.providers.items():
            for vendor_name, pconfig in vendors.items():
                result[f"{proto_type}.{vendor_name}"] = pconfig
        return result

    def _find_candidates(
        self, model_hint: str | None, exclude: set[str],
    ) -> list[tuple[str, ProviderConfig]]:
        """Return (provider_key, config) pairs that can serve *model_hint*."""
        candidates: list[tuple[str, ProviderConfig]] = []
        blocked_but_only = []  # Blocked providers that are the only option

        total_providers_for_model = self._count_providers_for_model(model_hint)

        for proto_type, vendors in self.config.providers.items():
            for vendor_name, pconfig in vendors.items():
                key = f"{proto_type}.{vendor_name}"
                if key in exclude:
                    continue
                if self.available_providers and key not in self.available_providers:
                    continue
                if model_hint and model_hint not in pconfig.models:
                    continue

                is_blocked = self.failure_tracker.is_blocked(key)
                if is_blocked:
                    if total_providers_for_model == 1:
                        _log.warning(
                            "Only provider for model %s is %s (blocked), using as last resort",
                            model_hint or "any", key,
                        )
                        blocked_but_only.append((key, pconfig))
                    continue

                candidates.append((key, pconfig))

        # If we have no candidates but have blocked-only providers, use them
        if not candidates and blocked_but_only:
            return blocked_but_only

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
