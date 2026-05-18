"""Routing engine — matches requests to providers based on rules."""

from datetime import datetime, time

from llmrouter.models import AppConfig, RoutingRule


def _parse_time(t_str: str) -> time:
    """Parse 'HH:MM' string to time object."""
    parts = t_str.split(":")
    return time(int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)


def _in_time_range(start: str, end: str, now: time | None = None) -> bool:
    """Check if *now* falls within [start, end].

    Supports cross-day ranges: start > end means the range crosses midnight.
    """
    if now is None:
        now = datetime.now().time()
    start_t = _parse_time(start)
    end_t = _parse_time(end)

    if start_t <= end_t:
        return start_t <= now <= end_t
    # Cross-day: e.g. 23:00-08:00
    return now >= start_t or now <= end_t


class Router:
    """Select the best provider+model for a request based on routing rules."""

    def __init__(self, config: AppConfig):
        self.config = config

    def select(self, model_hint: str | None = None) -> tuple[str, str]:
        """Return (provider_name, model_name) for the current context.

        1. Filter rules by current time.
        2. If *match_model* is set, rule only matches if request.model == match_model.
        3. Among remaining, pick highest-priority rule.
        4. If rule has *model*, use it as override; otherwise keep original model_hint.
        5. If no rule matches, fall back to default provider.
        """
        now = datetime.now().time()
        candidates: list[RoutingRule] = []

        for rule in self.config.rules:
            if rule.time_range and not _in_time_range(
                rule.time_range.start, rule.time_range.end, now
            ):
                continue
            if rule.match_model:
                models = [m.strip() for m in rule.match_model.split(",")]
                if model_hint not in models:
                    continue
            candidates.append(rule)

        if not candidates:
            return self._fallback(model_hint)

        candidates.sort(key=lambda r: r.priority, reverse=True)
        best = candidates[0]
        resolved_model = best.model or model_hint or ""
        return best.provider, resolved_model

    def _fallback(self, model_hint: str | None = None) -> tuple[str, str]:
        """Fallback to default provider or first available."""
        provider = self.config.default_provider or next(
            iter(self.config.providers), ""
        )
        return provider, model_hint or ""
