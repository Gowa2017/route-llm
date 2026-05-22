"""Daily usage tracking — writes JSONL files per provider/model."""

import json
import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

_log = logging.getLogger("llmrouter")


class UsageTracker:
    """Appends usage records to data/usage/YYYY-MM-DD.jsonl."""

    def __init__(self, data_dir: str = "data"):
        self._base = Path(data_dir) / "usage"
        self._base.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ):
        today = date.today().isoformat()
        path = self._base / f"{today}.jsonl"
        record = {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "timestamp": today,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def today_summary(self) -> list[dict]:
        """Return today's aggregated usage per provider+model."""
        return self.query()

    def query(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> list[dict]:
        """Return aggregated usage over a date range, optionally filtered.

        Params default to today.  Provider/model filters do substring match.
        """
        today = date.today()
        start = date.fromisoformat(start_date) if start_date else today
        end = date.fromisoformat(end_date) if end_date else today

        rows = []
        d = start
        while d <= end:
            path = self._base / f"{d.isoformat()}.jsonl"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if provider and provider not in r["provider"]:
                            continue
                        if model and model not in r["model"]:
                            continue
                        rows.append(r)
            d += timedelta(days=1)

        agg: dict = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_creation_tokens": 0}
        )
        for r in rows:
            key = f"{r['provider']}/{r['model']}"
            agg[key]["calls"] += 1
            agg[key]["input_tokens"] += r["input_tokens"]
            agg[key]["output_tokens"] += r["output_tokens"]
            agg[key]["cache_read_tokens"] += r.get("cache_read_tokens", 0)
            agg[key]["cache_creation_tokens"] += r.get("cache_creation_tokens", 0)

        return [{"provider_model": k, **v} for k, v in sorted(agg.items())]
