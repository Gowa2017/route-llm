"""Tests for usage cost calculation and table rendering."""

from route_llm.main import _calc_cost, _fmt_table
from route_llm.models import AppConfig


def _cfg(model: dict) -> AppConfig:
    return AppConfig(
        providers={
            "anthropic": {
                "v": {"api_key": "k", "base_url": "u", "models": {"m": model}}
            }
        }
    )


def _item(**overrides) -> dict:
    item = {
        "provider_model": "anthropic.v/m",
        "calls": 1,
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_read_tokens": 1_000_000,
    }
    item.update(overrides)
    return item


class TestCalcCost:
    def test_peak_bucket_uses_peak_prices(self):
        cfg = _cfg(
            {
                "input_price": 1.0,
                "cache_read_price": 0.5,
                "output_price": 2.0,
                "peak_input_price": 4.0,
                "peak_cache_read_price": 2.0,
                "peak_output_price": 8.0,
            }
        )
        off, peak = _calc_cost(
            [_item(bucket="offpeak"), _item(bucket="peak")], cfg
        )
        assert off["total_cost"] == 3.5
        assert peak["total_cost"] == 14.0

    def test_unbucketed_record_uses_base_prices(self):
        cfg = _cfg({"input_price": 1.0, "output_price": 2.0, "peak_input_price": 4.0})
        (item,) = _calc_cost([_item()], cfg)
        assert item["input_cost"] == 1.0
        assert item["output_cost"] == 2.0

    def test_peak_bucket_falls_back_per_field(self):
        cfg = _cfg({"input_price": 1.0, "output_price": 2.0, "peak_output_price": 8.0})
        (item,) = _calc_cost([_item(bucket="peak")], cfg)
        assert item["input_cost"] == 1.0     # 未配 peak_input_price → 基准价
        assert item["output_cost"] == 8.0

    def test_zero_price_still_counted(self):
        cfg = _cfg({"input_price": 0.0, "output_price": 0.0})
        (item,) = _calc_cost([_item()], cfg)
        assert item["total_cost"] == 0.0

    def test_model_without_config_untouched(self):
        cfg = _cfg({"input_price": 1.0})
        item = _item(provider_model="anthropic.v/other")
        assert _calc_cost([item], cfg) == [item]
        assert "total_cost" not in item


class TestFmtTable:
    def test_bucket_labels_rendered(self):
        rows = [
            {**_item(bucket="offpeak"), "total_cost": 0.1},
            {**_item(bucket="peak"), "total_cost": 0.9},
        ]
        table = _fmt_table(rows)
        assert "anthropic.v/m (非高峰)" in table
        assert "anthropic.v/m (高峰)" in table

    def test_no_bucket_keeps_plain_name(self):
        table = _fmt_table([{**_item(), "total_cost": 0.1}])
        assert "anthropic.v/m " in table
        assert "(高峰)" not in table
