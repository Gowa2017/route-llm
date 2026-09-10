"""Tests for peak / off-peak (分时) pricing."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from route_llm.models import AppConfig, ModelConfig, PeakWindow
from route_llm.pricing import (
    BUCKET_OFFPEAK,
    BUCKET_PEAK,
    effective_peak_windows,
    has_peak_pricing,
    is_peak,
    make_bucket_fn,
    prices_for,
)

# 2026-09-10 周四, 09-11 周五, 09-12 周六, 09-13 周日
WORKDAYS = ["mon", "tue", "wed", "thu", "fri"]


def _cfg(vendor_peak=None, model_peak="__unset__", peak_price=3.0):
    """Build a config with one anthropic vendor 'v' holding model 'm'."""
    model = {"input_price": 1.0, "output_price": 2.0}
    if peak_price is not None:
        model["peak_input_price"] = peak_price
        model["peak_output_price"] = peak_price * 2
    if model_peak != "__unset__":
        model["peak_windows"] = model_peak
    vendor = {"api_key": "k", "base_url": "u", "models": {"m": model}}
    if vendor_peak is not None:
        vendor["peak_windows"] = vendor_peak
    return AppConfig(providers={"anthropic": {"v": vendor}})


class TestPeakWindow:
    def test_days_normalized_to_weekday_numbers(self):
        w = PeakWindow(start="08:00", end="12:00", days=["Mon", "TUE", "wed"])
        assert w.days == {0, 1, 2}

    def test_full_names_accepted(self):
        assert PeakWindow(start="08:00", end="12:00", days=["monday"]).days == {0}

    def test_days_omitted_means_every_day(self):
        assert PeakWindow(start="08:00", end="12:00").days is None

    def test_invalid_day_rejected(self):
        with pytest.raises(ValidationError):
            PeakWindow(start="08:00", end="12:00", days=["funday"])


class TestIsPeak:
    def test_inside_window_on_matching_day(self):
        w = [PeakWindow(start="08:00", end="12:00", days=WORKDAYS)]
        assert is_peak(w, datetime(2026, 9, 10, 9, 0))  # 周四

    def test_outside_window(self):
        w = [PeakWindow(start="08:00", end="12:00", days=WORKDAYS)]
        assert not is_peak(w, datetime(2026, 9, 10, 13, 0))

    def test_weekend_excluded_even_inside_hours(self):
        w = [PeakWindow(start="08:00", end="12:00", days=WORKDAYS)]
        assert not is_peak(w, datetime(2026, 9, 12, 9, 0))  # 周六

    def test_days_omitted_covers_weekend(self):
        w = [PeakWindow(start="08:00", end="12:00")]
        assert is_peak(w, datetime(2026, 9, 12, 9, 0))

    def test_multiple_windows_any_match(self):
        w = [
            PeakWindow(start="08:00", end="12:00", days=WORKDAYS),
            PeakWindow(start="20:00", end="22:00"),
        ]
        assert is_peak(w, datetime(2026, 9, 13, 21, 0))  # 周日, 命中第二段

    def test_cross_midnight_spills_into_next_day(self):
        w = [PeakWindow(start="23:00", end="02:00", days=["fri"])]
        assert is_peak(w, datetime(2026, 9, 11, 23, 30))   # 周五 23:30
        assert is_peak(w, datetime(2026, 9, 12, 1, 0))     # 周六 01:00 归属周五

    def test_cross_midnight_day_filter_excludes_other_days(self):
        w = [PeakWindow(start="23:00", end="02:00", days=["fri"])]
        assert not is_peak(w, datetime(2026, 9, 12, 23, 30))  # 周六 23:30
        assert not is_peak(w, datetime(2026, 9, 10, 1, 0))    # 周四 01:00 归属周三

    def test_empty_windows_never_peak(self):
        assert not is_peak([], datetime(2026, 9, 10, 9, 0))


class TestEffectiveWindows:
    def test_vendor_level_applies_to_model(self):
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00"}])
        assert len(effective_peak_windows(cfg, "anthropic.v/m")) == 1

    def test_model_level_overrides_vendor(self):
        cfg = _cfg(
            vendor_peak=[{"start": "08:00", "end": "12:00"}],
            model_peak=[{"start": "20:00", "end": "22:00"}],
        )
        windows = effective_peak_windows(cfg, "anthropic.v/m")
        assert [w.start for w in windows] == ["20:00"]

    def test_empty_model_level_disables_vendor_default(self):
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00"}], model_peak=[])
        assert effective_peak_windows(cfg, "anthropic.v/m") == []

    def test_unknown_model_falls_back_to_vendor_windows(self):
        # 未配的模型拿不到 ModelConfig, 分桶处会因无高峰价而跳过, 这里只是厂商级查找
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00"}])
        assert len(effective_peak_windows(cfg, "anthropic.v/nope")) == 1

    def test_unknown_provider(self):
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00"}])
        assert effective_peak_windows(cfg, "anthropic.nope/m") == []
        assert effective_peak_windows(cfg, "no-slash") == []


class TestPricesFor:
    def test_peak_falls_back_to_base_per_field(self):
        mc = ModelConfig(input_price=1.0, output_price=2.0, peak_output_price=8.0)
        assert prices_for(mc, peak=True) == {
            "input": 1.0, "cache_read": None, "output": 8.0,
        }

    def test_offpeak_always_uses_base(self):
        mc = ModelConfig(input_price=1.0, output_price=2.0, peak_input_price=5.0)
        assert prices_for(mc, peak=False) == {
            "input": 1.0, "cache_read": None, "output": 2.0,
        }

    def test_zero_price_is_kept(self):
        mc = ModelConfig(input_price=0.0, peak_input_price=0.0)
        assert prices_for(mc, peak=True)["input"] == 0.0

    def test_has_peak_pricing(self):
        assert has_peak_pricing(ModelConfig(peak_input_price=0.0))
        assert not has_peak_pricing(ModelConfig(input_price=1.0))


class TestMakeBucketFn:
    def test_none_when_no_windows_configured(self):
        assert make_bucket_fn(_cfg()) is None

    def test_record_without_peak_price_is_not_bucketed(self):
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00"}], peak_price=None)
        fn = make_bucket_fn(cfg)
        assert fn is not None
        rec = {"provider": "anthropic.v", "model": "m", "timestamp": "2026-09-10T09:00:00"}
        assert fn(rec) is None

    def test_buckets_by_record_timestamp(self):
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00", "days": WORKDAYS}])
        fn = make_bucket_fn(cfg)
        base = {"provider": "anthropic.v", "model": "m"}
        assert fn({**base, "timestamp": "2026-09-10T09:00:00"}) == BUCKET_PEAK
        assert fn({**base, "timestamp": "2026-09-10T13:00:00"}) == BUCKET_OFFPEAK
        assert fn({**base, "timestamp": "2026-09-12T09:00:00"}) == BUCKET_OFFPEAK

    def test_unusable_timestamps_are_offpeak(self):
        cfg = _cfg(vendor_peak=[{"start": "08:00", "end": "12:00"}])
        fn = make_bucket_fn(cfg)
        base = {"provider": "anthropic.v", "model": "m"}
        assert fn(base) == BUCKET_OFFPEAK                                  # 缺 timestamp
        assert fn({**base, "timestamp": "2026-05-20"}) == BUCKET_OFFPEAK    # 旧数据只有日期
        assert fn({**base, "timestamp": "not-a-time"}) == BUCKET_OFFPEAK
        assert fn({**base, "timestamp": None}) == BUCKET_OFFPEAK
