"""Peak / off-peak (分时) pricing helpers.

基准价 input_price/cache_read_price/output_price 视为非高峰价;
peak_*_price 为高峰期价, 逐字段未配则回落基准价。

高峰窗口由 peak_windows 定义 (时段 + 星期), 厂商级配置, 模型级可覆盖。
判定基于每条 usage 记录的 timestamp, 因此历史数据可重算。
"""

from collections.abc import Callable
from datetime import datetime

from route_llm.models import AppConfig, ModelConfig, PeakWindow
from route_llm.router import _parse_time

BUCKET_PEAK = "peak"
BUCKET_OFFPEAK = "offpeak"


def _split_provider_model(provider_model: str) -> tuple[str, str, str] | None:
    """'anthropic.zhipu/glm-4.7' -> ('anthropic', 'zhipu', 'glm-4.7')."""
    if "/" not in provider_model:
        return None
    provider_key, model_name = provider_model.split("/", 1)
    if "." not in provider_key:
        return None
    proto, vendor = provider_key.split(".", 1)
    return proto, vendor, model_name


def _resolve(
    config: AppConfig, provider_model: str
) -> tuple[ModelConfig | None, list[PeakWindow]]:
    """Return (model config, effective peak windows) for a 'proto.vendor/model' key."""
    parts = _split_provider_model(provider_model)
    if not parts:
        return None, []
    proto, vendor, model_name = parts
    vendor_cfg = config.providers.get(proto, {}).get(vendor)
    if vendor_cfg is None:
        return None, []
    mc = vendor_cfg.models.get(model_name)
    # 模型级 peak_windows 覆盖厂商级; [] 显式表示该模型不分峰谷
    if mc is not None and mc.peak_windows is not None:
        return mc, mc.peak_windows
    return mc, vendor_cfg.peak_windows or []


def effective_peak_windows(config: AppConfig, provider_model: str) -> list[PeakWindow]:
    """高峰窗口, 模型级优先, 否则厂商级, 否则空列表。"""
    return _resolve(config, provider_model)[1]


def _in_window(window: PeakWindow, ts: datetime) -> bool:
    """判断 *ts* 是否落在窗口内 (时段 + 星期)。

    跨午夜时 (start > end), 午夜之后那段归属窗口起始的那一天。
    """
    t = ts.time()
    start, end = _parse_time(window.start), _parse_time(window.end)

    if start <= end:
        if not start <= t <= end:
            return False
        day = ts.weekday()
    elif t >= start:
        day = ts.weekday()
    elif t <= end:
        day = (ts.weekday() - 1) % 7
    else:
        return False

    return window.days is None or day in window.days


def is_peak(windows: list[PeakWindow], ts: datetime) -> bool:
    """任一窗口命中即为高峰。"""
    return any(_in_window(w, ts) for w in windows)


def has_peak_pricing(mc: ModelConfig) -> bool:
    """该模型是否配置了至少一个高峰价。"""
    return any(
        p is not None
        for p in (mc.peak_input_price, mc.peak_cache_read_price, mc.peak_output_price)
    )


def prices_for(mc: ModelConfig, peak: bool) -> dict[str, float | None]:
    """解析生效价格, 高峰价逐字段回落基准价。"""

    def pick(base: float | None, peak_price: float | None) -> float | None:
        return peak_price if peak and peak_price is not None else base

    return {
        "input": pick(mc.input_price, mc.peak_input_price),
        "cache_read": pick(mc.cache_read_price, mc.peak_cache_read_price),
        "output": pick(mc.output_price, mc.peak_output_price),
    }


def _record_time(record: dict) -> datetime | None:
    """记录时间戳; 缺失 / 只有日期 / 无法解析都返回 None。"""
    ts = record.get("timestamp")
    if not isinstance(ts, str) or "T" not in ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def make_bucket_fn(config: AppConfig) -> Callable[[dict], str | None] | None:
    """构造 usage record → bucket 的分桶函数。

    未配置任何高峰窗口时返回 None (调用方走无分桶的默认路径)。
    单个模型没有有效窗口或没配高峰价时, 该模型的记录返回 None, 不参与分桶。
    """
    if not _has_any_peak_window(config):
        return None

    memo: dict[str, tuple[list[PeakWindow], bool]] = {}

    def bucket_of(record: dict) -> str | None:
        key = f"{record.get('provider', '')}/{record.get('model', '')}"
        cached = memo.get(key)
        if cached is None:
            mc, windows = _resolve(config, key)
            cached = (windows, mc is not None and has_peak_pricing(mc))
            memo[key] = cached
        windows, priced = cached
        if not windows or not priced:
            return None
        ts = _record_time(record)
        if ts is None:
            return BUCKET_OFFPEAK
        return BUCKET_PEAK if is_peak(windows, ts) else BUCKET_OFFPEAK

    return bucket_of


def _has_any_peak_window(config: AppConfig) -> bool:
    for vendors in config.providers.values():
        for pcfg in vendors.values():
            if pcfg.peak_windows:
                return True
            if any(mc.peak_windows for mc in pcfg.models.values()):
                return True
    return False
