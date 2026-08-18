"""数据源工厂 + 带缓存的历史行情读取。"""
from __future__ import annotations

import pandas as pd

from config import DEFAULT_SOURCE
from . import cache
from .base import DataSource
from .akshare_source import AkshareSource
from .yfinance_source import YFinanceSource

_SOURCES = {
    "akshare": AkshareSource,
    "yfinance": YFinanceSource,
}


def get_data_source(name: str | None = None, **kwargs) -> DataSource:
    name = name or DEFAULT_SOURCE
    if name not in _SOURCES:
        raise ValueError(f"未知数据源: {name}，可选: {list(_SOURCES)}")
    return _SOURCES[name](**kwargs)


def load_history(
    symbol: str,
    start: str,
    end: str | None = None,
    freq: str = "daily",
    source: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取历史行情，优先读缓存；缓存未覆盖区间时回源并更新缓存。"""
    source = source or DEFAULT_SOURCE
    if use_cache:
        cached = cache.load(source, symbol, freq)
        if cached is not None and not cached.empty:
            lo, hi = pd.Timestamp(start), pd.Timestamp(end) if end else cached.index.max()
            if cached.index.min() <= lo and cached.index.max() >= hi:
                return cached.loc[lo:hi]

    ds = get_data_source(source)
    df = ds.get_history(symbol, start, end, freq)
    if use_cache and not df.empty:
        cache.save(source, symbol, freq, df)
    return df
