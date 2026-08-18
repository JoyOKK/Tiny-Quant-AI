"""美股数据源，基于 yfinance。"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .base import DataSource

_INTERVAL = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}


class YFinanceSource(DataSource):
    name = "yfinance"

    def get_history(self, symbol, start, end=None, freq="daily") -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=_INTERVAL.get(freq, "1d"),
            auto_adjust=True,
            progress=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        # 处理可能的多级列索引
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        return self._normalize(df)

    def get_realtime(self, symbol) -> dict:
        import yfinance as yf

        t = yf.Ticker(symbol)
        fi = getattr(t, "fast_info", {}) or {}
        price = fi.get("last_price") or fi.get("lastPrice")
        return {
            "symbol": symbol,
            "price": _to_float(price),
            "open": _to_float(fi.get("open")),
            "high": _to_float(fi.get("day_high")),
            "low": _to_float(fi.get("day_low")),
            "pre_close": _to_float(fi.get("previous_close")),
            "volume": _to_float(fi.get("last_volume")),
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        }

    def get_name(self, symbol: str) -> str:
        # Ticker.info 经常要几十秒，看板回测不能被它堵住；美股代码本身即可辨认。
        return symbol


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
