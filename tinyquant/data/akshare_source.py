"""A股数据源。

历史行情：优先东方财富（akshare.stock_zh_a_hist），失败自动回退新浪（stock_zh_a_daily）。
实时行情：优先新浪轻量行情端点（对单只更快），失败回退东方财富。
不同环境网络策略不同，双通道能显著提升可用性。
"""
from __future__ import annotations

import datetime as dt
import time

import pandas as pd

from .base import DataSource

_FREQ_MAP = {"daily": "daily", "weekly": "weekly", "monthly": "monthly"}
_SINA_HQ = "https://hq.sinajs.cn/list="


def _retry(fn, retries: int = 3, delay: float = 1.2):
    last = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries - 1:
                time.sleep(delay * (i + 1))
    raise last


def _sina_symbol(code: str) -> str:
    """把 '000001' 归一化成 'sz000001' 这类带交易所前缀的代码。"""
    c = code.lower()
    if c[:2] in ("sh", "sz", "bj"):
        return c
    if c.startswith("6") or c.startswith("9"):
        return "sh" + c
    if c.startswith(("4", "8")):
        return "bj" + c
    return "sz" + c


class AkshareSource(DataSource):
    name = "akshare"

    def __init__(self, adjust: str = "qfq"):
        # adjust: "qfq"前复权 / "hfq"后复权 / ""不复权
        self.adjust = adjust

    def get_name(self, symbol: str) -> str:
        """只查单只新浪行情里的简称，避免拉全市场代码表把回测卡住。"""
        try:
            import requests

            sym = _sina_symbol(symbol)
            headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
            r = requests.get(_SINA_HQ + sym, headers=headers, timeout=2)
            raw = r.text.split('"')[1] if '"' in r.text else ""
            name = raw.split(",")[0].strip() if raw else ""
            if name:
                return name
        except Exception:
            pass
        return symbol

    # ---------------- 历史行情 ----------------
    def get_history(self, symbol, start, end=None, freq="daily") -> pd.DataFrame:
        end = end or dt.date.today().strftime("%Y-%m-%d")
        try:
            return self._hist_eastmoney(symbol, start, end, freq)
        except Exception:
            return self._hist_sina(symbol, start, end, freq)

    def _hist_eastmoney(self, symbol, start, end, freq) -> pd.DataFrame:
        import akshare as ak

        df = _retry(
            lambda: ak.stock_zh_a_hist(
                symbol=symbol,
                period=_FREQ_MAP.get(freq, "daily"),
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=self.adjust,
            )
        )
        if df is None or df.empty:
            raise ValueError("eastmoney 无数据")
        df = df.rename(
            columns={
                "日期": "date", "开盘": "open", "最高": "high",
                "最低": "low", "收盘": "close", "成交量": "volume",
            }
        ).set_index("date")
        return self._normalize(df)

    def _hist_sina(self, symbol, start, end, freq) -> pd.DataFrame:
        import akshare as ak

        df = _retry(
            lambda: ak.stock_zh_a_daily(
                symbol=_sina_symbol(symbol),
                start_date=start.replace("-", ""),
                end_date=end.replace("-", ""),
                adjust=self.adjust or "",
            )
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.set_index("date")  # sina 已含 open/high/low/close/volume 列
        return self._normalize(df)

    # ---------------- 实时行情 ----------------
    def get_realtime(self, symbol) -> dict:
        try:
            return self._rt_sina(symbol)
        except Exception:
            return self._rt_eastmoney(symbol)

    def _rt_sina(self, symbol) -> dict:
        import requests

        sym = _sina_symbol(symbol)
        headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
        r = _retry(lambda: requests.get(_SINA_HQ + sym, headers=headers, timeout=8))
        text = r.text.strip()
        raw = text.split('"')[1] if '"' in text else ""
        f = raw.split(",")
        if len(f) < 9 or not f[3]:
            raise ValueError("sina 实时行情解析失败")
        return {
            "symbol": symbol,
            "name": f[0],
            "price": _to_float(f[3]),
            "open": _to_float(f[1]),
            "pre_close": _to_float(f[2]),
            "high": _to_float(f[4]),
            "low": _to_float(f[5]),
            "volume": _to_float(f[8]),
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        }

    def _rt_eastmoney(self, symbol) -> dict:
        import akshare as ak

        df = _retry(lambda: ak.stock_bid_ask_em(symbol=symbol))
        info = dict(zip(df["item"], df["value"]))
        price = info.get("最新") or info.get("最新价")
        return {
            "symbol": symbol,
            "price": _to_float(price),
            "open": _to_float(info.get("今开")),
            "high": _to_float(info.get("最高")),
            "low": _to_float(info.get("最低")),
            "pre_close": _to_float(info.get("昨收")),
            "volume": _to_float(info.get("总手")),
            "time": dt.datetime.now().isoformat(timespec="seconds"),
        }


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
