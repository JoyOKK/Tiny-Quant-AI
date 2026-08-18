"""基于 SQLite 的行情缓存，避免重复请求数据源。"""
from __future__ import annotations

import sqlite3
from contextlib import closing

import pandas as pd

from config import DB_PATH


def _table(source: str, symbol: str, freq: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in f"{source}_{symbol}_{freq}")
    return f"kline_{safe}"


def load(source: str, symbol: str, freq: str) -> pd.DataFrame | None:
    table = _table(source, symbol, freq)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        try:
            df = pd.read_sql(f'SELECT * FROM "{table}"', conn, parse_dates=["date"])
        except Exception:
            return None
    if df.empty:
        return None
    return df.set_index("date").sort_index()


def save(source: str, symbol: str, freq: str, df: pd.DataFrame) -> None:
    table = _table(source, symbol, freq)
    out = df.copy()
    out.index.name = "date"
    out = out.reset_index()
    with closing(sqlite3.connect(DB_PATH)) as conn:
        out.to_sql(table, conn, if_exists="replace", index=False)
