"""数据源抽象接口。

所有数据源都返回统一格式的 DataFrame：
    index: DatetimeIndex（升序）
    columns: ["open", "high", "low", "close", "volume"]
这样上层的指标 / 策略 / 回测代码无需关心具体是 A股还是美股。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataSource(ABC):
    """行情数据源抽象基类。新增数据源只需继承并实现两个方法。"""

    #: 数据源标识（用于工厂注册）
    name: str = "base"

    @abstractmethod
    def get_history(
        self,
        symbol: str,
        start: str,
        end: str | None = None,
        freq: str = "daily",
    ) -> pd.DataFrame:
        """获取历史 K 线。

        Args:
            symbol: 标的代码，如 A股 "000001"，美股 "AAPL"。
            start:  起始日期 "YYYY-MM-DD"。
            end:    结束日期，None 表示至今。
            freq:   周期，"daily"/"weekly"/"monthly"。

        Returns:
            标准 OHLCV DataFrame。
        """

    @abstractmethod
    def get_realtime(self, symbol: str) -> dict:
        """获取实时行情快照。

        Returns:
            dict，至少包含 {"symbol", "price", "time"}，尽量含 open/high/low/pre_close/volume。
        """

    def get_name(self, symbol: str) -> str:
        """股票简称。默认不发网络请求，避免拖慢回测。"""
        return symbol

    # --------- 通用工具 ---------
    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """统一列顺序、类型与排序。"""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        for col in OHLCV_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        cols = [c for c in OHLCV_COLUMNS if c in df.columns]
        return df[cols].dropna(how="all")
