"""双均线交叉策略（经典技术指标策略示例）。

短均线上穿长均线 -> 满仓；下穿 -> 空仓。
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import sma


@register("ma_cross")
class MACrossStrategy(Strategy):
    description = "双均线交叉：短均线上穿长均线做多，下穿平仓"

    @classmethod
    def default_params(cls) -> dict:
        return {"fast": 5, "slow": 20}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        ma_fast = sma(df["close"], fast)
        ma_slow = sma(df["close"], slow)
        signal = (ma_fast > ma_slow).astype(float)
        # 前 slow 根数据均线未成形，置空仓
        signal[ma_slow.isna()] = 0.0
        return signal
