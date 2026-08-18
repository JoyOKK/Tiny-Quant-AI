"""唐奇安通道突破策略（海龟交易法则的简化单标的版本）。

经典趋势跟踪：
- 收盘突破过去 entry 日的最高价 -> 开多（满仓）；
- 收盘跌破过去 exit 日的最低价 -> 平仓（空仓）。

比双均线更抗震荡假信号：只在价格创出实质性新高时才入场，
用较短的 exit 通道及时离场，是 CTA / 商品趋势策略的鼻祖思路。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register


@register("donchian")
class DonchianStrategy(Strategy):
    description = "唐奇安通道突破：破 N 日新高做多，破 M 日新低平仓（趋势跟踪）"

    @classmethod
    def default_params(cls) -> dict:
        return {"entry": 20, "exit": 10}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        entry = int(self.params["entry"])
        exit_ = int(self.params["exit"])
        high = df["high"] if "high" in df.columns else df["close"]
        low = df["low"] if "low" in df.columns else df["close"]
        close = df["close"]

        # 用「前一日为止」的通道，避免把当日最高/最低算进去造成未来函数
        upper = high.rolling(entry).max().shift(1)
        lower = low.rolling(exit_).min().shift(1)

        pos = np.zeros(len(close))
        holding = False
        c = close.to_numpy()
        u = upper.to_numpy()
        l = lower.to_numpy()
        for i in range(len(close)):
            if np.isnan(u[i]) or np.isnan(l[i]):
                pos[i] = 0.0
                continue
            if not holding and c[i] > u[i]:
                holding = True
            elif holding and c[i] < l[i]:
                holding = False
            pos[i] = 1.0 if holding else 0.0
        return pd.Series(pos, index=df.index)
