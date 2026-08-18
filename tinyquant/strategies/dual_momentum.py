"""双动量择时策略（Gary Antonacci 双动量的单标的版本）。

双动量 = 绝对动量 + 趋势过滤：
- 绝对动量：过去 lookback 日收益率 > 0（跑赢现金），说明标的本身处于上行；
- 趋势过滤：收盘价站上 trend_ma 日长期均线，过滤掉反弹陷阱。
两个条件同时满足才做多，否则空仓（持有现金）。

相比双均线，动量对「趋势的强度」更敏感，长期均线过滤能有效避开熊市。
"""
from __future__ import annotations

import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import sma, momentum


@register("dual_mom")
class DualMomentumStrategy(Strategy):
    description = "双动量择时：绝对动量为正且站上长期均线才做多（趋势择时）"

    @classmethod
    def default_params(cls) -> dict:
        return {"lookback": 120, "trend_ma": 200}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        lookback = int(self.params["lookback"])
        trend_ma = int(self.params["trend_ma"])
        close = df["close"]

        abs_mom = momentum(close, lookback)          # 过去 lookback 日涨跌幅
        ma = sma(close, trend_ma)

        long = (abs_mom > 0) & (close > ma)
        signal = long.astype(float)
        signal[abs_mom.isna() | ma.isna()] = 0.0
        return signal
