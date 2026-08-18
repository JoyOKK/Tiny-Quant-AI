"""布林带均值回归策略（与趋势策略互补，适合震荡市）。

- 收盘跌破布林下轨 -> 认为超跌，开多；
- 收盘回到中轨（均线）上方 -> 均值回归完成，平仓；
- 可选：收盘继续跌破 stop_std 倍标准差 -> 止损离场，防止单边下跌被套。

趋势策略在震荡市反复挨打，均值回归恰好在震荡市赚钱，
两类策略组合能显著平滑资金曲线。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import sma


@register("boll_mr")
class BollingerMeanReversion(Strategy):
    description = "布林带均值回归：跌破下轨买入，回到中轨卖出（震荡市策略）"

    @classmethod
    def default_params(cls) -> dict:
        return {"window": 20, "num_std": 2.0, "stop_std": 3.5}

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        window = int(self.params["window"])
        num_std = float(self.params["num_std"])
        stop_std = float(self.params["stop_std"])
        close = df["close"]
        mid = sma(close, window)
        std = close.rolling(window).std()
        lower = mid - num_std * std
        stop = mid - stop_std * std

        c = close.to_numpy()
        m = mid.to_numpy()
        lo = lower.to_numpy()
        sp = stop.to_numpy()
        pos = np.zeros(len(close))
        holding = False
        for i in range(len(close)):
            if np.isnan(m[i]) or np.isnan(lo[i]):
                pos[i] = 0.0
                continue
            if not holding and c[i] < lo[i]:
                holding = True
            elif holding and (c[i] >= m[i] or c[i] < sp[i]):
                holding = False
            pos[i] = 1.0 if holding else 0.0
        return pd.Series(pos, index=df.index)
