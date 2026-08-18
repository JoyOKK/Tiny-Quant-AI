"""波动率目标 + ATR 吊灯止损策略（现代 CTA 的仓位管理范式）。

三层叠加：
1. 趋势方向：收盘价站上 trend_ma 日均线才考虑做多；
2. 波动率目标（Volatility Targeting）：仓位 = 目标年化波动 / 实际年化波动，
   高波动时自动减仓、低波动时加仓，把组合波动稳定在 target_vol 附近，
   这是提升夏普比率最有效的手段之一；
3. ATR 吊灯止损（Chandelier Exit）：从持仓期间最高价回撤超过 atr_mult 倍 ATR 即离场，
   系统性压低最大回撤。

本策略输出 [0, 1] 的「连续仓位」（做多、上限不加杠杆），
无需修改回测引擎即可运行——引擎本就按 position * 收益 结算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import sma, atr

TRADING_DAYS = 252


@register("vol_target")
class VolTargetStrategy(Strategy):
    description = "波动率目标+ATR止损：趋势定方向，按目标波动动态调仓并吊灯止损"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "trend_ma": 100,     # 趋势过滤均线
            "target_vol": 0.15,  # 目标年化波动率
            "vol_window": 20,    # 实际波动率估计窗口
            "atr_window": 14,    # ATR 窗口
            "atr_mult": 3.0,     # 吊灯止损的 ATR 倍数
            "max_pos": 1.0,      # 仓位上限（不加杠杆=1.0）
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trend_ma = int(self.params["trend_ma"])
        target_vol = float(self.params["target_vol"])
        vol_window = int(self.params["vol_window"])
        atr_window = int(self.params["atr_window"])
        atr_mult = float(self.params["atr_mult"])
        max_pos = float(self.params["max_pos"])

        close = df["close"]
        ma = sma(close, trend_ma)
        ret = close.pct_change()
        realized_vol = ret.rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
        # 波动率目标仓位（实际波动越大仓位越小），并限制在 [0, max_pos]
        size = (target_vol / realized_vol).clip(upper=max_pos).fillna(0.0)

        if {"high", "low"}.issubset(df.columns):
            atr_series = atr(df, atr_window)
        else:  # 无高低价时用收益波动近似
            atr_series = ret.abs().rolling(atr_window).mean() * close

        c = close.to_numpy()
        m = ma.to_numpy()
        a = atr_series.to_numpy()
        sz = size.to_numpy()

        pos = np.zeros(len(close))
        holding = False
        peak = np.nan
        for i in range(len(close)):
            if np.isnan(m[i]) or np.isnan(a[i]) or np.isnan(sz[i]):
                pos[i] = 0.0
                continue
            if not holding:
                if c[i] > m[i]:            # 趋势转多，入场
                    holding = True
                    peak = c[i]
            else:
                peak = max(peak, c[i])
                stop = peak - atr_mult * a[i]
                if c[i] < m[i] or c[i] < stop:   # 趋势走坏或触发吊灯止损
                    holding = False
            pos[i] = sz[i] if holding else 0.0
        return pd.Series(pos, index=df.index)
