"""均线拐点反转策略。

用一条均线（默认 10 日）平滑价格，再看均线的「斜率」判断趋势方向：
- 斜率 > 0 均线上行，< 0 均线下行。
- 买点：前期均线明确下行一段后，下行放缓并翻头向上（均线谷底拐点）。
- 卖点：前期均线明确上行一段后，上行放缓并掉头向下（均线顶部拐点）。

买入后持有（仓位=1），出现卖点平仓（仓位=0），属长线 0/1 波段策略。
信号在收盘确认，回测按次日收盘成交，无未来函数。

为降低假信号做了三重优化：
1. 斜率按价格归一化（百分比），阈值在不同价位股票上通用；
2. 斜率死区 flat_eps：接近 0 的斜率视为「走平」，横盘不乱动；
3. 前期趋势强度 min_trend + 确认延迟 confirm_days：要求均线先真跌/真涨一段，
   且反转持续若干日才交易，过滤一日回头的假拐点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import sma


@register("ma_reversal")
class MAReversal(Strategy):
    description = "均线拐点反转：均线由降转升买入、由升转降卖出（波段）"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "ma_window": 10,      # 趋势均线周期
            "slope_span": 3,      # 用几日算斜率（平滑噪声）
            "trend_len": 5,       # 确认前期趋势的回看长度
            "min_trend": 0.02,    # 前期趋势累计幅度阈值（过滤横盘）
            "flat_eps": 0.0005,   # 斜率死区：日均 <0.05% 视为走平
            "confirm_days": 1,    # 反转持续几日才确认
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        ma_window = int(p["ma_window"])
        slope_span = max(int(p["slope_span"]), 1)
        trend_len = max(int(p["trend_len"]), 1)
        min_trend = float(p["min_trend"])
        flat_eps = float(p["flat_eps"])
        confirm_days = max(int(p["confirm_days"]), 1)

        close = df["close"]
        ma = sma(close, ma_window)
        # 归一化日均斜率：(MA_t - MA_{t-k}) / (k * MA_t)
        slope = (ma - ma.shift(slope_span)) / (slope_span * ma.replace(0, np.nan))

        ma_a = ma.to_numpy(dtype=float)
        sl = slope.to_numpy(dtype=float)
        n = len(close)
        pos = np.zeros(n)
        holding = False

        # 预热：保证前期趋势、斜率、确认窗口所需的历史都已成形
        need = ma_window + slope_span + trend_len + confirm_days
        for i in range(n):
            if i < need or np.isnan(ma_a[i]) or np.isnan(sl[i]):
                pos[i] = 1.0 if holding else 0.0
                continue

            # 前期趋势：用「昨日」相对 trend_len 日前的均线变化
            prev = ma_a[i - 1]
            base = ma_a[i - 1 - trend_len]
            prior = (prev - base) / prev if prev else 0.0

            # 反转前一刻的斜率（确认窗口之前一天）
            was = sl[i - confirm_days]
            # 确认窗口内的斜率是否方向一致且越过死区
            win = sl[i - confirm_days + 1 : i + 1]
            up_confirmed = bool(np.all(win > flat_eps))
            down_confirmed = bool(np.all(win < -flat_eps))

            buy = (prior <= -min_trend) and (was < -flat_eps) and up_confirmed
            sell = (prior >= min_trend) and (was > flat_eps) and down_confirmed

            if not holding and buy:
                holding = True
            elif holding and sell:
                holding = False
            pos[i] = 1.0 if holding else 0.0

        return pd.Series(pos, index=df.index, name="signal")
