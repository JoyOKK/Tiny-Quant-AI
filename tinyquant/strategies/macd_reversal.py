"""MACD 快慢均线拐点策略（低频、只抓真峰谷）。

用 MACD 体系里的两条移动均线判断买卖，参数沿用经典 12/26/9：
- 快线 = EMA(12)：对股价反应灵敏，斜率常有噪声抖动；
- 慢线 = EMA(26)：平滑，能较准确刻画下跌/上涨趋势；
- signal(9)：经典 MACD 信号周期，此处保留兼容，主逻辑以快慢均线为准。

核心思想（严格低频，只在下述条件同时成立才动手）：
- 买点（谷底）：快、慢线整体一致下行——慢线平滑地下降、快线可小幅抖动；
  当「快线斜率走平（由负转 0/翻头）」且「慢线下降在放缓（负斜率绝对值变小）」，
  判定接近谷底 → 全仓买入。
- 卖点（山顶）：快、慢线整体一致上行——慢线平滑地上升、快线可小幅抖动；
  当「快线斜率走平（由正转 0/掉头）」且「慢线上升在放缓（正斜率变小）」，
  判定接近山顶 → 全仓卖出。

为「不被假信号骗、降低交易频次」做的约束：
1. 斜率按收盘价归一化（每日变化 / 价格），阈值在不同价位股票通用；
2. flat_eps 死区：快线斜率落入死区才算「走平」，忽略细小抖动；
3. 慢线趋势确认：trend_len 窗口内慢线方向一致（平滑），且自窗口极值的
   累计幅度 ≥ min_trend，过滤横盘/假趋势；快线只看拐头、允许抖动；
4. cooldown 冷静期：两次交易至少间隔若干根 K 线，避免来回被噪声打脸。

信号在收盘确认，回测按次日收盘成交，无未来函数。仓位为 0/1 长线波段。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import ema


@register("macd_reversal")
class MACDReversal(Strategy):
    description = "MACD快慢均线拐点：快线走平+慢线趋势放缓，谷底买/峰顶卖（低频）"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "fast": 12,          # 快线 EMA 周期
            "slow": 26,          # 慢线 EMA 周期
            "signal": 9,         # 经典 MACD 信号周期（保留兼容，主逻辑未用）
            "slope_span": 3,     # 用几日算斜率，平滑快线噪声
            "trend_len": 8,      # 确认慢线平滑趋势的回看长度
            "min_trend": 0.03,   # 慢线自窗口极值的累计幅度阈值（/价格，过滤横盘）
            "flat_eps": 0.0008,  # 斜率死区：日均变化<该值(/价格)视为走平
            "cooldown": 10,      # 两次交易最小间隔 K 线数，压低频率
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        fast = max(int(p["fast"]), 1)
        slow = max(int(p["slow"]), 2)
        span = max(int(p["slope_span"]), 1)
        trend_len = max(int(p["trend_len"]), 2)
        min_trend = float(p["min_trend"])
        flat_eps = float(p["flat_eps"])
        cooldown = max(int(p["cooldown"]), 0)

        close = df["close"]
        fast_line = ema(close, fast)
        slow_line = ema(close, slow)

        # 归一化日均斜率：(x_t - x_{t-span}) / (span * price)，跨股票可比
        c = close.replace(0, np.nan)
        s_fast = ((fast_line - fast_line.shift(span)) / (span * c)).to_numpy(dtype=float)
        s_slow = ((slow_line - slow_line.shift(span)) / (span * c)).to_numpy(dtype=float)
        slow_a = slow_line.to_numpy(dtype=float)
        c_a = c.to_numpy(dtype=float)

        n = len(close)
        pos = np.zeros(n)
        holding = False
        last_trade = -(10**9)

        need = slow + trend_len + span + 1
        for i in range(n):
            if i < need or np.isnan(s_fast[i]) or np.isnan(s_slow[i]) or np.isnan(c_a[i]):
                pos[i] = 1.0 if holding else 0.0
                continue

            w0 = i - trend_len + 1  # 慢线趋势窗口 [w0, i]
            slow_slope_win = s_slow[w0:i + 1]
            slow_val_win = slow_a[w0:i + 1]

            # ---- 谷底：慢线先平滑下跌一段，快线走平 + 慢线降速放缓 ----
            slow_down_smooth = bool(np.all(slow_slope_win <= flat_eps))
            drop_from_peak = (np.nanmax(slow_val_win) - slow_a[i]) / c_a[i]
            slow_downtrend = slow_down_smooth and (drop_from_peak >= min_trend)
            fast_bottom = (s_fast[i - span] < -flat_eps) and (s_fast[i] >= -flat_eps)  # 快线由降转平/翻头
            slow_decel_down = (s_slow[i] < 0) and (s_slow[i] - s_slow[i - span] > 0)    # 慢线下降放缓
            buy = slow_downtrend and fast_bottom and slow_decel_down

            # ---- 山顶：慢线先平滑上涨一段，快线走平 + 慢线升速放缓 ----
            slow_up_smooth = bool(np.all(slow_slope_win >= -flat_eps))
            rise_from_trough = (slow_a[i] - np.nanmin(slow_val_win)) / c_a[i]
            slow_uptrend = slow_up_smooth and (rise_from_trough >= min_trend)
            fast_top = (s_fast[i - span] > flat_eps) and (s_fast[i] <= flat_eps)        # 快线由升转平/掉头
            slow_decel_up = (s_slow[i] > 0) and (s_slow[i] - s_slow[i - span] < 0)      # 慢线上升放缓
            sell = slow_uptrend and fast_top and slow_decel_up

            cooled = (i - last_trade) >= cooldown
            if not holding and buy and cooled:
                holding = True
                last_trade = i
            elif holding and sell and cooled:
                holding = False
                last_trade = i
            pos[i] = 1.0 if holding else 0.0

        return pd.Series(pos, index=df.index, name="signal")
