"""均线保形滤波(Savitzky-Golay)波峰波谷波段策略。

思路
----
先用 d 日均线平滑价格，再对该均线做 Savitzky-Golay「保形」滤波，得到一条
更干净、拐点清晰的曲线；据其一阶斜率的正负变化找波峰 / 波谷：

- 斜率由负转正（中间过 0 处即为波谷）→ 低点买入；
- 斜率由正转负（中间过 0 处即为波峰）→ 高点卖出。

以 5 日均线为例（ma_window=5）：均线保形曲线掉头向上时抄在谷底，
掉头向下时卖在峰顶，买入后满仓持有(1)，遇卖点平仓(0)，属长线 0/1 波段。

无未来函数
----------
Savitzky-Golay 常规实现是「中心窗口」，每点都会用到未来数据。这里改成
**因果（trailing）** 版：每个交易日只取「截至当日」的历史窗口做一次一侧
多项式拟合，取窗口末端的平滑值，因此当日信号只依赖当日及以前的行情。
回测引擎再按「次日收盘成交」执行，整体无前视偏差。

死区处理
--------
斜率按曲线价位归一化为「每日百分比」，slope_eps 定义一个死区：|斜率| 不超过
该阈值时视为「走平（≈0）」，不改变方向记忆；只有斜率真正翻越死区、且与上一个
明确方向相反时才触发买卖，从而把「中间过 0」的谷/峰识别得更稳，减少横盘抖动。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import sma


def _causal_savgol(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """因果 Savitzky-Golay 平滑：每点只用截至当日的历史窗口，取末端拟合值。

    这样得到的平滑序列不含未来信息，可安全用于生成交易信号。
    """
    n = len(y)
    out = np.full(n, np.nan)
    try:
        from scipy.signal import savgol_filter
    except Exception:  # noqa: BLE001 - 缺 scipy 时退化为因果滚动均值
        return pd.Series(y).rolling(window, min_periods=poly + 2).mean().to_numpy()

    for i in range(n):
        seg = y[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]        # 跳过均线预热期的前导 NaN
        w = len(valid)
        if w < poly + 2:
            continue
        win = w if w % 2 == 1 else w - 1   # savgol 窗口须为奇数且 ≤ 样本数
        p = min(poly, win - 1)
        out[i] = savgol_filter(valid[-win:], win, p)[-1]
    return out


@register("savgol_swing")
class SavgolSwing(Strategy):
    description = "均线保形滤波(SavGol)：曲线斜率由负转正买入、由正转负卖出（波段）"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "ma_window": 5,       # d 日均线周期（示例中的 5 日）
            "sg_window": 11,      # Savitzky-Golay 窗口(自动取奇数)
            "sg_poly": 3,         # 多项式阶数(< 窗口)
            "slope_span": 1,      # 计算斜率的间隔(日)
            "slope_eps": 0.0,     # 斜率死区(按价位归一化的日百分比)，0=纯过零
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        ma_window = max(int(p["ma_window"]), 1)
        sg_window = max(int(p["sg_window"]), 3)
        sg_poly = max(int(p["sg_poly"]), 1)
        slope_span = max(int(p["slope_span"]), 1)
        slope_eps = float(p["slope_eps"])

        close = df["close"]
        ma = sma(close, ma_window)
        curve = _causal_savgol(ma.to_numpy(dtype=float), sg_window, sg_poly)

        # 归一化日斜率：(curve_t - curve_{t-k}) / (k * curve_t)，得到每日百分比变化
        c = pd.Series(curve, index=df.index)
        slope = (c - c.shift(slope_span)) / (slope_span * c.replace(0, np.nan))
        sl = slope.to_numpy(dtype=float)

        n = len(close)
        pos = np.zeros(n)
        holding = False
        last_sign = 0  # 上一次明确的斜率方向：+1 上行 / -1 下行 / 0 未定

        for i in range(n):
            s = sl[i]
            if np.isnan(s):
                pos[i] = 1.0 if holding else 0.0
                continue

            cur_sign = 1 if s > slope_eps else (-1 if s < -slope_eps else 0)

            if not holding:
                # 由「下行」翻越死区转为「上行」→ 波谷，买入
                if cur_sign > 0 and last_sign < 0:
                    holding = True
            else:
                # 由「上行」翻越死区转为「下行」→ 波峰，卖出
                if cur_sign < 0 and last_sign > 0:
                    holding = False

            if cur_sign != 0:
                last_sign = cur_sign
            pos[i] = 1.0 if holding else 0.0

        return pd.Series(pos, index=df.index, name="signal")
