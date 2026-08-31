"""MA斜率(平滑)曲线的波峰波谷波段策略（严格因果，无未来函数）。

思路
----
用 k 日均线平滑价格得到均线曲线，对其一阶导（斜率）做**因果** Savitzky-Golay
滤波得到「MA斜率(平滑·因果)」曲线；再据这条斜率曲线自身的谷/峰买卖：

- 斜率曲线走到**谷底**（掉头回升确认）→ 买入；
- 斜率曲线走到**峰顶**（掉头回落确认）→ 卖出。

两种确认方式（mode）
--------------------
滞后主要来自「确认延迟」。提供两档：

- mode="turn"（默认，低滞后）：只要平滑斜率**掉头**（相邻变化越过一个很小的自适应
  死区 eps_k×局部std）并持续 persist 根，就在**当天**确认拐点。滞后≈persist 根，
  最及时，代价是横盘时可能多几次假信号。
- mode="zigzag"（稳健）：需等斜率从极值反向移动 rev×局部std 才确认，信号更少更稳，
  但滞后更大。

严格因果（无未来函数）
----------------------
平滑用因果(trailing) SavGol，波动阈值用尾部滚动 std，拐点用「只看当日及以前」的
在线状态机确认；决策所需信息当日收盘即可得。回测按次日成交执行，无前视偏差。
（已用「逐步追加未来数据、检验历史信号是否被改写」验证：改写比例 0%。）

lookahead 模式（仅供对照，默认关闭）
------------------------------------
lookahead=True 时改用中心式 SavGol + find_peaks，买卖点精确贴合看板中心式
「MA斜率(平滑)」的谷/峰，但**用到未来、不可实盘**，仅用于展示理想上限。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from .savgol_swing import _causal_savgol

try:
    from scipy.signal import find_peaks, savgol_filter
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _HAS_SCIPY = False


def _centered_savgol(s: pd.Series, window: int, poly: int) -> pd.Series:
    """中心式 Savitzky-Golay（用到未来），与看板 MA斜率(平滑) 一致；仅 lookahead 模式用。"""
    valid = s.dropna()
    if not _HAS_SCIPY or len(valid) < 3:
        return s.rolling(max(window, 1), min_periods=1, center=True).mean()
    win = window if window % 2 == 1 else window + 1
    if len(valid) < win:
        win = len(valid) if len(valid) % 2 == 1 else len(valid) - 1
    if win < 3:
        return s
    out = s.copy()
    out.loc[valid.index] = savgol_filter(valid.to_numpy(dtype=float), win, min(poly, win - 1))
    return out


@register("slope_swing")
class SlopeSwing(Strategy):
    description = "MA斜率(平滑)波段：斜率谷底买、峰顶卖（严格因果·低滞后；lookahead=True 仅对照）"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "ma_window": 5,      # k 日均线周期
            "sg_window": 11,     # 斜率保形滤波窗口(与看板 MA斜率(平滑·因果) 一致)
            "sg_poly": 3,        # 多项式阶数(< 窗口)
            "mode": "turn",      # turn=低滞后(掉头即确认) / zigzag=稳健(等反转)
            "eps_k": 0.0,        # [turn]死区＝eps_k×局部std；0=纯反转第一天(最及时,可能多抖)
            "persist": 1,        # [turn]掉头需连续确认的根数；1=反转第一天即触发
            "rev": 1.0,          # [zigzag]反转确认阈值＝rev×局部std
            "vol_win": 20,       # 计算局部波动(std)的滚动窗口
            "lookahead": False,  # True=中心式+find_peaks贴合可视谷峰(用未来,仅对照)
        }

    def _smoothed_slope(self, close: pd.Series, causal: bool) -> pd.Series:
        p = self.params
        w = max(int(p["ma_window"]), 1)
        slope = close.rolling(w, min_periods=1).mean().diff()   # 与看板一致
        sg_window = max(int(p["sg_window"]), 3)
        sg_poly = max(int(p["sg_poly"]), 1)
        if causal:
            return pd.Series(_causal_savgol(slope.to_numpy(dtype=float), sg_window, sg_poly),
                             index=close.index)
        return _centered_savgol(slope, sg_window, sg_poly)

    # ---- 理想对照：中心式 + find_peaks（用到未来，勿用于评估收益）----
    def _signals_lookahead(self, df: pd.DataFrame) -> pd.Series:
        sm = self._smoothed_slope(df["close"], causal=False)
        n = len(df)
        pos = np.zeros(n)
        cv = sm.to_numpy(dtype=float)
        mask = ~np.isnan(cv)
        if not _HAS_SCIPY or int(mask.sum()) < 3:
            return pd.Series(pos, index=df.index, name="signal")
        idx_valid = np.where(mask)[0]
        vv = cv[mask]
        std = float(np.nanstd(vv))
        prom = float(self.params["rev"]) * std if std > 0 else None
        peaks, _ = find_peaks(vv, prominence=prom)
        troughs, _ = find_peaks(-vv, prominence=prom)
        trough_idx = set(idx_valid[troughs].tolist())
        peak_idx = set(idx_valid[peaks].tolist())
        state = 0
        for i in range(n):
            if i in trough_idx:
                state = 1
            elif i in peak_idx:
                state = 0
            pos[i] = state
        return pd.Series(pos, index=df.index, name="signal")

    # ---- 低滞后：平滑斜率掉头即确认（严格因果）----
    def _signals_turn(self, sm: pd.Series, scale: np.ndarray) -> np.ndarray:
        p = self.params
        eps_k = max(float(p["eps_k"]), 0.0)
        persist = max(int(p["persist"]), 1)
        cv = sm.to_numpy(dtype=float)
        n = len(cv)
        pos = np.zeros(n)
        holding = False
        last_dir = 0          # 斜率曲线上一个「已确立」的涨跌方向：+1 升 / -1 降
        run_dir = 0           # 当前连续方向
        run_len = 0           # 当前方向已持续根数
        prev = np.nan
        for i in range(n):
            v, sc = cv[i], scale[i]
            # eps_k=0（纯反转第一天）时不依赖局部波动；eps_k>0 时需要有效 std 才能定死区
            bad_scale = eps_k > 0 and (np.isnan(sc) or sc <= 0)
            if np.isnan(v) or np.isnan(prev) or bad_scale:
                if not np.isnan(v):
                    prev = v
                pos[i] = 1.0 if holding else 0.0
                continue
            dead = eps_k * sc if eps_k > 0 else 0.0
            dv = v - prev
            d = 1 if dv > dead else (-1 if dv < -dead else 0)
            if d != 0 and d == run_dir:
                run_len += 1
            elif d != 0:
                run_dir, run_len = d, 1
            # d==0（走平）保持当前方向记忆，不重置
            # 方向连续 persist 根即「确立」；相对上一确立方向发生反转时买/卖
            if run_dir != 0 and run_len >= persist and run_dir != last_dir:
                if run_dir > 0 and last_dir < 0:      # 曲线由降转升第一天 → 谷底 → 买
                    holding = True
                elif run_dir < 0 and last_dir > 0:    # 曲线由升转降第一天 → 峰顶 → 卖
                    holding = False
                last_dir = run_dir
            prev = v
            pos[i] = 1.0 if holding else 0.0
        return pos

    # ---- 稳健：ZigZag 反转确认（严格因果）----
    def _signals_zigzag(self, sm: pd.Series, scale: np.ndarray) -> np.ndarray:
        rev = max(float(self.params["rev"]), 0.0)
        smv = sm.to_numpy(dtype=float)
        n = len(smv)
        pos = np.zeros(n)
        holding = False
        mode = 0
        hi = lo = np.nan
        for i in range(n):
            v, sc = smv[i], scale[i]
            if np.isnan(v) or np.isnan(sc) or sc <= 0:
                pos[i] = 1.0 if holding else 0.0
                continue
            if np.isnan(hi):
                hi = lo = v
                pos[i] = 1.0 if holding else 0.0
                continue
            thr = rev * sc
            hi = max(hi, v)
            lo = min(lo, v)
            if mode <= 0 and v >= lo + thr:
                holding, mode = True, 1
                hi = lo = v
            elif mode >= 0 and v <= hi - thr:
                holding, mode = False, -1
                hi = lo = v
            pos[i] = 1.0 if holding else 0.0
        return pos

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        if bool(p["lookahead"]):
            return self._signals_lookahead(df)

        sm = self._smoothed_slope(df["close"], causal=True)
        vol_win = max(int(p["vol_win"]), 2)
        scale = sm.rolling(vol_win, min_periods=vol_win // 2 + 1).std().to_numpy()

        mode = str(p.get("mode", "turn")).lower()
        pos = self._signals_zigzag(sm, scale) if mode == "zigzag" \
            else self._signals_turn(sm, scale)
        return pd.Series(pos, index=df.index, name="signal")
