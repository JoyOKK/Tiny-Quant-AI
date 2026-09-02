"""MA斜率(平滑)曲线的波峰波谷波段策略（严格因果，无未来函数）。

思路
----
用 k 日均线平滑价格得到均线曲线，对其一阶导（斜率）做**因果** Savitzky-Golay
滤波得到「MA斜率(平滑·因果)」曲线；再据这条斜率曲线自身的谷/峰买卖：

- 斜率曲线走到**谷底**（掉头回升确认）→ 买入；
- 斜率曲线走到**峰顶**（掉头回落确认）→ 卖出。

买点时点
--------
拐点最早只能在「波谷后第一根」（曲线由降转升的当天）因果确认。回测引擎
``position[t] = signal[t-1]``：确认日收盘决策，从确认日收盘价开始持有，
第一段盈亏是「确认日收盘 → 次日收盘」。图上 B/S 标在持仓生效日（谷/峰后 2 天），
成交价经济含义是确认日收盘（谷/峰后 1 天）。不把信号再提前 1 根，否则会把
确认日当天已经走完的涨跌算进收益，实盘买不到那个价。

假信号过滤
----------
因果平滑曲线上的短抖动会被「掉头即交易」当成完整波段，出现 1～3 天的来回单。
默认用三道闸门只做相对长期、幅度够大的趋势拐点：

- min_run：上一趋势至少持续这么多根，才承认这次掉头（滤掉两三天的毛刺）；
- min_retrace：从本段极值反向走了至少 min_retrace×斜率日变化std，才确认拐点
  （真 V 形第一天往往已够大，浅抖动会一直达不到阈值）；
- min_hold：成交后至少持有/空仓这么多根，才允许反向（挡住短线来回）。

两种确认方式（mode）
--------------------
- mode="turn"（默认）：掉头 + 上述闸门，买点对齐波谷后一天。
- mode="zigzag"：等斜率从极值反向移动 rev×局部std 才确认，更稳、滞后更大。

严格因果（无未来函数）
----------------------
平滑用因果(trailing) SavGol，波动阈值用尾部滚动 std，拐点用「只看当日及以前」的
在线状态机确认。lookahead=True 仅对照（中心式 SavGol + find_peaks，用到未来）。
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
    description = "MA斜率(平滑)波段：斜率谷底买、峰顶卖（严格因果；滤短抖动）"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "ma_window": 5,      # k 日均线周期
            "sg_window": 11,     # 斜率保形滤波窗口(与看板 MA斜率(平滑·因果) 一致)
            "sg_poly": 3,        # 多项式阶数(< 窗口)
            "mode": "turn",      # turn=低滞后(掉头即确认) / zigzag=稳健(等反转)
            "eps_k": 0.0,        # [turn]死区＝eps_k×局部std；0=纯反转第一天
            "persist": 1,        # [turn]新方向连续确认根数；1=反转第一天即可
            "min_run": 5,        # 上一趋势至少持续根数，不足则视为短抖动、不交易
            "min_retrace": 0.8,  # 从本段极值反向至少 min_retrace×斜率日变化std 才确认（真 V 常在第一天达标）
            "min_hold": 4,       # 成交后至少隔这么多根才允许反向，挡住 1～3 天来回
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
        min_run = max(int(p.get("min_run", 1)), 1)
        min_retrace = max(float(p.get("min_retrace", 0.0)), 0.0)
        min_hold = max(int(p.get("min_hold", 1)), 1)
        vol_win = max(int(p.get("vol_win", 20)), 2)
        cv = sm.to_numpy(dtype=float)
        # 回撤阈值按「斜率日变化」的波动，避免用水平 std（趋势越长阈值越大、买点被拖晚）
        dscale = sm.diff().rolling(vol_win, min_periods=vol_win // 2 + 1).std().to_numpy()
        n = len(cv)
        pos = np.zeros(n)
        holding = False
        last_dir = 0          # 斜率曲线上一个「已确立」的涨跌方向：+1 升 / -1 降
        last_run = 0          # 该确立方向已持续根数
        run_dir = 0           # 当前连续方向
        run_len = 0           # 当前方向已持续根数
        ext_hi = ext_lo = np.nan
        since_trade = 10**9   # 距上一笔成交的根数（首笔不受 min_hold 卡住）
        prev = np.nan
        for i in range(n):
            v, sc, dsc = cv[i], scale[i], dscale[i]
            bad_eps = eps_k > 0 and (np.isnan(sc) or sc <= 0)
            if np.isnan(v) or np.isnan(prev) or bad_eps:
                if not np.isnan(v):
                    prev = v
                    if last_dir != 0:
                        last_run += 1
                        ext_hi = v if np.isnan(ext_hi) else max(ext_hi, v)
                        ext_lo = v if np.isnan(ext_lo) else min(ext_lo, v)
                since_trade += 1
                pos[i] = 1.0 if holding else 0.0
                continue
            dead = eps_k * sc if eps_k > 0 else 0.0
            dv = v - prev
            d = 1 if dv > dead else (-1 if dv < -dead else 0)
            if d != 0 and d == run_dir:
                run_len += 1
            elif d != 0:
                run_dir, run_len = d, 1
            if last_dir != 0:
                last_run += 1
                ext_hi = v if np.isnan(ext_hi) else max(ext_hi, v)
                ext_lo = v if np.isnan(ext_lo) else min(ext_lo, v)
            # 新方向连续 persist 根、且相对上一确立方向反转时，再过三道闸门
            if run_dir != 0 and run_len >= persist and run_dir != last_dir:
                if last_dir == 0:
                    last_dir, last_run = run_dir, run_len
                    ext_hi = ext_lo = v
                else:
                    retrace = (v - ext_lo) if last_dir < 0 else (ext_hi - v)
                    ok_run = last_run >= min_run
                    thr = (min_retrace * dsc) if (min_retrace > 0 and dsc == dsc and dsc > 0) else 0.0
                    ok_retrace = retrace >= thr
                    ok_hold = since_trade >= min_hold
                    if ok_run and ok_retrace and ok_hold:
                        if run_dir > 0 and last_dir < 0:
                            holding = True
                        elif run_dir < 0 and last_dir > 0:
                            holding = False
                        last_dir, last_run = run_dir, run_len
                        ext_hi = ext_lo = v
                        since_trade = 0
                    # 未过闸门：不改 last_dir，短抖动被忽略，主趋势继续
            prev = v
            since_trade += 1
            pos[i] = 1.0 if holding else 0.0
        return pos

    # ---- 稳健：ZigZag 反转确认（严格因果）----
    def _signals_zigzag(self, sm: pd.Series, scale: np.ndarray) -> np.ndarray:
        rev = max(float(self.params["rev"]), 0.0)
        min_run = max(int(self.params.get("min_run", 1)), 1)
        min_hold = max(int(self.params.get("min_hold", 1)), 1)
        smv = sm.to_numpy(dtype=float)
        n = len(smv)
        pos = np.zeros(n)
        holding = False
        mode = 0
        hi = lo = np.nan
        run = 0
        since_trade = 10**9
        for i in range(n):
            v, sc = smv[i], scale[i]
            if np.isnan(v) or np.isnan(sc) or sc <= 0:
                since_trade += 1
                if mode != 0:
                    run += 1
                pos[i] = 1.0 if holding else 0.0
                continue
            if np.isnan(hi):
                hi = lo = v
                pos[i] = 1.0 if holding else 0.0
                continue
            thr = rev * sc
            hi = max(hi, v)
            lo = min(lo, v)
            run += 1
            can_flip = run >= min_run and since_trade >= min_hold
            if can_flip and mode <= 0 and v >= lo + thr:
                holding, mode = True, 1
                hi = lo = v
                run, since_trade = 0, 0
            elif can_flip and mode >= 0 and v <= hi - thr:
                holding, mode = False, -1
                hi = lo = v
                run, since_trade = 0, 0
            since_trade += 1
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
