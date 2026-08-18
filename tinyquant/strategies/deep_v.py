"""深 V 反转策略：加速下跌砸坑后在坑底抄底，反弹衰竭后离场。

买点（坑底）
-----------
观察窗口（默认 5 个交易日）内出现「多天下跌 + 加速砸坑」，且当日处于
窗口最低点附近。坑底优先用更稳健的衰竭反转 K 线判断：

- 主信号：锤子线 / 长下影，或收盘从当日低点明显拉回（深 V 的右拐点）；
- 兜底：前一日大阴线、当日收出十字星（你提出的形态）。

卖点（反弹衰竭）
---------------
买入后窗口内出现多日上涨、且已有一定反弹幅度时离场：

- 主信号：价格靠近持仓高点 + 滞涨 K 线（冲高回落 / 长上影 / 十字）+ 缩量，
  即短线量价背离见顶；
- 兜底：见到缩量十字星；
- 风控：假 V 止损、最长持仓（反弹策略不应久拖）。

信号在收盘确认，回测引擎按「次日收盘成交」执行，无未来函数。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register

_EPS = 1e-12


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / np.maximum(den, _EPS)


@register("deep_v")
class DeepVReversal(Strategy):
    description = "深V反转：加速砸坑后在坑底抄底，反弹缩量滞涨后卖出"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "lookback": 5,          # 观察窗口（交易日），默认约一周
            "min_down_days": 3,     # 窗口内最少下跌天数
            "min_drop": 0.05,       # 砸坑最小跌幅（窗口最高到最低）
            "accel_ratio": 1.0,     # 后半段跌幅/前半段，>=1 为加速；0 关闭
            "near_pit": 0.01,       # 当日/昨日低点距窗口最低点的容忍（1%）
            "close_pos": 0.55,      # 反转K：收盘须在当日振幅上半区
            "body_doji": 0.30,      # 十字星：实体/振幅上限
            "big_bear_body": 0.55,  # 大阴线：实体/振幅下限
            "vol_shrink": 0.75,     # 缩量：成交量 < 均量 × 该值
            "min_up_days": 3,       # 卖出前窗口内最少上涨天数
            "min_rally": 0.03,      # 买入后最少反弹幅度才考虑卖出
            "stop_loss": 0.07,      # 假V：相对买入价继续下跌则止损
            "max_hold": 10,         # 最长持仓（交易日）
        }

    def _build_features(self, df: pd.DataFrame) -> dict:
        """从 OHLCV 抽出买卖点判定所需的 K 线特征，供本策略与分批加仓版复用。"""
        p = self.params
        lookback = int(p["lookback"])
        close_pos_th = float(p["close_pos"])
        body_doji = float(p["body_doji"])
        big_bear_body = float(p["big_bear_body"])

        close = df["close"].to_numpy(dtype=float)
        n = len(close)
        high = (df["high"] if "high" in df.columns else df["close"]).to_numpy(dtype=float)
        low = (df["low"] if "low" in df.columns else df["close"]).to_numpy(dtype=float)
        if "open" in df.columns:
            open_ = df["open"].to_numpy(dtype=float)
        else:
            open_ = np.concatenate([[close[0]], close[:-1]])
        if "volume" in df.columns:
            volume = df["volume"].to_numpy(dtype=float)
            vol_ma = (
                pd.Series(volume).rolling(lookback, min_periods=lookback)
                .mean()
                .shift(1)
                .to_numpy()
            )
            has_vol = True
        else:
            volume = np.full(n, np.nan)
            vol_ma = np.full(n, np.nan)
            has_vol = False

        rng = np.maximum(high - low, _EPS)
        body = np.abs(close - open_)
        body_ratio = _safe_div(body, rng)
        close_pos = _safe_div(close - low, rng)
        lower_shadow = np.minimum(open_, close) - low
        upper_shadow = high - np.maximum(open_, close)
        # 一字板几乎无振幅，不能当成十字星反转
        has_range = _safe_div(high - low, close) >= 0.005

        is_bear = close < open_
        is_doji = has_range & (body_ratio <= body_doji)
        is_big_bear = has_range & is_bear & (body_ratio >= big_bear_body)
        is_hammer = (
            has_range
            & (lower_shadow >= 2.0 * np.maximum(body, _EPS))
            & (close_pos >= 0.5)
            & (upper_shadow <= np.maximum(body, _EPS))
        )
        is_bounce_bar = has_range & (close_pos >= close_pos_th) & ~is_big_bear
        is_reversal = is_hammer | is_bounce_bar
        is_stalling = has_range & (
            is_doji | (close_pos <= 1.0 - close_pos_th) | (upper_shadow >= 0.5 * rng)
        )
        is_shooting = (
            has_range
            & (upper_shadow >= 2.0 * np.maximum(body, _EPS))
            & (close_pos <= 0.5)
        )
        return {
            "lookback": lookback,
            "min_down_days": int(p["min_down_days"]),
            "min_drop": float(p["min_drop"]),
            "accel_ratio": float(p["accel_ratio"]),
            "near_pit": float(p["near_pit"]),
            "vol_shrink": float(p["vol_shrink"]),
            "min_up_days": int(p["min_up_days"]),
            "min_rally": float(p["min_rally"]),
            "stop_loss": float(p["stop_loss"]),
            "max_hold": int(p["max_hold"]),
            "n": n,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "vol_ma": vol_ma,
            "has_vol": has_vol,
            "is_reversal": is_reversal,
            "is_doji": is_doji,
            "is_big_bear": is_big_bear,
            "is_stalling": is_stalling,
            "is_shooting": is_shooting,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f = self._build_features(df)
        pos = np.zeros(f["n"])
        holding = False
        entry_i = -1
        entry_px = np.nan

        for i in range(f["n"]):
            if not holding:
                if self._is_buy(
                    i, f["lookback"], f["min_down_days"], f["min_drop"],
                    f["accel_ratio"], f["near_pit"],
                    f["close"], f["high"], f["low"],
                    f["is_reversal"], f["is_doji"], f["is_big_bear"],
                ):
                    holding = True
                    entry_i = i
                    entry_px = f["close"][i]
            else:
                hold_days = i - entry_i
                if self._is_sell(
                    i, entry_i, entry_px, hold_days,
                    f["lookback"], f["min_up_days"], f["min_rally"],
                    f["stop_loss"], f["max_hold"],
                    f["vol_shrink"], f["has_vol"],
                    f["close"], f["high"], f["volume"], f["vol_ma"],
                    f["is_stalling"], f["is_doji"], f["is_shooting"],
                ):
                    holding = False
                    entry_i = -1
                    entry_px = np.nan
            pos[i] = 1.0 if holding else 0.0

        return pd.Series(pos, index=df.index, name="signal")

    @staticmethod
    def _is_buy(
        i: int,
        lookback: int,
        min_down_days: int,
        min_drop: float,
        accel_ratio: float,
        near_pit: float,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        is_reversal: np.ndarray,
        is_doji: np.ndarray,
        is_big_bear: np.ndarray,
    ) -> bool:
        # 砸坑窗口取「昨天往前 lookback 根」，今日只做坑底确认，避免反转阳线稀释跌势
        if i < lookback:
            return False
        left, right = i - lookback, i - 1
        if left < 1:
            return False

        w_high = high[left : right + 1]
        w_low = low[left : right + 1]
        prev_close = close[left - 1 : right]
        w_close = close[left : right + 1]
        rets = w_close / np.maximum(prev_close, _EPS) - 1.0
        n_down = int(np.sum(rets < 0))
        if n_down < min_down_days:
            return False

        pit_high = float(np.max(w_high))
        pit_low = float(min(np.min(w_low), low[i]))
        if pit_high <= 0:
            return False
        drop = (pit_high - pit_low) / pit_high
        if drop < min_drop:
            return False

        if accel_ratio > 0:
            half = max(len(rets) // 2, 1)
            first = -float(np.minimum(rets[:half], 0).mean())
            second = -float(np.minimum(rets[half:], 0).mean())
            if first <= _EPS:
                if second <= _EPS:
                    return False
            elif second < first * accel_ratio:
                return False

        at_pit = min(low[i], low[i - 1]) <= pit_low * (1.0 + near_pit)
        if not at_pit:
            return False

        # 主信号：坑底衰竭反转（锤子 / 收盘拉回）
        if is_reversal[i]:
            return True
        # 兜底：大阴线 -> 十字星
        return bool(is_big_bear[i - 1] and is_doji[i])

    @staticmethod
    def _is_sell(
        i: int,
        entry_i: int,
        entry_px: float,
        hold_days: int,
        lookback: int,
        min_up_days: int,
        min_rally: float,
        stop_loss: float,
        max_hold: int,
        vol_shrink: float,
        has_vol: bool,
        close: np.ndarray,
        high: np.ndarray,
        volume: np.ndarray,
        vol_ma: np.ndarray,
        is_stalling: np.ndarray,
        is_doji: np.ndarray,
        is_shooting: np.ndarray,
    ) -> bool:
        if hold_days <= 0:
            return False
        if close[i] <= entry_px * (1.0 - stop_loss):
            return True
        if hold_days >= max_hold:
            return True

        rally = close[i] / entry_px - 1.0
        if rally < min_rally:
            return False

        win_start = max(entry_i + 1, i - lookback + 1)
        up_days = int(np.sum(np.diff(close[win_start - 1 : i + 1]) > 0))
        if up_days < min_up_days:
            return False

        shrinking = True
        if has_vol and not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            shrinking = volume[i] <= vol_ma[i] * vol_shrink

        hold_high = float(np.max(high[entry_i : i + 1]))
        near_high = high[i] >= hold_high * 0.98

        # 主信号：靠近持仓高点的缩量滞涨（量价背离见顶）
        if near_high and is_stalling[i] and shrinking:
            return True
        # 射击之星：高位长上影，放量/缩量都视为拒绝
        if near_high and is_shooting[i]:
            return True
        # 兜底：缩量十字星
        return bool(is_doji[i] and shrinking)
