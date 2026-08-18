"""深 V 分批加仓：首仓只试探两手，若不是真谷底、价格继续破新低则加仓。

当天 K 线无法确认是否已是谷底，因此：
- 出现与 `deep_v` 相同的坑底信号时，先按「两手」试探建仓（不是一次满仓）；
- 此后 `add_window` 个交易日内，若收盘跌破上一笔买入日的最低价、
  且相对上一笔成交价再跌至少 `add_step`，再加两手，层层下移；
- 卖出仍用 `deep_v` 的反弹衰竭规则（缩量滞涨 / 缩量十字星 / 超时）。
  加仓窗口内不按首仓价止损（继续破新低是加仓而不是离场）；
  窗口结束或已加满后，止损改为相对持仓均价。

仓位输出为 [0, 1] 的资金占比：每层 = 两手市值 / 账户资金，上限 1。
回测与模拟盘按该权重执行；K 线图上每次加仓都会标一个买点。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INIT_CASH
from .deep_v import DeepVReversal
from .registry import register

_EPS = 1e-12


@register("deep_v_scale")
class DeepVScaleIn(DeepVReversal):
    description = "深V分批加仓：试探两手，破前低再加两手，卖出规则同深V"

    @classmethod
    def default_params(cls) -> dict:
        params = dict(DeepVReversal.default_params())
        params.update(
            {
                "add_window": 5,       # 首仓后允许加仓的交易日（默认一周）
                "add_step": 0.02,      # 相对上一笔成交价至少再跌 2% 才加仓
                "max_layers": 5,       # 最多几层（含首仓）
                "lots_per_add": 2,     # 每层手数
                "lot_size": 100,       # A股 1 手 = 100 股
                "init_cash": INIT_CASH,
            }
        )
        return params

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        f = self._build_features(df)
        add_window = int(self.params["add_window"])
        add_step = float(self.params["add_step"])
        max_layers = max(int(self.params["max_layers"]), 1)
        lots_per_add = max(int(self.params["lots_per_add"]), 1)
        lot_size = max(int(self.params["lot_size"]), 1)
        init_cash = float(self.params["init_cash"])
        shares_per_layer = lots_per_add * lot_size

        n = f["n"]
        close, high, low = f["close"], f["high"], f["low"]
        pos = np.zeros(n)
        layers = 0
        weight = 0.0
        entry_i = -1
        avg_cost = np.nan
        last_fill = np.nan
        last_low = np.nan

        for i in range(n):
            if layers == 0:
                if self._is_buy(
                    i, f["lookback"], f["min_down_days"], f["min_drop"],
                    f["accel_ratio"], f["near_pit"],
                    close, high, low,
                    f["is_reversal"], f["is_doji"], f["is_big_bear"],
                ):
                    layers, weight, avg_cost = self._add_layer(
                        layers, weight, avg_cost, close[i],
                        shares_per_layer, init_cash,
                    )
                    entry_i = i
                    last_fill = close[i]
                    last_low = low[i]
            else:
                hold_days = i - entry_i
                # 加仓期内继续破新低是加仓，不是按首仓价止损离场
                still_scaling = hold_days < add_window and layers < max_layers
                effective_stop = 999.0 if still_scaling else f["stop_loss"]
                if self._is_sell(
                    i, entry_i, avg_cost, hold_days,
                    f["lookback"], f["min_up_days"], f["min_rally"],
                    effective_stop, f["max_hold"],
                    f["vol_shrink"], f["has_vol"],
                    close, high, f["volume"], f["vol_ma"],
                    f["is_stalling"], f["is_doji"], f["is_shooting"],
                ):
                    layers = 0
                    weight = 0.0
                    entry_i = -1
                    avg_cost = last_fill = last_low = np.nan
                elif still_scaling and self._should_add(
                    close[i], low[i], last_fill, last_low, add_step,
                ):
                    layers, weight, avg_cost = self._add_layer(
                        layers, weight, avg_cost, close[i],
                        shares_per_layer, init_cash,
                    )
                    last_fill = close[i]
                    last_low = min(last_low, low[i])
            pos[i] = weight if layers > 0 else 0.0

        return pd.Series(pos, index=df.index, name="signal")

    @staticmethod
    def _should_add(
        close_i: float,
        low_i: float,
        last_fill: float,
        last_low: float,
        add_step: float,
    ) -> bool:
        """收盘跌破上一笔买入日最低价，且相对上一笔成交价再跌至少 add_step。"""
        if last_fill <= 0 or last_low <= 0:
            return False
        broke_low = close_i < last_low - _EPS
        stepped = close_i <= last_fill * (1.0 - add_step)
        return bool(broke_low and stepped)

    @staticmethod
    def _add_layer(
        layers: int,
        weight: float,
        avg_cost: float,
        fill: float,
        shares_per_layer: int,
        init_cash: float,
    ) -> tuple[int, float, float]:
        add_w = shares_per_layer * fill / max(init_cash, _EPS)
        new_weight = min(1.0, weight + add_w)
        if np.isnan(avg_cost):
            new_avg = fill
        else:
            new_avg = (avg_cost * layers + fill) / (layers + 1)
        return layers + 1, new_weight, new_avg
