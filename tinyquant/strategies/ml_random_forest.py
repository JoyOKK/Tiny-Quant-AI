"""随机森林 ML 策略（AI 量化策略示例）。

思路：用技术指标作为特征，预测"下一根 K 线是否上涨"。
为避免未来函数，用前 train_ratio 段数据训练，仅在其后的样本上产生交易信号。
预测上涨概率 > buy_threshold 则满仓，否则空仓。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register

_FEATURES = [
    "ma5", "ma10", "ma20", "ema12", "ema26",
    "rsi14", "mom10", "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j", "atr14",
]


@register("ml_rf")
class RandomForestStrategy(Strategy):
    description = "随机森林预测次日涨跌，概率超阈值则做多（AI 策略示例）"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "train_ratio": 0.6,   # 训练集占比
            "buy_threshold": 0.55,  # 上涨概率买入阈值
            "n_estimators": 200,
            "max_depth": 5,
            "random_state": 42,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        from sklearn.ensemble import RandomForestClassifier

        feats = [f for f in _FEATURES if f in df.columns]
        data = df.copy()
        # 标签：次日收盘higher则为1（用 shift(-1) 只在训练时使用，不泄露给预测）
        data["label"] = (data["close"].shift(-1) > data["close"]).astype(int)
        data = data.dropna(subset=feats + ["label"])
        if len(data) < 60:
            return pd.Series(0.0, index=df.index)

        split = int(len(data) * float(self.params["train_ratio"]))
        train = data.iloc[:split]
        test = data.iloc[split:]
        if train.empty or test.empty:
            return pd.Series(0.0, index=df.index)

        model = RandomForestClassifier(
            n_estimators=int(self.params["n_estimators"]),
            max_depth=int(self.params["max_depth"]),
            random_state=int(self.params["random_state"]),
            n_jobs=-1,
        )
        model.fit(train[feats], train["label"])
        proba = model.predict_proba(test[feats])[:, 1]

        signal = pd.Series(0.0, index=df.index)
        buy = proba > float(self.params["buy_threshold"])
        signal.loc[test.index] = np.where(buy, 1.0, 0.0)
        return signal
