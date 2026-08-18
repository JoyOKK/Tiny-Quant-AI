"""梯度提升 + Walk-Forward + 三重障碍标签（工业级 ML 量化范式）。

相比示例里的 `ml_rf`，本策略做了三处关键升级：

1. 模型：优先用 LightGBM（表格数据上通常强于随机森林），
   未安装时自动回退到 sklearn 的 GradientBoostingClassifier；

2. Walk-Forward 滚动训练：不再"一次性 6:4 切分"，而是用扩张窗口反复
   "训练→预测下一段"，每 retrain 根 K 线重训一次，更贴近实盘、更抗过拟合；

3. 三重障碍标签（Triple-Barrier，López de Prado《金融机器学习进展》）：
   以 ATR 设定上/下障碍与最长持有期 horizon，标签为"先到止盈(1) 还是
   先到止损/超时(0)"，取代幼稚的"次日涨跌"，标签更贴近真实交易盈亏。

为避免标签泄露，训练集与预测段之间留出 horizon 根 K 线的禁区（embargo）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy
from .registry import register
from ..indicators import atr

_FEATURES = [
    "ma5", "ma10", "ma20", "ema12", "ema26",
    "rsi14", "mom10", "macd_dif", "macd_dea", "macd_hist",
    "kdj_k", "kdj_d", "kdj_j", "atr14",
]


def _make_model(params: dict):
    """优先 LightGBM，缺失则回退 sklearn 梯度提升。"""
    n_estimators = int(params["n_estimators"])
    max_depth = int(params["max_depth"])
    lr = float(params["learning_rate"])
    seed = int(params["random_state"])
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            num_leaves=max(2, 2 ** max_depth),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            random_state=seed,
        )


def _triple_barrier_labels(df: pd.DataFrame, horizon: int, pt: float, sl: float) -> pd.Series:
    """三重障碍标签：在未来 horizon 根内，先触上障碍=1，先触下障碍/超时=0。

    障碍宽度按 ATR 自适应：上 = close*(1+pt*atr/close)，下 = close*(1-sl*atr/close)。
    """
    close = df["close"].to_numpy(dtype=float)
    if {"high", "low"}.issubset(df.columns):
        a = atr(df, 14).to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
    else:
        a = df["close"].pct_change().abs().rolling(14).mean().to_numpy(dtype=float) * close
        high = low = close
    n = len(close)
    labels = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(a[i]) or a[i] <= 0:
            continue
        up = close[i] + pt * a[i]
        dn = close[i] - sl * a[i]
        end = min(i + horizon, n - 1)
        label = 0  # 默认超时按未触发止盈处理
        for j in range(i + 1, end + 1):
            if high[j] >= up:
                label = 1
                break
            if low[j] <= dn:
                label = 0
                break
        labels[i] = label
    return pd.Series(labels, index=df.index)


@register("ml_lgbm")
class GBMTripleBarrierStrategy(Strategy):
    description = "梯度提升(LightGBM)+Walk-Forward+三重障碍标签的 AI 策略"

    @classmethod
    def default_params(cls) -> dict:
        return {
            "horizon": 10,          # 三重障碍最长持有 K 线数
            "pt": 2.0,              # 止盈障碍 = pt * ATR
            "sl": 2.0,              # 止损障碍 = sl * ATR
            "min_train": 250,       # 首次训练所需最小样本
            "retrain": 20,          # 每隔多少根 K 线重训一次
            "buy_threshold": 0.55,  # 上涨概率买入阈值
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42,
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        horizon = int(self.params["horizon"])
        min_train = int(self.params["min_train"])
        retrain = int(self.params["retrain"])
        threshold = float(self.params["buy_threshold"])

        feats = [f for f in _FEATURES if f in df.columns]
        data = df.copy()
        data["label"] = _triple_barrier_labels(
            data, horizon, float(self.params["pt"]), float(self.params["sl"])
        )

        # 特征齐全的样本位置（标签可能因末尾 horizon 而为 NaN，训练时再过滤）
        feat_ok = data[feats].notna().all(axis=1).to_numpy()
        n = len(data)
        signal = pd.Series(0.0, index=df.index)
        if n < min_train + horizon + retrain:
            return signal  # 数据太短，直接空仓

        X = data[feats]
        y = data["label"]
        idx = data.index

        start = min_train
        while start < n:
            end = min(start + retrain, n)
            # 训练集：预测段之前、且标签已确定（留 horizon 禁区防泄露）
            train_hi = start - horizon
            if train_hi <= min_train // 2:
                start = end
                continue
            train_mask = feat_ok[:train_hi] & y.iloc[:train_hi].notna().to_numpy()
            if train_mask.sum() < 50 or y.iloc[:train_hi][train_mask].nunique() < 2:
                start = end
                continue

            model = _make_model(self.params)
            model.fit(X.iloc[:train_hi][train_mask], y.iloc[:train_hi][train_mask].astype(int))

            block = np.arange(start, end)
            block = block[feat_ok[start:end]]
            if len(block) == 0:
                start = end
                continue
            proba = model.predict_proba(X.iloc[block])[:, 1]
            buy = proba > threshold
            signal.loc[idx[block]] = np.where(buy, 1.0, 0.0)
            start = end
        return signal
