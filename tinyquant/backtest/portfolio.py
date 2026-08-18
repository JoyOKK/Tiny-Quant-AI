"""组合级（多标的截面）回测引擎。

现有 `engine.backtest` 是单标的、0/1 仓位的，无法表达"在一篮子股票里
挑最强的持有"这类截面策略。本模块补上组合级回测：

横截面动量（Cross-Sectional Momentum）：
- 每隔 rebalance 个交易日调仓一次；
- 用「过去 lookback 日、跳过最近 skip 日」的收益率给全体标的排序
  （跳过最近一段是为了规避短期反转，即经典的 12-1 动量）；
- 买入动量最强的 top_k 只（等权），可要求动量为正（绝对动量过滤）；
- 与「等权买入持有全部标的」基准对比。

A 股与美股的横截面动量都有较强的历史有效性，是最经典的多因子雏形。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import COMMISSION, INIT_CASH
from .metrics import performance_metrics


@dataclass
class PortfolioResult:
    equity: pd.Series
    benchmark: pd.Series
    weights: pd.DataFrame
    returns: pd.Series
    metrics: dict
    metrics_benchmark: dict

    def summary(self) -> str:
        m, b = self.metrics, self.metrics_benchmark
        rows = [
            ("总收益", m["total_return"], b["total_return"], "pct"),
            ("年化收益", m["annual_return"], b["annual_return"], "pct"),
            ("夏普比率", m["sharpe"], b["sharpe"], "num"),
            ("最大回撤", m["max_drawdown"], b["max_drawdown"], "pct"),
            ("卡玛比率", m["calmar"], b["calmar"], "num"),
        ]
        lines = [f"{'指标':<10}{'策略':>14}{'等权持有':>14}"]
        for name, sv, bv, kind in rows:
            lines.append(f"{name:<12}{_fmt(sv, kind):>12}{_fmt(bv, kind):>14}")
        return "\n".join(lines)


def _fmt(v, kind):
    if v != v:
        return "-"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    return f"{v:.2f}"


def build_close_panel(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把 {symbol: OHLCV DataFrame} 合并成 close 面板（列=标的，按日期对齐）。"""
    cols = {}
    for sym, df in prices.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        cols[sym] = df["close"]
    if not cols:
        raise ValueError("没有可用的行情数据")
    panel = pd.DataFrame(cols).sort_index()
    return panel.ffill()


def cross_sectional_momentum(
    prices: dict[str, pd.DataFrame],
    lookback: int = 120,
    skip: int = 20,
    top_k: int = 2,
    rebalance: int = 20,
    long_only_positive: bool = True,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION,
) -> PortfolioResult:
    panel = build_close_panel(prices)
    rets = panel.pct_change().fillna(0.0)
    n, m = panel.shape
    top_k = min(top_k, m)

    # 动量得分：过去 lookback 日、跳过最近 skip 日的收益率
    mom = panel.shift(skip) / panel.shift(skip + lookback) - 1.0

    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    dates = panel.index
    cur_w = pd.Series(0.0, index=panel.columns)
    for i, d in enumerate(dates):
        if i % rebalance == 0:
            score = mom.iloc[i]
            valid = score.dropna()
            if long_only_positive:
                valid = valid[valid > 0]
            if len(valid) > 0:
                chosen = valid.sort_values(ascending=False).head(top_k).index
                cur_w = pd.Series(0.0, index=panel.columns)
                cur_w[chosen] = 1.0 / len(chosen)
            else:
                cur_w = pd.Series(0.0, index=panel.columns)
        weights.iloc[i] = cur_w.to_numpy()

    # 次日执行，避免未来函数
    pos = weights.shift(1).fillna(0.0)
    gross_ret = (pos * rets).sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1).fillna(pos.abs().sum(axis=1))
    cost = turnover * commission
    strat_ret = gross_ret - cost
    equity = (1 + strat_ret).cumprod() * init_cash

    bench_ret = rets.mean(axis=1)  # 等权持有全部标的
    benchmark = (1 + bench_ret).cumprod() * init_cash

    exposure = pos.sum(axis=1)
    metrics = performance_metrics(equity, strat_ret, exposure)
    metrics_bench = performance_metrics(
        benchmark, bench_ret, pd.Series(1.0, index=panel.index)
    )
    return PortfolioResult(
        equity=equity,
        benchmark=benchmark,
        weights=weights,
        returns=strat_ret,
        metrics=metrics,
        metrics_benchmark=metrics_bench,
    )
