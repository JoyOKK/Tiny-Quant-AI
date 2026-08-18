"""向量化回测引擎。

模型很简单但足够反映策略优劣：
- 策略在第 t 日收盘产生目标仓位 signal[t]；
- 为避免未来函数，实际持仓 position[t] = signal[t-1]（次日按收盘价执行）；
- 每次仓位变动按 commission 收取手续费。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import COMMISSION, INIT_CASH
from .metrics import performance_metrics


@dataclass
class BacktestResult:
    equity: pd.Series          # 策略资金曲线
    benchmark: pd.Series       # 买入持有基准资金曲线
    position: pd.Series        # 每日仓位（0/1）
    returns: pd.Series         # 策略每日收益
    metrics: dict              # 绩效指标
    metrics_benchmark: dict    # 基准绩效指标

    def summary(self) -> str:
        m, b = self.metrics, self.metrics_benchmark
        rows = [
            ("总收益", m["total_return"], b["total_return"], "pct"),
            ("年化收益", m["annual_return"], b["annual_return"], "pct"),
            ("夏普比率", m["sharpe"], b["sharpe"], "num"),
            ("最大回撤", m["max_drawdown"], b["max_drawdown"], "pct"),
            ("卡玛比率", m["calmar"], b["calmar"], "num"),
            ("胜率", m["win_rate"], float("nan"), "pct"),
            ("交易次数", m["num_trades"], float("nan"), "int"),
        ]
        lines = [f"{'指标':<10}{'策略':>14}{'买入持有':>14}"]
        for name, sv, bv, kind in rows:
            lines.append(f"{name:<12}{_fmt(sv, kind):>12}{_fmt(bv, kind):>14}")
        return "\n".join(lines)


def _fmt(v, kind):
    if v != v:  # NaN
        return "-"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "int":
        return f"{int(v)}"
    return f"{v:.2f}"


def backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    init_cash: float = INIT_CASH,
    commission: float = COMMISSION,
) -> BacktestResult:
    df = df.dropna(subset=["close"])
    signals = signals.reindex(df.index).fillna(0).clip(0, 1)

    # 次日执行，避免未来函数
    position = signals.shift(1).fillna(0)
    asset_ret = df["close"].pct_change().fillna(0)

    # 仓位变动 -> 手续费（按换手比例）
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * commission

    strat_ret = position * asset_ret - cost
    equity = (1 + strat_ret).cumprod() * init_cash

    bench_ret = asset_ret
    benchmark = (1 + bench_ret).cumprod() * init_cash

    metrics = performance_metrics(equity, strat_ret, position)
    metrics_bench = performance_metrics(benchmark, bench_ret, pd.Series(1.0, index=df.index))

    return BacktestResult(
        equity=equity,
        benchmark=benchmark,
        position=position,
        returns=strat_ret,
        metrics=metrics,
        metrics_benchmark=metrics_bench,
    )
