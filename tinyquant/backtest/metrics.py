"""绩效指标计算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_metrics(
    equity: pd.Series,
    returns: pd.Series,
    position: pd.Series,
    bench_returns: pd.Series | None = None,
    commission: float = 0.0,
) -> dict:
    """根据资金曲线 / 每日收益 / 仓位序列计算常用绩效指标。

    Args:
        equity: 资金曲线。
        returns: 策略每日收益（已扣成本）。
        position: 每日仓位序列。
        bench_returns: 基准每日收益，提供后才计算 Beta / Alpha / 信息比率。
        commission: 单边手续费率，用于估算总交易成本。
    """
    equity = equity.dropna()
    returns = returns.fillna(0)
    if len(equity) < 2:
        return {}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n = len(equity)
    annual_return = (1 + total_return) ** (TRADING_DAYS / n) - 1

    std = returns.std()
    sharpe = np.sqrt(TRADING_DAYS) * returns.mean() / std if std > 0 else 0.0
    annual_volatility = std * np.sqrt(TRADING_DAYS)

    downside = returns[returns < 0].std()
    sortino = np.sqrt(TRADING_DAYS) * returns.mean() / downside if downside > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_drawdown = drawdown.min()
    max_dd_duration = _max_drawdown_duration(equity)

    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # 以"持仓段"统计每笔交易的收益与持仓天数
    trades = _extract_trades(equity, position)
    rets = [t["ret"] for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    win_rate = len(wins) / len(rets) if rets else 0.0

    gross_profit = float(sum(wins))
    gross_loss = float(-sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    best_trade = float(max(rets)) if rets else 0.0
    worst_trade = float(min(rets)) if rets else 0.0
    avg_holding_days = float(np.mean([t["days"] for t in trades])) if trades else 0.0

    pos = position.reindex(equity.index).fillna(0)
    exposure = float((pos > 0).mean())
    turnover = float(pos.diff().abs().fillna(pos.abs()).sum())
    total_cost = turnover * commission

    out = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "max_dd_duration": max_dd_duration,
        "calmar": calmar,
        "win_rate": win_rate,
        "num_trades": len(trades),
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_holding_days": avg_holding_days,
        "exposure": exposure,
        "turnover": turnover,
        "total_cost": total_cost,
    }
    out.update(_relative_metrics(returns, bench_returns))
    return out


def _relative_metrics(returns: pd.Series, bench_returns: pd.Series | None) -> dict:
    """相对基准的 Beta / 年化 Alpha / 信息比率。缺基准时返回空。"""
    if bench_returns is None:
        return {}
    b = bench_returns.reindex(returns.index).fillna(0)
    var_b = float(b.var())
    if var_b <= 0:
        return {"beta": float("nan"), "alpha": float("nan"), "info_ratio": float("nan")}
    beta = float(np.cov(returns, b)[0, 1] / var_b)
    alpha_daily = float(returns.mean() - beta * b.mean())
    alpha_annual = (1 + alpha_daily) ** TRADING_DAYS - 1
    active = returns - b
    act_std = float(active.std())
    info_ratio = np.sqrt(TRADING_DAYS) * float(active.mean()) / act_std if act_std > 0 else 0.0
    return {"beta": beta, "alpha": alpha_annual, "info_ratio": info_ratio}


def _max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤持续期（交易日）：从创新高到再次收复失地的最长间隔。"""
    roll_max = equity.cummax()
    in_dd = equity < roll_max
    longest = cur = 0
    for flag in in_dd:
        cur = cur + 1 if flag else 0
        longest = max(longest, cur)
    return int(longest)


def _extract_trades(equity: pd.Series, position: pd.Series) -> list[dict]:
    """把连续持仓区间切成一笔笔交易，返回每笔的收益率与持仓天数。"""
    pos = position.reindex(equity.index).fillna(0)
    trades: list[dict] = []
    in_pos = False
    entry_val = None
    entry_i = 0
    for i, t in enumerate(equity.index):
        holding = pos.loc[t] > 0
        if holding and not in_pos:
            in_pos = True
            entry_val = equity.loc[t]
            entry_i = i
        elif not holding and in_pos:
            in_pos = False
            if entry_val:
                trades.append({"ret": equity.loc[t] / entry_val - 1, "days": i - entry_i})
    if in_pos and entry_val:
        trades.append({"ret": equity.iloc[-1] / entry_val - 1, "days": len(equity) - 1 - entry_i})
    return trades
