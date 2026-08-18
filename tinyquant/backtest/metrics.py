"""绩效指标计算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def performance_metrics(equity: pd.Series, returns: pd.Series, position: pd.Series) -> dict:
    """根据资金曲线 / 每日收益 / 仓位序列计算常用绩效指标。"""
    equity = equity.dropna()
    returns = returns.fillna(0)
    if len(equity) < 2:
        return {}

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    n = len(equity)
    annual_return = (1 + total_return) ** (TRADING_DAYS / n) - 1

    std = returns.std()
    sharpe = np.sqrt(TRADING_DAYS) * returns.mean() / std if std > 0 else 0.0

    downside = returns[returns < 0].std()
    sortino = np.sqrt(TRADING_DAYS) * returns.mean() / downside if downside > 0 else 0.0

    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_drawdown = drawdown.min()

    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    # 以"持仓段"统计胜率：一次连续持仓视为一笔交易
    trades = _extract_trades(equity, position)
    wins = [t for t in trades if t > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "win_rate": win_rate,
        "num_trades": len(trades),
    }


def _extract_trades(equity: pd.Series, position: pd.Series) -> list[float]:
    """把连续持仓区间切成一笔笔交易，返回每笔收益率列表。"""
    pos = position.reindex(equity.index).fillna(0)
    trades = []
    in_pos = False
    entry_val = None
    for t in equity.index:
        holding = pos.loc[t] > 0
        if holding and not in_pos:
            in_pos = True
            entry_val = equity.loc[t]
        elif not holding and in_pos:
            in_pos = False
            if entry_val:
                trades.append(equity.loc[t] / entry_val - 1)
    if in_pos and entry_val:
        trades.append(equity.iloc[-1] / entry_val - 1)
    return trades
