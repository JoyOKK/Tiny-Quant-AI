"""模拟盘经纪商：用真实行情 + 虚拟资金跟踪策略每日表现。

用法：
  1. init  绑定策略与股票池，设定初始资金；
  2. run   每天（或任意时刻）执行一次：拉取最新行情 -> 跑策略 -> 调仓；
  3. status 查看持仓、市值、累计收益与资金曲线。
账户状态持久化到 config.PAPER_ACCOUNT_PATH（JSON），可长期跟踪。
"""
from __future__ import annotations

import datetime as dt
import json
import math

from config import COMMISSION, INIT_CASH, PAPER_ACCOUNT_PATH
from ..data.factory import get_data_source, load_history
from ..strategies import get_strategy


class PaperBroker:
    def __init__(self, path=PAPER_ACCOUNT_PATH):
        self.path = path
        self.acc = self._load()

    # ---------- 持久化 ----------
    def _load(self) -> dict | None:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return None

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.acc, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- 初始化 ----------
    def init_account(
        self,
        strategy: str,
        symbols: list[str],
        source: str,
        init_cash: float = INIT_CASH,
        strategy_params: dict | None = None,
        lot_size: int | None = None,
    ) -> dict:
        self.acc = {
            "created_at": _now(),
            "strategy": strategy,
            "strategy_params": strategy_params or {},
            "symbols": symbols,
            "source": source,
            "lot_size": lot_size if lot_size is not None else (100 if source == "akshare" else 1),
            "init_cash": init_cash,
            "cash": init_cash,
            "positions": {},   # symbol -> {"shares", "cost"}
            "trades": [],
            "equity_curve": [],
        }
        self._save()
        return self.acc

    def _require(self):
        if not self.acc:
            raise RuntimeError("模拟盘账户未初始化，请先运行 `paper init`。")

    # ---------- 运行一次调仓 ----------
    def run_once(self, verbose: bool = True) -> dict:
        self._require()
        acc = self.acc
        source = acc["source"]
        lot = acc.get("lot_size", 1)
        ds = get_data_source(source)
        strat = get_strategy(acc["strategy"], **acc.get("strategy_params", {}))

        # 1) 取最新价 + 目标信号
        prices, signals = {}, {}
        start = (dt.date.today() - dt.timedelta(days=730)).strftime("%Y-%m-%d")
        for sym in acc["symbols"]:
            df = load_history(sym, start=start, source=source)
            if df.empty:
                continue
            sig = strat.run(df)
            signals[sym] = float(sig.iloc[-1]) if len(sig) else 0.0
            try:
                rt = ds.get_realtime(sym)
                prices[sym] = rt.get("price") or float(df["close"].iloc[-1])
            except Exception:
                prices[sym] = float(df["close"].iloc[-1])

        # 2) 目标权重 = 策略信号（0~1 仓位）。多标的权重之和超过 1 时按比例缩小。
        total_equity = acc["cash"] + sum(
            p["shares"] * prices.get(s, p["cost"]) for s, p in acc["positions"].items()
        )
        weights = {
            s: max(float(signals.get(s, 0)), 0.0)
            for s in acc["symbols"] if s in prices
        }
        wsum = sum(weights.values())
        if wsum > 1.0:
            weights = {s: w / wsum for s, w in weights.items()}

        # 3) 生成目标持仓并撮合
        logs = []
        for sym in acc["symbols"]:
            price = prices.get(sym)
            if price is None or price <= 0:
                continue
            cur_shares = acc["positions"].get(sym, {}).get("shares", 0)
            w = weights.get(sym, 0.0)
            tgt_shares = int(math.floor(total_equity * w / price / lot) * lot) if w > 0 else 0
            diff = tgt_shares - cur_shares
            if diff == 0:
                continue
            action = "BUY" if diff > 0 else "SELL"
            amount = abs(diff) * price
            fee = amount * COMMISSION
            if action == "BUY" and acc["cash"] < amount + fee:
                # 现金不足，按可用现金买入整手
                affordable = int(math.floor(acc["cash"] / (price * (1 + COMMISSION)) / lot) * lot)
                diff = affordable
                if diff <= 0:
                    continue
                amount = diff * price
                fee = amount * COMMISSION
            self._apply_trade(sym, action, diff, price, fee)
            logs.append(f"{action} {sym} x{abs(diff)} @ {price:.3f} 费用 {fee:.2f}")

        # 4) 记录资金曲线
        market_value = sum(
            p["shares"] * prices.get(s, p["cost"]) for s, p in acc["positions"].items()
        )
        equity = acc["cash"] + market_value
        acc["equity_curve"].append(
            {"time": _now(), "equity": round(equity, 2),
             "cash": round(acc["cash"], 2), "market_value": round(market_value, 2)}
        )
        self._save()

        if verbose:
            print("信号:", {s: signals.get(s, 0) for s in acc["symbols"]})
            for l in logs:
                print(" ", l)
            if not logs:
                print("  无需调仓")
        return {"equity": equity, "signals": signals, "prices": prices, "logs": logs}

    def _apply_trade(self, sym, action, diff, price, fee):
        acc = self.acc
        pos = acc["positions"].get(sym, {"shares": 0, "cost": 0.0})
        if action == "BUY":
            new_shares = pos["shares"] + diff
            pos["cost"] = (pos["shares"] * pos["cost"] + diff * price) / new_shares
            pos["shares"] = new_shares
            acc["cash"] -= diff * price + fee
        else:  # SELL, diff<0
            pos["shares"] += diff  # diff 为负
            acc["cash"] += (-diff) * price - fee
            if pos["shares"] <= 0:
                pos = None
        if pos and pos["shares"] > 0:
            acc["positions"][sym] = pos
        else:
            acc["positions"].pop(sym, None)
        acc["trades"].append(
            {"time": _now(), "symbol": sym, "action": action,
             "shares": abs(diff), "price": round(price, 3), "fee": round(fee, 2)}
        )

    # ---------- 状态 ----------
    def status(self) -> dict:
        self._require()
        acc = self.acc
        ds = get_data_source(acc["source"])
        rows, market_value = [], 0.0
        for sym, pos in acc["positions"].items():
            try:
                price = ds.get_realtime(sym).get("price") or pos["cost"]
            except Exception:
                price = pos["cost"]
            val = pos["shares"] * price
            market_value += val
            pnl = (price - pos["cost"]) * pos["shares"]
            rows.append({
                "symbol": sym, "shares": pos["shares"], "cost": round(pos["cost"], 3),
                "price": round(price, 3), "value": round(val, 2), "pnl": round(pnl, 2),
            })
        equity = acc["cash"] + market_value
        total_return = equity / acc["init_cash"] - 1
        return {
            "equity": round(equity, 2),
            "cash": round(acc["cash"], 2),
            "market_value": round(market_value, 2),
            "total_return": total_return,
            "positions": rows,
            "strategy": acc["strategy"],
            "symbols": acc["symbols"],
        }


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
