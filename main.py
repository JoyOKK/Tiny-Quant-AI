"""Tiny-Quant-AI 命令行入口。

示例：
  python main.py strategies                              # 列出可用策略
  python main.py quote 000001                            # 查看实时行情
  python main.py indicators 000001 --start 2024-01-01    # 查看最新技术指标
  python main.py backtest 000001 --strategy ma_cross --start 2023-01-01
  python main.py backtest AAPL --source yfinance --strategy ml_rf --start 2022-01-01
  python main.py paper init --strategy ma_cross --symbols 000001,600519
  python main.py paper run                               # 每日跑一次调仓
  python main.py paper status                            # 查看模拟盘收益
  python main.py dashboard                               # 启动交互式网页看板
"""
from __future__ import annotations

import argparse
import datetime as dt

from config import DEFAULT_SOURCE, INIT_CASH
from tinyquant.data.factory import get_data_source, load_history
from tinyquant.indicators import add_indicators
from tinyquant.strategies import get_strategy, strategy_info
from tinyquant.backtest import backtest
from tinyquant.trading import PaperBroker


def _today():
    return dt.date.today().strftime("%Y-%m-%d")


def cmd_strategies(args):
    print("可用策略：")
    for info in strategy_info():
        print(f"  - {info['name']}: {info['description']}")
        if info["params"]:
            print(f"      默认参数: {info['params']}")


def cmd_quote(args):
    ds = get_data_source(args.source)
    rt = ds.get_realtime(args.symbol)
    print(f"[{rt['time']}] {rt['symbol']} 最新价 {rt['price']}")
    for k in ("open", "high", "low", "pre_close", "volume"):
        if rt.get(k) is not None:
            print(f"  {k}: {rt[k]}")


def cmd_indicators(args):
    df = load_history(args.symbol, start=args.start, end=args.end,
                      source=args.source)
    if df.empty:
        print("未获取到数据")
        return
    ind = add_indicators(df)
    print(f"{args.symbol} 最近一根 K 线的技术指标：")
    last = ind.iloc[-1]
    for col in ind.columns:
        print(f"  {col:<12}: {last[col]:.4f}" if last[col] == last[col] else f"  {col:<12}: -")


def _parse_params(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        try:
            out[k] = float(v) if "." in v else int(v)
        except ValueError:
            out[k] = v
    return out


def cmd_backtest(args):
    params = _parse_params(args.param)
    results = []
    for symbol in args.symbol.split(","):
        symbol = symbol.strip()
        try:
            df = load_history(symbol, start=args.start, end=args.end, source=args.source)
        except Exception as e:  # noqa: BLE001
            print(f"{symbol}: 获取行情失败（{type(e).__name__}: {e}），请检查网络/代理后重试")
            continue
        if df.empty:
            print(f"{symbol}: 未获取到数据，跳过")
            continue
        strat = get_strategy(args.strategy, **params)
        signals = strat.run(df)
        res = backtest(df, signals, init_cash=args.cash)
        print(f"\n===== {symbol} | 策略 {args.strategy} | {df.index.min().date()} ~ {df.index.max().date()} =====")
        print(res.summary())
        results.append((symbol, res))
        if args.plot:
            _plot(symbol, res)
    return results


def _plot(symbol, res):
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        res.equity.plot(ax=ax, label="策略")
        res.benchmark.plot(ax=ax, label="买入持有")
        ax.set_title(f"{symbol} 资金曲线")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"equity_{symbol}.png", dpi=120)
        print(f"  资金曲线已保存: equity_{symbol}.png")
    except Exception as e:
        print(f"  绘图失败: {e}")


def cmd_dashboard(args):
    import subprocess
    import sys
    from config import ROOT_DIR

    script = ROOT_DIR / "dashboard.py"
    print(f"启动 Streamlit 看板：http://localhost:{args.port} （Ctrl+C 退出）")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script),
         "--server.port", str(args.port)],
        check=False,
    )


def cmd_paper(args):
    broker = PaperBroker()
    if args.paper_cmd == "init":
        params = _parse_params(args.param)
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        broker.init_account(
            strategy=args.strategy, symbols=symbols, source=args.source,
            init_cash=args.cash, strategy_params=params,
        )
        print(f"模拟盘已初始化：策略={args.strategy} 股票池={symbols} 初始资金={args.cash}")
    elif args.paper_cmd == "run":
        broker.run_once()
    elif args.paper_cmd == "status":
        st = broker.status()
        print(f"策略: {st['strategy']} | 股票池: {st['symbols']}")
        print(f"总权益: {st['equity']}  现金: {st['cash']}  持仓市值: {st['market_value']}")
        print(f"累计收益率: {st['total_return'] * 100:.2f}%")
        if st["positions"]:
            print("持仓：")
            for r in st["positions"]:
                print(f"  {r['symbol']}: {r['shares']}股 成本 {r['cost']} 现价 {r['price']} "
                      f"市值 {r['value']} 浮盈 {r['pnl']}")
        else:
            print("当前空仓")


def build_parser():
    p = argparse.ArgumentParser(description="Tiny-Quant-AI 最小量化交易平台")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("strategies", help="列出可用策略")
    sp.set_defaults(func=cmd_strategies)

    sp = sub.add_parser("quote", help="实时行情")
    sp.add_argument("symbol")
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.set_defaults(func=cmd_quote)

    sp = sub.add_parser("indicators", help="技术指标")
    sp.add_argument("symbol")
    sp.add_argument("--start", default="2024-01-01")
    sp.add_argument("--end", default=None)
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.set_defaults(func=cmd_indicators)

    sp = sub.add_parser("backtest", help="回测")
    sp.add_argument("symbol", help="单只或多只(逗号分隔)")
    sp.add_argument("--strategy", default="ma_cross")
    sp.add_argument("--start", default="2023-01-01")
    sp.add_argument("--end", default=None)
    sp.add_argument("--source", default=DEFAULT_SOURCE)
    sp.add_argument("--cash", type=float, default=INIT_CASH)
    sp.add_argument("--param", nargs="*", help="策略参数，如 fast=5 slow=20")
    sp.add_argument("--plot", action="store_true")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("dashboard", help="启动交互式网页看板")
    sp.add_argument("--port", type=int, default=8501)
    sp.set_defaults(func=cmd_dashboard)

    sp = sub.add_parser("paper", help="模拟盘交易")
    psub = sp.add_subparsers(dest="paper_cmd", required=True)
    pi = psub.add_parser("init")
    pi.add_argument("--strategy", default="ma_cross")
    pi.add_argument("--symbols", required=True, help="逗号分隔的股票池")
    pi.add_argument("--source", default=DEFAULT_SOURCE)
    pi.add_argument("--cash", type=float, default=INIT_CASH)
    pi.add_argument("--param", nargs="*")
    psub.add_parser("run")
    psub.add_parser("status")
    sp.set_defaults(func=cmd_paper)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
