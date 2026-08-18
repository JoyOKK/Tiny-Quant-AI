"""Tiny-Quant-AI 交互式看板（Streamlit）。

启动：
    streamlit run dashboard.py
或：
    python main.py dashboard
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import DEFAULT_SOURCE, INIT_CASH
from tinyquant.backtest import backtest
from tinyquant.data.factory import get_data_source, load_history
from tinyquant.indicators import add_indicators
from tinyquant.strategies import get_strategy, strategy_info
from tinyquant.trading import PaperBroker

st.set_page_config(page_title="Tiny-Quant-AI 看板", page_icon="📈", layout="wide")

UP, DOWN = "#ef4444", "#22c55e"  # A股习惯：红涨绿跌


# ----------------------- 数据/计算辅助 -----------------------
@st.cache_data(show_spinner=False, ttl=1800)
def cached_history(symbol: str, start: str, end: str, source: str) -> pd.DataFrame:
    return load_history(symbol, start=start, end=end or None, source=source)


@st.cache_data(show_spinner=False, ttl=86400)
def cached_stock_name(symbol: str, source: str) -> str:
    """查简称，最多等 2 秒；超时或失败则显示代码，不阻塞回测。"""
    def _fetch():
        return get_data_source(source).get_name(symbol) or symbol

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_fetch).result(timeout=2)
    except Exception:
        return symbol


def compute_signals(df: pd.DataFrame, strategy: str, params: dict) -> pd.Series:
    strat = get_strategy(strategy, **params)
    return strat.run(df)


def signal_points(df: pd.DataFrame, signals: pd.Series):
    """返回买入 / 卖出的时间点与价格（按信号次日执行）。"""
    pos = signals.reindex(df.index).fillna(0)
    exec_pos = pos.shift(1).fillna(0)  # 与回测口径一致：次日成交
    change = exec_pos.diff().fillna(exec_pos)
    buy_idx = df.index[change > 0]
    sell_idx = df.index[change < 0]
    return (
        (buy_idx, df.loc[buy_idx, "close"]),
        (sell_idx, df.loc[sell_idx, "close"]),
    )


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


# ----------------------- 图表 -----------------------
def fig_kline(df: pd.DataFrame, signals: pd.Series, title: str) -> go.Figure:
    ind = add_indicators(df)
    (bx, by), (sx, sy) = signal_points(df, signals)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
        vertical_spacing=0.03, subplot_titles=(title, "成交量"),
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="K线", increasing_line_color=UP, decreasing_line_color=DOWN,
        ),
        row=1, col=1,
    )
    for ma, color in [("ma5", "#f59e0b"), ("ma20", "#3b82f6"), ("boll_upper", "#9ca3af"), ("boll_lower", "#9ca3af")]:
        if ma in ind:
            fig.add_trace(
                go.Scatter(x=ind.index, y=ind[ma], name=ma, line=dict(width=1, color=color)),
                row=1, col=1,
            )
    fig.add_trace(
        go.Scatter(x=bx, y=by, mode="markers", name="买入",
                   marker=dict(symbol="triangle-up", size=12, color=UP)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sx, y=sy, mode="markers", name="卖出",
                   marker=dict(symbol="triangle-down", size=12, color=DOWN)),
        row=1, col=1,
    )
    vol_colors = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["volume"], name="成交量", marker_color=vol_colors),
        row=2, col=1,
    )
    fig.update_layout(
        height=560, xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=0), margin=dict(t=40, b=10),
    )
    return fig


def fig_equity(res) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.equity.index, y=res.equity, name="策略",
                             line=dict(color="#3b82f6", width=2)))
    fig.add_trace(go.Scatter(x=res.benchmark.index, y=res.benchmark, name="买入持有",
                             line=dict(color="#9ca3af", width=1.5, dash="dot")))
    fig.update_layout(height=340, title="资金曲线", hovermode="x unified",
                      legend=dict(orientation="h", y=1.05, x=0), margin=dict(t=40, b=10))
    return fig


def fig_drawdown(res) -> go.Figure:
    dd = drawdown(res.equity) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd.index, y=dd, name="回撤", fill="tozeroy",
                             line=dict(color=DOWN, width=1)))
    fig.update_layout(height=240, title="回撤 (%)", hovermode="x unified",
                      margin=dict(t=40, b=10))
    return fig


# ----------------------- 侧边栏 -----------------------
st.sidebar.title("📈 Tiny-Quant-AI")
infos = {i["name"]: i for i in strategy_info()}

source = st.sidebar.selectbox("数据源", ["akshare", "yfinance"],
                              index=0 if DEFAULT_SOURCE == "akshare" else 1,
                              help="akshare=A股，yfinance=美股")
symbols_raw = st.sidebar.text_input("股票代码（逗号分隔）",
                                    value="000001,600519" if source == "akshare" else "AAPL,MSFT")
symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

strategy = st.sidebar.selectbox("策略", list(infos), format_func=lambda n: f"{n} · {infos[n]['description'][:14]}")

st.sidebar.caption("策略参数")
params = {}
for k, v in infos[strategy]["params"].items():
    if isinstance(v, bool):
        params[k] = st.sidebar.checkbox(k, v)
    elif isinstance(v, int):
        params[k] = int(st.sidebar.number_input(k, value=v, step=1))
    elif isinstance(v, float):
        params[k] = float(st.sidebar.number_input(k, value=v, step=0.05, format="%.3f"))
    else:
        params[k] = st.sidebar.text_input(k, str(v))

c1, c2 = st.sidebar.columns(2)
start = c1.date_input("开始", dt.date.today() - dt.timedelta(days=365 * 2))
end = c2.date_input("结束", dt.date.today())
init_cash = st.sidebar.number_input("初始资金", value=float(INIT_CASH), step=10000.0)


# ----------------------- 主区域 -----------------------
tab_bt, tab_ind, tab_paper, tab_quote = st.tabs(["🔬 回测", "📊 技术指标", "💼 模拟盘", "⚡ 实时行情"])

# ===== 回测 =====
with tab_bt:
    st.subheader(f"回测 · 策略 {strategy}")
    if st.button("运行回测", type="primary"):
        rows = []
        primary_ctx = None
        n = len(symbols) or 1
        progress = st.progress(0.0, text="准备回测…")
        for i, sym in enumerate(symbols):
            progress.progress(i / n, text=f"正在回测 {sym}（{i + 1}/{len(symbols)}）：拉取行情…")
            try:
                df = cached_history(sym, str(start), str(end), source)
            except Exception as e:  # noqa: BLE001
                st.error(f"{sym} 获取行情失败：{type(e).__name__}: {e}")
                continue
            if df.empty:
                st.warning(f"{sym} 无数据")
                continue
            progress.progress((i + 0.5) / n, text=f"正在回测 {sym}：计算信号与绩效…")
            signals = compute_signals(df, strategy, params)
            res = backtest(df, signals, init_cash=init_cash)
            m = res.metrics
            name = cached_stock_name(sym, source)
            rows.append({
                "名称": name,
                "股票": sym,
                "总收益": f"{m['total_return']*100:.2f}%",
                "年化": f"{m['annual_return']*100:.2f}%",
                "夏普": f"{m['sharpe']:.2f}",
                "最大回撤": f"{m['max_drawdown']*100:.2f}%",
                "胜率": f"{m['win_rate']*100:.1f}%",
                "交易数": m["num_trades"],
                "基准总收益": f"{res.metrics_benchmark['total_return']*100:.2f}%",
            })
            if primary_ctx is None:
                primary_ctx = (sym, name, df, signals, res)
        progress.progress(1.0, text="回测完成")
        if rows:
            st.session_state["bt"] = {"rows": rows, "primary": primary_ctx}
        else:
            st.warning("没有成功回测的标的，请检查代码或网络后重试。")

    if "bt" in st.session_state:
        data = st.session_state["bt"]
        st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True, hide_index=True)
        if data["primary"]:
            primary = data["primary"]
            # 兼容旧 session：以前是 (sym, df, signals, res)
            if len(primary) == 5:
                sym, name, df, signals, res = primary
            else:
                sym, df, signals, res = primary
                name = cached_stock_name(sym, source)
            m = res.metrics
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总收益", f"{m['total_return']*100:.2f}%",
                      f"{(m['total_return']-res.metrics_benchmark['total_return'])*100:.2f}% vs基准")
            k2.metric("年化收益", f"{m['annual_return']*100:.2f}%")
            k3.metric("夏普比率", f"{m['sharpe']:.2f}")
            k4.metric("最大回撤", f"{m['max_drawdown']*100:.2f}%")
            label = f"{name}（{sym}）" if name and name != sym else sym
            st.caption(f"主标的 {label}（列表第一只）：资金/回撤在上，K 线在下")
            cc1, cc2 = st.columns([3, 2])
            cc1.plotly_chart(fig_equity(res), use_container_width=True)
            cc2.plotly_chart(fig_drawdown(res), use_container_width=True)
            st.plotly_chart(fig_kline(df, signals, f"{label} K线 + 买卖点"), use_container_width=True)
    else:
        st.info("设置好左侧参数后点击「运行回测」。")

# ===== 技术指标 =====
with tab_ind:
    st.subheader("技术指标")
    sym = st.selectbox("查看标的", symbols, key="ind_sym") if symbols else None
    if sym and st.button("加载指标", key="load_ind"):
        try:
            df = cached_history(sym, str(start), str(end), source)
            ind = add_indicators(df)
            st.plotly_chart(fig_kline(df, pd.Series(0, index=df.index), f"{sym} K线 + 均线/布林"),
                            use_container_width=True)
            sub = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("MACD", "RSI / KDJ"))
            sub.add_trace(go.Bar(x=ind.index, y=ind["macd_hist"], name="MACD柱"), row=1, col=1)
            sub.add_trace(go.Scatter(x=ind.index, y=ind["macd_dif"], name="DIF"), row=1, col=1)
            sub.add_trace(go.Scatter(x=ind.index, y=ind["macd_dea"], name="DEA"), row=1, col=1)
            sub.add_trace(go.Scatter(x=ind.index, y=ind["rsi14"], name="RSI14"), row=2, col=1)
            if "kdj_k" in ind:
                sub.add_trace(go.Scatter(x=ind.index, y=ind["kdj_k"], name="KDJ-K"), row=2, col=1)
            sub.update_layout(height=420, hovermode="x unified", margin=dict(t=40, b=10))
            st.plotly_chart(sub, use_container_width=True)
            with st.expander("最近 10 行指标数据"):
                st.dataframe(ind.tail(10), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"加载失败：{type(e).__name__}: {e}")

# ===== 模拟盘 =====
with tab_paper:
    st.subheader("模拟盘")
    broker = PaperBroker()
    colA, colB, colC = st.columns(3)
    if colA.button("① 用当前策略初始化/重置账户"):
        broker.init_account(strategy=strategy, symbols=symbols, source=source,
                            init_cash=init_cash, strategy_params=params)
        st.success(f"已初始化：{strategy} · {symbols} · 初始资金 {init_cash:.0f}")
    if colB.button("② 运行一次调仓"):
        try:
            broker.run_once(verbose=False)
            st.success("已按最新行情执行一次调仓")
        except Exception as e:  # noqa: BLE001
            st.error(f"运行失败：{type(e).__name__}: {e}")
    if colC.button("③ 刷新状态"):
        st.rerun()

    if broker.acc:
        try:
            stt = broker.status()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("总权益", f"{stt['equity']:.0f}")
            k2.metric("现金", f"{stt['cash']:.0f}")
            k3.metric("持仓市值", f"{stt['market_value']:.0f}")
            k4.metric("累计收益率", f"{stt['total_return']*100:.2f}%")
            st.caption(f"策略 {stt['strategy']} · 股票池 {stt['symbols']}")
            if stt["positions"]:
                st.markdown("**当前持仓**")
                st.dataframe(pd.DataFrame(stt["positions"]), use_container_width=True, hide_index=True)
            else:
                st.info("当前空仓")
            curve = broker.acc.get("equity_curve", [])
            if len(curve) >= 2:
                ec = pd.DataFrame(curve)
                ec["time"] = pd.to_datetime(ec["time"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ec["time"], y=ec["equity"], name="总权益",
                                         line=dict(color="#3b82f6", width=2)))
                fig.update_layout(height=300, title="模拟盘权益曲线", hovermode="x unified",
                                  margin=dict(t=40, b=10))
                st.plotly_chart(fig, use_container_width=True)
            trades = broker.acc.get("trades", [])
            if trades:
                with st.expander(f"成交记录（{len(trades)} 笔）"):
                    st.dataframe(pd.DataFrame(trades[::-1]), use_container_width=True, hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"读取账户失败：{type(e).__name__}: {e}")
    else:
        st.info("尚未初始化模拟盘账户，点上方「①」创建。")

# ===== 实时行情 =====
with tab_quote:
    st.subheader("实时行情")
    if st.button("获取最新报价"):
        ds = get_data_source(source)
        cols = st.columns(min(len(symbols), 4) or 1)
        for i, sym in enumerate(symbols):
            try:
                rt = ds.get_realtime(sym)
                delta = None
                if rt.get("price") and rt.get("pre_close"):
                    delta = f"{(rt['price']/rt['pre_close']-1)*100:.2f}%"
                cols[i % len(cols)].metric(f"{rt.get('name', sym)} ({sym})",
                                           f"{rt.get('price')}", delta)
            except Exception as e:  # noqa: BLE001
                cols[i % len(cols)].error(f"{sym}: {type(e).__name__}")
