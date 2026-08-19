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
from tinyquant.data import watchlist as wl
from tinyquant.indicators import add_indicators
from tinyquant.strategies import get_strategy, strategy_info
from tinyquant.trading import PaperBroker

st.set_page_config(page_title="Tiny-Quant-AI 看板", page_icon="📈", layout="wide")

# 全局微调：缩小 metric 卡片字号、收紧分组容器间距，让指标区更紧凑美观
st.markdown(
    """
    <style>
      div[data-testid="stMetric"] { padding: 2px 0; }
      div[data-testid="stMetricValue"] { font-size: 1.15rem; line-height: 1.2; }
      div[data-testid="stMetricLabel"] p { font-size: 0.78rem; color: #64748b; }
      div[data-testid="stMetricDelta"] { font-size: 0.72rem; }
      div[data-testid="stMetricDelta"] div { font-size: 0.72rem; }
      /* 分组小标题更贴近卡片 */
      div[data-testid="stVerticalBlockBorderWrapper"] h6 {
        margin: 0 0 2px 0; color: #334155; font-weight: 600;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

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


@st.cache_data(show_spinner=False, ttl=300)
def cached_search(query: str, source: str) -> list[dict]:
    return wl.search(query, source, limit=6)


def _apply_pending_selection(source: str) -> None:
    """在「本次使用」控件创建之前，把待加入/删除的勾选写进 session_state。"""
    sel_key = f"pool_sel_{source}"
    add_sym = st.session_state.pop(f"_pending_add_{source}", None)
    del_sym = st.session_state.pop(f"_pending_del_{source}", None)
    if not add_sym and not del_sym:
        return
    cur = st.session_state.get(sel_key)
    if not isinstance(cur, list):
        cur = list(wl.selected(source))
    if add_sym and add_sym not in cur:
        cur = cur + [add_sym]
    if del_sym:
        cur = [s for s in cur if s != del_sym]
    st.session_state[sel_key] = cur


def render_symbol_picker(source: str) -> list[str]:
    """搜索加入股票池，勾选本次要用的标的；池子会写到本地文件。"""
    _apply_pending_selection(source)

    st.sidebar.markdown("**股票池**")
    query = st.sidebar.text_input(
        "搜索添加",
        placeholder="名称 / 代码 / 拼音，如 茅台 或 000001",
        key=f"search_{source}",
        label_visibility="collapsed",
    )
    st.sidebar.caption("搜索名称、代码或拼音，点结果即可加入")
    if query.strip():
        hits = cached_search(query.strip(), source)
        if not hits:
            st.sidebar.caption("没有匹配，可在下方「管理股票池」里手动加代码")
        already = {x["symbol"] for x in wl.pool(source)}
        for hit in hits:
            label = f"{hit['name']} · {hit['symbol']}"
            if hit["symbol"] in already:
                st.sidebar.caption(f"已在池中 · {label}")
                continue
            if st.sidebar.button(f"＋ {label}", key=f"add_{source}_{hit['symbol']}"):
                wl.add(source, hit["symbol"], hit["name"])
                st.session_state[f"_pending_add_{source}"] = hit["symbol"]
                st.rerun()

    items = wl.pool(source)
    options = [x["symbol"] for x in items]
    labels = {
        x["symbol"]: (f"{x.get('name') or x['symbol']} · {x['symbol']}"
                      if (x.get("name") and x["name"] != x["symbol"]) else x["symbol"])
        for x in items
    }
    saved = [s for s in wl.selected(source) if s in options]
    default = saved or options
    sel_key = f"pool_sel_{source}"
    ms_kwargs = dict(
        options=options,
        format_func=lambda s: labels.get(s, s),
        key=sel_key,
        help="勾选后，回测 / 指标 / 模拟盘 / 行情都用这些标的。股票池会保存在本地。",
    )
    if sel_key not in st.session_state:
        ms_kwargs["default"] = default
    chosen = st.sidebar.multiselect("本次使用", **ms_kwargs)
    if chosen != saved:
        wl.set_selected(source, chosen)
    if not items:
        st.sidebar.info("股票池是空的，先搜索添加一只。")
    elif not chosen:
        st.sidebar.warning("请至少勾选一只，再运行回测。")

    with st.sidebar.expander("管理股票池"):
        for x in items:
            c1, c2 = st.columns([4, 1])
            c1.caption(labels.get(x["symbol"], x["symbol"]))
            if c2.button("删", key=f"del_{source}_{x['symbol']}"):
                wl.remove(source, x["symbol"])
                st.session_state[f"_pending_del_{source}"] = x["symbol"]
                st.rerun()
        manual = st.text_input("手动加代码", placeholder="600519 或 AAPL", key=f"manual_{source}")
        if st.button("加入股票池", key=f"manual_add_{source}") and manual.strip():
            item = wl.add(source, manual.strip())
            name = cached_stock_name(item["symbol"], source)
            if name and name != item["symbol"]:
                wl.add(source, item["symbol"], name)
            st.session_state[f"_pending_add_{source}"] = item["symbol"]
            st.rerun()
    return chosen


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


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def _pct(v) -> str:
    return "-" if v is None or v != v else f"{v * 100:.2f}%"


def _num(v) -> str:
    if v is None or v != v:
        return "-"
    if v == float("inf"):
        return "∞"
    return f"{v:.2f}"


_GAUGE_GREEN, _GAUGE_YELLOW, _GAUGE_RED = "#22c55e", "#eab308", "#ef4444"


def _gauge(fig, row, col, value, rng, red_to, yellow_to):
    """在 make_subplots 的指定位置画一个带红/黄/绿区间的比率仪表盘。"""
    raw = value if value is not None and value == value and value != float("inf") else 0.0
    v = max(rng[0], min(rng[1], raw))
    num_color = _GAUGE_GREEN if raw >= yellow_to else (_GAUGE_YELLOW if raw >= red_to else _GAUGE_RED)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(v, 2),
            number=dict(font=dict(size=28, color=num_color), valueformat=".2f"),
            gauge=dict(
                axis=dict(range=rng, tickwidth=1, tickcolor="#94a3b8",
                          tickfont=dict(size=10, color="#94a3b8"), dtick=1),
                bar=dict(color="rgba(148,163,184,0.0)", thickness=0),
                borderwidth=0,
                steps=[
                    dict(range=[rng[0], red_to], color="rgba(239,68,68,0.22)"),
                    dict(range=[red_to, yellow_to], color="rgba(234,179,8,0.22)"),
                    dict(range=[yellow_to, rng[1]], color="rgba(34,197,94,0.22)"),
                ],
                threshold=dict(
                    line=dict(color=num_color, width=4),
                    thickness=0.82, value=v,
                ),
            ),
        ),
        row=row, col=col,
    )


def fig_ratio_gauges(m: dict) -> go.Figure:
    """夏普 / 索提诺 / 卡玛三个核心比率仪表盘，红黄绿一眼看好坏。"""
    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "indicator"}] * 3],
        subplot_titles=("夏普比率", "索提诺比率", "卡玛比率"),
        horizontal_spacing=0.08,
    )
    _gauge(fig, 1, 1, m.get("sharpe"), [-1, 3], 0, 1)
    _gauge(fig, 1, 2, m.get("sortino"), [-1, 3], 0, 1)
    _gauge(fig, 1, 3, m.get("calmar"), [0, 3], 0.5, 1)
    fig.update_layout(
        height=250, margin=dict(t=48, b=6, l=24, r=24),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
    )
    fig.update_annotations(font=dict(size=13, color="#cbd5e1"), yshift=6)
    return fig


def fig_metric_bars(m: dict, mb: dict) -> go.Figure:
    """收益/风险类指标的「策略 vs 基准」水平分组条形图（统一 %）。"""
    labels = ["总收益", "年化收益", "年化波动", "最大回撤"]
    keys = ["total_return", "annual_return", "annual_volatility", "max_drawdown"]
    strat = [(m.get(k) or 0) * 100 for k in keys]
    bench = [(mb.get(k) or 0) * 100 for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=bench, name="买入持有", orientation="h",
        marker=dict(color="#64748b", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in bench],
        textposition="outside", textfont=dict(size=14, color="#94a3b8"), cliponaxis=False,
    ))
    fig.add_trace(go.Bar(
        y=labels, x=strat, name="策略", orientation="h",
        marker=dict(color="#3b82f6", line=dict(width=0)),
        text=[f"{v:.1f}%" for v in strat],
        textposition="outside", textfont=dict(size=14, color="#60a5fa"), cliponaxis=False,
    ))
    lo = min(strat + bench + [0])
    hi = max(strat + bench + [0])
    pad = max((hi - lo) * 0.20, 3)
    fig.update_layout(
        barmode="group", bargap=0.28, bargroupgap=0.14, height=250,
        title=dict(text="策略 vs 基准（%）", font=dict(size=13, color="#cbd5e1"), x=0.02),
        uniformtext=dict(mode="show", minsize=12),
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=12, color="#cbd5e1"),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=6, l=6, r=24),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    fig.update_xaxes(range=[lo - pad, hi + pad], visible=False, zeroline=False)
    fig.add_vline(x=0, line_width=1, line_color="rgba(148,163,184,0.35)")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=13, color="#cbd5e1"),
                     showgrid=False)
    return fig


def _card_cell(col, spec: tuple) -> None:
    label, value = spec[0], spec[1]
    delta_val = spec[2] if len(spec) > 2 else None
    kwargs = spec[3] if len(spec) > 3 else {}
    col.metric(label, value, delta_val, **kwargs)


def _card_row(container, specs: list[tuple]) -> None:
    """在容器里铺一行 st.metric 卡片。specs: (label, value, delta, kwargs)。"""
    cols = container.columns(len(specs))
    for col, spec in zip(cols, specs):
        _card_cell(col, spec)


def _card_grid(container, specs: list[tuple], per_row: int) -> None:
    """固定列数的网格：每行都用同样的列宽，不足的位置留空，保证行列对齐。"""
    for start in range(0, len(specs), per_row):
        chunk = specs[start:start + per_row]
        cols = container.columns(per_row)
        for col, spec in zip(cols, chunk):
            _card_cell(col, spec)


def render_metric_cards(m: dict, mb: dict) -> None:
    """完整绩效指标：核心比率仪表盘 + 对比条形图 + 分组边框卡片。"""
    def delta(key, kind="pct"):
        if key not in m or key not in mb or m[key] != m[key] or mb[key] != mb[key]:
            return None
        d = m[key] - mb[key]
        return f"{d * 100:.2f}% vs基准" if kind == "pct" else f"{d:.2f} vs基准"

    # 顶部图形区：左仪表盘、右对比条形图
    g1, g2 = st.columns([3, 2])
    g1.plotly_chart(fig_ratio_gauges(m), use_container_width=True)
    g2.plotly_chart(fig_metric_bars(m, mb), use_container_width=True)

    box = st.container(border=True)
    box.markdown("###### 收益 & 风险")
    dd_delta = delta("max_drawdown")
    _card_row(box, [
        ("总收益", _pct(m.get("total_return")), delta("total_return")),
        ("年化收益", _pct(m.get("annual_return")), delta("annual_return")),
        ("年化波动", _pct(m.get("annual_volatility"))),
        ("最大回撤", _pct(m.get("max_drawdown")), dd_delta,
         {"delta_color": "inverse" if dd_delta else "normal"}),
        ("最长回撤期", f"{int(m.get('max_dd_duration', 0))} 天"),
    ])

    box = st.container(border=True)
    box.markdown("###### 相对基准（是否真有超额）")
    _card_row(box, [
        ("Beta", _num(m.get("beta")), None, {"help": "对基准的敏感度，越接近0越独立"}),
        ("Alpha(年化)", _pct(m.get("alpha")), None, {"help": "剔除Beta后的超额收益，>0才是真本事"}),
        ("信息比率", _num(m.get("info_ratio")), None, {"help": "超额收益/跟踪误差，>0.5较好"}),
        ("持仓时间占比", _pct(m.get("exposure")), None, {"help": "资金在场内的时间比例"}),
    ])

    box = st.container(border=True)
    box.markdown("###### 交易质量")
    _card_grid(box, [
        ("胜率", _pct(m.get("win_rate"))),
        ("交易数", f"{int(m.get('num_trades', 0))}"),
        ("盈亏比", _num(m.get("profit_factor")), None, {"help": "总盈利/总亏损，>1才赚钱"}),
        ("平均盈利", _pct(m.get("avg_win"))),
        ("平均亏损", _pct(m.get("avg_loss"))),
        ("平均持仓", f"{m.get('avg_holding_days', 0):.1f} 天"),
        ("最好一笔", _pct(m.get("best_trade"))),
        ("最差一笔", _pct(m.get("worst_trade"))),
        ("换手率", _num(m.get("turnover")), None, {"help": "累计仓位变动，越高交易越频繁"}),
        ("累计手续费", f"{m.get('total_cost', 0) * 100:.3f}%", None, {"help": "占初始资金比例"}),
    ], per_row=5)


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
symbols = render_symbol_picker(source)

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
    if not symbols:
        st.info("左侧股票池先搜索添加、再勾选至少一只，然后点「运行回测」。")
    elif st.button("运行回测", type="primary"):
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
            mb = res.metrics_benchmark
            render_metric_cards(m, mb)
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
    if not symbols:
        st.info("请先在左侧股票池勾选标的。")
        sym = None
    else:
        sym = st.selectbox(
            "查看标的",
            symbols,
            format_func=lambda s: wl.label_of(source, s),
            key="ind_sym",
        )
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
        if not symbols:
            st.warning("请先在左侧股票池勾选至少一只。")
        else:
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
    if not symbols:
        st.info("请先在左侧股票池勾选标的。")
    elif st.button("获取最新报价"):
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
