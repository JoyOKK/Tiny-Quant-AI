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
from tinyquant.data import strategy_lib as sl
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
      /* 「买卖点显示」横向单选：收紧标签与选项、以及与下方图表的间距 */
      div[data-testid="stRadio"] { margin-bottom: -0.6rem; }
      div[data-testid="stRadio"] > label { margin-bottom: 0; font-size: 0.8rem; color: #64748b; }
      div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 1.1rem; margin-top: -0.2rem; align-items: center;
      }
      div[data-testid="stRadio"] div[role="radiogroup"] label p { font-size: 0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

UP, DOWN = "#ef4444", "#22c55e"  # A股习惯：红涨绿跌
UP_FILL, DOWN_FILL = "rgba(239,68,68,0.12)", "rgba(34,197,94,0.12)"


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


def signal_points(df: pd.DataFrame, signals: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """买入 / 卖出当日 OHLC（与回测一致：信号次日按收盘价成交）。"""
    pos = signals.reindex(df.index).fillna(0)
    exec_pos = pos.shift(1).fillna(0)
    change = exec_pos.diff().fillna(exec_pos)
    cols = ["open", "high", "low", "close"]
    return df.loc[df.index[change > 0], cols], df.loc[df.index[change < 0], cols]


def _x_rangebreaks(index: pd.DatetimeIndex | pd.Index) -> list[dict]:
    """隐藏非交易日空隙（双休 + 节假日），使 K 线与指标曲线紧凑排列。"""
    idx = pd.DatetimeIndex(index).normalize().unique().sort_values()
    if len(idx) < 2:
        return [dict(bounds=["sat", "mon"])]
    # 日历日全集减去实际交易日 = 空隙；周末交给 bounds，values 只补工作日节假日
    missing = pd.date_range(idx.min(), idx.max(), freq="D").difference(idx)
    holidays = missing[missing.weekday < 5]  # 周一=0 … 周五=4
    breaks: list[dict] = [dict(bounds=["sat", "mon"])]
    if len(holidays):
        breaks.append(dict(values=[d.strftime("%Y-%m-%d") for d in holidays]))
    return breaks


_TRADE_COLS = ["交易", "日期", "方向", "成交价", "收益", "开", "高", "低"]


def trade_log(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """买卖点明细：按「买入→卖出」配对成一笔笔完整交易，卖出行给出该笔收益。"""
    buys, sells = signal_points(df, signals)
    events = [(ts, "买入", r) for ts, r in buys.iterrows()]
    events += [(ts, "卖出", r) for ts, r in sells.iterrows()]
    events.sort(key=lambda x: x[0])

    rows = []
    trade_no = 0
    buy_price = None
    for ts, side, r in events:
        close = float(r["close"])
        ret = ""
        if side == "买入":
            trade_no += 1
            buy_price = close
        elif buy_price:
            ret = f"{(close / buy_price - 1) * 100:+.2f}%"
        rows.append({
            "交易": f"T{trade_no}" if trade_no else "T1",
            "日期": pd.Timestamp(ts).strftime("%Y-%m-%d"),
            "方向": side,
            "成交价": round(close, 2),
            "收益": ret,
            "开": round(float(r["open"]), 2),
            "高": round(float(r["high"]), 2),
            "低": round(float(r["low"]), 2),
        })
    if not rows:
        return pd.DataFrame(columns=_TRADE_COLS)
    return pd.DataFrame(rows, columns=_TRADE_COLS)


# 交易配色：按笔循环使用，相邻交易颜色不同，便于区分
_TRADE_PALETTE = [
    "rgba(59,130,246,0.16)",   # 蓝
    "rgba(234,179,8,0.16)",    # 黄
    "rgba(168,85,247,0.16)",   # 紫
    "rgba(20,184,166,0.16)",   # 青
    "rgba(244,114,182,0.16)",  # 粉
    "rgba(148,163,184,0.16)",  # 灰
]


def style_trade_log(log: pd.DataFrame):
    """给每一笔完整交易上一种底色，收益按正负着色。"""
    def _row_bg(row):
        try:
            i = int(str(row["交易"])[1:]) - 1
        except ValueError:
            i = 0
        color = _TRADE_PALETTE[i % len(_TRADE_PALETTE)]
        return [f"background-color: {color}"] * len(row)

    def _ret_color(val):
        s = str(val)
        if s.startswith("+"):
            return "color: #ef4444; font-weight: 600"
        if s.startswith("-"):
            return "color: #22c55e; font-weight: 600"
        return ""

    return (
        log.style
        .apply(_row_bg, axis=1)
        .map(_ret_color, subset=["收益"])
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
def _trade_marker_offset(df: pd.DataFrame) -> float:
    """买卖点相对影线的价差：跟单根 K 线高度挂钩，放大后仍贴着对应那根。"""
    span = float(df["high"].max() - df["low"].min()) or 1.0
    bar = float((df["high"] - df["low"]).median() or 0.0)
    if bar <= 0:
        bar = span * 0.01
    return max(bar * 1.8, span * 0.012)


def _add_trade_markers(
    fig: go.Figure,
    pts: pd.DataFrame,
    *,
    side: str,
    offset: float,
    show_labels: bool,
    show_lines: bool,
    marker_size: int,
) -> None:
    """买卖点标记：买入在 K 线正下方引出红色「B」色块，卖出在正上方引出蓝色「S」色块，
    均以垂直虚线连到当日影线。marker_size 随交易密集程度缩放。
    """
    if pts.empty:
        return
    is_buy = side == "buy"
    letter = "B" if is_buy else "S"
    name = "买入" if is_buy else "卖出"
    block_color = "#ef4444" if is_buy else "#2563eb"  # 买入红 B / 卖出蓝 S
    wick = pts["low"] if is_buy else pts["high"]
    # 色块位置：买入在 K 线正下方、卖出在正上方，偏移量随 K 线高度自适应
    y_mark = wick - offset if is_buy else wick + offset

    # 垂直虚线：从当日 K 线影线引出到 B / S 色块
    xs, ys = [], []
    for x, yw, ym in zip(pts.index, wick, y_mark):
        xs += [x, x, None]
        ys += [float(yw), float(ym), None]
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=block_color, width=1.2, dash="dot"),
            legendgroup=name, showlegend=False, hoverinfo="skip",
            cliponaxis=False, opacity=0.75,
        ),
        row=1, col=1,
    )
    # B / S 色块：方块内嵌字母，替代原三角标
    block_size = marker_size + 6
    fig.add_trace(
        go.Scatter(
            x=pts.index, y=y_mark, mode="markers+text",
            name=name, legendgroup=name,
            marker=dict(symbol="square", size=block_size, color=block_color,
                        line=dict(width=1.2, color="#ffffff")),
            text=[letter] * len(pts), textposition="middle center",
            textfont=dict(color="#ffffff", size=max(10, int(marker_size * 0.8)),
                          family="PingFang SC, Microsoft YaHei, sans-serif"),
            hoverinfo="skip", cliponaxis=False,
        ),
        row=1, col=1,
    )


# 买卖点数量超过该阈值时自动切精简模式（不显示价格标签/指引线）
_DENSE_TRADE_THRESHOLD = 14


# 主图叠加指标：画在 K 线所在的价格面板上（key 与 add_indicators 输出列一致）→ (显示名, 颜色)
_OVERLAY_INDICATORS: dict[str, tuple[str, str]] = {
    "ma5": ("MA5", "#f59e0b"),
    "ma10": ("MA10", "#10b981"),
    "ma20": ("MA20", "#3b82f6"),
    "ema12": ("EMA12", "#8b5cf6"),
    "ema26": ("EMA26", "#ec4899"),
    "boll_upper": ("BOLL 上轨", "#9ca3af"),
    "boll_mid": ("BOLL 中轨", "#6b7280"),
    "boll_lower": ("BOLL 下轨", "#9ca3af"),
    # k 日均线曲线的一阶导(斜率)/二阶导(加速度)/三阶导(急动度)：数值量纲与价格差别大，
    # 画在主图右侧第二坐标轴上，围绕 0 波动，过 0 处即波峰/波谷
    "ma_slope": ("MA斜率(一阶导)", "#0ea5e9"),
    "ma_accel": ("MA加速度(二阶导)", "#f43f5e"),
    "ma_jerk": ("MA三阶导", "#a855f7"),
    # 斜率/加速度/三阶导的保形(Savitzky-Golay)滤波平滑版，去抖后更干净，便于识别真正的拐点
    # 注意：中心式(_smooth_series savgol)用到未来数据，仅供展示，勿据可视谷/峰做实盘判断
    "ma_slope_smooth": ("MA斜率(平滑)", "#22c55e"),
    "ma_accel_smooth": ("MA加速度(平滑)", "#eab308"),
    "ma_jerk_smooth": ("MA三阶导(平滑)", "#c084fc"),
    # 因果(trailing)版：每点只用「截至当日」的历史，实盘真正能看到的平滑曲线
    # 与 slope_swing 策略内部所用曲线一致；较中心式略滞后、末端不会回改(no repaint)
    "ma_slope_smooth_causal": ("MA斜率(平滑·因果)", "#0d9488"),
    "ma_accel_smooth_causal": ("MA加速度(平滑·因果)", "#d97706"),
    "ma_jerk_smooth_causal": ("MA三阶导(平滑·因果)", "#7c3aed"),
    # 固定 10 日均线斜率的因果平滑版（不随「导数 k日均线」控件变化）
    "ma10_slope_smooth_causal": ("MA10斜率(平滑·因果)", "#059669"),
}

# 需在主图右轴、且要按 k 日均线现算的导数类指标
_DERIV_KEYS = {
    "ma_slope", "ma_accel", "ma_jerk",
    "ma_slope_smooth", "ma_accel_smooth", "ma_jerk_smooth",
    "ma_slope_smooth_causal", "ma_accel_smooth_causal", "ma_jerk_smooth_causal",
    "ma10_slope_smooth_causal",
}

# 副图指标：量纲与价格不同，各自单独成一个子图 → 显示名
_PANEL_INDICATORS: dict[str, str] = {
    "macd": "MACD",
    "rsi14": "RSI",
    "kdj": "KDJ",
    "mom10": "动量 MOM10",
    "atr14": "波动 ATR14",
}

# 默认叠加的主图指标，保持与旧版一致
_DEFAULT_OVERLAYS = ["ma5", "ma20", "boll_upper", "boll_lower"]

# 曲线平滑（滤波）方式：key → 显示名。作用于各指标曲线，降低高频抖动
_SMOOTH_METHODS: dict[str, str] = {
    "none": "不平滑",
    "sma": "移动平均 SMA",
    "ema": "指数平滑 EMA",
    "median": "中位数（抗尖刺）",
    "savgol": "Savitzky-Golay（保形）",
}


def smooth_method_label(key: str) -> str:
    return _SMOOTH_METHODS.get(key, key)


def _smooth_series(s: pd.Series, method: str, strength: int) -> pd.Series:
    """按所选滤波方式平滑一条曲线；strength 为窗口/跨度，越大越平滑。"""
    if method in (None, "none") or not strength or int(strength) < 2:
        return s
    strength = int(strength)
    if method == "sma":
        return s.rolling(strength, min_periods=1).mean()
    if method == "ema":
        return s.ewm(span=strength, adjust=False).mean()
    if method == "median":
        return s.rolling(strength, min_periods=1).median()
    if method == "savgol":
        try:
            from scipy.signal import savgol_filter
        except Exception:  # noqa: BLE001 - 没装 scipy 时退化为移动平均
            return s.rolling(strength, min_periods=1).mean()
        win = strength if strength % 2 == 1 else strength + 1  # 窗口须为奇数
        valid = s.dropna()
        if len(valid) < win or win < 3:
            return s
        poly = min(3, win - 1)
        out = s.copy()
        out.loc[valid.index] = savgol_filter(valid.to_numpy(), win, poly)
        return out
    return s


def _causal_savgol_series(s: pd.Series, window: int, poly: int = 3) -> pd.Series:
    """因果(trailing) Savitzky-Golay：每点只用「截至当日」的历史窗口取末端拟合值。

    与 slope_swing 策略内部所用平滑一致，不含未来信息 → 实盘真正可见的平滑曲线。
    """
    y = s.to_numpy(dtype=float)
    n = len(y)
    out = [float("nan")] * n
    try:
        from scipy.signal import savgol_filter
    except Exception:  # noqa: BLE001 - 缺 scipy 时退化为因果滚动均值
        return s.rolling(max(window, 1), min_periods=poly + 2).mean()
    for i in range(n):
        seg = y[max(0, i - window + 1): i + 1]
        valid = seg[~pd.isna(seg)]             # 跳过均线预热期的前导 NaN
        w = len(valid)
        if w < poly + 2:
            continue
        win = w if w % 2 == 1 else w - 1       # 窗口须为奇数且 ≤ 样本数
        out[i] = float(savgol_filter(valid[-win:], win, min(poly, win - 1))[-1])
    return pd.Series(out, index=s.index)


def indicator_options() -> list[str]:
    """可供「叠加指标」下拉选择的全部指标 key（主图在前、副图在后）。"""
    return list(_OVERLAY_INDICATORS) + list(_PANEL_INDICATORS)


def indicator_label(key: str) -> str:
    """下拉选项显示文案，标明该指标画在主图还是副图。"""
    if key in _OVERLAY_INDICATORS:
        return f"主图 · {_OVERLAY_INDICATORS[key][0]}"
    return f"副图 · {_PANEL_INDICATORS.get(key, key)}"


def _add_indicator_panel(fig: go.Figure, ind: pd.DataFrame, key: str, row: int,
                         smooth: tuple[str, int] | None = None) -> None:
    """把一个副图指标画到指定子图行上；smooth=(方式, 强度) 时对曲线做平滑。"""
    sm_m, sm_s = smooth if smooth else ("none", 0)

    def sy(col: str) -> pd.Series:  # 取列并按需平滑
        return _smooth_series(ind[col], sm_m, sm_s)

    if key == "macd" and "macd_hist" in ind:
        colors = [UP if v >= 0 else DOWN for v in ind["macd_hist"].fillna(0)]
        fig.add_trace(go.Bar(x=ind.index, y=sy("macd_hist"), name="MACD柱",
                             marker_color=colors, showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=sy("macd_dif"), name="DIF",
                                 line=dict(width=1, color="#f59e0b")), row=row, col=1)
        fig.add_trace(go.Scatter(x=ind.index, y=sy("macd_dea"), name="DEA",
                                 line=dict(width=1, color="#3b82f6")), row=row, col=1)
    elif key == "rsi14" and "rsi14" in ind:
        fig.add_trace(go.Scatter(x=ind.index, y=sy("rsi14"), name="RSI14",
                                 line=dict(width=1, color="#8b5cf6")), row=row, col=1)
        for y0 in (70, 30):
            fig.add_hline(y=y0, line=dict(color="#94a3b8", width=0.8, dash="dot"),
                          row=row, col=1)
    elif key == "kdj":
        for col_name, nm, color in [("kdj_k", "K", "#f59e0b"),
                                    ("kdj_d", "D", "#3b82f6"),
                                    ("kdj_j", "J", "#ec4899")]:
            if col_name in ind:
                fig.add_trace(go.Scatter(x=ind.index, y=sy(col_name), name=f"KDJ-{nm}",
                                         line=dict(width=1, color=color)), row=row, col=1)
    elif key == "mom10" and "mom10" in ind:
        fig.add_trace(go.Scatter(x=ind.index, y=sy("mom10"), name="MOM10",
                                 line=dict(width=1, color="#14b8a6")), row=row, col=1)
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=0.8, dash="dot"),
                      row=row, col=1)
    elif key == "atr14" and "atr14" in ind:
        fig.add_trace(go.Scatter(x=ind.index, y=sy("atr14"), name="ATR14",
                                 line=dict(width=1, color="#ef4444")), row=row, col=1)


def _add_unified_hover(fig, df, ind, overlays, smooth, *, anchor_y,
                       buy_close=None, sell_close=None) -> None:
    """在图表最顶部放一个不可见锚点，用一个悬浮框汇总当日各项指标。

    - 叠加指标按类型分两列：均线类（MA/EMA）一列、布林类（BOLL）一列，列内对齐；
    - 数值用等宽字体 + 定宽格式对齐，空格用不换行空格避免被 HTML 折叠；
    - 若当日为买/卖点，额外追加「买入价 / 卖出价」一行。
    anchor_y 为锚点纵坐标序列（固定在所有 K 线上方）；hovermode="x" 横向偏移不挡曲线。
    """
    sm_m, sm_s = smooth
    ma_keys = [k for k in overlays if not k.startswith("boll")]
    boll_keys = [k for k in overlays if k.startswith("boll")]
    ma_vals = {k: _smooth_series(ind[k], sm_m, sm_s) for k in ma_keys}
    boll_vals = {k: _smooth_series(ind[k], sm_m, sm_s) for k in boll_keys}
    ma_lw = max((len(_OVERLAY_INDICATORS[k][0]) for k in ma_keys), default=0)
    boll_lw = max((len(_OVERLAY_INDICATORS[k][0]) for k in boll_keys), default=0)

    def _cell(key, val, label_w):
        label = _OVERLAY_INDICATORS[key][0].ljust(label_w)
        v = f"{val:<8.2f}" if val == val else "—"  # NaN → 破折号
        return f"{label} {v}"

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    vol = df["volume"]
    n_ind = max(len(ma_keys), len(boll_keys))
    texts = []
    for i, ts in enumerate(df.index):
        # 开/收 为第一列，高/低 为第二列，各列严格左对齐
        lines = [f"<b>{ts.strftime('%Y-%m-%d')}</b>",
                 f"开 {float(o.iloc[i]):<8.2f}    高 {float(h.iloc[i]):<8.2f}",
                 f"收 {float(c.iloc[i]):<8.2f}    低 {float(l.iloc[i]):<8.2f}"]
        for r in range(n_ind):
            left = _cell(ma_keys[r], float(ma_vals[ma_keys[r]].iloc[i]), ma_lw) if r < len(ma_keys) else ""
            right = _cell(boll_keys[r], float(boll_vals[boll_keys[r]].iloc[i]), boll_lw) if r < len(boll_keys) else ""
            if left and right:
                lines.append(f"{left}    {right}")
            else:
                lines.append((left or right).rstrip())
        lines.append(f"成交量 {float(vol.iloc[i]):,.0f}")
        if buy_close is not None and buy_close.iloc[i] == buy_close.iloc[i]:
            lines.append(f"<b>买入价 {float(buy_close.iloc[i]):.2f}</b>")
        elif sell_close is not None and sell_close.iloc[i] == sell_close.iloc[i]:
            lines.append(f"<b>卖出价 {float(sell_close.iloc[i]):.2f}</b>")
        texts.append("<br>".join(lines).replace(" ", "\u00a0"))

    fig.add_trace(
        go.Scatter(
            x=df.index, y=list(anchor_y),
            mode="markers", marker=dict(opacity=0, size=0.1),
            showlegend=False, name="", text=texts,
            hovertemplate="%{text}<extra></extra>",
            hoverlabel=dict(bgcolor="rgba(15,23,42,0.92)",
                            bordercolor="rgba(148,163,184,0.55)", align="left",
                            font=dict(color="#e2e8f0", size=12,
                                      family="Menlo, Consolas, 'Courier New', monospace")),
        ),
        row=1, col=1,
    )


def fig_kline(
    df: pd.DataFrame,
    signals: pd.Series,
    title: str,
    *,
    label_mode: str = "auto",
    overlays: list[str] | None = None,
    panels: list[str] | None = None,
    smooth: tuple[str, int] | None = None,
    deriv_ma_window: int = 5,
    slope_smooth_window: int = 11,
    accel_smooth_window: int = 11,
    jerk_smooth_window: int = 11,
) -> go.Figure:
    """K 线 + 买卖点。

    label_mode: auto/detailed/compact 控制是否显示价格标签与指引线。
    overlays:   主图叠加指标 key 列表（均线/布林/斜率/加速度/三阶导等），None 时用默认组合。
    panels:     副图指标 key 列表（MACD/RSI/KDJ 等），各自单独成图。
    smooth:     (滤波方式, 强度)，对各指标曲线平滑降噪；K 线与成交量不受影响。
    deriv_ma_window:    计算「斜率/加速度/三阶导」所用的 k 日均线周期（默认 5）。
    slope_smooth_window: 「MA斜率(平滑)」的保形滤波窗口（默认 11，值小更贴合真实斜率）。
    accel_smooth_window: 「MA加速度(平滑)」的保形滤波窗口（默认 11，与斜率同方案）。
    jerk_smooth_window:  「MA三阶导(平滑)」的保形滤波窗口（默认 11，与加速度同方案）。
    """
    sm_m, sm_s = smooth if smooth else ("none", 0)
    ind = add_indicators(df)
    overlays = _DEFAULT_OVERLAYS if overlays is None else overlays
    overlays = [k for k in overlays
                if k in _OVERLAY_INDICATORS and (k in ind or k in _DERIV_KEYS)]
    panels = [k for k in (panels or []) if k in _PANEL_INDICATORS]

    # 主图叠加拆两类：常规列指标（价格量纲，左轴）与导数指标（斜率/加速度/三阶导，右轴）
    col_overlays = [k for k in overlays if k not in _DERIV_KEYS]
    deriv_overlays = [k for k in overlays if k in _DERIV_KEYS]
    # 直接算好各导数曲线的最终显示序列（右轴）
    deriv_display: dict[str, pd.Series] = {}
    if deriv_overlays:
        w = max(int(deriv_ma_window), 1)
        dma = df["close"].rolling(w, min_periods=1).mean()
        slope = dma.diff()                       # 一阶导：均线曲线斜率（原始）
        accel = slope.diff()                     # 二阶导：斜率的变化＝加速度（原始）
        jerk = accel.diff()                      # 三阶导：加速度的变化＝急动度（原始）
        # 原始斜率/加速度/三阶导保持未平滑（真·原始，始终保留）；
        # 平滑版是对同一条原始曲线做保形(SavGol)滤波，各阶各用自己的窗口
        deriv_display["ma_slope"] = slope
        deriv_display["ma_accel"] = accel
        deriv_display["ma_jerk"] = jerk
        deriv_display["ma_slope_smooth"] = _smooth_series(
            slope, "savgol", int(slope_smooth_window))
        deriv_display["ma_accel_smooth"] = _smooth_series(
            accel, "savgol", int(accel_smooth_window))
        deriv_display["ma_jerk_smooth"] = _smooth_series(
            jerk, "savgol", int(jerk_smooth_window))
        # 因果版：只用截至当日的历史，实盘可见、末端不回改；与 slope_swing 策略一致
        deriv_display["ma_slope_smooth_causal"] = _causal_savgol_series(
            slope, int(slope_smooth_window))
        deriv_display["ma_accel_smooth_causal"] = _causal_savgol_series(
            accel, int(accel_smooth_window))
        deriv_display["ma_jerk_smooth_causal"] = _causal_savgol_series(
            jerk, int(jerk_smooth_window))
        # 固定 MA10 斜率因果平滑（独立于上方 k 日均线参数）
        ma10 = ind["ma10"] if "ma10" in ind.columns else df["close"].rolling(10, min_periods=1).mean()
        deriv_display["ma10_slope_smooth_causal"] = _causal_savgol_series(
            ma10.diff(), int(slope_smooth_window))

    buys, sells = signal_points(df, signals)
    offset = _trade_marker_offset(df)

    n_trades = len(buys) + len(sells)
    if label_mode == "detailed":
        dense = False
    elif label_mode == "compact":
        dense = True
    else:  # auto
        dense = n_trades > _DENSE_TRADE_THRESHOLD
    show_labels = not dense
    show_lines = not dense
    marker_size = 11 if dense else 18

    # 布局：K 线与成交量同一坐标系（成交量叠在主图底部），额外副图指标再往下排。
    # 说明：Plotly 的 spike 竖线无法跨子图域；把成交量画进主图后，十字虚线才能盖住量柱。
    n_panels = len(panels)
    n_rows = 1 + n_panels
    if n_panels:
        price_h = 0.72
        panel_h = (1 - price_h) / n_panels
        row_heights = [price_h] + [panel_h] * n_panels
    else:
        row_heights = [1.0]
    subplot_titles = [title] + [_PANEL_INDICATORS[k] for k in panels]

    # 只有选了导数指标时才在主图开启右侧第二坐标轴，避免影响原有布局
    specs = [[{"secondary_y": bool(deriv_overlays)}]] + \
            [[{"secondary_y": False}] for _ in range(n_rows - 1)]
    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, row_heights=row_heights,
        vertical_spacing=0.04, subplot_titles=subplot_titles, specs=specs,
    )
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="K线", increasing_line_color=UP, decreasing_line_color=DOWN,
            hoverinfo="skip",  # 悬浮值改由顶部统一提示承载，避免大色块盖住曲线
        ),
        row=1, col=1,
    )
    for key in col_overlays:
        zh, color = _OVERLAY_INDICATORS[key]
        fig.add_trace(
            go.Scatter(x=ind.index, y=_smooth_series(ind[key], sm_m, sm_s),
                       name=zh, line=dict(width=1, color=color), hoverinfo="skip"),
            row=1, col=1,
        )
    # 导数指标（斜率/加速度/三阶导）画在主图右轴，围绕 0 波动；过 0 处即波峰/波谷
    _CAUSAL_DERIV = {"ma_slope_smooth_causal", "ma_accel_smooth_causal",
                    "ma_jerk_smooth_causal", "ma10_slope_smooth_causal"}
    _CENTER_SMOOTH_DERIV = {"ma_slope_smooth", "ma_accel_smooth", "ma_jerk_smooth"}
    for key in deriv_overlays:
        zh, color = _OVERLAY_INDICATORS[key]
        # 中心式平滑用实线加粗；因果平滑用虚线(实盘可见)；原始导数用点线，便于对照
        is_causal = key in _CAUSAL_DERIV
        is_smooth = key in _CENTER_SMOOTH_DERIV or is_causal
        line = dict(width=1.8 if is_smooth else 1.2, color=color,
                    dash="dash" if is_causal else ("solid" if is_smooth else "dot"))
        fig.add_trace(
            go.Scatter(x=df.index, y=deriv_display[key], name=zh,
                       line=line, hoverinfo="skip"),
            row=1, col=1, secondary_y=True,
        )
    if deriv_overlays:
        fig.update_yaxes(
            title_text="斜率 / 加速度 / 三阶导", row=1, col=1, secondary_y=True,
            showgrid=False, zeroline=True, zerolinewidth=1,
            zerolinecolor="rgba(148,163,184,0.6)",
            tickfont=dict(size=10, color="#94a3b8"),
        )
    # 详细模式给标签留出高度；上下留白按走势包络自适应，给悬浮框腾位置
    pad = offset * (2.6 if show_labels else 1.4)
    # B/S 色块引线长度：拉长到基准偏移的 5 倍，让色块远离 K 线、彼此不堆叠
    lead = offset * 5.0
    # 自适应包络：用居中滚动极值勾出走势的下沿/上沿，让悬浮框贴着局部曲线而非钉死在全局极值
    env_win = max(5, len(df) // 30)
    low_env = df["low"].rolling(env_win, min_periods=1, center=True).min()
    high_env = df["high"].rolling(env_win, min_periods=1, center=True).max()
    # 留白同时兼顾悬浮框与更长的引线色块，避免色块被坐标轴裁切
    bottom_margin = max(pad * 1.9, lead + offset * 1.6)
    top_margin = max(pad * 1.7, lead + offset * 1.6)
    y_lo = float((low_env - bottom_margin).min())
    y_hi = float((high_env + top_margin).max())
    fig.add_trace(
        go.Scatter(
            x=[df.index[0], df.index[-1]],
            y=[y_lo, y_hi],
            mode="markers", marker=dict(opacity=0, size=0),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    # 成交量叠在主图底部（按价格轴缩放），与 K 线同域 → 纵向虚线可覆盖量柱
    # 注意：Bar 默认从 0 起画，必须设 base=y_lo，否则会铺满整个价格区
    price_span = max(y_hi - y_lo, 1e-9)
    vol_band = price_span * 0.18
    vol = df["volume"].astype(float)
    vmax = float(vol.max()) if len(vol) and float(vol.max()) > 0 else 1.0
    vol_height = vol / vmax * vol_band
    vol_colors = [UP if c >= o else DOWN for o, c in zip(df["open"], df["close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=vol_height, base=y_lo, name="成交量",
               marker_color=vol_colors, hoverinfo="skip"),
        row=1, col=1,
    )
    # 统一提示框固定显示在图表最顶部（在所有 K 线上方），配合 hovermode="x" 横向偏移不挡走势。
    top_anchor = float(high_env.max()) + pad * 1.6
    _add_unified_hover(fig, df, ind, col_overlays, (sm_m, sm_s),
                       anchor_y=[top_anchor] * len(df),
                       buy_close=buys["close"].reindex(df.index),
                       sell_close=sells["close"].reindex(df.index))
    marker_kw = dict(offset=lead, show_labels=show_labels,
                     show_lines=show_lines, marker_size=marker_size)
    # 买入在 K 线正下方引出「B」色块、卖出在正上方引出「S」色块
    _add_trade_markers(fig, buys, side="buy", **marker_kw)
    _add_trade_markers(fig, sells, side="sell", **marker_kw)
    for i, key in enumerate(panels):
        _add_indicator_panel(fig, ind, key, row=2 + i, smooth=(sm_m, sm_s))
    # 图例贴着图表顶部、向上排一行；主标题再抬到图例上方（标题 → 图例 → 图表）
    fig.update_layout(
        height=560 + 150 * n_panels, xaxis_rangeslider_visible=False,
        hovermode="x",  # 单一顶部锚点提示；十字准线由 spikes 承担
        spikedistance=-1,  # 悬停时始终画出十字虚线
        hoverdistance=40,
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        margin=dict(t=80, b=16, l=8, r=8),
        hoverlabel=dict(align="left"),
    )
    # 竖线 + 横线十字虚线：与 K 线/成交量同域，可完整覆盖量柱并做水平比价
    # rangebreaks 挖掉双休/节假日空隙，让 K 线与曲线紧凑连在一起
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikecolor="#94a3b8", spikedash="dot",
                     rangebreaks=_x_rangebreaks(df.index))
    fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor",
                     spikethickness=1, spikecolor="#94a3b8", spikedash="dot",
                     row=1, col=1, secondary_y=False)
    # 主标题固定在图表顶部上方 46px（图例之上）；其余子图标题略微下移留白
    for ann in fig.layout.annotations:
        if ann.text == title:
            ann.update(y=1.0, yanchor="bottom", yshift=46)
        elif ann.text in subplot_titles and ann.text:
            ann.update(yshift=-4)
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
_RGB_RED, _RGB_YELLOW, _RGB_GREEN = (239, 68, 68), (234, 179, 8), (34, 197, 94)


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _grad_color(t: float, alpha: float = 0.35) -> str:
    """t∈[0,1] 沿 红→黄→绿 插值，越靠 1 越绿（越好）。"""
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        c = _lerp(_RGB_RED, _RGB_YELLOW, t / 0.5)
    else:
        c = _lerp(_RGB_YELLOW, _RGB_GREEN, (t - 0.5) / 0.5)
    return f"rgba({c[0]},{c[1]},{c[2]},{alpha})"


def _grad_steps(rng, good_at, n: int = 28) -> list[dict]:
    """把量程切成 n 段做平滑渐变：达标线以下红→黄，以上黄→绿。"""
    lo, hi = rng
    steps = []
    for i in range(n):
        a, b = i / n, (i + 1) / n
        x0, x1 = lo + (hi - lo) * a, lo + (hi - lo) * b
        mid = (x0 + x1) / 2
        # 以「达标线 good_at」为绿黄分界：达标处 t=0.5，量程两端为 0 / 1
        if mid <= good_at:
            t = 0.5 * (mid - lo) / (good_at - lo) if good_at > lo else 0.5
        else:
            t = 0.5 + 0.5 * (mid - good_at) / (hi - good_at) if hi > good_at else 1.0
        steps.append(dict(range=[x0, x1], color=_grad_color(t)))
    return steps


def _gauge(fig, row, col, value, rng, red_to, yellow_to):
    """渐变仪表盘：底色红→黄→绿表示越高越好，虚线为达标线，粗线为当前值。"""
    raw = value if value is not None and value == value and value != float("inf") else 0.0
    v = max(rng[0], min(rng[1], raw))
    num_color = _GAUGE_GREEN if raw >= yellow_to else (_GAUGE_YELLOW if raw >= red_to else _GAUGE_RED)
    eps = (rng[1] - rng[0]) * 0.006
    steps = _grad_steps(rng, yellow_to)
    # 在「达标线」处插一段不透明窄条，视觉上就是一条参考竖线
    steps.append(dict(range=[yellow_to - eps, yellow_to + eps],
                      color="rgba(226,232,240,0.9)"))
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
                steps=steps,
                threshold=dict(
                    line=dict(color=num_color, width=4),
                    thickness=0.82, value=v,
                ),
            ),
        ),
        row=row, col=col,
    )


def fig_ratio_gauges(m: dict) -> go.Figure:
    """夏普 / 索提诺 / 卡玛三个核心比率仪表盘，渐变色 + 达标参考一眼看好坏。"""
    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "indicator"}] * 3],
        subplot_titles=("夏普比率", "索提诺比率", "卡玛比率"),
        horizontal_spacing=0.08,
    )
    _gauge(fig, 1, 1, m.get("sharpe"), [-1, 3], 0, 1)
    _gauge(fig, 1, 2, m.get("sortino"), [-1, 3], 0, 1)
    _gauge(fig, 1, 3, m.get("calmar"), [0, 3], 0.5, 1)
    fig.update_layout(
        height=270, margin=dict(t=46, b=30, l=24, r=24),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
    )
    fig.update_annotations(font=dict(size=13, color="#cbd5e1"), yshift=6)
    # 每个表盘下方标注达标参考（越高越好；虚白线=达标线）
    refs = [
        (0.13, "≥1 良好 · ≥2 优秀"),
        (0.5, "≥1 良好 · ≥2 优秀"),
        (0.87, "≥1 良好 · ≥3 优秀"),
    ]
    for x, text in refs:
        fig.add_annotation(
            x=x, y=-0.06, xref="paper", yref="paper",
            text=text, showarrow=False,
            font=dict(size=11, color="#94a3b8"),
            xanchor="center", yanchor="top",
        )
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
        # 标题与图例同处顶部一行，且都排在绘图区上方的边距里（底端贴图顶、向上延伸），避免压住柱子
        title=dict(text="策略 vs 基准（%）", font=dict(size=13, color="#cbd5e1"),
                   x=0.02, xanchor="left", y=1.0, yanchor="bottom", yref="paper"),
        uniformtext=dict(mode="show", minsize=12),
        legend=dict(orientation="h", y=1.0, yanchor="bottom",
                    x=1.0, xanchor="right", font=dict(size=12, color="#cbd5e1"),
                    bgcolor="rgba(0,0,0,0)", itemwidth=30),
        margin=dict(t=34, b=6, l=6, r=24),
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

    # 顶部图形区：左仪表盘、右对比条形图，套一个外框并用竖线分隔两块
    top = st.container(border=True)
    top.markdown("###### 核心比率 & 策略对比")
    g1, gsep, g2 = top.columns([3, 0.08, 2])
    g1.plotly_chart(fig_ratio_gauges(m), use_container_width=True)
    gsep.markdown(
        '<div style="border-left:1px solid rgba(148,163,184,0.35);'
        'height:230px;margin:6px auto 0;"></div>',
        unsafe_allow_html=True,
    )
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


# ----------------------- 策略库：对比计算与可视化 -----------------------
_CMP_PALETTE = ["#3b82f6", "#f59e0b", "#a855f7", "#14b8a6", "#ef4444",
                "#84cc16", "#ec4899", "#64748b"]


@st.cache_data(show_spinner=False, ttl=1800)
def run_snapshot_backtest(symbol: str, strategy: str, params: dict,
                          source: str, start: str, end: str, init_cash: float):
    """用某快照的策略/参数在指定标的上回测，返回 (归一化净值Series, 指标dict)。"""
    df = cached_history(symbol, start, end, source)
    if df.empty:
        return None, None
    signals = compute_signals(df, strategy, params)
    res = backtest(df, signals, init_cash=init_cash)
    nav = res.equity / res.equity.iloc[0]  # 归一化到 1.0，便于跨标的/跨时段对比
    return nav, res.metrics


def _uniq_label(label: str, used: set[str]) -> str:
    """保证对比标签唯一：重名时自动加后缀，避免曲线被 dict 覆盖而与表格行数不一致。"""
    if label not in used:
        used.add(label)
        return label
    i = 2
    while f"{label} #{i}" in used:
        i += 1
    new = f"{label} #{i}"
    used.add(new)
    return new


def fig_compare_equity(series_map: dict[str, pd.Series]) -> go.Figure:
    """多条归一化净值曲线叠加对比。"""
    fig = go.Figure()
    for i, (label, nav) in enumerate(series_map.items()):
        color = _CMP_PALETTE[i % len(_CMP_PALETTE)]
        fig.add_trace(go.Scatter(x=nav.index, y=nav, name=label,
                                 line=dict(color=color, width=2)))
    fig.update_layout(
        height=380, title="净值对比（起点归一化为 1.0）", hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=44, b=10),
    )
    fig.update_yaxes(title="净值")
    return fig


_CMP_METRICS = [
    ("total_return", "总收益", "pct"),
    ("annual_return", "年化收益", "pct"),
    ("sharpe", "夏普", "num"),
    ("sortino", "索提诺", "num"),
    ("calmar", "卡玛", "num"),
    ("max_drawdown", "最大回撤", "pct"),
    ("win_rate", "胜率", "pct"),
    ("num_trades", "交易数", "int"),
]


def compare_metrics_table(rows: list[dict]) -> pd.DataFrame:
    """把多条快照的绩效整理成对比表（每行一个快照）。"""
    out = []
    for r in rows:
        m = r["metrics"]
        rec = {"对比项": r["label"]}
        for key, zh, kind in _CMP_METRICS:
            v = m.get(key)
            if v is None or v != v:
                rec[zh] = "-"
            elif kind == "pct":
                rec[zh] = f"{v * 100:.2f}%"
            elif kind == "int":
                rec[zh] = f"{int(v)}"
            else:
                rec[zh] = f"{v:.2f}"
        out.append(rec)
    return pd.DataFrame(out)


def fig_compare_metrics_bars(rows: list[dict]) -> go.Figure:
    """核心指标分组柱状对比：总收益/年化/最大回撤（%）与夏普。"""
    labels = [r["label"] for r in rows]
    total = [(r["metrics"].get("total_return") or 0) * 100 for r in rows]
    annual = [(r["metrics"].get("annual_return") or 0) * 100 for r in rows]
    mdd = [(r["metrics"].get("max_drawdown") or 0) * 100 for r in rows]
    sharpe = [(r["metrics"].get("sharpe") or 0) for r in rows]
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=("收益 / 回撤（%）", "夏普比率"),
    )
    fig.add_trace(go.Bar(x=labels, y=total, name="总收益", marker_color="#3b82f6"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=annual, name="年化收益", marker_color="#22c55e"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=mdd, name="最大回撤", marker_color="#ef4444"), row=1, col=1)
    fig.add_trace(go.Bar(x=labels, y=sharpe, name="夏普", marker_color="#f59e0b",
                         showlegend=False), row=1, col=2)
    fig.update_layout(
        height=340, barmode="group", margin=dict(t=48, b=10),
        legend=dict(orientation="h", y=1.14, x=0),
    )
    return fig


# ----------------------- 我的自选：迷你走势 & 批量回测 -----------------------
def fig_indicator_subplots(ind: pd.DataFrame) -> go.Figure:
    """MACD + RSI/KDJ 双子图，供技术指标页与自选详情复用。"""
    sub = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("MACD", "RSI / KDJ"))
    sub.add_trace(go.Bar(x=ind.index, y=ind["macd_hist"], name="MACD柱"), row=1, col=1)
    sub.add_trace(go.Scatter(x=ind.index, y=ind["macd_dif"], name="DIF"), row=1, col=1)
    sub.add_trace(go.Scatter(x=ind.index, y=ind["macd_dea"], name="DEA"), row=1, col=1)
    sub.add_trace(go.Scatter(x=ind.index, y=ind["rsi14"], name="RSI14"), row=2, col=1)
    if "kdj_k" in ind:
        sub.add_trace(go.Scatter(x=ind.index, y=ind["kdj_k"], name="KDJ-K"), row=2, col=1)
    sub.update_layout(height=420, hovermode="x unified", margin=dict(t=40, b=10))
    return sub


@st.cache_data(show_spinner=False, ttl=900)
def watchlist_snapshot(symbol: str, source: str) -> dict:
    """自选行的迷你走势 + 最新价/涨跌幅：用近半年日线（带缓存），最后一根即最新。"""
    end = dt.date.today().strftime("%Y-%m-%d")
    start = (dt.date.today() - dt.timedelta(days=180)).strftime("%Y-%m-%d")
    try:
        df = load_history(symbol, start=start, end=end, source=source)
    except Exception:  # noqa: BLE001
        return {"ok": False}
    if df is None or df.empty or "close" not in df:
        return {"ok": False}
    closes = df["close"].dropna()
    if closes.empty:
        return {"ok": False}
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
    change = (last / prev - 1) if prev else 0.0
    return {
        "ok": True,
        "last": last,
        "change": change,
        "spark": closes.iloc[-60:],   # 近 60 个交易日走势
        "date": closes.index[-1].strftime("%Y-%m-%d"),
    }


def fig_sparkline(spark: pd.Series, change: float) -> go.Figure:
    """一行内的极简走势图：涨红跌绿，无坐标轴、无交互。"""
    color = UP if change >= 0 else DOWN
    fill = UP_FILL if change >= 0 else DOWN_FILL
    fig = go.Figure(go.Scatter(
        x=list(range(len(spark))), y=[float(v) for v in spark.values],
        mode="lines", line=dict(color=color, width=1.8),
        fill="tozeroy", fillcolor=fill, hoverinfo="skip",
    ))
    lo, hi = float(spark.min()), float(spark.max())
    pad = (hi - lo) * 0.12 or (hi * 0.02 or 1.0)
    fig.update_layout(
        height=46, margin=dict(t=4, b=4, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True, range=[lo - pad, hi + pad]),
    )
    return fig


def render_stock_detail(sym: str, source: str, start: str, end: str) -> None:
    """自选详情：K线+均线/布林、MACD/RSI/KDJ 子图与最近指标数据。"""
    label = wl.label_of(source, sym)
    st.markdown(f"#### 📈 {label} · 详细走势")
    try:
        df = cached_history(sym, start, end, source)
    except Exception as e:  # noqa: BLE001
        st.error(f"加载失败：{type(e).__name__}: {e}")
        return
    if df is None or df.empty:
        st.warning("这段区间没有数据，换个时间段试试。")
        return
    ind = add_indicators(df)
    st.plotly_chart(
        fig_kline(df, pd.Series(0, index=df.index), f"{label} K线 + 均线/布林"),
        use_container_width=True, key=f"detail_kline_{source}_{sym}",
    )
    st.plotly_chart(fig_indicator_subplots(ind), use_container_width=True,
                    key=f"detail_ind_{source}_{sym}")
    with st.expander("最近 10 行指标数据"):
        st.dataframe(ind.tail(10), use_container_width=True)


@st.cache_data(show_spinner=False, ttl=1800)
def run_batch_backtest(symbol: str, strategy: str, params: dict,
                       source: str, start: str, end: str, init_cash: float):
    """在单只标的上按给定策略回测，返回 (策略指标, 基准指标)；无数据返回 None。"""
    df = cached_history(symbol, start, end, source)
    if df is None or df.empty:
        return None
    signals = compute_signals(df, strategy, params)
    res = backtest(df, signals, init_cash=init_cash)
    return {"metrics": res.metrics, "bench": res.metrics_benchmark}


def fig_batch_returns(rows: list[dict]) -> go.Figure:
    """各自选股策略总收益的水平条形图：按收益排序，涨红跌绿。"""
    data = sorted(rows, key=lambda r: r["metrics"].get("total_return") or 0)
    labels = [f"{r['name']}（{r['symbol']}）" if r["name"] != r["symbol"] else r["symbol"]
              for r in data]
    vals = [(r["metrics"].get("total_return") or 0) * 100 for r in data]
    colors = [UP if v >= 0 else DOWN for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.1f}%" for v in vals], textposition="outside", cliponaxis=False,
    ))
    lo, hi = min(vals + [0]), max(vals + [0])
    pad = max((hi - lo) * 0.18, 3)
    fig.update_layout(
        height=max(260, 28 * len(vals) + 90),
        title=dict(text="各自选股策略总收益（%）", font=dict(size=13)),
        margin=dict(t=44, b=10, l=8, r=44), bargap=0.3,
    )
    fig.update_xaxes(range=[lo - pad, hi + pad], zeroline=False)
    fig.add_vline(x=0, line_width=1, line_color="rgba(148,163,184,0.4)")
    return fig


def render_batch_result(data: dict) -> None:
    """展示批量回测的整体聚合指标 + 分标的明细 + 收益条形图。"""
    import statistics

    rows = data["rows"]
    n = len(rows)

    def g(r, key):
        return r["metrics"].get(key) or 0

    rets = [g(r, "total_return") for r in rows]
    annuals = [g(r, "annual_return") for r in rows]
    mdds = [g(r, "max_drawdown") for r in rows]
    sharpes = [g(r, "sharpe") for r in rows]
    wins = [g(r, "win_rate") for r in rows]
    excess = [g(r, "total_return") - (r["bench"].get("total_return") or 0) for r in rows]
    profitable = sum(1 for x in rets if x > 0)
    beat = sum(1 for e in excess if e > 0)

    def avg(a):
        return sum(a) / len(a) if a else 0.0

    st.markdown(
        f"**策略：{data['label']}** · 区间 {data['start']} ~ {data['end']} · 共 {n} 只自选股")

    box = st.container(border=True)
    box.markdown("###### 整体表现（策略在自选股上的普适性）")
    _card_row(box, [
        ("回测股票数", f"{n}"),
        ("盈利占比", f"{profitable / n * 100:.0f}%", f"{profitable}/{n} 只盈利"),
        ("跑赢基准占比", f"{beat / n * 100:.0f}%", f"{beat}/{n} 只跑赢"),
        ("平均总收益", f"{avg(rets) * 100:.2f}%"),
        ("收益中位数", f"{statistics.median(rets) * 100:.2f}%"),
    ])

    box2 = st.container(border=True)
    box2.markdown("###### 收益与风险（自选股平均）")
    worst_dd = min(mdds) if mdds else 0.0
    _card_row(box2, [
        ("平均年化", f"{avg(annuals) * 100:.2f}%"),
        ("平均最大回撤", f"{avg(mdds) * 100:.2f}%"),
        ("最差回撤", f"{worst_dd * 100:.2f}%"),
        ("平均夏普", f"{avg(sharpes):.2f}"),
        ("平均胜率", f"{avg(wins) * 100:.1f}%"),
    ])

    st.markdown("###### 分标的明细")
    table = pd.DataFrame([{
        "名称": r["name"],
        "代码": r["symbol"],
        "总收益": f"{g(r, 'total_return') * 100:.2f}%",
        "年化": f"{g(r, 'annual_return') * 100:.2f}%",
        "夏普": f"{g(r, 'sharpe'):.2f}",
        "最大回撤": f"{g(r, 'max_drawdown') * 100:.2f}%",
        "胜率": f"{g(r, 'win_rate') * 100:.1f}%",
        "交易数": int(g(r, "num_trades")),
        "基准收益": f"{(r['bench'].get('total_return') or 0) * 100:.2f}%",
        "超额": f"{(g(r, 'total_return') - (r['bench'].get('total_return') or 0)) * 100:+.2f}%",
    } for r in rows])
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.plotly_chart(fig_batch_returns(rows), use_container_width=True, key="batch_ret_bar")


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
# 策略库「应用到股票」后，回测结果已写入 session；在页首给出跳转提示
_applied_msg = st.session_state.pop("lib_applied_msg", None)
if _applied_msg:
    st.success(_applied_msg, icon="✅")

tab_watch, tab_bt, tab_batch, tab_ind, tab_lib, tab_paper, tab_quote = st.tabs(
    ["⭐ 我的自选", "🔬 回测", "🎯 批量回测", "📊 技术指标",
     "📚 策略库", "💼 模拟盘", "⚡ 实时行情"])

# ===== 我的自选 =====
with tab_watch:
    st.subheader("⭐ 我的自选")
    watch_items = wl.pool(source)
    if not watch_items:
        st.info("自选列表是空的。用左侧「股票池 · 搜索添加」加入自选，或在「管理股票池」里手动加代码。")
    else:
        top1, top2 = st.columns([6, 1])
        top1.caption(
            f"共 {len(watch_items)} 只 · 数据源 {source}"
            f"（{'A股' if source == 'akshare' else '美股'}）· 点「详情」看完整 K 线与指标")
        if top2.button("🔄 刷新", key="watch_refresh"):
            watchlist_snapshot.clear()
            st.rerun()

        head = st.columns([3, 3, 2, 2, 1.2])
        for col, txt in zip(head, ["股票", "近 60 日走势", "最新价", "涨跌幅", "操作"]):
            col.markdown(f"<span style='color:#64748b;font-size:0.8rem'>{txt}</span>",
                         unsafe_allow_html=True)

        for x in watch_items:
            sym = x["symbol"]
            name = x.get("name") or sym
            snap = watchlist_snapshot(sym, source)
            row = st.columns([3, 3, 2, 2, 1.2])
            row[0].markdown(
                f"<div style='font-weight:600;font-size:0.98rem'>{name}</div>"
                f"<span style='color:#94a3b8;font-size:0.78rem'>{sym}</span>",
                unsafe_allow_html=True,
            )
            if snap.get("ok"):
                row[1].plotly_chart(
                    fig_sparkline(snap["spark"], snap["change"]),
                    use_container_width=True, config={"displayModeBar": False},
                    key=f"spark_{source}_{sym}",
                )
                color = UP if snap["change"] >= 0 else DOWN
                arrow = "▲" if snap["change"] >= 0 else "▼"
                row[2].markdown(
                    f"<div style='font-size:1.02rem;font-weight:600;color:{color};"
                    f"padding-top:8px'>{snap['last']:.2f}</div>",
                    unsafe_allow_html=True,
                )
                row[3].markdown(
                    f"<div style='font-size:1.0rem;font-weight:600;color:{color};"
                    f"padding-top:8px'>{arrow} {snap['change'] * 100:+.2f}%</div>",
                    unsafe_allow_html=True,
                )
            else:
                row[1].caption("暂无数据")
                row[2].caption("-")
                row[3].caption("-")
            if row[4].button("详情", key=f"watch_detail_btn_{source}_{sym}"):
                st.session_state["watch_detail"] = sym
                st.session_state["watch_detail_source"] = source

        detail_sym = st.session_state.get("watch_detail")
        pool_syms = {i["symbol"] for i in watch_items}
        if (detail_sym and detail_sym in pool_syms
                and st.session_state.get("watch_detail_source") == source):
            st.divider()
            render_stock_detail(detail_sym, source, str(start), str(end))

# ===== 批量回测 =====
with tab_batch:
    st.subheader("🎯 我的自选 · 批量回测")
    st.caption("选一套策略，在自选股票上批量回测，一眼看清整体成功率、收益与回撤，验证策略是否普适。")
    batch_items = wl.pool(source)
    if not batch_items:
        st.info("自选列表是空的，先在左侧添加股票再来批量回测。")
    else:
        strat_mode = st.radio("策略来源", ["当前侧栏策略", "我的策略库快照"],
                              horizontal=True, key="batch_strat_mode")
        run_strategy, run_params = strategy, params
        run_label = f"{strategy}（侧栏参数）"
        ready = True
        if strat_mode == "我的策略库快照":
            snaps = sl.list_all()
            if not snaps:
                st.warning("策略库还没有快照，先到「📚 策略库」保存一条，或改用「当前侧栏策略」。")
                ready = False
            else:
                def _p(p):
                    return ", ".join(f"{k}={v}" for k, v in p.items()) or "无参数"

                smap = {f"{s['name']} · {s['strategy']}（{_p(s['params'])}）": s for s in snaps}
                pick = st.selectbox("选择我的策略快照", list(smap), key="batch_snap_pick")
                sp = smap[pick]
                run_strategy, run_params = sp["strategy"], sp["params"]
                run_label = f"{sp['name']} · {sp['strategy']}"

        all_syms = [x["symbol"] for x in batch_items]
        chosen_syms = st.multiselect(
            "回测标的（默认全部自选）", all_syms, default=all_syms,
            format_func=lambda s: wl.label_of(source, s), key="batch_syms",
        )

        if st.button("🚀 运行批量回测", type="primary", key="batch_run", disabled=not ready):
            if not chosen_syms:
                st.warning("请至少选择一只自选股票。")
            else:
                result_rows = []
                n = len(chosen_syms)
                progress = st.progress(0.0, text="准备回测…")
                for i, sym in enumerate(chosen_syms):
                    progress.progress(i / n, text=f"回测 {sym}（{i + 1}/{n}）…")
                    try:
                        r = run_batch_backtest(
                            sym, run_strategy, run_params, source,
                            str(start), str(end), float(init_cash))
                    except Exception as e:  # noqa: BLE001
                        st.error(f"{sym} 回测失败：{type(e).__name__}: {e}")
                        continue
                    if not r:
                        st.warning(f"{sym} 无数据，跳过")
                        continue
                    result_rows.append({
                        "symbol": sym,
                        "name": cached_stock_name(sym, source),
                        "metrics": r["metrics"],
                        "bench": r["bench"],
                    })
                progress.progress(1.0, text="回测完成")
                if result_rows:
                    st.session_state["batch_result"] = {
                        "rows": result_rows, "label": run_label,
                        "start": str(start), "end": str(end),
                    }
                else:
                    st.session_state.pop("batch_result", None)
                    st.warning("没有成功回测的标的，请检查代码或网络后重试。")

        if st.session_state.get("batch_result"):
            render_batch_result(st.session_state["batch_result"])
        else:
            st.info("选好策略与标的后，点「运行批量回测」。")

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
            st.session_state["bt"] = {
                "rows": rows,
                "primary": primary_ctx,
                # 记录本次回测所用配置，保存快照时以此为准（避免与事后改动的侧栏不一致）
                "config": {
                    "symbols": list(symbols),
                    "strategy": strategy,
                    "params": dict(params),
                    "source": source,
                    "start": str(start),
                    "end": str(end),
                    "init_cash": float(init_cash),
                },
            }
        else:
            st.warning("没有成功回测的标的，请检查代码或网络后重试。")

    if "bt" in st.session_state:
        data = st.session_state["bt"]
        st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True, hide_index=True)

        with st.expander("⭐ 把当前回测存为策略快照（便于以后复用 / 对比）", expanded=False):
            # 以「本次回测所用配置」为准；兼容没有 config 的旧 session
            cfg = data.get("config") or {
                "symbols": symbols, "strategy": strategy, "params": params,
                "source": source, "start": str(start), "end": str(end),
                "init_cash": init_cash,
            }
            st.caption(
                f"将保存本次回测配置：{cfg['strategy']} · "
                f"{'/'.join(cfg['symbols'])} · {cfg['start']}~{cfg['end']}，及主标的绩效。")
            default_name = f"{cfg['strategy']}·{'/'.join(cfg['symbols'])}"
            snap_name = st.text_input("快照名称", value=default_name, key="snap_name_bt")
            snap_metrics = {}
            if data["primary"]:
                snap_metrics = (data["primary"][4] if len(data["primary"]) == 5
                                else data["primary"][3]).metrics
            if st.button("保存到策略库", key="save_snap_bt"):
                item = sl.add(
                    name=snap_name, symbols=cfg["symbols"], strategy=cfg["strategy"],
                    params=cfg["params"], source=cfg["source"], start=cfg["start"],
                    end=cfg["end"], init_cash=cfg["init_cash"], metrics=snap_metrics,
                )
                st.success(f"已保存快照「{item['name']}」到策略库（📚 策略库 页签查看）")

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
            log = trade_log(df, signals)
            mode_labels = {
                "auto": "自动（点多自动精简）",
                "detailed": "详细（价格标签+指引线）",
                "compact": "精简（只显示三角标）",
            }
            mc1, mc2 = st.columns([2, 3])
            with mc1:
                label_mode = st.radio(
                    "买卖点显示", list(mode_labels), index=0, horizontal=True,
                    format_func=lambda k: mode_labels[k], key="kline_label_mode",
                    help="交易频繁时选「精简」或「自动」，图更清爽；具体价格看下方明细或悬停。",
                )
            with mc2:
                selected_inds = st.multiselect(
                    "添加指标", indicator_options(),
                    default=["ma5", "ma_slope_smooth", "ma_accel_smooth"],
                    format_func=indicator_label, key="kline_indicators_v2",
                    help="主图指标（均线/布林/斜率/加速度/三阶导）叠加在 K 线上；副图指标（MACD/RSI/KDJ 等）单独成图。",
                )
            overlays = [k for k in selected_inds if k in _OVERLAY_INDICATORS]
            panels = [k for k in selected_inds if k in _PANEL_INDICATORS]
            sm1, sm2, sm3, sm4, sm5, sm6 = st.columns([2, 2, 1, 1, 1, 1])
            with sm1:
                smooth_method = st.selectbox(
                    "曲线平滑（滤波）", list(_SMOOTH_METHODS),
                    format_func=smooth_method_label, key="kline_smooth_method",
                    help="对各指标曲线降噪，减少高频抖动；K 线与成交量不受影响。",
                )
            with sm2:
                smooth_strength = st.slider(
                    "平滑强度（窗口 / 跨度）", 2, 30, 5, key="kline_smooth_strength",
                    disabled=(smooth_method == "none"),
                    help="值越大越平滑，但对拐点的滞后也越大。",
                )
            with sm3:
                deriv_k = st.number_input(
                    "导数 k日均线", min_value=1, max_value=120, value=5, step=1,
                    key="kline_deriv_k",
                    help="计算「MA斜率/加速度/三阶导」及其平滑版所用的均线周期，默认 5。",
                )
            with sm4:
                slope_smooth_win = st.number_input(
                    "斜率平滑窗口", min_value=3, max_value=61, value=11, step=2,
                    key="kline_slope_sw",
                    help="「MA斜率(平滑)」的保形(SavGol)滤波窗口，越大越平滑，默认 11（值小更贴合真实斜率）。",
                )
            with sm5:
                accel_smooth_win = st.number_input(
                    "加速度平滑窗口", min_value=3, max_value=61, value=11, step=2,
                    key="kline_accel_sw2",
                    help="「MA加速度(平滑)」的保形(SavGol)滤波窗口，与斜率同方案，默认 11。",
                )
            with sm6:
                jerk_smooth_win = st.number_input(
                    "三阶导平滑窗口", min_value=3, max_value=61, value=11, step=2,
                    key="kline_jerk_sw",
                    help="「MA三阶导(平滑·因果)」的保形(SavGol)滤波窗口，与加速度同方案，默认 11。",
                )
            st.plotly_chart(
                fig_kline(df, signals, f"{label} K线 + 买卖点", label_mode=label_mode,
                          overlays=overlays, panels=panels,
                          smooth=(smooth_method, smooth_strength),
                          deriv_ma_window=int(deriv_k),
                          slope_smooth_window=int(slope_smooth_win),
                          accel_smooth_window=int(accel_smooth_win),
                          jerk_smooth_window=int(jerk_smooth_win)),
                use_container_width=True,
            )
            if log.empty:
                st.caption("这段区间没有成交。")
            else:
                st.caption("三角标在影线外侧（买在下、卖在上）；悬停看成交价，密集时用下方明细核对。")
                n_trades = log["交易"].nunique()
                with st.expander(f"买卖点明细（{len(log)} 笔成交 · {n_trades} 组交易）", expanded=False):
                    st.caption("同一底色为一笔完整交易（买入→卖出）；收益红为盈、绿为亏。")
                    st.dataframe(style_trade_log(log), use_container_width=True, hide_index=True)
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
            st.plotly_chart(fig_indicator_subplots(ind), use_container_width=True)
            with st.expander("最近 10 行指标数据"):
                st.dataframe(ind.tail(10), use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"加载失败：{type(e).__name__}: {e}")

# ===== 策略库 =====
def _apply_snapshot_to_backtest(s: dict, target: str) -> None:
    """把某快照的策略/参数/区间套用到输入的股票上回测，结果写入「🔬 回测」页。"""
    sym = (target or "").strip()
    if not sym:
        st.warning("请先输入股票代码再点「应用回测」。")
        return
    src = s["source"]
    if src == "akshare" and sym.isdigit():
        sym = sym.zfill(6)
    elif src == "yfinance":
        sym = sym.upper()
    try:
        df = cached_history(sym, s["start"], s["end"], src)
    except Exception as e:  # noqa: BLE001
        st.error(f"{sym} 获取行情失败：{type(e).__name__}: {e}")
        return
    if df is None or df.empty:
        st.warning(f"{sym} 在 {s['start']} ~ {s['end']} 无数据，换个代码或时间段试试。")
        return
    signals = compute_signals(df, s["strategy"], s["params"])
    res = backtest(df, signals, init_cash=s["init_cash"])
    m = res.metrics
    name = cached_stock_name(sym, src)
    st.session_state["bt"] = {
        "rows": [{
            "名称": name,
            "股票": sym,
            "总收益": f"{m['total_return'] * 100:.2f}%",
            "年化": f"{m['annual_return'] * 100:.2f}%",
            "夏普": f"{m['sharpe']:.2f}",
            "最大回撤": f"{m['max_drawdown'] * 100:.2f}%",
            "胜率": f"{m['win_rate'] * 100:.1f}%",
            "交易数": m["num_trades"],
            "基准总收益": f"{res.metrics_benchmark['total_return'] * 100:.2f}%",
        }],
        "primary": (sym, name, df, signals, res),
        "config": {
            "symbols": [sym],
            "strategy": s["strategy"],
            "params": dict(s["params"]),
            "source": src,
            "start": s["start"],
            "end": s["end"],
            "init_cash": s["init_cash"],
        },
    }
    st.session_state["lib_applied_msg"] = (
        f"已用快照「{s['name']}」（{s['strategy']}）在 {name}（{sym}）上完成回测，"
        f"请切到「🔬 回测」页签查看 K 线、资金曲线与完整指标。")
    st.rerun()


def _lib_params_str(p: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items()) or "无参数"


def _render_lib_snapshot(s: dict) -> None:
    """在策略分组内渲染一条快照：以保存时间命名，可展开看参数并应用到股票。"""
    sid = s["id"]
    card = st.container(border=True)
    head = card.columns([5, 1])
    head[0].markdown(f"**🕒 {s['created_at']}**  ·  {s['name']}")
    head[0].caption(
        f"标的 {'/'.join(s['symbols']) or '未选'} · 数据源 {s['source']} · 区间 {s['start']} ~ {s['end']}")
    if not head[1].toggle("展开", key=f"lib_open_{sid}"):
        return

    card.markdown(f"**参数**：`{_lib_params_str(s['params'])}`")
    m = s.get("metrics") or {}
    if m.get("total_return") is not None:
        parts = [f"总收益 {m['total_return'] * 100:.2f}%"]
        if m.get("annual_return") is not None:
            parts.append(f"年化 {m['annual_return'] * 100:.2f}%")
        if m.get("sharpe") is not None:
            parts.append(f"夏普 {m['sharpe']:.2f}")
        if m.get("max_drawdown") is not None:
            parts.append(f"最大回撤 {m['max_drawdown'] * 100:.2f}%")
        card.caption("保存时绩效： " + " · ".join(parts))

    card.markdown("**应用到股票**")
    ap = card.columns([3, 1])
    default_sym = s["symbols"][0] if s["symbols"] else ""
    target = ap[0].text_input(
        "股票代码", value=default_sym, key=f"lib_apply_sym_{sid}",
        placeholder="如 000001 或 AAPL", label_visibility="collapsed",
        help=f"用数据源 {s['source']} 的行情，按此快照的策略/参数/区间回测后在「🔬 回测」页展示。",
    )
    ap[1].markdown("<div style='height:0.1rem'></div>", unsafe_allow_html=True)
    if ap[1].button("应用回测", key=f"lib_apply_btn_{sid}", type="primary",
                    use_container_width=True):
        _apply_snapshot_to_backtest(s, target)

    ops = card.columns([3, 1, 1])
    new_name = ops[0].text_input(
        "重命名", value=s["name"], key=f"lib_rename_in_{sid}", label_visibility="collapsed")
    if ops[1].button("保存名称", key=f"lib_rename_btn_{sid}", use_container_width=True):
        sl.rename(sid, new_name)
        st.rerun()
    if ops[2].button("🗑 删除", key=f"lib_del_btn_{sid}", use_container_width=True):
        sl.remove(sid)
        st.rerun()


with tab_lib:
    st.subheader("📚 策略库")
    st.caption("按策略分组管理你保存的参数快照：展开策略看历史快照（按保存时间命名），"
               "点开快照查看参数，输入股票代码「应用回测」即可在「🔬 回测」页看结果。")
    snaps = sl.list_all()

    def _snap_label(s: dict) -> str:
        return f"{s['name']} · {s['strategy']} · {'/'.join(s['symbols'])}（{s['created_at']}）"

    # ---------- A. 按策略分组列出全部快照 ----------
    if not snaps:
        st.info("还没有快照。去「🔬 回测」跑一次，展开底部「⭐ 存为策略快照」即可保存。")
    else:
        groups: dict[str, list[dict]] = {}
        for s in snaps:
            groups.setdefault(s["strategy"], []).append(s)  # list_all 已按时间倒序
        st.markdown(f"###### 共 {len(groups)} 种策略 · {len(snaps)} 个快照")
        for strat_name in sorted(groups):
            items = groups[strat_name]
            desc = infos.get(strat_name, {}).get("description", "")
            title = f"🧩 {strat_name}" + (f" · {desc}" if desc else "") + f"（{len(items)} 个快照）"
            with st.expander(title, expanded=False):
                for snap in items:
                    _render_lib_snapshot(snap)

    # ---------- C. 效果对比 ----------
    with st.expander("📊 效果对比：多套策略 / 多时期同台对比", expanded=False):
        if len(snaps) < 1:
            st.info("先保存至少一条快照。")
        else:
            lib_syms = sl.symbols_in_library()
            # 库里标的可能因删除而变化，先清理筛选器里的陈旧值
            if "lib_cmp_filter" in st.session_state:
                st.session_state["lib_cmp_filter"] = [
                    x for x in st.session_state["lib_cmp_filter"] if x in lib_syms]
            sym_filter = st.multiselect(
                "（可选）先按标的筛选快照", lib_syms, key="lib_cmp_filter",
                help="例如只看某只股票下、你不同时期存的多套策略。")
            pool = [s for s in snaps
                    if not sym_filter or any(x in sym_filter for x in s["symbols"])]
            c_map = {_snap_label(s): s for s in pool}
            # 用 session_state 预置默认并清理陈旧值（筛选/删除会改变可选项），避免与 default 冲突或报错
            if "lib_cmp_pick" not in st.session_state:
                st.session_state["lib_cmp_pick"] = list(c_map)[:min(3, len(c_map))]
            else:
                st.session_state["lib_cmp_pick"] = [
                    x for x in st.session_state["lib_cmp_pick"] if x in c_map]
            chosen = st.multiselect("选择要对比的快照（可多选）", list(c_map), key="lib_cmp_pick")
            override = st.text_input(
                "（可选）统一在此标的上对比，留空则各用快照自身标的", value="",
                key="lib_cmp_override",
                help="填一个代码，就把所选每套策略都在这只股票上回测后对比。")
            if st.button("运行对比", key="lib_cmp_run", type="primary"):
                series, rows, used = {}, [], set()
                for lab in chosen:
                    s = c_map[lab]
                    syms = [override.strip()] if override.strip() else s["symbols"]
                    for sym in syms:
                        try:
                            nav, met = run_snapshot_backtest(
                                sym, s["strategy"], s["params"], s["source"],
                                s["start"], s["end"], s["init_cash"])
                        except Exception as e:  # noqa: BLE001
                            st.error(f"{s['name']}@{sym} 回测失败：{type(e).__name__}: {e}")
                            continue
                        if nav is None:
                            st.warning(f"{s['name']}@{sym} 无数据")
                            continue
                        multi = bool(override.strip()) or len(syms) > 1
                        lbl = _uniq_label(f"{s['name']}@{sym}" if multi else s["name"], used)
                        series[lbl] = nav
                        rows.append({"label": lbl, "metrics": met})
                if series:
                    st.plotly_chart(fig_compare_equity(series), use_container_width=True)
                    st.plotly_chart(fig_compare_metrics_bars(rows), use_container_width=True)
                    st.markdown("###### 绩效对比明细")
                    st.dataframe(compare_metrics_table(rows), use_container_width=True, hide_index=True)
                else:
                    st.warning("没有可展示的结果，检查所选快照或标的。")

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
