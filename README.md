# Tiny-Quant-AI

一个**最小可用、组件可插拔**的 AI 量化交易平台，覆盖「数据 → 指标 → 策略 → 回测 → 模拟盘」完整闭环。支持 A 股（akshare）与美股（yfinance），策略以插件形式随时接入。

## 功能特性

1. **实时行情**：统一数据源接口，A 股用 akshare（免费无需 key），美股用 yfinance，历史行情自动 SQLite 缓存。
2. **技术指标**：纯 pandas 实现 MA/EMA/MACD/RSI/BOLL/KDJ/ATR/动量等，无需安装 TA-Lib。
3. **可插拔策略**：注册表 + 自动发现机制。新增策略只需在 `tinyquant/strategies/` 放一个文件，加 `@register("名字")` 装饰器即可，零侵入。内置双均线交叉（技术指标）与随机森林（AI/机器学习）两个示例策略。
4. **回测**：向量化回测引擎，次日成交避免未来函数，输出总收益/年化/夏普/最大回撤/卡玛/胜率，并与买入持有基准对比。支持单只/多只股票。
5. **模拟盘**：用真实行情 + 虚拟资金跟踪策略每日表现，账户状态持久化，方便每天观察策略收益。
6. **交互式看板**：基于 Streamlit + Plotly 的网页 Dashboard，浏览器里选股票/策略/参数，实时看回测指标、K线买卖点、资金曲线、回撤、模拟盘持仓与收益曲线。

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

```bash
# 1. 查看可用策略
python main.py strategies

# 2. 实时行情 / 技术指标
python main.py quote 000001
python main.py indicators 000001 --start 2024-01-01

# 3. 回测（A股双均线）
python main.py backtest 000001 --strategy ma_cross --start 2023-01-01

# 3b. 回测（美股随机森林），可多只、可绘图、可传参
python main.py backtest AAPL --source yfinance --strategy ml_rf --start 2022-01-01 --plot
python main.py backtest 000001,600519 --strategy ma_cross --param fast=10 slow=30

# 4. 模拟盘：初始化 -> 每日运行 -> 查看收益
python main.py paper init --strategy ma_cross --symbols 000001,600519
python main.py paper run          # 建议每个交易日收盘后跑一次
python main.py paper status

# 5. 交互式网页看板（推荐）
python main.py dashboard          # 等价于 streamlit run dashboard.py
```

## 交互式看板

```bash
python main.py dashboard          # 浏览器打开 http://localhost:8501
```

看板包含 4 个页签：

- **回测**：选股票/策略/参数一键回测，展示指标卡片、多标的对比表、K线+买卖点、资金曲线、回撤曲线；
- **技术指标**：K线叠加均线/布林 + MACD/RSI/KDJ 子图；
- **模拟盘**：一键初始化/调仓/刷新，查看持仓、权益曲线、成交记录；
- **实时行情**：批量查看最新报价与涨跌幅。

## 目录结构

```
Tiny-Quant-AI/
├── main.py                 # CLI 入口
├── dashboard.py            # Streamlit 交互式看板
├── config.py               # 全局配置（资金/手续费/路径/默认数据源）
├── requirements.txt
└── tinyquant/
    ├── data/               # 数据源：抽象接口 + akshare/yfinance + SQLite 缓存
    ├── indicators/         # 技术指标
    ├── strategies/         # 可插拔策略（base + registry + 各策略文件）
    ├── backtest/           # 回测引擎 + 绩效指标
    └── trading/            # 模拟盘经纪商
```

## 如何新增一个策略（插件）

在 `tinyquant/strategies/` 下新建 `my_strategy.py`：

```python
from .base import Strategy
from .registry import register

@register("my_strategy")
class MyStrategy(Strategy):
    description = "我的策略"

    @classmethod
    def default_params(cls):
        return {"threshold": 30}

    def generate_signals(self, df):
        # df 已带指标列，返回 0/1 目标仓位 Series
        return (df["rsi14"] < self.params["threshold"]).astype(float)
```

保存即可，`python main.py strategies` 会自动发现它，回测/模拟盘可直接用 `--strategy my_strategy` 调用。

## 说明与免责

- 回测采用「收盘产生信号、次日按收盘成交」的简化模型，含手续费不含滑点/涨跌停/停牌等细节，结果仅供策略相对比较。
- ML 策略为示例，使用训练/预测时间切分避免未来函数，但特征与调参都很基础，**不构成任何投资建议**。
