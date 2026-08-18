"""全局配置。"""
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent

# 数据缓存目录（历史行情 SQLite、模拟盘账户等）
DATA_DIR = ROOT_DIR / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

# 行情缓存 SQLite
DB_PATH = DATA_DIR / "market.db"

# 模拟盘账户状态文件
PAPER_ACCOUNT_PATH = DATA_DIR / "paper_account.json"

# 默认数据源: "akshare"(A股) 或 "yfinance"(美股)
DEFAULT_SOURCE = "akshare"

# 回测 / 模拟盘默认参数
INIT_CASH = 100_000.0      # 初始资金
COMMISSION = 0.0003        # 单边手续费率（万三）
SLIPPAGE = 0.0             # 滑点
