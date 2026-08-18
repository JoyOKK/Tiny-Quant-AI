"""策略抽象基类。

策略只做一件事：把带指标的行情 DataFrame 转成"目标仓位"信号 Series。
    信号取值约定：1 = 满仓持有，0 = 空仓。
回测引擎与模拟盘据此驱动交易，策略本身不关心资金/手续费。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..indicators import add_indicators


class Strategy(ABC):
    #: 策略唯一名称（由 @register 装饰器注入）
    name: str = "base"
    #: 人类可读的策略描述
    description: str = ""

    def __init__(self, **params):
        # 用默认参数兜底，再用外部传入参数覆盖
        merged = dict(self.default_params())
        merged.update({k: v for k, v in params.items() if v is not None})
        self.params = merged

    @classmethod
    def default_params(cls) -> dict:
        """子类可覆盖，返回默认超参数。"""
        return {}

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """默认给行情追加常用指标；子类可覆盖。"""
        return add_indicators(df)

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """输入带指标的行情，输出目标仓位信号（0/1）。"""

    def run(self, df: pd.DataFrame) -> pd.Series:
        """对外统一入口：prepare + generate_signals。"""
        prepared = self.prepare(df)
        signals = self.generate_signals(prepared)
        return signals.reindex(df.index).fillna(0).clip(0, 1)
