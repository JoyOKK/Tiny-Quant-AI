"""可插拔策略框架。"""
from .base import Strategy
from .registry import (
    register,
    discover,
    get_strategy,
    list_strategies,
    strategy_info,
)

__all__ = [
    "Strategy",
    "register",
    "discover",
    "get_strategy",
    "list_strategies",
    "strategy_info",
]
