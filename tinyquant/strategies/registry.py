"""策略注册表：实现可插拔机制。

新增策略的方式：在 strategies/ 目录下新建一个 .py 文件，
定义一个继承 Strategy 的类并加上 @register("your_name") 装饰器即可，
无需改动任何其它代码——启动时会自动发现并加载。
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Type

from .base import Strategy

_REGISTRY: dict[str, Type[Strategy]] = {}


def register(name: str):
    """类装饰器：把策略登记到全局注册表。"""

    def deco(cls: Type[Strategy]):
        if not issubclass(cls, Strategy):
            raise TypeError(f"{cls} 必须继承 Strategy")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def discover() -> None:
    """自动导入 strategies 包下的所有模块，触发注册。"""
    import tinyquant.strategies as pkg

    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.name in {"base", "registry"}:
            continue
        importlib.import_module(f"{pkg.__name__}.{mod.name}")


def get_strategy(name: str, **params) -> Strategy:
    if not _REGISTRY:
        discover()
    if name not in _REGISTRY:
        raise ValueError(f"未知策略: {name}，可用: {list_strategies()}")
    return _REGISTRY[name](**params)


def list_strategies() -> list[str]:
    if not _REGISTRY:
        discover()
    return sorted(_REGISTRY)


def strategy_info() -> list[dict]:
    if not _REGISTRY:
        discover()
    return [
        {
            "name": name,
            "description": cls.description,
            "params": cls.default_params(),
        }
        for name, cls in sorted(_REGISTRY.items())
    ]
