"""策略快照库：把「股票 + 策略 + 参数 + 绩效」保存到本地，便于复用与对比。

典型用法：
- 在某标的上跑出满意的策略后，存一个快照；
- 之后把同一份策略/参数应用到别的标的（复用 strategy/params）；
- 对同一标的存不同时期的多套策略，随时拉出来横向对比效果。

存储为一个 JSON，结构是快照列表；每条快照自带唯一 id。
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path

from config import STRATEGY_LIB_PATH


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def load(path: Path | None = None) -> list[dict]:
    p = path or STRATEGY_LIB_PATH
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def save(items: list[dict], path: Path | None = None) -> None:
    p = path or STRATEGY_LIB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def list_all() -> list[dict]:
    """按创建时间倒序返回全部快照。"""
    return sorted(load(), key=lambda x: x.get("created_at", ""), reverse=True)


def add(
    *,
    name: str,
    symbols: list[str],
    strategy: str,
    params: dict,
    source: str,
    start: str,
    end: str,
    init_cash: float,
    metrics: dict | None = None,
) -> dict:
    """新增一条策略快照并持久化，返回该快照。"""
    item = {
        "id": uuid.uuid4().hex[:8],
        "name": (name or "").strip() or f"{strategy}·{'/'.join(symbols) or '未选'}",
        "symbols": list(symbols),
        "strategy": strategy,
        "params": dict(params),
        "source": source,
        "start": str(start),
        "end": str(end),
        "init_cash": float(init_cash),
        "metrics": metrics or {},
        "created_at": _now(),
    }
    items = load()
    items.append(item)
    save(items)
    return item


def remove(snap_id: str) -> None:
    save([x for x in load() if x.get("id") != snap_id])


def rename(snap_id: str, new_name: str) -> None:
    items = load()
    for x in items:
        if x.get("id") == snap_id:
            x["name"] = new_name.strip() or x["name"]
            break
    save(items)


def get(snap_id: str) -> dict | None:
    for x in load():
        if x.get("id") == snap_id:
            return x
    return None


def symbols_in_library() -> list[str]:
    """库中出现过的全部标的（去重，便于按标的筛选对比）。"""
    seen: list[str] = []
    for x in load():
        for s in x.get("symbols", []):
            if s not in seen:
                seen.append(s)
    return seen
