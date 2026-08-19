"""看板股票池：本地持久化 + 按名称/代码/拼音搜索。

搜索优先走内置热门清单（无网络），不够再请求新浪联想，2 秒超时，
避免再出现全市场代码表把页面卡住的问题。
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from config import WATCHLIST_PATH

# 热门标的：名称 / 代码 / 拼音都能搜到。不是全市场，只保证常用股秒出结果。
_CN_HOT: list[tuple[str, str, str]] = [
    ("000001", "平安银行", "payh pingan"),
    ("000002", "万科A", "wka wanke"),
    ("000333", "美的集团", "mdjt media meidi"),
    ("000651", "格力电器", "gldq geli"),
    ("000725", "京东方A", "jdfa boe"),
    ("000858", "五粮液", "wly wuliangye"),
    ("002230", "科大讯飞", "kdxf iflytek"),
    ("002415", "海康威视", "hkws hikvision"),
    ("002475", "立讯精密", "lxjm luxshare"),
    ("002594", "比亚迪", "byd biyadi"),
    ("300059", "东方财富", "dfcf eastmoney"),
    ("300124", "汇川技术", "hcjs inovance"),
    ("300750", "宁德时代", "ndsd catl"),
    ("300760", "迈瑞医疗", "mryl mindray"),
    ("600036", "招商银行", "zsyh cmb"),
    ("600276", "恒瑞医药", "hryy hengrui"),
    ("600309", "万华化学", "whhx wanhua"),
    ("600519", "贵州茅台", "gzmt maotai moutai mt"),
    ("600887", "伊利股份", "ylgf yili"),
    ("600900", "长江电力", "cjdl yangtze"),
    ("601012", "隆基绿能", "ljln longi"),
    ("601166", "兴业银行", "xyyh cib"),
    ("601318", "中国平安", "zgpa pinganzg"),
    ("601398", "工商银行", "gsyh icbc"),
    ("601888", "中国中免", "zgzm dutyfree"),
    ("601899", "紫金矿业", "zjkj zijin"),
    ("601288", "农业银行", "nyyh abc"),
    ("603259", "药明康德", "ymkd wuxi"),
    ("603501", "韦尔股份", "wegf will"),
    ("688111", "金山办公", "jsbg kingsoft"),
    ("688981", "中芯国际", "zxgj smic"),
]

_US_HOT: list[tuple[str, str, str]] = [
    ("AAPL", "Apple", "apple"),
    ("MSFT", "Microsoft", "microsoft"),
    ("NVDA", "NVIDIA", "nvidia"),
    ("GOOGL", "Alphabet", "google alphabet"),
    ("AMZN", "Amazon", "amazon"),
    ("META", "Meta", "facebook meta"),
    ("TSLA", "Tesla", "tesla"),
    ("BRK-B", "Berkshire Hathaway", "berkshire"),
]

_DEFAULTS = {
    "akshare": {
        "pool": [
            {"symbol": "000001", "name": "平安银行"},
            {"symbol": "600519", "name": "贵州茅台"},
        ],
        "selected": ["000001", "600519"],
    },
    "yfinance": {
        "pool": [
            {"symbol": "AAPL", "name": "Apple"},
            {"symbol": "MSFT", "name": "Microsoft"},
        ],
        "selected": ["AAPL", "MSFT"],
    },
}


def _empty_bucket() -> dict:
    return {"pool": [], "selected": []}


def load(path: Path | None = None) -> dict:
    p = path or WATCHLIST_PATH
    if not p.exists():
        data = json.loads(json.dumps(_DEFAULTS))
        save(data, p)
        return data
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    for src, default in _DEFAULTS.items():
        bucket = data.get(src) or _empty_bucket()
        if not isinstance(bucket, dict):
            bucket = _empty_bucket()
        bucket.setdefault("pool", [])
        bucket.setdefault("selected", [])
        data[src] = bucket
        if not bucket["pool"]:
            bucket["pool"] = json.loads(json.dumps(default["pool"]))
            bucket["selected"] = list(default["selected"])
    return data


def save(data: dict, path: Path | None = None) -> None:
    p = path or WATCHLIST_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def pool(source: str) -> list[dict]:
    return list(load().get(source, _empty_bucket())["pool"])


def selected(source: str) -> list[str]:
    data = load()
    bucket = data.get(source, _empty_bucket())
    symbols = {x["symbol"] for x in bucket["pool"]}
    return [s for s in bucket.get("selected", []) if s in symbols]


def add(source: str, symbol: str, name: str | None = None) -> dict:
    symbol = _norm_symbol(source, symbol)
    name = (name or symbol).strip() or symbol
    data = load()
    bucket = data.setdefault(source, _empty_bucket())
    existing = {x["symbol"]: x for x in bucket["pool"]}
    if symbol in existing:
        if name and name != symbol:
            existing[symbol]["name"] = name
    else:
        bucket["pool"].append({"symbol": symbol, "name": name})
    if symbol not in bucket["selected"]:
        bucket["selected"].append(symbol)
    save(data)
    return {"symbol": symbol, "name": name}


def remove(source: str, symbol: str) -> None:
    data = load()
    bucket = data.get(source, _empty_bucket())
    bucket["pool"] = [x for x in bucket["pool"] if x["symbol"] != symbol]
    bucket["selected"] = [s for s in bucket.get("selected", []) if s != symbol]
    save(data)


def set_selected(source: str, symbols: list[str]) -> None:
    data = load()
    bucket = data.setdefault(source, _empty_bucket())
    allowed = {x["symbol"] for x in bucket["pool"]}
    bucket["selected"] = [s for s in symbols if s in allowed]
    save(data)


def label_of(source: str, symbol: str) -> str:
    for item in pool(source):
        if item["symbol"] == symbol:
            name = item.get("name") or symbol
            return f"{name} · {symbol}" if name != symbol else symbol
    return symbol


def search(query: str, source: str, limit: int = 8) -> list[dict]:
    q = (query or "").strip()
    if not q:
        return []
    hits = _search_local(q, source)
    seen = {x["symbol"] for x in hits}
    if len(hits) < limit:
        for item in _search_remote(q, source, limit=limit):
            if item["symbol"] not in seen:
                hits.append(item)
                seen.add(item["symbol"])
            if len(hits) >= limit:
                break
    return hits[:limit]


def _norm_symbol(source: str, symbol: str) -> str:
    s = symbol.strip().upper() if source == "yfinance" else symbol.strip()
    if source == "akshare" and s.isdigit():
        return s.zfill(6)
    return s


def _search_local(query: str, source: str) -> list[dict]:
    q = query.strip().lower()
    table = _CN_HOT if source == "akshare" else _US_HOT
    out = []
    for symbol, name, keys in table:
        blob = f"{symbol} {name} {keys}".lower()
        if q in blob.replace(" ", "") or q in blob or symbol.lower().startswith(q):
            out.append({"symbol": symbol, "name": name})
    return out


def _search_remote(query: str, source: str, limit: int = 8) -> list[dict]:
    try:
        if source == "akshare":
            return _search_sina(query, limit)
        return _search_yahoo(query, limit)
    except Exception:
        return []


def _search_sina(query: str, limit: int) -> list[dict]:
    import requests

    url = "https://suggest3.sinajs.cn/suggest/type=11,12,13&key=" + quote(query)
    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"},
        timeout=2,
    )
    text = r.content.decode("gbk", errors="ignore")
    if '"' not in text:
        return []
    payload = text.split('"', 1)[1].rsplit('"', 1)[0]
    out = []
    for item in payload.split(";"):
        parts = [p.strip() for p in item.split(",") if p is not None]
        if len(parts) < 4:
            continue
        name, code, mkt = parts[0], parts[2], parts[3].lower()
        parsed = _parse_cn_hit(name, code, mkt)
        if parsed:
            out.append(parsed)
        if len(out) >= limit:
            break
    return out


def _parse_cn_hit(name: str, code: str, mkt: str) -> dict | None:
    if not name or "指数" in name:
        return None
    if mkt.startswith("sh000") or mkt.startswith("sz399"):
        return None
    symbol = code.zfill(6) if code.isdigit() else code
    if not (symbol.isdigit() and len(symbol) == 6):
        return None
    if symbol[0] not in {"0", "3", "6", "8", "9"}:
        return None
    return {"symbol": symbol, "name": name}


def _search_yahoo(query: str, limit: int) -> list[dict]:
    import requests

    r = requests.get(
        "https://query1.finance.yahoo.com/v1/finance/search",
        params={"q": query, "quotesCount": limit, "newsCount": 0, "listsCount": 0},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=2,
    )
    r.raise_for_status()
    quotes = (r.json() or {}).get("quotes") or []
    out = []
    for q in quotes:
        symbol = str(q.get("symbol") or "").strip()
        if not symbol or q.get("quoteType") in {"OPTION", "CRYPTOCURRENCY"}:
            continue
        name = q.get("shortname") or q.get("longname") or symbol
        out.append({"symbol": symbol, "name": str(name)})
        if len(out) >= limit:
            break
    return out
