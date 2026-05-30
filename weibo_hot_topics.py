"""
微博热搜榜实时获取模块
────────────────────────────────
用法：
    from weibo_hot_topics import fetch_hot_topics
    topics = fetch_hot_topics(n=5)
    # → [{"rank":1, "name":"特朗普访华", "heat":"9999999",
    #      "tag":"沸", "search_url":"https://s.weibo.com/weibo?q=..."}, ...]
"""
import time
import urllib.parse
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

# ──────────────────────────── 常量 ────────────────────────────
_HOT_SEARCH_URL = "https://s.weibo.com/top/summary"
_BASE_SEARCH    = "https://s.weibo.com/weibo?q={query}&Refer=top"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
    "Referer":         "https://weibo.com/",
}

# 热搜标签图标 → 可读文字
_TAG_MAP = {
    "icon-热": "热",
    "icon-沸": "沸",
    "icon-爆": "爆",
    "icon-新": "新",
    "icon-商业": "商",
}

# 爬取失败时的备用话题（保持与 dashboard 中的演示数据兼容）
FALLBACK_TOPICS: List[Dict] = [
    {"rank": 1, "name": "特朗普访华",       "heat": "9999999", "tag": "沸", "search_url": ""},
    {"rank": 2, "name": "5G最新进展",        "heat": "8521000", "tag": "热", "search_url": ""},
    {"rank": 3, "name": "新冠病毒最新动态",  "heat": "7234000", "tag": "热", "search_url": ""},
    {"rank": 4, "name": "全国体育赛事",      "heat": "6108000", "tag": "新", "search_url": ""},
    {"rank": 5, "name": "气候极端天气预警",  "heat": "5423000", "tag": "热", "search_url": ""},
]


# ──────────────────────────── 核心函数 ────────────────────────────
def fetch_hot_topics(n: int = 5, timeout: int = 8, retry: int = 2) -> List[Dict]:
    """
    爬取微博热搜榜前 n 名话题。

    返回 list[dict]，每个 dict 含：
        rank        排名（int）
        name        话题名（不含 # 号）
        heat        热度值字符串
        tag         标签文字（热/沸/爆/新/商，或空字符串）
        search_url  微博搜索页 URL

    网络异常或解析失败时返回 FALLBACK_TOPICS[:n]。
    """
    for attempt in range(retry):
        try:
            session = requests.Session()
            resp = session.get(_HOT_SEARCH_URL, headers=_HEADERS,
                               timeout=timeout, allow_redirects=True)
            resp.raise_for_status()
            topics = _parse(resp.text, n)
            if topics:
                print(f"[热搜] 成功获取 {len(topics)} 个话题（第 {attempt+1} 次尝试）")
                return topics
        except Exception as exc:
            print(f"[热搜] 第 {attempt+1} 次尝试失败: {exc}")
            if attempt < retry - 1:
                time.sleep(1.5)

    print("[热搜] 全部重试失败，使用备用话题列表")
    return FALLBACK_TOPICS[:n]


def build_search_url(topic_name: str) -> str:
    """根据话题名构造微博搜索 URL（含 # 标签格式）。"""
    query = f"#{topic_name}#"
    return _BASE_SEARCH.format(query=urllib.parse.quote(query))


# ──────────────────────────── 解析函数 ────────────────────────────
def _parse(html: str, n: int) -> List[Dict]:
    """从热搜页 HTML 中提取话题列表。"""
    soup = BeautifulSoup(html, "html.parser")
    topics: List[Dict] = []

    # 尝试多种选择器以兼容页面结构变化
    rows = (
        soup.select("#pl_top_realtimehot table tbody tr")
        or soup.select("table.realtimehot tbody tr")
        or soup.select("table tbody tr")
    )

    for row in rows:
        rank_td = row.select_one("td.td-01")
        name_a  = row.select_one("td.td-02 a")
        heat_sp = row.select_one("td.td-02 span")
        tag_i   = row.select_one("td.td-03 i, td.td-02 i")

        if not name_a or not rank_td:
            continue

        rank_text = rank_td.get_text(strip=True)
        try:
            rank = int(rank_text)
        except ValueError:
            continue  # 跳过置顶/广告行

        name = name_a.get_text(strip=True).lstrip("#").rstrip("#")
        if not name:
            continue

        heat = heat_sp.get_text(strip=True) if heat_sp else "0"

        # 识别标签
        tag = ""
        if tag_i:
            classes = " ".join(tag_i.get("class", []))
            for cls, label in _TAG_MAP.items():
                if cls in classes:
                    tag = label
                    break

        href = name_a.get("href", "")
        if href.startswith("/"):
            url = f"https://s.weibo.com{href}"
        elif href.startswith("http"):
            url = href
        else:
            url = build_search_url(name)

        topics.append({
            "rank":       rank,
            "name":       name,
            "heat":       heat,
            "tag":        tag,
            "search_url": url,
        })
        if len(topics) >= n:
            break

    return topics


def format_heat(heat_str: str) -> str:
    """将热度数字格式化为可读字符串，例如 '9999999' → '999.9万'。"""
    try:
        val = int(heat_str)
        if val >= 10_000:
            return f"{val / 10_000:.1f}万"
        return str(val)
    except (ValueError, TypeError):
        return heat_str or "-"


# ──────────────────────────── 命令行测试 ────────────────────────────
if __name__ == "__main__":
    print("正在获取微博实时热搜…\n")
    results = fetch_hot_topics(10)
    for t in results:
        heat_fmt = format_heat(t["heat"])
        tag_str  = f"[{t['tag']}]" if t["tag"] else ""
        print(f"  {t['rank']:>2}. {t['name']:<20} {heat_fmt:>8} {tag_str}")
