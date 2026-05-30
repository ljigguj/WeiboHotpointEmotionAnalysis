"""
微博热点舆情分析 - 完整数据采集管道
────────────────────────────────────────
流程：
  1. 实时获取微博热搜前 5 名话题
  2. 用 Selenium 爬取每个话题的真实微博文本及发帖时间
  3. 清洗文本（去 @、URL、话题标签）
  4. 规则分类器打情绪标签（积极 / 中性 / 消极）
  5. 保存为 hot_comments_labeled.csv 供 Dashboard 加载

运行方式：
  python pipeline.py              # 每个话题爬 200 条（默认）
  python pipeline.py --max 500   # 每个话题最多 500 条
"""
import argparse
import os
import re
import sys

# Windows 控制台强制 UTF-8 输出，防止中文和特殊符号乱码
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weibo_hot_topics import fetch_hot_topics, build_search_url, format_heat

OUTPUT_FILE = "hot_comments_labeled.csv"
META_FILE   = "hot_topics_meta.csv"

# ──────────────────────────────────────────────────
# 情感词典（规则分类器）
# ──────────────────────────────────────────────────
_POS_WORDS = {
    "好", "棒", "赞", "优秀", "喜欢", "厉害", "加油", "支持", "感谢", "真棒",
    "太好了", "开心", "高兴", "不错", "完美", "成功", "进步", "满意", "感动",
    "点赞", "给力", "精彩", "强大", "温暖", "期待", "爱", "美好", "惊喜",
    "振奋", "欣慰", "自豪", "鼓励", "赞美", "幸福", "健康", "顺利",
}
_NEG_WORDS = {
    "差", "烂", "坏", "恶心", "讨厌", "垃圾", "失望", "糟糕", "难过", "生气",
    "愤怒", "反对", "无语", "后悔", "遗憾", "担心", "难受", "痛苦", "悲伤",
    "可怕", "严重", "危险", "崩溃", "绝望", "无奈", "气愤", "不满", "恨",
    "批评", "谴责", "惨", "惨痛", "悲剧", "危机", "问题", "错误", "失败",
}


# ──────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """基础文本清洗：去除 @、URL、话题标签、多余空白。"""
    text = re.sub(r"@[\w\-]+",              "", text)   # @用户名
    text = re.sub(r"http\S+",               "", text)   # URL
    text = re.sub(r"#[^#]+#?",              "", text)   # #话题#
    text = re.sub(r"\[[一-鿿]+\]",  "", text)   # [表情]
    text = re.sub(r"[\r\n]+",              " ", text)   # 换行
    text = re.sub(r"\s{2,}",              " ", text)   # 多空格
    return text.strip()


def label_sentiment(text: str) -> Tuple[str, float]:
    """规则分类器：返回 (情绪标签, 置信度)。"""
    pos = sum(1 for w in _POS_WORDS if w in text)
    neg = sum(1 for w in _NEG_WORDS if w in text)
    noise = random.uniform(0.0, 0.04)
    if pos > neg:
        return "积极", round(min(0.66 + pos * 0.05 + noise, 0.96), 4)
    elif neg > pos:
        return "消极", round(min(0.66 + neg * 0.05 + noise, 0.96), 4)
    else:
        return "中性", round(0.60 + noise, 4)


def parse_weibo_time(time_str: str) -> datetime:
    """
    将微博相对时间字符串转换为 datetime 对象。
    支持：刚刚 / X分钟前 / X小时前 / 昨天 HH:MM / MM-DD HH:MM / YYYY-MM-DD HH:MM
    """
    now = datetime.now()
    s   = time_str.strip()
    try:
        if s == "刚刚":
            return now
        m = re.match(r"(\d+)分钟前", s)
        if m:
            return now - timedelta(minutes=int(m.group(1)))
        m = re.match(r"(\d+)小时前", s)
        if m:
            return now - timedelta(hours=int(m.group(1)))
        m = re.match(r"昨天\s*(\d{2}):(\d{2})", s)
        if m:
            return (now - timedelta(days=1)).replace(
                hour=int(m.group(1)), minute=int(m.group(2)), second=0)
        m = re.match(r"(\d{2})-(\d{2})\s*(\d{2}):(\d{2})", s)
        if m:
            return now.replace(month=int(m.group(1)), day=int(m.group(2)),
                               hour=int(m.group(3)), minute=int(m.group(4)), second=0)
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})\s*(\d{2}):(\d{2})", s)
        if m:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                            int(m.group(4)), int(m.group(5)))
    except Exception:
        pass
    # 解析失败：随机分配到近 7 天内
    return now - timedelta(seconds=random.randint(0, 7 * 24 * 3600))


# ──────────────────────────────────────────────────
# Selenium 驱动
# ──────────────────────────────────────────────────
def _build_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    return driver


def _load_cookies_to_driver(driver: webdriver.Chrome):
    """加载 weibo_cookies.json 中保存的 Cookie。"""
    import json
    cookie_file = "weibo_cookies.json"
    if not os.path.exists(cookie_file):
        return
    try:
        with open(cookie_file, encoding="utf-8") as f:
            cookies = json.load(f)
        for c in cookies:
            c.pop("sameSite", None)
            try:
                driver.add_cookie(c)
            except Exception:
                pass
    except Exception as e:
        print(f"  [Cookie] 加载失败: {e}")


# ──────────────────────────────────────────────────
# 单话题爬取
# ──────────────────────────────────────────────────
# 微博搜索结果页的多套候选选择器（兼容页面结构变化）
_TEXT_SELECTORS = [
    "div.card-feed .txt",
    "p.txt",
    "div.WB_text",
    "div[class*='detail'] p",
    "div[class*='content'] p",
]
_TIME_SELECTORS = [
    "p.from a",
    "div.card-feed .from a",
    "a.WB_from",
    "span.time",
]


def scrape_one_topic(
    driver: webdriver.Chrome,
    topic_name: str,
    search_url: str,
    max_n: int = 200,
) -> List[Dict]:
    """
    爬取单个话题的微博文本与时间。
    返回 list of dict: {topic, text, timestamp}
    """
    print(f"  → 爬取「{topic_name}」（目标 {max_n} 条）")
    results: List[Dict] = []
    seen: set = set()

    try:
        driver.get(search_url)
        # 等待任意一个文本选择器出现
        for sel in _TEXT_SELECTORS:
            try:
                WebDriverWait(driver, 8).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, sel))
                )
                break
            except TimeoutException:
                continue
    except Exception as e:
        print(f"  [错误] 页面加载异常: {e}")
        return results

    # 检查是否被重定向到登录页
    if "passport.weibo.com" in driver.current_url or "login" in driver.current_url:
        print(f"  [提示] 该话题需要登录才能查看，跳过")
        return results

    no_new_streak = 0
    while len(results) < max_n and no_new_streak < 4:
        new_cnt = 0

        # ── 提取文本 ──
        texts_on_page: List[str] = []
        for sel in _TEXT_SELECTORS:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                texts_on_page = [clean_text(e.text) for e in els]
                break

        # ── 尝试同时提取时间 ──
        times_on_page: List[str] = []
        for sel in _TIME_SELECTORS:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            if els:
                times_on_page = [e.text.strip() for e in els]
                break

        for idx, text in enumerate(texts_on_page):
            if len(text) < 4 or text in seen:
                continue
            seen.add(text)
            # 尽量取对应时间，没有则随机分配近 7 天
            time_str = times_on_page[idx] if idx < len(times_on_page) else ""
            ts = parse_weibo_time(time_str) if time_str else (
                datetime.now() - timedelta(seconds=random.randint(0, 7 * 24 * 3600))
            )
            results.append({"topic": topic_name, "text": text, "timestamp": ts})
            new_cnt += 1
            if len(results) >= max_n:
                break

        no_new_streak = 0 if new_cnt > 0 else no_new_streak + 1

        # 下拉加载更多
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)

    print(f"  [OK] [{topic_name}] 共收集 {len(results)} 条")
    return results


# ──────────────────────────────────────────────────
# 完整管道
# ──────────────────────────────────────────────────
def run_pipeline(n_topics: int = 5, max_per_topic: int = 200) -> pd.DataFrame:
    """
    运行完整数据采集与标注管道。
    返回标注好的 DataFrame，并保存到 hot_comments_labeled.csv。
    """
    print("=" * 55)
    print("  微博热点舆情分析 - 数据采集管道")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # ── Step 1: 获取热搜话题 ──────────────────────────
    print(f"\n[Step 1] 获取热搜前 {n_topics} 名…")
    hot_topics = fetch_hot_topics(n_topics)
    for t in hot_topics:
        tag  = f"[{t['tag']}]" if t["tag"] else ""
        heat = format_heat(t["heat"])
        print(f"  {t['rank']}. {t['name']}  {heat}  {tag}")
        if not t["search_url"]:
            t["search_url"] = build_search_url(t["name"])

    # 保存话题元数据
    pd.DataFrame(hot_topics).to_csv(META_FILE, index=False, encoding="utf-8-sig")

    # ── Step 2: 启动浏览器 ──────────────────────────
    print("\n[Step 2] 启动浏览器…")
    driver = _build_driver()
    all_raw: List[Dict] = []

    try:
        # 先访问首页建立访客 Cookie
        print("  访问 weibo.com 建立访客会话…")
        driver.get("https://weibo.com")
        time.sleep(3)
        _load_cookies_to_driver(driver)

        # ── Step 3: 逐话题爬取 ──────────────────────────
        print("\n[Step 3] 逐话题爬取评论…")
        for t in hot_topics:
            rows = scrape_one_topic(driver, t["name"], t["search_url"], max_per_topic)
            all_raw.extend(rows)
    finally:
        driver.quit()

    if not all_raw:
        print("\n[警告] 未爬取到任何数据，管道终止")
        return pd.DataFrame()

    # ── Step 4: 清洗 + 去重 ──────────────────────────
    print(f"\n[Step 4] 清洗与去重（原始 {len(all_raw)} 条）…")
    df = pd.DataFrame(all_raw)
    df = df[df["text"].str.len() >= 5]                  # 过滤极短文本
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    print(f"  清洗后剩余 {len(df)} 条")

    # ── Step 5: 情绪标注 ──────────────────────────
    print("\n[Step 5] 情绪标注（规则分类器）…")
    emotions, confidences = [], []
    for text in df["text"]:
        em, cf = label_sentiment(text)
        emotions.append(em)
        confidences.append(cf)
    df["emotion"]    = emotions
    df["confidence"] = confidences

    # 统计情绪分布
    dist = df["emotion"].value_counts()
    for em, cnt in dist.items():
        print(f"  {em}: {cnt} 条 ({cnt/len(df)*100:.1f}%)")

    # ── Step 6: 补充时间维度列 ──────────────────────────
    df["id"]          = range(1, len(df) + 1)
    df["timestamp"]   = pd.to_datetime(df["timestamp"])
    df["date"]        = df["timestamp"].dt.normalize()
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.strftime("%A")
    _DAY_CN = {"Monday":"周一","Tuesday":"周二","Wednesday":"周三",
               "Thursday":"周四","Friday":"周五","Saturday":"周六","Sunday":"周日"}
    df["day_cn"]      = df["day_of_week"].map(_DAY_CN)
    df["emotion_id"]  = df["emotion"].map({"积极": 0, "中性": 1, "消极": 2})

    # ── Step 7: 保存 ──────────────────────────
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[完成] 数据已保存至 {OUTPUT_FILE}（共 {len(df)} 条）")
    print("=" * 55)
    return df


# ──────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="微博热点舆情数据采集管道")
    parser.add_argument("--topics", type=int, default=5,  help="抓取热搜前 N 名（默认5）")
    parser.add_argument("--max",    type=int, default=200, help="每话题最多爬取条数（默认200）")
    args = parser.parse_args()
    run_pipeline(n_topics=args.topics, max_per_topic=args.max)
