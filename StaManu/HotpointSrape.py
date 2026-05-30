"""
微博热点评论爬虫
────────────────────────────────
流程：
  1. 调用 weibo_hot_topics.fetch_hot_topics(5) 获取当前热搜前5名
  2. 为每个话题构造微博搜索页 URL
  3. 用 Selenium 逐页下拉收集评论
  4. 去重后保存至 raw_hot_comments.csv
"""
import sys
import os
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# 将上层目录加入 sys.path，以便 import weibo_hot_topics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from weibo_hot_topics import fetch_hot_topics, build_search_url, format_heat

# ──────────────────────────── 配置 ────────────────────────────
MAX_COMMENTS_PER_TOPIC = 2000   # 每个话题最多收集评论数
SCROLL_PAUSE            = 2.0   # 下滑后等待加载的秒数
LOAD_WAIT               = 3.0   # 页面初次加载等待秒数
OUTPUT_FILE             = "raw_hot_comments.csv"
TOP_N                   = 5     # 取热搜前 N 名


def _build_driver() -> webdriver.Chrome:
    """构建无头 Chrome 驱动。"""
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=opts)


def crawl_topic_comments(
    driver: webdriver.Chrome,
    topic_name: str,
    search_url: str,
    max_comments: int = MAX_COMMENTS_PER_TOPIC,
) -> list:
    """
    爬取单个话题的评论。

    参数
    ----
    driver      : 已初始化的 Selenium WebDriver
    topic_name  : 话题名称（用于标记数据来源）
    search_url  : 微博话题搜索页 URL
    max_comments: 最多收集评论数

    返回
    ----
    list of dict，每条含 topic / raw_comment 字段
    """
    print(f"  → 开始爬取「{topic_name}」，目标 {max_comments} 条，URL: {search_url}")
    collected: list = []
    seen_texts: set = set()

    try:
        driver.get(search_url)
        time.sleep(LOAD_WAIT)
    except WebDriverException as e:
        print(f"  [错误] 页面加载失败: {e}")
        return collected

    # 评论元素选择器（尝试多个，兼容微博页面结构）
    SELECTORS = [
        "div.WB_text",
        "p.txt",
        "div[class*='Feed_detail']",
        "div[class*='content'] p",
    ]

    consecutive_no_new = 0

    while len(collected) < max_comments:
        new_found = 0

        for selector in SELECTORS:
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            for el in elements:
                text = el.text.strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    collected.append({"topic": topic_name, "raw_comment": text})
                    new_found += 1
                    if len(collected) >= max_comments:
                        break
            if len(collected) >= max_comments:
                break

        if new_found == 0:
            consecutive_no_new += 1
            if consecutive_no_new >= 3:
                # 连续 3 次滚动没有新评论，视为已到底部
                break
        else:
            consecutive_no_new = 0

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(SCROLL_PAUSE)

    print(f"  ✓ 「{topic_name}」完成，共收集 {len(collected)} 条评论")
    return collected


def main():
    # ── 步骤 1：获取实时热搜 ──────────────────────────
    print(f"[Step 1] 获取微博热搜前 {TOP_N} 名…")
    hot_topics = fetch_hot_topics(TOP_N)

    print(f"\n当前热搜话题：")
    for t in hot_topics:
        heat_fmt = format_heat(t["heat"])
        tag_str  = f"[{t['tag']}]" if t["tag"] else ""
        print(f"  {t['rank']}. {t['name']}  {heat_fmt}  {tag_str}")

    # ── 步骤 2：构造搜索 URL ──────────────────────────
    for t in hot_topics:
        if not t["search_url"]:
            t["search_url"] = build_search_url(t["name"])

    # ── 步骤 3：启动浏览器并逐话题爬取 ──────────────────────────
    print(f"\n[Step 2] 启动浏览器，逐话题爬取评论…")
    driver = _build_driver()
    all_comments: list = []

    try:
        for topic in hot_topics:
            comments = crawl_topic_comments(
                driver,
                topic_name=topic["name"],
                search_url=topic["search_url"],
                max_comments=MAX_COMMENTS_PER_TOPIC,
            )
            all_comments.extend(comments)
    finally:
        driver.quit()

    # ── 步骤 4：全局去重并保存 ──────────────────────────
    print(f"\n[Step 3] 全局去重并保存…")
    df = pd.DataFrame(all_comments)

    before = len(df)
    df = df.drop_duplicates(subset="raw_comment").reset_index(drop=True)
    after  = len(df)
    print(f"  去重前 {before} 条 → 去重后 {after} 条")

    # 同时保存热搜元数据
    topic_df = pd.DataFrame(hot_topics)
    topic_df.to_csv("hot_topics_meta.csv", index=False, encoding="utf-8-sig")
    print(f"  热搜元数据已保存：hot_topics_meta.csv")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"  原始评论数据已保存：{OUTPUT_FILE}（共 {after} 条）")

    return df, hot_topics


if __name__ == "__main__":
    main()
