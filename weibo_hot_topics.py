"""
微博热搜榜实时获取模块（Selenium 版）
────────────────────────────────────────
问题根因：直接用 requests 访问 https://s.weibo.com/top/summary 会被
        302 跳转到 passport.weibo.com/visitor 做访客鉴权，
        返回的是登录页 HTML，所有热搜选择器均无法匹配。

解决方案：用 Selenium 自动完成访客跳转后获取热搜内容；
         支持保存/加载登录 Cookie，登录后体验更稳定。

用法：
    from weibo_hot_topics import fetch_hot_topics, format_heat
    topics = fetch_hot_topics(n=5)
"""
import json
import os
import time
import urllib.parse
from typing import List, Dict

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# ──────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────
_HOT_URL     = "https://s.weibo.com/top/summary"
_HOME_URL    = "https://weibo.com"
_COOKIES_FILE = "weibo_cookies.json"

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_TAG_MAP = {
    "icon-热": "热", "icon-沸": "沸",
    "icon-爆": "爆", "icon-新": "新", "icon-商业": "商",
}

# 爬取失败时的备用话题（与 dashboard 演示数据兼容）
FALLBACK_TOPICS: List[Dict] = [
    {"rank": 1, "name": "特朗普访华",      "heat": "9999999", "tag": "沸", "search_url": ""},
    {"rank": 2, "name": "5G最新进展",       "heat": "8521000", "tag": "热", "search_url": ""},
    {"rank": 3, "name": "新冠病毒最新动态", "heat": "7234000", "tag": "热", "search_url": ""},
    {"rank": 4, "name": "全国体育赛事",     "heat": "6108000", "tag": "新", "search_url": ""},
    {"rank": 5, "name": "气候极端天气预警", "heat": "5423000", "tag": "热", "search_url": ""},
]


# ──────────────────────────────────────────────────
# 公开接口
# ──────────────────────────────────────────────────
def fetch_hot_topics(n: int = 5, headless: bool = True, timeout: int = 20) -> List[Dict]:
    """
    获取微博热搜前 n 名话题。

    流程
    ────
    1. 打开 Chrome（无头模式）
    2. 先访问 weibo.com，让 Selenium 自动完成访客鉴权 Cookie
    3. 加载本地保存的登录 Cookie（若有）
    4. 访问热搜页，等待列表渲染
    5. 提取话题列表；成功则保存 Cookie，失败则返回备用话题

    参数
    ────
    headless : True=后台运行，False=弹出浏览器窗口（调试用）
    timeout  : 等待热搜列表出现的最长秒数
    """
    driver = None
    try:
        driver = _build_driver(headless)

        # ── Step 1: 访问首页，触发访客鉴权 ──
        print("[热搜] 访问 weibo.com 建立访客会话…")
        driver.get(_HOME_URL)
        time.sleep(3)                       # 等待访客 Cookie 写入

        # ── Step 2: 加载已保存的登录 Cookie ──
        _load_cookies(driver)

        # ── Step 3: 跳转热搜页 ──
        print("[热搜] 跳转热搜列表…")
        driver.get(_HOT_URL)

        # ── Step 4: 等待热搜表格出现 ──
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "table tbody tr"))
            )
        except TimeoutException:
            print(f"[热搜] 等待超时（>{timeout}s），尝试直接解析当前页面")

        # ── Step 5: 检查是否仍在鉴权/登录页 ──
        cur = driver.current_url
        if "passport.weibo.com" in cur or "login" in cur:
            print(f"[热搜] 仍在登录页 ({cur[:60]}…)")
            print("       → 可运行 save_login_cookies() 保存登录态后重试")
            return FALLBACK_TOPICS[:n]

        # ── Step 6: 提取话题 ──
        topics = _extract(driver, n)
        if topics:
            print(f"[热搜] 成功获取 {len(topics)} 个话题")
            _save_cookies(driver)           # 缓存 Cookie 供下次使用
            return topics

        # 获取到空列表：打印诊断信息
        print("[热搜] 列表为空，输出诊断信息…")
        _diagnose(driver)
        return FALLBACK_TOPICS[:n]

    except WebDriverException as e:
        print(f"[热搜] 浏览器错误: {e}")
        return FALLBACK_TOPICS[:n]
    except Exception as e:
        print(f"[热搜] 未知错误: {e}")
        return FALLBACK_TOPICS[:n]
    finally:
        if driver:
            driver.quit()


def save_login_cookies(timeout: int = 120):
    """
    弹出浏览器窗口，等待手动登录微博后自动保存 Cookie。
    登录成功后 Cookie 写入 weibo_cookies.json，
    之后 fetch_hot_topics() 会自动加载，无需再次登录。

    用法：
        python weibo_hot_topics.py --save-cookies
    """
    print("正在打开浏览器，请在页面中手动登录微博…")
    driver = _build_driver(headless=False)
    try:
        driver.get("https://weibo.com/login.php")
        print(f"请在 {timeout} 秒内完成登录…")
        WebDriverWait(driver, timeout).until(
            lambda d: "login" not in d.current_url and "passport" not in d.current_url
        )
        _save_cookies(driver)
        print(f"✓ 登录 Cookie 已保存至 {_COOKIES_FILE}")
    except TimeoutException:
        print("登录超时，未保存 Cookie")
    finally:
        driver.quit()


def build_search_url(topic_name: str) -> str:
    """根据话题名构造微博搜索页 URL。"""
    query = f"#{topic_name}#"
    return f"https://s.weibo.com/weibo?q={urllib.parse.quote(query)}&Refer=top"


def format_heat(heat_str: str) -> str:
    """将热度数字格式化为可读字符串，如 '9999999' → '999.9万'。"""
    try:
        val = int(heat_str)
        return f"{val / 10_000:.1f}万" if val >= 10_000 else str(val)
    except (ValueError, TypeError):
        return heat_str or "-"


# ──────────────────────────────────────────────────
# 内部工具
# ──────────────────────────────────────────────────
def _build_driver(headless: bool) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"user-agent={_UA}")
    # 隐藏自动化特征，降低被拦截概率
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    return driver


def _load_cookies(driver: webdriver.Chrome) -> bool:
    """从本地文件加载 Cookie，返回是否成功。"""
    if not os.path.exists(_COOKIES_FILE):
        return False
    try:
        with open(_COOKIES_FILE, encoding="utf-8") as f:
            cookies = json.load(f)
        for c in cookies:
            # Selenium 不接受 sameSite=None 的 cookie，过滤掉
            c.pop("sameSite", None)
            try:
                driver.add_cookie(c)
            except Exception:
                pass
        print(f"[热搜] 已加载本地 Cookie（{len(cookies)} 条）")
        return True
    except Exception as e:
        print(f"[热搜] Cookie 加载失败: {e}")
        return False


def _save_cookies(driver: webdriver.Chrome):
    """将当前会话 Cookie 保存到本地文件。"""
    try:
        cookies = driver.get_cookies()
        with open(_COOKIES_FILE, "w", encoding="utf-8") as f:
            json.dump(cookies, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[热搜] Cookie 保存失败: {e}")


def _extract(driver: webdriver.Chrome, n: int) -> List[Dict]:
    """从已加载的热搜页面提取话题列表。"""
    topics: List[Dict] = []

    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
    if not rows:
        # 备用选择器
        rows = driver.find_elements(By.CSS_SELECTOR, "tr")

    for row in rows:
        # ── 排名 ──
        try:
            rank_el   = row.find_element(By.CSS_SELECTOR, "td.td-01")
            rank_text = rank_el.text.strip()
            rank      = int(rank_text)
        except (Exception, ValueError):
            continue                        # 跳过广告/置顶行

        # ── 话题名 + URL ──
        try:
            name_el = row.find_element(By.CSS_SELECTOR, "td.td-02 a")
        except Exception:
            continue
        name = name_el.text.strip().lstrip("#").rstrip("#")
        href = name_el.get_attribute("href") or ""
        url  = href if href.startswith("http") else f"https://s.weibo.com{href}"
        if not url or not name:
            continue

        # ── 热度 ──
        heat = ""
        try:
            heat = row.find_element(By.CSS_SELECTOR, "td.td-02 span").text.strip()
        except Exception:
            pass

        # ── 标签（热/沸/爆/新）──
        tag = ""
        try:
            tag_el  = row.find_element(By.CSS_SELECTOR, "td.td-03 i, td.td-02 i")
            classes = tag_el.get_attribute("class") or ""
            for cls, label in _TAG_MAP.items():
                if cls in classes:
                    tag = label
                    break
        except Exception:
            pass

        topics.append({"rank": rank, "name": name,
                       "heat": heat, "tag": tag, "search_url": url})
        if len(topics) >= n:
            break

    return topics


def _diagnose(driver: webdriver.Chrome):
    """打印诊断信息，帮助定位选择器问题。"""
    print("=" * 50)
    print("[ 诊断 ] 当前 URL:", driver.current_url)
    print("[ 诊断 ] 页面标题:", driver.title)
    rows = driver.find_elements(By.CSS_SELECTOR, "tr")
    print(f"[ 诊断 ] <tr> 数量: {len(rows)}")
    tds  = driver.find_elements(By.CSS_SELECTOR, "td")
    print(f"[ 诊断 ] <td> 数量: {len(tds)}")
    if tds:
        sample = tds[0].get_attribute("outerHTML")[:200]
        print(f"[ 诊断 ] 首个 <td> 片段: {sample}")
    print("=" * 50)


# ──────────────────────────────────────────────────
# 命令行入口
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--save-cookies" in sys.argv:
        save_login_cookies()
    else:
        print("正在获取微博实时热搜…\n")
        results = fetch_hot_topics(n=10, headless=False)   # 调试时用非无头模式
        for t in results:
            tag_str  = f"[{t['tag']}]" if t["tag"] else "    "
            heat_fmt = format_heat(t["heat"])
            print(f"  {t['rank']:>2}. {t['name']:<25} {heat_fmt:>8}  {tag_str}")
