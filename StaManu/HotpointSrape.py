import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# 配置浏览器（无头模式，不弹出窗口）
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
driver = webdriver.Chrome(options=chrome_options)

# 定义需要爬取的微博热点链接（可添加多个热点）
hot_topic_urls = [
    "https://weibo.com/1234567890/ABCDEFG",  # 替换为真实热点微博链接
    "https://weibo.com/0987654321/HIJLMNO"
]
all_comments = []


# 爬取函数
def crawl_weibo_comments(url, max_comments=5000):
    driver.get(url)
    time.sleep(3)
    comment_count = 0

    # 循环加载评论（下拉页面）
    while comment_count < max_comments:
        # 定位评论元素
        comments = driver.find_elements(By.CSS_SELECTOR, "div.WB_text")
        for comment in comments:
            text = comment.text.strip()
            if text and text not in all_comments:  # 初步去重
                all_comments.append({"raw_comment": text})
                comment_count += 1
                if comment_count >= max_comments:
                    break

        # 下拉加载更多
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)

    print(f"热点链接爬取完成，累计评论：{len(all_comments)}")


# 批量爬取多个热点
for url in hot_topic_urls:
    crawl_weibo_comments(url, max_comments=6000)  # 单热点爬6000，多热点轻松破1万

# 保存原始数据
df = pd.DataFrame(all_comments)
df.to_csv("raw_hot_comments.csv", index=False, encoding="utf-8-sig")
driver.quit()
print("原始评论数据已保存：raw_hot_comments.csv")