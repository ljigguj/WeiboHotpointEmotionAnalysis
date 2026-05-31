import time, sys
sys.path.insert(0, '.')
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from weibo_hot_topics import _load_cookies, fetch_hot_topics

opts = Options()
opts.add_argument('--headless=new')
opts.add_argument('--no-sandbox')
opts.add_argument('--disable-dev-shm-usage')
opts.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36')
driver = webdriver.Chrome(options=opts)
driver.get('https://weibo.com')
time.sleep(2)
_load_cookies(driver)

topics = fetch_hot_topics(1)
driver.get(topics[0]['search_url'])
time.sleep(4)

print("话题:", topics[0]['name'])
print("URL:", driver.current_url[:80])

candidates = [
    'div.card-wrap',
    'div.card',
    'article.weibo-main',
    'div[action-type="feed_list_item"]',
    'p.txt',
    'p[node-type="feed_list_content"]',
    'div.content',
    'div.weibo-og',
    'div.m-imglayout',
    'div.list-group-item',
]
print("\n--- 选择器匹配情况 ---")
for sel in candidates:
    els = driver.find_elements(By.CSS_SELECTOR, sel)
    preview = els[0].text[:60].replace('\n', ' ') if els else ''
    print(f"[{len(els):>3}] {sel:<45}  {preview!r}")

# p.txt 父元素链
print("\n--- p.txt 父元素 class ---")
ptxts = driver.find_elements(By.CSS_SELECTOR, 'p.txt')
for i, p in enumerate(ptxts[:4]):
    parent_cls = driver.execute_script('return arguments[0].parentElement.className', p)
    gp_cls = driver.execute_script('return arguments[0].parentElement.parentElement.className', p)
    print(f"  [{i}] parent={parent_cls!r}  grandparent={gp_cls!r}")
    print(f"       text={p.text[:80]!r}")

# page=2 的匹配情况
url2 = topics[0]['search_url'] + '&page=2'
driver.get(url2)
time.sleep(3)
ptxt2 = driver.find_elements(By.CSS_SELECTOR, 'p.txt')
card2 = driver.find_elements(By.CSS_SELECTOR, 'div.card-wrap')
print(f"\npage=2:  p.txt={len(ptxt2)}  div.card-wrap={len(card2)}")

driver.quit()
