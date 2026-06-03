"""独立浏览器进程 — 不依赖 asyncio，直接弹窗"""
import sys, time, json
from playwright.sync_api import sync_playwright

url = sys.argv[1] if len(sys.argv) > 1 else "https://www.baidu.com"
search = sys.argv[2] if len(sys.argv) > 2 else ""

pw = sync_playwright().start()
browser = pw.chromium.launch(
    headless=False,
    args=[
        "--start-maximized",
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
    ]
)
page = browser.new_page(viewport={"width": 1280, "height": 900})

# Hide automation痕迹
page.add_init_script("""
    Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
    delete navigator.__proto__.webdriver;
""")

if search:
    from urllib.parse import quote as _quote
    # Try Bing first (no anti-bot), fallback to Baidu URL search
    urls = [
        f"https://www.bing.com/search?q={_quote(search)}",
        f"https://www.baidu.com/s?wd={_quote(search)}",
    ]
    for u in urls:
        try:
            page.goto(u, timeout=10000, wait_until="domcontentloaded")
            break
        except Exception:
            continue
    time.sleep(3)
else:
    page.goto(url, timeout=15000)
    time.sleep(1)

text = page.inner_text("body")
title = page.title()
print(json.dumps({"title": title, "url": url, "search": search, "preview": text[:500]}, ensure_ascii=False), flush=True)

# Keep browser open for user to see
time.sleep(120)
browser.close()
pw.stop()
