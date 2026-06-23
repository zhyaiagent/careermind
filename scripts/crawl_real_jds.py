"""Crawl real JDs: Bing search -> open job listing pages -> extract full text."""
import sys, os, json, time, re, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from playwright.sync_api import sync_playwright

SEARCHES = [
    ("AI算法工程师", "北京"), ("AI Agent开发", "上海"),
    ("大模型工程师", "深圳"), ("Python后端开发", "杭州"),
    ("NLP算法工程师", "北京"), ("数据分析师", "广州"),
    ("深度学习工程师", "上海"), ("RAG工程师", "北京"),
    ("Java架构师", "深圳"), ("前端开发工程师", "杭州"),
    ("AI产品经理", "北京"), ("CV算法工程师", "上海"),
    ("推荐算法工程师", "北京"), ("DevOps工程师", "深圳"),
    ("Go开发工程师", "杭州"), ("测试开发", "北京"),
    ("安全工程师", "深圳"), ("云计算工程师", "上海"),
    ("机器学习工程师", "北京"), ("数据工程师", "杭州"),
]

OUTPUT = "data/raw/jds.jsonl"

def extract_jd_from_page(page, title, city):
    """Try to extract JD-like content from any job listing page."""
    text = page.inner_text("body")
    # Look for job description patterns
    patterns = [
        r'(岗位职责.*?)(?=任职要求|岗位要求|职位要求|薪资|公司)',
        r'(职位描述.*?)(?=任职要求|岗位要求|职位要求|薪资|公司)',
        r'(工作内容.*?)(?=任职要求|岗位要求|职位要求|薪资|公司)',
        r'(岗位要求.*?)(?=薪资|公司|福利|联系|投递)',
        r'(任职要求.*?)(?=薪资|公司|福利|联系|投递)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m and len(m.group(1)) > 50:
            return m.group(1).strip()
    # Fallback: just return first 1000 chars
    return text[:1500]

def crawl():
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=False, args=["--no-first-run"])
    jds = []

    for title, city in SEARCHES:
        print(f"\n{'='*50}")
        print(f"  {title} @ {city}")
        print(f"{'='*50}")

        page = browser.new_page()
        try:
            # Search on Bing for job listings
            query = f"site:zhipin.com OR site:lagou.com OR site:51job.com {title} {city} 招聘"
            page.goto(f"https://www.bing.com/search?q={query}", timeout=15000, wait_until="commit")
            time.sleep(2)

            # Find job listing links (not ads, not Bing itself)
            links = page.locator("#b_results a[href]")
            count = min(links.count(), 5)
            found_jds = []

            for i in range(count):
                try:
                    link = links.nth(i)
                    url = link.get_attribute("href")
                    if not url or "bing.com" in url or "microsoft" in url:
                        continue
                    if not any(d in url for d in ["zhipin.com", "lagou.com", "51job.com", "liepin.com", "job"]):
                        continue

                    print(f"  Opening: {url[:60]}...")
                    # Open in same page
                    page.goto(url, timeout=15000, wait_until="commit")
                    time.sleep(2)

                    jd_text = extract_jd_from_page(page, title, city)
                    if len(jd_text) > 80:
                        jd = {
                            "job_title": title,
                            "company": "",
                            "city": city,
                            "salary_range": "",
                            "experience": "",
                            "education": "",
                            "jd_text": jd_text,
                            "tags": [],
                            "source_url": url,
                            "collected_at": time.strftime("%Y-%m-%d")
                        }
                        found_jds.append(jd)
                        print(f"    Got {len(jd_text)} chars")
                    else:
                        print(f"    Too short ({len(jd_text)} chars), skipping")

                    if len(found_jds) >= 2:  # 2 per search is enough
                        break
                    time.sleep(1)

                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            jds.extend(found_jds)
            print(f"  -> {len(found_jds)} JDs from this search")

        except Exception as e:
            print(f"  Search error: {e}")
        finally:
            page.close()
            time.sleep(2)  # Be polite to servers

    browser.close()
    pw.stop()

    # Save
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for jd in jds:
            f.write(json.dumps(jd, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"  DONE: {len(jds)} real JDs saved to {OUTPUT}")
    print(f"{'='*50}")

if __name__ == "__main__":
    crawl()
