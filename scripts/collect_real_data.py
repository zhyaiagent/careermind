"""Collect real JD data via browser automation."""
import sys, os, json, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Real job titles to search
SEARCHES = [
    ("AI算法工程师", "北京"),
    ("AI Agent开发", "上海"),
    ("大模型工程师", "深圳"),
    ("Python后端开发", "杭州"),
    ("NLP算法工程师", "北京"),
    ("数据分析师", "广州"),
    ("深度学习工程师", "上海"),
    ("RAG工程师", "北京"),
    ("Java架构师", "深圳"),
    ("前端开发工程师", "杭州"),
    ("AI产品经理", "北京"),
    ("CV算法工程师", "上海"),
    ("推荐算法工程师", "北京"),
    ("DevOps工程师", "深圳"),
    ("Go开发工程师", "杭州"),
    ("测试开发工程师", "北京"),
    ("安全工程师", "深圳"),
    ("云计算工程师", "上海"),
    ("机器学习工程师", "北京"),
    ("数据工程师", "杭州"),
]

OUTPUT = "data/raw/jds.jsonl"

def collect():
    from playwright.sync_api import sync_playwright
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=False, args=["--no-first-run"])
    page = browser.new_page()
    jds = []

    for title, city in SEARCHES:
        query = f"{title} {city} 招聘"
        url = f"https://www.bing.com/search?q={query}"
        print(f"\nSearching: {title} @ {city}")

        try:
            page.goto(url, timeout=15000, wait_until="commit")
            time.sleep(2)
            # Get search result snippets
            text = page.inner_text("body")

            # Parse snippets into JD entries
            lines = text.split("\n")
            snippets = []
            for line in lines:
                line = line.strip()
                if len(line) > 50 and any(kw in line for kw in ["要求","经验","学历","职责","薪资","K","年","招聘"]):
                    snippets.append(line)

            if snippets:
                jd_text = "\n".join(snippets[:5])
                jds.append({
                    "job_title": title,
                    "company": city,
                    "city": city,
                    "salary_range": "",
                    "experience": "",
                    "education": "",
                    "jd_text": jd_text,
                    "tags": title.replace("工程师","").split("开发")[0].split(),
                    "source_url": url
                })
                print(f"  Got {len(snippets)} snippets")
            else:
                print(f"  No results")

        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(2)  # Be polite

    browser.close()
    pw.stop()

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for jd in jds:
            f.write(json.dumps(jd, ensure_ascii=False) + "\n")

    print(f"\nDone: {len(jds)} JDs saved to {OUTPUT}")

if __name__ == "__main__":
    collect()
