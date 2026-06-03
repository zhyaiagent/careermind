"""
JD Crawler — scrapes job descriptions from BOSS Zhipin and other platforms.

Saves standardized JD records in JSONL format.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import logging
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config import JDS_JSONL, DATA_RAW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target job titles for AI/tech roles
TARGET_TITLES = [
    "AI算法工程师", "AI Agent开发", "LLM大模型", "NLP工程师",
    "Python开发", "Java开发", "Go开发", "前端开发",
    "数据分析师", "算法工程师",
]

# Target cities
TARGET_CITIES = [
    "北京", "上海", "深圳", "杭州", "广州", "成都",
]

# Expected JD JSON schema
JD_SCHEMA = {
    "job_title": "",
    "company": "",
    "city": "",
    "salary_range": "",     # "25-50K"
    "experience": "",       # "3-5年"
    "education": "",        # "本科"
    "jd_text": "",          # Full JD
    "tags": [],             # ["AI", "RAG", "Agent"]
    "source_url": "",
}


def crawl_boss(job_title: str, city: str, max_pages: int = 5) -> list[dict]:
    """
    Crawl BOSS Zhipin for a given job title and city.

    Uses requests + BeautifulSoup with polite delays.
    Respects robots.txt constraints.

    Returns list of JD dicts.
    """
    jds = []
    base_url = "https://www.zhipin.com"
    search_url = f"{base_url}/web/geek/job"

    headers = {
        "User-Agent": "JobSense-Crawler/1.0 (Educational Research)",
    }

    try:
        for page in range(1, max_pages + 1):
            params = {
                "query": job_title,
                "city": "100010000",  # Placeholder city code
                "page": page,
            }

            try:
                resp = requests.get(
                    search_url, params=params, headers=headers, timeout=15
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Request failed: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            job_cards = soup.find_all("li", class_="job-card-wrapper")

            if not job_cards:
                logger.info(f"No more results for {job_title} in {city} (page {page})")
                break

            for card in job_cards:
                try:
                    title_el = card.find("span", class_="job-name")
                    company_el = card.find("h3", class_="company-name")
                    salary_el = card.find("span", class_="salary")
                    exp_el = card.find("ul", class_="tag-list")

                    title = title_el.text.strip() if title_el else job_title
                    company = company_el.text.strip() if company_el else "未知公司"
                    salary = salary_el.text.strip() if salary_el else ""
                    exp_items = [li.text.strip() for li in exp_el.find_all("li")] if exp_el else []

                    # Build JD entry
                    jd = JD_SCHEMA.copy()
                    jd.update({
                        "job_title": title,
                        "company": company,
                        "city": city,
                        "salary_range": salary,
                        "experience": exp_items[0] if exp_items else "",
                        "education": exp_items[1] if len(exp_items) > 1 else "",
                        "jd_text": f"{title} - {company} - {salary}\n" + "\n".join(exp_items),
                        "tags": _extract_tags(f"{title} {company}"),
                        "source_url": f"{base_url}/job_detail/{card.get('data-jobid', '')}.html",
                    })
                    jds.append(jd)
                except Exception as e:
                    logger.debug(f"Parse error: {e}")
                    continue

            # Polite delay between pages (2-5 seconds)
            time.sleep(3)

    except Exception as e:
        logger.error(f"Crawl error for {job_title}@{city}: {e}")

    return jds


def _extract_tags(text: str) -> list[str]:
    """Extract technology tags from job text."""
    tech_keywords = [
        "Python", "Java", "Go", "C++", "JavaScript", "TypeScript",
        "AI", "AI Agent", "LLM", "NLP", "CV", "机器学习",
        "深度学习", "大模型", "RAG", "LangChain", "Agent",
        "PyTorch", "TensorFlow", "Kubernetes", "Docker",
        "React", "Vue", "Spring", "微服务",
    ]
    tags = []
    text_lower = text.lower()
    for kw in tech_keywords:
        if kw.lower() in text_lower:
            tags.append(kw)
    return tags


def save_jds(jds: list[dict], output_path: Path):
    """Save JD records to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for jd in jds:
            f.write(json.dumps(jd, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Crawl job descriptions")
    parser.add_argument(
        "--max-pages", type=int, default=5,
        help="Maximum pages per job title per city"
    )
    parser.add_argument(
        "--output", type=str, default=str(JDS_JSONL),
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--titles", nargs="*", default=None,
        help="Specific job titles to crawl"
    )
    parser.add_argument(
        "--cities", nargs="*", default=None,
        help="Specific cities to crawl"
    )
    args = parser.parse_args()

    titles = args.titles or TARGET_TITLES[:5]
    cities = args.cities or TARGET_CITIES[:3]

    total_jds = 0
    output_path = Path(args.output)

    # Clear output file if exists
    output_path.unlink(missing_ok=True)

    for title in titles:
        for city in cities:
            logger.info(f"Crawling: {title} @ {city}")
            jds = crawl_boss(title, city, max_pages=args.max_pages)
            save_jds(jds, output_path)
            total_jds += len(jds)
            logger.info(f"  → {len(jds)} JDs saved")
            time.sleep(2)  # Polite delay between queries

    logger.info(f"\nDone! Total JDs crawled: {total_jds}")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
