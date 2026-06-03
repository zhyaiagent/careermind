"""
Salary Database Builder — creates and populates the SQLite salary database.

Generates ~200 synthetic salary records based on realistic market data
for AI/tech roles across major Chinese cities.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import random
import logging
import argparse
from pathlib import Path

from config import SALARY_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Synthetic Data Parameters ─────────────────────
JOB_TITLES = [
    "AI算法工程师", "NLP工程师", "CV工程师", "大模型工程师",
    "Python开发", "Java开发", "Go开发", "前端开发",
    "后端开发", "数据分析师", "算法工程师", "机器学习工程师",
    "AI Agent开发", "RAG工程师", "深度学习工程师",
]

COMPANY_TYPES = ["互联网", "互联网", "互联网", "外企", "国企", "创业公司"]
CITIES = ["北京", "上海", "深圳", "杭州", "广州", "成都"]
EXPERIENCE_LEVELS = ["应届", "1-3年", "3-5年", "5-10年", "10年+"]
EDUCATION_LEVELS = ["本科", "本科", "硕士", "硕士", "博士", "大专"]

# Base salary (K) by city tier
CITY_BASE = {
    "北京": 18, "上海": 17, "深圳": 16,
    "杭州": 15, "广州": 14, "成都": 12,
}

# Salary multiplier by experience
EXP_MULTIPLIER = {
    "应届": 0.8,
    "1-3年": 1.0,
    "3-5年": 1.4,
    "5-10年": 2.0,
    "10年+": 2.8,
}


def create_tables(conn: sqlite3.Connection):
    """Create the salaries table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS salaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            -- 岗位名称
            company_type TEXT,
            -- 公司类型: 互联网/国企/外企/创业公司
            city TEXT NOT NULL,
            -- 城市
            experience TEXT,
            -- 经验要求
            education TEXT,
            -- 学历要求
            min_salary INTEGER,
            -- 最低薪资(K)
            max_salary INTEGER,
            -- 最高薪资(K)
            avg_salary REAL,
            -- 平均薪资(K)
            source TEXT,
            -- 数据来源
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def generate_salary_records(n: int = 200) -> list[dict]:
    """
    Generate synthetic but realistic salary records.

    Args:
        n: target number of records

    Returns:
        list of record dicts
    """
    records = []
    for _ in range(n):
        job_title = random.choice(JOB_TITLES)
        company_type = random.choice(COMPANY_TYPES)
        city = random.choice(CITIES)
        experience = random.choice(EXPERIENCE_LEVELS)
        education = random.choice(EDUCATION_LEVELS)

        # Base salary calculation
        base = CITY_BASE.get(city, 15)
        exp_mult = EXP_MULTIPLIER.get(experience, 1.0)

        # Add some randomness
        jitter = random.uniform(0.85, 1.15)

        # Title-specific adjustments
        if "AI" in job_title or "大模型" in job_title or "算法" in job_title:
            title_mult = random.uniform(1.2, 1.6)
        elif "Agent" in job_title or "RAG" in job_title:
            title_mult = random.uniform(1.1, 1.4)
        else:
            title_mult = random.uniform(0.9, 1.2)

        # Education adjustment
        edu_mult = {
            "博士": 1.3, "硕士": 1.1, "本科": 1.0, "大专": 0.8
        }.get(education, 1.0)

        # Company type adjustment
        company_mult = {
            "外企": 1.2, "互联网": 1.15, "国企": 0.9, "创业公司": 0.85
        }.get(company_type, 1.0)

        avg_salary = base * exp_mult * title_mult * edu_mult * company_mult * jitter
        avg_salary = round(avg_salary, 1)

        # Min/Max spread
        min_salary = round(avg_salary * 0.75)
        max_salary = round(avg_salary * 1.35)

        record = {
            "job_title": job_title,
            "company_type": company_type,
            "city": city,
            "experience": experience,
            "education": education,
            "min_salary": min_salary,
            "max_salary": max_salary,
            "avg_salary": avg_salary,
            "source": random.choice(["BOSS直聘", "猎聘", "拉勾", "脉脉"]),
        }
        records.append(record)

    return records


def insert_records(conn: sqlite3.Connection, records: list[dict]):
    """Insert salary records into the database."""
    for r in records:
        conn.execute("""
            INSERT INTO salaries
                (job_title, company_type, city, experience, education,
                 min_salary, max_salary, avg_salary, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["job_title"], r["company_type"], r["city"],
            r["experience"], r["education"],
            r["min_salary"], r["max_salary"], r["avg_salary"],
            r["source"],
        ))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Build salary database")
    parser.add_argument(
        "--num-records", type=int, default=200,
        help="Number of salary records to generate"
    )
    parser.add_argument(
        "--output", type=str, default=str(SALARY_DB_PATH),
        help="Output SQLite database path"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB
    output_path.unlink(missing_ok=True)

    conn = sqlite3.connect(str(output_path))
    try:
        create_tables(conn)
        records = generate_salary_records(args.num_records)
        insert_records(conn, records)
        logger.info(f"Created {len(records)} salary records")
        logger.info(f"Database: {output_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
