"""Salary Database Builder — PostgreSQL via SQLAlchemy."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random, logging
from sqlalchemy import create_engine, text
from config import DATABASE_URL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

JOB_TITLES = [
    "AI算法工程师","NLP工程师","CV工程师","大模型工程师",
    "Python开发","Java开发","Go开发","前端开发",
    "后端开发","数据分析师","算法工程师","机器学习工程师",
    "AI Agent开发","RAG工程师","深度学习工程师",
]
COMPANY_TYPES = ["互联网","互联网","互联网","外企","国企","创业公司"]
CITIES = ["北京","上海","深圳","杭州","广州","成都"]
EXPERIENCE_LEVELS = ["应届","1-3年","3-5年","5-10年","10年+"]
EDUCATION_LEVELS = ["本科","本科","硕士","硕士","博士","大专"]
CITY_BASE = {"北京":18,"上海":17,"深圳":16,"杭州":15,"广州":14,"成都":12}
EXP_MULTIPLIER = {"应届":0.8,"1-3年":1.0,"3-5年":1.4,"5-10年":2.0,"10年+":2.8}

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
random.seed(42)

def init_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS salaries (
                id SERIAL PRIMARY KEY,
                job_title TEXT NOT NULL,
                company_type TEXT,
                city TEXT NOT NULL,
                experience TEXT,
                education TEXT,
                min_salary INTEGER,
                max_salary INTEGER,
                avg_salary REAL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    logger.info("Table ready")

def generate(n=200):
    records = []
    for _ in range(n):
        title = random.choice(JOB_TITLES)
        company = random.choice(COMPANY_TYPES)
        city = random.choice(CITIES)
        exp = random.choice(EXPERIENCE_LEVELS)
        edu = random.choice(EDUCATION_LEVELS)
        base = CITY_BASE.get(city, 15)
        exp_m = EXP_MULTIPLIER.get(exp, 1.0)
        jitter = random.uniform(0.85, 1.15)
        title_m = 1.5 if "AI" in title or "大模型" in title or "算法" in title else (1.3 if "Agent" in title or "RAG" in title else 1.1)
        edu_m = {"博士":1.3,"硕士":1.1,"本科":1.0,"大专":0.8}.get(edu, 1.0)
        company_m = {"外企":1.2,"互联网":1.15,"国企":0.9,"创业公司":0.85}.get(company, 1.0)
        avg_salary = round(base * exp_m * title_m * edu_m * company_m * jitter, 1)
        records.append({
            "job_title": title, "company_type": company, "city": city,
            "experience": exp, "education": edu,
            "min_salary": round(avg_salary*0.75), "max_salary": round(avg_salary*1.35),
            "avg_salary": avg_salary,
            "source": random.choice(["BOSS直聘","猎聘","拉勾","脉脉"]),
        })
    return records

def insert(records):
    with engine.connect() as conn:
        for r in records:
            conn.execute(text(
                "INSERT INTO salaries (job_title,company_type,city,experience,education,min_salary,max_salary,avg_salary,source) VALUES (:a,:b,:c,:d,:e,:f,:g,:h,:i)"
            ), {"a":r["job_title"],"b":r["company_type"],"c":r["city"],"d":r["experience"],"e":r["education"],"f":r["min_salary"],"g":r["max_salary"],"h":r["avg_salary"],"i":r["source"]})
        conn.commit()
    logger.info(f"Inserted {len(records)} records")

if __name__ == "__main__":
    init_table()
    data = generate(200)
    insert(data)
    logger.info(f"Done: {len(data)} records in PostgreSQL")
