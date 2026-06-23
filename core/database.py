"""
Database layer — PostgreSQL via SQLAlchemy ORM.
Docker: docker compose up postgres -d
"""
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine)

# Ensure table exists
def init_db():
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


def get_db():
    """Dependency injection: yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def query_salary_db(job_title=None, city=None, experience=None):
    """
    Query salary data with optional filters.

    Uses parameterized queries (SQL injection safe).
    Returns list of dicts.
    """
    with SessionLocal() as session:
        sql = "SELECT job_title, company_type, city, experience, min_salary, max_salary, avg_salary, source FROM salaries WHERE 1=1"
        params = {}
        if job_title:
            sql += " AND job_title LIKE :job_title"
            params["job_title"] = f"%{job_title}%"
        if city:
            sql += " AND city = :city"
            params["city"] = city
        if experience:
            sql += " AND experience LIKE :experience"
            params["experience"] = f"%{experience}%"
        sql += " LIMIT 10"

        result = session.execute(text(sql), params)
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]
