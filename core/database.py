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


def init_db():
    """Create all tables on startup."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS salaries (
                id SERIAL PRIMARY KEY, job_title TEXT NOT NULL, company_type TEXT,
                city TEXT NOT NULL, experience TEXT, education TEXT,
                min_salary INTEGER, max_salary INTEGER, avg_salary REAL,
                source TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_thread ON conversations(thread_id)"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_memory (
                id SERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

def save_message(thread_id, role, content):
    with SessionLocal() as s:
        s.execute(text("INSERT INTO conversations (thread_id,role,content) VALUES (:tid,:r,:c)"),
                   {"tid":thread_id,"r":role,"c":content})
        s.commit()

def get_history(thread_id, max_turns=10):
    with SessionLocal() as s:
        r = s.execute(text(
            "SELECT role,content FROM conversations WHERE thread_id=:tid ORDER BY id DESC LIMIT :n"
        ), {"tid":thread_id,"n":max_turns*2})
        rows = r.fetchall()
        rows.reverse()
        return [{"role":row[0],"content":row[1]} for row in rows]

def clear_history(thread_id):
    with SessionLocal() as s:
        s.execute(text("DELETE FROM conversations WHERE thread_id=:tid"),{"tid":thread_id})
        s.commit()

def save_user_memory(thread_id, key, value):
    with SessionLocal() as s:
        s.execute(text("""
            INSERT INTO user_memory (thread_id,key,value) VALUES (:tid,:k,:v)
            ON CONFLICT (thread_id,key) DO UPDATE SET value=:v, updated_at=CURRENT_TIMESTAMP
        """),{"tid":thread_id,"k":key,"v":value})
        s.commit()

def get_user_memory(thread_id, key=None):
    with SessionLocal() as s:
        if key:
            r = s.execute(text("SELECT value FROM user_memory WHERE thread_id=:tid AND key=:k"),
                          {"tid":thread_id,"k":key})
            row = r.fetchone()
            return row[0] if row else None
        else:
            r = s.execute(text("SELECT key,value FROM user_memory WHERE thread_id=:tid"),
                          {"tid":thread_id})
            return {row[0]:row[1] for row in r.fetchall()}

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
