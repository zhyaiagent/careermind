"""JWT Auth — register, login, token verification."""
import os, bcrypt, jwt, datetime
from sqlalchemy import text
from core.database import engine, SessionLocal

SECRET = os.getenv("JWT_SECRET", "careermind-secret-key-change-in-production")

def init_auth_tables():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register(username: str, password: str) -> dict:
    with SessionLocal() as s:
        existing = s.execute(text("SELECT id FROM users WHERE username=:u"), {"u":username}).fetchone()
        if existing:
            return {"ok": False, "error": "Username already exists"}
        h = hash_password(password)
        s.execute(text("INSERT INTO users (username,password_hash) VALUES (:u,:p)"), {"u":username,"p":h})
        s.commit()
        return {"ok": True, "message": "Registered"}

def login(username: str, password: str) -> dict:
    with SessionLocal() as s:
        row = s.execute(text("SELECT id,password_hash FROM users WHERE username=:u"), {"u":username}).fetchone()
        if not row or not verify_password(password, row[1]):
            return {"ok": False, "error": "Invalid credentials"}
        token = jwt.encode({
            "user_id": row[0],
            "username": username,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30)
        }, SECRET, algorithm="HS256")
        return {"ok": True, "token": token, "user_id": row[0], "username": username}

def verify_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
