"""Auth helpers."""
import os, time, jwt, bcrypt, psycopg2
from db import get_pg

JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-prod")
JWT_EXP = 86400  # 24h


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def check_password(pw: str, hashed: str) -> bool:
    return bcrypt.checkpw(pw.encode(), hashed.encode())


def create_token(username: str) -> str:
    return jwt.encode({"sub": username, "exp": time.time() + JWT_EXP}, JWT_SECRET, algorithm="HS256")


def verify_token(token: str) -> str | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        if payload["exp"] < time.time():
            return None
        return payload["sub"]
    except Exception:
        return None


def authenticate(username: str, password: str) -> str | None:
    conn = get_pg()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM admin_users WHERE username=%s", (username,))
    row = cur.fetchone()
    cur.close()
    if row and check_password(password, row[0]):
        return create_token(username)
    return None
