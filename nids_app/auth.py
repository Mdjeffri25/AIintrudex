import secrets
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

from .database import execute, fetch_one, utc_now


def create_user(username: str, password: str, role: str = "user") -> int:
    password_hash = generate_password_hash(password)
    return execute(
        """
        INSERT INTO users (username, password_hash, role, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (username, password_hash, role, utc_now()),
    )


def authenticate_user(username: str, password: str) -> Optional[dict]:
    row = fetch_one("SELECT * FROM users WHERE username = ?", (username,))
    if not row:
        return None
    if not check_password_hash(row["password_hash"], password):
        return None
    return dict(row)


def create_session(user_id: int) -> str:
    token = secrets.token_hex(24)
    execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, utc_now()),
    )
    return token


def get_user_by_token(token: str) -> Optional[dict]:
    row = fetch_one(
        """
        SELECT users.id, users.username, users.role, users.created_at
        FROM sessions
        JOIN users ON users.id = sessions.user_id
        WHERE sessions.token = ?
        """,
        (token,),
    )
    return dict(row) if row else None

