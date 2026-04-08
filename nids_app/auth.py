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


def ensure_admin_account(preferred_username: str | None = None) -> None:
    admin_count = fetch_one("SELECT COUNT(*) AS count FROM users WHERE role = 'admin'")
    if admin_count and admin_count["count"] > 0:
        return

    candidate = None
    if preferred_username:
        candidate = fetch_one("SELECT id FROM users WHERE lower(username) = lower(?)", (preferred_username,))
    if not candidate:
        candidate = fetch_one("SELECT id FROM users WHERE lower(username) = 'admin' ORDER BY id DESC LIMIT 1")
    if not candidate:
        candidate = fetch_one("SELECT id FROM users ORDER BY id DESC LIMIT 1")
    if candidate:
        execute("UPDATE users SET role = 'admin' WHERE id = ?", (candidate["id"],))


def authenticate_user(username: str, password: str) -> Optional[dict]:
    ensure_admin_account(username)
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


def list_users() -> list[dict]:
    rows = fetch_one("SELECT COUNT(*) AS count FROM users")
    if not rows:
        return []
    from .database import fetch_all

    return [dict(row) for row in fetch_all("SELECT id, username, role, created_at FROM users ORDER BY username")]


def update_user_password(username: str, new_password: str) -> bool:
    user = fetch_one("SELECT id FROM users WHERE username = ?", (username,))
    if not user:
        return False
    password_hash = generate_password_hash(new_password)
    execute(
        "UPDATE users SET password_hash = ? WHERE username = ?",
        (password_hash, username),
    )
    return True
