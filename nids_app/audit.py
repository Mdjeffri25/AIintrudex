import hashlib
from typing import Any, Dict, Optional

from .database import execute, fetch_one, json_dumps, utc_now


def write_audit_log(event_type: str, payload: Dict[str, Any], user_id: Optional[int] = None) -> None:
    previous = fetch_one("SELECT record_hash FROM audit_logs ORDER BY id DESC LIMIT 1")
    previous_hash = previous["record_hash"] if previous else None
    payload_json = json_dumps(payload)
    record_hash = hashlib.sha256(f"{previous_hash or ''}|{event_type}|{payload_json}".encode("utf-8")).hexdigest()
    execute(
        """
        INSERT INTO audit_logs (user_id, event_type, record_hash, previous_hash, payload, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (user_id, event_type, record_hash, previous_hash, payload_json, utc_now()),
    )
