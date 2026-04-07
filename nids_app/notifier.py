from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any, Dict


def send_email_alert(settings: Dict[str, Any], subject: str, body: str) -> tuple[bool, str]:
    if not settings.get("email_enabled"):
        return False, "Email alerts disabled"

    host = (settings.get("smtp_host") or "").strip()
    port = int(settings.get("smtp_port") or 0)
    username = (settings.get("smtp_username") or "").strip()
    password = settings.get("smtp_password") or ""
    recipient = (settings.get("email_recipient") or "").strip()

    if not all([host, port, username, password, recipient]):
        return False, "SMTP configuration incomplete"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = username
    message["To"] = recipient
    message.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        return True, "Email sent"
    except Exception as exc:
        return False, f"Email failed: {exc}"

