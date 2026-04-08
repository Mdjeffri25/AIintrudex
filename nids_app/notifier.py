from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any, Dict

import requests


def _split_targets(raw_value: str) -> list[str]:
    return [item.strip() for item in (raw_value or "").split(",") if item.strip()]


def send_email_alert(settings: Dict[str, Any], subject: str, body: str) -> tuple[bool, str]:
    if not settings.get("email_enabled"):
        return False, "Email alerts disabled"

    host = (settings.get("smtp_host") or "").strip()
    port = int(settings.get("smtp_port") or 0)
    username = (settings.get("smtp_username") or "").strip()
    password = settings.get("smtp_password") or ""
    recipients = _split_targets(settings.get("email_recipient") or "")

    if not all([host, port, username, password]) or not recipients:
        return False, "SMTP configuration incomplete"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = username
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls()
            smtp.login(username, password)
            smtp.send_message(message)
        return True, "Email sent"
    except Exception as exc:
        return False, f"Email failed: {exc}"


def send_sms_alert(settings: Dict[str, Any], body: str) -> tuple[bool, str]:
    if not settings.get("sms_enabled"):
        return False, "SMS alerts disabled"

    account_sid = (settings.get("twilio_account_sid") or "").strip()
    auth_token = (settings.get("twilio_auth_token") or "").strip()
    messaging_service_sid = (settings.get("twilio_messaging_service_sid") or "").strip()
    recipients = _split_targets(settings.get("sms_number") or "")

    if not all([account_sid, auth_token, messaging_service_sid]) or not recipients:
        return False, "Twilio SMS configuration incomplete"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    failures: list[str] = []

    for recipient in recipients:
        try:
            response = requests.post(
                url,
                auth=(account_sid, auth_token),
                data={
                    "To": recipient,
                    "MessagingServiceSid": messaging_service_sid,
                    "Body": body,
                },
                timeout=20,
            )
            if response.status_code >= 400:
                failures.append(f"{recipient}: {response.text[:120]}")
        except Exception as exc:
            failures.append(f"{recipient}: {exc}")

    if failures:
        return False, " | ".join(failures[:3])
    return True, "SMS sent"
