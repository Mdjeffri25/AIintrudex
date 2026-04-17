from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict

from .audit import write_audit_log
from .database import execute, fetch_one, utc_now
from .live_monitor import LiveCaptureError, capture_live_window
from .notifier import send_email_alert, send_sms_alert


@dataclass
class MonitorJob:
    user_id: int
    stop_event: threading.Event
    thread: threading.Thread


_jobs: Dict[int, MonitorJob] = {}


def _load_settings(user_id: int) -> dict:
    row = fetch_one("SELECT * FROM alert_settings WHERE user_id = ?", (user_id,))
    return dict(row) if row else {}


def _store_live_event(user_id: int, result: dict) -> int:
    return execute(
        """
        INSERT INTO live_events
        (user_id, source_ip, destination_ip, protocol, packet_count, bytes_seen, anomaly_score, severity, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            result["source_ip"],
            result["destination_ip"],
            result["protocol"],
            result["packet_count"],
            result["bytes_seen"],
            result["confidence"],
            result["severity"],
            result["summary"],
            utc_now(),
        ),
    )


def _run_monitor_loop(user_id: int, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        settings = _load_settings(user_id)
        if not settings or not settings.get("monitor_enabled"):
            break

        packet_limit = int(settings.get("packet_limit") or 30)
        capture_seconds = int(settings.get("capture_seconds") or 10)

        try:
            result = capture_live_window(packet_limit=packet_limit, timeout=capture_seconds)
            event_id = _store_live_event(user_id, result)
            write_audit_log("continuous_monitor_event", {"user_id": user_id, "event_id": event_id}, user_id)

            if result["predicted_label"].lower() != "normal":
                subject = f"NIDS Alert: {result['predicted_label']} detected"
                body = (
                    f"User ID: {user_id}\n"
                    f"Predicted label: {result['predicted_label']}\n"
                    f"Confidence: {result['confidence']:.2f}%\n"
                    f"Severity: {result['severity']}\n"
                    f"Summary: {result['summary']}\n"
                    f"Top source IP: {result['source_ip']}\n"
                    f"Top destination IP: {result['destination_ip']}\n"
                    f"Recommended action: {result['recommended_action']}\n"
                )
                send_email_alert(settings, subject, body)
                send_sms_alert(
                    settings,
                    (
                        f"AI INTRUDEX ALERT: {result['predicted_label']} "
                        f"({result['confidence']:.2f}%). "
                        f"Severity {result['severity']}. "
                        f"Source {result['source_ip']} -> {result['destination_ip']}."
                    ),
                )
        except LiveCaptureError as exc:
            write_audit_log(
                "continuous_monitor_error",
                {"user_id": user_id, "error": str(exc), "stopped": True},
                user_id,
            )
            execute(
                "UPDATE alert_settings SET monitor_enabled = 0, updated_at = ? WHERE user_id = ?",
                (utc_now(), user_id),
            )
            break
        except Exception as exc:
            write_audit_log(
                "continuous_monitor_error",
                {"user_id": user_id, "error": str(exc), "stopped": False},
                user_id,
            )

        sleep_seconds = max(capture_seconds, 5)
        stop_event.wait(sleep_seconds)


def start_monitor(user_id: int) -> bool:
    if user_id in _jobs and _jobs[user_id].thread.is_alive():
        return False
    stop_event = threading.Event()
    thread = threading.Thread(target=_run_monitor_loop, args=(user_id, stop_event), daemon=True)
    _jobs[user_id] = MonitorJob(user_id=user_id, stop_event=stop_event, thread=thread)
    thread.start()
    return True


def stop_monitor(user_id: int) -> bool:
    job = _jobs.get(user_id)
    if not job:
        return False
    job.stop_event.set()
    return True


def monitor_status(user_id: int) -> dict:
    job = _jobs.get(user_id)
    return {
        "running": bool(job and job.thread.is_alive()),
    }
