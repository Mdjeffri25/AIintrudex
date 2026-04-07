from __future__ import annotations

from functools import wraps
from typing import Any, Dict, List

import pandas as pd
from flask import Flask, jsonify, request

from .agent_service import build_prediction_report
from .analyst_agent import build_ai_brief
from .audit import write_audit_log
from .auth import authenticate_user, create_session, create_user, get_user_by_token, update_user_password
from .database import execute, fetch_all, fetch_one, init_db, utc_now
from .live_monitor import capture_live_window
from .monitor_manager import monitor_status, start_monitor, stop_monitor
from .model_service import get_available_models, predict_records
from .notifier import send_email_alert


app = Flask(__name__)


def json_error(message: str, status: int = 400):
    return jsonify({"error": message}), status


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return json_error("Missing bearer token", 401)
        token = auth_header.split(" ", 1)[1].strip()
        user = get_user_by_token(token)
        if not user:
            return json_error("Invalid session token", 401)
        return fn(user, *args, **kwargs)

    return wrapper


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/register")
def register():
    payload = request.get_json(force=True)
    username = (payload.get("username") or "").strip()
    password = payload.get("password") or ""
    if len(username) < 3:
        return json_error("Username must be at least 3 characters")
    if len(password) < 4:
        return json_error("Password must be at least 4 characters")
    if fetch_one("SELECT id FROM users WHERE username = ?", (username,)):
        return json_error("Username already exists")

    existing_users = fetch_one("SELECT COUNT(*) AS count FROM users")
    role = "admin" if username.lower() == "admin" or existing_users["count"] == 0 else "user"
    user_id = create_user(username, password, role=role)
    token = create_session(user_id)
    write_audit_log("register", {"username": username}, user_id)
    return jsonify({"token": token, "user": {"id": user_id, "username": username, "role": role}})


@app.post("/api/login")
def login():
    payload = request.get_json(force=True)
    user = authenticate_user((payload.get("username") or "").strip(), payload.get("password") or "")
    if not user:
        return json_error("Invalid username or password", 401)
    token = create_session(user["id"])
    write_audit_log("login", {"username": user["username"]}, user["id"])
    return jsonify({"token": token, "user": {"id": user["id"], "username": user["username"], "role": user["role"]}})


@app.get("/api/me")
@require_auth
def me(user):
    return jsonify({"user": user})


@app.post("/api/predict-row")
@require_auth
def predict_row(user):
    payload = request.get_json(force=True)
    model_name = (payload.pop("model_name", "kdd") or "kdd").strip().lower()
    records = [payload]
    result = predict_records(records, model_name=model_name)[0]
    report = build_prediction_report(result.predicted_label, result.confidence, result.features)
    prediction_id = execute(
        """
        INSERT INTO predictions
        (user_id, upload_id, model_name, source_type, predicted_label, confidence, severity, summary, recommended_action, feature_snapshot, created_at)
        VALUES (?, NULL, ?, 'manual', ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
            model_name,
            result.predicted_label,
            result.confidence,
            report.severity,
            report.summary,
            report.recommended_action,
            pd.Series(result.features).to_json(),
            utc_now(),
        ),
    )
    write_audit_log("predict_row", {"prediction_id": prediction_id, "label": result.predicted_label}, user["id"])
    return jsonify(
        {
            "prediction_id": prediction_id,
            "predicted_label": result.predicted_label,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "model_name": model_name,
            "severity": report.severity,
            "summary": report.summary,
            "rationale": report.rationale,
            "recommended_action": report.recommended_action,
        }
    )


@app.post("/api/predict-csv")
@require_auth
def predict_csv(user):
    payload = request.get_json(force=True)
    rows: List[Dict[str, Any]] = payload.get("rows") or []
    filename = payload.get("filename") or "uploaded.csv"
    model_name = (payload.get("model_name") or "kdd").strip().lower()
    if not rows:
        return json_error("No rows provided")

    results = predict_records(rows, model_name=model_name)
    intrusion_rows = 0
    upload_id = execute(
        """
        INSERT INTO uploads (user_id, filename, total_rows, intrusion_rows, created_at)
        VALUES (?, ?, ?, 0, ?)
        """,
        (user["id"], filename, len(rows), utc_now()),
    )

    response_rows = []
    for result in results:
        report = build_prediction_report(result.predicted_label, result.confidence, result.features)
        if result.predicted_label.lower() != "normal":
            intrusion_rows += 1
        prediction_id = execute(
            """
            INSERT INTO predictions
            (user_id, upload_id, model_name, source_type, predicted_label, confidence, severity, summary, recommended_action, feature_snapshot, created_at)
            VALUES (?, ?, ?, 'csv', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                upload_id,
                model_name,
                result.predicted_label,
                result.confidence,
                report.severity,
                report.summary,
                report.recommended_action,
                pd.Series(result.features).to_json(),
                utc_now(),
            ),
        )
        response_rows.append(
            {
                "prediction_id": prediction_id,
                "model_name": model_name,
                "predicted_label": result.predicted_label,
                "confidence": result.confidence,
                "severity": report.severity,
                "summary": report.summary,
            }
        )

    execute("UPDATE uploads SET intrusion_rows = ? WHERE id = ?", (intrusion_rows, upload_id))
    write_audit_log("predict_csv", {"upload_id": upload_id, "rows": len(rows)}, user["id"])
    return jsonify(
        {
            "upload_id": upload_id,
            "filename": filename,
            "total_rows": len(rows),
            "intrusion_rows": intrusion_rows,
            "normal_rows": len(rows) - intrusion_rows,
            "results": response_rows,
        }
    )


@app.post("/api/live-monitor")
@require_auth
def live_monitor(user):
    payload = request.get_json(silent=True) or {}
    interface = payload.get("interface")
    packet_limit = int(payload.get("packet_limit") or 30)
    timeout = int(payload.get("timeout") or 10)

    try:
        result = capture_live_window(interface=interface, packet_limit=packet_limit, timeout=timeout)
    except Exception as exc:
        return json_error(f"Live capture failed: {exc}", 500)

    event_id = execute(
        """
        INSERT INTO live_events
        (user_id, source_ip, destination_ip, protocol, packet_count, bytes_seen, anomaly_score, severity, summary, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
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
    write_audit_log("live_monitor", {"event_id": event_id, "severity": result["severity"]}, user["id"])
    result["event_id"] = event_id
    return jsonify(result)


@app.get("/api/history")
@require_auth
def history(user):
    predictions = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, model_name, source_type, predicted_label, confidence, severity, summary, recommended_action, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 50
            """,
            (user["id"],),
        )
    ]
    uploads = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, filename, total_rows, intrusion_rows, created_at
            FROM uploads
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (user["id"],),
        )
    ]
    live_events = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, source_ip, destination_ip, protocol, packet_count, bytes_seen, anomaly_score, severity, summary, created_at
            FROM live_events
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (user["id"],),
        )
    ]
    return jsonify({"predictions": predictions, "uploads": uploads, "live_events": live_events})


@app.get("/api/ai-analyst")
@require_auth
def ai_analyst(user):
    predictions = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, model_name, source_type, predicted_label, confidence, severity, summary, recommended_action, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 25
            """,
            (user["id"],),
        )
    ]
    live_events = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, source_ip, destination_ip, protocol, anomaly_score, severity, summary, created_at
            FROM live_events
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 25
            """,
            (user["id"],),
        )
    ]
    brief = build_ai_brief(predictions, live_events)
    write_audit_log("ai_analyst_review", {"user_id": user["id"], "status": brief["status"]}, user["id"])
    return jsonify(brief)


@app.post("/api/ai-chat")
@require_auth
def ai_chat(user):
    payload = request.get_json(force=True)
    message = (payload.get("message") or "").strip()
    if not message:
        return json_error("Message is required")

    predictions = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, model_name, source_type, predicted_label, confidence, severity, summary, recommended_action, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user["id"],),
        )
    ]
    live_events = [
        dict(row)
        for row in fetch_all(
            """
            SELECT id, source_ip, destination_ip, protocol, anomaly_score, severity, summary, created_at
            FROM live_events
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (user["id"],),
        )
    ]
    suspicious_predictions = [row for row in predictions if str(row.get("predicted_label", "")).lower() != "normal"]
    elevated_live = [row for row in live_events if str(row.get("severity", "")).lower() in {"medium", "high", "critical"}]
    lower = message.lower()

    if "why" in lower and ("alert" in lower or "generated" in lower):
        if suspicious_predictions:
            latest = suspicious_predictions[0]
            answer = (
                f"The latest suspicious prediction was generated because the model classified the traffic as "
                f"{latest['predicted_label']} with {latest['confidence']:.2f}% confidence. "
                f"Summary: {latest['summary']} Recommended action: {latest['recommended_action']}"
            )
        elif elevated_live:
            latest = elevated_live[0]
            answer = (
                f"The latest live alert was generated from monitoring activity around source {latest.get('source_ip')} "
                f"with severity {latest.get('severity')}. Summary: {latest.get('summary')}"
            )
        else:
            answer = "No recent suspicious alert is available, so there is no alert reason to explain right now."
    elif "latest suspicious" in lower or "suspicious activity" in lower:
        answer = (
            f"I found {len(suspicious_predictions)} suspicious prediction records and {len(elevated_live)} elevated live events "
            f"in your recent history."
        )
    elif "dataset" in lower or "unsw" in lower or "kdd" in lower:
        answer = (
            "The website supports two model paths: the active KDD-style 41-feature model and an UNSW-NB15 enhanced model path. "
            "If UNSW artifacts are available, the UI model selector can route analysis to the UNSW model. "
            "If they are not trained yet, the UI keeps UNSW visible but marked unavailable."
        )
    elif "model" in lower and ("use" in lower or "using" in lower or "what model" in lower):
        answer = (
            "The website currently provides a KDD 41-feature model path and can also expose an UNSW-NB15 model path when its artifacts are trained and saved. "
            "Manual detection is currently optimized for KDD, while CSV analysis can be routed to the selected model."
        )
    elif "summarize" in lower and ("live" in lower or "monitoring" in lower):
        if live_events:
            latest = live_events[0]
            answer = (
                f"Latest live monitoring summary: protocol {latest.get('protocol')}, source {latest.get('source_ip')}, "
                f"destination {latest.get('destination_ip')}, severity {latest.get('severity')}. "
                f"{latest.get('summary')}"
            )
        else:
            answer = "There are no saved live monitoring events yet."
    elif "what should i do" in lower or "next" in lower or "action" in lower:
        if suspicious_predictions:
            latest = suspicious_predictions[0]
            answer = f"My recommendation is: {latest['recommended_action']}"
        elif elevated_live:
            answer = "Keep continuous monitoring enabled, inspect repeated source IPs, and review recent suspicious traffic windows."
        else:
            answer = "Continue monitoring, keep CSV or live monitoring active, and review alerts when new suspicious activity appears."
    elif "report" in lower:
        if suspicious_predictions:
            latest = suspicious_predictions[0]
            answer = (
                f"Incident report draft: suspicious traffic detected as {latest['predicted_label']} with "
                f"{latest['confidence']:.2f}% confidence on {latest['created_at']}. "
                f"Summary: {latest['summary']} Recommended action: {latest['recommended_action']}"
            )
        else:
            answer = "There is no suspicious record available to generate a report from right now."
    elif "41" in lower or "features" in lower:
        answer = (
            "The KDD model uses 41 features. UNSW-NB15 uses a different feature layout, so the app exposes it through a separate model path and selector rather than mixing both forms into one fixed input structure."
        )
    else:
        answer = (
            "I can help with alert explanation, suspicious activity summary, live monitoring summary, next-step guidance, "
            "incident report drafting, and explain which dataset and model flow the website is currently using."
        )

    write_audit_log("ai_chat", {"user_id": user["id"], "message": message[:120]}, user["id"])
    return jsonify({"reply": answer})


@app.get("/api/dashboard")
@require_auth
def dashboard(user):
    total_predictions = fetch_one("SELECT COUNT(*) AS count FROM predictions WHERE user_id = ?", (user["id"],))["count"]
    total_uploads = fetch_one("SELECT COUNT(*) AS count FROM uploads WHERE user_id = ?", (user["id"],))["count"]
    total_live_events = fetch_one("SELECT COUNT(*) AS count FROM live_events WHERE user_id = ?", (user["id"],))["count"]
    intrusion_predictions = fetch_one(
        "SELECT COUNT(*) AS count FROM predictions WHERE user_id = ? AND lower(predicted_label) != 'normal'",
        (user["id"],),
    )["count"]
    return jsonify(
        {
            "username": user["username"],
            "role": user["role"],
            "total_predictions": total_predictions,
            "intrusion_predictions": intrusion_predictions,
            "total_uploads": total_uploads,
            "total_live_events": total_live_events,
            "monitor_running": monitor_status(user["id"])["running"],
            "available_models": get_available_models(),
        }
    )


@app.get("/api/models")
@require_auth
def models(user):
    return jsonify({"models": get_available_models()})


@app.get("/api/admin/overview")
@require_auth
def admin_overview(user):
    if user["role"] != "admin":
        return json_error("Admin access required", 403)

    users = [
        dict(row)
        for row in fetch_all(
            """
            SELECT users.id, users.username, users.role, users.created_at,
                   COUNT(DISTINCT uploads.id) AS uploads_count,
                   COUNT(DISTINCT predictions.id) AS predictions_count,
                   SUM(CASE WHEN lower(predictions.predicted_label) != 'normal' THEN 1 ELSE 0 END) AS intrusion_count
            FROM users
            LEFT JOIN uploads ON uploads.user_id = users.id
            LEFT JOIN predictions ON predictions.user_id = users.id
            GROUP BY users.id, users.username, users.role, users.created_at
            ORDER BY users.id DESC
            """
        )
    ]
    recent_alerts = [
        dict(row)
        for row in fetch_all(
            """
            SELECT predictions.id, users.username, predictions.source_type, predictions.predicted_label,
                   predictions.confidence, predictions.severity, predictions.summary, predictions.created_at
            FROM predictions
            JOIN users ON users.id = predictions.user_id
            WHERE lower(predictions.predicted_label) != 'normal'
            ORDER BY predictions.id DESC
            LIMIT 20
            """
        )
    ]
    live_events = [
        dict(row)
        for row in fetch_all(
            """
            SELECT live_events.id, users.username, live_events.source_ip, live_events.destination_ip,
                   live_events.protocol, live_events.anomaly_score, live_events.severity, live_events.summary, live_events.created_at
            FROM live_events
            JOIN users ON users.id = live_events.user_id
            ORDER BY live_events.id DESC
            LIMIT 20
            """
        )
    ]
    return jsonify({"users": users, "recent_alerts": recent_alerts, "live_events": live_events})


@app.post("/api/admin/reset-password")
@require_auth
def admin_reset_password(user):
    if user["role"] != "admin":
        return json_error("Admin access required", 403)
    payload = request.get_json(force=True)
    username = (payload.get("username") or "").strip()
    new_password = payload.get("new_password") or ""
    if len(username) < 3:
        return json_error("Username is required")
    if len(new_password) < 4:
        return json_error("New password must be at least 4 characters")
    updated = update_user_password(username, new_password)
    if not updated:
        return json_error("User not found", 404)
    write_audit_log("admin_reset_password", {"admin": user["username"], "target": username}, user["id"])
    return jsonify({"updated": True, "username": username})


@app.get("/api/alert-settings")
@require_auth
def get_alert_settings(user):
    row = fetch_one("SELECT * FROM alert_settings WHERE user_id = ?", (user["id"],))
    if not row:
        return jsonify(
            {
                "monitor_enabled": 0,
                "packet_limit": 30,
                "capture_seconds": 10,
                "email_enabled": 0,
                "email_recipient": "",
                "smtp_host": "",
                "smtp_port": 587,
                "smtp_username": "",
                "smtp_password": "",
                "sms_enabled": 0,
                "sms_number": "",
                "monitor_running": monitor_status(user["id"])["running"],
            }
        )
    data = dict(row)
    data["monitor_running"] = monitor_status(user["id"])["running"]
    return jsonify(data)


@app.post("/api/alert-settings")
@require_auth
def save_alert_settings(user):
    payload = request.get_json(force=True)
    settings = {
        "monitor_enabled": int(bool(payload.get("monitor_enabled"))),
        "packet_limit": int(payload.get("packet_limit") or 30),
        "capture_seconds": int(payload.get("capture_seconds") or 10),
        "email_enabled": int(bool(payload.get("email_enabled"))),
        "email_recipient": payload.get("email_recipient") or "",
        "smtp_host": payload.get("smtp_host") or "",
        "smtp_port": int(payload.get("smtp_port") or 587),
        "smtp_username": payload.get("smtp_username") or "",
        "smtp_password": payload.get("smtp_password") or "",
        "sms_enabled": int(bool(payload.get("sms_enabled"))),
        "sms_number": payload.get("sms_number") or "",
        "updated_at": utc_now(),
    }
    exists = fetch_one("SELECT user_id FROM alert_settings WHERE user_id = ?", (user["id"],))
    if exists:
        execute(
            """
            UPDATE alert_settings
            SET monitor_enabled = ?, packet_limit = ?, capture_seconds = ?, email_enabled = ?,
                email_recipient = ?, smtp_host = ?, smtp_port = ?, smtp_username = ?, smtp_password = ?,
                sms_enabled = ?, sms_number = ?, updated_at = ?
            WHERE user_id = ?
            """,
            (
                settings["monitor_enabled"],
                settings["packet_limit"],
                settings["capture_seconds"],
                settings["email_enabled"],
                settings["email_recipient"],
                settings["smtp_host"],
                settings["smtp_port"],
                settings["smtp_username"],
                settings["smtp_password"],
                settings["sms_enabled"],
                settings["sms_number"],
                settings["updated_at"],
                user["id"],
            ),
        )
    else:
        execute(
            """
            INSERT INTO alert_settings
            (user_id, monitor_enabled, packet_limit, capture_seconds, email_enabled, email_recipient,
             smtp_host, smtp_port, smtp_username, smtp_password, sms_enabled, sms_number, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                settings["monitor_enabled"],
                settings["packet_limit"],
                settings["capture_seconds"],
                settings["email_enabled"],
                settings["email_recipient"],
                settings["smtp_host"],
                settings["smtp_port"],
                settings["smtp_username"],
                settings["smtp_password"],
                settings["sms_enabled"],
                settings["sms_number"],
                settings["updated_at"],
            ),
        )
    write_audit_log("save_alert_settings", {"user_id": user["id"]}, user["id"])
    settings["monitor_running"] = monitor_status(user["id"])["running"]
    return jsonify(settings)


@app.post("/api/monitor/start")
@require_auth
def start_monitoring(user):
    settings = fetch_one("SELECT * FROM alert_settings WHERE user_id = ?", (user["id"],))
    if not settings:
        return json_error("Save alert settings before starting continuous monitoring")
    execute(
        "UPDATE alert_settings SET monitor_enabled = 1, updated_at = ? WHERE user_id = ?",
        (utc_now(), user["id"]),
    )
    started = start_monitor(user["id"])
    write_audit_log("start_monitor", {"user_id": user["id"], "started": started}, user["id"])
    return jsonify({"started": started, "monitor_running": monitor_status(user["id"])["running"]})


@app.post("/api/monitor/stop")
@require_auth
def stop_monitoring(user):
    stopped = stop_monitor(user["id"])
    execute(
        "UPDATE alert_settings SET monitor_enabled = 0, updated_at = ? WHERE user_id = ?",
        (utc_now(), user["id"]),
    )
    write_audit_log("stop_monitor", {"user_id": user["id"], "stopped": stopped}, user["id"])
    return jsonify({"stopped": stopped, "monitor_running": monitor_status(user["id"])["running"]})


@app.post("/api/alert-settings/test")
@require_auth
def test_alert_settings(user):
    row = fetch_one("SELECT * FROM alert_settings WHERE user_id = ?", (user["id"],))
    if not row:
        return json_error("Save alert settings first")
    settings = dict(row)
    subject = "Intrudex Test Alert"
    body = (
        "This is a test alert from Intrudex.\n"
        f"User: {user['username']}\n"
        "The alert system is reachable and the interface test was triggered."
    )
    sent, message = send_email_alert(settings, subject, body)
    write_audit_log("test_alert", {"user_id": user["id"], "sent": sent, "message": message}, user["id"])
    if sent:
        return jsonify({"sent": True, "mode": "email", "message": message})
    return jsonify({"sent": True, "mode": "demo", "message": "Demo alert sent popup shown because email settings are incomplete."})


def create_app():
    init_db()
    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=8000, debug=True)
