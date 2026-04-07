from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentReport:
    severity: str
    summary: str
    rationale: str
    recommended_action: str


def _feature_flag(features: Dict[str, Any], key: str, threshold: float) -> bool:
    value = float(features.get(key, 0) or 0)
    return value >= threshold


def build_prediction_report(predicted_label: str, confidence: float, features: Dict[str, Any]) -> AgentReport:
    risk_points = 0
    reasons = []

    if predicted_label.lower() != "normal":
        risk_points += 2
        reasons.append("model classified the traffic as anomaly")
    if confidence >= 95:
        risk_points += 2
        reasons.append("prediction confidence is very high")
    elif confidence >= 80:
        risk_points += 1
        reasons.append("prediction confidence is high")

    if _feature_flag(features, "serror_rate", 0.5):
        risk_points += 1
        reasons.append("SYN error rate is elevated")
    if _feature_flag(features, "srv_serror_rate", 0.5):
        risk_points += 1
        reasons.append("service-level SYN error rate is elevated")
    if _feature_flag(features, "rerror_rate", 0.5):
        risk_points += 1
        reasons.append("reject error rate is elevated")
    if _feature_flag(features, "count", 150):
        risk_points += 1
        reasons.append("connection count is unusually high")
    if _feature_flag(features, "srv_count", 150):
        risk_points += 1
        reasons.append("service count is unusually high")
    if _feature_flag(features, "dst_host_count", 200):
        risk_points += 1
        reasons.append("destination host activity is heavy")

    if risk_points >= 6:
        severity = "critical"
    elif risk_points >= 4:
        severity = "high"
    elif risk_points >= 2:
        severity = "medium"
    else:
        severity = "low"

    summary = (
        f"Traffic is predicted as {predicted_label} with {confidence:.2f}% confidence "
        f"and severity {severity}."
    )
    rationale = "Key reasons: " + (", ".join(reasons) if reasons else "no strong risk indicators were triggered.")

    if severity in {"critical", "high"}:
        action = "Isolate the source host, inspect firewall and system logs, and review recent connection spikes."
    elif severity == "medium":
        action = "Monitor the source closely, verify the traffic source, and review repeated failed connections."
    else:
        action = "Keep the event in history and continue normal monitoring."

    return AgentReport(
        severity=severity,
        summary=summary,
        rationale=rationale,
        recommended_action=action,
    )


def build_live_monitor_report(packet_count: int, bytes_seen: int, tcp_syn_ratio: float, unique_destinations: int) -> AgentReport:
    anomaly_score = min(
        100.0,
        (packet_count * 0.25) + (bytes_seen / 10000.0) + (tcp_syn_ratio * 40.0) + (unique_destinations * 2.0),
    )
    if anomaly_score >= 75:
        severity = "critical"
    elif anomaly_score >= 50:
        severity = "high"
    elif anomaly_score >= 25:
        severity = "medium"
    else:
        severity = "low"

    summary = (
        f"Live traffic window captured {packet_count} packets and {bytes_seen} bytes. "
        f"Calculated live anomaly score is {anomaly_score:.2f}."
    )
    rationale = (
        "The live monitor looks at traffic burst size, TCP SYN behavior, and spread across destinations "
        f"(SYN ratio {tcp_syn_ratio:.2f}, unique destinations {unique_destinations})."
    )
    if severity in {"critical", "high"}:
        action = "Investigate the active source immediately and compare with recent alert history."
    elif severity == "medium":
        action = "Continue monitoring and validate whether the traffic spike is expected."
    else:
        action = "Traffic window looks stable; no immediate action is required."

    return AgentReport(
        severity=severity,
        summary=summary,
        rationale=rationale,
        recommended_action=action,
    )

