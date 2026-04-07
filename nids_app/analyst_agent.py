from __future__ import annotations

from typing import Any, Dict, List


def build_ai_brief(predictions: List[Dict[str, Any]], live_events: List[Dict[str, Any]]) -> Dict[str, Any]:
    suspicious_predictions = [row for row in predictions if str(row.get("predicted_label", "")).lower() != "normal"]
    high_live = [row for row in live_events if str(row.get("severity", "")).lower() in {"high", "critical", "medium"}]

    total_suspicious = len(suspicious_predictions)
    total_live = len(high_live)

    if total_suspicious == 0 and total_live == 0:
        status = "stable"
        summary = "AIintrudex agent review found no recent high-risk suspicious activity in the latest saved results."
        action = "Continue monitoring and keep CSV or live monitoring enabled."
    elif total_suspicious >= 3 or total_live >= 3:
        status = "warning"
        summary = (
            f"AIintrudex agent found repeated suspicious activity: {total_suspicious} suspicious prediction records and "
            f"{total_live} elevated live monitoring events."
        )
        action = "Prioritize firewall review, inspect repeated source IPs, and keep continuous monitoring enabled."
    else:
        status = "attention"
        summary = (
            f"AIintrudex agent found limited suspicious activity: {total_suspicious} suspicious prediction records and "
            f"{total_live} elevated live monitoring events."
        )
        action = "Review the most recent alert details and confirm whether the activity is expected."

    steps = [
        "Collect recent CSV and live monitoring evidence.",
        "Rank suspicious patterns by severity and frequency.",
        "Generate a short response plan for the user.",
    ]

    return {
        "status": status,
        "summary": summary,
        "recommended_action": action,
        "steps": steps,
        "suspicious_prediction_count": total_suspicious,
        "elevated_live_count": total_live,
    }

