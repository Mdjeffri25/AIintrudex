from __future__ import annotations

from io import StringIO
from typing import Dict

import pandas as pd
import requests
import streamlit as st

from nids_app.config import API_HOST, API_PORT
from nids_app.constants import FEATURE_COLUMNS


API_BASE = f"http://{API_HOST}:{API_PORT}/api"


def api_headers() -> Dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_post(path: str, payload: dict):
    return requests.post(f"{API_BASE}{path}", json=payload, headers=api_headers(), timeout=120)


def api_get(path: str):
    return requests.get(f"{API_BASE}{path}", headers=api_headers(), timeout=120)


def ensure_logged_in() -> bool:
    return bool(st.session_state.get("token"))


def inject_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(210, 230, 255, 0.75), transparent 28%),
                linear-gradient(135deg, #eef4ff 0%, #f9fbff 55%, #eef7f5 100%);
        }
        .hero {
            background: linear-gradient(135deg, #0b2a5b 0%, #204c8d 58%, #138a72 100%);
            color: #ffffff;
            padding: 2rem 2.2rem;
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(11, 42, 91, 0.22);
            margin-bottom: 1.4rem;
        }
        .hero h1 {
            margin: 0 0 0.45rem 0;
            font-size: 2.4rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0;
            font-size: 1.05rem;
            color: rgba(255,255,255,0.92);
        }
        .panel {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(32, 76, 141, 0.08);
            border-radius: 20px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 24px rgba(25, 40, 75, 0.08);
            margin-bottom: 1rem;
        }
        .mini-card {
            background: linear-gradient(180deg, #ffffff 0%, #f6f9ff 100%);
            border: 1px solid rgba(32, 76, 141, 0.1);
            border-radius: 18px;
            padding: 1rem;
            min-height: 118px;
            box-shadow: 0 8px 18px rgba(25, 40, 75, 0.07);
        }
        .mini-card .label {
            color: #5f6f89;
            font-size: 0.9rem;
            margin-bottom: 0.55rem;
        }
        .mini-card .value {
            color: #0f2346;
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }
        .mini-card .hint {
            margin-top: 0.55rem;
            color: #466285;
            font-size: 0.88rem;
        }
        .section-title {
            color: #102546;
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }
        .section-copy {
            color: #51647f;
            margin-bottom: 0.9rem;
        }
        .info-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            margin-right: 0.4rem;
            margin-bottom: 0.4rem;
            background: #e7f0ff;
            color: #16407f;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-ok {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #e8fff5 0%, #f2fffb 100%);
            border: 1px solid #b9efd8;
            color: #165a45;
        }
        .status-alert {
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: linear-gradient(135deg, #fff0f0 0%, #fff9f8 100%);
            border: 1px solid #f3c2c2;
            color: #8c2727;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth():
    inject_styles()
    st.markdown(
        """
        <div class="hero">
            <h1>AI Based Intrusion Detection Using Deep Learning</h1>
            <p>Sign in to analyze traffic, monitor live activity, save history, and generate user-friendly security explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])
    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">What This System Does</div>
                <div class="section-copy">
                    It studies network traffic behavior and tells whether the traffic looks normal or suspicious.
                    The upgraded version also stores per-user history and supports live monitoring.
                </div>
                <span class="info-chip">Deep learning detection</span>
                <span class="info-chip">CSV analysis</span>
                <span class="info-chip">Live monitoring</span>
                <span class="info-chip">User history</span>
                <span class="info-chip">Admin analytics</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True):
                response = api_post("/login", {"username": username, "password": password})
                if response.ok:
                    data = response.json()
                    st.session_state.token = data["token"]
                    st.session_state.user = data["user"]
                    st.rerun()
                else:
                    st.error(response.json().get("error", "Login failed"))
        with tab2:
            username = st.text_input("New username", key="register_username")
            password = st.text_input("New password", type="password", key="register_password")
            if st.button("Create account", use_container_width=True):
                response = api_post("/register", {"username": username, "password": password})
                if response.ok:
                    data = response.json()
                    st.session_state.token = data["token"]
                    st.session_state.user = data["user"]
                    st.rerun()
                else:
                    st.error(response.json().get("error", "Registration failed"))


def render_metric_cards(data):
    cards = [
        ("Predictions", data["total_predictions"], "Total detections saved for this account"),
        ("Intrusions", data["intrusion_predictions"], "Rows predicted as suspicious"),
        ("CSV Uploads", data["total_uploads"], "Batch analysis sessions"),
        ("Live Events", data["total_live_events"], "Live monitor sessions"),
    ]
    columns = st.columns(4)
    for column, (label, value, hint) in zip(columns, cards):
        with column:
            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_about_page():
    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Simple Explanation</div>
                <div class="section-copy">
                    This project works like a smart security guard for network traffic. It checks network behavior,
                    finds unusual patterns, and warns the user if the traffic looks suspicious.
                </div>
                <div class="section-title">Current Detection Core</div>
                <div class="section-copy">
                    The deep learning model reads 41 traffic features such as protocol, bytes, connection count,
                    and error rates, then predicts whether the traffic is <b>normal</b> or <b>anomaly</b>.
                </div>
                <div class="section-title">Upgraded 2026 Flow</div>
                <div class="section-copy">
                    User login, CSV upload, live capture, database history, admin analytics, and clear alert explanations
                    are now placed around the original model so the project becomes easier to use and easier to explain.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Project Highlights</div>
                <span class="info-chip">41-feature model</span>
                <span class="info-chip">User accounts</span>
                <span class="info-chip">Saved history</span>
                <span class="info-chip">CSV batch input</span>
                <span class="info-chip">Live packet capture</span>
                <span class="info-chip">Admin view</span>
                <span class="info-chip">Explainable alerts</span>
                <div style="height: 0.8rem;"></div>
                <div class="section-title">Best Line For Viva</div>
                <div class="section-copy">
                    The original project proved deep learning based intrusion detection. The upgraded version makes it more
                    practical by adding live monitoring, user-wise storage, alert explanation, and admin visibility.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_manual_prediction():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Single Traffic Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Use this page to test one network traffic record manually with the deep learning model.</div>',
        unsafe_allow_html=True,
    )

    defaults = {
        "duration": 0,
        "protocol_type": "tcp",
        "service": "http",
        "flag": "SF",
        "src_bytes": 181,
        "dst_bytes": 5450,
        "land": 0,
        "wrong_fragment": 0,
        "urgent": 0,
        "hot": 0,
        "num_failed_logins": 0,
        "logged_in": 1,
        "num_compromised": 0,
        "root_shell": 0,
        "su_attempted": 0,
        "num_root": 0,
        "num_file_creations": 0,
        "num_shells": 0,
        "num_access_files": 0,
        "num_outbound_cmds": 0,
        "is_host_login": 0,
        "is_guest_login": 0,
        "count": 9,
        "srv_count": 9,
        "serror_rate": 0.0,
        "srv_serror_rate": 0.0,
        "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0,
        "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0,
        "srv_diff_host_rate": 0.0,
        "dst_host_count": 9,
        "dst_host_srv_count": 9,
        "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0.0,
        "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0,
        "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0,
    }
    payload = {}
    cols = st.columns(2)
    for index, feature in enumerate(FEATURE_COLUMNS):
        with cols[index % 2]:
            payload[feature] = st.text_input(feature, value=str(defaults[feature]))

    if st.button("Analyze record", use_container_width=True):
        try:
            for key, value in list(payload.items()):
                if key not in {"protocol_type", "service", "flag"}:
                    payload[key] = float(value) if "." in value else int(value)
            response = api_post("/predict-row", payload)
            if response.ok:
                data = response.json()
                css_class = "status-ok" if data["predicted_label"].lower() == "normal" else "status-alert"
                st.markdown(
                    f'<div class="{css_class}"><b>{data["summary"]}</b><br>{data["rationale"]}<br><br>Recommended action: {data["recommended_action"]}</div>',
                    unsafe_allow_html=True,
                )
                st.json(data["probabilities"])
            else:
                st.error(response.json().get("error", "Prediction failed"))
        except ValueError:
            st.error("Please enter valid numbers for numeric fields.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_csv_prediction():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">CSV Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Upload a CSV file in the same 41-column format. This is useful for batch testing, lab data, or exported monitoring data.</div>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        content = uploaded.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        st.dataframe(df.head(), use_container_width=True)
        if st.button("Run CSV analysis", use_container_width=True):
            response = api_post("/predict-csv", {"filename": uploaded.name, "rows": df.to_dict(orient="records")})
            if response.ok:
                data = response.json()
                st.success(
                    f"Processed {data['total_rows']} rows. Intrusions: {data['intrusion_rows']}, Normal: {data['normal_rows']}."
                )
                st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
            else:
                st.error(response.json().get("error", "CSV prediction failed"))
    else:
        st.info("Example file available: `sample_nids_input.csv`")
    st.markdown("</div>", unsafe_allow_html=True)


def render_live_monitor():
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Live Network Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-copy">
            This page captures a short live traffic window from your machine, converts those packets into the model-style
            network features, and sends that feature record to the deep learning model.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Packet limit = how many packets to inspect in one live check. Capture seconds = how long to listen before building the live feature record."
    )
    st.caption(
        "The source or destination IP may look different from your own because the sniffer sees both local and remote endpoints in the captured traffic window."
    )

    interface = st.text_input("Interface name (leave blank for default)")
    packet_limit = st.slider("Packet limit", 10, 150, 30)
    timeout = st.slider("Capture seconds", 5, 30, 10)

    if st.button("Start live capture", use_container_width=True):
        response = api_post("/live-monitor", {"interface": interface or None, "packet_limit": packet_limit, "timeout": timeout})
        if response.ok:
            data = response.json()
            css_class = "status-ok" if data["predicted_label"].lower() == "normal" else "status-alert"
            st.markdown(
                f'<div class="{css_class}"><b>{data["summary"]}</b><br>{data["rationale"]}<br><br>Recommended action: {data["recommended_action"]}</div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted Label", data["predicted_label"])
            c2.metric("Confidence", f'{data["confidence"]:.2f}%')
            c3.metric("Protocol", data["protocol"])
            c4.metric("Service", data["service"])
            st.write(f"Top source IP seen: `{data['source_ip']}`")
            st.write(f"Top destination IP seen: `{data['destination_ip']}`")
            with st.expander("Extracted live feature record"):
                st.json(data["feature_record"])
            with st.expander("Class probabilities"):
                st.json(data["probabilities"])
        else:
            st.error(response.json().get("error", "Live monitoring failed"))
    st.markdown("</div>", unsafe_allow_html=True)


def render_history():
    response = api_get("/history")
    if not response.ok:
        st.error("Could not load history.")
        return
    data = response.json()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="panel"><div class="section-title">Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["predictions"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="panel"><div class="section-title">CSV Upload History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["uploads"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="panel"><div class="section-title">Live Monitoring History</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(data["live_events"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_admin_dashboard():
    response = api_get("/admin/overview")
    if not response.ok:
        st.error(response.json().get("error", "Admin overview unavailable"))
        return
    data = response.json()
    st.markdown('<div class="panel"><div class="section-title">User Overview</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(data["users"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel"><div class="section-title">Recent Suspicious Alerts</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["recent_alerts"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel"><div class="section-title">Recent Live Events</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["live_events"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="NIDS 2026", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    if not ensure_logged_in():
        render_auth()
        return

    dashboard_response = api_get("/dashboard")
    if not dashboard_response.ok:
        st.error("Backend is not reachable. Start `python run_backend.py` first.")
        return
    dashboard_data = dashboard_response.json()

    pages = ["About", "Dashboard", "Single Prediction", "CSV Upload", "Live Monitor", "History"]
    if dashboard_data.get("role") == "admin":
        pages.append("Admin Dashboard")

    with st.sidebar:
        st.markdown(
            f"""
            <div class="panel">
                <div class="section-title">User Session</div>
                <div class="section-copy" style="margin-bottom:0.4rem;">Logged in as <b>{st.session_state.user['username']}</b></div>
                <span class="info-chip">{dashboard_data.get("role", "user").title()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio("Navigation", pages)
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    st.markdown(
        """
        <div class="hero">
            <h1>AI Based Intrusion Detection Using Deep Learning</h1>
            <p>User-friendly intrusion analysis with deep learning, live monitoring, saved history, and explainable alert output.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(dashboard_data)
    st.write("")

    if page == "About":
        render_about_page()
    elif page == "Dashboard":
        st.markdown(
            """
            <div class="panel">
                <div class="section-title">Quick Overview</div>
                <div class="section-copy">
                    Use manual prediction for one record, CSV upload for many rows, and live monitor for real-time traffic windows.
                    Every result is saved to the user history automatically.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif page == "Single Prediction":
        render_manual_prediction()
    elif page == "CSV Upload":
        render_csv_prediction()
    elif page == "Live Monitor":
        render_live_monitor()
    elif page == "History":
        render_history()
    elif page == "Admin Dashboard":
        render_admin_dashboard()


if __name__ == "__main__":
    main()
