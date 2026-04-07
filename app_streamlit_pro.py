from __future__ import annotations

from io import StringIO
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import socket

from nids_app.config import API_HOST, API_PORT
from nids_app.constants import FEATURE_COLUMNS


API_BASE = f"http://{API_HOST}:{API_PORT}/api"
PRODUCT_NAME = "AIintrudex"

MODEL_ARCHITECTURE = [
    "Input Layer: 41 features",
    "Hidden Layer 1: 128 neurons (ReLU) + BatchNorm + Dropout 30%",
    "Hidden Layer 2: 64 neurons (ReLU) + BatchNorm + Dropout 30%",
    "Hidden Layer 3: 32 neurons (ReLU) + BatchNorm + Dropout 20%",
    "Output Layer: Softmax with 2 classes",
]

PERFORMANCE_ROWS = pd.DataFrame(
    [
        {"Class": "anomaly", "Precision": 0.99, "Recall": 0.99, "F1-Score": 0.99, "Support": 2349},
        {"Class": "normal", "Precision": 0.99, "Recall": 0.99, "F1-Score": 0.99, "Support": 2690},
    ]
)

TRAIN_EPOCHS = list(range(1, 22))
TRAIN_ACC = [0.9411, 0.9670, 0.9738, 0.9778, 0.9798, 0.9814, 0.9802, 0.9827, 0.9831, 0.9835, 0.9842, 0.9854, 0.9846, 0.9867, 0.9857, 0.9853, 0.9894, 0.9889, 0.9891, 0.9902, 0.9899]
VAL_ACC = [0.9792, 0.9839, 0.9841, 0.9849, 0.9861, 0.9861, 0.9866, 0.9861, 0.9856, 0.9869, 0.9873, 0.9876, 0.9876, 0.9866, 0.9864, 0.9871, 0.9878, 0.9881, 0.9878, 0.9873, 0.9881]
TRAIN_LOSS = [0.1579, 0.0874, 0.0726, 0.0613, 0.0546, 0.0518, 0.0516, 0.0474, 0.0444, 0.0430, 0.0428, 0.0403, 0.0405, 0.0363, 0.0373, 0.0386, 0.0294, 0.0303, 0.0274, 0.0291, 0.0290]
VAL_LOSS = [0.0607, 0.0512, 0.0507, 0.0490, 0.0529, 0.0441, 0.0450, 0.0387, 0.0482, 0.0437, 0.0381, 0.0410, 0.0387, 0.0387, 0.0458, 0.0436, 0.0441, 0.0429, 0.0427, 0.0420, 0.0461]
CONFUSION_MATRIX = [[2319, 30], [24, 2666]]

KEY_FEATURES = [
    "Real-time Detection: Instant analysis of network connections",
    "Deep Learning Model: 3-layer neural network with regularization",
    "High Accuracy: 98.93% detection rate",
    "User-friendly Interface: Easy-to-use dashboard",
]

TECHNICAL_SPECS = [
    "Model Type: Deep Neural Network (TensorFlow/Keras)",
    "Features Used: 41 network connection features",
    "Training Data: KDD Cup 99 Dataset",
    "Framework: TensorFlow, Streamlit, Flask, SQLite",
]

DETECTION_CAPABILITIES = [
    "DoS (Denial of Service)",
    "Probe attacks",
    "R2L (Remote to Local)",
    "U2R (User to Root)",
    "Normal traffic",
]

REQUIRED_FILES = [
    "model_weights.npz",
    "scaler.pkl",
    "label_encoders.pkl",
    "target_encoder.pkl",
]


def api_headers() -> Dict[str, str]:
    token = st.session_state.get("token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_post(path: str, payload: dict | None = None):
    return requests.post(f"{API_BASE}{path}", json=payload or {}, headers=api_headers(), timeout=120)


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
                radial-gradient(circle at top left, rgba(0, 184, 255, 0.14), transparent 28%),
                linear-gradient(180deg, #f3f9ff 0%, #fbfdff 100%);
            color: #10243e;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #082554 0%, #0c3b82 60%, #00a8f0 150%);
            border-right: 1px solid rgba(255,255,255,0.08);
            min-width: 205px !important;
            max-width: 205px !important;
        }
        [data-testid="stSidebar"] * {
            color: #f3fbff !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0 !important;
        }
        [data-testid="stToolbar"] {
            background: transparent !important;
            right: 1rem !important;
            top: 0.35rem !important;
        }
        [data-testid="stDecoration"] {
            display: none !important;
        }
        .block-container {
            max-width: 1380px !important;
            padding-top: 0.65rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .hero {
            background: linear-gradient(135deg, #002970 0%, #0a4ca8 58%, #00baf2 100%);
            color: white;
            border-radius: 24px;
            padding: 1.05rem 1.5rem;
            box-shadow: 0 15px 28px rgba(0, 41, 112, 0.18);
            margin-bottom: 0.75rem;
            text-align: center;
        }
        .hero h1 { margin: 0 0 0.18rem 0; font-size: 1.85rem; letter-spacing: -0.03em; line-height: 1.15; }
        .hero p { margin: 0; font-size: 0.88rem; color: rgba(255,255,255,0.92); }
        .hero .brand {
            font-size: 1.7rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            text-transform: none;
            margin-bottom: 0.12rem;
            color: rgba(255,255,255,0.98);
        }
        .card {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(10, 76, 168, 0.08);
            border-radius: 20px;
            padding: 0.9rem 0.95rem;
            box-shadow: 0 9px 20px rgba(16, 36, 62, 0.07);
            margin-bottom: 0.7rem;
        }
        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f6fbff 100%);
            border: 1px solid rgba(0, 102, 204, 0.1);
            border-radius: 17px;
            padding: 0.8rem 0.9rem;
            min-height: 88px;
            box-shadow: 0 8px 16px rgba(16, 36, 62, 0.05);
        }
        .metric-card .label { color: #537091; font-size: 0.8rem; margin-bottom: 0.32rem; }
        .metric-card .value { color: #06224d; font-size: 1.45rem; font-weight: 700; }
        .metric-card .hint { color: #6883a1; margin-top: 0.28rem; font-size: 0.78rem; }
        .section-title { font-size: 1.1rem; font-weight: 700; color: #0b2347; margin-bottom: 0.2rem; }
        .section-copy { color: #59708f; margin-bottom: 0.45rem; font-size: 0.9rem; }
        .status-ok {
            background: linear-gradient(135deg, #e8fff5 0%, #f3fffb 100%);
            border: 1px solid #bcefdc;
            color: #17614a;
            border-radius: 16px;
            padding: 1rem;
        }
        .status-alert {
            background: linear-gradient(135deg, #fff1f1 0%, #fff8f7 100%);
            border: 1px solid #f2c2c2;
            color: #8c2626;
            border-radius: 16px;
            padding: 1rem;
        }
        .badge {
            display: inline-block;
            background: #e5f6ff;
            color: #04507d;
            font-size: 0.82rem;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.3rem 0.65rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .stTextInput input, .stNumberInput input, .stTextArea textarea {
            background: #ffffff !important;
            color: #11243f !important;
            border: 1px solid #b8d4ef !important;
            border-radius: 12px !important;
        }
        .stSelectbox [data-baseweb="select"] > div,
        .stMultiSelect [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: #11243f !important;
            border: 1px solid #b8d4ef !important;
            border-radius: 12px !important;
        }
        .stSelectbox svg, .stMultiSelect svg {
            fill: #12345b !important;
        }
        [data-baseweb="select"] input {
            color: #11243f !important;
        }
        .stSlider [data-baseweb="slider"] {
            padding-top: 0.35rem;
        }
        .stSlider [data-testid="stTickBar"] {
            color: #12345b !important;
        }
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            background: linear-gradient(135deg, #003d97 0%, #00a8f0 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            min-height: 2.8rem !important;
            box-shadow: 0 10px 20px rgba(0, 61, 151, 0.18) !important;
        }
        .stButton > button p, .stDownloadButton > button p, .stFormSubmitButton > button p {
            color: #ffffff !important;
        }
        .stDownloadButton > button {
            border: 1px solid rgba(255,255,255,0.18) !important;
        }
        .stToggle label, .stRadio label, .stCheckbox label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label {
            color: #15345b !important;
            font-weight: 600 !important;
        }
        .stCaption, .stMarkdown, .stInfo, .stSuccess, .stWarning {
            color: #173154;
        }
        .stAlert {
            color: #173154 !important;
        }
        .soft-title {
            font-size: 1.45rem;
            font-weight: 700;
            color: #1f2c44;
            margin-bottom: 0.3rem;
        }
        .clean-list {
            color: #23354f;
            font-size: 0.9rem;
            line-height: 1.55;
        }
        .alert-card {
            border-left: 6px solid #ff5b5b;
            background: linear-gradient(180deg, #ffffff 0%, #fff6f6 100%);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            box-shadow: 0 10px 20px rgba(16, 36, 62, 0.06);
            margin-bottom: 0.9rem;
        }
        .alert-card.safe {
            border-left-color: #17b978;
            background: linear-gradient(180deg, #ffffff 0%, #f4fff9 100%);
        }
        .alert-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #0f2346;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }
        .alert-meta {
            color: #5c7290;
            font-size: 0.92rem;
            margin-bottom: 0.45rem;
        }
        .alert-summary {
            color: #1f3554;
            font-size: 0.96rem;
        }
        .premium-icon {
            font-size: 1.35rem;
            margin-right: 0.35rem;
        }
        .chat-wrap {
            display: flex;
            width: 100%;
            margin-bottom: 0.85rem;
        }
        .chat-wrap.user {
            justify-content: flex-end;
        }
        .chat-wrap.assistant {
            justify-content: flex-start;
        }
        .chat-bubble {
            max-width: 78%;
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 22px rgba(16, 36, 62, 0.08);
            line-height: 1.55;
        }
        .chat-bubble.user {
            background: linear-gradient(135deg, #e8f4ff 0%, #ffffff 100%);
            border: 1px solid #c9def7;
            color: #13355d;
        }
        .chat-bubble.assistant {
            background: linear-gradient(135deg, #003d97 0%, #00a8f0 100%);
            border: 1px solid rgba(255,255,255,0.12);
            color: #ffffff;
        }
        .chat-role {
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            opacity: 0.92;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }
        @media (max-width: 900px) {
            .hero {
                padding: 1.15rem 1rem;
                border-radius: 18px;
            }
            .hero .brand,
            .hero h1 {
                font-size: 1.55rem;
                line-height: 1.2;
            }
            .hero p {
                font-size: 0.85rem;
            }
            .card, .metric-card {
                border-radius: 14px;
                padding: 0.8rem;
            }
            .section-title {
                font-size: 1rem;
            }
            .soft-title {
                font-size: 1.25rem;
            }
            .clean-list {
                font-size: 0.9rem;
                line-height: 1.55;
            }
            .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
                min-height: 3rem !important;
                font-size: 1rem !important;
            }
            .chat-bubble {
                max-width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, icon: str):
    st.markdown(
        f"""
        <div class="card" style="padding:0.9rem 1.2rem; margin-bottom:0.9rem;">
            <div class="section-title">{icon} {title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_monitoring_host_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "Unavailable"


def render_auth():
    inject_styles()
    outer_left, center, outer_right = st.columns([0.04, 0.92, 0.04])
    with center:
        st.markdown(
            """
            <div class="hero" style="padding:1.15rem 1.6rem; max-width: 1120px; margin: 0 auto 0.9rem auto;">
                <div class="brand" style="font-size:1.85rem; margin-bottom:0.15rem;">AI INTRUDEX</div>
                <h1 style="font-size:2.05rem; margin-bottom:0.22rem;">Network Intrusion Detection System</h1>
                <p style="font-size:0.92rem;">AI INTRUDEX - Deep Learning for Advanced Network Security</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.1, 0.9], gap="large")
        with left:
            st.markdown(
                """
                <div class="card" style="max-width: 660px;">
                    <div class="section-title">Platform Features</div>
                    <span class="badge">User Accounts</span>
                    <span class="badge">CSV Prediction</span>
                    <span class="badge">Live Monitoring</span>
                    <span class="badge">24/7 Monitor Controls</span>
                    <span class="badge">Email Alert Setup</span>
                    <span class="badge">Admin Dashboard</span>
                    <div style="height:0.8rem;"></div>
                    <div class="section-copy">
                        This interface is designed for end users. They can log in, monitor traffic, upload CSV data,
                        review alerts, and configure continuous monitoring from one dashboard.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            login_tab, register_tab = st.tabs(["Login", "Register"])
            with login_tab:
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", key="login_password", type="password")
                if st.button("Login", use_container_width=True):
                    response = api_post("/login", {"username": username, "password": password})
                    if response.ok:
                        payload = response.json()
                        st.session_state.token = payload["token"]
                        st.session_state.user = payload["user"]
                        st.rerun()
                    else:
                        st.error(response.json().get("error", "Login failed"))
            with register_tab:
                username = st.text_input("New username", key="register_username")
                password = st.text_input("New password", key="register_password", type="password")
                if st.button("Create account", use_container_width=True):
                    response = api_post("/register", {"username": username, "password": password})
                    if response.ok:
                        payload = response.json()
                        st.session_state.token = payload["token"]
                        st.session_state.user = payload["user"]
                        st.rerun()
                    else:
                        st.error(response.json().get("error", "Registration failed"))


def render_metric_cards(dashboard_data: dict):
    items = [
        ("Predictions", dashboard_data["total_predictions"], "Manual + CSV analysis results"),
        ("Intrusions", dashboard_data["intrusion_predictions"], "Suspicious detections saved"),
        ("CSV Uploads", dashboard_data["total_uploads"], "Batch analysis sessions"),
        ("Live Events", dashboard_data["total_live_events"], "Live traffic sessions"),
    ]
    cols = st.columns(4)
    for col, item in zip(cols, items):
        label, value, hint = item
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                    <div class="hint">{hint}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_overview_page(dashboard_data: dict):
    st.markdown('<div class="card"><div class="section-title">About Network Intrusion Detection System</div></div>', unsafe_allow_html=True)
    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown('<div class="soft-title">🎯 System Overview</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="clean-list">
            This Network Intrusion Detection System (NIDS) uses <b>Deep Learning</b> to identify potential security threats in real-time network traffic with high accuracy.
            The platform supports both CSV-based prediction and live monitoring for continuous network visibility.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="soft-title">✨ Key Features</div>', unsafe_allow_html=True)
        for item in KEY_FEATURES:
            st.markdown(f'<div class="clean-list">• {item}</div>', unsafe_allow_html=True)
    with top_right:
        st.markdown('<div class="soft-title">🛠 Technical Specifications</div>', unsafe_allow_html=True)
        for item in TECHNICAL_SPECS:
            st.markdown(f'<div class="clean-list">• {item}</div>', unsafe_allow_html=True)
        st.markdown('<div class="soft-title" style="margin-top:1rem;">📊 Performance Metrics</div>', unsafe_allow_html=True)
        st.markdown('<div class="clean-list">• Accuracy: 98.93%</div>', unsafe_allow_html=True)
        st.markdown('<div class="clean-list">• Precision: 99%</div>', unsafe_allow_html=True)
        st.markdown('<div class="clean-list">• Recall: 99%</div>', unsafe_allow_html=True)
        st.markdown('<div class="clean-list">• F1-Score: 99%</div>', unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown('<div class="soft-title">🛡 Detection Capabilities</div>', unsafe_allow_html=True)
        for item in DETECTION_CAPABILITIES:
            st.markdown(f'<div class="clean-list">• {item}</div>', unsafe_allow_html=True)
    with bottom_right:
        st.markdown('<div class="soft-title">📁 Required Files</div>', unsafe_allow_html=True)
        for item in REQUIRED_FILES:
            st.markdown(f'<div class="clean-list">• {item}</div>', unsafe_allow_html=True)

    st.markdown('<div class="soft-title">🧪 Dataset Info</div>', unsafe_allow_html=True)
    ds1, ds2 = st.columns(2)
    with ds1:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">KDD-Based Current App</div>
                <div class="section-copy">
                    The current running application follows the original KDD-style 41-feature flow. This is the active model path
                    used by CSV prediction, manual prediction, and the current live feature extraction logic.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with ds2:
        st.markdown(
            """
            <div class="card">
                <div class="section-title">UNSW-NB15 Enhanced Training Pipeline</div>
                <div class="section-copy">
                    A separate UNSW-NB15 pipeline has been added for model modernization. It does not replace the current app yet,
                    but it is included for retraining, evaluation, and improved dataset relevance.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    status_text = "Running" if dashboard_data.get("monitor_running") else "Stopped"
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">System Status</div>
            <span class="badge">Model: Deep Learning</span>
            <span class="badge">Database: Connected</span>
            <span class="badge">Live Monitor: {status_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_simple_pdf(title: str, lines: list[str]) -> bytes:
    safe_lines = [title] + lines
    content_lines = ["BT", "/F1 14 Tf", "50 780 Td"]
    first = True
    for line in safe_lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if first:
            content_lines.append(f"({escaped}) Tj")
            first = False
        else:
            content_lines.append("0 -18 Td")
            content_lines.append(f"({escaped}) Tj")
    content_lines.append("ET")
    stream = "\n".join(content_lines).encode("latin-1", errors="replace")
    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objects.append(b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n")
    objects.append(f"4 0 obj << /Length {len(stream)} >> stream\n".encode("latin-1") + stream + b"\nendstream endobj\n")
    objects.append(b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n")
    pdf = b"%PDF-1.4\n"
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf += obj
    xref_offset = len(pdf)
    pdf += f"xref\n0 {len(objects)+1}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("latin-1")
    pdf += f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF".encode("latin-1")
    return pdf


def render_notifications_page():
    response = api_get("/history")
    if not response.ok:
        st.error("Could not load notifications.")
        return
    data = response.json()
    prediction_alerts = [row for row in data["predictions"] if str(row.get("predicted_label", "")).lower() != "normal"]
    live_alerts = [row for row in data["live_events"] if str(row.get("severity", "")).lower() in {"critical", "high", "medium"}]

    st.markdown('<div class="card"><div class="section-title">Notifications</div><div class="section-copy">Recent suspicious detections and live monitoring alerts appear here with cleaner alert cards.</div></div>', unsafe_allow_html=True)

    if not prediction_alerts and not live_alerts:
        st.success("No active suspicious alerts found in recent history.")
        return

    for row in prediction_alerts[:10]:
        pdf_bytes = build_simple_pdf(
            "Incident Report",
            [
                f"Type: Prediction Alert",
                f"Prediction: {row.get('predicted_label')}",
                f"Confidence: {row.get('confidence')}",
                f"Severity: {row.get('severity')}",
                f"Created At: {row.get('created_at')}",
                f"Summary: {row.get('summary')}",
                f"Recommended Action: {row.get('recommended_action')}",
            ],
        )
        st.markdown(
            f"""
            <div class="alert-card">
                <div class="alert-head"><span>Prediction Alert</span><span>{row.get('severity', '').upper()}</span></div>
                <div class="alert-meta">{row.get('created_at')} | {row.get('predicted_label')} | {row.get('confidence'):.2f}% confidence</div>
                <div class="alert-summary">{row.get('summary')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download Incident Report (PDF)",
            data=pdf_bytes,
            file_name=f"incident_report_prediction_{row.get('id')}.pdf",
            mime="application/pdf",
            key=f"pred_pdf_{row.get('id')}",
            use_container_width=True,
        )

    for row in live_alerts[:10]:
        pdf_bytes = build_simple_pdf(
            "Live Monitoring Incident Report",
            [
                f"Type: Live Monitoring Alert",
                f"Protocol: {row.get('protocol')}",
                f"Severity: {row.get('severity')}",
                f"Source IP: {row.get('source_ip')}",
                f"Destination IP: {row.get('destination_ip')}",
                f"Created At: {row.get('created_at')}",
                f"Summary: {row.get('summary')}",
            ],
        )
        st.markdown(
            f"""
            <div class="alert-card">
                <div class="alert-head"><span>Live Monitor Alert</span><span>{row.get('severity', '').upper()}</span></div>
                <div class="alert-meta">{row.get('created_at')} | {row.get('protocol')} | Source {row.get('source_ip')}</div>
                <div class="alert-summary">{row.get('summary')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download Live Incident Report (PDF)",
            data=pdf_bytes,
            file_name=f"incident_report_live_{row.get('id')}.pdf",
            mime="application/pdf",
            key=f"live_pdf_{row.get('id')}",
            use_container_width=True,
        )


def render_detection_page():
    st.markdown('<div class="card"><div class="section-title">Intrusion Detection</div><div class="section-copy">Fill the traffic details below and run the model just like the older application, but with clearer sections and labels.</div></div>', unsafe_allow_html=True)
    defaults = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF", "src_bytes": 181, "dst_bytes": 5450,
        "land": 0, "wrong_fragment": 0, "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0, "num_file_creations": 0,
        "num_shells": 0, "num_access_files": 0, "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 9, "srv_count": 9, "serror_rate": 0.0, "srv_serror_rate": 0.0, "rerror_rate": 0.0,
        "srv_rerror_rate": 0.0, "same_srv_rate": 1.0, "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0,
        "dst_host_count": 9, "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0, "dst_host_diff_srv_rate": 0.0,
        "dst_host_same_src_port_rate": 0.11, "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0, "dst_host_srv_rerror_rate": 0.0,
    }
    payload = {}
    tab1, tab2, tab3, tab4 = st.tabs(["Connection Info", "Traffic Data", "Security Metrics", "Host Statistics"])

    with tab1:
        render_section_header("Basic Connection Information", "🌐")
        c1, c2, c3 = st.columns(3)
        payload["duration"] = c1.number_input("Duration (seconds)", min_value=0, value=int(defaults["duration"]))
        payload["service"] = c2.selectbox("Service", ["http", "private", "domain_u", "ftp", "smtp", "telnet", "eco_i", "ecr_i"], index=0)
        payload["flag"] = c3.selectbox("Flag", ["SF", "S0", "REJ", "RSTR", "SH", "S1"], index=0)
        payload["protocol_type"] = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"], index=0)

    with tab2:
        render_section_header("Traffic Data", "📦")
        c1, c2, c3 = st.columns(3)
        payload["src_bytes"] = c1.number_input("Source Bytes", min_value=0, value=int(defaults["src_bytes"]))
        payload["dst_bytes"] = c2.number_input("Destination Bytes", min_value=0, value=int(defaults["dst_bytes"]))
        payload["count"] = c3.number_input("Connection Count", min_value=0, value=int(defaults["count"]))
        c4, c5, c6 = st.columns(3)
        payload["srv_count"] = c4.number_input("Service Count", min_value=0, value=int(defaults["srv_count"]))
        payload["land"] = 1 if c5.toggle("Land", value=False) else 0
        payload["logged_in"] = 1 if c6.toggle("Logged In", value=True) else 0

    with tab3:
        render_section_header("Security Metrics", "🔐")
        cols = st.columns(3)
        payload["serror_rate"] = cols[0].slider("SYN Error Rate", 0.0, 1.0, float(defaults["serror_rate"]), 0.01)
        payload["srv_serror_rate"] = cols[1].slider("Service SYN Error Rate", 0.0, 1.0, float(defaults["srv_serror_rate"]), 0.01)
        payload["rerror_rate"] = cols[2].slider("REJ Error Rate", 0.0, 1.0, float(defaults["rerror_rate"]), 0.01)
        cols = st.columns(3)
        payload["srv_rerror_rate"] = cols[0].slider("Service REJ Error Rate", 0.0, 1.0, float(defaults["srv_rerror_rate"]), 0.01)
        payload["same_srv_rate"] = cols[1].slider("Same Service Rate", 0.0, 1.0, float(defaults["same_srv_rate"]), 0.01)
        payload["diff_srv_rate"] = cols[2].slider("Different Service Rate", 0.0, 1.0, float(defaults["diff_srv_rate"]), 0.01)
        cols = st.columns(3)
        payload["srv_diff_host_rate"] = cols[0].slider("Service Diff Host Rate", 0.0, 1.0, float(defaults["srv_diff_host_rate"]), 0.01)
        payload["wrong_fragment"] = cols[1].number_input("Wrong Fragment", min_value=0, value=int(defaults["wrong_fragment"]))
        payload["urgent"] = cols[2].number_input("Urgent Packets", min_value=0, value=int(defaults["urgent"]))

    with tab4:
        render_section_header("Advanced Host Statistics", "📈")
        cols = st.columns(3)
        payload["dst_host_count"] = cols[0].number_input("Host Count", min_value=0, value=int(defaults["dst_host_count"]))
        payload["dst_host_srv_count"] = cols[1].number_input("Host Service Count", min_value=0, value=int(defaults["dst_host_srv_count"]))
        payload["dst_host_same_srv_rate"] = cols[2].slider("Host Same Service Rate", 0.0, 1.0, float(defaults["dst_host_same_srv_rate"]), 0.01)
        cols = st.columns(3)
        payload["dst_host_diff_srv_rate"] = cols[0].slider("Host Service Diff Rate", 0.0, 1.0, float(defaults["dst_host_diff_srv_rate"]), 0.01)
        payload["dst_host_same_src_port_rate"] = cols[1].slider("Host Same Src Port Rate", 0.0, 1.0, float(defaults["dst_host_same_src_port_rate"]), 0.01)
        payload["dst_host_srv_diff_host_rate"] = cols[2].slider("Host Service Diff Host Rate", 0.0, 1.0, float(defaults["dst_host_srv_diff_host_rate"]), 0.01)
        cols = st.columns(4)
        payload["dst_host_serror_rate"] = cols[0].slider("Host SYN Error Rate", 0.0, 1.0, float(defaults["dst_host_serror_rate"]), 0.01)
        payload["dst_host_srv_serror_rate"] = cols[1].slider("Host Service SYN Error Rate", 0.0, 1.0, float(defaults["dst_host_srv_serror_rate"]), 0.01)
        payload["dst_host_rerror_rate"] = cols[2].slider("Host REJ Error Rate", 0.0, 1.0, float(defaults["dst_host_rerror_rate"]), 0.01)
        payload["dst_host_srv_rerror_rate"] = cols[3].slider("Host Service REJ Error Rate", 0.0, 1.0, float(defaults["dst_host_srv_rerror_rate"]), 0.01)

    for feature, value in defaults.items():
        payload.setdefault(feature, value)

    if st.button("Run Deep Learning Prediction", use_container_width=True):
        response = api_post("/predict-row", payload)
        if response.ok:
            data = response.json()
            css_class = "status-ok" if data["predicted_label"].lower() == "normal" else "status-alert"
            st.markdown(f'<div class="{css_class}"><b>{data["summary"]}</b><br>{data["rationale"]}<br><br>Recommended action: {data["recommended_action"]}</div>', unsafe_allow_html=True)
            st.json(data["probabilities"])
        else:
            st.error(response.json().get("error", "Prediction failed"))


def render_csv_page():
    st.markdown('<div class="card"><div class="section-title">CSV Prediction</div><div class="section-copy">Upload a CSV file with the required 41 traffic feature columns to analyze multiple records at once.</div></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        content = uploaded.getvalue().decode("utf-8")
        df = pd.read_csv(StringIO(content))
        st.dataframe(df.head(), use_container_width=True)
        if st.button("Analyze Uploaded CSV", use_container_width=True):
            response = api_post("/predict-csv", {"filename": uploaded.name, "rows": df.to_dict(orient="records")})
            if response.ok:
                data = response.json()
                st.success(f"Processed {data['total_rows']} rows. Intrusions: {data['intrusion_rows']}, Normal: {data['normal_rows']}.")
                st.dataframe(pd.DataFrame(data["results"]), use_container_width=True)
            else:
                st.error(response.json().get("error", "CSV analysis failed"))
    else:
        st.info("Use `sample_nids_input.csv` as an example input format.")


def render_live_monitor_page(dashboard_data: dict):
    st.markdown('<div class="card"><div class="section-title">Live Monitoring</div><div class="section-copy">This feature captures packets from the machine or server where the backend is running. It extracts a live feature record, then sends that record to the deep learning model.</div></div>', unsafe_allow_html=True)
    st.info(f"Monitoring host local IP: {get_monitoring_host_ip()}")
    st.caption("The source and destination IPs shown after capture are the most active addresses seen in the live traffic window on the monitoring machine.")
    interface = st.text_input("Interface name (leave blank for default)")
    packet_limit = st.slider("Packet limit", 10, 150, 30)
    timeout = st.slider("Capture seconds", 5, 30, 10)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Run One Live Capture", use_container_width=True):
            response = api_post("/live-monitor", {"interface": interface or None, "packet_limit": packet_limit, "timeout": timeout})
            if response.ok:
                data = response.json()
                css_class = "status-ok" if data["predicted_label"].lower() == "normal" else "status-alert"
                st.markdown(f'<div class="{css_class}"><b>{data["summary"]}</b><br>{data["rationale"]}<br><br>Recommended action: {data["recommended_action"]}</div>', unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted Label", data["predicted_label"])
                m2.metric("Confidence", f'{data["confidence"]:.2f}%')
                m3.metric("Protocol", data["protocol"])
                m4.metric("Service", data["service"])
                st.write(f"Top source IP seen: `{data['source_ip']}`")
                st.write(f"Top destination IP seen: `{data['destination_ip']}`")
                with st.expander("Extracted live feature record"):
                    st.json(data["feature_record"])
            else:
                st.error(response.json().get("error", "Live monitoring failed"))
        latest_response = api_get("/history")
        if latest_response.ok:
            live_rows = latest_response.json().get("live_events", [])
            if live_rows:
                latest = live_rows[0]
                st.markdown(
                    f"""
                    <div class="card">
                        <div class="section-title">Latest Live Event</div>
                        <div class="section-copy">Source IP: <b>{latest.get('source_ip')}</b> | Destination IP: <b>{latest.get('destination_ip')}</b> | Severity: <b>{latest.get('severity')}</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    with c2:
        monitor_state = "Running" if dashboard_data.get("monitor_running") else "Stopped"
        st.metric("24/7 Monitor Status", monitor_state)


def render_ai_analyst_page():
    response = api_get("/ai-analyst")
    if not response.ok:
        st.error("Could not load AI analyst review.")
        return
    data = response.json()
    tone = "status-ok" if data["status"] == "stable" else "status-alert"
    st.markdown('<div class="card"><div class="section-title">AI Analyst</div><div class="section-copy">Agent-style review of recent predictions and live monitoring history.</div></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="{tone}"><b>{data["summary"]}</b><br><br>Recommended action: {data["recommended_action"]}</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    c1.metric("Suspicious Predictions", data["suspicious_prediction_count"])
    c2.metric("Elevated Live Events", data["elevated_live_count"])
    st.markdown('<div class="card"><div class="section-title">Agent Workflow</div></div>', unsafe_allow_html=True)
    for step in data["steps"]:
        st.markdown(f'<div class="clean-list">• {step}</div>', unsafe_allow_html=True)


def render_ai_assistant_page():
    st.markdown(
        '<div class="card"><div class="section-title">AI Assistant</div><div class="section-copy">Ask about alerts, suspicious activity, live monitoring, reports, or what action to take next.</div></div>',
        unsafe_allow_html=True,
    )
    sample_cols = st.columns(4)
    samples = [
        "Why was this alert generated?",
        "Show my latest suspicious activity",
        "Summarize today’s live monitoring",
        "What should I do next?",
    ]
    for col, sample in zip(sample_cols, samples):
        with col:
            if st.button(sample, use_container_width=True, key=f"sample_{sample}"):
                st.session_state["ai_chat_input"] = sample

    if "ai_chat_history" not in st.session_state:
        st.session_state["ai_chat_history"] = []

    message = st.text_input("Ask AI Assistant", key="ai_chat_input")
    if st.button("Send to AI Assistant", use_container_width=True):
        response = api_post("/ai-chat", {"message": message})
        if response.ok:
            reply = response.json()["reply"]
            st.session_state["ai_chat_history"].append({"role": "user", "text": message})
            st.session_state["ai_chat_history"].append({"role": "assistant", "text": reply})
        else:
            st.error(response.json().get("error", "AI assistant failed"))

    for item in st.session_state["ai_chat_history"][-10:]:
        css = "status-alert" if item["role"] == "assistant" else "card"
        label = "AI Assistant" if item["role"] == "assistant" else "You"
        st.markdown(
            f'<div class="{css}"><b>{label}:</b> {item["text"]}</div>',
            unsafe_allow_html=True,
        )


def render_alert_settings_page(dashboard_data: dict):
    response = api_get("/alert-settings")
    if not response.ok:
        st.error("Could not load alert settings.")
        return
    settings = response.json()
    st.markdown('<div class="card"><div class="section-title">Alert Settings</div><div class="section-copy">Configure continuous monitoring and email alerts here. SMS fields are placeholders for future integration.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("alert_settings_form"):
        monitor_enabled = st.toggle("Enable continuous monitoring", value=bool(settings.get("monitor_enabled")))
        packet_limit = st.slider("Packet limit for each cycle", 10, 150, int(settings.get("packet_limit", 30)))
        capture_seconds = st.slider("Capture seconds per cycle", 5, 30, int(settings.get("capture_seconds", 10)))
        st.markdown("### Email Alerts")
        email_enabled = st.toggle("Enable email alerts", value=bool(settings.get("email_enabled")))
        email_recipient = st.text_input("Recipient email", value=settings.get("email_recipient", ""))
        smtp_host = st.text_input("SMTP host", value=settings.get("smtp_host", "smtp.gmail.com"))
        smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(settings.get("smtp_port", 587)))
        smtp_username = st.text_input("SMTP username", value=settings.get("smtp_username", ""))
        smtp_password = st.text_input("SMTP password / app password", type="password", value=settings.get("smtp_password", ""))
        st.markdown("### SMS Placeholder")
        sms_enabled = st.toggle("Enable SMS placeholder", value=bool(settings.get("sms_enabled")))
        sms_number = st.text_input("SMS number", value=settings.get("sms_number", ""))
        submitted = st.form_submit_button("Save Settings", use_container_width=True)
    if submitted:
        payload = {
            "monitor_enabled": monitor_enabled,
            "packet_limit": packet_limit,
            "capture_seconds": capture_seconds,
            "email_enabled": email_enabled,
            "email_recipient": email_recipient,
            "smtp_host": smtp_host,
            "smtp_port": int(smtp_port),
            "smtp_username": smtp_username,
            "smtp_password": smtp_password,
            "sms_enabled": sms_enabled,
            "sms_number": sms_number,
        }
        save_response = api_post("/alert-settings", payload)
        if save_response.ok:
            st.success("Alert settings saved.")
        else:
            st.error(save_response.json().get("error", "Could not save settings"))
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start 24/7 Monitoring", use_container_width=True):
            start_response = api_post("/monitor/start")
            if start_response.ok:
                st.success("Continuous monitoring started.")
            else:
                st.error(start_response.json().get("error", "Could not start monitor"))
    with c2:
        if st.button("Stop 24/7 Monitoring", use_container_width=True):
            stop_response = api_post("/monitor/stop")
            if stop_response.ok:
                st.success("Continuous monitoring stopped.")
            else:
                st.error(stop_response.json().get("error", "Could not stop monitor"))
    running_text = "Running" if dashboard_data.get("monitor_running") else "Stopped"
    st.info(f"Current monitor status: {running_text}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_history_page():
    response = api_get("/history")
    if not response.ok:
        st.error("Could not load history.")
        return
    data = response.json()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="section-title">Prediction History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["predictions"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="section-title">CSV Upload History</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["uploads"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="section-title">Live Event History</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(data["live_events"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_admin_page():
    response = api_get("/admin/overview")
    if not response.ok:
        st.error(response.json().get("error", "Admin overview unavailable"))
        return
    data = response.json()
    st.markdown('<div class="card"><div class="section-title">User Overview</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(data["users"]), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="card"><div class="section-title">Recent Suspicious Alerts</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["recent_alerts"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="section-title">Recent Live Events</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(data["live_events"]), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_model_performance_page():
    st.markdown('<div class="card"><div class="section-title">Model Performance</div><div class="section-copy">Core deep learning architecture and baseline performance values from the trained project model.</div></div>', unsafe_allow_html=True)
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown('<div class="card"><div class="section-title">Model Architecture</div>', unsafe_allow_html=True)
        for row in MODEL_ARCHITECTURE:
            st.write(f"- {row}")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card"><div class="section-title">Classification Report</div>', unsafe_allow_html=True)
        st.dataframe(PERFORMANCE_ROWS, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", "98.93%")
    c2.metric("Precision", "99%")
    c3.metric("Recall", "99%")
    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=TRAIN_EPOCHS, y=TRAIN_ACC, mode="lines+markers", name="Training Accuracy", line=dict(color="#0057d8", width=3)))
    fig_acc.add_trace(go.Scatter(x=TRAIN_EPOCHS, y=VAL_ACC, mode="lines+markers", name="Validation Accuracy", line=dict(color="#00a8f0", width=3)))
    fig_acc.update_layout(title="Training History - Accuracy", height=360, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#163252"))
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=TRAIN_EPOCHS, y=TRAIN_LOSS, mode="lines+markers", name="Training Loss", line=dict(color="#ff6b6b", width=3)))
    fig_loss.add_trace(go.Scatter(x=TRAIN_EPOCHS, y=VAL_LOSS, mode="lines+markers", name="Validation Loss", line=dict(color="#7b61ff", width=3)))
    fig_loss.update_layout(title="Training History - Loss", height=360, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#163252"))
    heatmap = go.Figure(data=go.Heatmap(
        z=CONFUSION_MATRIX,
        x=["Predicted anomaly", "Predicted normal"],
        y=["Actual anomaly", "Actual normal"],
        text=CONFUSION_MATRIX,
        texttemplate="%{text}",
        colorscale="Blues"
    ))
    heatmap.update_layout(title="Confusion Matrix", height=420, paper_bgcolor="white", plot_bgcolor="white", font=dict(color="#163252"))
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(fig_acc, use_container_width=True)
    with g2:
        st.plotly_chart(fig_loss, use_container_width=True)
    st.plotly_chart(heatmap, use_container_width=True)


def render_reports_page():
    response = api_get("/history")
    if not response.ok:
        st.error("Could not load reports.")
        return
    data = response.json()
    rows = data["predictions"][:15]
    st.markdown('<div class="card"><div class="section-title">Incident Reports</div><div class="section-copy">Generate exportable PDF reports from saved prediction records.</div></div>', unsafe_allow_html=True)
    if not rows:
        st.info("No prediction records available yet.")
        return
    for row in rows:
        st.markdown(
            f"""
            <div class="card">
                <div class="section-title">Record #{row.get('id')}</div>
                <div class="section-copy">{row.get('created_at')} | {row.get('source_type')} | {row.get('predicted_label')} | Severity {row.get('severity')}</div>
                <div class="clean-list">{row.get('summary')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pdf_bytes = build_simple_pdf(
            "Prediction Report",
            [
                f"Record ID: {row.get('id')}",
                f"Source Type: {row.get('source_type')}",
                f"Predicted Label: {row.get('predicted_label')}",
                f"Confidence: {row.get('confidence')}",
                f"Severity: {row.get('severity')}",
                f"Created At: {row.get('created_at')}",
                f"Summary: {row.get('summary')}",
                f"Recommended Action: {row.get('recommended_action')}",
            ],
        )
        st.download_button(
            "Export PDF Report",
            data=pdf_bytes,
            file_name=f"prediction_report_{row.get('id')}.pdf",
            mime="application/pdf",
            key=f"report_pdf_{row.get('id')}",
            use_container_width=True,
        )


def main():
    st.set_page_config(page_title=PRODUCT_NAME, page_icon="🛡️", layout="wide")
    inject_styles()
    if not ensure_logged_in():
        render_auth()
        return
    dashboard_response = api_get("/dashboard")
    if not dashboard_response.ok:
        st.error("Backend is not reachable. Start `python run_backend.py` first.")
        return
    dashboard_data = dashboard_response.json()
    with st.sidebar:
        st.markdown("## Navigation")
        pages = ["Overview", "Intrusion Detection", "CSV Prediction", "Live Monitor", "AI Analyst", "AI Assistant", "Notifications", "Reports", "Alert Settings", "History", "Model Performance"]
        if dashboard_data.get("role") == "admin":
            pages.append("Admin Dashboard")
        page = st.radio("Select Page", pages)
        st.caption(f"Logged in as {st.session_state.user['username']} ({dashboard_data.get('role', 'user')})")
        if st.button("Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    st.markdown(
        """
        <div class="hero">
            <div class="brand">AI INTRUDEX</div>
            <h1>Network Intrusion Detection System</h1>
            <p>AI INTRUDEX - Deep Learning for Advanced Network Security</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(dashboard_data)
    st.write("")
    if page == "Overview":
        render_overview_page(dashboard_data)
    elif page == "Intrusion Detection":
        render_detection_page()
    elif page == "CSV Prediction":
        render_csv_page()
    elif page == "Live Monitor":
        render_live_monitor_page(dashboard_data)
    elif page == "AI Analyst":
        render_ai_analyst_page()
    elif page == "AI Assistant":
        render_ai_assistant_page()
    elif page == "Notifications":
        render_notifications_page()
    elif page == "Reports":
        render_reports_page()
    elif page == "Alert Settings":
        render_alert_settings_page(dashboard_data)
    elif page == "History":
        render_history_page()
    elif page == "Model Performance":
        render_model_performance_page()
    elif page == "Admin Dashboard":
        render_admin_page()


if __name__ == "__main__":
    main()
