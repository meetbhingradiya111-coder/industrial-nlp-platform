import streamlit as st
import requests
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

PREDICT_URL = "http://127.0.0.1:8000/predict"
RISK_URL = "http://127.0.0.1:8000/predict_risk"
SEQUENCE_URL = "http://industrial_api:8000/predict_sequence"

st.set_page_config(
    page_title="PredictAI",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_analyzed" not in st.session_state:
    st.session_state.last_analyzed = None

if "history" not in st.session_state:
    st.session_state.history = []

if "machine_result" not in st.session_state:
    st.session_state.machine_result = None

if "fleet_data" not in st.session_state:
    st.session_state.fleet_data = {}


# ============================================================
# PREMIUM GLASSMORPHIC UI & ANIMATIONS
# ============================================================

st.markdown(
    """
<style>

/* ---------------- GLOBAL IMPORTS & BASE ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #0b0e14;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(129, 140, 248, 0.05) 0px, transparent 50%);
    color: #f8fafc;
}

.block-container {
    max-width: 1350px;
    padding-top: 35px;
    padding-bottom: 60px;
}

/* ---------------- KEYFRAME ANIMATIONS ---------------- */
@keyframes smoothFadeUp {
    from {
        opacity: 0;
        transform: translateY(18px) scale(0.99);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes ambientGlowPulse {
    0%, 100% {
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.15), inset 0 0 15px rgba(99, 102, 241, 0.05);
    }
    50% {
        box-shadow: 0 0 35px rgba(99, 102, 241, 0.3), inset 0 0 25px rgba(99, 102, 241, 0.1);
    }
}

@keyframes luxuryRingPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.4);
    }
    70% {
        box-shadow: 0 0 0 16px rgba(245, 158, 11, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0);
    }
}

@keyframes statusDotPulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.35;
        transform: scale(0.85);
    }
}

@keyframes metallicSweep {
    0% {
        transform: translateX(-150%) rotate(25deg);
    }
    100% {
        transform: translateX(200%) rotate(25deg);
    }
}

@keyframes progressExpand {
    from {
        transform: scaleX(0);
        transform-origin: left;
    }
    to {
        transform: scaleX(1);
        transform-origin: left;
    }
}

@keyframes counterPop {
    0% {
        opacity: 0;
        transform: scale(0.92);
    }
    70% {
        transform: scale(1.03);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

/* ---------------- SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: rgba(13, 17, 23, 0.75);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

section[data-testid="stSidebar"] > div {
    padding: 28px 20px;
}

.logo {
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.03em;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.logo:hover {
    transform: translateX(4px);
}

.logo span {
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.logo-sub {
    color: #64748b;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

.sidebar-heading {
    color: #475569;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin: 32px 4px 12px;
}

.system-box {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(22, 27, 38, 0.5);
    backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 16px;
    margin-top: 30px;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.system-box:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 20px -6px rgba(0,0,0,0.4);
}

.system-title {
    color: #64748b;
    font-size: 0.58rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.system-online {
    color: #34d399;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}

.system-online::before {
    content: "";
    width: 6px;
    height: 6px;
    background: #34d399;
    border-radius: 50%;
    display: inline-block;
    animation: statusDotPulse 2s ease-in-out infinite;
}

.system-info {
    color: #475569;
    font-size: 0.62rem;
    margin-top: 4px;
}

/* ---------------- HEADER ---------------- */
.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.header-title {
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 4px;
}

.status {
    border: 1px solid rgba(52, 211, 153, 0.25);
    background: rgba(6, 78, 59, 0.2);
    backdrop-filter: blur(8px);
    color: #34d399;
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    animation: ambientGlowPulse 4s ease-in-out infinite;
}

/* ---------------- INTRO & CARDS ---------------- */
.intro {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(18, 24, 38, 0.6);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 18px;
    animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.05s both;
    position: relative;
    overflow: hidden;
}

.intro::after {
    content: "";
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(
        45deg,
        transparent 45%,
        rgba(255, 255, 255, 0.03) 50%,
        transparent 55%
    );
    animation: metallicSweep 7s ease-in-out infinite;
    pointer-events: none;
}

.intro-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
}

.intro-text {
    color: #64748b;
    font-size: 0.72rem;
    line-height: 1.6;
    margin-top: 6px;
}

/* ---------------- METRICS ---------------- */
.metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}

.metric {
    background: rgba(18, 24, 38, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 20px;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.metric:nth-child(1) { animation-delay: 0.08s; }
.metric:nth-child(2) { animation-delay: 0.14s; }
.metric:nth-child(3) { animation-delay: 0.20s; }
.metric:nth-child(4) { animation-delay: 0.26s; }

.metric:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.35);
    background: rgba(24, 32, 50, 0.7);
    box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.5), 0 0 20px rgba(99, 102, 241, 0.1);
}

.metric-label {
    color: #64748b;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.1rem;
    font-weight: 800;
    margin-top: 8px;
    animation: counterPop 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}

/* ---------------- CARDS & ANOMALY ---------------- */
.card, .anomaly, .risk, .history {
    background: rgba(18, 24, 38, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 22px;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.12s both;
}

.card:hover, .history:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(24, 32, 50, 0.65);
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.4);
}

.card-heading {
    color: #64748b;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.card-title {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 700;
    margin-top: 6px;
}

/* ---------------- CONDITION ---------------- */
.condition {
    text-align: center;
    padding: 24px;
}

.condition-symbol {
    width: 80px;
    height: 80px;
    border: 2px solid #f59e0b;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 24px auto 16px;
    color: #f59e0b;
    font-size: 2rem;
    font-weight: 800;
    transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.condition-symbol:hover {
    transform: scale(1.1) rotate(4deg);
}

.condition-symbol.condition-alert {
    animation: luxuryRingPulse 2.2s infinite;
}

.condition-name {
    color: #f59e0b;
    font-size: 1.2rem;
    font-weight: 800;
}

.condition-description {
    color: #64748b;
    font-size: 0.68rem;
    margin-top: 6px;
}

/* ---------------- DETAILS & RECOMMENDATIONS ---------------- */
.detail {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    transition: background 0.2s ease;
}

.detail:hover {
    background: rgba(255, 255, 255, 0.02);
}

.detail-label {
    color: #64748b;
    font-size: 0.68rem;
}

.detail-value {
    color: #e2e8f0;
    font-size: 0.68rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.diagnosis-title, .risk-value {
    animation: counterPop 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.recommendation {
    margin-top: 18px;
    padding-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.recommendation-title {
    color: #94a3b8;
    font-size: 0.58rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.recommendation-text {
    color: #94a3b8;
    font-size: 0.68rem;
    line-height: 1.6;
    margin-top: 6px;
}

/* ---------------- PROGRESS BARS & SHIMMER ---------------- */
.progress {
    height: 6px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    overflow: hidden;
    margin: 8px 0 18px;
    position: relative;
}

.progress > div {
    transform-origin: left;
    animation: progressExpand 1.2s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.progress::after {
    content: "";
    position: absolute;
    top: 0; left: -40%;
    width: 40%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    animation: metallicSweep 3s ease-in-out infinite;
    pointer-events: none;
}

/* ---------------- PIPELINE ANIMATIONS ---------------- */
.pipeline {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    position: relative;
}

.pipeline-card {
    background: rgba(18, 24, 38, 0.5);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 20px 12px;
    text-align: center;
    position: relative;
    z-index: 2;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.pipeline-card:nth-child(1) { animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.05s both; }
.pipeline-card:nth-child(2) { animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.15s both; }
.pipeline-card:nth-child(3) { animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.25s both; }
.pipeline-card:nth-child(4) { animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.35s both; }
.pipeline-card:nth-child(5) { animation: smoothFadeUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) 0.45s both; }

.pipeline-card:hover {
    transform: translateY(-6px) scale(1.02);
    background: rgba(24, 32, 50, 0.75);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 12px 28px rgba(99, 102, 241, 0.15);
}

.pipeline-icon {
    font-size: 1.4rem;
    margin: 10px 0;
    transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.pipeline-card:hover .pipeline-icon {
    transform: scale(1.25) rotate(-6deg);
}

.pipeline-step {
    color: #818cf8;
    font-size: 0.55rem;
    font-weight: 800;
}

.pipeline-name {
    color: #cbd5e1;
    font-size: 0.64rem;
    font-weight: 600;
}

/* Connector Line & Moving Signal */
.pipeline-card:not(:last-child)::after {
    content: "";
    position: absolute;
    top: 50%; right: -13px;
    width: 26px; height: 1px;
    background: rgba(255, 255, 255, 0.12);
    transform: translateY(-50%);
    z-index: 1;
}

.pipeline-card:not(:last-child)::before {
    content: "›";
    position: absolute;
    top: 50%; right: -18px;
    transform: translateY(-54%);
    color: #818cf8;
    font-size: 18px;
    font-weight: 700;
    z-index: 4;
    animation: statusDotPulse 1.8s ease-in-out infinite;
}

/* ---------------- STREAMLIT INTERACTION CONTROLS ---------------- */
.stButton > button {
    border-radius: 10px !important;
    min-height: 44px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35) !important;
}

.stButton > button:active {
    transform: translateY(1px) scale(0.98);
}

button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    color: #ffffff !important;
}

textarea, input {
    background: rgba(11, 14, 20, 0.8) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

textarea:focus, input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25), 0 0 20px rgba(99, 102, 241, 0.15) !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ---------------- RESPONSIVE & ACCESSIBILITY ---------------- */
@media (max-width: 900px) {
    .metrics, .pipeline {
        grid-template-columns: repeat(2, 1fr);
    }
    .pipeline-card:not(:last-child)::after,
    .pipeline-card:not(:last-child)::before {
        display: none;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# BEST DEMO UNHEALTHY SEQUENCE PRESETS
# ============================================================

LSTM_UNHEALTHY_SEQUENCE = [
    "slightly elevated readings near the gearbox drift in pressure noted but still within acceptable operating range",
    "a mild occasional noise was detected on the conveyor will keep an eye on it during upcoming shifts",
    "slightly elevated readings near the control panel play in the coupling noted but still within acceptable operating range",
    "technician report slight noise from motor shaft monitoring required",
    "during shift inspection minor misalignment noticed adjusted slightly",
    "shift handover note oil viscosity degraded complete oil change needed",
    "slight play in the coupling noticed on the motor during today s check machine continues to operate normally overall",
    "shift handover note small oil leak noticed near base cleaned",
    "belt tension seems marginally looser than the last inspection still functioning normally adjustment planned for next service",
    "slightly elevated readings near the gearbox hum noted but still within acceptable operating range."
]


# ============================================================
# HELPERS
# ============================================================

def render_html(content):
    st.html(content)


def page_header(title, subtitle):

    render_html(
        f"""
        <div class="header">

            <div>
                <div class="header-title">
                    {title}
                </div>

                <div class="header-subtitle">
                    {subtitle}
                </div>
            </div>

            <div class="status">
                ● AI SYSTEM ONLINE
            </div>

        </div>
        """
    )


def section(title, subtitle=""):

    render_html(
        f"""
        <div class="section">

            <div class="section-title">
                {title}
            </div>

            {
                f'<div class="section-subtitle">{subtitle}</div>'
                if subtitle else ""
            }

        </div>
        """
    )


# ============================================================
# API
# ============================================================

def predict_log(log):

    response = requests.post(
        PREDICT_URL,
        json={
            "log_text": log
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def predict_machine_risk(machine_id):

    response = requests.post(
        RISK_URL,
        json={
            "machine_id": machine_id
        },
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def predict_sequence(logs):
    urls_to_try = [SEQUENCE_URL, "http://host.docker.internal:8000/predict_sequence"]

    for url in urls_to_try:
        try:
            response = requests.post(url, json={"logs": logs}, timeout=15)
            if response.status_code == 200:
                return response.json()
        except requests.exceptions.RequestException:
            continue

    raise requests.exceptions.ConnectionError(
        "Sequence API (/predict_sequence) is unreachable or rejected the request."
    )


def extract_risk_pct(resp):
    for key in ["risk_probability", "failure_probability", "probability", "risk_score", "risk"]:
        if key in resp:
            try:
                val = float(resp[key])
                return val * 100 if val <= 1 else val
            except (TypeError, ValueError):
                continue

    raise KeyError(f"No risk probability key found in API response: {resp}")


# ============================================================
# RESULT HELPERS
# ============================================================

def get_failure(result):

    if not result:
        return "No Failure"

    return str(
        result.get(
            "predicted_failure_type",
            result.get(
                "failure_type",
                "No Failure"
            )
        )
    )


def get_confidence(result):

    if not result:
        return 0

    try:

        value = float(
            result.get(
                "confidence",
                0
            )
        )

        if value <= 1:
            value *= 100

        return value

    except:

        return 0


def get_component(result):

    if not result:
        return "Unspecified"

    return str(
        result.get(
            "component",
            "Unspecified"
        )
    )


def get_severity(result):

    if not result:
        return "Unknown"

    return str(
        result.get(
            "severity",
            "Unknown"
        )
    )


def get_risk(result):

    if not result:
        return "Normal"

    value = result.get(
        "risk_level",
        "Normal"
    )

    return clean_risk(value)


def clean_risk(value):

    value = str(value)

    for symbol in [
        "🔴",
        "🟠",
        "🟡",
        "🟢",
        "⚠️",
        "⚠",
        "●",
        "•"
    ]:

        value = value.replace(
            symbol,
            ""
        )

    return value.strip()


def get_risk_icon(result):

    risk = get_risk(result).lower()

    if "critical" in risk:
        return "🔴"

    if (
        "warning" in risk
        or "moderate" in risk
        or "medium" in risk
    ):
        return "🟠"

    return "🟢"


def get_risk_color(result):

    risk = get_risk(result).lower()

    if "critical" in risk:
        return "#f87171"

    if (
        "warning" in risk
        or "moderate" in risk
        or "medium" in risk
    ):
        return "#fbbf24"

    return "#34d399"


def get_anomaly(result):

    if not result:
        return False

    value = result.get(
        "is_anomaly",
        result.get(
            "anomaly",
            False
        )
    )

    if isinstance(value, str):

        return value.lower() in [
            "true",
            "yes",
            "1",
            "detected"
        ]

    return bool(value)


def get_condition(result):

    if not result:

        return (
            "Waiting",
            "Run an AI analysis to determine machine condition.",
            "#64748b",
            "?"
        )

    if get_anomaly(result):

        risk = get_risk(result).lower()

        if "critical" in risk:

            return (
                "Critical",
                "Immediate inspection recommended.",
                "#f87171",
                "!"
            )

        return (
            "Attention Required",
            "Unusual machine behavior detected.",
            "#fbbf24",
            "!"
        )

    risk = get_risk(result).lower()

    if "critical" in risk:

        return (
            "Critical",
            "Immediate inspection recommended.",
            "#f87171",
            "!"
        )

    if (
        "warning" in risk
        or "moderate" in risk
    ):

        return (
            "Monitor",
            "Continue monitoring machine behavior.",
            "#fbbf24",
            "◐"
        )

    return (
        "Normal",
        "Equipment operating within expected range.",
        "#34d399",
        "✓"
    )


def get_recommendation(result):

    if not result:

        return (
            "Run an AI analysis to receive "
            "a maintenance recommendation."
        )

    if get_anomaly(result):

        return (
            "An unusual maintenance pattern has been detected. "
            "Inspect the equipment and review recent operating "
            "conditions."
        )

    risk = get_risk(result).lower()

    if "critical" in risk:

        return (
            "Immediate equipment inspection is recommended. "
            "Consider stopping operation if abnormal symptoms continue."
        )

    if (
        "warning" in risk
        or "moderate" in risk
    ):

        return (
            "Schedule a maintenance inspection soon and continue "
            "monitoring equipment behavior."
        )

    return (
        "Continue routine monitoring. "
        "No immediate maintenance action is required."
    )


def save_result(result):

    now = datetime.now().strftime(
        "%H:%M:%S"
    )

    st.session_state.last_result = result

    st.session_state.last_analyzed = now

    st.session_state.history.insert(
        0,
        {
            "time": now,
            "failure": get_failure(result),
            "component": get_component(result),
            "severity": get_severity(result),
            "confidence": get_confidence(result),
            "risk": get_risk(result),
            "anomaly": (
                "Detected"
                if get_anomaly(result)
                else "Normal"
            )
        }
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="logo">
            ⚙ Predict<span>AI</span>
        </div>

        <div class="logo-sub">
            Industrial Predictive Intelligence
        </div>

        <div class="sidebar-heading">
            Navigation
        </div>
        """
    )

    selected_page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "AI Log Analyzer",
            "Machine Risk",
            "Prediction History",
            "Model Pipeline"
        ],
        label_visibility="collapsed"
    )

    render_html(
        """
        <div class="system-box">

            <div class="system-title">
                Platform Status
            </div>

            <div class="system-online">
                Operational
            </div>

            <div class="system-info">
                AI services connected
            </div>

        </div>
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

if selected_page == "Dashboard":

    page_header(
        "Machine Intelligence",
        "Predictive maintenance command center"
    )

    result = st.session_state.last_result

    render_html(
        """
        <div class="intro">

            <div class="intro-title">
                Monitor equipment before failure happens.
            </div>

            <div class="intro-text">
                Analyze maintenance logs using AI-powered
                failure classification, anomaly detection,
                and predictive risk analysis.
            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    condition_name = get_condition(result)[0]

    risk = get_risk(result)

    anomaly_status = (
        "Detected"
        if get_anomaly(result)
        else "Normal"
    )

    confidence = get_confidence(result)

    render_html(
        f"""
        <div class="metrics">

            <div class="metric">

                <div class="metric-label">
                    Machine Condition
                </div>

                <div
                    class="metric-value"
                    style="
                        color:{get_condition(result)[2]};
                    "
                >
                    {condition_name}
                </div>

            </div>


            <div class="metric">

                <div class="metric-label">
                    Risk Level
                </div>

                <div
                    class="metric-value"
                    style="
                        color:{get_risk_color(result)};
                    "
                >
                    {get_risk_icon(result)} {risk}
                </div>

            </div>


            <div class="metric">

                <div class="metric-label">
                    Anomaly Status
                </div>

                <div
                    class="metric-value"
                    style="
                        color:{
                            '#fbbf24'
                            if get_anomaly(result)
                            else '#34d399'
                        };
                    "
                >
                    {anomaly_status}
                </div>

            </div>


            <div class="metric">

                <div class="metric-label">
                    AI Confidence
                </div>

                <div class="metric-value">
                    {confidence:.2f}%
                </div>

            </div>

        </div>
        """
    )


    # --------------------------------------------------------
    # CONDITION / DIAGNOSIS
    # --------------------------------------------------------

    left, right = st.columns(
        [0.85, 1.5],
        gap="large"
    )


    with left:

        name, description, color, symbol = get_condition(
            result
        )

        render_html(
            f"""
            <div class="card condition">

                <div class="card-heading">
                    Equipment Condition
                </div>

                <div
                    class="condition-symbol {
                        'condition-alert'
                        if name in ['Critical', 'Attention Required', 'Monitor']
                        else ''
                    }"
                    style="
                        color:{color};
                        border-color:{color};
                    "
                >
                    {symbol}
                </div>

                <div
                    class="condition-name"
                    style="
                        color:{color};
                    "
                >
                    {name}
                </div>

                <div class="condition-description">
                    {description}
                </div>

            </div>
            """
        )


    with right:

        failure = get_failure(result)

        diagnosis_color = (
            "#fbbf24"
            if get_anomaly(result)
            else "#34d399"
        )

        diagnosis_symbol = (
            "⚠"
            if get_anomaly(result)
            else "✓"
        )

        render_html(
            f"""
            <div class="card">

                <div class="card-heading">
                    AI Diagnosis
                </div>

                <div
                    class="diagnosis-title"
                    style="
                        color:{diagnosis_color};
                    "
                >
                    {diagnosis_symbol}
                    {failure}
                </div>


                <div class="detail">

                    <span class="detail-label">
                        Component
                    </span>

                    <span class="detail-value">
                        {get_component(result)}
                    </span>

                </div>


                <div class="detail">

                    <span class="detail-label">
                        Severity
                    </span>

                    <span class="detail-value">
                        {get_severity(result)}
                    </span>

                </div>


                <div class="detail">

                    <span class="detail-label">
                        Confidence
                    </span>

                    <span class="detail-value">
                        {confidence:.2f}%
                    </span>

                </div>


                <div class="detail">

                    <span class="detail-label">
                        Risk
                    </span>

                    <span
                        class="detail-value"
                        style="
                            color:{get_risk_color(result)};
                        "
                    >
                        {get_risk_icon(result)}
                        {risk}
                    </span>

                </div>


                <div class="recommendation">

                    <div class="recommendation-title">
                        Recommended Action
                    </div>

                    <div class="recommendation-text">
                        {get_recommendation(result)}
                    </div>

                </div>

            </div>
            """
        )


    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    section(
        "Analyze Maintenance Log",
        "Enter a machine observation and run the AI pipeline."
    )

    log = st.text_area(
        "Maintenance Log",
        placeholder=(
            "Example: Loud grinding noise from motor shaft "
            "bearing, vibration increasing rapidly..."
        ),
        height=125,
        label_visibility="collapsed"
    )

    if st.button(
        "Analyze Maintenance Log",
        type="primary",
        use_container_width=True
    ):

        if not log.strip():

            st.warning(
                "Please enter a maintenance log."
            )

        else:

            with st.spinner(
                "Analyzing AI pipeline..."
            ):

                try:

                    result = predict_log(
                        log
                    )

                    save_result(
                        result
                    )

                    st.rerun()

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Backend API is not running."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )


# ============================================================
# AI LOG ANALYZER
# ============================================================

elif selected_page == "AI Log Analyzer":

    page_header(
        "AI Log Analyzer",
        "Failure classification and anomaly detection"
    )

    section(
        "Maintenance Log",
        "Describe symptoms, inspection results, noise, vibration, or temperature changes."
    )

    log = st.text_area(
        "Maintenance Log",
        placeholder=(
            "Example: Loud grinding noise from motor shaft "
            "bearing. Vibration increasing and temperature rising."
        ),
        height=180
    )

    col1, col2 = st.columns(
        [3, 1]
    )

    with col1:

        analyze = st.button(
            "Run AI Diagnosis",
            type="primary",
            use_container_width=True
        )

    with col2:

        clear = st.button(
            "Clear",
            use_container_width=True
        )


    if clear:

        st.session_state.last_result = None

        st.session_state.last_analyzed = None

        st.rerun()


    if analyze:

        if not log.strip():

            st.warning(
                "Enter a maintenance log."
            )

        else:

            with st.spinner(
                "Running AI models..."
            ):

                try:

                    result = predict_log(
                        log
                    )

                    save_result(
                        result
                    )

                except requests.exceptions.ConnectionError:

                    st.error(
                        "Backend API is not running."
                    )

                except Exception as e:

                    st.error(
                        f"Prediction error: {e}"
                    )


    result = st.session_state.last_result


    if result:

        section(
            "Diagnosis Result",
            f"Last analyzed: {st.session_state.last_analyzed}"
        )

        cols = st.columns(4)

        values = [
            (
                "Failure Type",
                get_failure(result)
            ),
            (
                "Component",
                get_component(result)
            ),
            (
                "Confidence",
                f"{get_confidence(result):.2f}%"
            ),
            (
                "Severity",
                get_severity(result)
            )
        ]

        for col, (label, value) in zip(
            cols,
            values
        ):

            with col:

                render_html(
                    f"""
                    <div class="metric">

                        <div class="metric-label">
                            {label}
                        </div>

                        <div class="metric-value">
                            {value}
                        </div>

                    </div>
                    """
                )


        section(
            "Risk Assessment"
        )

        render_html(
            f"""
            <div class="risk">

                <div class="risk-label">
                    Current Risk
                </div>

                <div
                    class="risk-value"
                    style="
                        color:{get_risk_color(result)};
                    "
                >
                    {get_risk_icon(result)}
                    {get_risk(result)}
                </div>

            </div>
            """
        )


        section(
            "Anomaly Detection",
            "Autoencoder reconstruction error compared with learned threshold."
        )

        error = float(
            result.get(
                "reconstruction_error",
                0
            )
        )

        threshold = float(
            result.get(
                "anomaly_threshold",
                1
            )
        )

        maximum = max(
            error,
            threshold,
            0.000001
        )

        error_width = min(
            error / maximum * 100,
            100
        )

        threshold_width = min(
            threshold / maximum * 100,
            100
        )

        is_anomaly = get_anomaly(
            result
        )

        color = (
            "#fbbf24"
            if is_anomaly
            else "#34d399"
        )

        render_html(
            f"""
            <div class="
                anomaly
                {
                    'anomaly-warning'
                    if is_anomaly
                    else 'anomaly-normal'
                }
            ">

                <div class="data-label">
                    Autoencoder Status
                </div>

                <div
                    class="anomaly-title"
                    style="
                        color:{color};
                    "
                >
                    {
                        "⚠ Anomaly Detected"
                        if is_anomaly
                        else "✓ No Anomaly Detected"
                    }
                </div>


                <div class="data-label">
                    Reconstruction Error
                </div>

                <div class="data-value">
                    {error:.6f}
                </div>

                <div class="progress">

                    <div
                        style="
                            width:{error_width}%;
                            height:100%;
                            background:{color};
                        "
                    ></div>

                </div>


                <div class="data-label">
                    Learned Threshold
                </div>

                <div class="data-value">
                    {threshold:.6f}
                </div>

                <div class="progress">

                    <div
                        style="
                            width:{threshold_width}%;
                            height:100%;
                            background:#6366f1;
                        "
                    ></div>

                </div>

            </div>
            """
        )


        section(
            "Recommended Maintenance Action"
        )

        render_html(
            f"""
            <div class="card">

                <div class="recommendation-title">
                    Maintenance Recommendation
                </div>

                <div class="recommendation-text">
                    {get_recommendation(result)}
                </div>

            </div>
            """
        )


# ============================================================
# MACHINE RISK (WITH 10 UNHEALTHY LOGS POPULATE BUTTON & NO THRESHOLD LINE)
# ============================================================

elif selected_page == "Machine Risk":

    page_header(
        "Fleet Multi-Log Risk Engine",
        "Configure industrial machine fleet and analyze operational log sequences to project failure timelines"
    )

    render_html(
        """
        <div class="card">
            <div class="card-heading">Fleet Setup</div>
            <div style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 12px;">
                Specify total machine count in your facility to configure diagnostic slots.
            </div>
        </div>
        """
    )

    col_setup1, col_setup2 = st.columns([2, 1])

    with col_setup1:
        num_machines = st.number_input(
            "Select Number of Active Machines in Industry",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
            help="Configures the dynamic slots for your machine fleet."
        )

    machine_options = [f"Machine #{i+1} (M-10{i+1})" for i in range(num_machines)]

    render_html(
        """
        <div style="margin-top: 24px; margin-bottom: 10px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Sequence Log Input</div>
            <div style="font-size: 0.78rem; color: #64748b;">Select a machine and enter historical logs (up to 10 entries) to analyze failure trajectory.</div>
        </div>
        """
    )

    selected_machine = st.selectbox("Target Machine", machine_options)

    if selected_machine not in st.session_state.fleet_data:
        st.session_state.fleet_data[selected_machine] = [""] * 10

    if st.button("Populate 10 Unhealthy Logs", key="btn_unhealthy_logs", use_container_width=True):
        st.session_state.fleet_data[selected_machine] = list(LSTM_UNHEALTHY_SEQUENCE)
        for idx, text in enumerate(LSTM_UNHEALTHY_SEQUENCE):
            st.session_state[f"{selected_machine}_log_{idx}"] = text
        st.rerun()

    with st.expander(f"View / Edit 10 Sequence Logs for {selected_machine}", expanded=True):
        updated_logs = []
        grid_col1, grid_col2 = st.columns(2, gap="medium")
        
        for i in range(10):
            target_col = grid_col1 if i < 5 else grid_col2
            widget_key = f"{selected_machine}_log_{i}"
            
            if widget_key not in st.session_state:
                st.session_state[widget_key] = st.session_state.fleet_data[selected_machine][i]

            with target_col:
                val = target_col.text_input(
                    f"Log Entry #{i+1}",
                    key=widget_key,
                    placeholder=f"Enter log observation sequence {i+1}..."
                )
                updated_logs.append(val)
        
        st.session_state.fleet_data[selected_machine] = updated_logs

    if st.button(f"Predict Failure Timeline for {selected_machine}", type="primary", use_container_width=True):
        active_logs = [l for l in st.session_state.fleet_data[selected_machine] if l.strip()]
        
        if len(active_logs) < 3:
            st.warning("Please enter at least 3 historical logs for time-series analysis.")
        else:
            with st.spinner("Processing temporal log sequence through DistilBERT + LSTM Risk Engine..."):
                try:
                    full_resp = predict_sequence(active_logs)
                    risk_pct = min(max(extract_risk_pct(full_resp), 0.0), 100.0)

                    trend = []
                    for k in range(1, len(active_logs) + 1):
                        try:
                            window = [active_logs[0]] * (10 - k) + active_logs[:k]
                            trend.append(extract_risk_pct(predict_sequence(window)))
                        except Exception:
                            trend.append(None)
                    trend[-1] = risk_pct

                    estimated_days = max(1, int((100 - risk_pct) * 0.45))
                    estimated_hours = estimated_days * 24

                    risk_color = "#f87171" if risk_pct > 60 else "#fbbf24" if risk_pct > 35 else "#34d399"

                    st.session_state.machine_result = {
                        "machine_id": selected_machine,
                        "risk_pct": risk_pct,
                        "rul_days": estimated_days,
                        "rul_hours": estimated_hours,
                        "risk_color": risk_color,
                        "total_logs": len(active_logs),
                        "trend": trend
                    }
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    m_res = st.session_state.machine_result

    if m_res and m_res.get("machine_id") == selected_machine:
        st.markdown("---")
        
        res_col1, res_col2, res_col3 = st.columns(3, gap="medium")

        with res_col1:
            render_html(
                f"""
                <div class="metric" style="background: rgba(18, 24, 38, 0.7); border: 1px solid {m_res['risk_color']};">
                    <div class="metric-label">Failure Probability</div>
                    <div class="metric-value" style="color: {m_res['risk_color']};">{m_res['risk_pct']:.1f}%</div>
                    <div style="font-size:0.65rem; color:#94a3b8; margin-top:4px;">Based on {m_res['total_logs']} sequential logs</div>
                </div>
                """
            )

        with res_col2:
            render_html(
                f"""
                <div class="metric" style="background: rgba(18, 24, 38, 0.7);">
                    <div class="metric-label">Estimated Time To Failure</div>
                    <div class="metric-value">{m_res['rul_days']} Days</div>
                    <div style="font-size:0.65rem; color:#94a3b8; margin-top:4px;">~{m_res['rul_hours']} operating hours</div>
                </div>
                """
            )

        with res_col3:
            status_text = "CRITICAL ACTION" if m_res['risk_pct'] > 60 else "SCHEDULE CHECK" if m_res['risk_pct'] > 35 else "HEALTHY"
            render_html(
                f"""
                <div class="metric" style="background: rgba(18, 24, 38, 0.7); border: 1px solid {m_res['risk_color']};">
                    <div class="metric-label">Maintenance Verdict</div>
                    <div class="metric-value" style="color: {m_res['risk_color']};">{status_text}</div>
                    <div style="font-size:0.65rem; color:#94a3b8; margin-top:4px;">Rule-based decision from risk</div>
                </div>
                """
            )

        steps = [f"Log #{i+1}" for i in range(m_res['total_logs'])]
        trend_data = m_res['trend']

        import plotly.graph_objects as go
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=steps,
            y=trend_data,
            mode='lines+markers',
            connectgaps=True,
            line=dict(color=m_res['risk_color'], width=3),
            marker=dict(size=8, color='#f8fafc'),
            hovertemplate="%{x}<br>Risk: %{y:.1f}%<extra></extra>",
            name="Degradation Trend"
        ))
        # Threshold line removed per user request
        fig_trend.update_layout(
            title=f"<b>Degradation Sequence Projection ({selected_machine})</b>",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Historical Log Sequence"),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Accumulated Risk (%)", range=[0, 100]),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ============================================================
# HISTORY
# ============================================================

elif selected_page == "Prediction History":

    page_header(
        "Prediction History",
        "Previous machine analysis results"
    )

    if not st.session_state.history:

        render_html(
            """
            <div class="card"
                 style="
                    text-align:center;
                    padding:60px;
                 ">

                <div style="
                    color:#64748b;
                    font-size:0.8rem;
                ">
                    No prediction history available.
                </div>

            </div>
            """
        )

    else:

        for item in st.session_state.history:

            render_html(
                f"""
                <div class="history">

                    <div
                        style="
                            display:flex;
                            justify-content:
                            space-between;
                        "
                    >

                        <div>

                            <div class="history-title">
                                {item["failure"]}
                            </div>

                            <div class="history-time">
                                {item["time"]}
                            </div>

                        </div>

                        <div
                            style="
                                color:
                                {
                                    '#fbbf24'
                                    if item["anomaly"]
                                    == "Detected"
                                    else '#34d399'
                                };
                                font-size:0.6rem;
                                font-weight:700;
                            "
                        >
                            ● {item["anomaly"]}
                        </div>

                    </div>


                    <div style="
                        display:grid;
                        grid-template-columns:
                        repeat(4,1fr);
                        gap:20px;
                        margin-top:14px;
                    ">

                        <div>

                            <div class="data-label">
                                Component
                            </div>

                            <div class="detail-value">
                                {item["component"]}
                            </div>

                        </div>


                        <div>

                            <div class="data-label">
                                Severity
                            </div>

                            <div class="detail-value">
                                {item["severity"]}
                            </div>

                        </div>


                        <div>

                            <div class="data-label">
                                Confidence
                            </div>

                            <div class="detail-value">
                                {item["confidence"]:.2f}%
                            </div>

                        </div>


                        <div>

                            <div class="data-label">
                                Risk
                            </div>

                            <div class="detail-value">
                                {item["risk"]}
                            </div>

                        </div>

                    </div>

                </div>
                """
            )


        if st.button(
            "Clear History",
            use_container_width=True
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# MODEL PIPELINE
# ============================================================

elif selected_page == "Model Pipeline":

    page_header(
        "Model Pipeline",
        "Predictive maintenance AI architecture"
    )

    section(
        "AI Pipeline",
        "From maintenance log to maintenance recommendation."
    )

    render_html(
        """
        <div class="pipeline">

            <div class="pipeline-card">
                <div class="pipeline-step">
                    01
                </div>

                <div class="pipeline-icon">
                    📄
                </div>

                <div class="pipeline-name">
                    Maintenance Log
                </div>
            </div>


            <div class="pipeline-card">
                <div class="pipeline-step">
                    02
                </div>

                <div class="pipeline-icon">
                    NLP
                </div>

                <div class="pipeline-name">
                    DistilBERT
                </div>
            </div>


            <div class="pipeline-card">
                <div class="pipeline-step">
                    03
                </div>

                <div class="pipeline-icon">
                    AE
                </div>

                <div class="pipeline-name">
                    Autoencoder
                </div>
            </div>


            <div class="pipeline-card">
                <div class="pipeline-step">
                    04
                </div>

                <div class="pipeline-icon">
                    LSTM
                </div>

                <div class="pipeline-name">
                    Risk Prediction
                </div>
            </div>


            <div class="pipeline-card">
                <div class="pipeline-step">
                    05
                </div>

                <div class="pipeline-icon">
                    ✓
                </div>

                <div class="pipeline-name">
                    Action
                </div>
            </div>

        </div>
        """
    )

    st.write("")

    left, right = st.columns(
        2,
        gap="large"
    )

    with left:

        render_html(
            """
            <div class="card">

                <div class="card-heading">
                    Failure Classification
                </div>

                <div class="card-title">
                    DistilBERT
                </div>

                <div class="recommendation-text">
                    Classifies maintenance text and identifies
                    the predicted failure condition.
                </div>

            </div>

            <br>

            <div class="card">

                <div class="card-heading">
                    Anomaly Detection
                </div>

                <div class="card-title">
                    Autoencoder
                </div>

                <div class="recommendation-text">
                    Detects unusual machine behavior using
                    reconstruction error and threshold comparison.
                </div>

            </div>
            """
        )


    with right:

        render_html(
            """
            <div class="card">

                <div class="card-heading">
                    Failure Prediction
                </div>

                <div class="card-title">
                    LSTM
                </div>

                <div class="recommendation-text">
                    Uses historical machine sequences to estimate
                    future equipment failure risk.
                </div>

            </div>

            <br>

            <div class="card">

                <div class="card-heading">
                    Final Decision
                </div>

                <div class="card-title">
                    Maintenance Recommendation
                </div>

                <div class="recommendation-text">
                    Converts AI predictions into a practical
                    maintenance recommendation.
                </div>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div style="
        text-align:center;
        color:#475569;
        font-size:0.6rem;
        margin-top:60px;
        padding-top:20px;
        border-top:1px solid rgba(255, 255, 255, 0.05);
    ">
        PredictAI · Predictive Maintenance Platform
    </div>
    """
)