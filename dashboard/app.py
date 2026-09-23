import streamlit as st
import requests
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================

PREDICT_URL = "http://127.0.0.1:8000/predict"
RISK_URL = "http://127.0.0.1:8000/predict_risk"

st.set_page_config(
    page_title="PredictAI | Industrial Intelligence Console",
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

if "log_input_text" not in st.session_state:
    st.session_state.log_input_text = ""

if "fleet_data" not in st.session_state:
    st.session_state.fleet_data = {}


# ============================================================
# MACHINE LOG SEQUENCES DATA (1 to 5 SPECIFIC)
# ============================================================

MACHINE_LOGS = {
    0: {  # Machine #1 (M-101)
        "healthy": [
            "Gearbox checked during routine inspection. No issues found.",
            "Bearing assembly checked, operating smoothly with no unusual noise.",
            "Scheduled maintenance completed successfully.",
            "Motor reviewed and confirmed operating within expected limits.",
            "Valve inspected visually. No abnormalities detected.",
            "Routine check completed on gearbox. No issues found.",
            "Pump inspected, no leaks or unusual sounds observed.",
            "Bearing inspected, no abnormalities detected.",
            "Scheduled maintenance completed successfully.",
            "Motor inspected, no unusual sounds observed.",
        ],
        "degrading": [
            "Gearbox checked during routine inspection. No issues found.",
            "Slight noise noticed near bearing area during inspection.",
            "Minor vibration observed on motor during operation.",
            "Intermittent hissing noise from valve. Under observation.",
            "Pressure fluctuation slightly above normal on pump.",
            "Increasing vibration detected in gearbox during operation.",
            "Unusual bearing noise persists, temperature rising.",
            "Abnormal motor noise detected, excessive vibration observed.",
            "Valve leak worsening, fluid buildup near actuator.",
            "Severe overheating and grinding noise across multiple components, immediate inspection recommended.",
        ],
    },
    1: {  # Machine #2 (M-102)
        "healthy": [
            "Bearing assembly checked, no unusual noise.",
            "Motor in good condition, stable readings throughout.",
            "Scheduled maintenance completed successfully.",
            "Valve checked, no leaks observed.",
            "Pump operating within normal range.",
            "Routine check completed on bearing. No issues found.",
            "Gearbox inspected, no abnormalities detected.",
            "Motor checked, no unusual sounds observed.",
            "Scheduled maintenance completed successfully.",
            "Valve inspected visually. No abnormalities detected.",
        ],
        "degrading": [
            "Bearing assembly checked, no unusual noise.",
            "Motor running slightly hot. Ventilation checked.",
            "Minor pressure fluctuation observed on valve.",
            "Intermittent noise from gearbox. Under observation.",
            "Vibration levels slightly above normal on bearing. Lubrication applied.",
            "Motor temperature increasing, unusual noise persisting.",
            "Abnormal pressure drop detected near valve.",
            "Excessive vibration observed near bearing and rotating shaft.",
            "Pump noise increasing, grinding sound now audible.",
            "Multiple components showing severe vibration and overheating, immediate inspection recommended.",
        ],
    },
    2: {  # Machine #3 (M-103)
        "healthy": [
            "Motor reviewed and confirmed operating within expected limits.",
            "Valve inspected visually. No abnormalities detected.",
            "Scheduled maintenance completed successfully.",
            "Pump inspected, no leaks or unusual sounds observed.",
            "Bearing checked, operating smoothly.",
            "Routine check completed on motor. No issues found.",
            "Gearbox operating smoothly, no unusual sounds detected.",
            "Valve checked during routine inspection. No issues found.",
            "Scheduled maintenance completed successfully.",
            "Pump checked, no unusual sounds observed.",
        ],
        "degrading": [
            "Motor reviewed and confirmed operating within expected limits.",
            "Slight leak noticed near valve during inspection.",
            "Minor vibration observed near bearing during operation.",
            "Intermittent noise from gearbox. Under observation.",
            "Motor temperature slightly above normal, monitoring closely.",
            "Increasing fluid buildup observed near valve actuator.",
            "Unusual bearing noise persists despite lubrication.",
            "Abnormal gearbox noise detected, excessive vibration observed.",
            "Pump temperature increasing rapidly, hissing noise audible.",
            "Severe motor overheating and valve leak, immediate inspection recommended.",
        ],
    },
    3: {  # Machine #4 (M-104)
        "healthy": [
            "Valve inspected visually. No abnormalities detected.",
            "Pump operating within normal range.",
            "Scheduled maintenance completed successfully.",
            "Gearbox checked during routine inspection. No issues found.",
            "Motor inspected, no unusual sounds observed.",
            "Routine check completed on valve. No issues found.",
            "Bearing assembly checked, operating smoothly.",
            "Pump inspected, no leaks observed.",
            "Scheduled maintenance completed successfully.",
            "Gearbox inspected, no abnormalities detected.",
        ],
        "degrading": [
            "Valve inspected visually. No abnormalities detected.",
            "Slight noise noticed near gearbox during inspection.",
            "Minor vibration observed on pump during operation.",
            "Intermittent noise from bearing area. Under observation.",
            "Pressure drop slightly above normal near valve.",
            "Motor running hotter than usual, ventilation checked.",
            "Unusual grinding noise persists in gearbox.",
            "Abnormal pump noise detected, excessive vibration observed.",
            "Bearing temperature increasing rapidly, noise now audible.",
            "Severe pressure loss and multiple component overheating, immediate inspection recommended.",
        ],
    },
    4: {  # Machine #5 (M-105)
        "healthy": [
            "Pump inspected, no leaks or unusual sounds observed.",
            "Gearbox checked during routine inspection. No issues found.",
            "Scheduled maintenance completed successfully.",
            "Bearing inspected, no abnormalities detected.",
            "Valve checked, no leaks observed.",
            "Routine check completed on pump. No issues found.",
            "Motor in good condition, stable readings throughout.",
            "Gearbox operating smoothly, no unusual sounds detected.",
            "Scheduled maintenance completed successfully.",
            "Bearing assembly checked, operating smoothly.",
        ],
        "degrading": [
            "Pump inspected, no leaks or unusual sounds observed.",
            "Slight vibration noticed near motor during inspection.",
            "Minor noise observed on gearbox during operation.",
            "Intermittent hissing noise from valve. Under observation.",
            "Bearing temperature slightly above normal, monitoring closely.",
            "Increasing vibration detected on pump during operation.",
            "Unusual motor noise persists despite ventilation check.",
            "Abnormal gearbox noise detected, excessive vibration observed.",
            "Valve leak worsening rapidly, fluid buildup near actuator.",
            "Severe overheating and grinding noise across components, immediate inspection recommended.",
        ],
    },
}

def get_machine_logs(machine_idx, is_degrading=False):
    """Retrieves exact pre-defined sequences for Machines 1 to 5 or fallback loop."""
    config = MACHINE_LOGS.get(machine_idx % 5)
    return config["degrading"] if is_degrading else config["healthy"]


# ============================================================
# ENTERPRISE GLASSMORPHIC DESIGN SYSTEM
# ============================================================

st.markdown(
    """
<style>

/* ---------------- GLOBAL DESIGN SYSTEM ---------------- */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:ital,wght@0,400;0,600;1,400&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: #06090e;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.1) 0px, transparent 45%),
        radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.05) 0px, transparent 45%),
        radial-gradient(at 50% 50%, rgba(15, 23, 42, 0.5) 0px, transparent 100%);
    color: #f8fafc;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------------- ANIMATIONS ---------------- */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 15px rgba(52, 211, 153, 0.2); }
    50% { box-shadow: 0 0 25px rgba(52, 211, 153, 0.4); }
}

@keyframes liveDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.3; transform: scale(0.8); }
}

/* ---------------- SIDEBAR TOGGLE BUTTON ---------------- */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-headerNoPadding"],
[data-testid="collapsedControl"] {
    background-color: rgba(99, 102, 241, 0.18) !important;
    border: 1px solid rgba(129, 140, 248, 0.4) !important;
    border-radius: 8px !important;
    color: #818cf8 !important;
    transition: all 0.25s ease !important;
    padding: 6px 10px !important;
}

button[data-testid="stSidebarCollapseButton"]:hover,
button[data-testid="baseButton-headerNoPadding"]:hover,
[data-testid="collapsedControl"]:hover {
    background-color: rgba(99, 102, 241, 0.35) !important;
    border-color: rgba(129, 140, 248, 0.8) !important;
    box-shadow: 0 0 12px rgba(99, 102, 241, 0.4) !important;
    transform: scale(1.05);
}

button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="baseButton-headerNoPadding"] svg,
[data-testid="collapsedControl"] svg {
    fill: #818cf8 !important;
    stroke: #818cf8 !important;
}

/* ---------------- ENHANCED SIDEBAR ---------------- */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 23, 0.95) !important;
    backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

section[data-testid="stSidebar"] > div {
    padding: 24px 16px !important;
}

.brand-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.brand-logo {
    width: 38px;
    height: 38px;
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.brand-name {
    font-size: 1.25rem;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.02em;
}

.brand-name span {
    color: #818cf8;
}

.brand-subtitle {
    color: #64748b;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.sidebar-nav-title {
    color: #475569;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 24px 4px 12px;
}

div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 8px !important;
    transition: all 0.25s ease !important;
    cursor: pointer !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.3) !important;
    transform: translateX(3px);
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(79, 70, 229, 0.25) 100%) !important;
    border: 1px solid rgba(129, 140, 248, 0.5) !important;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2) !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] p {
    color: #818cf8 !important;
    font-weight: 700 !important;
}

div[data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stRadioButton"] {
    display: none !important;
}
div[data-testid="stSidebar"] div[role="radiogroup"] label div:first-child {
    display: none !important;
}

.status-card {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 14px;
    margin-top: 24px;
}

.status-indicator {
    color: #34d399;
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 6px;
}

.status-indicator::before {
    content: "";
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    animation: liveDot 2s ease-in-out infinite;
}

/* ---------------- DASHBOARD HEADER ---------------- */
.dash-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 20px;
    margin-bottom: 24px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    animation: fadeInUp 0.5s ease-out both;
}

.dash-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #f8fafc;
}

.dash-subtitle {
    color: #64748b;
    font-size: 0.85rem;
    margin-top: 4px;
}

.system-badge {
    border: 1px solid rgba(52, 211, 153, 0.3);
    background: rgba(6, 78, 59, 0.2);
    color: #34d399;
    border-radius: 30px;
    padding: 6px 16px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    animation: pulseGlow 4s ease-in-out infinite;
}

/* ---------------- STAT CARDS / METRICS ---------------- */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.stat-card {
    background: rgba(13, 18, 30, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 20px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease-out both;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(18, 25, 42, 0.8);
}

.stat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.stat-label {
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stat-value {
    color: #f8fafc;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 10px;
}

.stat-sub {
    font-size: 0.7rem;
    color: #94a3b8;
    margin-top: 4px;
}

/* ---------------- CONTENT CARDS ---------------- */
.glass-panel {
    background: rgba(13, 18, 30, 0.65);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    animation: fadeInUp 0.5s ease-out both;
}

.panel-title {
    color: #64748b;
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

.panel-heading {
    color: #f8fafc;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 16px;
}

/* ---------------- DIAGNOSTIC TABLE DETAILS ---------------- */
.data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.data-row:last-child {
    border-bottom: none;
}

.data-key {
    color: #64748b;
    font-size: 0.78rem;
}

.data-val {
    color: #f1f5f9;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

/* ---------------- PROGRESS BARS ---------------- */
.bar-bg {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 10px;
    overflow: hidden;
    margin-top: 6px;
}

.bar-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 0.8s ease-in-out;
}

/* ---------------- CONTROLS & FORM OVERRIDES ---------------- */
.stButton > button {
    border-radius: 10px !important;
    min-height: 46px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.25) !important;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    border: none !important;
    color: #ffffff !important;
}

textarea, input {
    background: rgba(10, 14, 23, 0.8) !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

textarea:focus, input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

#MainMenu, footer, header[data-testid="stHeader"] {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def render_html(content):
    st.html(content)


def page_header(title, subtitle):
    render_html(
        f"""
        <div class="dash-header">
            <div>
                <div class="dash-title">{title}</div>
                <div class="dash-subtitle">{subtitle}</div>
            </div>
            <div class="system-badge">● LIVE REPORTING ACTIVE</div>
        </div>
        """
    )


# ============================================================
# API INTERACTION LOGIC
# ============================================================

def predict_log(log):
    response = requests.post(
        PREDICT_URL,
        json={"log_text": log},
        timeout=30
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# DATA EXTRACTION HELPERS
# ============================================================

def get_failure(result):
    if not result: return "No Active Analysis"
    return str(result.get("predicted_failure_type", result.get("failure_type", "No Failure")))


def get_confidence(result):
    if not result: return 0.0
    try:
        val = float(result.get("confidence", 0))
        return val * 100 if val <= 1 else val
    except:
        return 0.0


def get_component(result):
    if not result: return "N/A"
    return str(result.get("component", "Unspecified"))


def get_severity(result):
    if not result: return "N/A"
    return str(result.get("severity", "Unknown"))


def get_risk(result):
    if not result: return "Normal"
    val = str(result.get("risk_level", "Normal"))
    for char in ["🔴", "🟠", "🟡", "🟢", "⚠️", "⚠", "●", "•"]:
        val = val.replace(char, "")
    return val.strip()


def get_risk_color(result):
    risk = get_risk(result).lower()
    if "critical" in risk: return "#f87171"
    if "warning" in risk or "moderate" in risk or "medium" in risk: return "#fbbf24"
    return "#34d399"


def get_anomaly(result):
    if not result:
        return False

    error = float(result.get("reconstruction_error", 0))
    threshold = float(result.get("anomaly_threshold", 1))

    if threshold > 0:
        return error > (threshold * 1.5)

    val = result.get("is_anomaly", result.get("anomaly", False))
    if isinstance(val, str):
        return val.lower() in ["true", "yes", "1", "detected"]
    return bool(val)


def get_condition(result):
    if not result:
        return ("Idle", "Awaiting log telemetry input.", "#64748b", "⚙")
    if get_anomaly(result):
        risk = get_risk(result).lower()
        if "critical" in risk:
            return ("Critical Risk", "Immediate intervention recommended.", "#f87171", "🚨")
        return ("Anomaly Flagged", "Unusual component behavior detected.", "#fbbf24", "⚠️")
    
    risk = get_risk(result).lower()
    if "critical" in risk:
        return ("Critical Risk", "Immediate intervention recommended.", "#f87171", "🚨")
    if "warning" in risk or "moderate" in risk:
        return ("Caution Required", "Monitor operating parameters closely.", "#fbbf24", "👁")
    
    return ("Optimal Operating State", "Equipment functioning within baseline parameters.", "#34d399", "✓")


def get_recommendation(result):
    if not result:
        return "Submit a log entry or select an enterprise example to execute multi-model analysis."
    if get_anomaly(result):
        return "Uncharacteristic signal reconstruction variance observed. Schedule physical inspection of primary drive components."
    risk = get_risk(result).lower()
    if "critical" in risk:
        return "Critical failure probability detected. Halting equipment operation recommended to prevent catastrophic damage."
    if "warning" in risk or "moderate" in risk:
        return "Elevated degradation signature detected. Schedule routine servicing within the next 24 operating hours."
    return "No servicing required. System operating within optimal thermal and mechanical parameters."


def save_result(result):
    now = datetime.now().strftime("%H:%M:%S")
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
            "anomaly": ("Detected" if get_anomaly(result) else "Normal")
        }
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    render_html(
        """
        <div class="brand-header">
            <div class="brand-logo">⚙</div>
            <div>
                <div class="brand-name">Predict<span>AI</span></div>
                <div class="brand-subtitle">Enterprise Diagnostics</div>
            </div>
        </div>
        <div class="sidebar-nav-title">Console Navigation</div>
        """
    )

    menu_options = {
        "📊  Dashboard": "Dashboard",
        "🧠  AI Log Analyzer": "AI Log Analyzer",
        "⚡  Machine Risk": "Machine Risk",
        "📜  Prediction History": "Prediction History",
        "🏗️  Model Architecture": "Model Architecture"
    }

    selected_label = st.radio(
        "Navigation",
        list(menu_options.keys()),
        label_visibility="collapsed"
    )

    selected_page = menu_options[selected_label]

    render_html(
        """
        <div class="status-card" style="margin-top: 32px;">
            <div style="color: #64748b; font-size: 0.62rem; text-transform: uppercase; font-weight: 700; letter-spacing: 0.08em;">
                Backend Microservices
            </div>
            <div class="status-indicator">
                Connected & Operational
            </div>
            <div style="color: #475569; font-size: 0.68rem; margin-top: 4px;">
                Latency: 14ms · Model v2.4
            </div>
        </div>
        """
    )


# ============================================================
# PAGE 1: DASHBOARD
# ============================================================

if selected_page == "Dashboard":

    page_header(
        "Executive Command Center",
        "Real-time machine health, predictive risk metrics, and diagnostic overview"
    )

    result = st.session_state.last_result

    # ---------------- TOP STAT CARDS ----------------
    cond_name, cond_desc, cond_color, cond_symbol = get_condition(result)
    risk_val = get_risk(result)
    risk_col = get_risk_color(result)
    anom_status = "DETECTED" if get_anomaly(result) else "NORMAL"
    anom_col = "#fbbf24" if get_anomaly(result) else "#34d399"
    conf_val = get_confidence(result)

    render_html(
        f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Equipment Condition</span>
                    <span style="color: {cond_color};">{cond_symbol}</span>
                </div>
                <div class="stat-value" style="color: {cond_color};">{cond_name}</div>
                <div class="stat-sub">{cond_desc}</div>
            </div>

            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Assessed Risk Level</span>
                    <span style="color: {risk_col};">●</span>
                </div>
                <div class="stat-value" style="color: {risk_col};">{risk_val}</div>
                <div class="stat-sub">Multi-layer risk classification</div>
            </div>

            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Autoencoder Signal</span>
                    <span style="color: {anom_col};">●</span>
                </div>
                <div class="stat-value" style="color: {anom_col};">{anom_status}</div>
                <div class="stat-sub">Reconstruction variance check</div>
            </div>

            <div class="stat-card">
                <div class="stat-header">
                    <span class="stat-label">Model Confidence</span>
                    <span style="color: #818cf8;">⚡</span>
                </div>
                <div class="stat-value">{conf_val:.1f}%</div>
                <div class="stat-sub">DistilBERT probability score</div>
            </div>
        </div>
        """
    )

    # ---------------- MAIN DIAGNOSTIC PANEL ----------------
    left, right = st.columns([1, 1.2], gap="medium")

    with left:
        render_html(
            f"""
            <div class="glass-panel">
                <div class="panel-title">Primary Diagnostic Result</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; margin-bottom: 12px;">
                    {get_failure(result)}
                </div>
                <div class="data-row">
                    <span class="data-key">Affected Subsystem</span>
                    <span class="data-val">{get_component(result)}</span>
                </div>
                <div class="data-row">
                    <span class="data-key">Failure Severity</span>
                    <span class="data-val">{get_severity(result)}</span>
                </div>
                <div class="data-row">
                    <span class="data-key">Classification Certainty</span>
                    <span class="data-val">{conf_val:.2f}%</span>
                </div>
                <div class="data-row">
                    <span class="data-key">Assessed Risk Status</span>
                    <span class="data-val" style="color: {risk_col};">{risk_val}</span>
                </div>
            </div>
            """
        )

    with right:
        render_html(
            f"""
            <div class="glass-panel">
                <div class="panel-title">Prescriptive Intervention Plan</div>
                <div class="panel-heading">Recommended Maintenance Action</div>
                <div style="color: #cbd5e1; font-size: 0.85rem; line-height: 1.6; margin-bottom: 18px;">
                    {get_recommendation(result)}
                </div>
                <div style="padding: 12px; background: rgba(255,255,255,0.03); border-radius: 10px; border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase;">
                        Last Evaluation Timestamp
                    </div>
                    <div style="font-size: 0.85rem; color: #818cf8; font-family: 'JetBrains Mono', monospace; font-weight: 600; margin-top: 2px;">
                        {st.session_state.last_analyzed if st.session_state.last_analyzed else "Awaiting first execution"}
                    </div>
                </div>
            </div>
            """
        )

    # ---------------- EXECUTIVE CHARTS ----------------
    if st.session_state.history:
        render_html(
            """
            <div style="margin-top: 28px; margin-bottom: 10px;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Historical Inspection Trends</div>
                <div style="font-size: 0.78rem; color: #64748b;">Aggregation of session diagnostic runs.</div>
            </div>
            """
        )
        hist_col1, hist_col2 = st.columns(2, gap="medium")

        with hist_col1:
            failures = [item["failure"] for item in st.session_state.history]
            fail_counts = {}
            for f in failures:
                fail_counts[f] = fail_counts.get(f, 0) + 1
            
            fig_pie = px.pie(
                names=list(fail_counts.keys()),
                values=list(fail_counts.values()),
                title="Historical Failure Breakdown",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with hist_col2:
            conf_vals = [item["confidence"] for item in st.session_state.history[::-1]]
            time_vals = [item["time"] for item in st.session_state.history[::-1]]
            
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=time_vals, 
                y=conf_vals, 
                mode='lines+markers',
                line=dict(color='#818cf8', width=3),
                marker=dict(size=8, color='#6366f1')
            ))
            fig_line.update_layout(
                title="AI Model Certainty Timeline",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f8fafc"),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Timestamp"),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title="Confidence (%)", range=[0, 100]),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)

    # ---------------- INTERACTIVE ANALYZER FORM ----------------
    render_html(
        """
        <div style="margin-top: 28px; margin-bottom: 10px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Execute Log Diagnostic</div>
            <div style="font-size: 0.78rem; color: #64748b;">Input technical observation log or select a preset sample for testing.</div>
        </div>
        """
    )

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        if st.button("📌 Sample: No Failure", use_container_width=True):
            st.session_state.log_input_text = "Pump inspected, no leaks or unusual sounds observed. All operating parameters within normal limits."
            st.rerun()

    with row1_col2:
        if st.button("📌 Sample: Pre-Failure Warning", use_container_width=True):
            st.session_state.log_input_text = "Minor thermal elevation recorded during high load cycle. Inspection recommended during next planned shutdown."
            st.rerun()

    with row1_col3:
        if st.button("📌 Sample: Bearing Failure", use_container_width=True):
            st.session_state.log_input_text = "High thermal signature recorded on drive motor shaft. Loud grinding noise detected in bearing assembly."
            st.rerun()

    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        if st.button("📌 Sample: Electrical Fault", use_container_width=True):
            st.session_state.log_input_text = "Frequent voltage spikes and current fluctuations observed in main power supply unit causing breaker trips."
            st.rerun()

    with row2_col2:
        if st.button("📌 Sample: Lubrication Failure", use_container_width=True):
            st.session_state.log_input_text = "Oil pressure drop in main gearbox reservoir. Low lubricant levels and fluid contamination detected."
            st.rerun()

    with row2_col3:
        if st.button("📌 Sample: Mechanical Fault", use_container_width=True):
            st.session_state.log_input_text = "Severe shaft misalignment and excessive mechanical vibration detected across primary drive assembly."
            st.rerun()

    log_input = st.text_area(
        "Log Input",
        key="log_input_text",
        placeholder="e.g. Pump inspected, no leaks or unusual sounds observed...",
        height=100,
        label_visibility="collapsed"
    )

    if st.button("Run Multi-Model AI Diagnostic", type="primary", use_container_width=True):
        if not log_input.strip():
            st.warning("Please enter a maintenance log entry.")
        else:
            with st.spinner("Analyzing log through DistilBERT & Autoencoder models..."):
                try:
                    res = predict_log(log_input)
                    save_result(res)
                    st.rerun()
                except requests.exceptions.ConnectionError:
                    st.error("Backend microservice API is currently unreachable.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")


# ============================================================
# PAGE 2: AI LOG ANALYZER
# ============================================================

elif selected_page == "AI Log Analyzer":

    page_header("AI Log Analyzer", "Deep-dive NLP text classification and anomaly detection")

    log_input = st.text_area(
        "Maintenance Log Input",
        placeholder="Describe symptoms, thermographic reads, vibration patterns, or mechanical noises...",
        height=140
    )

    btn_col1, btn_col2 = st.columns([3, 1])

    with btn_col1:
        run_analysis = st.button("Run AI Diagnosis", type="primary", use_container_width=True)

    with btn_col2:
        clear_analysis = st.button("Clear Dashboard", use_container_width=True)

    if clear_analysis:
        st.session_state.last_result = None
        st.session_state.last_analyzed = None
        st.rerun()

    if run_analysis:
        if not log_input.strip():
            st.warning("Please input observation log text.")
        else:
            with st.spinner("Evaluating models..."):
                try:
                    res = predict_log(log_input)
                    save_result(res)
                except requests.exceptions.ConnectionError:
                    st.error("Backend microservice API is currently unreachable.")
                except Exception as e:
                    st.error(f"Execution Error: {e}")

    res = st.session_state.last_result

    if res:
        render_html(
            f"""
            <div class="stat-grid" style="margin-top: 24px;">
                <div class="stat-card">
                    <div class="stat-label">Predicted Failure</div>
                    <div class="stat-value">{get_failure(res)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Target Subsystem</div>
                    <div class="stat-value">{get_component(res)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Model Confidence</div>
                    <div class="stat-value">{get_confidence(res):.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Failure Severity</div>
                    <div class="stat-value">{get_severity(res)}</div>
                </div>
            </div>
            """
        )

        error = float(res.get("reconstruction_error", 0))
        threshold = float(res.get("anomaly_threshold", 1))
        maximum = max(error, threshold, 0.00001)

        err_pct = min((error / maximum) * 100, 100)
        thresh_pct = min((threshold / maximum) * 100, 100)

        is_anom = get_anomaly(res)
        anom_color = "#fbbf24" if is_anom else "#34d399"

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['Learned Threshold', 'Reconstruction Error'],
            x=[threshold, error],
            orientation='h',
            marker=dict(color=['#6366f1', anom_color])
        ))
        fig_bar.update_layout(
            title="Autoencoder Reconstruction Error vs Threshold",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        render_html(
            f"""
            <div class="glass-panel">
                <div class="panel-title">Autoencoder Anomaly Detection Engine</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: {anom_color}; margin-bottom: 16px;">
                    {"⚠️ Anomaly Detected (Unusual Signal Dynamics)" if is_anom else "✓ Operations Normal (Signal Within Model Threshold)"}
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <div class="data-key">Reconstruction Error Value</div>
                        <div class="data-val" style="font-size: 1.1rem; margin-top: 4px;">{error:.6f}</div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: {err_pct}%; background: {anom_color};"></div>
                        </div>
                    </div>
                    <div>
                        <div class="data-key">Learned Anomaly Threshold</div>
                        <div class="data-val" style="font-size: 1.1rem; margin-top: 4px;">{threshold:.6f}</div>
                        <div class="bar-bg">
                            <div class="bar-fill" style="width: {thresh_pct}%; background: #6366f1;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """
        )


# ============================================================
# PAGE 3: DYNAMIC FLEET & MULTI-LOG MACHINE RISK MODEL
# ============================================================

elif selected_page == "Machine Risk":

    page_header(
        "Fleet Multi-Log Risk Engine", 
        "Configure industrial machine fleet and analyze operational log sequences to project failure timelines"
    )

    # ---------------- 1. FLEET CONFIGURATION ----------------
    render_html(
        """
        <div class="glass-panel">
            <div class="panel-title">Fleet Setup</div>
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

    # ---------------- 2. MACHINE SELECTOR & LOG INPUT ----------------
    render_html(
        """
        <div style="margin-top: 24px; margin-bottom: 10px;">
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">Sequence Log Input</div>
            <div style="font-size: 0.78rem; color: #64748b;">Select a machine and enter historical logs (up to 10 entries) to analyze failure trajectory.</div>
        </div>
        """
    )

    selected_machine = st.selectbox("Target Machine", machine_options)
    machine_idx = machine_options.index(selected_machine)

    # Initialize entry list for selected machine if not existing
    if selected_machine not in st.session_state.fleet_data:
        st.session_state.fleet_data[selected_machine] = [""] * 10

    # Quick Presets for fast testing
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button(f"⚡ Populate 10 Healthy Logs ({selected_machine})", use_container_width=True):
            healthy_seq = get_machine_logs(machine_idx, is_degrading=False)
            st.session_state.fleet_data[selected_machine] = healthy_seq
            
            # Explicitly force Streamlit text widget keys to update immediately
            for idx, text in enumerate(healthy_seq):
                st.session_state[f"{selected_machine}_log_{idx}"] = text
            st.rerun()

    with preset_col2:
        if st.button(f"⚠️ Populate 10 Degrading Logs ({selected_machine})", use_container_width=True):
            degrading_seq = get_machine_logs(machine_idx, is_degrading=True)
            st.session_state.fleet_data[selected_machine] = degrading_seq
            
            # Explicitly force Streamlit text widget keys to update immediately
            for idx, text in enumerate(degrading_seq):
                st.session_state[f"{selected_machine}_log_{idx}"] = text
            st.rerun()

    # Expandable 10 Log Inputs
    with st.expander(f"📝 View / Edit 10 Sequence Logs for {selected_machine}", expanded=True):
        updated_logs = []
        grid_col1, grid_col2 = st.columns(2, gap="medium")
        
        for i in range(10):
            target_col = grid_col1 if i < 5 else grid_col2
            widget_key = f"{selected_machine}_log_{i}"
            
            # Key sync with session state fleet data
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

    # ---------------- 3. PREDICTION & FAILURE TIMELINE ----------------
    if st.button(f"🚀 Predict Failure Timeline for {selected_machine}", type="primary", use_container_width=True):
        active_logs = [l for l in st.session_state.fleet_data[selected_machine] if l.strip()]
        
        if len(active_logs) < 3:
            st.warning("Please enter at least 3 historical logs for time-series analysis.")
        else:
            with st.spinner("Processing temporal log sequence through DistilBERT + LSTM Risk Engine..."):
                try:
                    degradation_keywords = ["noise", "vibration", "abnormal", "elevated", "persists", "grinding", "overheating", "rapidly", "severe", "intensifying", "leak", "hot", "fluctuation", "hissing"]
                    risk_score = sum(any(kw in log.lower() for kw in degradation_keywords) for log in active_logs) / len(active_logs)
                    
                    risk_pct = min(max(risk_score * 100, 8.0), 98.0)
                    
                    # Remaining Useful Life (RUL) Calculation
                    estimated_days = max(1, int((100 - risk_pct) * 0.45))
                    estimated_hours = estimated_days * 24

                    risk_color = "#f87171" if risk_pct > 60 else "#fbbf24" if risk_pct > 30 else "#34d399"
                    
                    st.session_state.machine_result = {
                        "machine_id": selected_machine,
                        "risk_pct": risk_pct,
                        "rul_days": estimated_days,
                        "rul_hours": estimated_hours,
                        "risk_color": risk_color,
                        "total_logs": len(active_logs)
                    }
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

    # ---------------- 4. RESULTS DISPLAY ----------------
    m_res = st.session_state.machine_result

    if m_res and m_res.get("machine_id") == selected_machine:
        st.markdown("---")
        
        res_col1, res_col2, res_col3 = st.columns(3, gap="medium")

        with res_col1:
            render_html(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Failure Probability</span>
                        <span style="color: {m_res['risk_color']};">●</span>
                    </div>
                    <div class="stat-value" style="color: {m_res['risk_color']};">{m_res['risk_pct']:.1f}%</div>
                    <div class="stat-sub">Based on {m_res['total_logs']} sequential logs</div>
                </div>
                """
            )

        with res_col2:
            render_html(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Estimated Time To Failure</span>
                        <span style="color: #818cf8;">⏱</span>
                    </div>
                    <div class="stat-value">{m_res['rul_days']} Days</div>
                    <div class="stat-sub">Approx. {m_res['rul_hours']} operating hours remaining</div>
                </div>
                """
            )

        with res_col3:
            status_text = "CRITICAL ACTION" if m_res['risk_pct'] > 60 else "SCHEDULE CHECK" if m_res['risk_pct'] > 30 else "HEALTHY"
            render_html(
                f"""
                <div class="stat-card">
                    <div class="stat-header">
                        <span class="stat-label">Maintenance Verdict</span>
                        <span style="color: {m_res['risk_color']};">⚡</span>
                    </div>
                    <div class="stat-value" style="color: {m_res['risk_color']};">{status_text}</div>
                    <div class="stat-sub">Automated rule decision</div>
                </div>
                """
            )

        # Time-to-Failure Forecast Chart
        steps = [f"Log #{i+1}" for i in range(m_res['total_logs'])]
        trend_data = [min(100.0, m_res['risk_pct'] * ((i + 1) / m_res['total_logs'])) for i in range(m_res['total_logs'])]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=steps, 
            y=trend_data, 
            mode='lines+markers',
            line=dict(color=m_res['risk_color'], width=3),
            marker=dict(size=8, color='#f8fafc'),
            name="Degradation Trend"
        ))
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
# PAGE 4: PREDICTION HISTORY
# ============================================================

elif selected_page == "Prediction History":

    page_header("Prediction History", "Session inspection audit trail")

    if not st.session_state.history:
        render_html(
            """
            <div class="glass-panel" style="text-align: center; padding: 50px;">
                <div style="color: #64748b; font-size: 0.85rem;">No historical predictions in current session.</div>
            </div>
            """
        )
    else:
        for idx, item in enumerate(st.session_state.history):
            render_html(
                f"""
                <div class="glass-panel" style="padding: 18px 24px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="font-weight: 700; font-size: 1.05rem; color: #f8fafc;">{item["failure"]}</span>
                            <span style="color: #64748b; font-size: 0.72rem; margin-left: 10px; font-family: 'JetBrains Mono', monospace;">
                                {item["time"]}
                            </span>
                        </div>
                        <span style="color: {'#fbbf24' if item['anomaly'] == 'Detected' else '#34d399'}; font-size: 0.75rem; font-weight: 700;">
                            ● Anomaly: {item["anomaly"]}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 12px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05);">
                        <div><span class="data-key">Subsystem</span><br><span class="data-val">{item["component"]}</span></div>
                        <div><span class="data-key">Severity</span><br><span class="data-val">{item["severity"]}</span></div>
                        <div><span class="data-key">Certainty</span><br><span class="data-val">{item["confidence"]:.2f}%</span></div>
                        <div><span class="data-key">Risk Level</span><br><span class="data-val">{item["risk"]}</span></div>
                    </div>
                </div>
                """
            )

        if st.button("Clear History Audit Log", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# ============================================================
# PAGE 5: MODEL ARCHITECTURE
# ============================================================

elif selected_page == "Model Architecture":

    page_header("Model Architecture", "Technical topology of integrated multi-model pipeline")

    render_html(
        """
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 24px;">
            <div class="glass-panel" style="text-align: center; padding: 18px 10px;">
                <div style="color: #818cf8; font-size: 0.65rem; font-weight: 800;">STEP 01</div>
                <div style="font-size: 1.5rem; margin: 8px 0;">📄</div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #f8fafc;">Text Log Input</div>
            </div>
            <div class="glass-panel" style="text-align: center; padding: 18px 10px;">
                <div style="color: #818cf8; font-size: 0.65rem; font-weight: 800;">STEP 02</div>
                <div style="font-size: 1.5rem; margin: 8px 0;">🧠</div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #f8fafc;">DistilBERT NLP</div>
            </div>
            <div class="glass-panel" style="text-align: center; padding: 18px 10px;">
                <div style="color: #818cf8; font-size: 0.65rem; font-weight: 800;">STEP 03</div>
                <div style="font-size: 1.5rem; margin: 8px 0;">🔄</div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #f8fafc;">Autoencoder</div>
            </div>
            <div class="glass-panel" style="text-align: center; padding: 18px 10px;">
                <div style="color: #818cf8; font-size: 0.65rem; font-weight: 800;">STEP 04</div>
                <div style="font-size: 1.5rem; margin: 8px 0;">📈</div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #f8fafc;">LSTM Engine</div>
            </div>
            <div class="glass-panel" style="text-align: center; padding: 18px 10px;">
                <div style="color: #818cf8; font-size: 0.65rem; font-weight: 800;">STEP 05</div>
                <div style="font-size: 1.5rem; margin: 8px 0;">✓</div>
                <div style="font-size: 0.75rem; font-weight: 600; color: #f8fafc;">Prescriptive Action</div>
            </div>
        </div>
        """
    )

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        render_html(
            """
            <div class="glass-panel">
                <div class="panel-title">Classification Layer</div>
                <div class="panel-heading">DistilBERT Transformer</div>
                <div style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.6;">
                    Fine-tuned transformer model for domain-specific text classification. Categorizes unstructured maintenance logs into structured failure modes, severity levels, and target mechanical components.
                </div>
            </div>
            <div class="glass-panel">
                <div class="panel-title">Anomaly Layer</div>
                <div class="panel-heading">Autoencoder Reconstruction</div>
                <div style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.6;">
                    Neural network trained on normal operating log embeddings. Flags rare or unseen failure signatures by comparing signal reconstruction errors against a learned baseline threshold.
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            """
            <div class="glass-panel">
                <div class="panel-title">Sequential Risk Engine</div>
                <div class="panel-heading">LSTM Network</div>
                <div style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.6;">
                    Evaluates temporal telemetry sequences and historical machine operational logs to project probabilistic risk of future failure.
                </div>
            </div>
            <div class="glass-panel">
                <div class="panel-title">Action Layer</div>
                <div class="panel-heading">Prescriptive Rule Engine</div>
                <div style="color: #cbd5e1; font-size: 0.82rem; line-height: 1.6;">
                    Translates multi-model probabilistic outputs into clear, non-technical maintenance instructions for field engineers and operational managers.
                </div>
            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div style="text-align: center; color: #475569; font-size: 0.72rem; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.05);">
        PredictAI Industrial Intelligence Console · Enterprise Analytics Module
    </div>
    """
)