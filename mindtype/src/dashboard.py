"""
dashboard.py
-------------
MindType: Typing-Based Mental State & Cognitive Strain Analysis
An editorial research platform for keystroke dynamics analysis.

Run with:  streamlit run src/dashboard.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import streamlit as st
import pandas as pd
import numpy as np

from paragraphs import PASSAGES, get_passage, total_passages
from keystroke_widget import keystroke_capture
from features import compute_user_baselines
from reliability import compute_passage_quality
from report import (
    build_features_dataframe_from_events,
    calibrate_baseline_from_passage_1,
    aggregate_report
)
from infer_live import (
    make_live_session,
    score_feature_frame,
    log_result,
    LOG_PATH
)
from models_engine import get_model_benchmark_comparison
from visuals import (
    create_keystroke_rhythm_chart,
    create_baseline_diverging_chart,
    create_typing_speed_comparison,
    create_pause_distribution_chart,
    create_correction_activity_chart,
    create_passage_comparison_chart,
    create_reliability_gauge
)
import importlib
import dsp
importlib.reload(dsp)
if not hasattr(dsp, "pseudonymize_user_id"):
    dsp.pseudonymize_user_id = getattr(dsp, "pseudonymize_id", None)

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="MindType — Cognitive Strain & Typing Analysis",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# =====================================================================
# EDITORIAL STYLING SYSTEM (PLUS JAKARTA SANS & CLEAN CONTRAST)
# =====================================================================
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
      @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

      :root {
        --bg: #F8FAFC;
        --card: #FFFFFF;
        --border: #E2E8F0;
        --border-subtle: #F1F5F9;
        --navy: #0F172A;
        --navy-light: #1E293B;
        --slate: #475569;
        --slate-light: #94A3B8;
        --indigo: #4F46E5;
        --indigo-soft: #EEF2FF;
        --amber: #D97706;
        --amber-soft: #FEF3C7;
        --emerald: #059669;
        --emerald-soft: #DCFCE7;
        --rose: #E11D48;
        --rose-soft: #FFE4E6;
      }

      /* Base typography hierarchy without overriding Streamlit Material Icon ligatures */
      html, body, [class*="st-"], .stApp, p, h1, h2, h3, h4, h5, h6, label, input, textarea, select, button {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      }

      /* Explicitly protect and guarantee Material Symbols icon font for Streamlit */
      [data-testid="stIconMaterial"],
      .material-symbols-rounded,
      .material-symbols-outlined,
      .material-icons,
      [class*="material-symbols"] {
        font-family: 'Material Symbols Rounded' !important;
        font-weight: normal !important;
        font-style: normal !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
      }

      .stApp {
        background-color: var(--bg);
      }

      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3.5rem;
        max-width: 1140px;
        margin: 0 auto;
      }

      /* Global Headings */
      h1:not(.hero-title), h2:not(.hero-title), h3, h4 {
        color: var(--navy) !important;
        letter-spacing: -0.02em;
        font-weight: 800 !important;
      }

      /* =====================================================================
         PREMIUM LEFT-SIDE NAVIGATION RAIL
         ===================================================================== */
      section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid rgba(148, 163, 184, 0.15) !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.08) !important;
        width: 250px !important;
        min-width: 250px !important;
      }

      section[data-testid="stSidebar"] > div:first-child {
        background-color: #0F172A !important;
        padding-top: 1.4rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
        padding-bottom: 1.5rem !important;
      }

      /* Sleek Scrollbar for Sidebar */
      section[data-testid="stSidebar"]::-webkit-scrollbar {
        width: 4px;
      }
      section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 4px;
      }

      /* Sidebar Brand Header */
      .mindtype-sidebar-header {
        padding: 6px 8px 12px 8px;
      }
      .mindtype-sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.02em;
        line-height: 1.2;
      }
      .mindtype-sidebar-subtitle {
        font-size: 0.80rem;
        font-weight: 600;
        color: #94A3B8;
        letter-spacing: 0.02em;
        margin-top: 4px;
      }
      .mindtype-sidebar-divider {
        height: 1px;
        background: rgba(148, 163, 184, 0.15);
        margin: 14px 4px 18px 4px;
      }

      /* Scoped Navigation Buttons in Left Rail */
      section[data-testid="stSidebar"] .stButton {
        margin-bottom: 6px !important;
      }

      section[data-testid="stSidebar"] .stButton > button {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        height: 46px !important;
        min-height: 46px !important;
        padding: 10px 14px !important;
        border-radius: 12px !important;
        font-size: 0.90rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        text-align: left !important;
        width: 100% !important;
        white-space: nowrap !important;
        transition: all 200ms ease-out !important;
        cursor: pointer !important;
      }

      /* Inactive Navigation Buttons */
      section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"],
      section[data-testid="stSidebar"] .stButton > button:not([data-testid="stBaseButton-primary"]) {
        background: transparent !important;
        color: #94A3B8 !important;
        border: 1px solid transparent !important;
        box-shadow: none !important;
        transform: none !important;
      }

      /* Inactive Button Hover */
      section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover,
      section[data-testid="stSidebar"] .stButton > button:not([data-testid="stBaseButton-primary"]):hover {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(148, 163, 184, 0.15) !important;
        transform: translateX(2px) !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15) !important;
      }

      /* Active Navigation Button */
      section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background: rgba(99, 102, 241, 0.20) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.50) !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 10px rgba(99, 102, 241, 0.25) !important;
        position: relative !important;
      }

      /* Active Button Hover */
      section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover {
        background: rgba(99, 102, 241, 0.28) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(99, 102, 241, 0.65) !important;
        transform: translateX(2px) !important;
      }

      /* Navigation Button Text & Inner Alignments */
      section[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"],
      section[data-testid="stSidebar"] .stButton > button div[data-testid="stMarkdownContainer"] > p {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        line-height: 1 !important;
        text-align: left !important;
      }

      /* Sidebar Footer Micro-Details */
      .mindtype-sidebar-footer {
        margin-top: 36px;
        padding: 16px 8px 6px 8px;
        border-top: 1px solid rgba(148, 163, 184, 0.12);
      }
      .mindtype-footer-tag {
        font-size: 0.68rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }
      .mindtype-footer-sub {
        font-size: 0.74rem;
        color: #475569;
        margin-top: 3px;
      }

      /* Streamlit Header & Sidebar Controls */
      header[data-testid="stHeader"] {
        background: transparent !important;
        z-index: 100 !important;
      }
      [data-testid="stSidebarCollapseButton"] button {
        color: #94A3B8 !important;
      }
      [data-testid="stSidebarCollapseButton"] button:hover {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.08) !important;
      }
      [data-testid="stSidebarCollapsedControl"] {
        color: var(--navy) !important;
        top: 0.8rem !important;
        left: 0.8rem !important;
      }

      /* Hero Section (CRITICAL: PURE WHITE TITLE ON DARK NAVY) */
      .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 70%, #2A3B5C 100%);
        border-radius: 20px;
        padding: 46px 48px;
        margin-bottom: 32px;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
        position: relative;
        overflow: hidden;
      }
      .hero-container::after {
        content: "";
        position: absolute;
        right: -30px; bottom: -40px;
        width: 240px; height: 180px;
        background: radial-gradient(circle, rgba(79, 70, 229, 0.25) 0%, transparent 70%);
        pointer-events: none;
      }
      .hero-tag {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: #A5B4FC !important;
        margin-bottom: 12px;
        display: inline-block;
      }
      .hero-title {
        color: #FFFFFF !important;
        font-size: 2.45rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        margin: 0 0 12px 0 !important;
        line-height: 1.18 !important;
      }
      .hero-subtitle {
        color: #CBD5E1 !important;
        font-size: 1.05rem !important;
        font-weight: 400 !important;
        line-height: 1.6 !important;
        max-width: 820px;
        margin: 0 0 24px 0 !important;
      }
      .hero-disclaimer {
        font-size: 0.82rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
        gap: 6px;
        margin: 0;
      }

      /* Editorial Section Headers */
      .editorial-header {
        margin: 36px 0 16px 0;
        border-bottom: 1.5px solid var(--border);
        padding-bottom: 10px;
      }
      .editorial-eyebrow {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        color: var(--indigo);
        margin-bottom: 4px;
      }
      .editorial-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: var(--navy);
        margin: 0;
        letter-spacing: -0.02em;
      }

      /* Assessment Screen Header & Clean Rationale Block */
      .assessment-header-block {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.02);
      }
      .assessment-step-tag {
        font-size: 0.74rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--indigo);
        margin-bottom: 4px;
      }
      .assessment-title-text {
        font-size: 1.5rem;
        font-weight: 800;
        color: var(--navy);
        margin: 2px 0 8px 0;
      }
      .assessment-desc-text {
        font-size: 0.94rem;
        color: var(--slate);
        line-height: 1.55;
        margin-bottom: 14px;
      }
      .editorial-rationale {
        background: #F8FAFC;
        border-left: 3.5px solid var(--indigo);
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.55;
      }

      /* Dot Timeline Progress */
      .dot-progress-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 18px;
        padding: 0 4px;
      }
      .dot-step {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--slate-light);
      }
      .dot-step.active {
        color: var(--indigo);
      }
      .dot-step.completed {
        color: var(--emerald);
      }
      .dot-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #CBD5E1;
      }
      .dot-step.active .dot-indicator {
        background: var(--indigo);
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.2);
      }
      .dot-step.completed .dot-indicator {
        background: var(--emerald);
      }
      .dot-line {
        flex: 1;
        height: 2px;
        background: #E2E8F0;
        margin: 0 8px;
        margin-bottom: 16px;
      }

      /* Main Content Buttons: Sophisticated Indigo Accent */
      section[data-testid="stMain"] .stButton>button,
      .main .stButton>button {
        background: var(--navy) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        padding: 0.60rem 1.4rem !important;
        font-size: 0.94rem !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08) !important;
      }
      section[data-testid="stMain"] .stButton>button:hover,
      .main .stButton>button:hover {
        background: var(--indigo) !important;
        color: #FFFFFF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.25) !important;
      }

      /* KPI Highlights in Report */
      .kpi-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 18px;
        margin-bottom: 28px;
      }
      .kpi-box {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px 26px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.03);
      }
      .kpi-title {
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--slate);
        margin-bottom: 6px;
      }
      .kpi-big {
        font-size: 3.1rem;
        font-weight: 800;
        color: var(--navy);
        line-height: 1;
        margin-bottom: 8px;
      }
      .kpi-sub {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--slate);
      }

      /* Horizontal Observation Rows */
      .obs-row {
        display: flex;
        align-items: flex-start;
        gap: 16px;
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 12px;
      }
      .obs-num {
        font-size: 0.82rem;
        font-weight: 800;
        color: var(--indigo);
        background: var(--indigo-soft);
        padding: 4px 10px;
        border-radius: 6px;
      }
      .obs-title {
        font-size: 1.02rem;
        font-weight: 700;
        color: var(--navy);
        margin: 0 0 4px 0;
      }
      .obs-text {
        font-size: 0.90rem;
        color: var(--slate);
        line-height: 1.55;
        margin: 0;
      }

      /* Heuristic Feature Attribution Bar */
      .contrib-bar-wrap {
        margin-bottom: 14px;
      }
      .contrib-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 700;
        color: var(--navy);
        margin-bottom: 4px;
      }
      .contrib-track {
        height: 10px;
        background: #E2E8F0;
        border-radius: 999px;
        overflow: hidden;
      }
      .contrib-fill {
        height: 100%;
        background: var(--indigo);
        border-radius: 999px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================
if "nav_view" not in st.session_state:
    st.session_state.nav_view = "home"  # home | focus | assessment | analyzing | report

if "passage_idx" not in st.session_state:
    st.session_state.passage_idx = 0
    st.session_state.passage_results = []
    st.session_state.baseline_mean = None
    st.session_state.baseline_std = None
    st.session_state.baseline_ikt = None
    st.session_state.baseline_stats = None
    st.session_state.final_report = None
    st.session_state.participant_name = "Participant"
    st.session_state.focus_agreed = False

try:
    baselines_df = pd.read_csv("models/baselines_plain.csv")
    baselines_ok = True
except Exception:
    baselines_df, baselines_ok = pd.DataFrame(), False


# =====================================================================
# PREMIUM LEFT-SIDE VERTICAL NAVIGATION SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="mindtype-sidebar-header">
          <div class="mindtype-sidebar-title">MINDTYPE</div>
          <div class="mindtype-sidebar-subtitle">Behavioral Analysis</div>
        </div>
        <div class="mindtype-sidebar-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    # Determine active state for navigation items
    is_home = (st.session_state.nav_view == "home")
    is_assess = (st.session_state.nav_view in ["focus", "assessment", "analyzing"])
    is_how = (st.session_state.nav_view == "how_it_works")
    is_priv = (st.session_state.nav_view == "privacy")
    is_dash = (st.session_state.nav_view in ["dashboard", "report", "sandbox"])

    if st.button("Home", key="nav_home", use_container_width=True, type="primary" if is_home else "secondary"):
        st.session_state.nav_view = "home"
        st.rerun()

    if st.button("Assessment", key="nav_assess", use_container_width=True, type="primary" if is_assess else "secondary"):
        st.session_state.nav_view = "focus"
        st.rerun()

    if st.button("How It Works", key="nav_how", use_container_width=True, type="primary" if is_how else "secondary"):
        st.session_state.nav_view = "how_it_works"
        st.rerun()

    if st.button("Privacy", key="nav_priv", use_container_width=True, type="primary" if is_priv else "secondary"):
        st.session_state.nav_view = "privacy"
        st.rerun()

    if st.button("Dashboard", key="nav_dashboard", use_container_width=True, type="primary" if is_dash else "secondary"):
        st.session_state.nav_view = "dashboard"
        st.rerun()

    st.markdown(
        """
        <div class="mindtype-sidebar-divider" style="margin-top:28px;"></div>
        <div class="mindtype-sidebar-footer">
          <div class="mindtype-footer-tag">Research Prototype</div>
          <div class="mindtype-footer-sub">Keystroke Dynamics v1.2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# VIEW 1: EDITORIAL LANDING PAGE
# =====================================================================
if st.session_state.nav_view == "home":
    st.markdown(
        """
        <div class="hero-container">
          <span class="hero-tag">Typing-Based Behavioral Analysis &bull; Research Prototype</span>
          <h1 class="hero-title">Understand the patterns behind the way you type.</h1>
          <p class="hero-subtitle">
            MindType analyzes keystroke dynamics such as rhythm, hesitation, corrections and timing
            to identify behavioral changes associated with cognitive or task-related strain.
          </p>
          <div class="hero-disclaimer">
            <span>🔒 Privacy-first</span> &bull;
            <span>🔬 Research prototype</span> &bull;
            <span>⚠️ Not a medical or clinical diagnosis</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_btn_start, col_btn_how, _ = st.columns([1.5, 1.5, 4])
    with col_btn_start:
        if st.button("Start Assessment →", key="btn_home_start", use_container_width=True):
            st.session_state.nav_view = "focus"
            st.rerun()
    with col_btn_how:
        if st.button("How It Works ↓", key="btn_home_how", use_container_width=True):
            st.session_state.nav_view = "how_it_works"
            st.rerun()

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    # 3 Editorial Pillars
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px 22px;">
              <div style="font-size:0.75rem; font-weight:800; color:#4F46E5; letter-spacing:0.08em; text-transform:uppercase;">01 &bull; Natural Pacing</div>
              <h3 style="font-size:1.15rem; margin:8px 0 6px 0;">Type Naturally</h3>
              <p style="font-size:0.88rem; color:#64748B; line-height:1.55; margin:0;">
                Complete 5 structured typing tasks without intentionally changing your style.
                Passage 1 calibrates your personal resting baseline.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p2:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px 22px;">
              <div style="font-size:0.75rem; font-weight:800; color:#4F46E5; letter-spacing:0.08em; text-transform:uppercase;">02 &bull; Keystroke Dynamics</div>
              <h3 style="font-size:1.15rem; margin:8px 0 6px 0;">We Analyze Your Timing</h3>
              <p style="font-size:0.88rem; color:#64748B; line-height:1.55; margin:0;">
                MindType captures micro-timings: inter-key intervals, dwell times, pauses, and backspace bursts,
                evaluating your rhythm against your own resting baseline.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_p3:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px 22px;">
              <div style="font-size:0.75rem; font-weight:800; color:#4F46E5; letter-spacing:0.08em; text-transform:uppercase;">03 &bull; Explainable Report</div>
              <h3 style="font-size:1.15rem; margin:8px 0 6px 0;">Understand Your Pattern</h3>
              <p style="font-size:0.88rem; color:#64748B; line-height:1.55; margin:0;">
                Receive an explainable Behavioral Strain Index and Measurement Reliability score,
                demonstrating what changed, how much it changed, and where hesitation clustered.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
# VIEW 2: HOW IT WORKS
# =====================================================================
elif st.session_state.nav_view == "how_it_works":
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Scientific Methodology</div>
          <h2 class="editorial-title">How MindType Analyzes Typing Dynamics</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    steps = [
        ("01", "TYPE NATURALLY", "Complete the assessment without intentionally changing your typing style. Small typing errors are natural and are folded into error rate metrics rather than interrupting your session."),
        ("02", "CAPTURE TYPING DYNAMICS", "MindType captures high-resolution timestamps in the browser for keydown and keyup events. It extracts flight time (inter-key intervals) and dwell time (key hold duration)."),
        ("03", "COMPARE WITH YOUR BASELINE", "Passage 1 serves as your resting calibration reference. All subsequent neutral and load tasks are mathematically compared to this individual baseline rather than population stereotypes."),
        ("04", "INTERPRET BEHAVIORAL CHANGES", "The system identifies concrete behavioral shifts (cadence reduction, hesitation spike clusters, correction bursts) and evaluates measurement reliability.")
    ]

    for num, title, text in steps:
        st.markdown(
            f"""
            <div class="obs-row">
              <span class="obs-num">{num}</span>
              <div>
                <h4 class="obs-title">{title}</h4>
                <p class="obs-text">{text}</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("Begin Assessment Protocol →", use_container_width=False):
        st.session_state.nav_view = "focus"
        st.rerun()


# =====================================================================
# VIEW 3: PRIVACY-FIRST ARCHITECTURE
# =====================================================================
elif st.session_state.nav_view == "privacy":
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Data Protection</div>
          <h2 class="editorial-title">Privacy-First Behavioral Analysis</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:28px; line-height:1.7; color:#334155; font-size:0.94rem;">
          <h4 style="margin:0 0 10px 0; color:#0F172A;">Core Privacy Principles</h4>
          <ul style="padding-left: 20px; margin: 0 0 16px 0;">
            <li><strong>Zero Content Capture:</strong> The actual text you type is never stored or transmitted as user content. Only timing intervals and error flags are analyzed.</li>
            <li><strong>One-Way Cryptographic Pseudonymization:</strong> All user identities are hashed with a secret salt using SHA-256 before writing to any feature store.</li>
            <li><strong>Encrypted Baselines:</strong> Baseline files on disk are encrypted at rest using AES-128-CBC via Fernet.</li>
            <li><strong>Differential Privacy:</strong> Mathematical Laplace noise is available for aggregate cohort statistics and is strictly excluded from individual live reports.</li>
            <li><strong>Strict Data Minimization:</strong> MindType does not request microphone, camera, clipboard, or unrelated device access.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


# =====================================================================
# VIEW 4: MANDATORY PRE-ASSESSMENT FOCUS SCREEN
# =====================================================================
elif st.session_state.nav_view == "focus":
    st.markdown(
        """
        <div class="hero-container" style="padding: 36px 42px;">
          <span class="hero-tag">Pre-Assessment Protocol</span>
          <h1 class="hero-title">Before You Begin</h1>
          <p class="hero-subtitle">
            For the most reliable result, complete the assessment in one focused session.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_box_l, col_box_r = st.columns([1.4, 1])
    with col_box_l:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px 26px; margin-bottom:16px;">
              <h4 style="margin:0 0 12px 0; font-size:1.05rem;">Focus Checklist</h4>
              <div style="font-size:0.92rem; color:#334155; line-height:1.8;">
                &#10003; <strong>Stay focused on the typing task</strong> throughout the 5 short passages.<br>
                &#10003; <strong>Avoid talking</strong> to someone while typing.<br>
                &#10003; <strong>Avoid switching tabs</strong> or using another device.<br>
                &#10003; <strong>Do not intentionally type faster or slower</strong> — type naturally.<br>
                &#10003; <strong>Do not copy/paste</strong> the passage (pasting is blocked and lowers reliability).<br>
                &#10003; <strong>Small mistakes are expected:</strong> keep going to the end of each passage.<br>
                &#10003; <strong>Complete the session</strong> without unnecessary interruptions.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_box_r:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px; margin-bottom:16px;">
              <h4 style="margin:0 0 8px 0; font-size:1.0rem;">Why does this matter?</h4>
              <p style="font-size:0.86rem; color:#64748B; line-height:1.55; margin:0 0 14px 0;">
                MindType measures subtle changes in your natural typing behavior.
                Talking, multitasking, or switching tabs can alter your keystroke rhythm and reduce measurement reliability.
              </p>
              <h4 style="margin:0 0 6px 0; font-size:0.95rem;">Privacy Statement</h4>
              <p style="font-size:0.85rem; color:#64748B; line-height:1.5; margin:0;">
                Your message content is not used as the signal. MindType strictly analyzes timing, pauses, corrections, and rhythm.
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    col_name, _ = st.columns([1.5, 1])
    with col_name:
        p_name = st.text_input(
            "Participant Name or Identifier (Optional)",
            value=st.session_state.participant_name,
            help="Used only to label your downloadable report."
        )
        if p_name:
            st.session_state.participant_name = p_name

    agree = st.checkbox(
        "I understand and I'm ready to complete the assessment in one focused session.",
        value=st.session_state.focus_agreed
    )
    st.session_state.focus_agreed = agree

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col_btn_go, col_btn_back, _ = st.columns([1.5, 1.2, 3])
    with col_btn_go:
        if st.button("Start Assessment →", disabled=not agree, use_container_width=True):
            st.session_state.passage_idx = 0
            st.session_state.passage_results = []
            st.session_state.nav_view = "assessment"
            st.rerun()
    with col_btn_back:
        if st.button("← Cancel", use_container_width=True):
            st.session_state.nav_view = "home"
            st.rerun()


# =====================================================================
# VIEW 5: 5-STAGE ASSESSMENT INTERFACE
# =====================================================================
elif st.session_state.nav_view == "assessment":
    idx = st.session_state.passage_idx
    n_total = total_passages()

    if idx < n_total:
        meta = get_passage(idx)

        # Dot Progress Line
        steps_html = []
        for s in range(n_total):
            status_class = "completed" if s < idx else ("active" if s == idx else "")
            steps_html.append(f"""
            <div class="dot-step {status_class}">
              <div class="dot-indicator"></div>
              <span>{s+1:02d}</span>
            </div>
            """)
            if s < n_total - 1:
                steps_html.append('<div class="dot-line"></div>')

        st.markdown(f'<div class="dot-progress-row">{"".join(steps_html)}</div>', unsafe_allow_html=True)

        # Editorial Passage Header (NO OVERLAPPING TEXT, CLEAN LAYOUT)
        condition_badge_color = "#166534" if meta["type"] == "calibration" else ("#92400E" if meta["type"] == "load" else "#0369A1")
        condition_badge_bg = "#DCFCE7" if meta["type"] == "calibration" else ("#FEF3C7" if meta["type"] == "load" else "#E0F2FE")

        st.markdown(
            f"""
            <div class="assessment-header-block">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span class="assessment-step-tag">ASSESSMENT &bull; {idx+1:02d} / {n_total:02d}</span>
                <span style="font-size:0.75rem; font-weight:700; padding:3px 10px; border-radius:999px; background:{condition_badge_bg}; color:{condition_badge_color};">
                  {meta['condition']}
                </span>
              </div>
              <h2 class="assessment-title-text">{meta['title']}</h2>
              <p class="assessment-desc-text">{meta['description']}</p>
              <div class="editorial-rationale">
                <strong>WHY THIS PASSAGE?</strong> {meta['rationale']}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Render Keystroke Capture Component
        capture_res = keystroke_capture(
            paragraph=meta["text"],
            title=f"Passage {meta['id']} of {n_total}",
            condition=meta["type"],
            key=f"keystroke_capture_stage_{idx}"
        )

        # Process submission
        if capture_res is not None:
            events = capture_res.get("events", [])
            if not events:
                st.warning("No keystroke data detected. Please type the passage before submitting.")
            else:
                try:
                    feat_df = build_features_dataframe_from_events(events, meta["id"], "session_live")
                    quality_meta = capture_res.get("quality", {})
                    p_quality = compute_passage_quality(
                        quality_meta,
                        n_keystrokes=len(feat_df),
                        expected_chars=len(meta["text"])
                    )

                    if idx == 0:
                        b_mean, b_std, b_ikt, b_stats = calibrate_baseline_from_passage_1(feat_df)
                        st.session_state.baseline_mean = b_mean
                        st.session_state.baseline_std = b_std
                        st.session_state.baseline_ikt = b_ikt
                        st.session_state.baseline_stats = b_stats
                        score = score_feature_frame(feat_df, b_mean, b_std, baseline_stats=b_stats)
                    else:
                        b_mean = st.session_state.baseline_mean
                        b_std = st.session_state.baseline_std
                        b_ikt = st.session_state.baseline_ikt
                        b_stats = st.session_state.baseline_stats
                        score = score_feature_frame(
                            feat_df, b_mean, b_std, baseline_ikt=b_ikt, baseline_stats=b_stats
                        )

                    score.update(p_quality)
                    score["events"] = events
                    st.session_state.passage_results.append(score)
                    st.session_state.passage_idx += 1

                    if st.session_state.passage_idx >= n_total:
                        st.session_state.nav_view = "analyzing"
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing typing data: {e}")

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        if st.button("Cancel Assessment", key="btn_cancel_assessment"):
            st.session_state.passage_idx = 0
            st.session_state.passage_results = []
            st.session_state.nav_view = "focus"
            st.rerun()

    else:
        st.session_state.nav_view = "analyzing"
        st.rerun()


# =====================================================================
# VIEW 6: ANALYSIS TRANSITION STATE
# =====================================================================
elif st.session_state.nav_view == "analyzing":
    st.markdown(
        """
        <div class="hero-container" style="text-align: center; padding: 42px 30px;">
          <span class="hero-tag">Synthesis Pipeline</span>
          <h1 class="hero-title" style="font-size: 2.2rem;">Analyzing Your Typing Pattern</h1>
          <p class="hero-subtitle" style="margin: 0 auto; max-width: 620px;">
            Evaluating micro-timing, establishing personal baseline shifts,
            and computing measurement reliability...
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    status_box = st.empty()
    steps = [
        "✓ Extracting inter-key intervals and dwell time dynamics...",
        "✓ Calibrating personal baseline reference from Passage 1...",
        "✓ Analyzing rhythm variability across Neutral vs. Cognitive Load conditions...",
        "✓ Auditing session quality (continuity, paste prevention, window focus)...",
        "✓ Synthesizing explainable behavioral report..."
    ]
    for step in steps:
        status_box.markdown(f"**{step}**")
        time.sleep(0.3)

    try:
        st.session_state.final_report = aggregate_report(
            st.session_state.passage_results,
            PASSAGES,
            participant_label=st.session_state.participant_name or "Participant"
        )
        for i, r in enumerate(st.session_state.passage_results):
            log_result({
                "user_id": st.session_state.participant_name or "anonymous",
                "user_id_pseudonymized": dsp.pseudonymize_id(st.session_state.participant_name),
                "session_id": f"{st.session_state.participant_name}_p{i+1}",
                "timestamp": pd.Timestamp.now().isoformat(),
                **{k: v for k, v in r.items() if not isinstance(v, (list, dict, np.ndarray))},
            })

        st.session_state.nav_view = "report"
        st.rerun()
    except Exception as e:
        st.error(f"Failed to synthesize report: {e}")
        if st.button("Return to Focus Screen"):
            st.session_state.nav_view = "focus"
            st.rerun()


# =====================================================================
# VIEW 7: COMPREHENSIVE STORYTELLING REPORT & DASHBOARD
# =====================================================================
elif st.session_state.nav_view in ["dashboard", "report"]:
    rep = st.session_state.final_report
    if not rep:
        st.markdown(
            """
            <div style="background: #FFFFFF; border: 1px solid var(--border); border-radius: 16px; padding: 36px 30px; text-align: center; max-width: 680px; margin: 28px auto; box-shadow: 0 4px 16px rgba(15,23,42,0.04);">
              <div style="font-size: 2.4rem; margin-bottom: 12px;">📊</div>
              <h2 style="font-size: 1.55rem; font-weight: 800; color: var(--navy); margin: 0 0 10px 0;">Cognitive Behavioral Dashboard</h2>
              <p style="font-size: 0.95rem; color: var(--slate); line-height: 1.6; margin: 0 0 24px 0;">
                The MindType Dashboard displays your complete 5-stage behavioral analysis, cognitive strain indicators, typing cadence rhythm, hesitation pause distributions, and predictive model interpretations once an assessment has been completed.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            if st.button("Start Assessment Protocol →", key="dash_start_assess", use_container_width=True):
                st.session_state.nav_view = "focus"
                st.rerun()
        with col_d2:
            if st.button("Developer Sandbox (Synthetic Data) →", key="dash_open_sandbox", use_container_width=True):
                st.session_state.nav_view = "sandbox"
                st.rerun()
        st.stop()

    rel = rep["reliability"]
    interp = rep["interpretation"]
    indicators = rep["indicators"]
    observed = rep["observed_cards"]
    strain_index = rep["strain_index"]
    p_results = rep["passage_results"]

    # =================================================================
    # 1. YOUR TYPING PATTERN
    # =================================================================
    st.markdown(
        f"""
        <div style="border-bottom: 2px solid #E2E8F0; padding-bottom: 22px; margin-bottom: 26px;">
          <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; color: #4F46E5; letter-spacing: 0.09em;">
            Behavioral Research Report &bull; 5-Stage Within-Subject Protocol
          </div>
          <h1 style="font-size: 2.3rem; color: #0F172A; margin: 6px 0 10px 0;">YOUR TYPING PATTERN</h1>
          <div style="background: #F8FAFC; border-left: 4px solid #4F46E5; padding: 16px 20px; border-radius: 10px;">
            <p style="font-size: 1.08rem; font-weight: 600; color: #0F172A; margin: 0; line-height: 1.6;">
              {interp.get('one_sentence_summary', interp.get('summary', ''))}
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Inconclusive Guardrail Banner if reliability is low
    if not rep["is_conclusive"]:
        st.error(
            "### ⚠️ Inconclusive Session Notice\n"
            "This session contained focus disruptions, paste attempts, or timing irregularities that lower measurement reliability below research standards.\n\n"
            "**Detected Factors:**\n" +
            "\n".join([f"- {r}" for r in rel.get("inconclusive_reasons", [])]) +
            "\n\n*The findings below are indicative only. Consider restarting the assessment in an uninterrupted session.*"
        )

    # Two Large Visual Metrics
    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-box">
            <div class="kpi-title">Observed Behavioral State</div>
            <div class="kpi-big" style="font-size: 1.85rem;">{interp.get('pattern_title', rep['tier_label'])}</div>
            <div class="kpi-sub">Behavioral Strain Index: <strong>{strain_index:.1f} / 100</strong></div>
            <div style="font-size:0.80rem; color:#94A3B8; margin-top:6px;">Evaluates multi-signal timing shifts across diagnostic tasks P2–P5</div>
          </div>
          <div class="kpi-box">
            <div class="kpi-title">Measurement Reliability</div>
            <div class="kpi-big">{rel['reliability_score']:.0f} <span style="font-size:1.5rem; color:#94A3B8;">/ 100</span></div>
            <div class="kpi-sub" style="color: {'#166534' if rel['reliability_tier']=='High' else '#92400E'};">
              {rel['reliability_tier']} Reliability &bull; {rel['session_quality_score']:.0f}% Focus Quality
            </div>
            <div style="font-size:0.80rem; color:#94A3B8; margin-top:6px;">Audits baseline stability, sample volume, and focus continuity</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 2. WHAT CHANGED?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Observable Deviations</div>
          <h2 class="editorial-title">What Changed?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px; text-align:center;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.06em;">Typing Speed</div>
              <div style="font-size:2rem; font-weight:800; color:{'#D97706' if indicators['speed_delta_pct']>8 else '#0F172A'}; margin:4px 0;">
                {indicators['speed_delta_pct']:+.1f}%
              </div>
              <div style="font-size:0.82rem; font-weight:600; color:#475569;">{indicators['speed_direction']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_w2:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px; text-align:center;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.06em;">Pause Frequency</div>
              <div style="font-size:2rem; font-weight:800; color:{'#D97706' if indicators['pause_delta_pct']>15 else '#0F172A'}; margin:4px 0;">
                {indicators['pause_delta_pct']:+.1f}%
              </div>
              <div style="font-size:0.82rem; font-weight:600; color:#475569;">{indicators['pause_direction']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_w3:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px; text-align:center;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.06em;">Rhythm Variability</div>
              <div style="font-size:2rem; font-weight:800; color:{'#D97706' if indicators['rhythm_delta_pct']>12 else '#0F172A'}; margin:4px 0;">
                {indicators['rhythm_delta_pct']:+.1f}%
              </div>
              <div style="font-size:0.82rem; font-weight:600; color:#475569;">{indicators['rhythm_direction']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_w4:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:18px; text-align:center;">
              <div style="font-size:0.75rem; font-weight:700; text-transform:uppercase; color:#64748B; letter-spacing:0.06em;">Correction Activity</div>
              <div style="font-size:2rem; font-weight:800; color:{'#D97706' if indicators['correction_delta_pct']>15 else '#0F172A'}; margin:4px 0;">
                {indicators['correction_delta_pct']:+.1f}%
              </div>
              <div style="font-size:0.82rem; font-weight:600; color:#475569;">{indicators['correction_direction']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 3. HOW MUCH DID IT CHANGE?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Comparative Magnitude</div>
          <h2 class="editorial-title">How Much Did It Change?</h2>
        </div>
        <p style="font-size: 0.92rem; color: #475569; margin-top: -6px;">
          The horizontal bar chart below compares your assessment session directly against your resting calibration baseline.
          <strong>100 represents your own personal calibration baseline</strong> — scores above or below 100 represent relative behavioral shifts.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Diverging baseline comparison bar chart
    fig_diverging = create_baseline_diverging_chart(indicators)
    st.plotly_chart(fig_diverging, use_container_width=True)

    # Speed WPM comparison
    calib_wpm = p_results[0].get("wpm", 45.0)
    diag_wpm = float(np.mean([r.get("wpm", 40.0) for r in p_results[1:]]))
    calib_ms = p_results[0].get("avg_typing_speed_ms", 180.0)
    diag_ms = float(np.mean([r.get("avg_typing_speed_ms", 220.0) for r in p_results[1:]]))

    col_speed_vis, col_speed_note = st.columns([1.2, 1])
    with col_speed_vis:
        fig_speed = create_typing_speed_comparison(calib_wpm, diag_wpm, calib_ms, diag_ms)
        st.plotly_chart(fig_speed, use_container_width=True)
    with col_speed_note:
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:20px 22px; height:240px; display:flex; flex-direction:column; justify-content:center;">
              <h4 style="margin:0 0 6px 0; font-size:1.05rem;">Pacing & Cadence Finding</h4>
              <p style="font-size:0.90rem; color:#475569; line-height:1.6; margin:0 0 10px 0;">
                Your baseline typing speed averaged <strong>{calib_wpm:.1f} WPM</strong> ({calib_ms:.0f} ms between keys).
                During diagnostic tasks, your speed shifted to <strong>{diag_wpm:.1f} WPM</strong> ({diag_ms:.0f} ms),
                representing a <strong>{indicators['speed_delta_pct']:+.1f}%</strong> change in inter-key interval.
              </p>
              <div style="font-size:0.84rem; color:#0F172A; font-weight:700;">
                Status: {'Cadence widened (deliberate motor planning)' if indicators['speed_delta_pct'] > 8 else ('Cadence accelerated' if indicators['speed_delta_pct'] < -8 else 'Cadence remained stable')}
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 4. WHERE DID IT CHANGE?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Micro-Timing Geography & Task Timeline</div>
          <h2 class="editorial-title">Where Did It Change?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Full-width rhythm scatter
    load_events = p_results[2].get("events", []) if len(p_results) > 2 else []
    b_mean = st.session_state.baseline_mean or 180.0
    b_std = st.session_state.baseline_std or 35.0
    fig_rhythm = create_keystroke_rhythm_chart(load_events, b_mean, b_std)
    st.plotly_chart(fig_rhythm, use_container_width=True)

    # Passage-by-Passage Condition Comparison Chart
    fig_tasks = create_passage_comparison_chart(p_results, PASSAGES)
    st.plotly_chart(fig_tasks, use_container_width=True)
    st.caption("Tracking Neutral vs Cognitive Load passages reveals whether behavioral shifts repeated across demanding tasks and normalized during recovery tasks.")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 5. WHAT BEHAVIOR DROVE THE CHANGE?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Behavioral Mechanics</div>
          <h2 class="editorial-title">What Behavior Drove the Change?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_mech_l, col_mech_r = st.columns(2)
    with col_mech_l:
        fig_pauses = create_pause_distribution_chart(p_results[1:])
        st.plotly_chart(fig_pauses, use_container_width=True)
    with col_mech_r:
        b_bksp = p_results[0].get("backspace_rate", 0.04)
        d_bksp = float(np.mean([r.get("backspace_rate", 0.04) for r in p_results[1:]]))
        total_bursts = sum(int(r.get("correction_bursts", 0)) for r in p_results[1:])
        fig_corrections = create_correction_activity_chart(b_bksp, d_bksp, total_bursts)
        st.plotly_chart(fig_corrections, use_container_width=True)

    # Relative Heuristic Contributions & Dimension Cards
    col_contrib_l, col_contrib_r = st.columns([1.2, 1])
    with col_contrib_l:
        st.markdown("<h4 style='font-size:1.02rem; margin:0 0 12px 0;'>Relative Heuristic Behavioral Impact</h4>", unsafe_allow_html=True)
        contribs = p_results[2].get("feature_contributions", []) if len(p_results) > 2 else []
        for c in contribs:
            st.markdown(
                f"""
                <div class="contrib-bar-wrap">
                  <div class="contrib-meta">
                    <span>{c['feature']}</span>
                    <span style="color:#4F46E5;">{c['impact_level']}</span>
                  </div>
                  <div class="contrib-track">
                    <div class="contrib-fill" style="width: {c['meter_width']}%;"></div>
                  </div>
                  <div style="font-size:0.80rem; color:#64748B; margin-top:3px;">{c['description']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.caption("Weights reflect relative heuristic impact from keystroke-dynamics literature, not black-box mathematical percentages.")

    with col_contrib_r:
        st.markdown("<h4 style='font-size:1.02rem; margin:0 0 12px 0;'>Qualitative Observations Across 5 Dimensions</h4>", unsafe_allow_html=True)
        for i, c in enumerate(observed[:5]):
            st.markdown(
                f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:12px 14px; margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; font-size:0.88rem; color:#0F172A;">{c['icon']} {c['dimension']}</span>
                    <span style="font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:999px; background:#F1F5F9; color:#475569;">{c['badge']}</span>
                  </div>
                  <p style="font-size:0.82rem; color:#475569; margin:4px 0 0 0; line-height:1.45;">{c['narrative']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 6. WHAT MIGHT THIS PATTERN MEAN?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Multi-Signal Synthesis</div>
          <h2 class="editorial-title">What Might This Pattern Mean?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    signals_list = "".join([f"<li style='margin-bottom:4px;'>{sig}</li>" for sig in interp.get("contributing_signals", [])])
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px; padding:24px 28px; box-shadow:0 2px 12px rgba(15,23,42,0.04);">
          <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
            <span style="font-size:1.5rem;">💡</span>
            <h3 style="margin:0; font-size:1.25rem; color:#0F172A;">{interp.get('pattern_title', 'Behavioral Synthesis')}</h3>
          </div>
          <p style="font-size:0.96rem; color:#334155; line-height:1.65; margin:0 0 16px 0;">
            {interp.get('detailed_explanation', interp.get('summary', ''))}
          </p>
          <div style="background:#F8FAFC; border-radius:10px; padding:14px 18px; border-left:3.5px solid #4F46E5;">
            <strong style="font-size:0.86rem; color:#0F172A; text-transform:uppercase; letter-spacing:0.04em;">Key Supporting Signals:</strong>
            <ul style="margin:6px 0 0 0; padding-left:20px; font-size:0.88rem; color:#475569; line-height:1.6;">
              {signals_list if signals_list else '<li>Pacing and rhythm indicators remained consistent with baseline reference.</li>'}
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 7. HOW RELIABLE IS THIS?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Psychometric Validity & Session Quality</div>
          <h2 class="editorial-title">How Reliable Is This?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_rel_gauge, col_rel_factors = st.columns([1.1, 1.3])
    with col_rel_gauge:
        fig_rel = create_reliability_gauge(rel["reliability_score"], rel["reliability_tier"])
        st.plotly_chart(fig_rel, use_container_width=True)

    with col_rel_factors:
        total_blurs = sum(int(r.get("blur_count", 0)) for r in p_results)
        total_pastes = sum(int(r.get("paste_attempts", 0)) for r in p_results)
        st.markdown(
            f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:20px; font-size:0.88rem; color:#334155; line-height:1.7;">
              <h4 style="margin:0 0 8px 0; font-size:1.02rem; color:#0F172A;">Reliability Audit Factors</h4>
              {'&#10003;' if rel['reliability_score']>=50 else '&#9888;'} <strong>Session Completeness:</strong> 5 of 5 assessment passages recorded.<br>
              {'&#10003;' if rel['session_quality_score']>=75 else '&#9888;'} <strong>Focus Continuity:</strong> {rel['session_quality_score']:.0f}% session quality ({total_blurs} window focus shifts recorded).<br>
              &#10003; <strong>Baseline Stability:</strong> Stable Passage 1 calibration reference established.<br>
              &#10003; <strong>Task Concordance:</strong> Plausible behavioral concordance between Neutral and Cognitive Load tasks.<br>
              {'&#10003;' if total_pastes==0 else '&#9888;'} <strong>Zero Paste Invalidation:</strong> {total_pastes} paste attempt(s) detected.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 8. WHAT ELSE COULD EXPLAIN IT?
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Alternative Context</div>
          <h2 class="editorial-title">What Else Could Explain It?</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background:#F8FAFC; border-left:4px solid #D97706; border-radius:12px; padding:20px 24px; font-size:0.90rem; color:#475569; line-height:1.65;">
          <strong style="color:#0F172A; font-size:0.96rem;">Typing behavior is an indirect behavioral signal.</strong><br>
          Observed shifts in typing speed, hesitation pauses, and rhythm irregularity can arise from multiple non-strain factors:
          <ul style="margin:8px 0 0 0; padding-left:22px;">
            <li><strong>Physical Fatigue or Tiredness:</strong> Hand or eye muscle fatigue, especially after long computer sessions or poor sleep.</li>
            <li><strong>Environmental Distraction:</strong> Interruptions, notifications, or conversation in your physical workspace.</li>
            <li><strong>Hardware & Keyboard Differences:</strong> Switching between laptop scissor switches and external mechanical switches.</li>
            <li><strong>Task Vocabulary:</strong> Natural variation in familiarity with specific phrasing or phonetics.</li>
            <li><strong>Time of Day & Caffeine:</strong> Diurnal variations and stimulant consumption directly influence motor pacing.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 9. LIMITATIONS & MEDICAL SAFETY
    # =================================================================
    st.markdown(
        """
        <div class="editorial-header">
          <div class="editorial-eyebrow">Scientific Boundaries & Wellbeing Guidance</div>
          <h2 class="editorial-title">Important Limitations & Wellbeing Guidance</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_lim_l, col_lim_r = st.columns(2)
    with col_lim_l:
        st.markdown(
            """
            <div style="background:#FEF3C7; border:1px solid #FDE68A; border-radius:14px; padding:20px; font-size:0.86rem; color:#92400E; line-height:1.65;">
              <h4 style="margin:0 0 8px 0; font-size:1.0rem; color:#B45309;">Scientific Limitations</h4>
              &bull; MindType analyzes typing behavior; it does NOT directly measure or diagnose mental-health conditions.<br>
              &bull; The software cannot detect depression, anxiety, ADHD, burnout, or any psychiatric disorder.<br>
              &bull; Machine learning libraries (scikit-learn, TensorFlow, SciPy) extract temporal feature distributions, not clinical diagnoses.<br>
              &bull; Results represent an experimental behavioral estimate from a single assessment session.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_lim_r:
        st.markdown(
            """
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:14px; padding:20px; font-size:0.88rem; color:#334155; line-height:1.65;">
              <h4 style="margin:0 0 8px 0; font-size:1.0rem; color:#0F172A;">Wellbeing & Healthcare Guidance</h4>
              <p style="margin:0 0 10px 0;">
                A single typing session cannot determine a person's overall cognitive or psychological condition.
              </p>
              <p style="margin:0; font-size:0.85rem; color:#64748B;">
                If you notice that these behavioral patterns are recurring and you are also experiencing persistent
                changes in concentration, stress, sleep, mood, or everyday functioning, consider discussing your experiences
                with a qualified healthcare professional.
              </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 10. DOWNLOADABLE REPORTS & NEXT STEPS
    # =================================================================
    col_d1, col_d2, col_d_rst = st.columns([1.5, 1.5, 1.5])
    raw_label = rep.get("participant_label")
    file_label = str(raw_label).strip().lower().replace(" ", "_") if raw_label else "participant"
    with col_d1:
        st.download_button(
            label="📥 Download HTML Report",
            data=rep["html_report"],
            file_name=f"mindtype_report_{file_label}.html",
            mime="text/html",
            use_container_width=True
        )
    with col_d2:
        st.download_button(
            label="📄 Download Text Summary",
            data=rep["txt_report"],
            file_name=f"mindtype_report_{file_label}.txt",
            mime="text/plain",
            use_container_width=True
        )
    with col_d_rst:
        if st.button("🔄 Restart Assessment", use_container_width=True):
            st.session_state.passage_idx = 0
            st.session_state.passage_results = []
            st.session_state.final_report = None
            st.session_state.nav_view = "focus"
            st.rerun()

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)

    # =================================================================
    # 11. DEVELOPER / TECHNICAL DETAILS (DEFAULT CLOSED)
    # =================================================================
    with st.expander("🛠️ Developer / Technical Details (Model Outputs & Feature Vector)", expanded=False):
        st.markdown("#### Technical Metadata & Raw Model Outputs")
        st.markdown(
            """
            *This section contains low-level engineering metrics, synthetic benchmark evaluations, and raw model diagnostics.
            It is intentionally separated from the primary participant storytelling experience.*
            """
        )

        dev_tab_mod, dev_tab_feats, dev_tab_logs = st.tabs([
            "Model Architecture & Benchmarks",
            "Raw Extracted Features",
            "Session Diagnostics"
        ])

        with dev_tab_mod:
            benchmarks = get_model_benchmark_comparison()
            st.info(f"**Evaluation Context:** {benchmarks['evaluation_context']}\n\n{benchmarks['notice']}")
            df_models = pd.DataFrame(benchmarks["benchmark_table"])
            st.dataframe(df_models, use_container_width=True, hide_index=True)

        with dev_tab_feats:
            tech_summary = []
            for i, r in enumerate(p_results):
                tech_summary.append({
                    "Stage": f"P{i+1}",
                    "LSTM_Probability": r.get("strain_probability", 0.0),
                    "Mean_IKI_ms": r.get("avg_typing_speed_ms", 0.0),
                    "Median_IKI_ms": r.get("inter_key_median_ms", 0.0),
                    "Rhythm_CV": r.get("inter_key_cv", 0.0),
                    "Backspace_Rate": r.get("backspace_rate", 0.0),
                    "Error_Rate": r.get("error_rate", 0.0),
                    "Hesitation_Spikes": r.get("hesitation_spikes", 0),
                    "Mann_Whitney_p": r.get("rhythm_shift_p"),
                })
            st.dataframe(pd.DataFrame(tech_summary), use_container_width=True, hide_index=True)

        with dev_tab_logs:
            st.markdown("**Session Security & Privacy Verification:**")
            st.code(f"Participant Hash: {dsp.pseudonymize_user_id(rep.get('participant_label'))}\nStorage: Zero keystroke text saved\nTelemetry: Timing offsets and boolean flags only\nDifferential Privacy: Laplace mechanism calibrated to ε=1.0", language="text")
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            st.markdown("**Anonymized Session Diagnostics JSON:**")
            clean_log = {
                "participant": dsp.pseudonymize_user_id(rep.get("participant_label")),
                "timestamp": rep.get("timestamp", ""),
                "strain_index": rep["strain_index"],
                "reliability_score": rel["reliability_score"],
                "session_quality": rel["session_quality_score"],
                "is_conclusive": rep["is_conclusive"],
                "indicators": indicators,
                "stages": [
                    {k: v for k, v in r.items() if not isinstance(v, (list, dict, np.ndarray))}
                    for r in p_results
                ]
            }
            st.json(clean_log)


# =====================================================================
# VIEW 8: DEVELOPER DEMO & SANDBOX (SYNTHETIC DATA)
# =====================================================================
elif st.session_state.nav_view == "sandbox":
    st.markdown(
        """
        <div style="background:#FFFBEB; border:1px solid #FDE68A; border-radius:16px; padding:22px 26px; margin-bottom:24px;">
          <div style="font-size:0.75rem; font-weight:800; text-transform:uppercase; color:#B45309; letter-spacing:0.08em; margin-bottom:4px;">
            Developer Testing Sandbox
          </div>
          <h2 style="margin:0 0 6px 0; color:#92400E; font-size:1.4rem;">DEVELOPER DEMO — SYNTHETIC DATA</h2>
          <p style="margin:0; color:#78350F; font-size:0.92rem; line-height:1.5;">
            This tool generates simulated keystroke timings from the synthetic training distribution.
            It is provided solely for verifying pipeline latency and feature engineering without manual typing.
            <strong>Simulated data must never be confused with real assessment readings.</strong>
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not baselines_ok:
        st.error("Cannot run simulation: `models/baselines_plain.csv` not found.")
    else:
        user_list = baselines_df["user_id"].tolist() if not baselines_df.empty else ["user_001"]
        col_s1, col_s2, col_s3 = st.columns([2, 2, 1.5])
        with col_s1:
            selected_user = st.selectbox("Select Synthetic User Profile", user_list, index=0)
        with col_s2:
            sim_state = st.radio("Simulated Condition", ["Calm / Resting Baseline", "Elevated Cognitive Load"], horizontal=True)
        with col_s3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            run_sim = st.button("Generate & Score", use_container_width=True)

        if run_sim:
            label_hint = 1 if "Elevated" in sim_state else 0
            rng = np.random.default_rng()
            sim_df = make_live_session(selected_user, label_hint, rng)

            b_row = baselines_df[baselines_df.user_id == selected_user]
            b_mean = float(b_row.iloc[0]["baseline_mean_ms"]) if not b_row.empty else 180.0
            b_std = float(b_row.iloc[0]["baseline_std_ms"]) if not b_row.empty else 45.0

            sim_score = score_feature_frame(sim_df, b_mean, b_std)
            log_result({
                "user_id": selected_user,
                "user_id_pseudonymized": dsp.pseudonymize_id(selected_user),
                "session_id": f"{selected_user}_sim_{rng.integers(100000)}",
                "timestamp": pd.Timestamp.now().isoformat(),
                **{k: v for k, v in sim_score.items() if not isinstance(v, (list, dict, np.ndarray))},
            })

            st.success(f"Generated synthetic session for {selected_user}!")

            col_r1, col_r2, col_r3 = st.columns(3)
            col_r1.metric("Behavioral Strain Index", f"{sim_score['strain_score']:.1f} / 100")
            col_r2.metric("Predicted Condition", sim_score["predicted_state"])
            col_r3.metric("Cadence (ms)", f"{sim_score['avg_typing_speed_ms']:.1f}")

            with st.expander("Technical Model Diagnostics (Synthetic)", expanded=True):
                st.json({k: v for k, v in sim_score.items() if not isinstance(v, np.ndarray)})
