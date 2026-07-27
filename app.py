import html
import json
import os
import re
import textwrap
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import shap
import streamlit.components.v1 as components

# Configure page layout and metadata
st.set_page_config(
    page_title="Review Authenticity Engine | FraudScope",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# CONFIGURABLE THRESHOLDS & CONSTANTS
# ==============================================================================

CRITICAL_RISK_THRESHOLD = 0.80
HIGH_RISK_THRESHOLD = 0.50

UNIQUE_WORD_RATIO_THRESHOLD = 0.45
MIN_WORD_COUNT_FOR_RATIO_CHECK = 5
CONSECUTIVE_REPEAT_COUNT = 3

# ==============================================================================
# PREMIUM ENTERPRISE DESIGN SYSTEM (CSS)
# ==============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300..900;1,14..32,300..900&display=swap');

/* ── Reset & tokens ─────────────────────────────────────────────────────── */
:root {
    --bg:            #06080f;
    --bg-mid:        #090d18;
    --panel:         rgba(14, 20, 36, 0.92);
    --panel-2:       rgba(11, 17, 30, 0.88);
    --line:          rgba(148, 163, 184, 0.13);
    --line-hi:       rgba(129, 140, 248, 0.38);
    --text:          #f1f5f9;
    --muted:         #94a3b8;
    --faint:         #64748b;
    --indigo:        #6366f1;
    --indigo-hi:     #818cf8;
    --blue:          #38bdf8;
    --purple:        #a855f7;
    --amber:         #f59e0b;
    --crimson:       #ef4444;
    --emerald:       #10b981;

    /* Glow palette */
    --glow-indigo:   0 0 40px rgba(99,102,241,0.18);
    --glow-blue:     0 0 40px rgba(56,189,248,0.12);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── App background ─────────────────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 80% 55% at 50% -10%, rgba(99,102,241,0.13) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 90% 110%, rgba(168,85,247,0.09) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 5%  80%,  rgba(56,189,248,0.07) 0%, transparent 60%),
        #06080f;
    color: var(--text);
}

/* ── Container ──────────────────────────────────────────────────────────── */
.main .block-container,
[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1080px !important;
    padding: 0 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Chrome clean-up ────────────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"]  { display: none; }
[data-testid="stToolbar"]     { display: none; }
h1, h2, h3, h4, p { letter-spacing: 0; }

/* ── Dividers ───────────────────────────────────────────────────────────── */
hr {
    border: 0 !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.18), transparent) !important;
    margin: 2rem 0 !important;
}

/* ── Navigation tabs ────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: rgba(11, 17, 30, 0.80);
    padding: 0.4rem;
    border-radius: 14px;
    border: 1px solid var(--line);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.04),
        0 20px 60px rgba(0,0,0,0.30);
    margin-bottom: 1.4rem;
    backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 0.96rem !important;
    font-weight: 600 !important;
    min-height: 44px !important;
    padding: 0 1.35rem !important;
    transition: all 200ms cubic-bezier(0.4,0,0.2,1) !important;
    letter-spacing: -0.01em !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(255,255,255,0.045) !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(145deg, rgba(99,102,241,0.96), rgba(79,70,229,0.92)) !important;
    box-shadow:
        0 8px 24px rgba(79,70,229,0.38),
        inset 0 1px 0 rgba(255,255,255,0.20) !important;
}

/* ── Hero card ──────────────────────────────────────────────────────────── */
.hero-card {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(129, 140, 248, 0.28);
    border-radius: 22px;
    padding: 2.6rem 2.8rem 2.4rem;
    margin: 0 0 1.6rem;
    background:
        linear-gradient(145deg,
            rgba(14,20,36,0.98) 0%,
            rgba(11,17,30,0.94) 50%,
            rgba(22,30,50,0.80) 100%),
        linear-gradient(120deg, rgba(99,102,241,0.18), rgba(168,85,247,0.10));
    box-shadow:
        0 32px 80px rgba(0,0,0,0.44),
        inset 0 1px 0 rgba(255,255,255,0.07),
        0 0 0 1px rgba(129,140,248,0.10);
    backdrop-filter: blur(20px);
}

/* Grid texture */
.hero-card::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.030) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 32px 32px;
    mask-image: linear-gradient(135deg, black 20%, transparent 75%);
    pointer-events: none;
}

/* Corner glow */
.hero-card::after {
    content: "";
    position: absolute;
    top: -60px; left: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,0.22), transparent 65%);
    pointer-events: none;
}

.hero-content {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2.5rem;
}
.hero-left  { flex: 1 1 0; min-width: 0; }
.hero-right {
    flex: 0 0 270px;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.hero-kicker {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    color: #a5b4fc;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    padding: 0.32rem 0.82rem;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.26);
    border-radius: 999px;
}

.hero-title {
    color: #ffffff;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.035em;
    margin: 0 0 0.8rem;
    background: linear-gradient(135deg, #ffffff 35%, rgba(165,180,252,0.88) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.12rem;
    line-height: 1.62;
    margin-bottom: 1.5rem;
    font-weight: 400;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
    border: 1px solid rgba(129,140,248,0.28);
    background: rgba(99,102,241,0.10);
    color: #c7d2fe;
    font-size: 0.88rem;
    font-weight: 600;
    backdrop-filter: blur(8px);
    transition: border-color 200ms ease, background 200ms ease;
}
.hero-badge:hover {
    border-color: rgba(129,140,248,0.50);
    background: rgba(99,102,241,0.18);
}

/* Hero right-side stat tiles */
.hero-stat {
    border: 1px solid rgba(129,140,248,0.18);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    background: rgba(99,102,241,0.07);
    backdrop-filter: blur(8px);
    display: flex;
    align-items: center;
    gap: 0.85rem;
}
.hero-stat-icon {
    font-size: 1.5rem;
    flex-shrink: 0;
}
.hero-stat-label {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.18rem;
}
.hero-stat-value {
    color: #ffffff;
    font-size: 1.08rem;
    font-weight: 750;
    letter-spacing: -0.01em;
}

@media (max-width: 760px) {
    .hero-content { flex-direction: column; }
    .hero-right    { flex: 1 1 auto; width: 100%; flex-direction: row; flex-wrap: wrap; }
    .hero-stat     { flex: 1 1 140px; }
}

/* ── Section typography ─────────────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    margin: 1.8rem 0 0.3rem;
}
.section-title {
    color: var(--text);
    font-size: 1.55rem;
    font-weight: 750;
    letter-spacing: -0.02em;
}
.section-pill {
    background: rgba(99,102,241,0.14);
    color: #a5b4fc;
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    border: 1px solid rgba(129,140,248,0.24);
    text-transform: uppercase;
}
.section-copy {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
    margin: 0 0 0.9rem;
    font-weight: 400;
}

/* ── Input card ─────────────────────────────────────────────────────────── */
.input-shell {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.4rem 1.55rem 1.1rem;
    background: linear-gradient(175deg, rgba(14,20,36,0.95), rgba(11,17,30,0.88));
    box-shadow:
        0 20px 50px rgba(0,0,0,0.28),
        inset 0 1px 0 rgba(255,255,255,0.05);
    margin-bottom: 0.9rem;
    backdrop-filter: blur(12px);
    transition: border-color 200ms ease;
}
.input-shell:hover { border-color: rgba(148,163,184,0.22); }

.input-title {
    color: #f1f5f9;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.01em;
}
.input-copy {
    color: var(--muted);
    font-size: 0.96rem;
    line-height: 1.55;
    margin-bottom: 0;
}

/* ── Textarea & inputs ──────────────────────────────────────────────────── */
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label,
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label {
    color: #cbd5e1 !important;
    font-size: 0.91rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.005em !important;
}
.stTextArea textarea,
.stTextInput input {
    color: #e5e7eb !important;
    background: rgba(2, 6, 23, 0.60) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.025) !important;
    transition: border-color 180ms ease, box-shadow 180ms ease !important;
    font-size: 1.02rem !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea { min-height: 120px !important; line-height: 1.6 !important; }
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(99,102,241,0.65) !important;
    box-shadow:
        0 0 0 3px rgba(99,102,241,0.14),
        inset 0 1px 0 rgba(255,255,255,0.025) !important;
    background: rgba(2,6,23,0.76) !important;
}
.stTextArea textarea::placeholder { color: #475569 !important; }

/* Select box */
.stSelectbox [data-baseweb="select"] > div {
    color: #e5e7eb !important;
    background: rgba(2,6,23,0.60) !important;
    border: 1px solid rgba(148,163,184,0.16) !important;
    border-radius: 10px !important;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBarMax"] {
    color: var(--faint) !important;
    font-size: 0.82rem !important;
}

/* ── Primary CTA button ─────────────────────────────────────────────────── */
div.stButton > button[kind="primary"],
div.stButton > button[type="primary"] {
    width: 100% !important;
    min-height: 52px !important;
    border: 0 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.01em !important;
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 55%, #7c3aed 100%) !important;
    box-shadow:
        0 14px 36px rgba(79,70,229,0.34),
        inset 0 1px 0 rgba(255,255,255,0.18) !important;
    transition: transform 160ms cubic-bezier(0.4,0,0.2,1),
                box-shadow 160ms cubic-bezier(0.4,0,0.2,1),
                filter 160ms ease !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[type="primary"]:hover {
    transform: translateY(-2px) !important;
    filter: brightness(1.09) !important;
    box-shadow:
        0 20px 44px rgba(79,70,229,0.44),
        inset 0 1px 0 rgba(255,255,255,0.22) !important;
}
div.stButton > button[kind="primary"]:active,
div.stButton > button[type="primary"]:active {
    transform: translateY(0px) scale(0.99) !important;
    filter: brightness(0.96) !important;
}

/* ── Metric / KPI cards ─────────────────────────────────────────────────── */
.metric-card, .kpi-card {
    position: relative;
    overflow: hidden;
    min-height: 160px;
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.45rem 1.55rem;
    background: linear-gradient(175deg, rgba(14,20,36,0.95), rgba(11,17,30,0.80));
    box-shadow:
        0 16px 44px rgba(0,0,0,0.26),
        inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    transition:
        transform 200ms cubic-bezier(0.4,0,0.2,1),
        border-color 200ms ease,
        box-shadow 200ms ease;
}
.metric-card:hover, .kpi-card:hover {
    transform: translateY(-4px);
    border-color: var(--line-hi);
    box-shadow:
        0 24px 56px rgba(0,0,0,0.34),
        0 0 0 1px rgba(99,102,241,0.10);
}

/* Accent left strip */
.metric-card::before, .kpi-card::before {
    content: "";
    position: absolute;
    inset: 12px auto 12px 0;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: linear-gradient(180deg, var(--blue), var(--indigo));
}
.accent-success::before  { background: linear-gradient(180deg, #34d399, #10b981); }
.accent-danger::before   { background: linear-gradient(180deg, #fb7185, #ef4444); }
.accent-warning::before  { background: linear-gradient(180deg, #fbbf24, #f59e0b); }
.accent-purple::before   { background: linear-gradient(180deg, #c084fc, #8b5cf6); }
.accent-blue::before     { background: linear-gradient(180deg, #7dd3fc, #38bdf8); }

/* Inner corner glow */
.metric-card::after, .kpi-card::after {
    content: "";
    position: absolute;
    bottom: -30px; right: -30px;
    width: 100px; height: 100px;
    border-radius: 50%;
    background: rgba(99,102,241,0.06);
    pointer-events: none;
}

.card-icon {
    width: 2.2rem;
    height: 2.2rem;
    display: grid;
    place-items: center;
    border-radius: 10px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(129,140,248,0.20);
    font-size: 1.15rem;
    margin-bottom: 0.9rem;
}
.card-label, .kpi-label {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.card-value, .kpi-value {
    color: #ffffff;
    font-size: 2.3rem;
    line-height: 1;
    font-weight: 800;
    letter-spacing: -0.03em;
    margin-bottom: 0.45rem;
}
.card-subtitle, .kpi-desc {
    color: var(--muted);
    font-size: 0.88rem;
    font-weight: 400;
}

/* ── SHAP evidence panel ─────────────────────────────────────────────────── */
.shap-card {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.3rem 1.55rem;
    background: linear-gradient(175deg, rgba(14,20,36,0.95), rgba(11,17,30,0.88));
    box-shadow: 0 16px 44px rgba(0,0,0,0.24), inset 0 1px 0 rgba(255,255,255,0.05);
    margin-top: 1.4rem;
    backdrop-filter: blur(10px);
}
.shap-title {
    color: #f1f5f9;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.22rem;
    letter-spacing: -0.01em;
}
.shap-caption {
    color: var(--muted);
    font-size: 0.93rem;
    line-height: 1.55;
    margin-bottom: 0.85rem;
}
.shap-frame {
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 12px;
    padding: 0.75rem;
    background: rgba(2,6,23,0.48);
}

/* ── Filter panel ───────────────────────────────────────────────────────── */
.filter-shell {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 1.3rem 1.55rem 1rem;
    background: linear-gradient(175deg, rgba(14,20,36,0.95), rgba(11,17,30,0.88));
    box-shadow: 0 16px 44px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.05);
    margin: 1.2rem 0 1rem;
    backdrop-filter: blur(10px);
}
.filter-title {
    color: #f1f5f9;
    font-size: 1.06rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    letter-spacing: -0.01em;
}
.filter-copy {
    color: var(--muted);
    font-size: 0.93rem;
    line-height: 1.55;
    margin-bottom: 0.6rem;
}

/* ── Risk badges ─────────────────────────────────────────────────────────── */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.28rem;
    border-radius: 999px;
    padding: 0.28rem 0.72rem;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}
.badge-critical { color: #fff1f2; background: rgba(239,68,68,0.92); box-shadow: 0 0 12px rgba(239,68,68,0.28); }
.badge-high     { color: #1c0a00; background: rgba(245,158,11,0.95); box-shadow: 0 0 12px rgba(245,158,11,0.22); }
.badge-medium   { color: #1c1400; background: rgba(250,204,21,0.92); }
.badge-low      { color: #012414; background: rgba(52,211,153,0.90); }

/* ── Fraud ring cards ───────────────────────────────────────────────────── */
.ring-detail-card {
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    background: rgba(2,6,23,0.32);
    margin-bottom: 0.9rem;
    backdrop-filter: blur(6px);
}
.ring-detail-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
}
.ring-title {
    color: #f1f5f9;
    font-size: 1.03rem;
    font-weight: 750;
    letter-spacing: -0.01em;
}
.ring-meta { color: var(--muted); font-size: 0.79rem; margin-top: 0.2rem; }

/* Progress bar */
.progress-label {
    display: flex;
    justify-content: space-between;
    color: #cbd5e1;
    font-size: 0.79rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.progress-track {
    height: 0.62rem;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(51,65,85,0.65);
    border: 1px solid rgba(148,163,184,0.12);
}
.progress-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #38bdf8 0%, #6366f1 45%, #ef4444 100%);
    transition: width 600ms cubic-bezier(0.4,0,0.2,1);
}

/* Ring stats grid */
.ring-stats {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.65rem;
    margin-top: 0.9rem;
}
.detail-tile {
    border: 1px solid rgba(148,163,184,0.11);
    border-radius: 12px;
    padding: 0.8rem 0.9rem;
    background: rgba(2,6,23,0.40);
    transition: border-color 180ms ease;
}
.detail-tile:hover { border-color: rgba(129,140,248,0.28); }
.detail-tile span {
    color: var(--muted);
    display: block;
    font-size: 0.74rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.detail-tile strong {
    color: #f1f5f9;
    font-size: 1.14rem;
    font-weight: 750;
    letter-spacing: -0.01em;
}
.detail-tile strong.metric-emphasis {
    font-size: 1.3rem;
    font-weight: 800;
    color: #ffffff;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid rgba(148,163,184,0.13) !important;
    border-radius: 18px !important;
    background: linear-gradient(175deg, rgba(14,20,36,0.94), rgba(11,17,30,0.82)) !important;
    box-shadow: 0 14px 38px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.04);
    margin-bottom: 0.85rem;
    overflow: hidden;
    transition:
        transform 200ms cubic-bezier(0.4,0,0.2,1),
        border-color 200ms ease,
        box-shadow 200ms ease;
    backdrop-filter: blur(10px);
}
[data-testid="stExpander"]:hover {
    transform: translateY(-2px);
    border-color: rgba(129,140,248,0.36) !important;
    box-shadow: 0 18px 48px rgba(0,0,0,0.30);
}
[data-testid="stExpander"] details summary {
    padding: 0.95rem 1.1rem !important;
}
[data-testid="stExpander"] details summary p {
    margin: 0 !important;
    width: 100%;
    font-size: 0.95rem !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
}
[data-testid="stExpanderDetails"] {
    border-top: 1px solid rgba(148,163,184,0.10);
    padding: 1rem 1.1rem 1.15rem !important;
}

/* ── Review table ────────────────────────────────────────────────────────── */
.review-table-wrap {
    max-height: 440px;
    overflow: auto;
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 14px;
    background: rgba(2,6,23,0.44);
    scrollbar-width: thin;
    scrollbar-color: rgba(99,102,241,0.3) transparent;
}
.review-table-wrap::-webkit-scrollbar { width: 6px; height: 6px; }
.review-table-wrap::-webkit-scrollbar-track { background: transparent; }
.review-table-wrap::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.30); border-radius: 999px; }
.review-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    table-layout: fixed;
    color: #e5e7eb;
    font-size: 0.87rem;
}
.review-table th {
    position: sticky;
    top: 0;
    z-index: 1;
    text-align: left;
    color: #94a3b8;
    background: rgba(11,17,30,0.99);
    border-bottom: 1px solid rgba(148,163,184,0.16);
    padding: 0.75rem 0.88rem;
    font-size: 0.72rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    font-weight: 700;
}
.review-table td {
    border-bottom: 1px solid rgba(148,163,184,0.08);
    padding: 0.8rem 0.88rem;
    vertical-align: top;
    word-wrap: break-word;
    white-space: normal;
    line-height: 1.5;
}
.review-table tr:nth-child(even) td { background: rgba(15,23,42,0.30); }
.review-table tr:hover td {
    background: rgba(99,102,241,0.07);
    transition: background 120ms ease;
}
.col-reviewer { width: 17%; }
.col-product  { width: 17%; }
.col-rating   { width: 9%;  }
.col-text     { width: 57%; }

/* ── Confidence ring (SVG) ───────────────────────────────────────────────── */
.conf-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
}
.conf-ring-label {
    color: var(--muted);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3.5rem 1rem;
    border: 1px dashed rgba(148,163,184,0.18);
    border-radius: 18px;
    background: rgba(14,20,36,0.50);
    margin: 1rem 0;
}
.empty-state-icon { font-size: 2.6rem; margin-bottom: 0.7rem; }
.empty-state-title {
    color: #e2e8f0;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    letter-spacing: -0.01em;
}
.empty-state-body { color: var(--muted); font-size: 0.95rem; line-height: 1.55; }

/* ── Error card ──────────────────────────────────────────────────────────── */
.error-card {
    border: 1px solid rgba(239,68,68,0.28);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    background: rgba(239,68,68,0.06);
    margin: 1rem 0;
    backdrop-filter: blur(8px);
}
.error-card-title { color: #fca5a5; font-size: 1.05rem; font-weight: 700; margin-bottom: 0.4rem; }
.error-card-body  { color: #f87171; font-size: 0.93rem; line-height: 1.55; }

/* ── Spinner override ────────────────────────────────────────────────────── */
.stSpinner > div {
    border-color: var(--indigo) transparent transparent transparent !important;
}

/* ── Info / warning messages ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(129,140,248,0.22) !important;
    border-radius: 12px !important;
    color: #c7d2fe !important;
}

/* ── Footer ──────────────────────────────────────────────────────────────── */
.footer-note {
    color: var(--faint);
    text-align: center;
    font-size: 0.84rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(148,163,184,0.10);
}

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 900px) {
    .main .block-container,
    [data-testid="stAppViewContainer"] .main .block-container {
        padding: 0 1.1rem 3rem !important;
    }
    .hero-card { padding: 2rem 1.8rem; border-radius: 18px; }
    .hero-title { font-size: 2.3rem; }
    .ring-stats { grid-template-columns: repeat(3, minmax(0,1fr)); }
}
@media (max-width: 560px) {
    .hero-title { font-size: 1.9rem; }
    .hero-card  { padding: 1.5rem 1.25rem; }
    .ring-stats { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ML PREDICTION PIPELINE (STRICTLY UNTOUCHED - EXACT ORIGINAL IMPLEMENTATION)
# ==============================================================================

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("Smitvkohale/review-authenticity-engine")
    model = DistilBertForSequenceClassification.from_pretrained("Smitvkohale/review-authenticity-engine")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def render_html(markup: str) -> None:
    """Render custom HTML without leading indentation becoming Markdown code."""
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)

def predict_proba(texts):
    enc = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=-1).numpy()


# ==============================================================================
# DATA LOADING, RISK SCORING, & DISPLAY-ONLY QUALITY FILTERING
# ==============================================================================

@st.cache_data
def load_rings_data():
    json_path = os.path.join(os.path.dirname(__file__), "detected_rings.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_risk_level(fake_ratio: float) -> tuple[str, str, str]:
    if fake_ratio >= CRITICAL_RISK_THRESHOLD:
        return "CRITICAL", "🔴", "badge-critical"
    elif fake_ratio >= HIGH_RISK_THRESHOLD:
        return "HIGH", "🟠", "badge-high"
    elif fake_ratio >= 0.25:
        return "MEDIUM", "🟡", "badge-medium"
    else:
        return "LOW", "🟢", "badge-low"

def is_low_quality_review(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    if re.search(r'\b(\w+)(?:\s+\1){' + str(CONSECUTIVE_REPEAT_COUNT - 1) + r',}\b', text.lower()):
        return True
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) >= MIN_WORD_COUNT_FOR_RATIO_CHECK:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < UNIQUE_WORD_RATIO_THRESHOLD:
            return True
    return False

def make_progress_bar_html(ratio: float) -> str:
    pct = max(0, min(100, round(ratio * 100, 1)))
    return textwrap.dedent(f"""
    <div>
        <div class="progress-label"><span>Fake ratio</span><span>{pct:.1f}%</span></div>
        <div class="progress-track"><div class="progress-fill" style="width: {pct}%;"></div></div>
    </div>
    """).strip()

def render_review_table(reviews: list[dict]) -> str:
    rows = []
    for review in reviews:
        reviewer = html.escape(str(review.get("user_id", "N/A")))
        product  = html.escape(str(review.get("asin",    "N/A")))
        rating   = html.escape(str(review.get("rating",  "N/A")))
        text     = html.escape(str(review.get("text",    "")))
        rows.append(
            f"<tr><td>{reviewer}</td><td>{product}</td><td>{rating}</td><td>{text}</td></tr>"
        )
    return textwrap.dedent(f"""
    <style>
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; background: #020617; color: #e5e7eb;
               font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
        ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(99,102,241,0.32); border-radius: 999px; }}
        .review-table-wrap {{
            height: 340px; overflow: auto;
            border: 1px solid rgba(148,163,184,0.16);
            border-radius: 12px;
            background: rgba(2,6,23,0.80);
        }}
        .review-table {{
            width: 100%; border-collapse: separate; border-spacing: 0;
            table-layout: fixed; font-size: 13.5px;
        }}
        .review-table th {{
            position: sticky; top: 0; z-index: 1;
            text-align: left; color: #94a3b8;
            background: #0b1120;
            border-bottom: 1px solid rgba(148,163,184,0.18);
            padding: 11px 13px; font-size: 11.5px;
            letter-spacing: 0.09em; text-transform: uppercase; font-weight: 700;
        }}
        .review-table td {{
            border-bottom: 1px solid rgba(148,163,184,0.09);
            padding: 11px 13px; vertical-align: top;
            overflow-wrap: anywhere; white-space: normal; line-height: 1.5;
        }}
        .review-table tr:nth-child(even) td {{ background: rgba(15,23,42,0.36); }}
        .review-table tr:hover td {{ background: rgba(99,102,241,0.08); }}
        .col-reviewer {{ width: 17%; }} .col-product {{ width: 17%; }}
        .col-rating   {{ width: 9%;  }} .col-text    {{ width: 57%; }}
    </style>
    <div class="review-table-wrap">
        <table class="review-table">
            <thead>
                <tr>
                    <th class="col-reviewer">Reviewer ID</th>
                    <th class="col-product">Product ID</th>
                    <th class="col-rating">Rating</th>
                    <th class="col-text">Review Text</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
    """).strip()

def render_confidence_ring(value: float, label: str, color: str = "#6366f1") -> str:
    """SVG circular progress ring for confidence/probability visualization."""
    r = 38
    circ = 2 * 3.14159 * r
    offset = circ * (1 - value)
    pct_text = f"{value:.1%}"
    return textwrap.dedent(f"""
    <div class="conf-ring-wrap">
        <svg width="100" height="100" viewBox="0 0 100 100" style="transform:rotate(-90deg)">
            <circle cx="50" cy="50" r="{r}" fill="none"
                stroke="rgba(51,65,85,0.7)" stroke-width="9"/>
            <circle cx="50" cy="50" r="{r}" fill="none"
                stroke="{color}" stroke-width="9"
                stroke-linecap="round"
                stroke-dasharray="{circ:.2f}"
                stroke-dashoffset="{offset:.2f}"
                style="transition: stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)"/>
        </svg>
        <div style="position:relative;text-align:center;margin-top:-72px;height:64px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <div style="color:#ffffff;font-size:1.22rem;font-weight:800;
                        letter-spacing:-0.02em;line-height:1;">{pct_text}</div>
        </div>
        <div style="margin-top:12px;" class="conf-ring-label">{label}</div>
    </div>
    """).strip()


# ==============================================================================
# MAIN APPLICATION INTERFACE
# ==============================================================================

tab_scorer, tab_dashboard = st.tabs(["🛡️  Single Review Scorer", "🕸️  Fraud Ring Dashboard"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: REVIEW SCORER
# ─────────────────────────────────────────────────────────────────────────────
with tab_scorer:

    render_html("""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-left">
                <div class="hero-kicker">🛡 FraudScope AI · Security Platform</div>
                <div class="hero-title">Review Authenticity Engine</div>
                <div class="hero-subtitle">
                    AI-powered fake review detection using natural language understanding
                    and behavioral fraud pattern analysis. Powered by fine-tuned DistilBERT.
                </div>
                <div class="hero-badges">
                    <span class="hero-badge">✓ DistilBERT NLP</span>
                    <span class="hero-badge">✓ Explainable AI</span>
                    <span class="hero-badge">✓ Fraud Ring Detection</span>
                    <span class="hero-badge">✓ Network Analytics</span>
                </div>
            </div>
            <div class="hero-right">
                <div class="hero-stat">
                    <div class="hero-stat-icon">🤖</div>
                    <div>
                        <div class="hero-stat-label">Model</div>
                        <div class="hero-stat-value">DistilBERT</div>
                    </div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-icon">🔬</div>
                    <div>
                        <div class="hero-stat-label">Explainability</div>
                        <div class="hero-stat-value">SHAP Attribution</div>
                    </div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-icon">🕸️</div>
                    <div>
                        <div class="hero-stat-label">Analysis Type</div>
                        <div class="hero-stat-value">NLP + Graph</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """)

    render_html("""
    <div class="input-shell">
        <div class="input-title">📄 Review Intake</div>
        <div class="input-copy">
            Paste any product review below to score its authenticity, estimate fake-review probability,
            and generate token-level attribution evidence via SHAP.
        </div>
    </div>
    """)

    review_text = st.text_area(
        label="Review Content",
        height=130,
        placeholder="e.g. Amazing product, highly recommend — best purchase I've ever made, five stars...",
        label_visibility="visible"
    )

    analyze = st.button("🔍  Analyze Review Authenticity", type="primary", use_container_width=True)

    if analyze and not review_text.strip():
        render_html("""
        <div class="empty-state">
            <div class="empty-state-icon">📋</div>
            <div class="empty-state-title">No Review Provided</div>
            <div class="empty-state-body">Please paste a review into the field above before running analysis.</div>
        </div>
        """)

    elif analyze and review_text.strip():
        with st.spinner("Executing DistilBERT inference & SHAP attribution — this may take a moment..."):
            try:
                probs = predict_proba([review_text])[0]
                fake_prob  = float(probs[1])
                real_prob  = 1.0 - fake_prob
                confidence = max(fake_prob, real_prob)
                is_fake    = fake_prob > 0.5
                risk_score = fake_prob * confidence

                render_html('<div class="section-header"><div class="section-title">🎯 Analysis Results</div></div>')
                render_html('<div class="section-copy">Authenticity readout calibrated for analyst review and rapid triage.</div>')

                # ── Three KPI cards ──────────────────────────────────────────
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    verdict_class = "accent-danger" if is_fake else "accent-success"
                    verdict_text  = "Likely FAKE" if is_fake else "Likely REAL"
                    verdict_sub   = "High fake risk detected" if is_fake else "Authentic review patterns"
                    render_html(f"""
                    <div class="metric-card {verdict_class}">
                        <div class="card-icon">🛡️</div>
                        <div class="card-label">Verdict</div>
                        <div class="card-value">{verdict_text}</div>
                        <div class="card-subtitle">{verdict_sub}</div>
                    </div>
                    """)
                with rc2:
                    render_html(f"""
                    <div class="metric-card accent-warning">
                        <div class="card-icon">📊</div>
                        <div class="card-label">Fake Probability</div>
                        <div class="card-value">{fake_prob:.1%}</div>
                        <div class="card-subtitle">Probability score (DistilBERT)</div>
                    </div>
                    """)
                with rc3:
                    render_html(f"""
                    <div class="metric-card accent-purple">
                        <div class="card-icon">🎯</div>
                        <div class="card-label">Model Confidence</div>
                        <div class="card-value">{confidence:.1%}</div>
                        <div class="card-subtitle">Decision certainty</div>
                    </div>
                    """)

                # ── Confidence visualization (circular rings) ────────────────
                render_html('<div class="section-header" style="margin-top:1.8rem;"><div class="section-title">📈 Confidence Visualization</div></div>')
                render_html('<div class="section-copy">Visual breakdown of the model\'s probability distribution and composite risk score.</div>')

                v1, v2, v3 = st.columns(3)
                fake_color = "#ef4444" if fake_prob > 0.5 else "#10b981"
                with v1:
                    render_html(render_confidence_ring(fake_prob, "Fake Probability", fake_color))
                with v2:
                    render_html(render_confidence_ring(confidence, "Model Confidence", "#6366f1"))
                with v3:
                    render_html(render_confidence_ring(min(risk_score * 2, 1.0), "Composite Risk", "#f59e0b"))

                # ── SHAP evidence panel ──────────────────────────────────────
                render_html("""
                <div class="shap-card">
                    <div class="shap-title">🔬 Why the model reached this decision</div>
                    <div class="shap-caption">
                        Words highlighted in <strong style="color:#ef4444;">red</strong> increased the fake-review score.
                        Words highlighted in <strong style="color:#38bdf8;">blue</strong> supported genuine authenticity.
                        Token attribution computed via SHAP kernel explainer.
                    </div>
                </div>
                """)

                # SHAP explanation — exact original code preserved
                explainer  = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))
                shap_values = explainer([review_text])
                html_output = shap.plots.text(shap_values[0, :, 1], display=False)
                render_html('<div class="shap-card"><div class="shap-frame">')
                components.html(html_output, height=220, scrolling=True)
                render_html('</div></div>')

            except Exception as e:
                render_html(f"""
                <div class="error-card">
                    <div class="error-card-title">⚠️ Analysis Error</div>
                    <div class="error-card-body">
                        The model encountered an issue during inference.<br><br>
                        <strong>Possible causes:</strong> Network connectivity issue loading the Hugging Face model,
                        or an unexpected input format.<br><br>
                        <strong>How to fix:</strong> Verify your internet connection and try again. If the problem
                        persists, the model host may be temporarily unavailable.
                    </div>
                </div>
                """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: FRAUD RING DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:

    render_html("""
    <div class="hero-card">
        <div class="hero-content">
            <div class="hero-kicker">🕸 Network Intelligence · Behavioral Analysis</div>
            <div class="hero-title">Detected Fraud Rings</div>
            <div class="hero-subtitle">
                A fraud ring is a coordinated cluster of reviewer accounts exhibiting suspicious
                collective behavior — shared product targets, timing patterns, and artificially inflated ratings.
                Use this dashboard to prioritize investigation by fake-review density, reviewer cluster size,
                and affected product scope.
            </div>
            <div class="hero-badges">
                <span class="hero-badge">🕸 Coordinated reviewer networks</span>
                <span class="hero-badge">📊 Behavioral fraud signals</span>
                <span class="hero-badge">🛡 Analyst-ready triage</span>
            </div>
        </div>
    </div>
    """)

    rings_data = load_rings_data()

    if not rings_data:
        render_html("""
        <div class="empty-state">
            <div class="empty-state-icon">🗂️</div>
            <div class="empty-state-title">No Fraud Ring Data Available</div>
            <div class="empty-state-body">
                <code>detected_rings.json</code> is missing or empty.<br>
                Run the fraud ring detection pipeline and place the output file in the application directory.
            </div>
        </div>
        """)
    else:
        # ── Dynamic KPIs ─────────────────────────────────────────────────────
        total_rings          = len(rings_data)
        avg_fake_ratio       = sum(r.get("fake_ratio", 0) for r in rings_data) / total_rings
        total_flagged_reviews = sum(len(r.get("reviews", [])) for r in rings_data)
        critical_rings       = sum(1 for r in rings_data if r.get("fake_ratio", 0) >= CRITICAL_RISK_THRESHOLD)
        high_rings           = sum(1 for r in rings_data if HIGH_RISK_THRESHOLD <= r.get("fake_ratio", 0) < CRITICAL_RISK_THRESHOLD)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_html(f"""
            <div class="kpi-card accent-blue">
                <div class="card-icon">🕸️</div>
                <div class="kpi-label">Total Rings</div>
                <div class="kpi-value">{total_rings:,}</div>
                <div class="kpi-desc">Detected networks</div>
            </div>
            """)
        with k2:
            render_html(f"""
            <div class="kpi-card accent-purple">
                <div class="card-icon">📊</div>
                <div class="kpi-label">Avg Fake Ratio</div>
                <div class="kpi-value">{avg_fake_ratio:.1%}</div>
                <div class="kpi-desc">Mean network density</div>
            </div>
            """)
        with k3:
            render_html(f"""
            <div class="kpi-card accent-danger">
                <div class="card-icon">🔴</div>
                <div class="kpi-label">Critical Rings</div>
                <div class="kpi-value">{critical_rings:,}</div>
                <div class="kpi-desc">≥ {CRITICAL_RISK_THRESHOLD:.0%} fake ratio</div>
            </div>
            """)
        with k4:
            render_html(f"""
            <div class="kpi-card accent-warning">
                <div class="card-icon">📄</div>
                <div class="kpi-label">Flagged Reviews</div>
                <div class="kpi-value">{total_flagged_reviews:,}</div>
                <div class="kpi-desc">Total network entries</div>
            </div>
            """)

        # ── Filter panel ─────────────────────────────────────────────────────
        render_html("""
        <div class="filter-shell">
            <div class="filter-title">🔍 Search & Filter Controls</div>
            <div class="filter-copy">
                Filter coordinated clusters by Ring ID, Reviewer ID, or ASIN — or narrow by fake ratio threshold and sort order.
            </div>
        </div>
        """)

        f1, f2, f3, f4 = st.columns([2.2, 1.8, 1.8, 1.2])
        with f1:
            search_query = st.text_input(
                "Search Ring ID, Reviewer ID, or ASIN",
                placeholder="e.g. 2, FAKEUSER, or B09YD16CPP"
            )
        with f2:
            min_fake_ratio = st.slider("Min Fake Ratio", 0.0, 1.0, 0.0, 0.05, format="%.0f%%")
        with f3:
            sort_option = st.selectbox(
                "Sort By",
                ["Fake Ratio (High → Low)", "Ring Size (Large → Small)", "Cluster ID (Ascending)"]
            )
        with f4:
            st.markdown("<br>", unsafe_allow_html=True)
            reset = st.button("↺ Reset", use_container_width=True)

        if reset:
            search_query   = ""
            min_fake_ratio = 0.0
            sort_option    = "Fake Ratio (High → Low)"

        # ── Apply filter & sort ───────────────────────────────────────────────
        filtered_rings = []
        query_str = search_query.strip().lower()
        for ring in rings_data:
            fake_ratio = ring.get("fake_ratio", 0)
            cluster_id = str(ring.get("cluster_id", ""))
            if fake_ratio < min_fake_ratio:
                continue
            if query_str:
                reviews      = ring.get("reviews", [])
                match_cluster = query_str in cluster_id.lower()
                match_user    = any(query_str in str(r.get("user_id", "")).lower() for r in reviews)
                match_asin    = any(query_str in str(r.get("asin",    "")).lower() for r in reviews)
                if not (match_cluster or match_user or match_asin):
                    continue
            filtered_rings.append(ring)

        if sort_option == "Fake Ratio (High → Low)":
            filtered_rings.sort(key=lambda r: r.get("fake_ratio", 0), reverse=True)
        elif sort_option == "Ring Size (Large → Small)":
            filtered_rings.sort(key=lambda r: len(r.get("reviews", [])), reverse=True)
        else:
            filtered_rings.sort(key=lambda r: r.get("cluster_id", 0))

        render_html(
            f'<div class="section-copy" style="margin-top:0.4rem;">Displaying '
            f'<strong style="color:#e2e8f0;">{len(filtered_rings)}</strong> of '
            f'<strong style="color:#e2e8f0;">{total_rings}</strong> detected fraud rings</div>'
        )

        if not filtered_rings:
            render_html("""
            <div class="empty-state">
                <div class="empty-state-icon">🔎</div>
                <div class="empty-state-title">No Rings Match Your Filters</div>
                <div class="empty-state-body">
                    Try broadening your search query or lowering the minimum fake ratio threshold.
                </div>
            </div>
            """)

        # ── Render each ring ──────────────────────────────────────────────────
        for ring in filtered_rings:
            cluster_id      = ring.get("cluster_id", "N/A")
            fake_ratio      = ring.get("fake_ratio", 0)
            original_reviews = ring.get("reviews", [])
            total_ring_reviews = len(original_reviews)

            displayed_reviews = [r for r in original_reviews if not is_low_quality_review(r.get("text", ""))]
            hidden_count      = total_ring_reviews - len(displayed_reviews)

            risk_label, risk_icon, badge_class = get_risk_level(fake_ratio)
            unique_asins = len(set(r.get("asin", "") for r in original_reviews if "asin" in r))

            expander_title = (
                f"{risk_icon}  Ring #{cluster_id}  ·  {risk_label}  "
                f"·  {fake_ratio:.1%} fake  ·  {len(displayed_reviews)}/{total_ring_reviews} reviews  "
                f"·  {unique_asins} products"
            )

            with st.expander(expander_title):
                render_html(f"""
                <div class="ring-detail-card">
                    <div class="ring-detail-head">
                        <div>
                            <div class="ring-title">🛑 Fraud Ring #{html.escape(str(cluster_id))}</div>
                            <div class="ring-meta">Coordinated reviewer cluster · Behavioral anomaly detected</div>
                        </div>
                        <span class="risk-badge {badge_class}">{risk_icon} {risk_label}</span>
                    </div>
                    {make_progress_bar_html(fake_ratio)}
                    <div class="ring-stats">
                        <div class="detail-tile">
                            <span>Fake Ratio</span>
                            <strong class="metric-emphasis">{fake_ratio:.1%}</strong>
                        </div>
                        <div class="detail-tile">
                            <span>Visible Reviews</span>
                            <strong>{len(displayed_reviews):,}</strong>
                        </div>
                        <div class="detail-tile">
                            <span>Total Reviews</span>
                            <strong>{total_ring_reviews:,}</strong>
                        </div>
                        <div class="detail-tile">
                            <span>Products</span>
                            <strong>{unique_asins:,}</strong>
                        </div>
                        <div class="detail-tile">
                            <span>Hidden Artifacts</span>
                            <strong>{hidden_count:,}</strong>
                        </div>
                    </div>
                </div>
                """)

                if hidden_count > 0:
                    st.info(
                        f"ℹ️ **{hidden_count} low-quality synthetic artifact "
                        f"review{'s were' if hidden_count > 1 else ' was'} hidden** "
                        f"for readability — repetitive paraphrasing patterns detected."
                    )

                render_html('<div class="section-copy" style="margin-top:0.5rem;"><strong>📄 Review Table</strong></div>')
                if displayed_reviews:
                    components.html(render_review_table(displayed_reviews), height=370, scrolling=False)
                else:
                    render_html("""
                    <div class="empty-state">
                        <div class="empty-state-icon">🧹</div>
                        <div class="empty-state-title">All Reviews Filtered</div>
                        <div class="empty-state-body">
                            Every review in this ring was identified as a low-quality synthetic artifact
                            and hidden for readability.
                        </div>
                    </div>
                    """)

st.markdown("---")
render_html(
    '<div class="footer-note">'
    'Part of <strong>FraudScope</strong> — a two-layer fake review & coordination detection system '
    '(DistilBERT text classifier + reviewer network graph analysis).'
    '</div>'
)
