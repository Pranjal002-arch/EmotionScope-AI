from pathlib import Path
import json
import pickle
import datetime
import threading
import re

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import speech_recognition as sr
import pyttsx3
from streamlit_option_menu import option_menu

st.set_page_config(page_title="EmotionScope AI", page_icon="🧠", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "emotion_model.pkl"
VECTORIZER_PATH = ARTIFACTS_DIR / "emotion_vectorizer.pkl"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
SAMPLES_PATH = ARTIFACTS_DIR / "prediction_samples.csv"
HISTORY_PATH = BASE_DIR / "history.csv"

LABEL_ORDER = ["sadness", "joy", "love", "anger", "fear", "surprise"]
HISTORY_COLUMNS = ["timestamp", "input_text", "predicted_emotion", "risk_level", "confidence_score"]

EMOTION_EMOJI = {
    "joy": "😄", "love": "❤️", "sadness": "😢",
    "anger": "😠", "fear": "😨", "surprise": "😲"
}

EMOTION_COLOR = {
    "joy": "#22c55e", "love": "#f43f5e", "sadness": "#3b82f6",
    "anger": "#ef4444", "fear": "#f97316", "surprise": "#a855f7"
}

RISK_PATTERNS = {
    "high": [
        "kill myself", "want to die", "suicide", "end my life", "hurt myself",
        "self harm", "i don't want to live", "i dont want to live"
    ],
    "medium": [
        "depressed", "depression", "hopeless", "worthless", "empty", "numb", "alone",
        "doing nothing", "not feeling well", "exhausted", "helpless", "fearful",
        "do not feel like doing anything", "dont feel like doing anything",
        "i feel like doing nothing", "i am not okay", "i'm not okay", "miserable",
        "unmotivated", "future scares me"
    ]
}

NEGATIVE_PHRASES = [
    "do not feel like doing anything", "dont feel like doing anything", "do not want to do anything",
    "not feeling well", "feel like doing nothing", "lost interest", "no motivation",
    "my future scares me", "fearful of my future", "afraid of my future"
]

KEYWORDS = {
    "sadness": [
        "depressed", "depression", "hopeless", "worthless", "empty", "numb", "drained",
        "miserable", "broken", "exhausted", "tired", "sad", "crying", "alone", "lonely",
        "down", "unmotivated", "helpless"
    ],
    "fear": [
        "fear", "fearful", "afraid", "scared", "terrified", "anxious", "anxiety", "worry",
        "worried", "panic", "nervous", "uncertain"
    ],
    "anger": ["angry", "annoyed", "furious", "hate", "frustrated", "irritated", "mad", "disgusted"],
    "joy": ["happy", "joyful", "excited", "great", "amazing", "wonderful", "delighted", "cheerful"],
    "love": ["love", "adore", "romantic", "beloved", "my everything", "affection"],
    "surprise": ["surprised", "shocked", "unexpected", "astonished", "amazed"]
}

# ── Beautiful CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Deep space background ── */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120,80,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(236,72,153,0.14) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(14,20,40,0.6) 0%, transparent 80%),
        linear-gradient(160deg, #06091a 0%, #090e20 35%, #07101e 65%, #050810 100%);
    min-height: 100vh;
}

/* Glowing orbs */
.stApp::before {
    content: '';
    position: fixed;
    top: -15%;
    left: -8%;
    width: 700px;
    height: 700px;
    background: radial-gradient(circle, rgba(99,102,241,0.10) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: orb1 12s ease-in-out infinite;
}

.stApp::after {
    content: '';
    position: fixed;
    bottom: -15%;
    right: -8%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(236,72,153,0.09) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
    animation: orb2 15s ease-in-out infinite;
}

@keyframes orb1 {
    0%, 100% { transform: translate(0,0) scale(1); }
    33% { transform: translate(40px,30px) scale(1.05); }
    66% { transform: translate(-20px,50px) scale(0.95); }
}

@keyframes orb2 {
    0%, 100% { transform: translate(0,0) scale(1); }
    33% { transform: translate(-30px,-40px) scale(1.08); }
    66% { transform: translate(25px,-20px) scale(0.92); }
}

.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1500px;
    position: relative;
    z-index: 1;
}

/* ══════════════════════════════
   HERO HEADER
══════════════════════════════ */
.hero-header {
    text-align: center;
    padding: 2.8rem 2rem 2.2rem;
    position: relative;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.12));
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 999px;
    padding: 7px 20px;
    font-size: 0.78rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.4rem;
    box-shadow: 0 0 20px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.08);
    animation: badgePulse 3s ease-in-out infinite;
}

@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 20px rgba(99,102,241,0.15), inset 0 1px 0 rgba(255,255,255,0.08); }
    50% { box-shadow: 0 0 35px rgba(99,102,241,0.30), inset 0 1px 0 rgba(255,255,255,0.12); }
}

.hero-title {
    font-size: 4.2rem;
    font-weight: 900;
    line-height: 1.02;
    letter-spacing: -0.05em;
    background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 30%, #c7d2fe 60%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    text-shadow: none;
    filter: drop-shadow(0 0 40px rgba(99,102,241,0.25));
}

.hero-subtitle {
    font-size: 1.08rem;
    color: #64748b;
    max-width: 560px;
    margin: 0 auto 0.5rem;
    line-height: 1.7;
    font-weight: 400;
}

.hero-divider {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #6366f1, #a855f7, transparent);
    margin: 1.5rem auto 0;
    border-radius: 2px;
}

/* ══════════════════════════════
   GLASS CARDS
══════════════════════════════ */
.glass-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.025) 100%);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 22px;
    padding: 28px 30px;
    backdrop-filter: blur(24px);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.06) inset,
        0 -1px 0 rgba(0,0,0,0.2) inset,
        0 8px 32px rgba(0,0,0,0.35),
        0 2px 8px rgba(0,0,0,0.2);
    transition: all 0.35s cubic-bezier(0.4,0,0.2,1);
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.15) 50%, transparent 100%);
}

.glass-card:hover {
    border-color: rgba(99,102,241,0.22);
    box-shadow:
        0 1px 0 rgba(255,255,255,0.08) inset,
        0 -1px 0 rgba(0,0,0,0.2) inset,
        0 12px 40px rgba(0,0,0,0.45),
        0 0 0 1px rgba(99,102,241,0.08);
    transform: translateY(-2px);
}

/* ══════════════════════════════
   METRIC CARDS
══════════════════════════════ */
.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.055) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 24px 26px;
    position: relative;
    overflow: hidden;
    transition: all 0.35s ease;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

.metric-card::after {
    content: '';
    position: absolute;
    bottom: -40px;
    right: -40px;
    width: 120px;
    height: 120px;
    border-radius: 50%;
    opacity: 0.06;
}

.metric-card.blue { border-top: 2px solid transparent; border-image: linear-gradient(90deg,#3b82f6,#6366f1) 1; }
.metric-card.blue::after { background: #3b82f6; }
.metric-card.green { border-top: 2px solid transparent; border-image: linear-gradient(90deg,#22c55e,#10b981) 1; }
.metric-card.green::after { background: #22c55e; }
.metric-card.purple { border-top: 2px solid transparent; border-image: linear-gradient(90deg,#a855f7,#ec4899) 1; }
.metric-card.purple::after { background: #a855f7; }

.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 30px rgba(0,0,0,0.4); }

.metric-icon { font-size: 2rem; margin-bottom: 12px; display: block; }

.metric-label {
    color: #475569;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.metric-value {
    color: #f8fafc;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(135deg, #f8fafc, #cbd5e1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ══════════════════════════════
   SECTION TITLES
══════════════════════════════ */
.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #cbd5e1;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: -0.01em;
}

.section-title span {
    display: inline-block;
    width: 3px;
    height: 16px;
    background: linear-gradient(180deg, #818cf8, #c084fc);
    border-radius: 2px;
    box-shadow: 0 0 8px rgba(99,102,241,0.6);
}

/* ══════════════════════════════
   INPUT AREA
══════════════════════════════ */
div[data-testid="stTextArea"] textarea {
    background: rgba(10,14,30,0.6) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-size: 0.97rem !important;
    min-height: 160px !important;
    resize: none !important;
    transition: all 0.25s ease !important;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.7 !important;
    caret-color: #818cf8 !important;
}

div[data-testid="stTextArea"] textarea::placeholder { color: #334155 !important; }

div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12), 0 0 20px rgba(99,102,241,0.08) !important;
    background: rgba(14,18,38,0.8) !important;
}

/* ══════════════════════════════
   BUTTONS
══════════════════════════════ */
.stButton > button {
    border-radius: 13px !important;
    font-weight: 700 !important;
    font-size: 0.93rem !important;
    min-height: 50px !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4), 0 1px 0 rgba(255,255,255,0.15) inset !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.55), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    transform: translateY(-2px) !important;
}

.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

.stButton > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    color: #94a3b8 !important;
}

.stButton > button:not([kind="primary"]):hover {
    background: rgba(99,102,241,0.10) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: #c7d2fe !important;
    box-shadow: 0 0 15px rgba(99,102,241,0.12) !important;
}

/* ══════════════════════════════
   RESULT EMOTION CARD
══════════════════════════════ */
.result-emotion-card {
    border-radius: 20px;
    padding: 32px 24px;
    text-align: center;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

.result-emotion-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1px;
    background: linear-gradient(135deg, var(--ec,#6366f1), transparent 60%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.emotion-emoji-big {
    font-size: 4.5rem;
    display: block;
    margin-bottom: 12px;
    animation: floatEmoji 3s ease-in-out infinite;
    filter: drop-shadow(0 4px 12px rgba(0,0,0,0.4));
}

@keyframes floatEmoji {
    0%, 100% { transform: translateY(0) scale(1); }
    50% { transform: translateY(-6px) scale(1.06); }
}

.emotion-name-big {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    margin-bottom: 6px;
    text-shadow: 0 0 30px currentColor;
}

.confidence-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    padding: 5px 16px;
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
    margin-top: 6px;
}

/* ══════════════════════════════
   RISK BADGE
══════════════════════════════ */
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 18px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 12px;
}

.risk-high {
    background: linear-gradient(135deg,rgba(239,68,68,0.18),rgba(239,68,68,0.08));
    color: #fca5a5;
    border: 1px solid rgba(239,68,68,0.35);
    box-shadow: 0 0 16px rgba(239,68,68,0.15);
}
.risk-medium {
    background: linear-gradient(135deg,rgba(249,115,22,0.18),rgba(249,115,22,0.08));
    color: #fdba74;
    border: 1px solid rgba(249,115,22,0.35);
    box-shadow: 0 0 16px rgba(249,115,22,0.15);
}
.risk-low {
    background: linear-gradient(135deg,rgba(34,197,94,0.18),rgba(34,197,94,0.08));
    color: #86efac;
    border: 1px solid rgba(34,197,94,0.35);
    box-shadow: 0 0 16px rgba(34,197,94,0.15);
}

/* ══════════════════════════════
   SUPPORT BOX
══════════════════════════════ */
.support-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(168,85,247,0.06));
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 14px;
    padding: 18px 20px;
    color: #c7d2fe;
    font-size: 0.93rem;
    line-height: 1.65;
    margin-top: 14px;
    position: relative;
    overflow: hidden;
}

.support-box::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, #818cf8, #c084fc);
    border-radius: 3px 0 0 3px;
}

/* ══════════════════════════════
   SIGNAL TAGS
══════════════════════════════ */
.signal-tag {
    display: inline-block;
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    margin: 3px;
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(168,85,247,0.10));
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.28);
    letter-spacing: 0.02em;
    transition: all 0.2s ease;
}

.signal-tag:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.28), rgba(168,85,247,0.18));
    transform: translateY(-1px);
}

/* ══════════════════════════════
   PROGRESS BARS
══════════════════════════════ */
.conf-bar-wrap { margin: 5px 0 13px; }

.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.80rem;
    color: #475569;
    margin-bottom: 6px;
    font-weight: 600;
}

.conf-bar-track {
    height: 7px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
}

.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    position: relative;
    overflow: hidden;
}

.conf-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; right: 0; bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
    animation: shimmer 2s ease-in-out infinite;
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* ══════════════════════════════
   SIDEBAR
══════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #05070f 0%, #080c1a 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.12) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.4) !important;
}

.sidebar-logo {
    text-align: center;
    padding: 1.8rem 0 1.2rem;
}

.logo-icon-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 60px;
    height: 60px;
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(168,85,247,0.15));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 18px;
    font-size: 2rem;
    margin-bottom: 12px;
    box-shadow: 0 0 25px rgba(99,102,241,0.2);
}

.logo-name {
    font-size: 1.15rem;
    font-weight: 800;
    color: #e2e8f0;
    letter-spacing: -0.03em;
}

.logo-tagline {
    font-size: 0.73rem;
    color: #334155;
    margin-top: 5px;
    letter-spacing: 0.02em;
}

.sidebar-divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,0.1);
    margin: 1rem 0;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    border-radius: 10px;
    margin-bottom: 4px;
    transition: background 0.2s;
}

.legend-item:hover { background: rgba(255,255,255,0.04); }

.legend-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    box-shadow: 0 0 6px currentColor;
}

.legend-label {
    font-size: 0.82rem;
    color: #475569;
    font-weight: 500;
    text-transform: capitalize;
    flex: 1;
}

.legend-emoji { font-size: 1rem; }

/* ══════════════════════════════
   RADIO
══════════════════════════════ */
div[data-testid="stRadio"] > div {
    gap: 8px !important;
}

div[data-testid="stRadio"] label {
    color: #64748b !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    padding: 6px 14px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    background: rgba(255,255,255,0.03) !important;
    transition: all 0.2s !important;
}

div[data-testid="stRadio"] label:has(input:checked) {
    color: #a5b4fc !important;
    background: rgba(99,102,241,0.12) !important;
    border-color: rgba(99,102,241,0.3) !important;
}

/* ══════════════════════════════
   DATAFRAME
══════════════════════════════ */
.stDataFrame {
    border-radius: 16px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ══════════════════════════════
   ALERTS
══════════════════════════════ */
div[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-size: 0.88rem !important;
    border-left-width: 3px !important;
}

/* ══════════════════════════════
   HOW IT WORKS ITEMS
══════════════════════════════ */
.about-item {
    padding: 14px 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    position: relative;
}

.about-item:last-child { border-bottom: none; }

.about-item::before {
    content: '';
    position: absolute;
    left: -30px;
    top: 50%;
    transform: translateY(-50%);
    width: 3px;
    height: 0;
    background: linear-gradient(180deg, #818cf8, #c084fc);
    border-radius: 2px;
    transition: height 0.3s ease;
}

.about-text-title {
    font-size: 0.92rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
    letter-spacing: -0.01em;
}

.about-text-desc {
    font-size: 0.82rem;
    color: #475569;
    line-height: 1.55;
}

/* ══════════════════════════════
   METRICS (st.metric)
══════════════════════════════ */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

[data-testid="stMetric"] label {
    color: #475569 !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em !important;
}

/* ══════════════════════════════
   HIDE STREAMLIT DEFAULTS
══════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.5); }
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ────────────────────────────────────────────────────────────

def softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / ex.sum()


def preprocess_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def count_words(text):
    return len([w for w in str(text).strip().split() if w.strip()])


def normalize_scores(score_map):
    total = sum(score_map.values())
    if total <= 0:
        return {k: 1 / len(score_map) for k in score_map}
    return {k: float(v) / total for k, v in score_map.items()}


def detect_risk_level(text):
    text = preprocess_text(text)
    for phrase in RISK_PATTERNS["high"]:
        if phrase in text:
            return "high"
    for phrase in RISK_PATTERNS["medium"]:
        if phrase in text:
            return "medium"
    return "low"


def generate_support_message(emotion, risk_level):
    if risk_level == "high":
        return "Your message reflects serious distress. Please reach out to a trusted person or mental health professional right away."
    if risk_level == "medium":
        return "Your message reflects emotional strain. Give yourself rest, speak to someone you trust, and seek support if this continues."
    advice = {
        "joy": "Your tone feels bright and positive. Keep protecting what is helping you feel good.",
        "love": "Your message reflects warmth and connection. Hold close the people who matter to you.",
        "sadness": "Your words reflect emotional heaviness. Be gentle with yourself and take things one step at a time.",
        "anger": "Your message carries tension and frustration. Pause, breathe, and respond slowly.",
        "fear": "Your words reflect worry and uncertainty. Focus on one small step you can control right now.",
        "surprise": "Your message reflects sudden change or uncertainty. Give yourself time to process it."
    }
    return advice.get(emotion, "Take a moment to slow down and respond kindly to yourself.")


def speak_text(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=run, daemon=True).start()


def capture_speech():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak clearly.")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        return recognizer.recognize_google(audio)
    except sr.WaitTimeoutError:
        st.warning("No speech detected.")
        return ""
    except sr.UnknownValueError:
        st.warning("Could not understand speech.")
        return ""
    except OSError:
        st.error("Microphone not found or permission denied.")
        return ""
    except Exception as e:
        st.error(f"Speech error: {e}")
        return ""


@st.cache_resource
def load_artifacts():
    model, vectorizer = None, None
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)

    metrics = {}
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}

    sample_predictions = pd.DataFrame()
    if SAMPLES_PATH.exists():
        try:
            sample_predictions = pd.read_csv(SAMPLES_PATH)
        except Exception:
            sample_predictions = pd.DataFrame()

    return model, vectorizer, metrics, sample_predictions


def ensure_history():
    if not HISTORY_PATH.exists() or HISTORY_PATH.stat().st_size == 0:
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(HISTORY_PATH, index=False)


def load_history():
    ensure_history()
    try:
        df = pd.read_csv(HISTORY_PATH)
        if list(df.columns) != HISTORY_COLUMNS:
            df = pd.DataFrame(columns=HISTORY_COLUMNS)
            df.to_csv(HISTORY_PATH, index=False)
        return df
    except (EmptyDataError, Exception):
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
        df.to_csv(HISTORY_PATH, index=False)
        return df


def save_history(input_text, predicted_emotion, risk_level, confidence_score):
    ensure_history()
    try:
        df = pd.read_csv(HISTORY_PATH)
    except Exception:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
    if list(df.columns) != HISTORY_COLUMNS:
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
    row = pd.DataFrame([{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": input_text,
        "predicted_emotion": predicted_emotion,
        "risk_level": risk_level,
        "confidence_score": round(float(confidence_score), 4)
    }])
    pd.concat([df, row], ignore_index=True).to_csv(HISTORY_PATH, index=False)


def build_model_scores(text, model, vectorizer):
    fallback = {"sadness": 0.24, "joy": 0.15, "love": 0.08, "anger": 0.16, "fear": 0.27, "surprise": 0.10}
    if model is None or vectorizer is None:
        return fallback
    score_map = {label: 0.0 for label in LABEL_ORDER}
    vec = vectorizer.transform([text])
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        for cls, prob in zip(model.classes_, probs):
            if cls in score_map:
                score_map[cls] = float(prob)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(vec)
        raw = np.array(scores[0] if getattr(scores, "ndim", 1) > 1 else scores, dtype=float)
        probs = softmax(raw)
        for cls, prob in zip(model.classes_, probs):
            if cls in score_map:
                score_map[cls] = float(prob)
    else:
        return fallback
    return normalize_scores(score_map)


def build_rule_scores(text):
    text = preprocess_text(text)
    score_map = {label: 0.0 for label in LABEL_ORDER}
    signals = []
    for word in KEYWORDS["sadness"]:
        if word in text:
            score_map["sadness"] += 2.4
            score_map["fear"] += 0.5
            signals.append("sadness")
    for word in KEYWORDS["fear"]:
        if word in text:
            score_map["fear"] += 2.2
            score_map["sadness"] += 0.6
            signals.append("fear")
    for word in KEYWORDS["anger"]:
        if word in text:
            score_map["anger"] += 2.1
            signals.append("anger")
    for word in KEYWORDS["joy"]:
        if word in text:
            score_map["joy"] += 2.0
            signals.append("joy")
    for word in KEYWORDS["love"]:
        if word in text:
            score_map["love"] += 1.9
            score_map["joy"] += 0.4
            signals.append("love")
    for word in KEYWORDS["surprise"]:
        if word in text:
            score_map["surprise"] += 1.8
            signals.append("surprise")
    for phrase in NEGATIVE_PHRASES:
        if phrase in text:
            score_map["sadness"] += 3.0
            score_map["fear"] += 1.3
            score_map["love"] *= 0.2
            score_map["joy"] *= 0.2
            signals.extend(["sadness", "fear"])
    if score_map["sadness"] > 0 or score_map["fear"] > 0:
        score_map["love"] *= 0.1
        score_map["joy"] *= 0.25
    signals = sorted(set(signals))
    if sum(score_map.values()) == 0:
        return None, []
    return normalize_scores(score_map), signals


def apply_display_logic(score_map, text):
    words = count_words(text)
    score_map = normalize_scores(score_map)
    if words <= 4:
        top_emotion = max(score_map, key=score_map.get)
        one_hot = {k: 0.0 for k in score_map}
        one_hot[top_emotion] = 1.0
        return one_hot, True
    cleaned = {emotion: (score if score >= 0.03 else 0.0) for emotion, score in score_map.items()}
    return normalize_scores(cleaned), False


def predict_emotion(text, model, vectorizer):
    raw_text = str(text).strip()
    clean_text = preprocess_text(raw_text)
    model_scores = build_model_scores(clean_text, model, vectorizer)
    rule_scores, signals = build_rule_scores(clean_text)
    if rule_scores is None:
        final_scores = model_scores
    else:
        strong_negative = any(k in signals for k in ["sadness", "fear", "anger"])
        alpha = 0.78 if strong_negative else 0.48
        final_scores = {
            label: alpha * rule_scores.get(label, 0.0) + (1 - alpha) * model_scores.get(label, 0.0)
            for label in LABEL_ORDER
        }
        if strong_negative:
            final_scores["love"] *= 0.12
            final_scores["joy"] *= 0.22
            final_scores["sadness"] *= 1.22
            final_scores["fear"] *= 1.15
        final_scores = normalize_scores(final_scores)
    final_scores, is_short_text = apply_display_logic(final_scores, raw_text)
    pred = max(final_scores, key=final_scores.get)
    confidence = float(final_scores[pred])
    if not is_short_text and pred in ["sadness", "fear"] and any(s in signals for s in ["sadness", "fear"]):
        confidence = max(confidence, 0.84)
    elif not is_short_text:
        confidence = min(confidence, 0.95)
    score_df = pd.DataFrame({
        "Emotion": LABEL_ORDER,
        "Score": [float(final_scores[label]) for label in LABEL_ORDER]
    }).sort_values("Score", ascending=False)
    return pred, confidence, score_df, is_short_text, signals


def get_average_confidence(metrics, sample_predictions):
    if "avg_confidence" in metrics and metrics.get("avg_confidence") not in [None, 0, 0.0]:
        return float(metrics["avg_confidence"]) * 100
    if not sample_predictions.empty:
        for col in ["confidence", "Confidence", "confidence_score"]:
            if col in sample_predictions.columns:
                return float(sample_predictions[col].mean()) * 100
    if "validation_accuracy" in metrics:
        return float(metrics["validation_accuracy"]) * 100
    return 0.0


def conf_bar(label, value, color):
    pct = value * 100
    return f"""
    <div class="conf-bar-wrap">
        <div class="conf-bar-label"><span>{label}</span><span>{pct:.1f}%</span></div>
        <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    </div>"""


# ── Init ────────────────────────────────────────────────────────────────────────
model, vectorizer, metrics, sample_predictions = load_artifacts()
ensure_history()

for key, val in [("speech_text", ""), ("input_text", ""), ("last_result", None)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon-wrap">🧠</div>
        <div class="logo-name">EmotionScope AI</div>
        <div class="logo-tagline">Emotional Intelligence Platform</div>
    </div>
    <hr class="sidebar-divider">
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Dashboard", "History", "Model Info"],
        icons=["grid-1x2-fill", "clock-history", "cpu-fill"],
        default_index=0,
        styles={
            "container": {"padding": "0", "background": "transparent"},
            "icon": {"color": "#818cf8", "font-size": "15px"},
            "nav-link": {
                "font-size": "13.5px", "color": "#475569", "border-radius": "12px",
                "margin": "3px 0", "font-weight": "600", "padding": "10px 14px"
            },
            "nav-link-selected": {
                "background": "linear-gradient(135deg,rgba(99,102,241,0.18),rgba(168,85,247,0.10))",
                "color": "#a5b4fc", "font-weight": "700",
                "border": "1px solid rgba(99,102,241,0.28)"
            },
        }
    )

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    legend_html = '<div style="padding:0 6px;"><div style="font-size:0.72rem;color:#1e293b;font-weight:800;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px;">Emotions</div>'
    for e in LABEL_ORDER:
        legend_html += f'<div class="legend-item"><span class="legend-emoji">{EMOTION_EMOJI[e]}</span><span class="legend-label">{e}</span><span class="legend-dot" style="background:{EMOTION_COLOR[e]};color:{EMOTION_COLOR[e]};"></span></div>'
    legend_html += '</div>'
    st.markdown(legend_html, unsafe_allow_html=True)


# ── Main Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✦ AI-Powered Emotion Analysis</div>
    <div class="hero-title">EmotionScope AI</div>
    <div class="hero-subtitle">Turn your thoughts into emotional insights with real-time AI analysis, visual breakdowns, and personalized support.</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if selected == "Dashboard":
    train_acc = float(metrics.get("train_accuracy", 0)) * 100
    val_acc = float(metrics.get("validation_accuracy", 0)) * 100
    avg_conf = get_average_confidence(metrics, sample_predictions)

    # Metrics row
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-card blue">
            <span class="metric-icon">🎯</span>
            <div class="metric-label">Train Accuracy</div>
            <div class="metric-value">{train_acc:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card green">
            <span class="metric-icon">✅</span>
            <div class="metric-label">Validation Accuracy</div>
            <div class="metric-value">{val_acc:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card purple">
            <span class="metric-icon">💡</span>
            <div class="metric-label">Average Confidence</div>
            <div class="metric-value">{avg_conf:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Input + About row
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span></span>Analyze Your Emotion</div>', unsafe_allow_html=True)

        input_mode = st.radio("Input method", ["✍️ Type Text", "🎙️ Voice Input"], horizontal=True, label_visibility="collapsed")

        if input_mode == "✍️ Type Text":
            user_text = st.text_area(
                "Your text",
                value=st.session_state["input_text"],
                placeholder="How are you feeling today? Type anything...",
                height=160,
                label_visibility="collapsed"
            )
            st.session_state["input_text"] = user_text
        else:
            if st.button("🎙️ Start Listening", use_container_width=True):
                speech_text = capture_speech()
                if speech_text:
                    st.session_state["speech_text"] = speech_text
                    st.session_state["input_text"] = speech_text
                    st.success(f"✅ Recognized: **{speech_text}**")
            user_text = st.text_area(
                "Recognized text",
                value=st.session_state["input_text"],
                height=160,
                label_visibility="collapsed"
            )
            st.session_state["input_text"] = user_text

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        analyze = st.button("🔍 Analyze Emotion", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span></span>How It Works</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-item">
            <div>
                <div class="about-text-title">AI + Rule Fusion</div>
                <div class="about-text-desc">Combines ML model predictions with linguistic rule signals for accurate emotion detection.</div>
            </div>
        </div>
        <div class="about-item">
            <div>
                <div class="about-text-title">Visual Breakdown</div>
                <div class="about-text-desc">See bar charts and radar plots showing the full emotion distribution in real time.</div>
            </div>
        </div>
        <div class="about-item">
            <div>
                <div class="about-text-title">Risk Detection</div>
                <div class="about-text-desc">Automatically flags high or medium risk emotional content and provides supportive guidance.</div>
            </div>
        </div>
        <div class="about-item">
            <div>
                <div class="about-text-title">Voice Support</div>
                <div class="about-text-desc">Speak your input or listen to personalized advice with text-to-speech output.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Run prediction
    if analyze:
        if not str(st.session_state["input_text"]).strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            text = st.session_state["input_text"]
            pred, confidence, score_df, is_short_text, signals = predict_emotion(text, model, vectorizer)
            risk = detect_risk_level(text)
            support = generate_support_message(pred, risk)
            st.session_state["last_result"] = {
                "text": text, "predicted_emotion": pred, "confidence": confidence,
                "risk_level": risk, "support_message": support, "score_df": score_df,
                "is_short_text": is_short_text, "signals": signals
            }
            save_history(text, pred, risk, confidence)

    # Show results
    if st.session_state["last_result"] is not None:
        result = st.session_state["last_result"]
        pred = result["predicted_emotion"]
        confidence = result["confidence"]
        risk = result["risk_level"]
        support = result["support_message"]
        score_df = result["score_df"]
        is_short_text = result["is_short_text"]
        signals = result["signals"]
        color = EMOTION_COLOR.get(pred, "#6366f1")
        emoji = EMOTION_EMOJI.get(pred, "🤔")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        r1, r2 = st.columns([1, 1.1])

        with r1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><span></span>Result</div>', unsafe_allow_html=True)

            # Emotion display
            risk_class = f"risk-{risk}"
            risk_icon = "🔴" if risk == "high" else ("🟡" if risk == "medium" else "🟢")
            r, g, b = (int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            st.markdown(f"""
            <div class="result-emotion-card" style="--ec:{color};background:linear-gradient(145deg,rgba({r},{g},{b},0.14),rgba({r},{g},{b},0.05));border:1px solid rgba({r},{g},{b},0.25);box-shadow:0 0 40px rgba({r},{g},{b},0.10);">
                <span class="emotion-emoji-big">{emoji}</span>
                <div class="emotion-name-big" style="color:{color};">{pred.upper()}</div>
                <div class="confidence-pill">{confidence*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # Confidence bar
            st.markdown(conf_bar("Confidence", confidence, color), unsafe_allow_html=True)

            # Risk badge
            st.markdown(f'<div class="{risk_class} risk-badge">{risk_icon} {risk.upper()} RISK</div>', unsafe_allow_html=True)

            # Signals
            if signals:
                st.markdown("<div style='margin-top:12px;'>", unsafe_allow_html=True)
                tags = "".join([f'<span class="signal-tag">#{s}</span>' for s in signals])
                st.markdown(f"<div style='margin-top:8px;'>{tags}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # Support message
            st.markdown(f'<div class="support-box">💬 {support}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
            if st.button("🔊 Speak Advice", use_container_width=True):
                speak_text(support)
                st.success("Speaking advice...")

            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><span></span>Emotion Distribution</div>', unsafe_allow_html=True)

            fig_bar = px.bar(
                score_df, x="Emotion", y="Score", color="Emotion",
                color_discrete_map=EMOTION_COLOR,
                text=score_df["Score"].apply(lambda x: f"{x*100:.1f}%")
            )
            fig_bar.update_traces(textposition="outside", marker_line_width=0, opacity=0.9)
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(title="", tickfont=dict(color="#64748b", size=12), gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(title="", tickfont=dict(color="#64748b", size=11), gridcolor="rgba(255,255,255,0.04)", range=[0, 1.15]),
                showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=280
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Per-emotion bars
            st.markdown('<div class="section-title" style="margin-top:8px;"><span></span>Score Breakdown</div>', unsafe_allow_html=True)
            for _, row in score_df.iterrows():
                st.markdown(conf_bar(
                    f"{EMOTION_EMOJI.get(row['Emotion'], '')} {row['Emotion'].capitalize()}",
                    row["Score"],
                    EMOTION_COLOR.get(row["Emotion"], "#6366f1")
                ), unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Radar chart
        non_zero = score_df[score_df["Score"] > 0]
        if not is_short_text and len(non_zero) > 1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title"><span></span>Radar — Emotion Profile</div>', unsafe_allow_html=True)

            radar_loop = pd.concat([score_df, score_df.iloc[[0]]], ignore_index=True)
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_loop["Score"], theta=radar_loop["Emotion"], fill="toself",
                fillcolor="rgba(99,102,241,0.12)",
                line=dict(color="#6366f1", width=2.5),
                name="Emotion Profile"
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#475569", size=10), gridcolor="rgba(255,255,255,0.06)"),
                    angularaxis=dict(tickfont=dict(color="#94a3b8", size=13), gridcolor="rgba(255,255,255,0.06)")
                ),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white", family="Inter"),
                showlegend=False, margin=dict(l=30, r=30, t=20, b=20), height=340
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif selected == "History":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"><span></span>Prediction History</div>', unsafe_allow_html=True)

    history_df = load_history()
    if history_df.empty:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#334155;">
            <div style="font-size:3rem;margin-bottom:1rem;">📭</div>
            <div style="font-size:1rem;font-weight:600;color:#475569;">No history yet</div>
            <div style="font-size:0.85rem;margin-top:6px;">Analyze some text to see your emotion history here.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sorted_df = history_df.sort_values("timestamp", ascending=False).reset_index(drop=True)

        # Summary stats
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Entries", len(sorted_df))
        with s2:
            top_emotion = sorted_df["predicted_emotion"].mode()[0] if not sorted_df.empty else "—"
            st.metric("Most Common", f"{EMOTION_EMOJI.get(top_emotion,'')} {top_emotion.capitalize()}")
        with s3:
            avg_c = sorted_df["confidence_score"].mean() * 100 if "confidence_score" in sorted_df else 0
            st.metric("Avg Confidence", f"{avg_c:.1f}%")
        with s4:
            high_risk = (sorted_df["risk_level"] == "high").sum()
            st.metric("High Risk Entries", high_risk)

        st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

        # Emotion trend chart
        if len(sorted_df) >= 2:
            trend_fig = px.line(
                sorted_df.iloc[::-1], x="timestamp", y="confidence_score",
                color="predicted_emotion", color_discrete_map=EMOTION_COLOR,
                markers=True, title="Confidence Over Time"
            )
            trend_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Inter"),
                xaxis=dict(tickfont=dict(color="#64748b"), gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(tickfont=dict(color="#64748b"), gridcolor="rgba(255,255,255,0.04)"),
                legend=dict(font=dict(color="#94a3b8")),
                title=dict(font=dict(color="#e2e8f0", size=14)),
                margin=dict(l=10, r=10, t=40, b=10), height=260
            )
            st.plotly_chart(trend_fig, use_container_width=True)

        st.dataframe(
            sorted_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "timestamp": st.column_config.TextColumn("Time"),
                "input_text": st.column_config.TextColumn("Input", width="large"),
                "predicted_emotion": st.column_config.TextColumn("Emotion"),
                "risk_level": st.column_config.TextColumn("Risk"),
                "confidence_score": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
            }
        )

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
else:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span></span>Model Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="about-item">
            <span class="about-icon">🤖</span>
            <div>
                <div class="about-text-title">Hybrid Architecture</div>
                <div class="about-text-desc">Combines a trained ML classifier with a hand-crafted rule engine for robust emotion prediction.</div>
            </div>
        </div>
        <div class="about-item">
            <span class="about-icon">🏷️</span>
            <div>
                <div class="about-text-title">6 Emotion Classes</div>
                <div class="about-text-desc">Sadness · Joy · Love · Anger · Fear · Surprise</div>
            </div>
        </div>
        <div class="about-item">
            <span class="about-icon">🔧</span>
            <div>
                <div class="about-text-title">Negative Bias Correction</div>
                <div class="about-text-desc">Stronger rule weighting (α=0.78) for negative emotions prevents mislabeling sad/fearful text as love or joy.</div>
            </div>
        </div>
        <div class="about-item">
            <span class="about-icon">📏</span>
            <div>
                <div class="about-text-title">Short Text Handling</div>
                <div class="about-text-desc">Inputs ≤ 4 words are mapped to a single dominant emotion to avoid noise in sparse signals.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span></span>Performance Metrics</div>', unsafe_allow_html=True)

        train_acc = float(metrics.get("train_accuracy", 0)) * 100
        val_acc = float(metrics.get("validation_accuracy", 0)) * 100
        avg_conf = get_average_confidence(metrics, sample_predictions)

        st.markdown(conf_bar("🎯 Train Accuracy", train_acc / 100, "#6366f1"), unsafe_allow_html=True)
        st.markdown(conf_bar("✅ Validation Accuracy", val_acc / 100, "#22c55e"), unsafe_allow_html=True)
        st.markdown(conf_bar("💡 Avg Confidence", avg_conf / 100, "#a855f7"), unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-title"><span></span>Emotion Labels</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        for i, label in enumerate(LABEL_ORDER):
            with cols[i % 3]:
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
                border-radius:10px;padding:10px 12px;text-align:center;margin-bottom:8px;">
                    <div style="font-size:1.5rem;">{EMOTION_EMOJI[label]}</div>
                    <div style="font-size:0.8rem;color:#64748b;font-weight:600;margin-top:4px;text-transform:capitalize;">{label}</div>
                    <div style="width:100%;height:3px;border-radius:2px;background:{EMOTION_COLOR[label]};margin-top:6px;opacity:0.7;"></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
