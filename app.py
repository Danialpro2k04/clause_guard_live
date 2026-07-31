"""
ClauseGuard — Live Multi-Tenant Compliance Review (BYOK)
------------------------------------------------------------
A public, session-isolated deployment of the ClauseGuard pipeline for
Streamlit Community Cloud / Hugging Face Spaces.

Every visitor brings their own Groq API key, uploads up to 10 policy
documents, and gets an isolated in-memory Qdrant index + HITL review queue
that lives only for their browser session. Nothing is ever written to
server disk — no `/corpus`, no `pending_reviews.json`, no shared Qdrant path.

Pipeline (unchanged conceptually from the original repo):
    Intake Agent -> Retrieval Agent (Qdrant) -> Risk-Scoring Agent -> HITL

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import groq
import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from pipeline import extract_text_from_upload, run_compliance_pipeline
from vector_store import DEFAULT_COLLECTION_NAME, ingest_policy_files


MAX_POLICY_FILES = 10
COLLECTION_NAME = DEFAULT_COLLECTION_NAME
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GITHUB_REPO_URL = "https://github.com/Danialpro2k04/clause_guard_live"

RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

RISK_STYLE = {
    "HIGH": {"color": "#FF4757", "bg": "rgba(255,71,87,0.12)", "border": "#FF4757", "icon": "⬆"},
    "MEDIUM": {"color": "#FFA502", "bg": "rgba(255,165,2,0.12)", "border": "#FFA502", "icon": "◆"},
    "LOW": {"color": "#2ED573", "bg": "rgba(46,213,115,0.12)", "border": "#2ED573", "icon": "⬇"},
}

STATUS_STYLE = {
    "PENDING": {"color": "#A8B2D8", "bg": "rgba(168,178,216,0.12)"},
    "RESOLVED": {"color": "#64FFDA", "bg": "rgba(100,255,218,0.12)"},
}

DECISION_STYLE = {
    "APPROVED": {"color": "#2ED573", "bg": "rgba(46,213,115,0.12)", "icon": "✓"},
    "REJECTED": {"color": "#FF4757", "bg": "rgba(255,71,87,0.12)", "icon": "✕"},
    "MODIFIED": {"color": "#A78BFA", "bg": "rgba(167,139,250,0.12)", "icon": "⟳"},
}

SESSION_DEFAULTS = {
    "groq_api_key": "",
    "qdrant_client": None,
    "policy_stats": None,
    "pending_reviews": [],
    "reviewer_name": "",
    "flash": None,
    "contracts_audited": 0,
}


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------


def init_session_state() -> None:
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.qdrant_client is None:
        st.session_state.qdrant_client = QdrantClient(":memory:")


def reset_session() -> None:
    for key in SESSION_DEFAULTS:
        st.session_state.pop(key, None)
    for key in list(st.session_state.keys()):
        if key.startswith(("notes__", "show_modify__", "newlevel__")):
            st.session_state.pop(key, None)
    init_session_state()


def get_groq_client() -> groq.Groq | None:
    key = st.session_state.get("groq_api_key", "").strip()
    if not key:
        return None
    return groq.Groq(api_key=key)


def flash(kind: str, message: str) -> None:
    st.session_state.flash = (kind, message)


# --------------------------------------------------------------------------
# HITL record helpers
# --------------------------------------------------------------------------


def make_review_id(record: dict[str, Any]) -> str:
    basis = f"{record.get('contract_name', '')}||{record.get('clause_text', '')}"
    digest = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]
    return f"rev_{digest}"


def push_flagged_findings(final_report: dict[str, Any]) -> int:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    existing_ids = {r["review_id"] for r in st.session_state.pending_reviews}
    added = 0

    for clause in final_report.get("evaluations", []):
        if clause.get("risk_level") not in ("HIGH", "MEDIUM"):
            continue
        record = {
            "contract_name": final_report.get("contract_name", "Unknown Contract"),
            "clause_text": clause.get("clause_text", ""),
            "risk_level": clause.get("risk_level", "MEDIUM"),
            "justification": clause.get("justification", ""),
            "recommendation": clause.get("recommendation", ""),
            "status": "PENDING",
            "timestamp": now,
            "human_decision": None,
            "human_notes": "",
            "reviewed_by": None,
            "reviewed_at": None,
        }
        record["review_id"] = make_review_id(record)
        if record["review_id"] in existing_ids:
            continue
        st.session_state.pending_reviews.append(record)
        existing_ids.add(record["review_id"])
        added += 1

    return added


def commit_decision(review_id: str, decision: str, notes: str, reviewer: str) -> None:
    if not reviewer.strip():
        flash("warning", "Enter your name above before recording a decision — it's part of the audit trail.")
        st.rerun()
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in st.session_state.pending_reviews:
        if r["review_id"] == review_id:
            r["human_decision"] = decision
            r["human_notes"] = notes.strip()
            r["status"] = "RESOLVED"
            r["reviewed_by"] = reviewer.strip()
            r["reviewed_at"] = now
    flash("success", f"Recorded **{decision}** for `{review_id}`.")
    st.rerun()


def modify_risk_level(review_id: str, new_level: str, notes: str, reviewer: str) -> None:
    if not reviewer.strip():
        flash("warning", "Enter your name above before recording a decision — it's part of the audit trail.")
        st.rerun()
        return

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in st.session_state.pending_reviews:
        if r["review_id"] == review_id:
            r.setdefault("original_risk_level", r["risk_level"])
            r["risk_level"] = new_level
            r["human_decision"] = "MODIFIED"
            r["human_notes"] = notes.strip()
            r["status"] = "RESOLVED"
            r["reviewed_by"] = reviewer.strip()
            r["reviewed_at"] = now
    flash("success", f"Risk level for `{review_id}` updated to **{new_level}**.")
    st.rerun()


def filter_and_sort(records: list[dict[str, Any]], filters: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(records)

    if filters["status_filter"] == "Pending only":
        rows = [r for r in rows if r["status"] == "PENDING"]
    elif filters["status_filter"] == "Resolved only":
        rows = [r for r in rows if r["status"] != "PENDING"]

    if filters["risk_filter"]:
        rows = [r for r in rows if r["risk_level"] in filters["risk_filter"]]

    if filters["contract_filter"]:
        rows = [r for r in rows if r["contract_name"] in filters["contract_filter"]]

    if filters["search"]:
        q = filters["search"]
        rows = [
            r for r in rows
            if q in r["clause_text"].lower() or q in r["justification"].lower()
        ]

    if filters["sort_by"] == "Risk (High → Low)":
        rows.sort(key=lambda r: (RISK_ORDER.get(r["risk_level"], 99), r["contract_name"]))
    else:
        rows.sort(key=lambda r: (r["contract_name"], RISK_ORDER.get(r["risk_level"], 99)))

    return rows


def records_to_markdown(records: list[dict[str, Any]]) -> str:
    lines = ["# ClauseGuard Audit Report", ""]
    for r in records:
        lines.append(f"## {r['contract_name']} — {r['risk_level']}")
        lines.append(f"- **Status:** {r['status']}")
        lines.append(f"- **Clause:** {r['clause_text']}")
        lines.append(f"- **Justification:** {r['justification']}")
        if r.get("recommendation"):
            lines.append(f"- **Recommendation:** {r['recommendation']}")
        if r.get("human_decision"):
            lines.append(
                f"- **Reviewer decision:** {r['human_decision']} "
                f"by {r.get('reviewed_by', '—')} on {r.get('reviewed_at', '—')}"
            )
        if r.get("human_notes"):
            lines.append(f"- **Reviewer notes:** {r['human_notes']}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Design System — CSS
# --------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg-base:       #0A0E1A;
    --bg-surface:    #0F1629;
    --bg-card:       #141B2D;
    --bg-card-hover: #1A2238;
    --bg-elevated:   #1E2840;
    --border:        rgba(100,120,200,0.14);
    --border-strong: rgba(100,120,200,0.28);
    --accent:        #4F8EF7;
    --accent-dim:    rgba(79,142,247,0.15);
    --accent-glow:   rgba(79,142,247,0.35);
    --text-primary:  #E8EDF8;
    --text-secondary:#8892B0;
    --text-muted:    #546079;
    --font-main:     'Inter', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     14px;
    --radius-xl:     18px;
    --shadow-card:   0 4px 24px rgba(0,0,0,0.45);
    --shadow-glow:   0 0 32px rgba(79,142,247,0.18);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    color: var(--text-primary) !important;
}

.stApp {
    background: var(--bg-base) !important;
}

/* ── Main container ── */
.block-container {
    padding-top: 1.75rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1160px !important;
}

/* ── Hide default Streamlit header chrome ── */
header[data-testid="stHeader"] { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }

section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

/* ── Headings ── */
h1 { font-size: 1.9rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; letter-spacing: -0.01em !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; }

/* ── Tabs ── */
div[data-testid="stTabs"] > div:first-child {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 4px !important;
    gap: 2px !important;
    margin-bottom: 1.5rem !important;
}

button[data-testid="stTab"] {
    border-radius: var(--radius-md) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    padding: 0.55rem 1.1rem !important;
    border: none !important;
    transition: all 0.18s ease !important;
}

button[data-testid="stTab"]:hover {
    color: var(--text-primary) !important;
    background: var(--accent-dim) !important;
}

button[data-testid="stTab"][aria-selected="true"] {
    background: var(--accent) !important;
    color: #fff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px var(--accent-glow) !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.1rem 0.8rem !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.2s ease !important;
}
div[data-testid="stMetric"]:hover { border-color: var(--border-strong) !important; }

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.85rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    line-height: 1.2 !important;
}

/* ── Containers / cards ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    box-shadow: var(--shadow-card) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--border-strong) !important;
    box-shadow: var(--shadow-card), var(--shadow-glow) !important;
}

/* ── Inputs ── */
input, textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-main) !important;
    font-size: 0.9rem !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
input:focus, textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-dim) !important;
    outline: none !important;
}
input[type="password"] { font-family: var(--font-mono) !important; letter-spacing: 0.1em !important; }

label, .stTextInput label, .stTextArea label, .stSelectbox label, .stFileUploader label {
    color: var(--text-secondary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 0.3rem !important;
}

/* ── Selectbox / multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-elevated) !important;
    border-color: var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    box-shadow: 0 0 16px var(--accent-glow) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    color: #fff !important;
    border-color: var(--accent) !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
}
.stButton > button[kind="primary"]:hover {
    background: #3a7ae0 !important;
    box-shadow: 0 6px 24px var(--accent-glow) !important;
    color: #fff !important;
}
.stButton > button:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-strong) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: var(--accent-dim) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-elevated) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.18s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

/* ── Radio ── */
[data-testid="stRadio"] label { text-transform: none !important; letter-spacing: normal !important; }

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border-left-width: 3px !important;
    font-size: 0.9rem !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* ── Status widget ── */
[data-testid="stStatusWidget"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* ── Caption / small text ── */
[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* ────────────────────────────────────────────────
   Custom component classes
   ──────────────────────────────────────────────── */

/* Pill badges */
.cg-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px 3px 9px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
    margin-right: 6px;
    border: 1px solid transparent;
    font-family: var(--font-main);
}

/* Section header band */
.cg-section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid var(--border);
}
.cg-section-icon {
    width: 34px;
    height: 34px;
    border-radius: var(--radius-sm);
    background: var(--accent-dim);
    border: 1px solid rgba(79,142,247,0.25);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.cg-section-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.cg-section-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 1px;
}

/* Clause content box */
.cg-clause-box {
    background: rgba(10,14,26,0.6);
    border: 1px solid var(--border);
    border-left: 3px solid var(--border-strong);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.1rem;
    font-size: 0.9rem;
    line-height: 1.65;
    color: var(--text-primary);
    margin: 0.4rem 0 1rem 0;
    font-family: var(--font-main);
}

/* Justification box */
.cg-just-box {
    background: rgba(79,142,247,0.06);
    border: 1px solid rgba(79,142,247,0.18);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #A8C4FF;
    margin: 0.4rem 0 1rem 0;
}

/* Recommendation box */
.cg-rec-box {
    background: rgba(46,213,115,0.06);
    border: 1px solid rgba(46,213,115,0.2);
    border-left: 3px solid #2ED573;
    border-radius: var(--radius-md);
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #7DFFC0;
    margin: 0.4rem 0 1rem 0;
}

/* Decision note box */
.cg-decision-box {
    background: rgba(167,139,250,0.06);
    border: 1px solid rgba(167,139,250,0.2);
    border-radius: var(--radius-md);
    padding: 0.8rem 1rem;
    font-size: 0.875rem;
    color: #C4B5FD;
    margin-top: 0.5rem;
}

/* Meta line */
.cg-meta {
    font-size: 0.77rem;
    color: var(--text-muted);
    font-family: var(--font-mono);
    margin-bottom: 0.25rem;
}

/* Field label inside cards */
.cg-field-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.3rem;
    margin-top: 0.6rem;
}

/* Key input key ring icon */
.cg-key-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-xl);
    padding: 1.5rem 1.75rem 1.75rem;
    margin-bottom: 0.5rem;
    box-shadow: var(--shadow-card);
}
.cg-key-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(46,213,115,0.1);
    border: 1px solid rgba(46,213,115,0.3);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: #2ED573;
    margin-top: 0.6rem;
}
.cg-key-badge-off {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(168,178,216,0.08);
    border: 1px solid rgba(168,178,216,0.2);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    margin-top: 0.6rem;
}

/* Status dot */
.cg-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    flex-shrink: 0;
}
.cg-dot-on  { background: #2ED573; box-shadow: 0 0 6px #2ED573; }
.cg-dot-off { background: var(--text-muted); }

/* Sidebar status rows */
.cg-sidebar-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 0;
    font-size: 0.83rem;
    color: var(--text-secondary);
    border-bottom: 1px solid var(--border);
}
.cg-sidebar-row:last-child { border-bottom: none; }

/* Hero title area */
.cg-hero {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 0.15rem;
}
.cg-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-dim);
    border: 1px solid rgba(79,142,247,0.3);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Audit result card */
.cg-audit-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s ease;
}
.cg-audit-card:hover { border-color: var(--border-strong); }

/* Empty state */
.cg-empty {
    text-align: center;
    padding: 3.5rem 2rem;
    color: var(--text-muted);
}
.cg-empty-icon { font-size: 2.8rem; margin-bottom: 0.75rem; opacity: 0.5; }
.cg-empty-title { font-size: 1rem; font-weight: 600; color: var(--text-secondary); margin-bottom: 0.4rem; }
.cg-empty-sub { font-size: 0.85rem; }

/* Info banner */
.cg-info-banner {
    background: var(--accent-dim);
    border: 1px solid rgba(79,142,247,0.25);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    font-size: 0.875rem;
    color: #A8C4FF;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin-bottom: 0.75rem;
}

/* Step indicator */
.cg-step {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 1.1rem;
}
.cg-step-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: var(--accent-dim);
    border: 1.5px solid var(--accent);
    color: var(--accent);
    font-size: 0.75rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.cg-step-body { flex: 1; }
.cg-step-title { font-size: 0.88rem; font-weight: 600; color: var(--text-primary); margin-bottom: 2px; }
.cg-step-desc  { font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; }

/* Count badge */
.cg-count {
    display: inline-block;
    min-width: 22px;
    height: 22px;
    border-radius: 999px;
    background: var(--accent);
    color: #fff;
    font-size: 0.72rem;
    font-weight: 700;
    text-align: center;
    line-height: 22px;
    padding: 0 6px;
    margin-left: 6px;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
</style>
"""


# --------------------------------------------------------------------------
# HTML helpers
# --------------------------------------------------------------------------

def pill(text: str, color: str, bg: str, border: str = "") -> str:
    border_css = f"border-color:{border};" if border else ""
    return (
        f'<span class="cg-pill" '
        f'style="color:{color};background:{bg};{border_css}">'
        f'{text}</span>'
    )


def section_header(icon: str, title: str, sub: str = "") -> str:
    sub_html = f'<div class="cg-section-sub">{sub}</div>' if sub else ""
    return f"""
<div class="cg-section-header">
  <div class="cg-section-icon">{icon}</div>
  <div>
    <div class="cg-section-title">{title}</div>
    {sub_html}
  </div>
</div>"""


def field_label(text: str) -> str:
    return f'<div class="cg-field-label">{text}</div>'


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------


def render_sidebar() -> None:
    key_ok = bool(st.session_state.groq_api_key.strip())
    policies_ok = bool(st.session_state.policy_stats)
    n_reviews = len(st.session_state.pending_reviews)
    n_pending = sum(1 for r in st.session_state.pending_reviews if r["status"] == "PENDING")

    # Brand
    st.sidebar.markdown(
        """
        <div style="padding:0.5rem 0 1.2rem 0; border-bottom:1px solid var(--border); margin-bottom:1.1rem;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <div style="font-size:1.4rem;">⚖️</div>
                <div>
                    <div style="font-size:1.05rem;font-weight:700;color:var(--text-primary);letter-spacing:-0.01em;">ClauseGuard</div>
                    <div style="font-size:0.72rem;color:var(--text-muted);margin-top:1px;">MCP-native Compliance Review</div>
                </div>
            </div>
            <span class="cg-hero-badge">● Live · BYOK Demo</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Session status
    st.sidebar.markdown(
        '<div style="font-size:0.72rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.6rem;">Session Status</div>',
        unsafe_allow_html=True,
    )

    dot_key = '<span class="cg-dot cg-dot-on"></span>' if key_ok else '<span class="cg-dot cg-dot-off"></span>'
    dot_pol = '<span class="cg-dot cg-dot-on"></span>' if policies_ok else '<span class="cg-dot cg-dot-off"></span>'

    policy_txt = (
        f"{st.session_state.policy_stats['num_chunks']} passages · "
        f"{st.session_state.policy_stats['num_files']} file(s)"
        if policies_ok else "No policies indexed"
    )

    st.sidebar.markdown(
        f"""
        <div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:var(--radius-md);padding:0.7rem 0.85rem;margin-bottom:1.1rem;">
            <div class="cg-sidebar-row">{dot_key} Groq API key {'active' if key_ok else 'not set'}</div>
            <div class="cg-sidebar-row">{dot_pol} {policy_txt}</div>
            <div class="cg-sidebar-row">
                <span style="color:var(--accent);font-size:1em;">📄</span>
                {st.session_state.contracts_audited} contract(s) audited
            </div>
            <div class="cg-sidebar-row">
                <span style="color:#FFA502;font-size:1em;">⚖️</span>
                {n_reviews} finding(s) · {n_pending} pending
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Architecture
    with st.sidebar.expander("🧭 Pipeline Architecture"):
        st.markdown(
            """
            <div class="cg-step">
                <div class="cg-step-num">1</div>
                <div class="cg-step-body">
                    <div class="cg-step-title">Intake Agent</div>
                    <div class="cg-step-desc">Extracts risk-bearing clauses & formulates compliance queries</div>
                </div>
            </div>
            <div class="cg-step">
                <div class="cg-step-num">2</div>
                <div class="cg-step-body">
                    <div class="cg-step-title">Retrieval Agent</div>
                    <div class="cg-step-desc">Searches your in-memory Qdrant policy index</div>
                </div>
            </div>
            <div class="cg-step">
                <div class="cg-step-num">3</div>
                <div class="cg-step-body">
                    <div class="cg-step-title">Risk-Scoring Agent</div>
                    <div class="cg-step-desc">Assigns HIGH / MEDIUM / LOW with rationale</div>
                </div>
            </div>
            <div class="cg-step">
                <div class="cg-step-num">4</div>
                <div class="cg-step-body">
                    <div class="cg-step-title">Human-in-the-Loop</div>
                    <div class="cg-step-desc">You approve, reject, or modify every finding</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="font-size:0.76rem;color:var(--text-muted);line-height:1.5;margin-bottom:0.9rem;">'
        'Everything runs in memory, scoped to this browser session. '
        'No data is written to server disk or shared between visitors.'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("↺  Reset Session Data", use_container_width=True):
        reset_session()
        st.rerun()

    st.sidebar.markdown(
        f'<div style="margin-top:0.8rem;font-size:0.76rem;text-align:center;">'
        f'<a href="{GITHUB_REPO_URL}" style="color:var(--accent);text-decoration:none;">View source on GitHub ↗</a>'
        f'</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Tab 1 — API Setup & Policy Corpus Ingestion
# --------------------------------------------------------------------------


def render_setup_tab() -> None:
    # ── Step 1: API Key ──────────────────────────────────────────────────
    st.markdown(section_header("🔑", "Groq API Key", "Held in memory for this session only — never written to disk"), unsafe_allow_html=True)

    key_set = bool(st.session_state.groq_api_key.strip())
    badge = (
        '<span class="cg-key-badge"><span class="cg-dot cg-dot-on" style="margin-right:0;"></span> Active for this session</span>'
        if key_set else
        '<span class="cg-key-badge-off"><span class="cg-dot cg-dot-off" style="margin-right:0;"></span> Not set</span>'
    )

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="margin-bottom:0.8rem;">
                <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6;">
                    ClauseGuard runs on <strong style="color:var(--text-primary);">your own Groq key</strong> —
                    it's held in session memory, gone the moment you close the tab or hit Reset.
                    &nbsp;<a href="https://console.groq.com/keys" style="color:var(--accent);text-decoration:none;">
                    Get a free key at console.groq.com →</a>
                </div>
                {badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

        key_input = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_••••••••••••••••••••••••••••••••",
            key="groq_key_input",
        )
        if key_input != st.session_state.groq_api_key:
            st.session_state.groq_api_key = key_input
            st.rerun()

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # ── Step 2: Policy Documents ─────────────────────────────────────────
    st.markdown(section_header("📚", "Policy Corpus", f"Upload up to {MAX_POLICY_FILES} .txt or .md files to build your compliance index"), unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="font-size:0.85rem;color:var(--text-secondary);line-height:1.6;margin-bottom:0.9rem;">
                Documents are chunked, embedded with
                <span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--accent);">all-MiniLM-L6-v2</span>
                and stored in an in-memory Qdrant vector index — <strong style="color:var(--text-primary);">session-scoped only</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_files = st.file_uploader(
            "Drop policy documents here",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="policy_uploader",
        )

        if uploaded_files and len(uploaded_files) > MAX_POLICY_FILES:
            st.warning(
                f"You selected {len(uploaded_files)} files — the limit is {MAX_POLICY_FILES}. "
                f"Only the first {MAX_POLICY_FILES} will be indexed."
            )
            uploaded_files = uploaded_files[:MAX_POLICY_FILES]

        col_btn, col_info = st.columns([1.5, 3])
        with col_btn:
            index_clicked = st.button(
                "⬆  Index Documents",
                disabled=not uploaded_files,
                use_container_width=True,
            )
        with col_info:
            if uploaded_files:
                names = ", ".join(f.name for f in uploaded_files[:3])
                suffix = f" +{len(uploaded_files)-3} more" if len(uploaded_files) > 3 else ""
                st.markdown(
                    f'<div style="padding-top:0.55rem;font-size:0.82rem;color:var(--text-muted);">'
                    f'{len(uploaded_files)} file(s) selected: {names}{suffix}</div>',
                    unsafe_allow_html=True,
                )

        if index_clicked:
            with st.spinner("Embedding and indexing your policy documents…"):
                embedder = load_embedder()
                try:
                    stats = ingest_policy_files(
                        st.session_state.qdrant_client,
                        embedder,
                        uploaded_files,
                        collection_name=COLLECTION_NAME,
                    )
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    stats = None

            if stats:
                st.session_state.policy_stats = stats
                st.success(
                    f"✓ Successfully indexed **{stats['num_chunks']}** passages "
                    f"across **{stats['num_files']}** file(s)."
                )

    if st.session_state.policy_stats:
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
        with st.expander("📊 Indexed Policy Corpus"):
            for fname, count in st.session_state.policy_stats["per_file"].items():
                st.markdown(
                    f"""
                    <div style="display:flex;justify-content:space-between;align-items:center;
                         padding:6px 0;border-bottom:1px solid var(--border);">
                        <span style="font-size:0.85rem;color:var(--text-primary);
                               font-family:var(--font-mono);">{fname}</span>
                        <span style="font-size:0.78rem;color:var(--text-muted);">{count} passages</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# --------------------------------------------------------------------------
# Tab 2 — Contract Compliance Audit Pipeline
# --------------------------------------------------------------------------


def render_audit_tab() -> None:
    st.markdown(section_header("🔍", "Contract Compliance Audit", "Run the multi-agent pipeline against your indexed policy corpus"), unsafe_allow_html=True)

    groq_client = get_groq_client()

    # Pre-flight checks
    if groq_client is None or not st.session_state.policy_stats:
        st.markdown(
            '<div class="cg-info-banner">ℹ️ <span>Complete <strong>Tab 1</strong> first — '
            'add your Groq API key and index at least one policy document before running an audit.</span></div>',
            unsafe_allow_html=True,
        )

    with st.container(border=True):
        st.markdown(field_label("Contract Document"), unsafe_allow_html=True)
        contract_file = st.file_uploader(
            "Upload draft contract",
            type=["txt", "pdf", "docx"],
            key="contract_uploader",
            label_visibility="collapsed",
        )

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        st.markdown(field_label("Retrieval Depth"), unsafe_allow_html=True)
        top_k = st.slider(
            "Policy passages to retrieve per clause",
            min_value=1,
            max_value=5,
            value=2,
            label_visibility="collapsed",
        )
        st.markdown(
            f'<div style="font-size:0.78rem;color:var(--text-muted);margin-top:-0.4rem;margin-bottom:0.6rem;">'
            f'Retrieve <strong style="color:var(--text-primary);">{top_k}</strong> policy passage(s) per clause during retrieval</div>',
            unsafe_allow_html=True,
        )

        run_disabled = groq_client is None or not st.session_state.policy_stats or contract_file is None
        st.button(
            "🚀  Run Compliance Audit",
            disabled=run_disabled,
            type="primary",
            use_container_width=True,
            key="run_audit_btn",
            on_click=lambda: None,
        )

    # Execute audit on button press
    if st.session_state.get("run_audit_btn") and not run_disabled and contract_file:
        run_audit(contract_file, groq_client, top_k)


def run_audit(contract_file, groq_client: groq.Groq, top_k: int) -> None:
    try:
        contract_text = extract_text_from_upload(contract_file)
    except ValueError as e:
        st.error(str(e))
        return

    if not contract_text.strip():
        st.error("Couldn't extract any text from that file — is it a scanned/image-only PDF?")
        return

    embedder = load_embedder()
    step_labels = {
        1: "Step 1 / 3 — Intake Agent: extracting risk-bearing clauses…",
        2: "Step 2 / 3 — Retrieval Agent: searching your indexed policies…",
        3: "Step 3 / 3 — Risk-Scoring Agent: evaluating compliance…",
    }

    with st.status("Running the compliance pipeline…", expanded=True) as status:

        def on_step(n: int, label: str) -> None:
            msg = step_labels.get(n, label)
            status.update(label=msg)
            st.write(msg)

        try:
            final_report = run_compliance_pipeline(
                contract_text,
                contract_file.name,
                groq_client,
                st.session_state.qdrant_client,
                embedder,
                top_k_policies=top_k,
                model_name=GROQ_MODEL_NAME,
                collection_name=COLLECTION_NAME,
                on_step=on_step,
            )
        except groq.AuthenticationError:
            status.update(label="Failed — invalid API key", state="error")
            st.error("🔒 Your Groq API key was rejected. Double-check it in Tab 1 and try again.")
            return
        except groq.RateLimitError:
            status.update(label="Failed — rate limited", state="error")
            st.error("⏱️ You've hit your Groq quota or rate limit. Wait a moment, or enter a different key.")
            return
        except groq.APIError as e:
            status.update(label="Failed — Groq API error", state="error")
            st.error(f"Groq API error: {e}")
            return
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong while auditing this contract: {e}")
            return

        status.update(label="Compliance audit complete ✓", state="complete")

    st.session_state.contracts_audited += 1
    added = push_flagged_findings(final_report)

    evaluations = final_report.get("evaluations", [])
    high_count = sum(1 for c in evaluations if c.get("risk_level") == "HIGH")
    med_count  = sum(1 for c in evaluations if c.get("risk_level") == "MEDIUM")
    low_count  = sum(1 for c in evaluations if c.get("risk_level") == "LOW")

    st.markdown(
        f"""
        <div style="background:rgba(46,213,115,0.07);border:1px solid rgba(46,213,115,0.25);
             border-radius:var(--radius-md);padding:1rem 1.2rem;margin:0.8rem 0 1.2rem 0;
             display:flex;align-items:center;gap:1.2rem;flex-wrap:wrap;">
            <div style="font-size:1.3rem;">✅</div>
            <div style="flex:1;min-width:200px;">
                <div style="font-size:0.92rem;font-weight:600;color:#2ED573;margin-bottom:2px;">Audit Complete</div>
                <div style="font-size:0.82rem;color:var(--text-muted);">
                    {len(evaluations)} clause(s) reviewed &nbsp;·&nbsp;
                    {added} new finding(s) sent to HITL Dashboard
                </div>
            </div>
            <div style="display:flex;gap:10px;">
                {pill(f"⬆ {high_count} HIGH", RISK_STYLE['HIGH']['color'], RISK_STYLE['HIGH']['bg'], RISK_STYLE['HIGH']['border'])}
                {pill(f"◆ {med_count} MEDIUM", RISK_STYLE['MEDIUM']['color'], RISK_STYLE['MEDIUM']['bg'], RISK_STYLE['MEDIUM']['border'])}
                {pill(f"⬇ {low_count} LOW", RISK_STYLE['LOW']['color'], RISK_STYLE['LOW']['bg'], RISK_STYLE['LOW']['border'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.78rem;font-weight:600;color:var(--text-muted);text-transform:uppercase;'
        'letter-spacing:0.07em;margin-bottom:0.7rem;">Clause-by-Clause Results</div>',
        unsafe_allow_html=True,
    )

    for clause in evaluations:
        risk = clause.get("risk_level", "LOW")
        style = RISK_STYLE.get(risk, RISK_STYLE["LOW"])
        with st.container(border=True):
            header_col, badge_col = st.columns([5, 1.2])
            with header_col:
                st.markdown(
                    pill(f"{style['icon']} {risk}", style["color"], style["bg"], style["border"])
                    + f' <strong style="font-size:0.92rem;color:var(--text-primary);">'
                    f'{clause.get("clause_title", "Untitled Clause")}</strong>',
                    unsafe_allow_html=True,
                )
            st.markdown(field_label("Clause Text"), unsafe_allow_html=True)
            st.markdown(f'<div class="cg-clause-box">{clause.get("clause_text", "")}</div>', unsafe_allow_html=True)
            st.markdown(field_label("AI Assessment"), unsafe_allow_html=True)
            st.markdown(f'<div class="cg-just-box">{clause.get("justification", "")}</div>', unsafe_allow_html=True)
            if clause.get("recommendation"):
                st.markdown(field_label("Suggested Fix"), unsafe_allow_html=True)
                st.markdown(f'<div class="cg-rec-box">{clause["recommendation"]}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Tab 3 — HITL Review Dashboard
# --------------------------------------------------------------------------


def render_dashboard_filters(records: list[dict[str, Any]]) -> dict[str, Any]:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 1.8])
        status_filter = c1.radio("Status", ["Pending only", "Resolved only", "All"], index=0)
        risk_filter = c2.multiselect("Risk Level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
        contracts = sorted({r["contract_name"] for r in records})
        contract_filter = c3.multiselect("Contract", contracts, default=[], placeholder="All contracts")
        search = c4.text_input("Search", placeholder="e.g. encryption, indemnification…")
        sort_by = st.selectbox("Sort By", ["Risk (High → Low)", "Contract name"], index=0)

    return {
        "status_filter": status_filter,
        "risk_filter": risk_filter,
        "contract_filter": contract_filter,
        "search": search.strip().lower(),
        "sort_by": sort_by,
    }


def render_summary(records: list[dict[str, Any]]) -> None:
    pending  = [r for r in records if r["status"] == "PENDING"]
    high     = sum(1 for r in pending if r["risk_level"] == "HIGH")
    medium   = sum(1 for r in pending if r["risk_level"] == "MEDIUM")
    low      = sum(1 for r in pending if r["risk_level"] == "LOW")
    resolved = sum(1 for r in records if r["status"] != "PENDING")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pending Review", len(pending))
    c2.metric("⬆ High Risk",    high)
    c3.metric("◆ Medium Risk",  medium)
    c4.metric("⬇ Low Risk",     low)
    c5.metric("Resolved",       resolved)


def render_card(record: dict[str, Any], reviewer: str) -> None:
    rid       = record["review_id"]
    risk      = record.get("risk_level", "LOW")
    risk_s    = RISK_STYLE.get(risk, RISK_STYLE["LOW"])
    status    = record.get("status", "PENDING")
    status_s  = STATUS_STYLE.get(status, STATUS_STYLE["PENDING"])
    is_pending = status == "PENDING"

    with st.container(border=True):
        # ── Header row ──────────────────────────────────────────────────
        header_l, header_r = st.columns([5, 2])

        with header_l:
            risk_pill   = pill(f"{risk_s['icon']} {risk}", risk_s["color"], risk_s["bg"], risk_s["border"])
            status_pill = pill(status, status_s["color"], status_s["bg"])
            st.markdown(risk_pill + status_pill, unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:0.9rem;font-weight:600;color:var(--text-primary);margin-top:5px;">'
                f'{record["contract_name"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="cg-meta">ID: {rid} &nbsp;·&nbsp; Flagged: {record["timestamp"]}</div>',
                unsafe_allow_html=True,
            )

        with header_r:
            if not is_pending and record.get("human_decision"):
                d   = record["human_decision"]
                d_s = DECISION_STYLE.get(d, DECISION_STYLE["APPROVED"])
                decision_pill = pill(f'{d_s["icon"]} {d}', d_s["color"], d_s["bg"])
                st.markdown(
                    f'<div style="text-align:right;margin-top:4px;">{decision_pill}</div>',
                    unsafe_allow_html=True,
                )
                if record.get("reviewed_by"):
                    st.markdown(
                        f'<div class="cg-meta" style="text-align:right;">'
                        f'by {record["reviewed_by"]}<br>{record.get("reviewed_at","")}</div>',
                        unsafe_allow_html=True,
                    )

        # ── Clause ──────────────────────────────────────────────────────
        st.markdown(field_label("Clause Text"), unsafe_allow_html=True)
        st.markdown(f'<div class="cg-clause-box">{record["clause_text"]}</div>', unsafe_allow_html=True)

        # ── Justification ───────────────────────────────────────────────
        st.markdown(field_label("Why It Was Flagged"), unsafe_allow_html=True)
        st.markdown(f'<div class="cg-just-box">{record["justification"]}</div>', unsafe_allow_html=True)

        # ── Recommendation ───────────────────────────────────────────────
        if record.get("recommendation"):
            st.markdown(field_label("Suggested Remedy"), unsafe_allow_html=True)
            st.markdown(f'<div class="cg-rec-box">{record["recommendation"]}</div>', unsafe_allow_html=True)

        # ── Resolved state ───────────────────────────────────────────────
        if not is_pending:
            if record.get("human_notes"):
                st.markdown(field_label("Reviewer Notes"), unsafe_allow_html=True)
                st.markdown(f'<div class="cg-decision-box">{record["human_notes"]}</div>', unsafe_allow_html=True)
            return

        # ── HITL controls ────────────────────────────────────────────────
        st.markdown(
            '<div style="border-top:1px solid var(--border);margin:0.8rem 0 0.9rem;"></div>',
            unsafe_allow_html=True,
        )
        st.markdown(field_label("Your Decision"), unsafe_allow_html=True)

        notes_key = f"notes__{rid}"
        notes = st.text_area(
            "Notes",
            key=notes_key,
            placeholder="e.g. Confirmed with vendor legal — escalating to legal team…",
            height=72,
            label_visibility="collapsed",
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("✓  Approve", key=f"approve__{rid}", use_container_width=True):
            commit_decision(rid, "APPROVED", notes, reviewer)
        if b2.button("✕  Reject", key=f"reject__{rid}", use_container_width=True):
            commit_decision(rid, "REJECTED", notes, reviewer)
        if b3.button("⟳  Modify Risk", key=f"modify__{rid}", use_container_width=True):
            st.session_state[f"show_modify__{rid}"] = True

        if st.session_state.get(f"show_modify__{rid}"):
            mc1, mc2 = st.columns([2, 1])
            with mc1:
                new_level = st.selectbox(
                    "New risk level",
                    ["HIGH", "MEDIUM", "LOW"],
                    key=f"newlevel__{rid}",
                )
            with mc2:
                st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                if st.button("Save", key=f"savelevel__{rid}", use_container_width=True):
                    modify_risk_level(rid, new_level, notes, reviewer)


def render_export_controls(records: list[dict[str, Any]]) -> None:
    st.markdown(section_header("📤", "Export Audit Report", "Download the full audit trail in your preferred format"), unsafe_allow_html=True)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                '<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:0.5rem;">'
                'Structured data — ideal for programmatic ingestion and CI/CD pipelines</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇  Download as JSON",
                data=json.dumps(records, indent=2),
                file_name="clauseguard_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )
        with col2:
            st.markdown(
                '<div style="font-size:0.82rem;color:var(--text-muted);margin-bottom:0.5rem;">'
                'Human-readable format — great for legal review and documentation</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇  Download as Markdown",
                data=records_to_markdown(records),
                file_name="clauseguard_audit_report.md",
                mime="text/markdown",
                use_container_width=True,
            )


def render_dashboard_tab() -> None:
    records = st.session_state.pending_reviews

    if not records:
        st.markdown(
            """
            <div class="cg-empty">
                <div class="cg-empty-icon">⚖️</div>
                <div class="cg-empty-title">No Findings Yet</div>
                <div class="cg-empty-sub">Run a compliance audit in <strong>Tab 2</strong> to populate this dashboard.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Reviewer identity
    with st.container(border=True):
        id_col, _ = st.columns([2, 3])
        with id_col:
            reviewer = st.text_input(
                "Reviewer Name",
                value=st.session_state.reviewer_name,
                placeholder="e.g. J. Alvarez",
                help="Attached to any decision you make — part of the audit trail.",
            )
        st.session_state.reviewer_name = reviewer
        if not reviewer.strip():
            st.markdown(
                '<div style="font-size:0.8rem;color:#FFA502;margin-top:-0.3rem;">'
                '⚠ Enter your name to enable decision recording</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Metrics
    render_summary(records)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Filters
    filters = render_dashboard_filters(records)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Findings
    visible = filter_and_sort(records, filters)
    if not visible:
        st.markdown(
            '<div class="cg-empty" style="padding:2rem;"><div class="cg-empty-sub">No findings match the current filters.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:0.78rem;color:var(--text-muted);margin-bottom:0.7rem;">'
            f'Showing <strong style="color:var(--text-primary);">{len(visible)}</strong> of '
            f'<strong style="color:var(--text-primary);">{len(records)}</strong> findings</div>',
            unsafe_allow_html=True,
        )
        for record in visible:
            render_card(record, reviewer)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    render_export_controls(records)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="ClauseGuard — Live Compliance Review",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    # Hero header
    st.markdown(
        """
        <div class="cg-hero" style="margin-bottom:0.1rem;">
            <span style="font-size:2rem;">⚖️</span>
            <div>
                <h1 style="margin:0;padding:0;line-height:1.1;">ClauseGuard</h1>
            </div>
            <span class="cg-hero-badge">MCP-Native · Multi-Agent</span>
        </div>
        <div style="font-size:0.875rem;color:var(--text-muted);margin-bottom:1.5rem;padding-left:2px;">
            Multi-agent contract compliance review — bring your own Groq key; your policies and findings never leave your session.
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_sidebar()

    # Flash messages
    if st.session_state.flash:
        kind, message = st.session_state.flash
        getattr(st, kind)(message)
        st.session_state.flash = None

    tab1, tab2, tab3 = st.tabs(
        [
            "🔑  API Setup & Policy Corpus",
            "🔍  Contract Compliance Audit",
            "⚖️  HITL Review Dashboard",
        ]
    )

    with tab1:
        render_setup_tab()
    with tab2:
        render_audit_tab()
    with tab3:
        render_dashboard_tab()


if __name__ == "__main__":
    main()
