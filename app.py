"""
ClauseGuard — Live Multi-Tenant Compliance Review (BYOK)
------------------------------------------------------------
A public, session-isolated deployment of the ClauseGuard pipeline for
Streamlit Community Cloud / Hugging Face Spaces.

Every visitor brings their own Groq API key, uploads up to 10 policy
documents, and gets an isolated in-memory Qdrant index + HITL review queue
that lives only for their browser session. Nothing is ever written to
server disk — no `/corpus`, no `pending_reviews.json`, no shared Qdrant path.[cite: 1]

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
GITHUB_REPO_URL = "https://github.com/Danialpro2k04/ClauseGuard"

RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

RISK_STYLE = {
    "HIGH": {"color": "#DC2626", "bg": "#FEF2F2", "border": "#FCA5A5", "icon": "🔴"},
    "MEDIUM": {"color": "#D97706", "bg": "#FFFBEB", "border": "#FCD34D", "icon": "🟠"},
    "LOW": {"color": "#65A30D", "bg": "#F7FEE7", "border": "#BEF264", "icon": "🟢"},
}

STATUS_STYLE = {
    "PENDING": {"color": "#4B5563", "bg": "#F3F4F6"},
    "RESOLVED": {"color": "#1D4ED8", "bg": "#EFF6FF"},
}

DECISION_STYLE = {
    "APPROVED": {"color": "#15803D", "bg": "#F0FDF4", "icon": "✅"},
    "REJECTED": {"color": "#B91C1C", "bg": "#FEF2F2", "icon": "❌"},
    "MODIFIED": {"color": "#7C3AED", "bg": "#F5F3FF", "icon": "✏️"},
}

SESSION_DEFAULTS = {
    "groq_api_key": "",
    "qdrant_client": None,
    "policy_stats": None,
    "pending_reviews": [],
    "reviewer_name": "",
    "flash": None,
    "contracts_audited": 0,
    "current_step": 1, # Wizard step tracker
}



@st.cache_resource(show_spinner="Loading embedding model (first run only)…")
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)



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


def next_step():
    st.session_state.current_step += 1

def prev_step():
    st.session_state.current_step -= 1


def get_groq_client() -> groq.Groq | None:
    key = st.session_state.get("groq_api_key", "").strip()
    if not key:
        return None
    return groq.Groq(api_key=key)


def flash(kind: str, message: str) -> None:
    st.session_state.flash = (kind, message)



# HITL record helpers 


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



# Styling


# Integrated Premium Neutral CSS + Bug fixes
CUSTOM_CSS = """
<style>
    /* ── 1. App Canvas & Typography ── */
    .stApp {
        background-color: #F4F7F9 !important;
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    h1, h2, h3 {
        color: #0F172A !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* ── 2. The Sidebar: Engaging & Premium Neutral ── */
    section[data-testid="stSidebar"] {
        background-color: #EBF0F5 !important;
        border-right: 1px solid #D9E2EC !important;
        box-shadow: inset -1px 0 0 rgba(0,0,0,0.03) !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    
    section[data-testid="stSidebar"] button {
        background-color: #FFFFFF !important;
        border: 1px solid #D9E2EC !important;
        color: #0F172A !important;
    }

    /* ── 3. Elevated Cards (Pure White against the Gray Canvas) ── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"], 
    div[data-testid="stForm"],
    .st-emotion-cache-1wmy9hl {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        border-top: 3px solid #4F46E5 !important; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03) !important;
        padding: 1.75rem !important;
        transition: all 0.2s ease-in-out;
        /* BUG FIX: Prevent spinner from overflowing out of the card */
        height: auto !important;
        overflow: visible !important;
    }
    
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"]:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08) !important;
        border-color: #CBD5E1 !important;
    }

    /* ── 4. Tactile Inputs & Focus Rings ── */
    .stTextInput input, div[data-baseweb="input"] {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        color: #0F172A !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15) !important; 
        background-color: #FFFFFF !important;
    }

    /* ── 5. Primary Buttons: Premium Gradient ── */
    .stButton button {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 12px -2px rgba(79, 70, 229, 0.3) !important;
    }

    .stButton button:disabled {
        background: #E2E8F0 !important;
        color: #94A3B8 !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* Original pill styling kept intact */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-right: 6px;
        white-space: nowrap;
    }

    .meta-line { color: #6B7280; font-size: 0.82rem; margin-bottom: 0.3rem; }
    hr { margin: 0.6rem 0 1.2rem 0; }
</style>
"""


def pill(text: str, color: str, bg: str) -> str:
    return f'<span class="pill" style="color:{color};background:{bg};">{text}</span>'



# Sidebar


def render_sidebar() -> None:
    st.sidebar.markdown("### ⚖️ ClauseGuard")
    
    st.sidebar.markdown(
        '<div style="font-weight: 600; color: #334155; margin-bottom: 1rem;">'
        '<span style="color: #09ed59; margin-right: 4px;">●</span> Live Review'
        '</div>', 
        unsafe_allow_html=True
    )

    key_ok = bool(st.session_state.groq_api_key.strip())
    policies_ok = bool(st.session_state.policy_stats)

    st.sidebar.markdown("**Session status**")
    st.sidebar.markdown(f"{'✅' if key_ok else '🔒'} Groq API key {'set' if key_ok else 'not set'}")
    if policies_ok:
        st.sidebar.markdown(
            f"✅ {st.session_state.policy_stats['num_chunks']} policy passages indexed "
            f"({st.session_state.policy_stats['num_files']} file(s))"
        )
    else:
        st.sidebar.markdown("🔒 No policies indexed yet")
    st.sidebar.markdown(f"📄 {st.session_state.contracts_audited} contract(s) audited this session")
    st.sidebar.markdown(f"⚖️ {len(st.session_state.pending_reviews)} finding(s) in HITL queue")

    with st.sidebar.expander("🧭 System architecture"):
        st.markdown(
            "1. **Intake Agent** — extracts risk-bearing clauses & formulates "
            "compliance statements\n"
            "2. **Retrieval Agent** — searches your in-memory Qdrant policy index\n"
            "3. **Risk-Scoring Agent** — assigns HIGH / MEDIUM / LOW with rationale\n"
            "4. **Human-in-the-Loop** — you approve, reject, or modify every finding"
        )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Everything above runs in memory, scoped to this browser session. "
        "Nothing is written to server disk, and no data is shared between visitors."
    )

    if st.sidebar.button("🔄 Reset Session Data", use_container_width=True):
        reset_session()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
    f'''
    <a href="{GITHUB_REPO_URL}" target="_blank" style="text-decoration: none; color: inherit; font-size: 0.8em;">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="16" style="vertical-align: middle; margin-right: 5px; filter: invert(0.5);"/>
        View MCP server + Full architecture
    </a>
    ''',
    unsafe_allow_html=True
)



# Step 1 — API Setup & Policy Corpus Ingestion


def render_setup_step() -> None:
    st.subheader("1. Connect your Groq API key")
    st.markdown(
        "ClauseGuard runs on **your own** Groq key for this session only — it's "
        "held in memory, never written to disk, and gone the moment you close this "
        "tab or hit Reset. [Get a free key at console.groq.com →](https://console.groq.com/keys)"
    )

    key_input = st.text_input(
        "Groq API key",
        value=st.session_state.groq_api_key,
        type="password",
        placeholder="gsk_...",
        key="groq_key_input",
    )
    if key_input != st.session_state.groq_api_key:
        st.session_state.groq_api_key = key_input
        st.rerun()

    if st.session_state.groq_api_key:
        st.success("✅ Key saved for this session.")
    else:
        st.info("Enter a key above to unlock the audit pipeline.")

    st.markdown("---")
    st.subheader("2. Upload your company's policy documents")
    st.caption(
        f"Up to {MAX_POLICY_FILES} `.txt` or `.md` files. These are chunked, embedded, "
        "and stored in an in-memory vector index that lives only for this session."
    )

    uploaded_files = st.file_uploader(
        "Policy documents",
        type=["txt", "md"],
        accept_multiple_files=True,
        key="policy_uploader",
    )

    if uploaded_files and len(uploaded_files) > MAX_POLICY_FILES:
        st.warning(
            f"You selected {len(uploaded_files)} files, but the limit is "
            f"{MAX_POLICY_FILES}. Only the first {MAX_POLICY_FILES} will be indexed — "
            "remove some and re-upload to include the rest."
        )
        uploaded_files = uploaded_files[:MAX_POLICY_FILES]

    if st.button("📚 Index Policy Documents", disabled=not uploaded_files):
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
                f"Successfully indexed {stats['num_chunks']} policy passages "
                f"across {stats['num_files']} file(s)."
            )

    if st.session_state.policy_stats:
        with st.expander("📊 Indexed policy corpus"):
            for fname, count in st.session_state.policy_stats["per_file"].items():
                st.markdown(f"- **{fname}** — {count} passage(s)")



# Step 2 — Contract Compliance Audit Pipeline


def render_audit_step() -> None:
    st.subheader("Audit a draft contract")

    groq_client = get_groq_client()
    if groq_client is None:
        st.warning("Ensure your Groq API key is setup in the previous step.")
    if not st.session_state.policy_stats:
        st.warning(
            "Go back and index at least one policy document so the Retrieval "
            "Agent has something to compare against."
        )

    contract_file = st.file_uploader(
        "Draft contract", type=["txt", "pdf", "docx"], key="contract_uploader"
    )
    top_k = st.slider("Policy passages to retrieve per clause", min_value=1, max_value=5, value=2)

    run_disabled = groq_client is None or not st.session_state.policy_stats or contract_file is None
    if st.button("🚀 Run Compliance Audit", disabled=run_disabled, type="primary"):
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
        1: "Step 1/3 — Intake Agent: extracting risk-bearing clauses…",
        2: "Step 2/3 — Retrieval Agent: searching your indexed policies…",
        3: "Step 3/3 — Risk-Scoring Agent: evaluating compliance…",
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
            st.error(
                "⏱️ You've hit your Groq quota or rate limit. Wait a bit, or enter "
                "a different key in Tab 1."
            )
            return
        except groq.APIError as e:
            status.update(label="Failed — Groq API error", state="error")
            st.error(f"Groq API error: {e}")
            return
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"Something went wrong while auditing this contract: {e}")
            return

        status.update(label="Compliance audit complete.", state="complete")

    st.session_state.contracts_audited += 1
    added = push_flagged_findings(final_report)

    st.success(
        f"Audit complete — {len(final_report['evaluations'])} clause(s) reviewed, "
        f"{added} new finding(s) sent to the HITL dashboard (Next Step)."
    )

    for clause in final_report["evaluations"]:
        risk = clause.get("risk_level", "LOW")
        style = RISK_STYLE.get(risk, RISK_STYLE["LOW"])
        with st.container(border=True):
            st.markdown(pill(f"{style['icon']} {risk}", style["color"], style["bg"]), unsafe_allow_html=True)
            st.markdown(f"**{clause.get('clause_title', 'Untitled Clause')}**")
            st.caption(clause.get("clause_text", ""))
            st.markdown(f"*{clause.get('justification', '')}*")
            if clause.get("recommendation"):
                st.markdown(f"**Suggested fix:** {clause['recommendation']}")



# Step 3 — HITL Review Dashboard


def render_dashboard_filters(records: list[dict[str, Any]]) -> dict[str, Any]:
    c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 1.8])
    status_filter = c1.radio("Status", ["Pending only", "Resolved only", "All"], index=0)
    risk_filter = c2.multiselect("Risk level", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM", "LOW"])
    contracts = sorted({r["contract_name"] for r in records})
    contract_filter = c3.multiselect("Contract", contracts, default=[], placeholder="All contracts")
    search = c4.text_input("Search clause text", placeholder="e.g. encryption, indemnification…")
    sort_by = st.selectbox("Sort by", ["Risk (High → Low)", "Contract name"], index=0)
    return {
        "status_filter": status_filter,
        "risk_filter": risk_filter,
        "contract_filter": contract_filter,
        "search": search.strip().lower(),
        "sort_by": sort_by,
    }


def render_summary(records: list[dict[str, Any]]) -> None:
    pending = [r for r in records if r["status"] == "PENDING"]
    high = sum(1 for r in pending if r["risk_level"] == "HIGH")
    medium = sum(1 for r in pending if r["risk_level"] == "MEDIUM")
    low = sum(1 for r in pending if r["risk_level"] == "LOW")
    resolved = sum(1 for r in records if r["status"] != "PENDING")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Pending review", len(pending))
    c2.metric("🔴 High risk", high)
    c3.metric("🟠 Medium risk", medium)
    c4.metric("🟢 Low risk", low)
    c5.metric("Resolved", resolved)


def render_card(record: dict[str, Any], reviewer: str) -> None:
    rid = record["review_id"]
    risk = record.get("risk_level", "LOW")
    risk_s = RISK_STYLE.get(risk, RISK_STYLE["LOW"])
    status = record.get("status", "PENDING")
    status_s = STATUS_STYLE.get(status, STATUS_STYLE["PENDING"])
    is_pending = status == "PENDING"

    with st.container(border=True):
        header_l, header_r = st.columns([5, 2])

        with header_l:
            st.markdown(
                pill(f"{risk_s['icon']} {risk}", risk_s["color"], risk_s["bg"])
                + pill(status, status_s["color"], status_s["bg"]),
                unsafe_allow_html=True,
            )
            st.markdown(f"**{record['contract_name']}**")
            st.markdown(
                f'<div class="meta-line">ID: {rid} &nbsp;•&nbsp; Flagged: {record["timestamp"]}</div>',
                unsafe_allow_html=True,
            )

        with header_r:
            if not is_pending and record.get("human_decision"):
                d = record["human_decision"]
                d_s = DECISION_STYLE.get(d, DECISION_STYLE["APPROVED"])
        
                pill_label = f"{d_s['icon']} {d}"
                pill_html = pill(pill_label, d_s['color'], d_s['bg'])
        
                st.markdown(
                    f'<div style="text-align:right;">{pill_html}</div>',
                    unsafe_allow_html=True,
                )
                if record.get("reviewed_by"):
                    st.markdown(
                        f'<div class="meta-line" style="text-align:right;">'
                        f'by {record["reviewed_by"]} on {record.get("reviewed_at", "")}</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("**Clause Text**")
        st.info(record["clause_text"])

        st.markdown("**Why It Was Flagged**")
        st.warning(record["justification"])

        if record.get("recommendation"):
            st.markdown("**Suggested Remedy**")
            st.success(record["recommendation"])

        if not is_pending:
            if record.get("human_notes"):
                st.markdown("**Reviewer notes**")
                st.write(record["human_notes"])
            return

        st.markdown("**Your Decision**")
        notes_key = f"notes__{rid}"
        notes = st.text_area(
            "Notes (optional, but recommended — visible in the audit trail)",
            key=notes_key,
            placeholder="e.g. Confirmed with vendor legal this is a placeholder clause; escalating.",
            height=80,
            label_visibility="collapsed",
        )

        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Approve", key=f"approve__{rid}", use_container_width=True):
            commit_decision(rid, "APPROVED", notes, reviewer)
        if b2.button("❌ Reject", key=f"reject__{rid}", use_container_width=True):
            commit_decision(rid, "REJECTED", notes, reviewer)
        if b3.button("✏️ Modify risk level", key=f"modify__{rid}", use_container_width=True):
            st.session_state[f"show_modify__{rid}"] = True

        if st.session_state.get(f"show_modify__{rid}"):
            new_level = st.selectbox(
                "New risk level", ["HIGH", "MEDIUM", "LOW"], key=f"newlevel__{rid}"
            )
            if st.button("Save new risk level", key=f"savelevel__{rid}"):
                modify_risk_level(rid, new_level, notes, reviewer)


def render_export_controls(records: list[dict[str, Any]]) -> None:
    st.subheader("📤 Export audit report")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download as JSON",
            data=json.dumps(records, indent=2),
            file_name="clauseguard_audit_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download as Markdown",
            data=records_to_markdown(records),
            file_name="clauseguard_audit_report.md",
            mime="text/markdown",
            use_container_width=True,
        )


def render_dashboard_step() -> None:
    records = st.session_state.pending_reviews

    if not records:
        st.info("No findings yet — run an audit in **Step 2** to populate this dashboard.")
        return

    reviewer = st.text_input(
        "Your name",
        value=st.session_state.reviewer_name,
        placeholder="e.g. J. Alvarez",
        help="Attached to any decision you make below, for the audit trail.",
    )
    st.session_state.reviewer_name = reviewer

    filters = render_dashboard_filters(records)
    render_summary(records)
    st.markdown("---")

    visible = filter_and_sort(records, filters)
    if not visible:
        st.info("No findings match the current filters.")
    else:
        st.caption(f"Showing {len(visible)} of {len(records)} findings.")
        for record in visible:
            render_card(record, reviewer)

    st.markdown("---")
    render_export_controls(records)


# Main Flow

def main() -> None:
    st.set_page_config(
        page_title="ClauseGuard — Live Compliance Review",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()

    st.title("⚖️ ClauseGuard")
    st.caption(
        "Multi-agent contract compliance review — bring your own Groq "
        "key; your policies and findings never leave your session."
    )

    render_sidebar()

    if st.session_state.flash:
        kind, message = st.session_state.flash
        getattr(st, kind)(message)
        st.session_state.flash = None

    if st.session_state.current_step == 1:
        render_setup_step()
        st.divider()
        
        # Validates if user has entered Groq key AND indexed a policy
        can_proceed = bool(st.session_state.groq_api_key.strip()) and bool(st.session_state.policy_stats)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col3:
            st.button(
                "Next: Run Audit ➔", 
                on_click=next_step, 
                disabled=not can_proceed, 
                use_container_width=True
            )

    elif st.session_state.current_step == 2:
        render_audit_step()
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.button("⬅ Previous: Setup", on_click=prev_step, use_container_width=True)
        with col3:
            st.button("Next: Review Findings ➔", on_click=next_step, use_container_width=True)

    elif st.session_state.current_step == 3:
        render_dashboard_step()
        st.divider()
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.button("⬅ Previous: Audit", on_click=prev_step, use_container_width=True)


if __name__ == "__main__":
    main()
