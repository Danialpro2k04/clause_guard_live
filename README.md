# ⚖️ ClauseGuard Live

**Multi-agent contract compliance review — Bring Your Own Key (BYOK)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://clauseguardlive.streamlit.app/) 

ClauseGuard is a live, session-isolated web application built with Streamlit that automates contract compliance reviews. By combining a multi-agent LLM pipeline with Retrieval-Augmented Generation (RAG), it evaluates draft contracts against your internal company policies and flags risky clauses for human review.

This repository hosts the **Live/Public Demo** version of the tool. It is designed with a strict zero-retention architecture: every visitor brings their own API key, and all data (keys, indexed policies, and contract findings) is held entirely in memory and wiped the moment the browser session ends.

## Why the live demo doesn't run through MCP

The real ClauseGurad repo includes a full MCP (Model Context Protocol) server in
`server/mcp_server.py`, but this hosted live demo doesn't call it. That's
intentional, not a shortcut.

MCP servers communicate over local transport (stdio) with a client running
on the same machine — like Claude Desktop or Cursor — not with anonymous
visitors over the public web. There's no equivalent of a "public MCP
endpoint" the way there is with a REST API.

So this project has two front doors to the same core agents:

- **Live demo** (`clauseguardlive.streamlit.app`) — a Streamlit app that
  calls the Intake, Retrieval, and Risk-Scoring agents directly, so anyone
  can try the full pipeline instantly with zero setup.
- **MCP server** (`server/mcp_server.py`) — exposes `search_policy_docs`
  and `log_for_human_review` as MCP tools, so the same capability can be
  plugged directly into an MCP client like Claude Desktop for local use.

Same agents, same logic — two different doors, built for two different
audiences.

---

## ✨ Key Features

* **Privacy-First & Session Isolated:** No database, no server disk writes. Qdrant vector stores and document corpora exist solely in the user's volatile memory (`QdrantClient(":memory:")`, scoped to `st.session_state`).
* **Bring Your Own Key (BYOK):** Powered by Groq (`llama-3.1-8b-instant`). Users securely inject their own API key for the session — it's never persisted to disk.
* **Multi-Agent Pipeline:**
  1. **Intake Agent:** Extracts risk-bearing clauses and formulates compliance statements.
  2. **Retrieval Agent:** Embeds and searches your custom Qdrant policy index.
  3. **Risk-Scoring Agent:** Assigns HIGH / MEDIUM / LOW risk scores with detailed justifications.
* **Human-in-the-Loop (HITL) Dashboard:** A sleek, dedicated workspace to approve, reject, or modify risk levels, complete with an exportable audit trail.
* **Guided Wizard UI:** A clean, premium-neutral step-by-step flow (Setup ➔ Audit ➔ Review) ensuring a smooth user experience.

---

## 🏗️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM Provider:** [Groq](https://groq.com/) (Llama 3.1)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Vector Database:** [Qdrant](https://qdrant.tech/) (In-Memory mode)

> **Note on the embedding model:** the `SentenceTransformer` instance is loaded once via `st.cache_resource` and shared across all visitors' sessions. This is safe from a privacy standpoint — it holds only pretrained weights, never a visitor's documents or keys — and keeps cold-start times reasonable on a free-tier host. The Qdrant client, policy index, Groq key, and HITL queue are never shared; those are always created fresh per session.

---

## 🚀 Running Locally

To run this application on your local machine, follow these steps:

### 1. Clone the repository

```bash
git clone https://github.com/Danialpro2k04/clause_guard_live.git
cd clause_guard_live
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📖 How to Use

ClauseGuard is broken down into a simple 3-step wizard flow:

### Step 1: API Setup & Policy Corpus

* Enter your Groq API key (get a free one at [console.groq.com](https://console.groq.com/keys)).
* Upload up to 10 policy documents (`.txt` or `.md`). The app will chunk, embed, and index these into a temporary Qdrant vector store.

### Step 2: Contract Compliance Audit

* Upload a draft contract (`.txt`, `.pdf`, or `.docx`).
* The multi-agent pipeline will scan the contract, retrieve relevant policies from your index, and evaluate compliance.

### Step 3: HITL Review Dashboard

* Review the flagged findings (High, Medium, Low risk).
* Read the LLM's justification and suggested remedies.
* Add your reviewer notes and choose to **Approve**, **Reject**, or **Modify the risk level**.
* Export the final audit report as JSON or Markdown.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Danialpro2k04/clause_guard_live/issues).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
