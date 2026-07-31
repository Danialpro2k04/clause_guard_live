"""
Pipeline orchestrator for the live ClauseGuard deployment.
--------------------------------------------------------------
Runs the Intake -> Retrieval -> Risk-Scoring sequence against a single
uploaded contract, using session-scoped Groq/Qdrant clients so nothing here
touches local disk or a process-wide shared client.

Two changes from the original `pipeline.py`:
  1. `extract_text_from_file(path)` -> `extract_text_from_upload(uploaded_file)`:
     reads a Streamlit `UploadedFile` straight out of memory instead of
     opening a path on disk (there is no server-side upload directory in the
     live app).
  2. `review_contract(file_path)` -> `run_compliance_pipeline(...)`: takes
     the Groq client, Qdrant client, and embedder as arguments (BYOK +
     per-session isolation) and accepts an `on_step` callback so a caller can
     drive progress spinners without this module importing Streamlit.
"""

from __future__ import annotations

import io
import os
from typing import Callable, Optional

import pypdf
from docx import Document

from agents.intake import run_intake_agent
from agents.retrieval import run_retrieval_agent
from agents.risk_scorer import run_risk_scoring_agent
from vector_store import DEFAULT_COLLECTION_NAME

DEFAULT_MODEL = "llama-3.1-8b-instant"


def extract_text_from_upload(uploaded_file) -> str:
    """Extracts raw text from an in-memory uploaded contract file.

    Args:
        uploaded_file: A Streamlit `UploadedFile` (or any object exposing
            `.name` and `.getvalue()`). Supports .txt, .pdf, and .docx.

    Returns:
        str: Extracted plain text.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    raw = uploaded_file.getvalue()

    if ext == ".txt":
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("utf-8", errors="replace")

    elif ext == ".pdf":
        reader = pypdf.PdfReader(io.BytesIO(raw))
        extracted_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text)
        return "\n".join(extracted_text)

    elif ext == ".docx":
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError(
            f"Unsupported file format: '{ext}'. Supported formats are .txt, .pdf, and .docx"
        )


def run_compliance_pipeline(
    contract_text: str,
    contract_name: str,
    groq_client,
    qdrant_client,
    embedder,
    top_k_policies: int = 2,
    model_name: str = DEFAULT_MODEL,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    on_step: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """Executes the full ClauseGuard compliance pipeline against contract text
    already extracted in memory.

    Pipeline sequence:
      1. Intake Agent — extracts clauses into structured JSON.
      2. Retrieval Agent — queries the session's Qdrant collection for policies.
      3. Risk-Scoring Agent — evaluates risk for each clause.

    Args:
        contract_text: Already-extracted contract text (see
            `extract_text_from_upload`).
        contract_name: Display name of the contract (e.g. uploaded filename).
        groq_client: Session-scoped `groq.Groq` client (BYOK).
        qdrant_client: Session-scoped, in-memory `QdrantClient`.
        embedder: Shared embedding model (`.encode(text)`).
        top_k_policies: Number of policy passages to retrieve per clause.
        model_name: Groq model to use for both LLM agents.
        collection_name: Qdrant collection holding this session's policies.
        on_step: Optional `callback(step_number, label)` invoked before each
            stage starts, so a UI layer can drive spinners/status without
            this module depending on Streamlit.

    Returns:
        dict: Final report — 'contract_name', 'document_type', and
        'evaluations' (a list of scored clauses).
    """

    def _step(n: int, label: str) -> None:
        if on_step:
            on_step(n, label)

    _step(1, "Intake Agent — extracting risk-bearing clauses")
    intake_data = run_intake_agent(contract_text, groq_client, model_name=model_name)

    _step(2, "Retrieval Agent — searching indexed policies")
    retrieval_data = run_retrieval_agent(
        intake_data,
        qdrant_client,
        embedder,
        top_k=top_k_policies,
        collection_name=collection_name,
    )

    _step(3, "Risk-Scoring Agent — evaluating compliance")
    final_report = run_risk_scoring_agent(
        retrieval_data, contract_name, groq_client, model_name=model_name
    )

    return final_report
