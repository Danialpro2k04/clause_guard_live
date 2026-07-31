"""
Risk-Scoring Agent — Step 3 of the ClauseGuard pipeline.
------------------------------------------------------------
Compares each clause against its retrieved policy context and assigns a
HIGH / MEDIUM / LOW risk verdict, a justification, and a suggested remedy.

Refactored for the live deployment in two ways:
  1. The Groq client is injected (BYOK), same as the Intake Agent.
  2. This no longer calls an MCP tool to append HIGH/MEDIUM findings to a
     shared `pending_reviews.json` on disk. It just returns the scored
     clauses; the caller (app.py) decides what to do with them — in the
     live app, that means appending to `st.session_state["pending_reviews"]`
     so findings stay scoped to the visitor's own session.
"""

from __future__ import annotations

import json

DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert Corporate Compliance Risk Assessor. Your task is to compare a "
    "proposed contract clause against retrieved internal company compliance policies.\n\n"
    "Assign one of the following risk levels:\n"
    "- HIGH: Direct violation or contradiction of company policy, severe legal/security risk.\n"
    "- MEDIUM: Ambiguous language, partial mismatch, or missing required protective terms.\n"
    "- LOW: Fully compliant with company policy, zero or minimal risk.\n\n"
    "Return ONLY valid JSON with this structure:\n"
    "{\n"
    '  "risk_level": "HIGH | MEDIUM | LOW",\n'
    '  "justification": "Clear, objective breakdown of why this risk score was assigned relative to company policy.",\n'
    '  "recommendation": "Suggested modification or action for the legal team."\n'
    "}"
)


def score_clause_risk(
    contract_name: str, clause_info: dict, groq_client, model_name: str = DEFAULT_MODEL
) -> dict:
    """Evaluates a single clause against its retrieved policy context.

    Args:
        contract_name: Name of the contract this clause came from.
        clause_info: dict with 'clause_title', 'clause_text', and
            'retrieved_policy_context' (from the Retrieval Agent).
        groq_client: Session-scoped `groq.Groq` client (BYOK).
        model_name: Groq model to use.

    Returns:
        dict with 'clause_title', 'clause_text', 'risk_level',
        'justification', and 'recommendation'.
    """
    clause_title = clause_info.get("clause_title", "Untitled Clause")
    clause_text = clause_info.get("clause_text", "")
    policy_context = clause_info.get("retrieved_policy_context", "")

    user_prompt = (
        f"CONTRACT CLAUSE TITLE: {clause_title}\n"
        f'CONTRACT CLAUSE TEXT:\n"{clause_text}"\n\n'
        f"RETRIEVED COMPANY POLICY CONTEXT:\n{policy_context}"
    )

    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    score_data = json.loads(response.choices[0].message.content)

    return {
        "clause_title": clause_title,
        "clause_text": clause_text,
        "risk_level": str(score_data.get("risk_level", "MEDIUM")).strip().upper(),
        "justification": score_data.get("justification", ""),
        "recommendation": score_data.get("recommendation", ""),
    }


def run_risk_scoring_agent(
    retrieval_payload: dict,
    contract_name: str,
    groq_client,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """Runs the risk-scoring agent across every retrieved clause in a payload."""
    clauses = retrieval_payload.get("clauses", [])
    scored_clauses = [
        score_clause_risk(contract_name, clause, groq_client, model_name=model_name)
        for clause in clauses
    ]
    return {
        "contract_name": contract_name,
        "document_type": retrieval_payload.get("document_type", "Unknown"),
        "evaluations": scored_clauses,
    }
