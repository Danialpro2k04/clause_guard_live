"""
Intake Agent — Step 1 of the ClauseGuard pipeline.
-----------------------------------------------------
Reads raw contract text and extracts risk-bearing clauses as declarative
compliance statements, ready to be semantically matched against policy text
by the Retrieval Agent.

Refactored for the live multi-tenant deployment: the Groq client is built by
the caller from a per-visitor "bring your own key" and passed in here,
rather than being instantiated once at import time from a shared
`GROQ_API_KEY` environment variable. This is what makes it safe for many
concurrent visitors, each with their own key, to hit this module at once.
"""

from __future__ import annotations

import json

DEFAULT_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = (
    "You are an expert Legal Intake Compliance Agent. Your job is to analyze contract text "
    "and identify key risk-bearing clauses (e.g., data security, liability, data retention, IP).\n\n"
    "CRITICAL INSTRUCTION: Do NOT generate questions. Instead, formulate clear, direct, "
    "declarative STATEMENTS summarizing what the contract stipulates or permits. "
    "These statements will be semantically matched against company policy documents.\n\n"
    "Return ONLY valid JSON with this exact structure:\n"
    "{\n"
    '  "document_type": "NDA | MSA | Vendor Agreement | Unknown",\n'
    '  "clauses": [\n'
    "    {\n"
    '      "clause_title": "Title or summary of clause",\n'
    '      "clause_text": "Exact or verbatim snippet from the contract",\n'
    '      "compliance_statement": "Declarative statement of what the clause permits/requires '
    "(e.g., 'Data storage at rest is not required to be encrypted.')\"\n"
    "    }\n"
    "  ]\n"
    "}"
)


def run_intake_agent(
    contract_text: str, groq_client, model_name: str = DEFAULT_MODEL
) -> dict:
    """Parses contract text and identifies risk-bearing clauses.

    Args:
        contract_text: The text of the incoming contract.
        groq_client: An already-instantiated `groq.Groq` client, built from
            the current session's BYOK key.
        model_name: Groq model to use.

    Returns:
        dict with 'document_type' and a list of 'clauses' (each with
        'clause_title', 'clause_text', and 'compliance_statement').
    """
    response = groq_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this contract text:\n\n{contract_text}"},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)
