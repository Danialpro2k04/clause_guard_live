"""
Retrieval Agent — Step 2 of the ClauseGuard pipeline.
--------------------------------------------------------
Takes the Intake Agent's declarative compliance statements and, for each
one, searches the session's indexed policy corpus for relevant passages.

Refactored for the live deployment: previously this imported
`search_policy_docs` from `server.mcp_server`, which held a single
process-wide Qdrant client pointed at an on-disk database — one shared
policy corpus for everyone. That's replaced here with `vector_store`, which
takes the session's own in-memory Qdrant client and embedder as arguments,
so each visitor only ever searches their own uploaded policies.
"""

from __future__ import annotations

from vector_store import DEFAULT_COLLECTION_NAME, search_policy_docs


def run_retrieval_agent(
    intake_data: dict,
    qdrant_client,
    embedder,
    top_k: int = 2,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict:
    """Queries the session's policy index for each clause's compliance statement.

    Args:
        intake_data: Output of `run_intake_agent()`.
        qdrant_client: The session's in-memory QdrantClient.
        embedder: Shared embedding model (`.encode(text)`).
        top_k: Number of policy passages to retrieve per clause.
        collection_name: Qdrant collection to search.

    Returns:
        dict with 'document_type' and 'clauses', each clause now carrying a
        'retrieved_policy_context' string for the Risk-Scoring Agent.
    """
    evaluated_clauses = []
    clauses = intake_data.get("clauses", [])

    for clause in clauses:
        statement = clause.get("compliance_statement", "")
        retrieved_context = search_policy_docs(
            qdrant_client,
            embedder,
            statement,
            limit=top_k,
            collection_name=collection_name,
        )
        evaluated_clauses.append(
            {
                "clause_title": clause.get("clause_title", "Untitled Clause"),
                "clause_text": clause.get("clause_text", ""),
                "compliance_statement": statement,
                "retrieved_policy_context": retrieved_context,
            }
        )

    return {
        "document_type": intake_data.get("document_type", "Unknown"),
        "clauses": evaluated_clauses,
    }
