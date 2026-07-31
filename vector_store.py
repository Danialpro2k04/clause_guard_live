"""
Vector store adapter for the live, multi-tenant ClauseGuard deployment.
------------------------------------------------------------------------
Replaces `server/mcp_server.py`'s Qdrant tools for the hosted app. The
original dev version opened one *persistent, on-disk* Qdrant database
(`QdrantClient(path=...)`) shared by every process on the machine — fine for
a single local user, but unsafe for a public multi-tenant deployment where
concurrent visitors must never see each other's policy documents.

Here, every function takes the Qdrant client and embedder as arguments
instead of importing/instantiating global ones. The caller (app.py) hands in
a client built as `QdrantClient(":memory:")` and stored in
`st.session_state["qdrant_client"]`, so each browser session gets its own
isolated, in-memory vector index that simply evaporates when the session
ends — nothing is ever written to server disk.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

DEFAULT_COLLECTION_NAME = "user_policies"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Splits raw policy text into overlapping passages for embedding.

    Same defaults (500 chars / 50 overlap) as the original ingest_corpus.py,
    kept for context retention across chunk boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def ensure_fresh_collection(
    qdrant_client, collection_name: str = DEFAULT_COLLECTION_NAME
) -> None:
    """(Re)creates an empty collection. Re-indexing always starts clean —
    there is no incremental/partial update, so a visitor who re-uploads a
    corrected policy file doesn't end up with stale duplicate passages."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name in existing:
        qdrant_client.delete_collection(collection_name)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


def ingest_policy_files(
    qdrant_client,
    embedder,
    uploaded_files: list[Any],
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> dict:
    """Chunks, embeds, and upserts a batch of uploaded policy files into the
    session's in-memory Qdrant collection.

    Args:
        qdrant_client: A QdrantClient (expected to be the session's
            in-memory instance, e.g. `st.session_state["qdrant_client"]`).
        embedder: Any object exposing `.encode(text) -> vector` (a
            SentenceTransformer instance in practice).
        uploaded_files: Streamlit `UploadedFile` objects (`.name`,
            `.getvalue()`) — read entirely in memory, never saved to disk.
        collection_name: Qdrant collection to (re)populate.

    Returns:
        dict with 'num_files', 'num_chunks', and a 'per_file' breakdown —
        used to render the "Successfully indexed N passages across M
        files" confirmation in the UI.
    """
    ensure_fresh_collection(qdrant_client, collection_name)

    points: list[PointStruct] = []
    per_file_counts: dict[str, int] = {}
    point_id = 1

    for uploaded_file in uploaded_files:
        raw = uploaded_file.getvalue()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = raw.decode("utf-8", errors="replace")

        chunks = chunk_text(content)
        per_file_counts[uploaded_file.name] = len(chunks)

        for chunk_idx, chunk in enumerate(chunks):
            vector = embedder.encode(chunk).tolist()
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": uploaded_file.name,
                        "chunk_id": chunk_idx,
                    },
                )
            )
            point_id += 1

    if points:
        qdrant_client.upsert(collection_name=collection_name, points=points)

    return {
        "num_files": len(uploaded_files),
        "num_chunks": len(points),
        "per_file": per_file_counts,
    }


def search_policy_docs(
    qdrant_client,
    embedder,
    query_text: str,
    limit: int = 3,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> str:
    """Searches the session's indexed policies for passages relevant to a
    compliance statement. Same tool contract as the original
    `mcp_server.search_policy_docs`, just parameterized instead of relying on
    a shared global client + embedder.

    Returns a formatted string (not raw hits) since this is consumed
    directly as LLM context by the Risk-Scoring Agent.
    """
    try:
        existing = [c.name for c in qdrant_client.get_collections().collections]
    except Exception as e:
        return f"Error querying Qdrant database: {e}"

    if collection_name not in existing:
        return "No policy documents have been indexed yet for this session."

    query_vector = embedder.encode(query_text).tolist()

    try:
        results = qdrant_client.query_points(
            collection_name=collection_name, query=query_vector, limit=limit
        ).points
    except Exception as e:
        return f"Error querying Qdrant database: {e}"

    if not results:
        return "No relevant policy documents found matching the query."

    formatted_passages = []
    for idx, hit in enumerate(results, start=1):
        source = hit.payload.get("source", "Unknown File")
        text = hit.payload.get("text", "")
        formatted_passages.append(
            f"--- [Policy Result {idx}] (Source: {source} | Similarity Score: {hit.score:.2f}) ---\n{text}"
        )

    return "\n\n".join(formatted_passages)
