"""
rag_agent.py

RAGAgent is intentionally thin - all the real retrieval logic lives in
rag_pipeline.py. This file's only job is to call that pipeline and package
the result into the RagResult structure the orchestrator expects.
"""

from app.models import RagResult
from app.rag_pipeline import retrieve_chunks


def run_rag_agent(vector_store, query: str) -> RagResult:
    chunks = retrieve_chunks(vector_store, query, k=3)
    return RagResult(chunks=chunks, query_used=query)
