"""
rag_pipeline.py
This file builds the retrieval half of RAG: turning the FAQ text file into
searchable chunks stored in a local vector database.

Everything here runs locally and for free, using ChromaDB's built-in 
ONNX runtime for embeddings. This avoids downloading PyTorch (~2.5GB) 
and sentence-transformers, keeping the footprint tiny.
"""
import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "faq_chunks"


class LocalLightweightEmbeddings(Embeddings):
    """
    Uses ChromaDB's built-in ONNX embedding function (all-MiniLM-L6-v2).
    This avoids downloading PyTorch and sentence-transformers.
    """
    def __init__(self):
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]


def chunk_faq_document(file_path: str) -> list[Document]:
    """Split the FAQ file into one chunk per Q&A pair."""
    with open(file_path, "r") as f:
        text = f.read()
    
    raw_chunks = text.split("\nQ:")
    documents = []
    for i, chunk in enumerate(raw_chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        if i > 0:
            chunk = "Q: " + chunk
        documents.append(Document(page_content=chunk, metadata={"chunk_id": i}))
    return documents


def get_embedding_model() -> Embeddings:
    return LocalLightweightEmbeddings()


def build_or_load_vector_store(faq_path: str) -> Chroma:
    """
    If a vector store already exists on disk, just load it. Otherwise
    chunk the FAQ, embed the chunks, and create a new one.
    """
    embeddings = get_embedding_model()
    
    store_already_exists = os.path.isdir(VECTOR_STORE_DIR) and os.listdir(VECTOR_STORE_DIR)
    
    if store_already_exists:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=VECTOR_STORE_DIR,
        )
        
    documents = chunk_faq_document(faq_path)
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )
    return vector_store


def retrieve_chunks(vector_store: Chroma, query: str, k: int = 3) -> list[str]:
    """Return the top-k most relevant FAQ chunks for a query."""
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]