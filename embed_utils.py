# chroma_utils.py
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")

def get_chroma_client():
    """
    Returns a chroma client instance.
    By default uses local in-process chroma; for remote server adjust settings/environment.
    """
    settings = Settings()  # default in-process
    client = chromadb.Client(settings=settings)
    return client

def get_or_create_collection(client: chromadb.Client, name: Optional[str] = None, embedding_dim: Optional[int] = None):
    col_name = name or CHROMA_COLLECTION_NAME
    try:
        collection = client.get_collection(col_name)
    except Exception:
        collection = client.create_collection(col_name)
    return collection

def upsert_documents(collection, ids: List[str], metadatas: List[Dict[str, Any]], documents: List[str], embeddings: List[List[float]]):
    """
    Upsert into chroma collection. The collection must exist.
    """
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )

def query_collection(collection, query_embeddings: List[float], top_k: int = 8):
    """
    Query collection by embedding. Returns list of dicts with ids, documents, metadatas, distances.
    """
    res = collection.query(
        query_embeddings=[query_embeddings],
        n_results=top_k,
        include=['documents', 'metadatas', 'distances', 'ids']
    )
    # res fields: ids, distances, documents, metadatas
    return res
