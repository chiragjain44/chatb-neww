# ingest.py
import os
import uuid
import psycopg2
import json
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
from embeddings_utils import chunk_text, embed_texts
from chroma_utils import get_chroma_client, get_or_create_collection, upsert_documents

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "ragdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "raguser")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "changeme")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))

def fetch_documents(conn, limit=0):
    cur = conn.cursor()
    q = "SELECT id, title, content, metadata FROM documents"
    if limit and limit > 0:
        q += f" LIMIT {limit}"
    cur.execute(q)
    rows = cur.fetchall()
    return rows

def connect_db():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    return conn

def ingest_all(limit=0):
    conn = connect_db()
    rows = fetch_documents(conn, limit=limit)
    client = get_chroma_client()
    collection = get_or_create_collection(client, CHROMA_COLLECTION_NAME)

    all_ids = []
    all_docs = []
    all_metadatas = []
    # We'll batch embeddings to BATCH_SIZE after chunking
    for doc_id, title, content, metadata in tqdm(rows, desc="docs"):
        metadata_obj = metadata if metadata else {}
        # Compose a doc_text: include title + content
        doc_text = f"Title: {title}\n\n{content}"
        chunks = chunk_text(doc_text)
        for idx, chunk in enumerate(chunks):
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}_{idx}"))
            md = {"source_doc_id": doc_id, "chunk_index": idx, "title": title}
            if isinstance(metadata_obj, dict):
                md.update(metadata_obj)
            all_ids.append(pid)
            all_docs.append(chunk)
            all_metadatas.append(md)

            # When we reach BATCH_SIZE, embed and upsert 
            if len(all_docs) >= BATCH_SIZE:
                embeddings = embed_texts(all_docs)
                upsert_documents(collection, all_ids, all_metadatas, all_docs, embeddings)
                all_ids = []
                all_docs = []
                all_metadatas = []

    # Final remaining
    if all_docs:
        embeddings = embed_texts(all_docs)
        upsert_documents(collection, all_ids, all_metadatas, all_docs, embeddings)

    conn.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_all(limit=0)  # set limit for testing small set
