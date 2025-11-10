import os
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from chroma_utils import get_chroma_vectorstore
from db_utils import fetch_documents

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

def ingest_docs(limit=None):
    rows = fetch_documents(limit)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = []

    for row in tqdm(rows, desc="Processing"):
        doc_id, title, content, metadata = row
        metadata = metadata or {}
        base_text = f"Title: {title or 'Untitled'}\n\n{content}"
        chunks = splitter.split_text(base_text)
        for idx, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={"source_id": doc_id, "chunk_index": idx, **metadata}
            ))

    vectordb = get_chroma_vectorstore()
    vectordb.add_documents(docs)
    vectordb.persist()
    print(f"✅ Ingested {len(docs)} chunks into ChromaDB.")

if __name__ == "__main__":
    ingest_docs()
