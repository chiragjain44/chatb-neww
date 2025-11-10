# app.py
import os
import openai
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List
from embeddings_utils import embed_texts
from chroma_utils import get_chroma_client, get_or_create_collection, query_collection

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")
TOP_K = int(os.getenv("TOP_K", "8"))
MAX_CONTEXT_ITEMS = int(os.getenv("MAX_CONTEXT_ITEMS", "6"))

openai.api_key = OPENAI_API_KEY

app = FastAPI(title="RAG Chatbot")

client = get_chroma_client()
collection = get_or_create_collection(client, CHROMA_COLLECTION_NAME)

class AskRequest(BaseModel):
    question: str
    top_k: int = TOP_K

def build_prompt(question: str, contexts: List[str]) -> str:
    """
    Build a prompt for the LLM that provides contexts and asks the model to answer using only the contexts.
    """
    header = (
        "You are an assistant that answers user questions using only the provided context snippets. "
        "If the answer is not contained in the context, say 'I don't know' or ask for clarification.\n\n"
    )
    ctx_text = "\n\n---\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)])
    prompt = f"{header}Context:\n{ctx_text}\n\nUser Question: {question}\n\nAnswer succinctly and cite the context numbers if used (e.g., [Context 1])."
    return prompt

@app.post("/ask")
async def ask(req: AskRequest):
    question = req.question
    top_k = req.top_k

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")

    # 1) embed the query
    q_emb = embed_texts([question])[0]

    # 2) retrieve from chroma
    res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=['documents', 'metadatas', 'distances', 'ids'])
    docs = res.get('documents', [[]])[0]  # list of docs
    metadatas = res.get('metadatas', [[]])[0]
    distances = res.get('distances', [[]])[0]
    ids = res.get('ids', [[]])[0]

    # select top contexts (truncate to MAX_CONTEXT_ITEMS)
    contexts = []
    for i, doc_text in enumerate(docs[:MAX_CONTEXT_ITEMS]):
        md = metadatas[i] if i < len(metadatas) else {}
        ctx = f"{doc_text}\n\n[metadata: {md}]"
        contexts.append(ctx)

    if not contexts:
        # If no contexts found, optionally ask LLM to say "I don't know" or fallback
        return {"answer": "I don't know. No relevant information found in the knowledge base.", "contexts": []}

    # 3) Build prompt and call LLM
    prompt = build_prompt(question, contexts)

    try:
        completion = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.0
        )
        answer = completion['choices'][0]['message']['content'].strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # return answer + traces (ids, distances)
    trace = [{"id": ids[i], "distance": distances[i], "metadata": (metadatas[i] if i < len(metadatas) else {})} for i in range(min(len(ids), MAX_CONTEXT_ITEMS))]
    return {"answer": answer, "contexts": contexts, "trace": trace}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
