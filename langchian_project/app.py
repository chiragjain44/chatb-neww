import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from chroma_utils import get_chroma_vectorstore

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "6"))

app = FastAPI(title="LangChain RAG Chatbot")

# Initialize vectorstore retriever
vectordb = get_chroma_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# LLM
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)

# RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # can use "refine" or "map_reduce" for larger docs
    return_source_documents=True
)

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_rag(question: Question):
    if not question.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        response = qa_chain({"query": question.query})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

    answer = response.get("result", "")
    sources = [
        {"content": doc.page_content[:200] + "...", "metadata": doc.metadata}
        for doc in response.get("source_documents", [])
    ]
    return {"answer": answer, "sources": sources}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
