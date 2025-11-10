import os
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_collection")

def get_chroma_vectorstore(persist_directory: str = "./chroma_db"):
    embeddings = OpenAIEmbeddings(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    vectordb = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )
    return vectordb
