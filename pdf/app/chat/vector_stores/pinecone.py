from langchain_pinecone import PineconeVectorStore
from app.chat.embeddings.openai import embeddings
from pinecone import Pinecone as PineconeClient
import os

_vector_store = None

def _get_vector_store():
    global _vector_store
    if _vector_store is None:
        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
        _vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    return _vector_store

def build_retriever(chat_args, k):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}, "k": k}
    return _get_vector_store().as_retriever(search_kwargs=search_kwargs)
