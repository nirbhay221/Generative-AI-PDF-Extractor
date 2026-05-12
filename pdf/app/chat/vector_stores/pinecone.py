from langchain_pinecone import PineconeVectorStore
from app.chat.embeddings.openai import embeddings
from pinecone import Pinecone as PineconeClient
import os

_pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
_index = _pc.Index(os.getenv("PINECONE_INDEX_NAME"))

vector_stores = PineconeVectorStore(index=_index, embedding=embeddings)

def build_retriever(chat_args, k):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}, "k": k}
    return vector_stores.as_retriever(search_kwargs=search_kwargs)
