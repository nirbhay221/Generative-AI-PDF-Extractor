from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    question: str
    original_question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str
    retry_count: int
