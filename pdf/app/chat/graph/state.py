from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    # Core conversation state
    question: str
    original_question: str
    chat_history: List[BaseMessage]
    documents: List[Document]
    answer: str

    # Adaptive RAG - query routing
    route: str               # "retrieval" | "direct" | "web_search"

    # CRAG - retrieval correction loop
    retry_count: int         # increments on rewrite; triggers web_search after 1 rewrite
    source: str              # "pdf" | "web" | "llm_direct"

    # Self-RAG - generation quality loop
    generate_retry_count: int   # increments when hallucination grader fails; max 3 attempts
    answer_retry_count: int     # increments when answer grader fails; max 1 rewrite
    hallucination_grade: str    # "yes" (grounded) | "no" (hallucinated)
    answer_grade: str           # "yes" (useful) | "no" (off-topic)
