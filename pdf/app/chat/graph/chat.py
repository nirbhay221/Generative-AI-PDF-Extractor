import logging
import threading
from queue import Queue

from flask import current_app
from langchain_core.messages import HumanMessage, AIMessage

from app.chat.callbacks.stream import StreamingHandler
from app.chat.memories.histories.sql_history import SqlMessageHistory


class GraphChat:
    def __init__(self, graph, history: SqlMessageHistory):
        self.graph = graph
        self.history = history

    def _initial_state(self, question: str) -> dict:
        return {
            "question": question,
            "original_question": question,
            "chat_history": self.history.messages,
            "documents": [],
            "answer": "",
            # Adaptive RAG
            "route": "retrieval",
            # CRAG
            "retry_count": 0,
            "source": "pdf",
            # Self-RAG
            "generate_retry_count": 0,
            "answer_retry_count": 0,
            "hallucination_grade": "",
            "answer_grade": "",
        }

    def run(self, question: str) -> str:
        result = self.graph.invoke(self._initial_state(question))
        answer = result.get("answer", "")
        self.history.add_message(HumanMessage(content=question))
        self.history.add_message(AIMessage(content=answer))
        return answer

    def stream(self, question: str):
        queue = Queue()
        handler = StreamingHandler(queue)
        app_context = current_app.app_context()

        def task():
            app_context.push()
            try:
                result = self.graph.invoke(
                    self._initial_state(question),
                    config={"callbacks": [handler]},
                )
                answer = result.get("answer", "")
                self.history.add_message(HumanMessage(content=question))
                self.history.add_message(AIMessage(content=answer))
            except Exception as e:
                logging.error(f"[GraphChat] graph execution failed: {e}")
            finally:
                # Always put the sentinel exactly once, after the entire graph
                # finishes (including any hallucination/answer grader retries).
                # StreamingHandler.on_llm_end no longer puts it, so this is
                # the single point of stream termination.
                queue.put(None)

        threading.Thread(target=task).start()

        while True:
            token = queue.get()
            if token is None:
                break
            yield token
