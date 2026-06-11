from langchain_core.callbacks import BaseCallbackHandler


class StreamingHandler(BaseCallbackHandler):
    def __init__(self, queue):
        self.queue = queue
        self.streaming_run_ids = set()

    def on_chat_model_start(self, serialized, messages, run_id, **kwargs):
        # Use .get so models that don't include "streaming" don't raise KeyError
        if serialized.get("kwargs", {}).get("streaming", False):
            self.streaming_run_ids.add(run_id)

    def on_llm_new_token(self, token, **kwargs):
        self.queue.put(token)

    def on_llm_end(self, response, run_id, **kwargs):
        # Do NOT put sentinel here. With Self-RAG the generate node can run
        # multiple times (hallucination retries). Closing the stream after the
        # first generate would hide all retry output. The sentinel is placed by
        # GraphChat.stream() in the task() finally block once the whole graph
        # finishes, so the stream always closes exactly once.
        self.streaming_run_ids.discard(run_id)

    def on_llm_error(self, error, **kwargs):
        # Don't close the stream here either. A grader or rewriter LLM error
        # should not kill the user's stream - GraphChat.stream() task() will
        # handle cleanup via its finally block.
        pass
