from app.chat.tracing.langfuse import langfuse

class TraceableChain:
    def __call__(self, *args, **kwargs):
        if langfuse:
            trace = langfuse.trace(
                id=self.metadata["conversation_id"],
                metadata=self.metadata
            )
            callbacks = kwargs.get("callbacks", [])
            callbacks.append(trace.get_langchain_handler())
            kwargs["callbacks"] = callbacks

        return super().__call__(*args, **kwargs)
