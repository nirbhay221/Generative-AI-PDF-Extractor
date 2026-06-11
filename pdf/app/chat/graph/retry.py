import logging
import backoff


def _is_retriable(exc):
    """Only retry on transient network/rate-limit errors, not logic errors."""
    retriable = (ConnectionError, TimeoutError, OSError)
    try:
        from openai import RateLimitError, APIConnectionError, APITimeoutError
        retriable += (RateLimitError, APIConnectionError, APITimeoutError)
    except ImportError:
        pass
    return isinstance(exc, retriable)


def invoke_with_retry(chain, inputs):
    """Invoke a LangChain chain with exponential backoff on transient failures."""
    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=3,
        max_time=30,
        giveup=lambda e: not _is_retriable(e),
        on_backoff=lambda details: logging.warning(
            f"LLM call failed (attempt {details['tries']}/3), "
            f"retrying in {details['wait']:.1f}s - {type(details['exception']).__name__}"
        ),
    )
    def _call():
        return chain.invoke(inputs)

    return _call()
