"""
Small retry-with-backoff decorator for calls to the model endpoint (see
app/llm_client.py — Qwen2.5-VL via Ollama, on a separate GPU machine,
reached through a tunnel).

A brief tunnel hiccup, a momentarily busy GPU, or a cold-starting Ollama
process shouldn't crash an ingestion run or a live call. With this
decorator, the call pauses briefly and retries a few times before giving
up — turns a transient failure into a short delay instead of an error.

Usage:
    @retry_with_backoff()
    def call_model(...):
        ...
"""
import functools
import time


def _is_retryable(e: Exception) -> bool:
    """Some exceptions we want to retry (rate limits, transient overload,
    a model cold-starting) stringify with useful text; others — notably
    plain HTTP timeouts — stringify to an EMPTY string, so string-matching
    alone would miss them. This checks three signals: the message text,
    an HTTP status code if the exception carries one (e.g. an APIStatusError
    from the openai client), and the exception's class name (catches
    *Timeout*/*Connect* classes regardless of what, if anything, they put
    in their message)."""
    message = str(e).lower()
    if any(marker in message for marker in
           ("429", "rate limit", "quota", "resource exhausted", "503", "overloaded",
            "timeout", "timed out", "connection", "connect")):
        return True
    status = getattr(getattr(e, "response", None), "status_code", None)
    if status in (429, 503):
        return True
    if "timeout" in type(e).__name__.lower() or "connect" in type(e).__name__.lower():
        return True
    return False


def retry_with_backoff(max_attempts: int = 4, base_delay: float = 2.0, max_delay: float = 30.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                attempt += 1
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt >= max_attempts or not _is_retryable(e):
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    print(f"[retry] {fn.__name__} hit a transient error ({e.__class__.__name__}) — "
                          f"retrying in {delay:.0f}s (attempt {attempt}/{max_attempts})...")
                    time.sleep(delay)
        return wrapper
    return decorator
