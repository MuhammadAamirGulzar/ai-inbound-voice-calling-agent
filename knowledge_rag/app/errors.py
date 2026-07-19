"""
Turns any exception into a message that's never blank, plus logs the full
traceback to the console.

Why this exists: some exception types — notably timeout/connection
exceptions raised by the HTTP client underneath the `openai` package (used
to talk to the Ollama endpoint) — stringify to an EMPTY string (`str(e) ==
""`). Without this, an ingestion failure from a plain timeout showed up in
the console/API/CLI as just "Ingestion failed:" with nothing after the
colon, which isn't useful to anyone. This wraps that so there's always at
least the exception type shown, and always a full traceback in the
terminal for real debugging.
"""
import traceback


def describe_exception(e: BaseException) -> str:
    message = str(e).strip()
    if message:
        return f"{type(e).__name__}: {message}"
    return (
        f"{type(e).__name__} (the library raised this with no message attached — "
        f"common for a request timeout or a cold-starting model). Check the terminal "
        f"running this process for the full traceback, or try again in a moment."
    )


def log_exception(context: str, e: BaseException) -> None:
    """Print a clear one-liner plus the full traceback to the console."""
    print(f"[error] {context}: {describe_exception(e)}")
    print(traceback.format_exc())
