"""
Centralized model client — talks to Qwen2.5-VL running under Ollama on a
separate GPU machine, reached through a tunnel (ngrok or similar).

Every AI call in this project (vision/OCR transcription, text structuring,
and RAG answer generation) goes through the two functions in this module:

    chat_text(prompt, ...)                 -> plain text-in / text-out
    chat_vision(prompt, image, ...)        -> text + image -> text

This is the ONLY module that talks to the model backend. Swapping the
endpoint, the model, or the backend entirely later means editing this one
file — nothing in extraction.py, structuring.py, or rag.py needs to change.

--------------------------------------------------------------------------
How this works (for readers new to this setup)
--------------------------------------------------------------------------
Ollama runs on a separate machine that has a GPU, and exposes an
OpenAI-compatible API at <endpoint>/v1/chat/completions on that machine.
Since it's a separate machine (not localhost), it needs to be exposed over
a tunnel (ngrok or similar) and that public tunnel URL pasted into
MODEL_ENDPOINT_URL in your `.env` file, e.g.:

    MODEL_ENDPOINT_URL=https://your-static-domain.ngrok-free.dev/v1

We talk to it with the standard `openai` python package pointed at that
`base_url` — no Ollama-specific SDK required, since Ollama speaks the same
wire protocol as the OpenAI API.

One model, MODEL_NAME (default `qwen2.5vl:7b`, configurable in `.env`, see
app/config.py), is used for everything: it's a vision-language model, so
it handles both plain-text calls (structuring extracted text into JSON,
generating RAG answers) and vision calls (transcribing menu photos and
scanned PDF pages) — there's no separate text-only model to configure.
See the README for the full Ollama install + tunnel setup guide, and for
how to point this at a different model or a bigger/smaller Qwen2.5-VL
variant.
"""
import base64
import io
import os
from typing import Optional

from openai import OpenAI
from PIL import Image

from . import config
from .retry import retry_with_backoff

# Ollama's context window (num_ctx) defaults to a small value (commonly
# 2048-4096 tokens depending on the model) UNLESS explicitly overridden per
# request via the "options" field. This is NOT the same thing as max_tokens/
# num_predict (which only caps the *output*) - num_ctx caps prompt + output
# COMBINED. Without setting this, Ollama silently truncates whatever doesn't
# fit (typically from the start of the prompt) with no error and no warning.
# For this app that means: a long menu's raw text can get silently cut down
# before the model ever sees the back half of it, and the amount cut varies
# run-to-run with tiny differences in extracted text length - which is why
# the SAME pdf can come out with a wildly different number of menu items
# between runs. Set generously here; lower it if the GPU doesn't have the
# VRAM to support it (Ollama will error clearly on OOM rather than silently
# truncating, which is a much easier problem to spot and fix).
MODEL_NUM_CTX = int(os.getenv("MODEL_NUM_CTX", "32768"))

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not config.MODEL_ENDPOINT_URL:
            raise RuntimeError(
                "MODEL_ENDPOINT_URL is not set. Point it at your tunneled Ollama "
                "endpoint, WITH a trailing /v1, e.g. "
                "https://your-static-domain.ngrok-free.dev/v1 (see .env.example)."
            )
        _client = OpenAI(
            base_url=config.MODEL_ENDPOINT_URL,
            api_key=config.MODEL_API_KEY,
            timeout=config.MODEL_REQUEST_TIMEOUT,
        )
    return _client


def _image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _connection_error(e: Exception) -> RuntimeError:
    return RuntimeError(
        f"Couldn't reach the model at MODEL_ENDPOINT_URL={config.MODEL_ENDPOINT_URL!r}. "
        f"Checklist: (1) is Ollama running on the GPU machine ('ollama serve' / "
        f"the Ollama app)? (2) is the tunnel (ngrok or similar) still up? tunnels "
        f"restart with a new URL unless you're on a static/reserved domain. "
        f"(3) does MODEL_ENDPOINT_URL in .env end with /v1 and match the current "
        f"tunnel URL exactly? Original error: {e}"
    )


def _is_connection_error(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return "connect" in name or "connect" in str(e).lower()


@retry_with_backoff()
def _text_completion_call(prompt: str, temperature: float, max_tokens: int):
    client = get_client()
    try:
        return client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"options": {"num_ctx": MODEL_NUM_CTX}},
        )
    except Exception as e:
        if _is_connection_error(e):
            raise _connection_error(e) from e
        raise


@retry_with_backoff()
def _vision_completion_call(prompt: str, image: Image.Image, temperature: float, max_tokens: int):
    client = get_client()
    try:
        return client.chat.completions.create(
            model=config.MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
                ],
            }],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body={"options": {"num_ctx": MODEL_NUM_CTX}},
        )
    except Exception as e:
        if _is_connection_error(e):
            raise _connection_error(e) from e
        raise


def chat_text(prompt: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """Send a plain text prompt to the configured MODEL_NAME and return
    the generated text (never raises on empty content, returns "")."""
    text, _finish_reason = chat_text_ex(prompt, temperature, max_tokens)
    return text


def chat_text_ex(prompt: str, temperature: float = 0.0, max_tokens: int = 4096):
    """Same as chat_text, but also returns the completion's finish_reason
    ("stop" = model finished naturally, "length" = it was cut off because
    max_tokens was reached). Callers that need to know whether a response
    might be missing content because of a hit token cap - e.g. structuring
    a big menu into JSON - should use this instead of chat_text, since a
    truncated JSON response silently missing the back half of a menu is
    otherwise indistinguishable from a complete one."""
    response = _text_completion_call(prompt, temperature, max_tokens)
    choice = response.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason


def chat_vision(prompt: str, image: Image.Image, temperature: float = 0.0, max_tokens: int = 4096) -> str:
    """Send a text prompt + an image to the configured MODEL_NAME and
    return the generated text."""
    text, _finish_reason = chat_vision_ex(prompt, image, temperature, max_tokens)
    return text


def chat_vision_ex(prompt: str, image: Image.Image, temperature: float = 0.0, max_tokens: int = 4096):
    """Same as chat_vision, but also returns finish_reason - see chat_text_ex."""
    response = _vision_completion_call(prompt, image, temperature, max_tokens)
    choice = response.choices[0]
    return (choice.message.content or "").strip(), choice.finish_reason
