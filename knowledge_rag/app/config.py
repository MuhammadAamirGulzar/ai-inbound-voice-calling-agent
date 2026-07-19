"""
Central configuration: model names, timeouts, and data directory paths.

Everything here is read from environment variables (see .env.example) — no
secrets or credentials are hardcoded in the codebase.

This pipeline runs every AI step (reading images/scanned PDFs, structuring
extracted text into JSON, and generating RAG answers) through Qwen2.5-VL
served by Ollama on a separate GPU machine, reached over a tunnel (ngrok or
similar). See app/llm_client.py for the client itself, and the README for a
step-by-step Ollama + tunnel setup guide.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# --- Qwen2.5-VL via Ollama, on a separate GPU machine, reached through a tunnel ---
# OpenAI-compatible base URL, WITH a trailing /v1. See .env.example for the
# full Ollama install + tunnel setup walkthrough.
MODEL_ENDPOINT_URL = os.getenv("MODEL_ENDPOINT_URL", "")

# Single vision-language model used for BOTH plain text calls (structuring,
# RAG answers) and vision calls (reading menu photos / scanned PDF pages).
# Must exactly match the tag shown by `ollama list` on the GPU machine.
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5vl:7b")

# Ollama ignores this value, but the OpenAI-compatible client requires
# *some* non-empty string in the Authorization header.
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "ollama")

# Per-request timeout (seconds) for calls to the model endpoint. Paired
# with the retry-with-backoff decorator in app/retry.py so a single slow
# request can't hang the whole ingestion indefinitely. Kept generous by
# default for slow tunnel + local-GPU round trips.
MODEL_REQUEST_TIMEOUT = float(os.getenv("MODEL_REQUEST_TIMEOUT", "180"))

# --- Embeddings (local, CPU, no API key, no network call) ---
# Deliberately kept local rather than routed through Hugging Face's hosted
# API: this model is tiny (~80MB) and fast on CPU, and it's on the hot path
# the voice agent calls *during a live call* (see app/vectorstore.py ->
# query_menu_items). Adding network latency and hosted rate limits to that
# lookup would be a bad trade for a real-time voice agent. It's still a
# Hugging Face model (downloaded once from the Hub) — it's just run
# locally instead of over the API.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Data directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")             # original uploaded files, per business
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")  # structured JSON output, per business
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")     # vector store persistence

for _dir in (RAW_DIR, PROCESSED_DIR, CHROMA_DIR):
    os.makedirs(_dir, exist_ok=True)
