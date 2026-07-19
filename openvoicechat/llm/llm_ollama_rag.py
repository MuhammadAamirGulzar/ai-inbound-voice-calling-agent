"""
Chatbot_OllamaRAG — drop-in replacement for Chatbot_LLM.

Same interface (run() generator + post_process()) so twilio_routes.py and
sip_routes.py need only a one-line import swap.  Groq is not involved at all.

How it works per turn:
  1. Append the user utterance to self.messages (full history kept, same as
     Chatbot_LLM).
  2. Run two RAG lookups against the business's Chroma index (both are CPU-only
     local calls, sub-millisecond):
       a. vectorstore.query_menu_items()  — resolves spoken phrases to priced
          menu items (order-taking).
       b. vectorstore.query()             — retrieves general text chunks
          (policies, hours, delivery info, persona).
  3. Inject the combined RAG context into the system message for this turn
     only — the history itself is NOT polluted with context strings.
  4. Call knowledge_rag's llm_client.chat_text() (Ollama, OpenAI-compat API)
     with the full history + injected context turn, then yield the reply as a
     single chunk (Ollama's non-streaming path is used here because the
     existing conversation_loop already concatenates yielded chunks before
     passing to TTS — streaming adds no benefit on this hot path).
  5. post_process() appends the assistant reply to self.messages exactly as
     Chatbot_LLM does.

Prerequisites:
  - Ollama running locally (or tunnelled) with MODEL_NAME pulled.
  - MODEL_ENDPOINT_URL set in .env (default: http://localhost:11434/v1).
  - Business knowledge already ingested + menu published via the RAG console.
  - RAG_BUSINESS_ID set in .env (the business_id string used during ingestion).
"""

import os
import sys
import requests
from openai import OpenAI

from .base import BaseChatbot

# ---------------------------------------------------------------------------
# Talk to the RAG API service running on localhost.
# ---------------------------------------------------------------------------
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://127.0.0.1:8001")

# Number of menu matches and general chunks to retrieve per turn.
_MENU_TOP_K = 3
_CHUNK_TOP_K = 4


class Chatbot_OllamaRAG(BaseChatbot):
    """
    Conversational chatbot backed by local Ollama + RAG context, with full
    multi-turn message history.  Groq is not used.

    Parameters
    ----------
    sys_prompt : str
        System prompt loaded from the restaurant's AgentConfiguration.
    business_id : str
        The RAG business_id used during knowledge ingestion (e.g.
        "clove_cafe").  Falls back to the RAG_BUSINESS_ID env var if empty.
    """

    def __init__(self, sys_prompt: str = "", business_id: str = ""):
        self.business_id = business_id or os.getenv("RAG_BUSINESS_ID", "")
        self.messages: list[dict] = []
        if sys_prompt:
            self.messages.append({"role": "system", "content": sys_prompt})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rag_context(self, user_text: str) -> str:
        """
        Run both RAG lookups and return a single context block to inject into
        the current turn's user message.  Returns an empty string if the
        business_id is not set or the collections are empty.
        """
        if not self.business_id:
            return ""

        sections: list[str] = []

        # ── Menu item lookup ──────────────────────────────────────────────
        try:
            resp = requests.post(
                f"{RAG_BASE_URL}/menu-search",
                json={"business_id": self.business_id, "phrase": user_text, "top_k": _MENU_TOP_K},
                timeout=5.0
            )
            if resp.status_code == 200:
                menu_matches = resp.json().get("matches", [])
                if menu_matches:
                    lines = []
                    for m in menu_matches:
                        name = m.get("name", "")
                        cat = m.get("category", "")
                        price = m.get("price", "")
                        variants = m.get("variants", [])
                        # Build a compact one-liner per item
                        parts = [f"{cat}: {name}" if cat else name]
                        if price:
                            parts.append(f"Price: {price}")
                        if variants:
                            v_str = ", ".join(
                                f"{v.get('name','')} {v.get('price','')}".strip()
                                for v in variants
                                if v.get("name")
                            )
                            if v_str:
                                parts.append(f"Variants: {v_str}")
                        lines.append(" | ".join(parts))
                    sections.append("MENU ITEMS:\n" + "\n".join(lines))
        except Exception as e:
            print(f"[Chatbot_OllamaRAG] Menu lookup error (non-fatal): {e}")

        # ── General Q&A chunk lookup ──────────────────────────────────────
        try:
            resp = requests.post(
                f"{RAG_BASE_URL}/vector-query",
                json={"business_id": self.business_id, "text": user_text, "top_k": _CHUNK_TOP_K},
                timeout=5.0
            )
            if resp.status_code == 200:
                chunk_results = resp.json().get("results", [])
                if chunk_results:
                    chunks_text = "\n\n".join(res["document"] for res in chunk_results if "document" in res)
                    sections.append("BUSINESS KNOWLEDGE:\n" + chunks_text)
        except Exception as e:
            print(f"[Chatbot_OllamaRAG] Chunk lookup error (non-fatal): {e}")

        if not sections:
            return ""

        return (
            "\n\n---\n"
            "[Retrieved context — use this to answer accurately. "
            "Do NOT mention that you used a database or context.]\n"
            + "\n\n".join(sections)
            + "\n---"
        )

    def _build_messages_with_context(self, user_text: str) -> list[dict]:
        """
        Return a copy of self.messages with the current user turn appended,
        optionally enriched with inline RAG context.  self.messages itself is
        NOT modified here — post_process() does that after the call succeeds.
        """
        context = self._build_rag_context(user_text)
        augmented_user_content = user_text
        if context:
            augmented_user_content = user_text + context

        return self.messages + [{"role": "user", "content": augmented_user_content}]

    # ------------------------------------------------------------------
    # BaseChatbot interface
    # ------------------------------------------------------------------

    def run(self, input_text: str, minibot_args=None):
        """
        Yield the Ollama response as a single chunk (matches the generator
        protocol that twilio_routes / sip_routes expect).
        """
        print("Fetching response from Ollama (RAG-augmented).")
        messages = self._build_messages_with_context(input_text)

        # Build a single prompt string from the message list
        prompt = _messages_to_prompt(messages)

        endpoint = os.getenv("MODEL_ENDPOINT_URL", "http://localhost:11434/v1")
        api_key = os.getenv("MODEL_API_KEY", "ollama")
        model_name = os.getenv("MODEL_NAME", "qwen2.5vl:7b")
        timeout = float(os.getenv("MODEL_REQUEST_TIMEOUT", "180"))

        try:
            client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
                timeout=timeout,
            )
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[Chatbot_OllamaRAG] Ollama call failed: {e}")
            answer = "Maafi chahta hoon, abhi jawab dene mein mushkil ho rahi hai. Thodi der baad dobara poochein."

        # Append the raw user text (without injected context) to history so
        # the history stays clean for subsequent context injections.
        self.messages.append({"role": "user", "content": input_text})

        yield answer

    def post_process(self, response: str) -> str:
        """Append the assistant reply to history — same as Chatbot_LLM."""
        self.messages.append({"role": "assistant", "content": response})
        return response


# ---------------------------------------------------------------------------
# Utility: convert a messages list to a plain prompt string.
# Ollama's API IS OpenAI-compatible and supports the messages[] format, but
# we format the history ourselves so context and history are both preserved.
# ---------------------------------------------------------------------------

def _messages_to_prompt(messages: list[dict]) -> str:
    """
    Flatten a messages list into a readable transcript the model can follow.

    Format:
        [SYSTEM]
        <system content>

        [CUSTOMER]
        <user turn>

        [AGENT]
        <assistant turn>

        ...

        [CUSTOMER]
        <latest user turn with RAG context already injected>

        [AGENT]
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            parts.append(f"[CUSTOMER]\n{content}")
        elif role == "assistant":
            parts.append(f"[AGENT]\n{content}")

    # Trailing cue so the model knows to generate the next agent turn
    parts.append("[AGENT]")
    return "\n\n".join(parts)
