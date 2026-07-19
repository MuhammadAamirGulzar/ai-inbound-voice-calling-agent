"""
StreamingLLM — token-streaming chat completion against any
OpenAI-compatible endpoint (Groq, OpenAI, Ollama, vLLM, ...).

The pipeline consumes LLMEvents:

    token       incremental text (feed to the sentence chunker)
    tool_call   the model asked to transfer_call / end_call
    done        stream finished
    error       provider failure — the session speaks a fallback line

Tool calls are how call control stays inside the conversation: the model
decides "this needs a human" and emits transfer_call with a summary that
the transfer layer whispers to the receiving agent (warm transfer).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI

# One HTTP client per (endpoint, key) for the whole process. A per-call
# client re-handshakes TLS on every turn: httpx's default keep-alive is
# 5 s, and the caller talks longer than that between turns — measured as
# +1.5-2.5 s of LLM TTFT on every single turn.
_shared_clients: dict = {}


def _get_client(base_url: str, api_key: str) -> AsyncOpenAI:
    cache_key = (base_url, api_key)
    if cache_key not in _shared_clients:
        http_client = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_connections=100,
                                max_keepalive_connections=20,
                                keepalive_expiry=120.0),
        )
        _shared_clients[cache_key] = AsyncOpenAI(
            base_url=base_url, api_key=api_key or "none",
            http_client=http_client, timeout=30.0)
    return _shared_clients[cache_key]

CALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transfer_call",
            "description": (
                "Transfer the caller to a human staff member. ONLY use when "
                "the caller explicitly asks for a human, agent or manager, "
                "or is angry / complaining about service, or demands a "
                "refund. NEVER use this for ordinary business you handle "
                "yourself: menu questions, prices, delivery areas or times, "
                "opening hours, placing or confirming an order. When in "
                "doubt, answer the caller yourself instead of transferring."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why the call is being transferred."},
                    "summary": {"type": "string", "description": "One-sentence handoff summary for the human agent (caller need, any order details so far)."},
                },
                "required": ["reason", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": (
                "Hang up politely. ONLY use after the conversation has "
                "clearly concluded: the order (if any) is confirmed with a "
                "delivery address and the caller said goodbye or that they "
                "are done. NEVER use it to avoid answering a question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "farewell": {"type": "string", "description": "Final goodbye sentence to speak before hanging up."},
                },
                "required": ["farewell"],
            },
        },
    },
]


@dataclass
class LLMEvent:
    kind: str                      # token | tool_call | done | error
    text: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


class StreamingLLM:
    def __init__(self, config):
        self.config = config
        self.client = _get_client(config.llm_base_url, config.llm_api_key)

    async def stream_reply(self, messages: list[dict],
                           tools_enabled: bool = True) -> AsyncIterator[LLMEvent]:
        kwargs = dict(
            model=self.config.llm_model,
            messages=messages,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            stream=True,
        )
        if tools_enabled:
            kwargs["tools"] = CALL_TOOLS
            kwargs["tool_choice"] = "auto"

        # Streamed tool-call fragments accumulate here until finish_reason.
        pending_tools: dict[int, dict] = {}
        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield LLMEvent(kind="token", text=delta.content)
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        slot = pending_tools.setdefault(
                            tc.index, {"name": "", "arguments": ""})
                        if tc.function and tc.function.name:
                            slot["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        except Exception as e:
            yield LLMEvent(kind="error", text=str(e))
            return

        for slot in pending_tools.values():
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            yield LLMEvent(kind="tool_call", tool_name=slot["name"], tool_args=args)
        yield LLMEvent(kind="done")
