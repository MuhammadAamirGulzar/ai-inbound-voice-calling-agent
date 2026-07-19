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

from openai import AsyncOpenAI

CALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "transfer_call",
            "description": (
                "Transfer the caller to a human staff member. Use when the "
                "caller explicitly asks for a human, is upset, or has a "
                "request you cannot handle (complaints, refunds, complex "
                "changes to existing orders)."
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
                "Hang up politely. Use only after the conversation has "
                "clearly concluded (order confirmed and caller said goodbye, "
                "or caller asked to end the call)."
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
        self.client = AsyncOpenAI(
            base_url=config.llm_base_url,
            api_key=config.llm_api_key or "none",
            timeout=30.0,
        )

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
