"""
Async RAG context lookup against the local knowledge_rag service.

Voice constraint: this sits on the hot path between "user stopped talking"
and "LLM starts thinking", so both lookups run concurrently under one hard
time budget (config.rag_timeout_s). A slow or down RAG service degrades to
no context — never to a slow answer.
"""

from __future__ import annotations

import asyncio

import httpx

_MENU_TOP_K = 3
_CHUNK_TOP_K = 4


async def build_rag_context(config, user_text: str) -> str:
    if not config.rag_business_id:
        return ""

    async def menu_lookup(client: httpx.AsyncClient) -> str:
        resp = await client.post(f"{config.rag_base_url}/menu-search", json={
            "business_id": config.rag_business_id,
            "phrase": user_text,
            "top_k": _MENU_TOP_K,
        })
        if resp.status_code != 200:
            return ""
        matches = resp.json().get("matches", [])
        lines = []
        for m in matches:
            parts = [f"{m.get('category')}: {m.get('name')}" if m.get("category") else m.get("name", "")]
            if m.get("price"):
                parts.append(f"Price: {m['price']}")
            variants = ", ".join(
                f"{v.get('name', '')} {v.get('price', '')}".strip()
                for v in m.get("variants", []) if v.get("name")
            )
            if variants:
                parts.append(f"Variants: {variants}")
            lines.append(" | ".join(p for p in parts if p))
        return ("MENU ITEMS:\n" + "\n".join(lines)) if lines else ""

    async def chunk_lookup(client: httpx.AsyncClient) -> str:
        resp = await client.post(f"{config.rag_base_url}/vector-query", json={
            "business_id": config.rag_business_id,
            "text": user_text,
            "top_k": _CHUNK_TOP_K,
        })
        if resp.status_code != 200:
            return ""
        docs = [r["document"] for r in resp.json().get("results", []) if "document" in r]
        return ("BUSINESS KNOWLEDGE:\n" + "\n\n".join(docs)) if docs else ""

    sections: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=config.rag_timeout_s) as client:
            results = await asyncio.wait_for(
                asyncio.gather(menu_lookup(client), chunk_lookup(client),
                               return_exceptions=True),
                timeout=config.rag_timeout_s,
            )
        sections = [r for r in results if isinstance(r, str) and r]
    except (asyncio.TimeoutError, Exception):
        return ""

    if not sections:
        return ""
    return (
        "\n\n---\n[Retrieved context — use this to answer accurately. "
        "Do NOT mention that you used a database or context.]\n"
        + "\n\n".join(sections) + "\n---"
    )
