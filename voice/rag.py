"""
Async RAG context lookup against the local knowledge_rag service.

Voice constraint: this sits on the hot path between "user stopped talking"
and "LLM starts thinking", so both lookups run concurrently under one hard
time budget (config.rag_timeout_s). A slow or down RAG service degrades to
no context — never to a slow answer.
"""

from __future__ import annotations

import asyncio
import time

import httpx

_MENU_TOP_K = 3
_CHUNK_TOP_K = 4

# Circuit breaker: a down/hanging RAG service must not tax every turn.
# (Measured: on Windows a refused localhost connect burns ~1.7 s — the
# whole lookup budget — on every single turn until something gives.)
#
# Recovery happens OFF the hot path: once open, live turns skip RAG
# immediately and a background probe of /health decides when to close the
# breaker again. The old scheme ("cooldown expiry closes it") made a real
# caller pay the ~1.7 s discovery cost twice after every cooldown.
_BREAKER_THRESHOLD = 2
_BREAKER_COOLDOWN_S = 30.0
_PROBE_TIMEOUT_S = 1.0
_consecutive_failures = 0
_breaker_open = False
_last_probe_at = 0.0
_probe_task = None


def rag_healthy() -> bool:
    """False while the breaker is open (recent consecutive failures)."""
    return not _breaker_open


def record_rag_result(ok: bool):
    global _consecutive_failures, _breaker_open
    if ok:
        _consecutive_failures = 0
        _breaker_open = False
    else:
        _consecutive_failures += 1
        if _consecutive_failures >= _BREAKER_THRESHOLD:
            _breaker_open = True


async def _probe_health(base_url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            resp = await client.get(f"{base_url}/health")
        return resp.status_code == 200
    except Exception:
        return False


async def _reprobe(base_url: str):
    if await _probe_health(base_url):
        record_rag_result(True)


def _maybe_reprobe(base_url: str):
    """Kick a background /health probe at most once per cooldown."""
    global _last_probe_at, _probe_task
    now = time.monotonic()
    if now - _last_probe_at < _BREAKER_COOLDOWN_S:
        return
    if _probe_task is not None and not _probe_task.done():
        return
    _last_probe_at = now
    _probe_task = asyncio.create_task(_reprobe(base_url))


async def startup_probe(base_url: str, attempts: int = 6,
                        interval_s: float = 3.0) -> bool:
    """
    Establish breaker state at boot, before the first call. Retries for a
    bit to let an auto-started sidecar finish booting; if RAG stays down,
    the breaker is opened so no caller ever pays the discovery cost.
    """
    for attempt in range(attempts):
        if await _probe_health(base_url):
            record_rag_result(True)
            return True
        if attempt < attempts - 1:
            await asyncio.sleep(interval_s)
    for _ in range(_BREAKER_THRESHOLD):
        record_rag_result(False)
    return False


async def build_rag_context(config, user_text: str) -> str:
    if not config.rag_business_id:
        return ""
    if not rag_healthy():
        _maybe_reprobe(config.rag_base_url)
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
        # Service is "up" if at least one lookup came back at all.
        record_rag_result(any(isinstance(r, str) for r in results))
        sections = [r for r in results if isinstance(r, str) and r]
    except (asyncio.TimeoutError, Exception):
        record_rag_result(False)
        return ""

    if not sections:
        return ""
    return (
        "\n\n---\n[Retrieved context — use this to answer accurately. "
        "Do NOT mention that you used a database or context.]\n"
        + "\n\n".join(sections) + "\n---"
    )
