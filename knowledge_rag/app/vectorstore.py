"""
Vector store layer: local, free, persistent — no API key, no cost.

Uses:
  - sentence-transformers (all-MiniLM-L6-v2) to embed text locally on CPU
  - ChromaDB (persistent, on-disk) to store + search embeddings

Two Chroma "collections" per business, kept deliberately separate:

  business_<id>        general text chunks -> for open-ended Q&A / persona /
                        FAQ-style retrieval (app/rag.py). Good for fuzzy
                        questions, not precise enough for exact order-taking.

  business_<id>_menu    one entry per structured menu item (name + category +
                        description embedded, price/variants/customizations
                        kept as metadata) -> for the voice agent to resolve
                        what a caller said ("the spicy chicken thing") to a
                        real, priced menu item. This is what Module 5
                        (order-taking) should call — not the generic chunk
                        search above.

Menu items are only ever written here via `add_menu_items`, which `ingest.py`
now only calls AFTER a human has reviewed/published a batch — see
`ingest.publish_review()`. Nothing in this module enforces that on its own;
it's a contract with the caller, documented here so it isn't lost.

Every collection is namespaced per business_id, so retrieval never mixes
brands. `delete_business_data` drops both collections for a business_id.
"""
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass
import json
from typing import List, Tuple
from . import config

_embedder = None
_client = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"[vectorstore] Loading local embedding model '{config.EMBEDDING_MODEL_NAME}' (first run downloads it once)...")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client
def _collection_name(business_id: str) -> str:
    # Chroma collection names must be simple — normalize the business_id defensively
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in business_id)
    return f"business_{safe}"


def _menu_collection_name(business_id: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in business_id)
    return f"business_{safe}_menu"


def get_collection(business_id: str):
    client = get_client()
    return client.get_or_create_collection(name=_collection_name(business_id))


def get_menu_collection(business_id: str):
    client = get_client()
    return client.get_or_create_collection(name=_menu_collection_name(business_id))


def add_chunks(business_id: str, chunks: List[str], source_file: str) -> None:
    if not chunks:
        return
    collection = get_collection(business_id)
    embedder = get_embedder()
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{source_file}-{i}" for i in range(len(chunks))]
    metadatas = [{"source_file": source_file, "business_id": business_id} for _ in chunks]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def query(business_id: str, question: str, top_k: int = 5) -> List[Tuple[str, dict]]:
    collection = get_collection(business_id)
    if collection.count() == 0:
        return []
    embedder = get_embedder()
    q_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=min(top_k, collection.count()))
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(docs)
    
    filtered_results = []
    for doc, meta, dist in zip(docs, metas, distances):
        # Only return matches with L2 distance <= 1.2 to filter out irrelevant noise
        if dist <= 1.2:
            filtered_results.append((doc, meta))
    return filtered_results


def _normalize_variants(variants) -> List[dict]:
    """Accepts either the old [string] shape or the new [{name, price}]
    shape and normalizes to the latter, so older ingested JSON on disk
    doesn't crash this layer."""
    normalized = []
    for v in variants or []:
        if isinstance(v, dict):
            normalized.append({"name": v.get("name", ""), "price": v.get("price", "")})
        elif isinstance(v, str):
            normalized.append({"name": v, "price": ""})
    return normalized


def add_menu_items(business_id: str, menu_items: List[dict], source_file: str) -> None:
    """
    Embed each structured menu item individually (category + name +
    description) so the voice agent can resolve a spoken phrase to an exact,
    priced item — instead of hoping the item and its price land in the same
    text chunk. Category is included in the embedded text (not just as
    metadata) so that same-named items in different categories don't
    collide on retrieval.

    Re-ingesting the same source_file overwrites its previous items (upsert
    on a deterministic id), so correcting a menu via human review and
    re-publishing doesn't leave stale duplicate entries behind.

    IMPORTANT: this should only be called with reviewed/approved menu
    items — see ingest.publish_review(). Calling it directly with
    unreviewed, freshly-extracted items skips the human-review gate.
    """
    if not menu_items:
        return
    collection = get_menu_collection(business_id)
    embedder = get_embedder()

    texts, ids, metadatas = [], [], []
    for i, item in enumerate(menu_items):
        name = (item.get("item") or "").strip()
        if not name:
            continue
        category = (item.get("category") or "").strip()
        description = (item.get("description") or "").strip()
        embed_text = ". ".join(p for p in (category, name, description) if p)
        texts.append(embed_text)
        ids.append(f"{source_file}-menu-{i}")
        metadatas.append({
            "business_id": business_id,
            "source_file": source_file,
            "name": name,
            "category": category,
            "price": item.get("price") or "",
            "variants": json.dumps(_normalize_variants(item.get("variants"))),
            "customizations": json.dumps(item.get("customizations") or []),
        })

    if not texts:
        return
    embeddings = embedder.encode(texts).tolist()
    collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def delete_menu_items_for_source(business_id: str, source_file: str) -> None:
    """Remove all menu-item embeddings previously published for a given
    source file — used when a review is re-published with corrected items,
    or when an ingestion is deleted outright."""
    collection = get_menu_collection(business_id)
    existing = collection.get(where={"source_file": source_file})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def delete_chunks_for_source(business_id: str, source_file: str) -> None:
    """Remove all general text chunks previously published for a given
    source file — used when a review is re-published with corrected text,
    or when an ingestion is deleted outright."""
    collection = get_collection(business_id)
    existing = collection.get(where={"source_file": source_file})
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)


def query_menu_items(business_id: str, spoken_text: str, top_k: int = 3) -> List[dict]:
    """
    Given what a caller said (however phrased), return the closest matching
    real menu items with their exact price/variants/customizations attached.

    Intended use in the voice agent: try an exact/substring match against the
    cached menu first (fast, free, no vector search needed); fall back to
    this only when that fails or the utterance is ambiguous.
    """
    collection = get_menu_collection(business_id)
    if collection.count() == 0:
        return []
    embedder = get_embedder()
    q_embedding = embedder.encode([spoken_text]).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=min(top_k, collection.count()))
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0] if results.get("distances") else [0.0] * len(metas)

    matches = []
    for meta, dist in zip(metas, distances):
        # Only return matches with L2 distance <= 1.2 to filter out irrelevant noise
        if dist <= 1.2:
            matches.append({
                "name": meta.get("name", ""),
                "category": meta.get("category", ""),
                "price": meta.get("price", ""),
                "variants": json.loads(meta.get("variants", "[]")),
                "customizations": json.loads(meta.get("customizations", "[]")),
                "source_file": meta.get("source_file", ""),
                "match_distance": dist,
            })
    return matches


def reset_menu_collection(business_id: str):
    """Delete and recreate the menu collection to clear any corrupted indexes."""
    client = get_client()
    name = _menu_collection_name(business_id)
    try:
        client.delete_collection(name=name)
    except Exception:
        pass
    return client.get_or_create_collection(name=name)


def delete_business_data(business_id: str) -> None:
    """Drop both collections for a business_id entirely (used when an
    owner wants to fully reset/remove a business's ingested data)."""
    client = get_client()
    for name in (_collection_name(business_id), _menu_collection_name(business_id)):
        try:
            client.delete_collection(name=name)
        except Exception:
            pass  # collection may not exist yet — fine
