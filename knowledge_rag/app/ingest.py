"""
Ingestion orchestration.

Pipeline for one uploaded file, for one business:

  1. Extract raw text      (app/extraction.py  — native / OCR / vision as needed)
  2. Structure it          (app/structuring.py — persona / policies / details / menu JSON)
  3. Chunk + embed + store general text for Q&A (app/chunking.py + app/vectorstore.py)
  4. Save the structured JSON to disk with status "pending_review"
  5. STOP — menu items are NOT embedded into the order-taking vector store yet.

A human then reviews the structured menu (see `list_pending_reviews` /
`get_review` / `publish_review` / `reject_review` below) and either publishes
it — which embeds the (possibly-corrected) menu items into the menu vector
store used for live order-taking — or rejects it, which does nothing further.

Why gate only the menu behind review and not the general Q&A text chunks:
the menu is what a caller's *order and bill* are built from, so a wrong,
un-reviewed price there is the costliest kind of mistake this system can
make. General Q&A retrieval already hedges with "I don't have that
information" rather than inventing answers, so it's lower-stakes to make
available immediately. This gate is exactly what Module 3/4 of the project
plan calls for ("human-review screen so extracted menu items can be
corrected before publishing").
"""
import json
import os
import re
import shutil
import time
import html as html_lib
from typing import List, Optional

from . import chunking, config, extraction, structuring, vectorstore


_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>\n]{0,60}>")


def _looks_like_html(text: str) -> bool:
    """Heuristic: a handful of stray '<' characters can show up in normal
    text (e.g. 'price < 500'), but real markup shows up as several actual
    tag-shaped tokens close together — a menu exported straight from a web
    page/HTML table is the case this catches."""
    return len(_HTML_TAG_RE.findall(text)) >= 5


def _html_to_plain_text(text: str) -> str:
    """Best-effort conversion of an HTML table/document dump into readable
    plain text, so downstream structuring and Q&A retrieval never see raw
    markup: row-ish tags become line breaks, cell-ish tags become a ' | '
    separator, everything else is stripped and entities are unescaped."""
    # Collapse whitespace/newlines that exist purely between tags (the
    # original document's indentation) so the row/cell separators below are
    # the only thing controlling line breaks — otherwise each cell of a
    # pretty-printed table ends up on its own line instead of one row.
    text = re.sub(r">\s+<", "><", text)
    text = re.sub(r"(?is)<script.*?</script>", "", text)
    text = re.sub(r"(?is)<style.*?</style>", "", text)
    text = re.sub(r"(?i)</(tr|p|div|li|h[1-6])\s*>", "\n", text)
    text = re.sub(r"(?i)<(tr|p|div|li|h[1-6])[^>]*>", "\n", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(td|th)\s*>", " | ", text)
    text = re.sub(r"(?i)<(td|th)[^>]*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)

    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s*\|\s*", " | ", line).strip(" |").strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _clean_raw_text(raw_text: str) -> str:
    """Single choke point both ingestion paths (file upload and pasted
    text) run through before structuring/chunking. Some source documents
    (e.g. a menu exported straight from a website) are literally HTML —
    without this, that markup was showing up verbatim in structuring
    output and, worse, in the Q&A vector store, so answers could echo raw
    <tr>/<td> tags back to a customer."""
    if raw_text and _looks_like_html(raw_text):
        print("[ingest] Extracted text looks like raw HTML markup — converting to plain text before structuring/embedding.")
        return _html_to_plain_text(raw_text)
    return raw_text



def _business_processed_dir(business_id: str) -> str:
    path = os.path.join(config.PROCESSED_DIR, business_id)
    os.makedirs(path, exist_ok=True)
    return path


def _record_path(business_id: str, ingestion_id: str) -> str:
    return os.path.join(_business_processed_dir(business_id), f"{ingestion_id}.json")


def ingest_file(business_id: str, file_path: str, progress_cb=None) -> dict:
    """Runs extraction + structuring + Q&A-chunk embedding, and writes a
    'pending_review' record. Returns that record (including an
    `ingestion_id` you'll need to call publish_review / reject_review).

    `progress_cb`, if given, is forwarded to app/extraction.py and called as
    progress_cb(current_page, total_pages) while a multi-page scanned PDF is
    being read page-by-page — useful for showing live progress in a UI
    instead of one long silent wait."""
    business_raw_dir = os.path.join(config.RAW_DIR, business_id)
    os.makedirs(business_raw_dir, exist_ok=True)
    source_basename = os.path.basename(file_path)
    dest_path = os.path.join(business_raw_dir, source_basename)
    if os.path.abspath(file_path) != os.path.abspath(dest_path):
        shutil.copy(file_path, dest_path)

    print(f"[ingest] Extracting text from '{source_basename}' ...")
    raw_text = extraction.extract_text(dest_path, progress_cb=progress_cb)
    print(f"[ingest] Extracted {len(raw_text)} characters.")

    return _structure_and_store(business_id, raw_text, source_basename)


def ingest_text(business_id: str, text: str, source_name: str = "pasted_text") -> dict:
    """Same pipeline as ingest_file, but for raw text typed/pasted directly
    (no file involved) — e.g. a business owner pasting their menu straight
    into a text box instead of uploading a document. The text is still
    saved to disk under data/raw/<business_id>/ so there's a durable record
    of what was ingested, exactly like a file upload."""
    business_raw_dir = os.path.join(config.RAW_DIR, business_id)
    os.makedirs(business_raw_dir, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in source_name) or "pasted_text"
    if not safe_name.endswith(".txt"):
        safe_name += ".txt"
    # Avoid clobbering a previous paste with the same name.
    source_basename = f"{os.path.splitext(safe_name)[0]}__{int(time.time())}.txt"
    dest_path = os.path.join(business_raw_dir, source_basename)
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"[ingest] Using {len(text)} characters of pasted text as '{source_basename}' ...")
    return _structure_and_store(business_id, text, source_basename)


def _structure_and_store(business_id: str, raw_text: str, source_basename: str) -> dict:
    """Shared tail end of the ingestion pipeline for both file- and
    text-based ingestion: structure -> chunk + embed (if published) -> write review record."""
    raw_text = _clean_raw_text(raw_text)

    print("[ingest] Structuring extracted text (persona / policies / details / menu) ...")
    structured = structuring.structure_business_data(raw_text)
    structured["_raw_text"] = raw_text

    # A structuring call can "succeed" (no exception) while still producing
    # nothing usable — the model returned non-JSON text, or there was no
    # extracted text to structure in the first place (e.g. the vision model
    # silently returned an empty/near-empty transcription). Auto-publishing
    # in that case is how a completely blank profile used to sail straight
    # to "published" with no chance for a human to notice or fix it — so
    # treat either condition as requiring review, exactly like a found menu
    # does, instead of only checking for menu items.
    structuring_failed = "_structuring_failed_raw_response" in structured
    extraction_was_empty = len(raw_text.strip()) < 20
    needs_review = bool(structured.get("menu")) or structuring_failed or extraction_was_empty

    ingestion_id = f"{os.path.splitext(source_basename)[0]}__{int(time.time())}"
    status = "pending_review" if needs_review else "published"

    if status == "published":
        print("[ingest] Chunking + embedding raw text for general Q&A retrieval (published immediately) ...")
        chunks = chunking.chunk_text(raw_text)
        vectorstore.add_chunks(business_id, chunks, source_file=source_basename)
        print(f"[ingest] Embedded {len(chunks)} text chunks for '{business_id}'.")

    record = {
        "ingestion_id": ingestion_id,
        "source_file": source_basename,
        "status": status,
        "ingested_at": time.time(),
        "data": structured,
    }
    if structuring_failed:
        record["warning"] = (
            "The structuring model's response could not be parsed as JSON — the extracted "
            "profile below is empty/incomplete. See data['_structuring_failed_raw_response'] "
            "for the model's raw output, or check the terminal log."
        )
    elif extraction_was_empty:
        record["warning"] = (
            "Almost no text was extracted from this document (raw text was "
            f"{len(raw_text.strip())} character(s)). This usually means the vision/OCR model "
            "call failed, returned 'NONE', or the file itself is unreadable — check the "
            "terminal log for the exact error, and try re-uploading."
        )
    # If there's no menu content at all AND nothing went wrong (e.g. a pure
    # "about us"/persona file), there's nothing price-sensitive to review —
    # publish it immediately so persona/details/policies are usable right
    # away without busywork.
    if record["status"] == "published":
        record["published_at"] = record["ingested_at"]

    with open(_record_path(business_id, ingestion_id), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Saved as ingestion '{ingestion_id}' with status '{record['status']}'.")

    if record["status"] == "pending_review":
        print(f"[ingest] Menu items extracted but NOT yet live — call publish_review("
              f"'{business_id}', '{ingestion_id}') after review to make them orderable.")
    else:
        print("[ingest] No menu content found — nothing pending review.")

    return record


def list_pending_reviews(business_id: str) -> List[dict]:
    """List ingestions awaiting human review for a business, oldest first."""
    directory = _business_processed_dir(business_id)
    pending = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
            record = json.load(f)
        if record.get("status") == "pending_review":
            pending.append(record)
    pending.sort(key=lambda r: r.get("ingested_at", 0))
    return pending


def get_review(business_id: str, ingestion_id: str) -> Optional[dict]:
    path = _record_path(business_id, ingestion_id)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compile_profile_to_text(data: dict) -> str:
    lines = []
    if data.get("business_name"):
        lines.append(f"Business Name: {data['business_name']}")
    
    persona = data.get("persona") or {}
    persona_lines = []
    for k, v in persona.items():
        if v:
            key_name = k.replace("_", " ").title()
            persona_lines.append(f"- {key_name}: {v}")
    if persona_lines:
        lines.append("Business Persona / Tone:\n" + "\n".join(persona_lines))
        
    details = data.get("details") or {}
    details_lines = []
    for k, v in details.items():
        if v:
            key_name = k.replace("_", " ").title()
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            details_lines.append(f"- {key_name}: {v}")
    if details_lines:
        lines.append("Business Details:\n" + "\n".join(details_lines))
        
    policies = data.get("policies") or {}
    policies_lines = []
    for k, v in policies.items():
        if v:
            key_name = k.replace("_", " ").title()
            policies_lines.append(f"- {key_name}: {v}")
    if policies_lines:
        lines.append("Business Policies:\n" + "\n".join(policies_lines))
        
    if data.get("system_prompt"):
        lines.append(f"System Prompt / Behaviour Guidelines:\n{data['system_prompt']}")

    return "\n\n".join(lines)


def publish_review(business_id: str, ingestion_id: str, corrected_data: Optional[dict] = None) -> dict:
    """
    Approve an ingestion and make its menu items live for order-taking.

    `corrected_data`, if given, should be the full structured `data` object
    (same shape as `record["data"]`) with any human corrections applied —
    e.g. a fixed price, a removed hallucinated item, a renamed category. If
    omitted, the originally-extracted data is published as-is.

    Safe to call again after a re-review: previously-published menu
    embeddings for this source file are cleared first, so corrections never
    leave stale duplicate entries behind.
    """
    record = get_review(business_id, ingestion_id)
    if record is None:
        raise ValueError(f"No ingestion '{ingestion_id}' found for business '{business_id}'.")

    if corrected_data is not None:
        record["data"] = corrected_data

    menu_items = record["data"].get("menu") or []
    # Vector store insertion is handled by the main application's sync endpoint after saving to PostgreSQL.

    # Remove previous chunks for this source file to prevent duplicate entries
    vectorstore.delete_chunks_for_source(business_id, record["source_file"])

    # Chunk and embed the corrected structured details + raw text
    corrected_text = compile_profile_to_text(record["data"])
    raw_text = record["data"].get("_raw_text") or ""
    
    combined_text = corrected_text
    if raw_text:
        combined_text += "\n\n=== RAW EXTRACTED DOCUMENT CONTENT ===\n\n" + raw_text

    chunks = chunking.chunk_text(combined_text)
    vectorstore.add_chunks(business_id, chunks, source_file=record["source_file"])
    print(f"[ingest] Embedded {len(chunks)} general text chunks for '{business_id}' at publish time.")

    record["status"] = "published"
    record["published_at"] = time.time()
    with open(_record_path(business_id, ingestion_id), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"[ingest] Published ingestion '{ingestion_id}' — {len(menu_items)} menu item(s) now live.")
    return record


def reject_review(business_id: str, ingestion_id: str, reason: str = "") -> dict:
    """Mark an ingestion as rejected. Its menu items are never embedded and
    it's excluded from the merged business profile. The record is kept
    (not deleted) so there's an audit trail of what was extracted and why
    it was turned down."""
    record = get_review(business_id, ingestion_id)
    if record is None:
        raise ValueError(f"No ingestion '{ingestion_id}' found for business '{business_id}'.")
    record["status"] = "rejected"
    record["rejected_reason"] = reason
    with open(_record_path(business_id, ingestion_id), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def get_merged_business_profile(business_id: str) -> dict:
    """
    Merge every PUBLISHED ingestion for a business into one profile.
    Pending-review and rejected ingestions are excluded — an unreviewed
    price should never surface to the voice agent or a caller.

    Menu items are deduplicated by (category, item name), case-insensitive —
    if the same item was extracted from two different files (e.g. a menu
    PDF and a later corrected photo), the one from the most recently
    published ingestion wins rather than both appearing.

    persona/details/policies fields get overwritten by whichever published
    ingestion most recently had a non-empty value for that field.
    """
    directory = _business_processed_dir(business_id)
    if not os.path.isdir(directory):
        return {}

    merged = structuring._empty_structure()
    all_areas: List[str] = []
    records = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
            record = json.load(f)
        if record.get("status") == "published":
            records.append(record)

    # Oldest first, so the most-recently-published ingestion wins on conflicts.
    records.sort(key=lambda r: r.get("published_at", r.get("ingested_at", 0)))

    menu_by_key = {}
    for record in records:
        data = record.get("data") or {}
        if data.get("business_name"):
            merged["business_name"] = data["business_name"]
        if data.get("system_prompt"):
            merged["system_prompt"] = data["system_prompt"]
        for k, v in (data.get("persona") or {}).items():
            if v:
                merged["persona"][k] = v
        for k, v in (data.get("details") or {}).items():
            if k == "delivery_areas":
                # list field: accumulate across published files instead of
                # the most-recent file's list silently replacing earlier ones.
                all_areas.extend(v or [])
                continue
            if v:
                merged["details"][k] = v
        for k, v in (data.get("policies") or {}).items():
            if v:
                merged["policies"][k] = v
        for item in data.get("menu") or []:
            key = ((item.get("category") or "").strip().lower(), (item.get("item") or "").strip().lower())
            if key == ("", ""):
                continue
            menu_by_key[key] = item  # later (more recent) publish overwrites earlier

    merged["menu"] = list(menu_by_key.values())

    # dedupe delivery_areas while preserving order
    seen = set()
    areas = []
    for area in all_areas:
        if area not in seen:
            seen.add(area)
            areas.append(area)
    merged["details"]["delivery_areas"] = areas

    return merged


def delete_business(business_id: str) -> None:
    """Fully remove a business: raw files, processed/review JSON, and both
    vector-store collections. Irreversible — used to let an owner start
    over or to remove a wrongly-onboarded business."""
    for base_dir in (config.RAW_DIR, config.PROCESSED_DIR):
        path = os.path.join(base_dir, business_id)
        if os.path.isdir(path):
            shutil.rmtree(path)
    vectorstore.delete_business_data(business_id)
    print(f"[ingest] Deleted all data for business '{business_id}'.")


def update_business_profile(business_id: str, profile_patch: dict) -> dict:
    """
    Update profile fields (persona / details / policies / business_name) across
    all published records for a business, then re-embed the corrected text so
    the RAG vector store reflects the latest admin-approved values.

    `profile_patch` should be a partial or full profile dict, e.g.:
        {
            "business_name": "My Restaurant",
            "persona": {"agent_name": "Zara", "tone": "friendly"},
            "details": {"address": "123 Main St", "phone": "0300-1234567"},
            "policies": {"upsell_strategy": "Offer drinks with every order"}
        }

    Only non-empty / non-None values in the patch are applied.
    Returns the resulting merged profile.
    """
    directory = _business_processed_dir(business_id)
    records = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            rec = json.load(f)
        if rec.get("status") == "published":
            records.append((fpath, rec))

    if not records:
        # No published record yet — create a synthetic one to store the patch
        ingestion_id = f"manual_profile__{int(time.time())}"
        synthetic_data: dict = structuring._empty_structure()

        def _deep_merge(base: dict, patch: dict) -> None:
            for k, v in patch.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _deep_merge(base[k], v)
                elif v not in (None, "", [], {}):
                    base[k] = v

        _deep_merge(synthetic_data, profile_patch)
        record = {
            "ingestion_id": ingestion_id,
            "source_file": "manual_profile",
            "status": "published",
            "ingested_at": time.time(),
            "published_at": time.time(),
            "data": synthetic_data,
        }
        fpath = _record_path(business_id, ingestion_id)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        records = [(fpath, record)]

    def _deep_merge(base: dict, patch: dict) -> None:
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                _deep_merge(base[k], v)
            elif v not in (None, "", [], {}):
                base[k] = v

    for fpath, record in records:
        data = record.get("data") or {}
        _deep_merge(data, profile_patch)
        record["data"] = data
        record["published_at"] = time.time()
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

        # Re-embed: clear old chunks then write updated compiled text + raw
        source_file = record.get("source_file", "manual_profile")
        vectorstore.delete_chunks_for_source(business_id, source_file)
        corrected_text = compile_profile_to_text(record["data"])
        raw_text = record["data"].get("_raw_text") or ""
        combined_text = corrected_text
        if raw_text:
            combined_text += "\n\n=== RAW EXTRACTED DOCUMENT CONTENT ===\n\n" + raw_text
        chunks = chunking.chunk_text(combined_text)
        vectorstore.add_chunks(business_id, chunks, source_file=source_file)
        print(f"[ingest] Re-embedded {len(chunks)} chunks for '{business_id}' after profile update.")

    return get_merged_business_profile(business_id)
