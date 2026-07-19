"""
FastAPI app — the integration point for the voice call agent. Wraps the
same ingest -> review -> publish -> query/menu-search pipeline used by the
CLI and the Streamlit console (see app/ingest.py, app/rag.py,
app/vectorstore.py) behind an HTTP API.

Run with:
    uvicorn main:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger docs.
"""
import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import errors, ingest, rag, vectorstore

app = FastAPI(title="Business Knowledge RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Without this, FastAPI's default behavior on an unhandled exception
    is a bare 500 with no detail — unhelpful for a client integrating
    against this API (e.g. the voice agent) trying to figure out what went
    wrong. This logs the full traceback server-side and returns a always-
    non-empty, clear message in the response instead."""
    errors.log_exception(f"{request.method} {request.url.path}", exc)
    return JSONResponse(status_code=500, content={"detail": errors.describe_exception(exc)})


class TextIngestRequest(BaseModel):
    business_id: str
    text: str
    source_name: Optional[str] = "pasted_text"


class QueryRequest(BaseModel):
    business_id: str
    question: str


class MenuSearchRequest(BaseModel):
    business_id: str
    phrase: str
    top_k: int = 3


class PublishRequest(BaseModel):
    corrected_data: Optional[dict] = None


class RejectRequest(BaseModel):
    reason: str = ""


@app.post("/ingest")
async def ingest_endpoint(business_id: str = Form(...), file: UploadFile = File(...)):
    """Upload a .txt, .pdf, .png, or .jpg file for a business — runs
    extraction + structuring + Q&A-chunk embedding, and returns a
    pending-review record. Menu items are NOT live yet — see /review/*.

    For raw pasted/typed text instead of a file, use POST /ingest-text."""
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        record = ingest.ingest_file(business_id, tmp_path)
    finally:
        os.remove(tmp_path)
    return record


@app.post("/ingest-text")
async def ingest_text_endpoint(req: TextIngestRequest):
    """Ingest raw text pasted/typed directly — no file involved. Same
    pipeline as /ingest (extraction is skipped since the text is already
    plain text), runs structuring + Q&A-chunk embedding, and returns a
    pending-review record exactly like a file upload would."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    return ingest.ingest_text(req.business_id, req.text, source_name=req.source_name or "pasted_text")


@app.get("/businesses/{business_id}/reviews")
async def list_reviews_endpoint(business_id: str):
    """Ingestions awaiting human review — this is what a dashboard's
    'review extracted menu' screen should list."""
    return {"pending": ingest.list_pending_reviews(business_id)}


@app.get("/businesses/{business_id}/reviews/{ingestion_id}")
async def get_review_endpoint(business_id: str, ingestion_id: str):
    record = ingest.get_review(business_id, ingestion_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Ingestion not found")
    return record


@app.post("/businesses/{business_id}/reviews/{ingestion_id}/publish")
async def publish_review_endpoint(business_id: str, ingestion_id: str, req: PublishRequest):
    """Approve (optionally with corrections) — makes the menu items live
    for order-taking. `corrected_data` should be the full structured `data`
    object with any human edits applied; omit to publish as extracted."""
    try:
        return ingest.publish_review(business_id, ingestion_id, corrected_data=req.corrected_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/businesses/{business_id}/reviews/{ingestion_id}/reject")
async def reject_review_endpoint(business_id: str, ingestion_id: str, req: RejectRequest):
    try:
        return ingest.reject_review(business_id, ingestion_id, reason=req.reason)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


class VectorQueryRequest(BaseModel):
    business_id: str
    text: str
    top_k: int = 4


@app.post("/query")
async def query_endpoint(req: QueryRequest):
    """Ask a question about a business's ingested data — returns a grounded answer."""
    return rag.answer_question(req.business_id, req.question)


@app.post("/vector-query")
async def vector_query_endpoint(req: VectorQueryRequest):
    """Retrieve raw business knowledge text chunks for a given text query."""
    results = vectorstore.query(req.business_id, req.text, top_k=req.top_k)
    return {"results": [{"document": doc, "metadata": meta} for doc, meta in results]}


@app.post("/menu-search")
async def menu_search_endpoint(req: MenuSearchRequest):
    """
    Resolve a spoken/typed phrase to real, PUBLISHED, priced menu items —
    this is the endpoint the voice call agent calls during order-taking,
    not /query.
    """
    matches = vectorstore.query_menu_items(req.business_id, req.phrase, top_k=req.top_k)
    return {"matches": matches}


@app.get("/businesses/{business_id}/profile")
async def profile_endpoint(business_id: str):
    """Return the merged, published-only structured profile (persona,
    policies, details, menu) for a business."""
    return ingest.get_merged_business_profile(business_id)


class ProfilePatchRequest(BaseModel):
    business_name: Optional[str] = None
    system_prompt: Optional[str] = None
    persona: Optional[dict] = None
    details: Optional[dict] = None
    policies: Optional[dict] = None


@app.patch("/businesses/{business_id}/profile")
async def update_profile_endpoint(business_id: str, req: ProfilePatchRequest):
    """Update persona / details / policies / business_name for a business and
    re-embed the updated compiled text so the RAG reflects the new values."""
    patch = {}
    if req.business_name is not None:
        patch["business_name"] = req.business_name
    if req.system_prompt is not None:
        patch["system_prompt"] = req.system_prompt
    if req.persona is not None:
        patch["persona"] = req.persona
    if req.details is not None:
        patch["details"] = req.details
    if req.policies is not None:
        patch["policies"] = req.policies
    updated_profile = ingest.update_business_profile(business_id, patch)
    return {"status": "updated", "profile": updated_profile}


@app.delete("/businesses/{business_id}")
async def delete_business_endpoint(business_id: str):
    """Permanently delete all raw files, structured data, and vector-store
    entries for a business."""
    ingest.delete_business(business_id)
    return {"status": "deleted", "business_id": business_id}


class MenuSyncItem(BaseModel):
    name: str
    category: str
    description: Optional[str] = ""
    price: float
    variants: Optional[list] = []
    customizations: Optional[list] = []

class MenuSyncRequest(BaseModel):
    menu_items: list[MenuSyncItem]

@app.post("/businesses/{business_id}/menu/sync")
async def sync_menu_endpoint(business_id: str, req: MenuSyncRequest):
    """
    Directly sync menu items from PostgreSQL database into Chroma menu collection.
    Clears the entire menu collection for this business and updates it with the new items.
    """
    try:
        collection = vectorstore.reset_menu_collection(business_id)
        
        if req.menu_items:
            formatted_items = []
            for it in req.menu_items:
                formatted_items.append({
                    "item": it.name,
                    "category": it.category,
                    "description": it.description or "",
                    "price": str(it.price),
                    "variants": it.variants or [],
                    "customizations": it.customizations or [],
                })
            # Add to Chroma with source_file="postgres"
            vectorstore.add_menu_items(business_id, formatted_items, source_file="postgres")
        return {"status": "success", "count": len(req.menu_items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}

