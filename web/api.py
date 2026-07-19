"""JSON APIs: RAG reverse proxy and call-log access."""

import re

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi import Response as FastAPIResponse
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from sql import crud, models
from services.menu_sync import (RAG_UPSTREAM, save_published_menu_to_postgres,
                                sync_postgres_menu_to_chroma)
from web.core import get_db

router = APIRouter()

@router.api_route("/rag/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def rag_proxy(path: str, request: Request, db: Session = Depends(get_db)):
    url = f"{RAG_UPSTREAM}/{path}"
    # No timeout limit — image ingestion through the ngrok tunnel (vision model
    # + structuring) can take several minutes on slow connections.
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            body = await request.body()
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
            resp = await client.request(
                method=request.method,
                url=url,
                params=dict(request.query_params),
                headers=headers,
                content=body,
            )
        
        # Intercept post-publish response to save to Postgres & sync to Chroma
        if resp.status_code == 200 and request.method == "POST" and "reviews" in path and path.endswith("/publish"):
            try:
                publish_match = re.match(r"businesses/([^/]+)/reviews/([^/]+)/publish", path)
                if publish_match:
                    restaurant_id = int(publish_match.group(1))
                    published_data = resp.json()
                    menu_items = published_data.get("data", {}).get("menu") or []
                    save_published_menu_to_postgres(db, restaurant_id, menu_items)
                    await sync_postgres_menu_to_chroma(db, restaurant_id)
            except Exception as e:
                print(f"Error handling post-publish sync to Postgres/Chroma: {e}")

        return FastAPIResponse(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
            media_type=resp.headers.get("content-type"),
        )
    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={"detail": "Knowledge RAG service is not available. It may still be starting up — please wait a moment and try again."},
        )
    except httpx.ReadTimeout:
        return JSONResponse(
            status_code=504,
            content={"detail": "The RAG service took too long to respond. The ingestion may still be processing in the background — check Pending Reviews in a moment."},
        )


@router.get("/api/call-logs")
async def api_call_logs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    """Return recent Twilio call logs for the v2 call-history page."""
    logs = crud.get_chat_histories(db, skip=skip, limit=limit)
    return [
        {
            "id": log.id,
            "session_id": log.session_id,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "response_time": log.response_time,
            "message_count": len(log.chat_data) if log.chat_data else 0,
        }
        for log in logs
    ]


@router.get("/api/call-logs/{log_id}")
async def api_call_log_detail(log_id: int, db: Session = Depends(get_db)):
    """Return full chat transcript for a single call log."""
    log = crud.get_chat_history(db, log_id)
    if not log:
        return JSONResponse(status_code=404, content={"error": "Call log not found."})
    return {
        "id": log.id,
        "session_id": log.session_id,
        "timestamp": log.timestamp.isoformat() if log.timestamp else None,
        "response_time": log.response_time,
        "chat_data": log.chat_data,
    }


