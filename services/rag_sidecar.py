"""Lifecycle of the knowledge_rag sidecar (menu/knowledge service).

Started with the main app unless RAG_AUTOSTART=false (cloud deployments
run it as its own service or not at all).
"""

import os
import subprocess

import httpx

from sql import models
from sql.database import SessionLocal
from services.menu_sync import sync_postgres_menu_to_chroma
from utils.logger import print_info, print_error


async def start(app):
    import sys
    import asyncio
    if os.getenv("RAG_AUTOSTART", "true").lower() in ("0", "false", "no"):
        print_info("RAG_AUTOSTART disabled — skipping knowledge_rag sidecar.")
        return
    proj_root = os.path.dirname(os.path.abspath(__file__))
    rag_dir = os.path.join(proj_root, "knowledge_rag")

    rag_python = os.path.join(rag_dir, "venv", "Scripts", "python.exe")
    if not os.path.exists(rag_python):
        rag_python = sys.executable

    cmd = [rag_python, "-m", "uvicorn", "main:app", "--port", "8001"]

    try:
        print_info(f"Auto-starting knowledge_rag backend on port 8001...")
        # No CREATE_NO_WINDOW — inherit parent's stdout/stderr so RAG logs
        # appear in the same terminal as the main uvicorn server.
        process = subprocess.Popen(
            cmd,
            cwd=rag_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        app.state.rag_process = process

        # Wait until the RAG server is healthy (up to 60s) before accepting traffic
        health_url = "http://127.0.0.1:8001/health"
        async with httpx.AsyncClient(timeout=3.0) as client:
            for attempt in range(20):
                await asyncio.sleep(3)
                try:
                    r = await client.get(health_url)
                    if r.status_code == 200:
                        print_info("knowledge_rag backend is ready on port 8001.")
                        # Sync existing postgres menus to Chroma
                        try:
                            db_session = SessionLocal()
                            restaurants = db_session.query(models.Restaurant).all()
                            for rest in restaurants:
                                await sync_postgres_menu_to_chroma(db_session, rest.id)
                            db_session.close()
                        except Exception as sync_err:
                            print_error(f"Failed to run initial startup menu sync: {sync_err}")
                        break
                except Exception:
                    pass
            else:
                print_error("knowledge_rag backend did not become ready within 60s.")
    except Exception as e:
        print_error(f"Failed to auto-start knowledge_rag backend: {e}")



def stop(app):
    process = getattr(app.state, "rag_process", None)
    if process:
        print_info("Stopping knowledge_rag backend...")
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
