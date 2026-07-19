"""
OrderSaathi — multi-tenant AI voice agent platform (entrypoint).

Assembles the FastAPI app from focused packages:

    voice/      streaming STT -> LLM -> TTS engine (barge-in, metrics)
    telephony/  Twilio/SIP media-stream transport + legacy pipeline
    web/        dashboard pages and JSON APIs (auth, admin, superadmin)
    services/   menu sync, RAG sidecar lifecycle, billing reminders
    sql/        models, CRUD, additive migrations

Run: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os

from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

load_dotenv()

# ── Keep model/tool caches inside the project (avoids C: drive issues) ──
project_root = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(project_root, ".cache")
os.makedirs(cache_dir, exist_ok=True)

os.environ["HF_HOME"] = os.path.join(cache_dir, "huggingface")
os.environ["HF_HUB_CACHE"] = os.environ["HF_HOME"]
os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]
os.environ["TORCH_HOME"] = os.path.join(cache_dir, "torch")

temp_dir = os.path.join(cache_dir, "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir

# ── Startup guards ──
SESSION_MIDDLEWARE_SECRET_KEY = os.getenv("SESSION_MIDDLEWARE_SECRET_KEY")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not SESSION_MIDDLEWARE_SECRET_KEY or not JWT_SECRET_KEY:
    raise RuntimeError(
        "Missing required environment variables: SESSION_MIDDLEWARE_SECRET_KEY and JWT_SECRET_KEY"
    )

# ── Database ──
from sql import models
from sql.database import engine, ensure_columns

models.Base.metadata.create_all(bind=engine)
try:
    ensure_columns()
except Exception as _mig_err:
    print(f"[db] column migration warning: {_mig_err}")

# ── App assembly ──
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SESSION_MIDDLEWARE_SECRET_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")

from telephony.twilio_routes import router as twilio_router
from web.auth import router as auth_router
from web.admin import router as admin_router
from web.superadmin import router as superadmin_router
from web.api import router as api_router

app.include_router(twilio_router)
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(superadmin_router)
app.include_router(api_router)

# Browser-mic test console needs the local ML stack (torch/whisper);
# skip it gracefully on cloud-streaming-only deployments.
try:
    from web.test_console import router as test_router
    app.include_router(test_router)
except ImportError as _test_err:
    print(f"Local test console disabled (missing local ML deps): {_test_err}")

# SIP media-stream bridge, opt-in.
if os.getenv("SIP_ENABLED", "false").lower() == "true":
    try:
        from telephony.sip_routes import router as sip_router
        app.include_router(sip_router)
        print("SIP routes registered successfully.")
    except Exception as e:
        print(f"Failed to register SIP routes: {e}")


# ── Prometheus scrape target for the voice engine ──
# Aggregates only (latency histogram, call/turn/barge-in counters).
from voice.telemetry import telemetry as _voice_telemetry


@app.get("/metrics")
async def metrics():
    return Response(content=_voice_telemetry.render_prometheus(),
                    media_type="text/plain; version=0.0.4")


# ── Lifecycle ──
from services import rag_sidecar, reminders


@app.on_event("startup")
async def _startup():
    await rag_sidecar.start(app)
    reminders.start_due_date_reminder_cron()


@app.on_event("shutdown")
async def _shutdown():
    rag_sidecar.stop(app)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="keys/key.pem",
        ssl_certfile="keys/cert.pem",
    )
