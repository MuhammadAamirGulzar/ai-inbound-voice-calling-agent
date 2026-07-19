"""
Twilio webhook + Media Streams endpoints.

Two call pipelines are dispatched from /twilio-media-stream:

  streaming (default when keys exist)  voice/pipeline.py — cloud streaming
      Deepgram live STT → Groq/OpenAI LLM (token stream) → Aura/ElevenLabs
      streaming TTS, with barge-in, warm transfers and per-turn metrics.

  legacy (VOICE_PIPELINE=legacy, or no cloud keys)  twilio_legacy.py —
      local Faster-Whisper + Ollama + Piper, thread per call. Works fully
      offline; sequential, so several seconds per turn.

Selection: VOICE_PIPELINE env = auto (default) | streaming | legacy.
"""

import asyncio
import base64
import json
import os
import time
import urllib.parse
from typing import Optional

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect

from sql.database import SessionLocal
from sql import models
from voice.config import VoiceConfig
from voice.pipeline import CallSession, TwilioTransport

router = APIRouter()

DEFAULT_SYSTEM_PROMPT = (
    "You are the friendly AI voice agent taking phone orders for a "
    "restaurant. Be extremely brief — under 30 words per reply. Collect "
    "the caller's delivery address before concluding an order. Use the "
    "transfer_call tool if the caller asks for a human or has a complaint; "
    "use end_call once the conversation has clearly finished."
)


def resolve_restaurants_for_call(to_number: str, db) -> list:
    from sql.models import Restaurant
    if not to_number:
        return []
    return db.query(Restaurant).filter(Restaurant.order_phone_number == to_number).all()


# ─────────────────────────────────────────────────────────────────────────
# Incoming-call webhook → TwiML that opens the media stream
# ─────────────────────────────────────────────────────────────────────────
@router.post("/voice")
async def voice(request: Request):
    """
    Twilio Webhook for incoming calls.
    Returns TwiML that connects the call to a WebSocket Media Stream.
    """
    form_data = await request.form()
    to_number = form_data.get("To", "")
    caller_number = form_data.get("From", "")

    db = SessionLocal()
    restaurants = []
    try:
        restaurants = resolve_restaurants_for_call(to_number, db)
        if not restaurants:
            print(f"WARNING: resolve_restaurants_for_call('{to_number}') returned zero matches in voice webhook.")
    except Exception as e:
        print(f"Error querying restaurants in voice webhook: {e}")
    finally:
        db.close()

    response = VoiceResponse()

    # Check if ALL matching restaurants are suspended. If so, reject the call.
    if restaurants and all(r.is_suspended for r in restaurants):
        print("Call rejected: All matching restaurants are suspended.")
        response.say("We are sorry, this service is temporarily unavailable. Goodbye.")
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Check if the AI voice agent is manually paused
    if restaurants and restaurants[0].agent_configuration and not restaurants[0].agent_configuration.is_active:
        print("Agent is paused. Hanging up call.")
        response.say("Thank you for calling. The restaurant's AI assistant is currently offline. Goodbye.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")

    connect = Connect()

    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https" else "ws"

    caller_number_encoded = urllib.parse.quote(caller_number or "")
    to_number_encoded = urllib.parse.quote(to_number or "")
    ws_url = f"{scheme}://{host}/twilio-media-stream?caller_number={caller_number_encoded}&to_number={to_number_encoded}"

    print(f"Connecting Twilio call to WebSocket URL: {ws_url}")

    connect.stream(url=ws_url)
    response.append(connect)

    return HTMLResponse(content=str(response), media_type="application/xml")


# ─────────────────────────────────────────────────────────────────────────
# Warm-transfer whisper: played to the STAFF member before bridging,
# so the human hears the AI's handoff summary while the caller waits.
# ─────────────────────────────────────────────────────────────────────────
@router.api_route("/twilio/transfer-whisper", methods=["GET", "POST"])
async def transfer_whisper(request: Request):
    summary = request.query_params.get("summary", "")[:400]
    response = VoiceResponse()
    response.say("Incoming call transferred from the A I assistant.")
    if summary:
        response.say(summary)
    response.say("Connecting you now.")
    return HTMLResponse(content=str(response), media_type="application/xml")


# ─────────────────────────────────────────────────────────────────────────
# Media stream endpoint — dispatches to streaming or legacy pipeline
# ─────────────────────────────────────────────────────────────────────────
@router.websocket("/twilio-media-stream")
async def twilio_media_stream(
    websocket: WebSocket,
    caller_number: Optional[str] = None,
    to_number: Optional[str] = None,
):
    await websocket.accept()
    print(f"Twilio Media Stream WebSocket Connected (Caller: {caller_number}, To: {to_number}).")

    config = VoiceConfig.from_env()
    mode = os.getenv("VOICE_PIPELINE", "auto").strip().lower()
    use_streaming = (mode == "streaming") or (mode == "auto" and config.streaming_ready)

    if use_streaming:
        await run_streaming_call(websocket, config, caller_number or "", to_number or "")
    else:
        if mode == "auto":
            print("[voice] cloud keys missing (DEEPGRAM_API_KEY / GROQ_API_KEY) — "
                  "falling back to legacy local pipeline.")
        from twilio_legacy import run_legacy_call
        await run_legacy_call(websocket, caller_number, to_number,
                              resolve_restaurants_for_call)


# ─────────────────────────────────────────────────────────────────────────
# Streaming pipeline plumbing
# ─────────────────────────────────────────────────────────────────────────
def _load_agent_settings(to_number: str) -> dict:
    """Per-tenant prompt/voice settings (runs in a worker thread)."""
    db = SessionLocal()
    try:
        restaurant = db.query(models.Restaurant).filter(
            models.Restaurant.order_phone_number == to_number).first()
        if not restaurant:
            return {}
        agent_config = restaurant.agent_configuration
        return {
            "restaurant_id": restaurant.id,
            "restaurant_name": restaurant.name,
            "system_prompt": (agent_config.system_prompt
                              if agent_config else DEFAULT_SYSTEM_PROMPT),
            "is_active": agent_config.is_active if agent_config else True,
            "voice_settings": (getattr(agent_config, "voice_settings", None)
                               if agent_config else None) or {},
        }
    finally:
        db.close()


async def _fetch_greeting(config: VoiceConfig, restaurant_name: str,
                          business_id: str) -> str:
    """Greeting from the RAG persona profile, with a sane default."""
    default = (f"Hello! Thank you for calling {restaurant_name}. "
               f"What would you like to order today?"
               if restaurant_name else "Hello! How can I help you today?")
    if not business_id:
        return default
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(
                f"{config.rag_base_url}/businesses/{business_id}/profile")
        if r.status_code == 200:
            persona = (r.json().get("persona") or {})
            if persona.get("greeting_script"):
                return persona["greeting_script"]
    except Exception:
        pass
    return default


def _create_call_logs(stream_sid: str, to_number: str, caller_number: str):
    db = SessionLocal()
    try:
        restaurants = resolve_restaurants_for_call(to_number, db)
        for r in restaurants:
            db.add(models.ChatHistory(
                session_id=stream_sid,
                restaurant_id=r.id,
                caller_number=caller_number,
                chat_data=[],
                response_time=0.0,
                status="in_progress",
                duration_seconds=None,
                recording_url=None,
            ))
        db.commit()
    except Exception as e:
        print(f"Error creating call_logs rows on start: {e}")
        db.rollback()
    finally:
        db.close()


def _finalize_call(stream_sid: str, to_number: str, duration_seconds: int,
                   messages: list, metrics_dict: dict, call_failed: bool):
    db = SessionLocal()
    try:
        restaurants = resolve_restaurants_for_call(to_number, db)
        duration_minutes = duration_seconds / 60.0
        for r in restaurants:
            r.used_minutes += duration_minutes
            if r.used_minutes >= r.assigned_minutes:
                r.is_suspended = True
                print(f"Restaurant '{r.name}' exceeded its quota — auto-suspending.")

        user_spoke = any(m.get("role") == "user" for m in messages)
        final_status = "failed" if call_failed else (
            "completed" if user_spoke else "missed")

        avg_response = None
        turns = (metrics_dict.get("summary") or {})
        if turns.get("response_ms_p50") is not None:
            avg_response = turns["response_ms_p50"] / 1000.0

        logs = db.query(models.ChatHistory).filter(
            models.ChatHistory.session_id == stream_sid).all()
        for log in logs:
            log.chat_data = [m for m in messages if m.get("role") != "system"]
            log.duration_seconds = duration_seconds
            log.status = final_status
            if avg_response is not None:
                log.response_time = avg_response
            if hasattr(log, "metrics"):
                log.metrics = metrics_dict
            if final_status == "completed":
                from utils.order_extractor import extract_order_from_transcript
                try:
                    extract_order_from_transcript(log, db)
                except Exception as parse_err:
                    print(f"Error auto-extracting order from call transcript: {parse_err}")
        db.commit()
    except Exception as e:
        print(f"Error updating call logs/minutes: {e}")
        db.rollback()
    finally:
        db.close()


async def run_streaming_call(websocket: WebSocket, config: VoiceConfig,
                             caller_number: str, to_number: str):
    """Drive one call on the async streaming engine."""
    from websocket_registry import active_connections

    stream_sid = None
    call_sid = ""
    session: Optional[CallSession] = None
    session_task = None
    call_start = None

    async def handle_start(message: dict):
        nonlocal stream_sid, call_sid, session, session_task, call_start
        start = message.get("start", {})
        stream_sid = start.get("streamSid") or message.get("streamSid")
        call_sid = start.get("callSid", "")
        call_start = time.time()
        active_connections[stream_sid] = websocket
        print(f"[voice] stream started sid={stream_sid} call={call_sid}")

        settings = await asyncio.to_thread(_load_agent_settings, to_number)
        config.system_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        config.apply_overrides(settings.get("voice_settings"))
        if settings.get("restaurant_id") and not config.rag_business_id:
            config.rag_business_id = str(settings["restaurant_id"])
        config.greeting = await _fetch_greeting(
            config, settings.get("restaurant_name", ""), config.rag_business_id)

        await asyncio.to_thread(_create_call_logs, stream_sid, to_number,
                                caller_number)

        transport = TwilioTransport(websocket, stream_sid)
        session = CallSession(config, transport, call_sid=call_sid,
                              caller_number=caller_number)
        session_task = asyncio.create_task(session.run())

    receive_task = asyncio.create_task(websocket.receive_text())
    ended_task = None
    call_failed = False
    try:
        while True:
            wait_for = {receive_task}
            if session is not None and ended_task is None:
                ended_task = asyncio.create_task(session.ended.wait())
            if ended_task is not None:
                wait_for.add(ended_task)
            done, _ = await asyncio.wait(wait_for,
                                         return_when=asyncio.FIRST_COMPLETED)

            if ended_task is not None and ended_task in done:
                # Session ended on its own (agent hangup / watchdog).
                break

            data = receive_task.result()  # raises on disconnect
            message = json.loads(data)
            event = message.get("event")
            if event == "media":
                if session is not None:
                    await session.feed_audio(
                        base64.b64decode(message["media"]["payload"]))
            elif event == "mark":
                if session is not None:
                    session.on_mark(message.get("mark", {}).get("name", ""))
            elif event == "start":
                await handle_start(message)
            elif event == "connected":
                pass
            elif event == "stop":
                print("[voice] stop event received")
                break
            receive_task = asyncio.create_task(websocket.receive_text())
    except WebSocketDisconnect:
        print("[voice] media stream disconnected")
    except Exception as e:
        call_failed = True
        print(f"[voice] media stream error: {e}")
    finally:
        receive_task.cancel()
        if ended_task is not None:
            ended_task.cancel()
        if session is not None:
            await session.shutdown()
        if session_task is not None:
            session_task.cancel()
        if stream_sid:
            active_connections.pop(stream_sid, None)

        if session is not None and call_start is not None:
            duration = int(time.time() - call_start)
            metrics_dict = session.metrics.to_dict()
            print(session.metrics.log_line())
            await asyncio.to_thread(
                _finalize_call, stream_sid, to_number, duration,
                session.messages, metrics_dict, call_failed)

        try:
            await websocket.close()
        except Exception:
            pass
