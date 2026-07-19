"""
DeepgramLiveSTT — async streaming speech-to-text over Deepgram's live
WebSocket API.

Twilio's mu-law 8 kHz frames are forwarded as-is (encoding=mulaw), so there
is no decode/resample step between the phone call and the recognizer.

Emits STTEvent objects onto an asyncio.Queue consumed by the CallSession:

    speech_started   caller began talking (VAD) — barge-in signal
    interim          partial transcript (also a barge-in signal, safer)
    final            finalized transcript segment; speech_final=True means
                     the endpointer saw enough silence to close the turn
    utterance_end    word-gap fallback turn closure (noisy line endings)
    error / closed   stream lifecycle

Turn-taking contract: the session accumulates `final` segments and fires
the user turn on the first of (speech_final=True, utterance_end).
"""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from dataclasses import dataclass

import websockets

log = logging.getLogger("voice.stt")

DEEPGRAM_WSS = "wss://api.deepgram.com/v1/listen"

# Outbound audio buffer: ~5 s of 20 ms Twilio frames. If Deepgram's socket
# stalls longer than that, old frames are dropped to stay realtime.
_SEND_QUEUE_FRAMES = 256


@dataclass
class STTEvent:
    kind: str            # speech_started | interim | final | utterance_end | error | closed
    text: str = ""
    speech_final: bool = False
    confidence: float = 0.0


class DeepgramLiveSTT:
    def __init__(self, config, events: asyncio.Queue):
        self.config = config
        self.events = events
        self._ws = None
        self._tasks: list[asyncio.Task] = []
        self._closed = False
        self._send_q: asyncio.Queue = asyncio.Queue(maxsize=_SEND_QUEUE_FRAMES)
        self._dropped_frames = 0

    def _build_url(self) -> str:
        params = {
            "model": self.config.stt_model,
            "language": self.config.stt_language,
            "encoding": "mulaw",
            "sample_rate": "8000",
            "channels": "1",
            "punctuate": "true",
            "smart_format": "true",
            "interim_results": "true",
            "endpointing": str(self.config.endpointing_ms),
            "utterance_end_ms": str(max(self.config.utterance_end_ms, 1000)),
            "vad_events": "true",
        }
        query = urllib.parse.urlencode(params)
        # Keyterm prompting boosts recognition of menu items / proper nouns.
        for term in filter(None, (t.strip() for t in self.config.stt_keywords.split(","))):
            query += "&keyterm=" + urllib.parse.quote(term)
        return f"{DEEPGRAM_WSS}?{query}"

    async def start(self):
        headers = {"Authorization": f"Token {self.config.deepgram_api_key}"}
        url = self._build_url()
        try:
            # websockets<14 uses extra_headers, >=14 additional_headers
            try:
                self._ws = await websockets.connect(url, extra_headers=headers)
            except TypeError:
                self._ws = await websockets.connect(url, additional_headers=headers)
        except Exception as e:
            await self.events.put(STTEvent(kind="error", text=f"deepgram connect failed: {e}"))
            raise
        self._tasks.append(asyncio.create_task(self._receiver()))
        self._tasks.append(asyncio.create_task(self._keepalive()))
        self._tasks.append(asyncio.create_task(self._sender()))

    async def send_audio(self, mulaw_bytes: bytes):
        """
        Non-blocking enqueue. The actual socket send happens in _sender(),
        so a slow/stalled Deepgram connection can never back-pressure the
        Twilio receive loop (which also carries mark/stop events for every
        live call). On overflow the oldest frame is dropped — staying
        realtime beats perfect audio for a phone conversation.
        """
        if self._ws is None or self._closed:
            return
        try:
            self._send_q.put_nowait(mulaw_bytes)
        except asyncio.QueueFull:
            self._dropped_frames += 1
            try:
                self._send_q.get_nowait()
                self._send_q.put_nowait(mulaw_bytes)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    async def _sender(self):
        try:
            while not self._closed:
                data = await self._send_q.get()
                await self._ws.send(data)
        except asyncio.CancelledError:
            raise
        except Exception:
            if not self._closed:
                await self.events.put(STTEvent(kind="error", text="deepgram send failed"))

    async def finish(self):
        self._closed = True
        if self._dropped_frames:
            log.warning("dropped %d audio frames (slow Deepgram socket)",
                        self._dropped_frames)
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except Exception:
                pass
        for task in self._tasks:
            task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ── internals ────────────────────────────────────────────────────────
    async def _keepalive(self):
        # Twilio streams silence continuously, but during transfers or media
        # gaps Deepgram will close an idle socket after ~10 s without this.
        while not self._closed:
            await asyncio.sleep(5)
            try:
                await self._ws.send(json.dumps({"type": "KeepAlive"}))
            except Exception:
                return

    async def _receiver(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "Results":
                    alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
                    text = (alt.get("transcript") or "").strip()
                    if msg.get("is_final"):
                        if text:
                            await self.events.put(STTEvent(
                                kind="final", text=text,
                                speech_final=bool(msg.get("speech_final")),
                                confidence=float(alt.get("confidence") or 0.0),
                            ))
                        elif msg.get("speech_final"):
                            # Endpoint reached with no words in this segment —
                            # still forward so a pending turn can close.
                            await self.events.put(STTEvent(kind="final", text="",
                                                           speech_final=True))
                    elif text:
                        await self.events.put(STTEvent(kind="interim", text=text))
                elif mtype == "SpeechStarted":
                    await self.events.put(STTEvent(kind="speech_started"))
                elif mtype == "UtteranceEnd":
                    await self.events.put(STTEvent(kind="utterance_end"))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not self._closed:
                await self.events.put(STTEvent(kind="error", text=str(e)))
        finally:
            await self.events.put(STTEvent(kind="closed"))
