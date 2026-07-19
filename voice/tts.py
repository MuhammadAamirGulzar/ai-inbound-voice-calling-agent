"""
Streaming TTS providers that emit telephony-native audio.

Both providers are asked for mu-law 8 kHz output, so synthesized bytes go
straight into Twilio media frames — no resampling, no pydub/librosa on the
hot path. Chunks are yielded as they arrive from the HTTP stream, which is
what makes sentence-level pipelining effective: the first ~100 ms of the
first sentence is already on the wire while the LLM is still generating
the rest of the reply.

Cancellation contract: the pipeline cancels the task consuming these
generators on barge-in; httpx closes the connection when the generator is
garbage-collected/aclosed, aborting synthesis server-side.
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx
import websockets

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
DEEPGRAM_SPEAK_WSS = "wss://api.deepgram.com/v1/speak"
ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


class DeepgramAuraWS:
    """
    Deepgram Aura over the TTS WebSocket — the actually-streaming
    interface. The REST /v1/speak endpoint synthesizes the full clip
    before sending a single byte (measured: TTFB == total, scaling with
    text length), so for live calls we hold one speak socket per call
    and stream each sentence through it:

        {"type": "Speak", "text": ...} then {"type": "Flush"}
        <- binary audio frames as they are synthesized
        <- {"type": "Flushed"} marks end of the utterance

    Barge-in: the pipeline cancels the consuming task; the next
    synthesize() sends {"type": "Clear"} first, dropping any audio the
    socket still had buffered for the cancelled sentence.
    """

    provider_name = "deepgram-aura-ws"

    def __init__(self, config):
        self.config = config
        self._ws = None
        self._lock = asyncio.Lock()   # sentences are strictly serialized
        self._need_clear = False

    def _url(self) -> str:
        return (f"{DEEPGRAM_SPEAK_WSS}?model={self.config.tts_voice}"
                f"&encoding=mulaw&sample_rate=8000")

    async def _connect(self):
        api_key = self.config.deepgram_tts_api_key or self.config.deepgram_api_key
        headers = {"Authorization": f"Token {api_key}"}
        try:
            self._ws = await websockets.connect(self._url(), extra_headers=headers)
        except TypeError:
            self._ws = await websockets.connect(self._url(), additional_headers=headers)

    async def prewarm(self):
        """Open the speak socket ahead of the first sentence (call start),
        overlapping this cold start with the STT connect."""
        async with self._lock:
            if self._ws is None:
                try:
                    await self._connect()
                except Exception:
                    pass  # synthesize() will retry and surface the error

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        async with self._lock:
            try:
                if self._ws is None:
                    await self._connect()
                if self._need_clear:
                    await self._ws.send(json.dumps({"type": "Clear"}))
                    self._need_clear = False
                self._need_clear = True   # cleared once we see Flushed
                await self._ws.send(json.dumps({"type": "Speak", "text": text}))
                await self._ws.send(json.dumps({"type": "Flush"}))
                while True:
                    frame = await asyncio.wait_for(self._ws.recv(), timeout=20.0)
                    if isinstance(frame, bytes):
                        yield frame
                    else:
                        msg = json.loads(frame)
                        if msg.get("type") == "Flushed":
                            self._need_clear = False
                            return
                        if msg.get("type") == "Error":
                            raise RuntimeError(f"Aura WS error: {msg}")
            except (asyncio.CancelledError, GeneratorExit):
                # Barge-in mid-sentence: leave _need_clear set so the next
                # sentence flushes stale audio first.
                raise
            except websockets.exceptions.ConnectionClosed:
                self._ws = None
                raise RuntimeError("Aura WS closed mid-synthesis")

    async def close(self):
        if self._ws is not None:
            try:
                await self._ws.send(json.dumps({"type": "Close"}))
                await self._ws.close()
            except Exception:
                pass
            self._ws = None


class DeepgramAuraTTS:
    """Deepgram Aura over REST — full-clip synthesis (TTFB grows with
    text length); kept as fallback and for offline tools."""

    provider_name = "deepgram-aura"

    def __init__(self, config):
        self.config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        params = {
            "model": self.config.tts_voice,
            "encoding": "mulaw",
            "sample_rate": "8000",
            "container": "none",
        }
        api_key = self.config.deepgram_tts_api_key or self.config.deepgram_api_key
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }
        async with self._client.stream(
            "POST", DEEPGRAM_SPEAK_URL, params=params, headers=headers,
            json={"text": text},
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Deepgram TTS {resp.status_code}: {body[:200]!r}")
            async for chunk in resp.aiter_bytes(chunk_size=1600):  # 200 ms
                if chunk:
                    yield chunk

    async def close(self):
        await self._client.aclose()


class ElevenLabsTTS:
    """ElevenLabs Flash v2.5 — lowest-latency EL model, ulaw_8000 output."""

    provider_name = "elevenlabs"

    def __init__(self, config):
        self.config = config
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0))

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        url = ELEVENLABS_STREAM_URL.format(voice_id=self.config.elevenlabs_voice_id)
        params = {"output_format": "ulaw_8000", "optimize_streaming_latency": "3"}
        headers = {
            "xi-api-key": self.config.elevenlabs_api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.config.elevenlabs_model,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5},
        }
        async with self._client.stream(
            "POST", url, params=params, headers=headers, json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"ElevenLabs TTS {resp.status_code}: {body[:200]!r}")
            async for chunk in resp.aiter_bytes(chunk_size=1600):
                if chunk:
                    yield chunk

    async def close(self):
        await self._client.aclose()


def make_tts(config):
    """Primary provider by config; the session falls back at call time."""
    if config.tts_provider == "elevenlabs" and config.elevenlabs_api_key:
        return ElevenLabsTTS(config)
    return DeepgramAuraWS(config)


def make_fallback_tts(config, primary):
    """Return a second provider for graceful degradation, if keys allow."""
    if primary.provider_name != "elevenlabs" and config.elevenlabs_api_key:
        return ElevenLabsTTS(config)
    if config.deepgram_api_key:
        return DeepgramAuraTTS(config)   # REST path as last resort
    return None
