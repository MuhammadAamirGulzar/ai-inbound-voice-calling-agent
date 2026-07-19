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

from typing import AsyncIterator

import httpx

DEEPGRAM_SPEAK_URL = "https://api.deepgram.com/v1/speak"
ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"


class DeepgramAuraTTS:
    """Deepgram Aura — same API key as STT, mulaw@8k native output."""

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
        headers = {
            "Authorization": f"Token {self.config.deepgram_api_key}",
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
    return DeepgramAuraTTS(config)


def make_fallback_tts(config, primary):
    """Return a second provider for graceful degradation, if keys allow."""
    if primary.provider_name != "elevenlabs" and config.elevenlabs_api_key:
        return ElevenLabsTTS(config)
    if primary.provider_name != "deepgram-aura" and config.deepgram_api_key:
        return DeepgramAuraTTS(config)
    return None
