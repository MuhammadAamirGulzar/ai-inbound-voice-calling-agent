"""
VoiceConfig — one immutable-ish config object per call.

Values come from three layers, later layers override earlier ones:
  1. Hard defaults tuned for telephony latency.
  2. Environment variables (deployment-wide).
  3. The restaurant's AgentConfiguration.voice_settings JSON (per tenant).

Per-tenant overrides let one deployment serve e.g. an English hotel desk
with Deepgram+Aura and an Urdu restaurant with a different STT language
and ElevenLabs voice, without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _placeholder(value: str) -> bool:
    return (not value) or value.startswith("replace-with")


@dataclass
class VoiceConfig:
    # ── Providers ────────────────────────────────────────────────────────
    stt_provider: str = "deepgram"            # deepgram
    llm_provider: str = "openai-compatible"   # groq / openai / ollama — all OpenAI-compatible
    tts_provider: str = "deepgram"            # deepgram (Aura) | elevenlabs

    # ── Credentials / endpoints ──────────────────────────────────────────
    deepgram_api_key: str = ""
    elevenlabs_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    # ── STT tuning ───────────────────────────────────────────────────────
    stt_model: str = "nova-3"
    # "multi" enables Nova-3 code-switching (English + others in one call).
    stt_language: str = "multi"
    # Silence (ms) after speech before Deepgram finalizes the utterance.
    # This is the endpointing knob: lower = snappier turns but more
    # mid-sentence cutoffs; 300 ms is a good telephony default.
    endpointing_ms: int = 300
    # Word-gap fallback finalization for noisy endings.
    utterance_end_ms: int = 1000
    stt_keywords: str = ""  # comma-separated boost terms (menu items, names)

    # ── TTS tuning ───────────────────────────────────────────────────────
    tts_voice: str = "aura-2-thalia-en"       # Deepgram Aura voice
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    elevenlabs_model: str = "eleven_flash_v2_5"   # lowest-latency EL model

    # ── Turn-taking / barge-in ───────────────────────────────────────────
    barge_in_enabled: bool = True
    # "transcript": interrupt only once Deepgram hears actual words
    #               (robust against phone-line echo / coughs).
    # "vad":        interrupt on Deepgram SpeechStarted (fastest, but can
    #               false-trigger on speakerphone echo).
    barge_in_mode: str = "transcript"
    # Minimum interim-transcript length (chars) that counts as a barge-in.
    barge_in_min_chars: int = 3
    # Backchannels that should NOT interrupt the agent mid-sentence.
    barge_in_ignore: tuple = ("ok", "okay", "yes", "yeah", "hmm", "mhm",
                              "acha", "achha", "haan", "han", "ji", "theek")

    # ── Conversation ─────────────────────────────────────────────────────
    system_prompt: str = "You are a helpful restaurant voice assistant."
    greeting: str = "Hello! How can I help you today?"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.3
    # Total time budget for RAG context lookups on the hot path.
    rag_timeout_s: float = 1.5
    rag_base_url: str = "http://127.0.0.1:8001"
    rag_business_id: str = ""

    # ── Transfers / call control ─────────────────────────────────────────
    transfer_number: str = ""          # human escalation target (E.164)
    public_base_url: str = ""          # needed for warm-transfer whisper TwiML
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""

    # ── Misc ─────────────────────────────────────────────────────────────
    max_call_seconds: int = 900

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        cfg = cls(
            deepgram_api_key=_env("DEEPGRAM_API_KEY"),
            elevenlabs_api_key=_env("ELEVENLABS_API_KEY"),
            llm_api_key=_env("GROQ_API_KEY") or _env("LLM_API_KEY") or _env("MODEL_API_KEY"),
            llm_base_url=_env("LLM_BASE_URL") or _env("MODEL_ENDPOINT_URL") or "https://api.groq.com/openai/v1",
            llm_model=_env("LLM_MODEL") or _env("MODEL_NAME") or "llama-3.3-70b-versatile",
            stt_model=_env("STT_MODEL", "nova-3"),
            stt_language=_env("STT_LANGUAGE", "multi"),
            tts_provider=_env("TTS_PROVIDER", "deepgram"),
            tts_voice=_env("TTS_VOICE", "aura-2-thalia-en"),
            elevenlabs_voice_id=_env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            transfer_number=_env("TRANSFER_NUMBER"),
            public_base_url=_env("PUBLIC_BASE_URL").rstrip("/"),
            twilio_account_sid=_env("TWILIO_ACCOUNT_SID"),
            twilio_auth_token=_env("TWILIO_AUTH_TOKEN"),
            rag_base_url=_env("RAG_BASE_URL", "http://127.0.0.1:8001"),
            rag_business_id=_env("RAG_BUSINESS_ID"),
        )
        if _env("ENDPOINTING_MS"):
            cfg.endpointing_ms = int(_env("ENDPOINTING_MS"))
        if _env("BARGE_IN_MODE"):
            cfg.barge_in_mode = _env("BARGE_IN_MODE")
        if _env("BARGE_IN_ENABLED"):
            cfg.barge_in_enabled = _env("BARGE_IN_ENABLED").lower() not in ("0", "false", "no")
        if _placeholder(cfg.twilio_account_sid):
            cfg.twilio_account_sid = ""
        if _placeholder(cfg.twilio_auth_token):
            cfg.twilio_auth_token = ""
        if _placeholder(cfg.deepgram_api_key):
            cfg.deepgram_api_key = ""
        if _placeholder(cfg.elevenlabs_api_key):
            cfg.elevenlabs_api_key = ""
        return cfg

    def apply_overrides(self, overrides: dict | None) -> "VoiceConfig":
        """Apply a per-tenant voice_settings dict (unknown keys ignored)."""
        if not overrides:
            return self
        valid = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in valid and value not in (None, ""):
                setattr(self, key, value)
        return self

    @property
    def streaming_ready(self) -> bool:
        """True when the cloud streaming pipeline can run."""
        return bool(self.deepgram_api_key and self.llm_api_key)
