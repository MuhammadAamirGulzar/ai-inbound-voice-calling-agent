"""
voice — real-time streaming voice engine for telephony calls.

This package implements the production call pipeline:

    Twilio Media Stream (mu-law 8 kHz)
        └─> DeepgramLiveSTT   (streaming, interim results, endpointing)
              └─> CallSession (turn manager, barge-in, metrics)
                    └─> StreamingLLM  (token streaming, tool calls)
                          └─> sentence chunker
                                └─> streaming TTS (mu-law 8 kHz out)
                                      └─> Twilio media/mark/clear frames

Design goals:
- No blocking work on the event loop (DB access is pushed to threads).
- No GPU / local model on the hot path: one asyncio task tree per call,
  so a single instance can hold many concurrent calls.
- Audio stays in Twilio's native mu-law 8 kHz end to end (Deepgram accepts
  it, Aura/ElevenLabs emit it) — zero resampling on the hot path.
- Every turn is measured: STT finalization, LLM first token, TTS first
  byte, first audio frame to the caller, barge-ins.
"""

from .config import VoiceConfig
from .metrics import CallMetrics, TurnMetrics

__all__ = ["VoiceConfig", "CallMetrics", "TurnMetrics"]
