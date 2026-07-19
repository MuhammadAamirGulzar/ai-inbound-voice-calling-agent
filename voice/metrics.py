"""
Per-turn latency instrumentation.

Every conversational turn records a timeline of monotonic timestamps:

    user stops speaking
      └─ endpointing wait (provider-side, ≈ endpointing_ms)
    stt_final          — final transcript for the utterance arrived
    llm_first_token    — first token streamed back from the LLM
    tts_first_byte     — first audio byte from TTS for the first sentence
    first_audio_sent   — first media frame handed to the telephony socket

The headline number is `response_ms`: user-stopped-speaking → first audio
frame to the caller (what a caller actually perceives as "how fast it
answers"). We estimate user-stop as (stt_final arrival − endpointing_ms)
since the provider by definition waited that long in silence.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


def _now_ms() -> float:
    return time.monotonic() * 1000.0


@dataclass
class TurnMetrics:
    turn_index: int = 0
    user_text: str = ""
    agent_text: str = ""

    # Monotonic ms timestamps (None until the event happens)
    t_stt_final: Optional[float] = None
    t_llm_first_token: Optional[float] = None
    t_llm_done: Optional[float] = None
    t_tts_first_byte: Optional[float] = None
    t_first_audio_sent: Optional[float] = None

    endpointing_ms: int = 0
    sentences: int = 0
    barged_in: bool = False
    tool_call: str = ""
    error: str = ""

    # ── event markers ────────────────────────────────────────────────────
    def mark_stt_final(self):
        if self.t_stt_final is None:
            self.t_stt_final = _now_ms()

    def mark_llm_first_token(self):
        if self.t_llm_first_token is None:
            self.t_llm_first_token = _now_ms()

    def mark_llm_done(self):
        self.t_llm_done = _now_ms()

    def mark_tts_first_byte(self):
        if self.t_tts_first_byte is None:
            self.t_tts_first_byte = _now_ms()

    def mark_first_audio_sent(self):
        if self.t_first_audio_sent is None:
            self.t_first_audio_sent = _now_ms()

    # ── derived latencies (ms) ───────────────────────────────────────────
    @staticmethod
    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return round(b - a, 1)

    @property
    def llm_ttft_ms(self) -> Optional[float]:
        """STT final → first LLM token."""
        return self._delta(self.t_stt_final, self.t_llm_first_token)

    @property
    def llm_total_ms(self) -> Optional[float]:
        return self._delta(self.t_stt_final, self.t_llm_done)

    @property
    def tts_ttfb_ms(self) -> Optional[float]:
        """First LLM token → first TTS audio byte."""
        return self._delta(self.t_llm_first_token, self.t_tts_first_byte)

    @property
    def pipeline_ms(self) -> Optional[float]:
        """STT final → first audio frame sent (server-side work only)."""
        return self._delta(self.t_stt_final, self.t_first_audio_sent)

    @property
    def response_ms(self) -> Optional[float]:
        """Perceived: user stopped speaking → first audio frame sent."""
        p = self.pipeline_ms
        if p is None:
            return None
        return round(p + self.endpointing_ms, 1)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn_index,
            "user_text": self.user_text,
            "agent_text": self.agent_text,
            "endpointing_ms": self.endpointing_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "llm_total_ms": self.llm_total_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
            "pipeline_ms": self.pipeline_ms,
            "response_ms": self.response_ms,
            "sentences": self.sentences,
            "barged_in": self.barged_in,
            "tool_call": self.tool_call,
            "error": self.error,
        }


def _percentile(values: list, pct: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


@dataclass
class CallMetrics:
    call_sid: str = ""
    stream_sid: str = ""
    started_at: float = field(default_factory=time.time)
    turns: list = field(default_factory=list)  # list[TurnMetrics]
    greeting_first_audio_ms: Optional[float] = None
    stt_connect_ms: Optional[float] = None
    stt_reconnects: int = 0
    stt_provider: str = ""
    llm_model: str = ""
    tts_provider: str = ""
    disconnect_reason: str = ""

    def new_turn(self, user_text: str, endpointing_ms: int) -> TurnMetrics:
        turn = TurnMetrics(
            turn_index=len(self.turns) + 1,
            user_text=user_text,
            endpointing_ms=endpointing_ms,
        )
        turn.mark_stt_final()
        self.turns.append(turn)
        return turn

    def summary(self) -> dict:
        responses = [t.response_ms for t in self.turns if t.response_ms is not None]
        return {
            "turn_count": len(self.turns),
            "barge_ins": sum(1 for t in self.turns if t.barged_in),
            "errors": sum(1 for t in self.turns if t.error),
            "response_ms_p50": _percentile(responses, 50),
            "response_ms_p95": _percentile(responses, 95),
            "response_ms_max": max(responses) if responses else None,
            "greeting_first_audio_ms": self.greeting_first_audio_ms,
            "stt_connect_ms": self.stt_connect_ms,
            "stt_reconnects": self.stt_reconnects,
            "providers": {
                "stt": self.stt_provider,
                "llm": self.llm_model,
                "tts": self.tts_provider,
            },
            "disconnect_reason": self.disconnect_reason,
        }

    def to_dict(self) -> dict:
        return {
            "call_sid": self.call_sid,
            "stream_sid": self.stream_sid,
            "summary": self.summary(),
            "turns": [t.to_dict() for t in self.turns],
        }

    def log_line(self) -> str:
        return json.dumps({"event": "call_metrics", **self.to_dict()},
                          ensure_ascii=False)
