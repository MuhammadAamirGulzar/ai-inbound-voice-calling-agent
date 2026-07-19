"""
Process-wide voice telemetry, exposed in Prometheus text format.

Aggregates across all calls handled by this instance:

  voice_calls_active                gauge   calls in flight right now
  voice_calls_total                 counter calls handled since start
  voice_turns_total                 counter conversational turns
  voice_barge_ins_total             counter caller interruptions
  voice_transfers_total             counter successful human handoffs
  voice_provider_errors_total       counter STT/LLM/TTS failures (labeled)
  voice_tts_fallbacks_total         counter primary→secondary TTS switches
  voice_turn_response_ms            histogram perceived response latency

Scrape /metrics with Prometheus and the p50/p95 panels come from
histogram_quantile() over voice_turn_response_ms_bucket. Only aggregates
are exposed — no transcripts, numbers, or tenant data.
"""

from __future__ import annotations

import threading

_BUCKETS = (250, 500, 750, 1000, 1500, 2000, 3000, 5000)


class VoiceTelemetry:
    def __init__(self):
        self._lock = threading.Lock()
        self.calls_active = 0
        self.calls_total = 0
        self.turns_total = 0
        self.barge_ins_total = 0
        self.transfers_total = 0
        self.tts_fallbacks_total = 0
        self.provider_errors = {"stt": 0, "llm": 0, "tts": 0}
        self._bucket_counts = [0] * (len(_BUCKETS) + 1)  # +Inf last
        self._response_sum_ms = 0.0
        self._response_count = 0

    # ── recording ────────────────────────────────────────────────────────
    def call_started(self):
        with self._lock:
            self.calls_active += 1
            self.calls_total += 1

    def call_ended(self):
        with self._lock:
            self.calls_active = max(0, self.calls_active - 1)

    def provider_error(self, stage: str):
        with self._lock:
            if stage in self.provider_errors:
                self.provider_errors[stage] += 1

    def tts_fallback(self):
        with self._lock:
            self.tts_fallbacks_total += 1

    def record_turn(self, turn):
        """Aggregate one finished TurnMetrics (also on barge-in/cancel)."""
        with self._lock:
            self.turns_total += 1
            if turn.barged_in:
                self.barge_ins_total += 1
            if turn.tool_call == "transfer_call":
                self.transfers_total += 1
            # STT/TTS failures are counted where they happen; only LLM
            # errors are attributed here (they surface as turn errors).
            if turn.error and turn.error.startswith("llm"):
                self.provider_errors["llm"] += 1
            response_ms = turn.response_ms
            if response_ms is not None:
                self._response_sum_ms += response_ms
                self._response_count += 1
                for i, edge in enumerate(_BUCKETS):
                    if response_ms <= edge:
                        self._bucket_counts[i] += 1
                        break
                else:
                    self._bucket_counts[-1] += 1

    # ── exposition ───────────────────────────────────────────────────────
    def render_prometheus(self) -> str:
        with self._lock:
            lines = [
                "# TYPE voice_calls_active gauge",
                f"voice_calls_active {self.calls_active}",
                "# TYPE voice_calls_total counter",
                f"voice_calls_total {self.calls_total}",
                "# TYPE voice_turns_total counter",
                f"voice_turns_total {self.turns_total}",
                "# TYPE voice_barge_ins_total counter",
                f"voice_barge_ins_total {self.barge_ins_total}",
                "# TYPE voice_transfers_total counter",
                f"voice_transfers_total {self.transfers_total}",
                "# TYPE voice_tts_fallbacks_total counter",
                f"voice_tts_fallbacks_total {self.tts_fallbacks_total}",
                "# TYPE voice_provider_errors_total counter",
            ]
            for stage, count in self.provider_errors.items():
                lines.append(
                    f'voice_provider_errors_total{{stage="{stage}"}} {count}')
            lines.append("# TYPE voice_turn_response_ms histogram")
            cumulative = 0
            for i, edge in enumerate(_BUCKETS):
                cumulative += self._bucket_counts[i]
                lines.append(
                    f'voice_turn_response_ms_bucket{{le="{edge}"}} {cumulative}')
            cumulative += self._bucket_counts[-1]
            lines.append(
                f'voice_turn_response_ms_bucket{{le="+Inf"}} {cumulative}')
            lines.append(f"voice_turn_response_ms_sum {self._response_sum_ms}")
            lines.append(f"voice_turn_response_ms_count {self._response_count}")
            return "\n".join(lines) + "\n"


telemetry = VoiceTelemetry()
