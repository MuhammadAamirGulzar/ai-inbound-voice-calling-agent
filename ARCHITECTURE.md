# Architecture

## System context

Inbound AI voice agent for restaurants: Twilio delivers PSTN calls as
bidirectional μ-law 8 kHz WebSocket media streams; the platform answers
as a per-tenant AI agent, takes orders, escalates to humans, and records
transcripts, orders, and voice-quality metrics.

## Call path (streaming pipeline)

```
Twilio Media Stream WS
   │  {event: start | media | mark | stop}
   ▼
twilio_routes.run_streaming_call            ── plumbing per call
   │  • resolves tenant by dialed number (system prompt, voice_settings)
   │  • creates call_logs rows, registers in live-call registry
   │  • races WS receive against session.ended
   ▼
voice.pipeline.CallSession                  ── one asyncio task tree per call
   │
   ├── voice.stt.DeepgramLiveSTT
   │     wss://api.deepgram.com/v1/listen?encoding=mulaw&sample_rate=8000
   │     &model=nova-3&language=multi&interim_results=true
   │     &endpointing=300&utterance_end_ms=1000&vad_events=true
   │     → STTEvent queue: interim / final(speech_final) / utterance_end /
   │       speech_started
   │
   ├── turn detection
   │     accumulate `final` segments → close turn on speech_final
   │     (endpointing silence) or utterance_end (word-gap fallback)
   │
   ├── _respond() task (one per turn)
   │     RAG context (≤1.5s hard budget, concurrent menu+chunks lookups)
   │     → StreamingLLM (OpenAI-compatible, stream=True, tools)
   │     → SentenceStream: emit speakable chunk at first sentence boundary
   │     → TTS stream (Aura/ElevenLabs, μ-law 8k native, provider fallback)
   │     → transport.send_media frames + send_mark after each sentence
   │
   ├── barge-in
   │     interim transcript with real words while agent speaking
   │     → cancel _respond task → snapshot played marks → Twilio `clear`
   │     → truncate assistant history to sentences actually heard
   │
   └── call control tools (LLM-invoked)
         transfer_call → speak hold line → Twilio REST redirect to <Dial>
                         with whisper-summary callback on the staff leg
         end_call      → speak farewell → wait for its mark → REST hangup
```

## Design decisions

1. **μ-law end to end.** Twilio, Deepgram (input) and Aura/ElevenLabs
   (output) all speak μ-law 8 kHz. The hot path does zero transcoding —
   the numpy codec in `voice/audio.py` exists only for the simulator and
   wav export. Removes CPU, latency, and the old librosa/audioop deps.

2. **Sentence-level pipelining, not full-reply TTS.** First-sentence
   synthesis starts while the LLM is mid-generation. Combined with token
   streaming and Groq-class inference, perceived response time is
   dominated by the endpointing wait, which is a *tunable* (300 ms
   default, per tenant).

3. **Marks as ground truth for "what the caller heard".** Twilio `mark`
   frames come back only when the audio ahead of them has played. On
   barge-in the assistant history is truncated to played sentences, so
   the LLM's view of the conversation matches the caller's reality —
   otherwise the agent believes it delivered information the caller
   never heard.

4. **`clear` on barge-in.** Stopping our own sending is not enough;
   Twilio buffers outbound audio aggressively and keeps playing for
   seconds. The `clear` frame flushes Twilio's buffer — this is the
   difference between an agent that *feels* interruptible and one that
   talks over the caller.

5. **Turn closure = speech_final OR utterance_end.** Endpointing handles
   the common case fast; `UtteranceEnd` (word-timing based) closes turns
   on noisy lines where the VAD never sees clean silence.

6. **Async everywhere, threads nowhere (hot path).** Per-call cost is a
   few coroutines and one upstream WS + short-lived HTTPS streams.
   Sync SQLAlchemy is confined to `asyncio.to_thread`. Concurrency
   limits come from provider rate limits, not server CPU.

7. **Tools for call control.** `transfer_call` / `end_call` are LLM
   function calls, so escalation policy lives in the per-tenant prompt
   ("transfer complaints and refund requests"), not in code.

8. **Degradation ladder.** TTS falls back Aura⇄ElevenLabs per sentence;
   STT retries once on connect; RAG times out to no-context; a missing
   Twilio credential disables transfers/hangup but never breaks the
   conversation; missing cloud keys entirely drops the deployment to the
   legacy offline pipeline (`twilio_legacy.py`).

## Metrics model

`voice/metrics.py` — per turn:

| field | meaning |
|---|---|
| `response_ms` | caller stopped speaking → first audio frame sent (incl. endpointing wait) |
| `llm_ttft_ms` | STT final → first LLM token |
| `tts_ttfb_ms` | first LLM token → first TTS audio byte |
| `pipeline_ms` | STT final → first audio frame sent (server-side work) |
| `barged_in`, `tool_call`, `error`, `sentences` | turn events |

Call summary adds p50/p95/max response, barge-in and error counts, and
provider identifiers. Persisted to `call_logs.metrics` (JSON), rendered
in the admin call-detail page, and emitted as one structured
`call_metrics` log line per call for log-based dashboards.

## Testing / evals

- `tests/test_voice.py` — provider-free unit tests: μ-law codec
  round-trip, sentence chunking (decimals, Urdu marks, forced cuts),
  latency math, and barge-in behaviour (clear sent, history truncated,
  backchannels ignored, mid-response speech queued) using fake
  STT/LLM/TTS.
- `tools/call_simulator.py` — a Twilio impersonator for E2E evals
  without telephony: real-time framing, line silence, playback-buffer
  emulation with mark echo, `clear` verification, multi-turn scripted
  scenarios (caller lines synthesized via TTS), concurrency mode with
  aggregate percentiles, JSON reports for CI regression tracking.

## Data layer

SQLAlchemy models: users/restaurants (tenants), `AgentConfiguration`
(+ `voice_settings` JSON overriding VoiceConfig per tenant),
`call_logs` (transcript, status, duration, `metrics` JSON), orders
(auto-extracted from completed transcripts). Additive column migration
runs at startup (`sql/database.ensure_columns`); SQLite for local dev,
Postgres in production.

## Scaling notes

- Stateless per call → horizontal scale behind a WS-capable LB; no
  sticky sessions needed beyond the lifetime of one WebSocket.
- Watch provider ceilings: Deepgram concurrent stream limits and LLM
  TPM/RPM are the real capacity bounds; the structured metrics stream
  is the input for capacity dashboards.
- Post-call work (order extraction) already runs off the hot path;
  a task queue is the next step if it grows.
