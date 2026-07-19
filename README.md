# AI Inbound Voice Agent — Real-Time Restaurant Call Answering

A production-oriented **inbound voice AI platform**: callers dial a
restaurant's phone number, Twilio streams the call audio to this server,
and a streaming STT → LLM → TTS pipeline answers, takes the order,
handles interruptions, escalates to a human when needed, and logs a
per-turn latency waterfall for every call.

Built and operated as a multi-tenant system (multiple restaurants, one
deployment) with an admin dashboard for live calls, transcripts, orders,
and voice-quality metrics.

```
Caller ──PSTN──> Twilio ──Media Stream (WS, μ-law 8kHz)──> FastAPI
                                                             │
                       ┌─────────────────────────────────────┤
                       │            CallSession (asyncio, one task tree per call)
                       │
   Deepgram Nova-3 ◄───┤ streaming STT: interim results, endpointing,
   (live WebSocket)    │ VAD events, keyterm boosting, language=multi
                       │
   Groq / OpenAI   ◄───┤ token-streaming LLM + tool calls
   (any OpenAI-        │ (transfer_call, end_call) + RAG context
    compatible API)    │
                       │ sentence chunker (emits speakable chunks
                       │ as tokens arrive — Urdu/Arabic aware)
                       │
   Deepgram Aura   ◄───┤ streaming TTS, μ-law 8kHz native output
   or ElevenLabs       │ (zero transcoding on the hot path)
                       │
                       └──> media frames + mark/clear ──> Twilio ──> Caller
```

## Why this architecture

| Concern | Approach |
|---|---|
| **Latency** | Everything streams. STT finalizes turns with ~300 ms endpointing; the LLM streams tokens; TTS synthesis starts on the *first complete sentence*, not the full reply; first audio is typically on the wire while the LLM is still generating. |
| **Barge-in** | STT runs full-duplex during agent speech. Real caller words (not backchannels like "ok/haan") cancel the response task **and send Twilio a `clear` frame** — without the clear, Twilio keeps playing its buffer for seconds after the caller interrupts. Chat history is truncated to the sentences the caller actually *heard*, tracked via Twilio `mark` events. |
| **Audio formats** | Twilio's native μ-law 8 kHz goes straight to Deepgram (`encoding=mulaw`), and both Aura and ElevenLabs emit μ-law 8 kHz — **no resampling or transcoding anywhere on the hot path**. |
| **Concurrency** | One asyncio task tree per call; no threads, no local models on the hot path; DB writes pushed to worker threads. A small instance can hold dozens of simultaneous calls; capacity is provider-rate-limited, not CPU-bound. |
| **Transfers** | The LLM decides via a `transfer_call` tool. The caller hears a hold line; the staff member hears an AI-generated **whisper summary** of the call before being bridged (warm transfer); no-answer falls back to an apology instead of dead air. |
| **Reliability** | TTS provider fallback (Aura ⇄ ElevenLabs), STT reconnect-on-start retry, RAG lookups under a hard 1.5 s budget (degrade to no context, never a slow answer), max-call-duration watchdog, graceful `end_call` hangup via Twilio REST. |
| **Observability** | Every turn records a timestamp waterfall: STT final → LLM first token → TTS first byte → first audio frame sent, plus barge-ins, tool calls, and errors. Persisted per call (JSON), rendered in the admin call-detail page, and emitted as a structured `call_metrics` log line. |
| **Multi-tenant** | Prompt, greeting, language, voices, endpointing, and transfer number are per-restaurant (`AgentConfiguration.voice_settings`), resolved per call by the dialed number. |

## Two pipelines, one platform

| | Streaming (production) | Legacy (offline fallback) |
|---|---|---|
| STT | Deepgram Nova-3 live WS | Faster-Whisper (local) |
| LLM | Groq / OpenAI / any compatible | Ollama (local) |
| TTS | Deepgram Aura / ElevenLabs | Piper (local) |
| Turn latency | sub-second design target | multi-second (sequential) |
| Barge-in | yes (with Twilio `clear`) | no |
| Transfers / hangup tools | yes | no |
| Dependencies | ~50 MB, no GPU | torch + models (~4 GB) |

`VOICE_PIPELINE=auto` (default) picks streaming when `DEEPGRAM_API_KEY`
and an LLM key are present, and falls back to the fully local legacy
pipeline otherwise — the platform still answers calls air-gapped.

## Quick start (streaming)

```bash
git clone https://github.com/MuhammadAamirGulzar/ai-inbound-voice-calling-agent.git
cd ai-inbound-voice-calling-agent
python -m venv .venv && .venv\Scripts\activate   # source .venv/bin/activate on Linux
pip install -r requirements-streaming.txt

cp .env.example .env
# Set at minimum:
#   DEEPGRAM_API_KEY   — free $200 credit, no card: console.deepgram.com/signup
#   GROQ_API_KEY       — free tier, no card: console.groq.com
#   DATABASE_URL       — postgresql://... (or sqlite:///./local.db for a quick demo)
#   SESSION_MIDDLEWARE_SECRET_KEY / JWT_SECRET_KEY — any long random strings

uvicorn app:app --host 0.0.0.0 --port 8000
```

### Test a full call without a phone

The repo ships a **call simulator** that impersonates a Twilio Media
Stream — including real-time 20 ms μ-law framing, continuous line
silence, playback-buffer emulation with `mark` echo, and `clear`
handling — so the entire pipeline can be exercised and measured locally:

```bash
# 3-turn conversation; caller speech is synthesized via Deepgram TTS
python tools/call_simulator.py \
  --say "Hi, do you deliver to DHA phase five?" \
        "How much is a chicken burger?" \
        "Okay, one chicken burger then. That's all, thank you."

# Barge-in test: interrupt the bot 1.5s into its reply,
# verify a clear frame arrives and measure reaction time
python tools/call_simulator.py --say "Tell me the whole menu please" --barge-in-after 1.5

# Concurrency: 10 simultaneous calls, aggregate p50/p95
python tools/call_simulator.py --say "Hi, what are your opening hours?" --calls 10 --report load.json
```

The report prints per-turn response latency (last caller frame → first
bot audio frame) and saves the bot's audio to `sim_output/*.wav`.

### Go live on a real number

1. Buy/claim a Twilio number (trial credit works) and point its Voice
   webhook to `https://<your-host>/voice` (use ngrok for local dev).
2. Set the number as a restaurant's `order_phone_number` in the admin UI.
3. Set `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` (enables hangup +
   transfers) and `PUBLIC_BASE_URL` + `TRANSFER_NUMBER` (enables warm
   transfer whisper).
4. Call the number.

## Latency budget (per turn, streaming pipeline)

```
caller stops speaking
  ├─ endpointing wait (Deepgram, configurable)        ~300 ms
  ├─ STT final → LLM first token (Groq)               ~100–250 ms
  ├─ first sentence → TTS first byte (Aura)           ~100–200 ms
  └─ first media frame on the Twilio socket           ~10 ms
                                        target E2E:    ≲ 700–800 ms
```

Each stage is measured per turn and persisted — the admin call-detail
page shows the waterfall, flags responses > 1.5 s, barge-ins, and tool
calls. The `ENDPOINTING_MS` / `BARGE_IN_MODE` knobs trade snappiness
against cutoffs per tenant.

## Multilingual

The demo tenant runs an Urdu/English code-switching agent (Roman-Urdu
prompts, `STT_LANGUAGE=multi` for Nova-3 code-switching, Urdu-capable
voices). The sentence chunker handles Urdu/Arabic punctuation (۔ ؟ ،),
and barge-in backchannel filtering includes Urdu fillers ("haan", "ji",
"acha"). Language, STT model and voices are per-tenant settings.

## Repository layout

```
voice/                  streaming engine (the interesting part)
  pipeline.py           CallSession: turn state machine, barge-in, marks
  stt.py                Deepgram live WS client (interim/endpoint/VAD events)
  llm.py                token-streaming OpenAI-compatible client + call tools
  tts.py                Aura + ElevenLabs streaming TTS (μ-law native)
  sentence_stream.py    incremental sentence chunking for token streams
  metrics.py            per-turn latency waterfall, call summaries
  transfer.py           Twilio REST call control (warm transfer, hangup)
  rag.py                budget-capped async RAG context lookup
  config.py             env + per-tenant VoiceConfig
  audio.py              numpy μ-law codec (simulator/legacy edges only)
twilio_routes.py        webhook, media-stream endpoint, pipeline dispatch
twilio_legacy.py        offline local pipeline (whisper/ollama/piper)
tools/call_simulator.py Twilio-free eval harness (latency/barge-in/load)
tests/test_voice.py     unit tests: codec, chunker, metrics, barge-in
knowledge_rag/          RAG sidecar: menu + business knowledge (Chroma)
app.py, templates/      multi-tenant admin (live calls, orders, analytics)
sql/                    SQLAlchemy models incl. per-call metrics JSON
```

## Operations notes

- **Scaling**: instances are stateless per call; scale horizontally
  behind a WS-capable proxy. Session state is per-connection; call logs
  and metrics land in Postgres.
- **Deploy slim**: `requirements-streaming.txt` (no torch) for cloud;
  `requirements-cpu.txt` adds the offline stack.
- **Keys**: never commit `.env`; all providers are optional-degradable
  except STT+LLM for the streaming path.
- **Known trade-offs**: TTS is per-sentence HTTP streaming (WS input
  streaming to TTS would shave another ~50–100 ms); transcript-mode
  barge-in costs one interim-result round-trip (~200 ms) vs VAD mode but
  is robust against phone-line echo.

## License

CC BY-NC 4.0 — see `LICENSE`.
