# Changelog

All notable changes to this repository will be documented in this file.

The format follows Keep a Changelog, and this project uses semantic versioning principles for release notes.

## [Unreleased]

### Added
- **Streaming voice engine** (`voice/`): cloud pipeline with Deepgram
  Nova-3 live STT (interim results, endpointing, VAD events), token-
  streaming OpenAI-compatible LLM (Groq/OpenAI/Ollama), and streaming
  μ-law-native TTS (Deepgram Aura / ElevenLabs) — no transcoding on the
  hot path, one asyncio task tree per call.
- Barge-in with Twilio `clear` buffer flush, mark-based playback
  tracking, and history truncation to sentences the caller actually heard.
- LLM call-control tools: `transfer_call` (warm transfer with whisper
  summary to staff) and `end_call` (graceful REST hangup).
- Per-turn latency metrics (STT final → LLM TTFT → TTS TTFB → first
  audio frame), call p50/p95 summaries, persisted to `call_logs.metrics`
  and rendered in the admin call-detail page.
- Call simulator (`tools/call_simulator.py`): Twilio-free eval harness
  with scripted multi-turn scenarios, barge-in verification, concurrency
  mode, and JSON latency reports.
- Unit test suite (`tests/test_voice.py`) covering the μ-law codec,
  sentence chunker, metrics math, and barge-in state machine.
- `requirements-streaming.txt` slim torch-free deployment profile;
  `VOICE_PIPELINE` / `RAG_AUTOSTART` deployment knobs; per-tenant
  `AgentConfiguration.voice_settings` overrides with additive startup
  column migration.

### Changed
- `twilio_routes.py` now dispatches between the streaming engine and the
  legacy local pipeline (moved intact to `twilio_legacy.py`, still fully
  offline-capable).
- README/ARCHITECTURE rewritten for this repository's actual product
  (previously duplicated from the outbound sales project).

### Previous release

### Added
- Production-grade repository documentation and governance files.
- Secure environment template for deployment configuration.

### Changed
- Main README rewritten to reflect product architecture and operational setup.
- Repository hygiene rules expanded in `.gitignore`.

### Security
- Removed tracked secrets, certificates, and generated runtime artifacts from version control.
- Updated user authentication persistence to store password hashes instead of plaintext.
