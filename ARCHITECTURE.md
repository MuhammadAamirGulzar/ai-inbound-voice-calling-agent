# Architecture

## System Context

AIColdCaller provides real-time voice conversations between end users and AI sales agents.

Core responsibilities:
- Ingest audio from client sessions
- Transcribe speech to text
- Generate contextual responses through LLM backends
- Synthesize voice responses in near real time
- Persist interaction data for analytics and traceability

## High-Level Components

1. Web Application Layer
- FastAPI routes for UI and API endpoints
- WebSocket endpoint for streaming conversation flow
- Jinja templates and static assets for dashboards and setup pages

2. Conversation Orchestration Layer
- Session-aware chat orchestration in the main app service
- Agent configuration and prompt assembly from campaign context
- Response-time instrumentation and operational logging

3. AI Runtime Layer
- STT adapters: faster-whisper, Deepgram, HF variants
- LLM adapters: OpenAI-compatible and local runtime adapters
- TTS adapters: ElevenLabs, XTTS, Piper, and alternatives

4. Data Layer
- SQLAlchemy models for users, organizations, teams, agents, and chat history
- CRUD abstraction for transactional operations
- SQLite default storage with migration path to managed relational databases

5. Knowledge Ingestion Layer
- Embedded crawler module (`gpt-crawler/`) for extracting product/site context
- Crawl output condensation path for prompt budget control

## Request Flow (Realtime)

1. Client opens WebSocket session.
2. Audio chunks are streamed to backend.
3. STT adapter emits transcribed user utterances.
4. LLM adapter generates response text from active agent context.
5. TTS adapter returns audio chunks to the same session.
6. Interaction metrics and transcript entries are persisted.

## Operational Concerns

- Secrets are sourced from environment variables.
- Runtime artifacts are excluded from version control.
- Passwords are persisted as secure hashes.
- Metrics should be exported to external observability tooling in production.

## Scaling Notes

- Separate API and heavy inference workloads where possible.
- Introduce task queues for asynchronous post-call processing.
- Move to managed Postgres for concurrent production workloads.
- Add Redis (or equivalent) for session coordination in multi-instance deployments.
