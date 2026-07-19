"""
telephony — transport layer between phone networks and the voice engine.

    twilio_routes   Twilio webhook + Media Streams WS, pipeline dispatch
    legacy          offline thread-per-call pipeline (whisper/ollama/piper)
    sip_routes      SIP media-stream bridge (behind SIP_ENABLED)
    twilio_audio    mu-law <-> PCM helpers for the legacy Twilio path
    sip_audio       PCM helpers for the SIP path
    registry        stream-sid -> live WebSocket map (admin live-calls page)
"""
