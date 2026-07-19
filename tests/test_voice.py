"""
Unit tests for the streaming voice engine — no network, no API keys.

Run:  python -m pytest tests/test_voice.py -q
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice.audio import mulaw_to_pcm16, pcm16_to_mulaw, resample_pcm16
from voice.config import VoiceConfig
from voice.metrics import CallMetrics, TurnMetrics, _percentile
from voice.sentence_stream import SentenceStream, clean_for_tts
from voice.transfer import TwilioCallControl


# ─────────────────────────────────────────────────────────────────────────
# audio codec
# ─────────────────────────────────────────────────────────────────────────
def test_mulaw_roundtrip_accuracy():
    t = np.arange(0, 0.05, 1 / 8000.0)
    pcm = (np.sin(2 * np.pi * 440 * t) * 12000).astype(np.int16)
    decoded = mulaw_to_pcm16(pcm16_to_mulaw(pcm))
    assert len(decoded) == len(pcm)
    # mu-law is lossy; correlation should still be ~1
    corr = np.corrcoef(pcm.astype(float), decoded.astype(float))[0, 1]
    assert corr > 0.99


def test_mulaw_silence_and_extremes():
    silence = np.zeros(100, dtype=np.int16)
    assert np.abs(mulaw_to_pcm16(pcm16_to_mulaw(silence))).max() <= 8
    loud = np.array([32767, -32768, 20000, -20000], dtype=np.int16)
    decoded = mulaw_to_pcm16(pcm16_to_mulaw(loud))
    assert np.sign(decoded).tolist() == [1, -1, 1, -1]


def test_resample_lengths():
    pcm = np.zeros(16000, dtype=np.int16)  # 1 s @ 16 kHz
    assert len(resample_pcm16(pcm, 16000, 8000)) == 8000
    assert len(resample_pcm16(pcm, 16000, 16000)) == 16000


# ─────────────────────────────────────────────────────────────────────────
# sentence chunking
# ─────────────────────────────────────────────────────────────────────────
def feed(stream, text):
    out = []
    for token in text.split(" "):
        out.extend(stream.push(token + " "))
    return out


def test_sentence_stream_basic_split():
    s = SentenceStream()
    out = feed(s, "Hello there. How are you today? I am fine.")
    out.extend(s.flush())
    assert out == ["Hello there.", "How are you today?", "I am fine."]


def test_sentence_stream_decimal_not_split():
    s = SentenceStream()
    out = feed(s, "The burger costs 3.99 dollars. Anything else?")
    out.extend(s.flush())
    assert out[0] == "The burger costs 3.99 dollars."


def test_sentence_stream_urdu_marks():
    s = SentenceStream()
    out = list(s.push("Ji bilkul۔ Aap ka order confirm hai۔ "))
    out.extend(s.flush())
    assert out[0] == "Ji bilkul۔"


def test_sentence_stream_forced_cut_long_text():
    s = SentenceStream(max_chars=50)
    long_text = "word " * 30  # no sentence punctuation at all
    out = list(s.push(long_text))
    out.extend(s.flush())
    assert len(out) >= 2
    assert all(len(c) <= 60 for c in out)


def test_clean_for_tts_strips_markdown():
    assert clean_for_tts("**Hello** *there* `code`") == "Hello there code"
    assert clean_for_tts("[menu](http://x.com) here") == "menu here"


# ─────────────────────────────────────────────────────────────────────────
# metrics
# ─────────────────────────────────────────────────────────────────────────
def test_turn_latency_math():
    turn = TurnMetrics(endpointing_ms=300)
    turn.t_stt_final = 1000.0
    turn.t_llm_first_token = 1250.0
    turn.t_tts_first_byte = 1400.0
    turn.t_first_audio_sent = 1450.0
    assert turn.llm_ttft_ms == 250.0
    assert turn.tts_ttfb_ms == 150.0
    assert turn.pipeline_ms == 450.0
    assert turn.response_ms == 750.0  # includes endpointing wait


def test_call_summary_percentiles():
    metrics = CallMetrics()
    for latency in (400, 500, 600, 700, 2000):
        turn = metrics.new_turn("hi", endpointing_ms=0)
        turn.t_stt_final = 0.0
        turn.t_first_audio_sent = float(latency)
    summary = metrics.summary()
    assert summary["turn_count"] == 5
    assert summary["response_ms_p50"] == 600.0
    assert summary["response_ms_max"] == 2000.0
    assert _percentile([], 50) is None


# ─────────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────────
def test_config_overrides_and_readiness():
    cfg = VoiceConfig(deepgram_api_key="dg", llm_api_key="groq")
    assert cfg.streaming_ready
    cfg.apply_overrides({"tts_provider": "elevenlabs", "endpointing_ms": 500,
                         "bogus_key": "ignored", "stt_language": ""})
    assert cfg.tts_provider == "elevenlabs"
    assert cfg.endpointing_ms == 500
    assert cfg.stt_language == "multi"  # empty override ignored
    assert not hasattr(cfg, "bogus_key")
    assert not VoiceConfig().streaming_ready


def test_transfer_twiml_includes_whisper():
    cfg = VoiceConfig(twilio_account_sid="AC1", twilio_auth_token="tok",
                      public_base_url="https://example.com")
    control = TwilioCallControl(cfg)
    twiml = control.build_transfer_twiml("+15551234567", "Caller wants a refund")
    assert "<Dial" in twiml and "+15551234567" in twiml
    assert "transfer-whisper" in twiml and "refund" in twiml
    # no whisper url without a public base
    cfg.public_base_url = ""
    twiml2 = control.build_transfer_twiml("+15551234567", "x")
    assert "transfer-whisper" not in twiml2


# ─────────────────────────────────────────────────────────────────────────
# pipeline: barge-in behaviour with fake providers
# ─────────────────────────────────────────────────────────────────────────
class FakeTransport:
    def __init__(self):
        self.stream_sid = "SIM"
        self.media_frames = []
        self.marks = []
        self.clears = 0

    async def send_media(self, chunk):
        self.media_frames.append(chunk)

    async def send_mark(self, name):
        self.marks.append(name)

    async def send_clear(self):
        self.clears += 1


class FakeLLM:
    """Streams two sentences slowly, so a barge-in can land mid-reply."""

    def __init__(self, tokens=None, delay=0.02):
        self.tokens = tokens or ["Hello ", "there. ", "Second ", "sentence ",
                                 "coming. "]
        self.delay = delay

    async def stream_reply(self, messages, tools_enabled=True):
        from voice.llm import LLMEvent
        for token in self.tokens:
            await asyncio.sleep(self.delay)
            yield LLMEvent(kind="token", text=token)
        yield LLMEvent(kind="done")


class FakeTTS:
    provider_name = "fake"

    def __init__(self, chunk_count=5, delay=0.01, fail=False):
        self.chunk_count = chunk_count
        self.delay = delay
        self.fail = fail

    async def synthesize(self, text):
        if self.fail:
            raise RuntimeError("fake tts down")
        for _ in range(self.chunk_count):
            await asyncio.sleep(self.delay)
            yield b"\xff" * 160

    async def close(self):
        pass


def make_session(transport, llm=None, tts=None):
    from voice.pipeline import CallSession
    cfg = VoiceConfig(deepgram_api_key="x", llm_api_key="y",
                      system_prompt="test agent", rag_business_id="")
    session = CallSession.__new__(CallSession)
    # Wire only what the response/barge-in path needs (skip real STT/network).
    session.config = cfg
    session.transport = transport
    session.call_sid = ""
    session.caller_number = ""
    session.metrics = CallMetrics()
    session.messages = [{"role": "system", "content": cfg.system_prompt}]
    session.stt_events = asyncio.Queue()
    session.llm = llm or FakeLLM()
    session.tts = tts or FakeTTS()
    session.tts_fallback = None
    from voice.transfer import TwilioCallControl
    session.control = TwilioCallControl(cfg)
    session.state = "listening"
    session.ended = asyncio.Event()
    session._respond_task = None
    session._pending_finals = []
    session._queued_user_text = ""
    session._turn_seq = 0
    session._mark_events = {}
    session._spoken_sentences = []
    session._played_marks = 0
    session._greeted = False
    return session


def test_full_turn_speaks_and_records_metrics():
    async def scenario():
        transport = FakeTransport()
        session = make_session(transport)
        await session._respond("what's on the menu?")
        return transport, session

    transport, session = asyncio.run(scenario())
    assert len(transport.media_frames) > 0
    assert len(transport.marks) == 2          # two sentences → two marks
    turn = session.metrics.turns[0]
    assert turn.sentences == 2
    assert turn.response_ms is not None and turn.response_ms > 0
    roles = [m["role"] for m in session.messages]
    assert roles == ["system", "user", "assistant"]


def test_barge_in_clears_and_truncates_history():
    async def scenario():
        from voice.stt import STTEvent
        transport = FakeTransport()
        session = make_session(
            transport, llm=FakeLLM(delay=0.05), tts=FakeTTS(delay=0.03))
        task = asyncio.create_task(session._respond("tell me everything"))
        session._respond_task = task   # normally set by _close_user_turn
        # wait until the first sentence's mark is queued, then "play" it
        while not transport.marks:
            await asyncio.sleep(0.01)
        session.on_mark(transport.marks[0])
        # caller interrupts with real words while agent is responding
        session._maybe_barge_in(STTEvent(kind="interim", text="wait stop"))
        await asyncio.sleep(0.1)
        try:
            await task
        except asyncio.CancelledError:
            pass
        return transport, session

    transport, session = asyncio.run(scenario())
    assert transport.clears == 1
    assert session.state == "listening"
    assert session.metrics.turns[0].barged_in
    # history keeps only the sentence that was actually played, cut-marked
    assistant = [m for m in session.messages if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"].endswith("—")


def test_backchannel_does_not_barge_in():
    async def scenario():
        from voice.stt import STTEvent
        transport = FakeTransport()
        session = make_session(
            transport, llm=FakeLLM(delay=0.03), tts=FakeTTS(delay=0.02))
        task = asyncio.create_task(session._respond("hello"))
        await asyncio.sleep(0.05)
        session._maybe_barge_in(STTEvent(kind="interim", text="ok"))
        session._maybe_barge_in(STTEvent(kind="interim", text="haan"))
        await task
        return transport, session

    transport, session = asyncio.run(scenario())
    assert transport.clears == 0
    assert not session.metrics.turns[0].barged_in


def test_tts_failure_falls_back_to_secondary():
    async def scenario():
        transport = FakeTransport()
        session = make_session(transport, tts=FakeTTS(fail=True))
        session.tts_fallback = FakeTTS()
        await session._respond("hi")
        return transport, session

    transport, session = asyncio.run(scenario())
    assert len(transport.media_frames) > 0  # fallback provider spoke
    assert not session.metrics.turns[0].error


def test_user_speech_while_responding_is_answered_after():
    async def scenario():
        transport = FakeTransport()
        session = make_session(
            transport, llm=FakeLLM(delay=0.03), tts=FakeTTS(delay=0.02))
        session.config.barge_in_enabled = False
        task = asyncio.create_task(session._respond("first question"))
        await asyncio.sleep(0.05)
        session._pending_finals = ["and also"]
        session._close_user_turn()          # arrives mid-response
        await task
        # the queued turn is answered right after the current one finishes
        if session._respond_task and not session._respond_task.done():
            await session._respond_task
        return session

    session = asyncio.run(scenario())
    assert session._queued_user_text == ""
    assert len(session.metrics.turns) == 2
    assert session.metrics.turns[1].user_text == "and also"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ─────────────────────────────────────────────────────────────────────────
# telemetry
# ─────────────────────────────────────────────────────────────────────────
def test_telemetry_counters_and_histogram():
    from voice.telemetry import VoiceTelemetry
    t = VoiceTelemetry()
    t.call_started()
    turn = TurnMetrics(endpointing_ms=0)
    turn.t_stt_final = 0.0
    turn.t_first_audio_sent = 600.0
    t.record_turn(turn)
    barged = TurnMetrics(endpointing_ms=0)
    barged.barged_in = True
    t.record_turn(barged)
    t.call_ended()

    out = t.render_prometheus()
    assert "voice_calls_active 0" in out
    assert "voice_calls_total 1" in out
    assert "voice_turns_total 2" in out
    assert "voice_barge_ins_total 1" in out
    # 600ms falls in the le=750 bucket, cumulative counts include it upward
    assert 'voice_turn_response_ms_bucket{le="500"} 0' in out
    assert 'voice_turn_response_ms_bucket{le="750"} 1' in out
    assert 'voice_turn_response_ms_bucket{le="+Inf"} 1' in out
    assert "voice_turn_response_ms_count 1" in out


def test_respond_records_turn_in_telemetry():
    from voice.telemetry import telemetry

    async def scenario():
        before = telemetry.turns_total
        transport = FakeTransport()
        session = make_session(transport)
        await session._respond("hello there")
        return before

    before = asyncio.run(scenario())
    assert telemetry.turns_total == before + 1


# ─────────────────────────────────────────────────────────────────────────
# WER (tools/stt_eval.py)
# ─────────────────────────────────────────────────────────────────────────
def test_word_error_rate():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from stt_eval import word_error_rate
    assert word_error_rate("one chicken burger", "one chicken burger") == 0.0
    assert word_error_rate("one chicken burger", "one cheese burger") == round(1 / 3, 4)
    assert word_error_rate("hello", "") == 1.0
    # punctuation/case insensitive
    assert word_error_rate("Hello, there!", "hello there") == 0.0
    # insertion counts against hypothesis
    assert word_error_rate("hello there", "well hello there") == 0.5


def test_barge_in_during_buffered_playback():
    """Respond task finished, audio still playing from the buffer:
    interruption must still clear and rewrite history."""
    async def scenario():
        from voice.stt import STTEvent
        transport = FakeTransport()
        session = make_session(transport)
        await session._respond("tell me the menu")   # completes fully
        assert session.state == "listening"
        # only the first of two sentences was actually played
        session.on_mark(transport.marks[0])
        session._maybe_barge_in(STTEvent(kind="interim", text="wait stop"))
        await asyncio.sleep(0.05)
        return transport, session

    transport, session = asyncio.run(scenario())
    assert transport.clears == 1
    assert session.metrics.turns[0].barged_in
    assistant = [m for m in session.messages if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert assistant[0]["content"].endswith("—")
    # a second interruption must not re-fire (nothing left unplayed)
    assert not session._agent_audibly_speaking()
