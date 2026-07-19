"""
CallSession — the per-call orchestrator.

One asyncio task tree per phone call:

    run()                       consumes STT events (turn detection, barge-in)
      ├─ DeepgramLiveSTT._receiver / _keepalive
      ├─ _respond() task        LLM stream → sentences → TTS → media frames
      └─ _watchdog() task       max call duration

Turn-taking:
    Deepgram emits `final` segments; the turn closes on speech_final
    (endpointing silence) or utterance_end (word-gap fallback). Segments
    between closures are joined into one user turn.

Barge-in ("transcript" mode, default):
    While the agent is speaking, STT keeps running on the caller's audio.
    The first interim transcript with real words (not a backchannel like
    "ok"/"haan") cancels the response task, sends Twilio a `clear` frame to
    flush its ~buffered audio (without this the bot talks for seconds after
    being interrupted — Twilio buffers aggressively), and truncates the
    assistant's chat history to the sentences that were actually played
    (tracked via Twilio `mark` events), so the LLM never believes it said
    things the caller never heard.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from typing import Optional

from .config import VoiceConfig
from .llm import StreamingLLM
from .metrics import CallMetrics, TurnMetrics
from .rag import build_rag_context
from .sentence_stream import SentenceStream
from .stt import DeepgramLiveSTT, STTEvent
from .transfer import TwilioCallControl
from .tts import make_tts, make_fallback_tts

FALLBACK_LINE = "Sorry, I'm having a little trouble right now. Could you say that again?"
TRANSFER_LINE = "Sure, let me connect you with a member of our team. One moment please."
TRANSFER_FAILED_LINE = "I'm sorry, I couldn't reach a team member right now. Is there anything else I can help you with?"


class TwilioTransport:
    """Thin wrapper over the FastAPI WebSocket for outbound frames."""

    def __init__(self, websocket, stream_sid: str):
        self.websocket = websocket
        self.stream_sid = stream_sid
        self.bytes_sent = 0

    async def send_media(self, mulaw_bytes: bytes):
        self.bytes_sent += len(mulaw_bytes)
        await self.websocket.send_json({
            "event": "media",
            "streamSid": self.stream_sid,
            "media": {"payload": base64.b64encode(mulaw_bytes).decode("ascii")},
        })

    async def send_mark(self, name: str):
        await self.websocket.send_json({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": name},
        })

    async def send_clear(self):
        await self.websocket.send_json({
            "event": "clear",
            "streamSid": self.stream_sid,
        })


class CallSession:
    def __init__(self, config: VoiceConfig, transport: TwilioTransport,
                 call_sid: str = "", caller_number: str = ""):
        self.config = config
        self.transport = transport
        self.call_sid = call_sid
        self.caller_number = caller_number

        self.metrics = CallMetrics(
            call_sid=call_sid,
            stream_sid=transport.stream_sid,
            stt_provider=f"deepgram/{config.stt_model}",
            llm_model=config.llm_model,
        )
        self.messages: list[dict] = [
            {"role": "system", "content": config.system_prompt}
        ]

        self.stt_events: asyncio.Queue = asyncio.Queue()
        self.stt = DeepgramLiveSTT(config, self.stt_events)
        self.llm = StreamingLLM(config)
        self.tts = make_tts(config)
        self.tts_fallback = make_fallback_tts(config, self.tts)
        self.control = TwilioCallControl(config)
        self.metrics.tts_provider = self.tts.provider_name

        self.state = "listening"           # listening | responding | ending
        self.ended = asyncio.Event()
        self._respond_task: Optional[asyncio.Task] = None
        self._pending_finals: list[str] = []
        self._queued_user_text = ""        # turn closed while still responding
        self._turn_seq = 0
        self._mark_events: dict[str, asyncio.Event] = {}
        self._spoken_sentences: list[str] = []   # current turn, queued order
        self._played_marks = 0                   # current turn, confirmed played
        self._greeted = False

    # ── inbound hooks (called by the route) ──────────────────────────────
    async def feed_audio(self, mulaw_bytes: bytes):
        await self.stt.send_audio(mulaw_bytes)

    def on_mark(self, name: str):
        event = self._mark_events.get(name)
        if event:
            event.set()
        if name.startswith(f"t{self._turn_seq}:") or name.startswith("greet:"):
            self._played_marks += 1

    # ── lifecycle ────────────────────────────────────────────────────────
    async def run(self):
        """Main loop; returns when the call is over."""
        started = time.monotonic()
        try:
            await self.stt.start()
        except Exception:
            # One retry for transient connect failures, then give up cleanly.
            await asyncio.sleep(0.5)
            try:
                await self.stt.start()
            except Exception:
                self.metrics.disconnect_reason = "stt_connect_failed"
                self.ended.set()
                return

        watchdog = asyncio.create_task(self._watchdog())
        greeting = asyncio.create_task(self._speak_greeting(started))

        try:
            while not self.ended.is_set():
                event: STTEvent = await self.stt_events.get()
                if event.kind == "interim":
                    self._maybe_barge_in(event)
                elif event.kind == "speech_started":
                    if self.config.barge_in_mode == "vad":
                        self._maybe_barge_in(event)
                elif event.kind == "final":
                    if event.text:
                        self._maybe_barge_in(event)
                        self._pending_finals.append(event.text)
                    if event.speech_final and self._pending_finals:
                        self._close_user_turn()
                elif event.kind == "utterance_end":
                    if self._pending_finals:
                        self._close_user_turn()
                elif event.kind in ("closed", "error"):
                    if event.kind == "error":
                        print(f"[voice] STT error: {event.text}")
                    if not self.metrics.disconnect_reason:
                        self.metrics.disconnect_reason = event.kind
                    break
        finally:
            watchdog.cancel()
            greeting.cancel()
            await self.shutdown()

    async def shutdown(self):
        self.ended.set()
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        await self.stt.finish()
        try:
            await self.tts.close()
            if self.tts_fallback:
                await self.tts_fallback.close()
        except Exception:
            pass

    async def _watchdog(self):
        await asyncio.sleep(self.config.max_call_seconds)
        self.metrics.disconnect_reason = "max_duration"
        if self.call_sid:
            await self.control.hangup(self.call_sid)
        self.ended.set()

    # ── turn management ──────────────────────────────────────────────────
    def _close_user_turn(self):
        user_text = " ".join(self._pending_finals).strip()
        self._pending_finals.clear()
        if not user_text:
            return
        if self.state == "responding":
            # Barge-in disabled (or filtered) and the caller kept talking:
            # queue the text; it merges into the next turn.
            self._queued_user_text = (self._queued_user_text + " " + user_text).strip()
            return
        if self._queued_user_text:
            user_text = (self._queued_user_text + " " + user_text).strip()
            self._queued_user_text = ""
        self._respond_task = asyncio.create_task(self._respond(user_text))

    def _maybe_barge_in(self, event: STTEvent):
        if (self.state != "responding"
                or not self.config.barge_in_enabled):
            return
        if event.kind in ("interim", "final"):
            text = re.sub(r"[^\w\s]", "", event.text).strip().lower()
            if len(text) < self.config.barge_in_min_chars:
                return
            if text in self.config.barge_in_ignore:
                return
        asyncio.create_task(self._barge_in())

    async def _barge_in(self):
        if self.state != "responding":
            return
        self.state = "listening"
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        # Snapshot BEFORE clearing: Twilio may flush queued marks on clear,
        # which must not count as "the caller heard this".
        played = self._spoken_sentences[:self._played_marks]
        try:
            await self.transport.send_clear()
        except Exception:
            pass
        if self.metrics.turns:
            turn = self.metrics.turns[-1]
            turn.barged_in = True
            turn.agent_text = " ".join(played)
        if played:
            self.messages.append({
                "role": "assistant",
                "content": " ".join(played) + " —",  # em-dash: cut off mid-reply
            })
        print(f"[voice] barge-in: cleared playback after "
              f"{self._played_marks}/{len(self._spoken_sentences)} sentences")

    # ── speaking ─────────────────────────────────────────────────────────
    async def _speak_greeting(self, session_start: float):
        if self._greeted:
            return
        self._greeted = True
        try:
            first = await self._speak_sentences(
                [self.config.greeting], mark_prefix="greet", turn=None)
            if first is not None:
                self.metrics.greeting_first_audio_ms = round(
                    (first - session_start) * 1000.0, 1)
            self.messages.append(
                {"role": "assistant", "content": self.config.greeting})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[voice] greeting failed: {e}")

    async def _synthesize(self, sentence: str, turn: Optional[TurnMetrics]):
        """Yield audio chunks, falling back to the secondary TTS provider."""
        providers = [self.tts] + ([self.tts_fallback] if self.tts_fallback else [])
        last_error = None
        for provider in providers:
            try:
                got_audio = False
                async for chunk in provider.synthesize(sentence):
                    if not got_audio:
                        got_audio = True
                        if turn:
                            turn.mark_tts_first_byte()
                    yield chunk
                if got_audio:
                    return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                print(f"[voice] TTS {provider.provider_name} failed: {e}")
        if last_error:
            raise last_error

    async def _speak_sentences(self, sentences, mark_prefix: str,
                               turn: Optional[TurnMetrics]) -> Optional[float]:
        """Speak sentences in order; returns monotonic time of first frame."""
        first_frame_at = None
        for idx, sentence in enumerate(sentences):
            mark_name = f"{mark_prefix}:{idx}"
            self._mark_events[mark_name] = asyncio.Event()
            self._spoken_sentences.append(sentence)
            async for chunk in self._synthesize(sentence, turn):
                if first_frame_at is None:
                    first_frame_at = time.monotonic()
                    if turn:
                        turn.mark_first_audio_sent()
                await self.transport.send_media(chunk)
            await self.transport.send_mark(mark_name)
        return first_frame_at

    async def _wait_for_mark(self, name: str, timeout: float):
        event = self._mark_events.get(name)
        if not event:
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

    # ── the response turn ────────────────────────────────────────────────
    async def _respond(self, user_text: str):
        self.state = "responding"
        self._turn_seq += 1
        seq = self._turn_seq
        self._spoken_sentences = []
        self._played_marks = 0

        turn = self.metrics.new_turn(user_text, self.config.endpointing_ms)
        print(f"[voice] turn {turn.turn_index} user: {user_text!r}")

        try:
            context = await build_rag_context(self.config, user_text)
            request_messages = self.messages + [
                {"role": "user", "content": user_text + context}
            ]
            self.messages.append({"role": "user", "content": user_text})

            chunker = SentenceStream()
            spoken: list[str] = []
            full_reply: list[str] = []
            tool_calls: list = []
            sentence_idx = 0

            async def speak(sentence: str):
                nonlocal sentence_idx
                mark_name = f"t{seq}:{sentence_idx}"
                self._mark_events[mark_name] = asyncio.Event()
                self._spoken_sentences.append(sentence)
                async for chunk in self._synthesize(sentence, turn):
                    if turn.t_first_audio_sent is None:
                        turn.mark_first_audio_sent()
                    await self.transport.send_media(chunk)
                await self.transport.send_mark(mark_name)
                spoken.append(sentence)
                turn.sentences += 1
                sentence_idx += 1

            async for event in self.llm.stream_reply(request_messages):
                if event.kind == "token":
                    turn.mark_llm_first_token()
                    full_reply.append(event.text)
                    for sentence in chunker.push(event.text):
                        await speak(sentence)
                elif event.kind == "tool_call":
                    tool_calls.append(event)
                elif event.kind == "error":
                    turn.error = f"llm: {event.text}"
                    print(f"[voice] LLM error: {event.text}")
            turn.mark_llm_done()

            if turn.error and not full_reply and not tool_calls:
                await speak(FALLBACK_LINE)
                full_reply = [FALLBACK_LINE]

            for sentence in chunker.flush():
                await speak(sentence)

            reply_text = "".join(full_reply).strip()
            if reply_text:
                self.messages.append({"role": "assistant", "content": reply_text})
            turn.agent_text = " ".join(spoken)

            for call in tool_calls:
                turn.tool_call = call.tool_name
                if call.tool_name == "transfer_call":
                    await self._do_transfer(call.tool_args, seq)
                elif call.tool_name == "end_call":
                    await self._do_end_call(call.tool_args, seq)

            self.state = "listening"
        except asyncio.CancelledError:
            # Barge-in path; _barge_in() owns state and history cleanup.
            raise
        except Exception as e:
            turn.error = turn.error or str(e)
            print(f"[voice] respond failed: {e}")
            try:
                await self._speak_sentences([FALLBACK_LINE],
                                            mark_prefix=f"t{seq}f", turn=turn)
            except Exception:
                pass
            self.state = "listening"

    # ── call control ─────────────────────────────────────────────────────
    async def _do_transfer(self, args: dict, seq: int):
        target = self.config.transfer_number
        summary = args.get("summary", "")
        print(f"[voice] transfer requested: {args.get('reason', '')!r}")
        if not target or not self.control.enabled or not self.call_sid:
            await self._speak_sentences([TRANSFER_FAILED_LINE],
                                        mark_prefix=f"t{seq}x", turn=None)
            self.messages.append(
                {"role": "assistant", "content": TRANSFER_FAILED_LINE})
            return
        self.state = "ending"
        await self._speak_sentences([TRANSFER_LINE],
                                    mark_prefix=f"t{seq}tr", turn=None)
        await self._wait_for_mark(f"t{seq}tr:0", timeout=10.0)
        ok = await self.control.transfer(self.call_sid, target, summary)
        if ok:
            self.metrics.disconnect_reason = "transferred"
            # Twilio redirects the call; the media stream will stop shortly.
        else:
            self.state = "listening"
            await self._speak_sentences([TRANSFER_FAILED_LINE],
                                        mark_prefix=f"t{seq}xx", turn=None)

    async def _do_end_call(self, args: dict, seq: int):
        farewell = args.get("farewell") or "Thank you for calling. Goodbye!"
        self.state = "ending"
        await self._speak_sentences([farewell], mark_prefix=f"t{seq}bye", turn=None)
        self.messages.append({"role": "assistant", "content": farewell})
        await self._wait_for_mark(f"t{seq}bye:0", timeout=15.0)
        self.metrics.disconnect_reason = "agent_hangup"
        if self.call_sid and self.control.enabled:
            await self.control.hangup(self.call_sid)
        self.ended.set()
