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
import logging
import re
import time
from typing import Optional

from .config import VoiceConfig
from .llm import StreamingLLM
from .metrics import CallMetrics, TurnMetrics
from .rag import build_rag_context
from .sentence_stream import SentenceStream
from .stt import DeepgramLiveSTT, STTEvent
from .telemetry import telemetry
from .transfer import TwilioCallControl
from .tts import make_tts, make_fallback_tts

log = logging.getLogger("voice.pipeline")

FALLBACK_LINE = "Sorry, I'm having a little trouble right now. Could you say that again?"
TRANSFER_LINE = "Sure, let me connect you with a member of our team. One moment please."
TRANSFER_FAILED_LINE = "I'm sorry, I couldn't reach a team member right now. Is there anything else I can help you with?"
STT_DOWN_LINE = "I'm sorry, we're having technical difficulties right now. Please call back in a few minutes."


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
        self._make_stt = lambda: DeepgramLiveSTT(config, self.stt_events)
        self.stt = self._make_stt()
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
        self._turn_marks: set[str] = set()       # mark names of the current reply
        self._greeted = False
        self._greeting_task: Optional[asyncio.Task] = None
        self._stt_reconnects = 0

    # ── inbound hooks (called by the route) ──────────────────────────────
    async def feed_audio(self, mulaw_bytes: bytes):
        await self.stt.send_audio(mulaw_bytes)

    def on_mark(self, name: str):
        # Popping keeps the dict bounded on long calls; a waiter that
        # already grabbed the Event still gets woken by set().
        event = self._mark_events.pop(name, None)
        if event:
            event.set()
        # Count playback only for marks belonging to the reply currently
        # tracked in _spoken_sentences. Matching by exact name (not prefix)
        # covers fallback/transfer/goodbye lines and ignores marks Twilio
        # echoes late for a reply we already reset.
        if name in self._turn_marks:
            self._turn_marks.discard(name)
            self._played_marks += 1

    # ── lifecycle ────────────────────────────────────────────────────────
    async def run(self):
        """Main loop; returns when the call is over."""
        started = time.monotonic()
        telemetry.call_started()
        # Greeting synthesis and the STT socket connect are independent
        # cold starts — run them concurrently so the caller hears audio
        # as early as possible.
        watchdog = asyncio.create_task(self._watchdog())
        if hasattr(self.tts, "prewarm"):
            asyncio.create_task(self.tts.prewarm())
        greeting = asyncio.create_task(self._speak_greeting(started))
        self._greeting_task = greeting
        try:
            await self.stt.start()
        except Exception:
            # One retry for transient connect failures, then give up cleanly.
            await asyncio.sleep(0.5)
            try:
                self.stt = self._make_stt()
                await self.stt.start()
            except Exception:
                self.metrics.disconnect_reason = "stt_connect_failed"
                telemetry.provider_error("stt")
                greeting.cancel()
                watchdog.cancel()
                # Best effort: tell the caller instead of dead air, and end
                # the call deterministically rather than waiting for Twilio.
                try:
                    await asyncio.wait_for(self._speak_sentences(
                        [STT_DOWN_LINE], mark_prefix="err", turn=None), timeout=5.0)
                    await self._wait_for_mark("err:0", timeout=8.0)
                except Exception:
                    pass
                if self.call_sid and self.control.enabled:
                    await self.control.hangup(self.call_sid)
                self.ended.set()
                return
        self.metrics.stt_connect_ms = round(
            (time.monotonic() - started) * 1000.0, 1)

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
                        log.warning("STT error: %s", event.text)
                    if (not self.ended.is_set() and self.state != "ending"
                            and await self._reconnect_stt()):
                        continue
                    if not self.metrics.disconnect_reason:
                        self.metrics.disconnect_reason = event.kind
                    break
        finally:
            watchdog.cancel()
            greeting.cancel()
            await self.shutdown()

    async def _reconnect_stt(self) -> bool:
        """
        Survive a mid-call Deepgram drop: a transient WS failure used to
        end the whole call; instead we swap in a fresh STT stream (audio
        during the gap is lost — the caller repeats a word at worst).
        """
        while self._stt_reconnects < self.config.stt_max_reconnects:
            self._stt_reconnects += 1
            self.metrics.stt_reconnects = self._stt_reconnects
            try:
                await self.stt.finish()
            except Exception:
                pass
            # Drop events queued by the dead socket (its trailing "closed",
            # stale partials) so they can't trigger another reconnect.
            while not self.stt_events.empty():
                try:
                    self.stt_events.get_nowait()
                except asyncio.QueueEmpty:
                    break
            self.stt = self._make_stt()
            try:
                await self.stt.start()
                log.info("STT reconnected (attempt %d)", self._stt_reconnects)
                return True
            except Exception as e:
                telemetry.provider_error("stt")
                log.warning("STT reconnect attempt %d failed: %s",
                            self._stt_reconnects, e)
                await asyncio.sleep(0.3)
        return False

    async def shutdown(self):
        # shutdown() can be invoked from both run() and the route's finally.
        if not getattr(self, "_shutdown_done", False):
            self._shutdown_done = True
            telemetry.call_ended()
        self.ended.set()
        if self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        if self.stt is not None:
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
        if self.state in ("responding", "ending"):
            # Responding: barge-in disabled (or filtered) and the caller
            # kept talking — queue the text; it merges into the next turn.
            # Ending: a transfer/goodbye is in flight — starting a second
            # respond task here would interleave two replies.
            self._queued_user_text = (self._queued_user_text + " " + user_text).strip()
            return
        if self._queued_user_text:
            user_text = (self._queued_user_text + " " + user_text).strip()
            self._queued_user_text = ""
        self._respond_task = asyncio.create_task(self._respond(user_text))

    def _agent_audibly_speaking(self) -> bool:
        """
        True while the caller can still hear the agent. The respond task
        often finishes seconds before playback does — Twilio buffers the
        whole reply — so "speaking" means unplayed sentences are queued
        (marks not yet echoed), not that a task is running.
        """
        return (self.state == "responding"
                or self._played_marks < len(self._spoken_sentences))

    def _maybe_barge_in(self, event: STTEvent):
        if self.state == "ending":
            # Transfer/goodbye in flight: interrupting now would clear the
            # farewell audio and the call would end in silence anyway.
            return
        if not self.config.barge_in_enabled or not self._agent_audibly_speaking():
            return
        if event.kind in ("interim", "final"):
            text = re.sub(r"[^\w\s]", "", event.text).strip().lower()
            if len(text) < self.config.barge_in_min_chars:
                return
            if text in self.config.barge_in_ignore:
                return
        asyncio.create_task(self._barge_in())

    async def _barge_in(self):
        mid_response = self.state == "responding"
        if not self._agent_audibly_speaking():
            return
        self.state = "listening"
        if mid_response and self._respond_task and not self._respond_task.done():
            self._respond_task.cancel()
        if self._greeting_task and not self._greeting_task.done():
            self._greeting_task.cancel()
        # Snapshot BEFORE clearing: Twilio may flush queued marks on clear,
        # which must not count as "the caller heard this". Turn flags are
        # set before the first await so the cancelled respond task's
        # cleanup never records the turn without them.
        played = self._spoken_sentences[:self._played_marks]
        queued_count = len(self._spoken_sentences)
        # Stop counting this reply as "audibly speaking" from now on.
        self._spoken_sentences = played
        self._played_marks = len(played)
        if self.metrics.turns:
            turn = self.metrics.turns[-1]
            turn.barged_in = True
            turn.agent_text = " ".join(played)
        truncated = (" ".join(played) + " —") if played else ""
        if mid_response:
            # Task cancelled before it appended its reply to history.
            if truncated:
                self.messages.append({"role": "assistant", "content": truncated})
        else:
            # Reply was fully generated and appended, but the caller only
            # heard part of it: rewrite history to what was actually heard.
            for message in reversed(self.messages):
                if message["role"] == "assistant":
                    message["content"] = truncated or message["content"]
                    break
        try:
            await self.transport.send_clear()
        except Exception:
            pass
        log.info("barge-in (%s): cleared after %d/%d sentences",
                 "mid-response" if mid_response else "during playback",
                 len(played), queued_count)

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
            log.warning("greeting failed: %s", e)

    async def _synthesize(self, sentence: str, turn: Optional[TurnMetrics]):
        """Yield audio chunks, falling back to the secondary TTS provider."""
        providers = [self.tts] + ([self.tts_fallback] if self.tts_fallback else [])
        last_error = None
        for provider in providers:
            got_audio = False
            try:
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
                log.warning("TTS %s failed: %s", provider.provider_name, e)
                if got_audio:
                    # The caller already heard part of this sentence;
                    # re-synthesizing it on the fallback would replay it.
                    # Truncate this sentence and move on.
                    telemetry.provider_error("tts")
                    return
                telemetry.tts_fallback()
        if last_error:
            telemetry.provider_error("tts")
            raise last_error

    async def _speak_sentences(self, sentences, mark_prefix: str,
                               turn: Optional[TurnMetrics]) -> Optional[float]:
        """Speak sentences in order; returns monotonic time of first frame."""
        first_frame_at = None
        for idx, sentence in enumerate(sentences):
            mark_name = f"{mark_prefix}:{idx}"
            self._mark_events[mark_name] = asyncio.Event()
            self._turn_marks.add(mark_name)
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
        self._turn_marks = set()

        # If the greeting is still being synthesized (caller spoke over
        # it with barge-in filtered), let it finish sending first so two
        # replies never interleave frames on the same stream.
        if self._greeting_task and not self._greeting_task.done():
            await asyncio.wait({self._greeting_task}, timeout=10.0)

        turn = self.metrics.new_turn(user_text, self.config.endpointing_ms)
        log.info("turn %d user: %r", turn.turn_index, user_text)

        completed = False
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
                self._turn_marks.add(mark_name)
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
                    log.warning("LLM error: %s", event.text)
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

            if self.state != "ending":   # keep a transfer/hangup in flight
                self.state = "listening"
            completed = True
        except asyncio.CancelledError:
            # Barge-in path; _barge_in() owns state and history cleanup.
            raise
        except Exception as e:
            turn.error = turn.error or str(e)
            log.exception("respond failed: %s", e)
            try:
                await self._speak_sentences([FALLBACK_LINE],
                                            mark_prefix=f"t{seq}f", turn=turn)
            except Exception:
                pass
            if self.state != "ending":
                self.state = "listening"
            completed = True
        finally:
            telemetry.record_turn(turn)
            # A turn that closed while we were speaking (barge-in filtered
            # or disabled) would otherwise sit queued forever: answer it
            # now. Skipped on cancellation — the caller is still talking
            # and their next endpoint will merge the queued text — and when
            # the call is ending (transfer/goodbye already spoken).
            if (completed and self._queued_user_text
                    and not self.ended.is_set() and self.state != "ending"):
                queued = self._queued_user_text
                self._queued_user_text = ""
                self._respond_task = asyncio.create_task(self._respond(queued))

    # ── call control ─────────────────────────────────────────────────────
    async def _do_transfer(self, args: dict, seq: int):
        target = self.config.transfer_number
        summary = args.get("summary", "")
        log.info("transfer requested: %r", args.get("reason", ""))
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
