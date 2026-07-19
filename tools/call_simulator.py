"""
Call simulator + latency/eval harness.

Emulates a Twilio Media Stream client against a running server, so the
entire voice pipeline can be exercised and measured WITHOUT a phone,
a Twilio account, or a public URL:

  - streams caller audio as real-time 20 ms mu-law frames (with continuous
    silence between utterances, like a real phone line — endpointing and
    VAD behave exactly as in production)
  - emulates Twilio's playback buffer: `mark` events are echoed back when
    the virtual playhead reaches them; `clear` flushes the buffer
  - measures per-turn response latency (last caller frame → first bot
    audio frame) and prints/saves a report
  - barge-in test: interrupts the bot mid-reply and verifies a `clear`
    frame arrives + measures reaction time
  - concurrency test: N simultaneous calls, aggregate p50/p95

Caller speech comes from WAV files (--wav) or is synthesized with
Deepgram Aura TTS (--say, needs DEEPGRAM_API_KEY) — so a full closed-loop
eval needs zero recordings.

Examples:
  python tools/call_simulator.py --say "Hi, do you deliver to DHA phase five?" \
      "How much is a chicken burger?" "Okay that's all, thank you."
  python tools/call_simulator.py --say "Tell me the whole menu please" \
      --barge-in-after 1.5
  python tools/call_simulator.py --say "Hi there" --calls 10
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice.audio import (FRAME_BYTES_20MS, mulaw_to_pcm16, pcm16_to_mulaw,
                         resample_pcm16)

SILENCE_FRAME = b"\xff" * FRAME_BYTES_20MS  # mu-law digital silence
FRAME_SECONDS = 0.02
OUT_DIR = Path("sim_output")


# ─────────────────────────────────────────────────────────────────────────
# Caller-audio sources
# ─────────────────────────────────────────────────────────────────────────
def wav_to_mulaw(path: str) -> bytes:
    with wave.open(path, "rb") as wav:
        assert wav.getsampwidth() == 2, f"{path}: need 16-bit PCM wav"
        rate = wav.getframerate()
        pcm_bytes = wav.readframes(wav.getnframes())
    import numpy as np
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    if wav.getnchannels() == 2:
        pcm = pcm[::2]
    pcm = resample_pcm16(pcm, rate, 8000)
    return pcm16_to_mulaw(pcm)


async def synthesize_caller_line(text: str, api_key: str,
                                 cache_dir: Path) -> bytes:
    """Generate caller speech with Deepgram Aura (cached by text hash)."""
    import hashlib
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (hashlib.sha1(text.encode()).hexdigest() + ".ulaw")
    if cached.exists():
        return cached.read_bytes()
    params = {"model": "aura-2-orion-en", "encoding": "mulaw",
              "sample_rate": "8000", "container": "none"}
    headers = {"Authorization": f"Token {api_key}"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post("https://api.deepgram.com/v1/speak",
                                 params=params, headers=headers,
                                 json={"text": text})
        resp.raise_for_status()
        audio = resp.content
    cached.write_bytes(audio)
    return audio


# ─────────────────────────────────────────────────────────────────────────
# One simulated call
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class TurnResult:
    text: str
    response_ms: float | None = None
    bot_audio_ms: float = 0.0


@dataclass
class CallResult:
    call_index: int
    turns: list = field(default_factory=list)
    barge_in_clear_ms: float | None = None
    error: str = ""


class SimulatedCall:
    def __init__(self, url: str, utterances: list[bytes],
                 utterance_texts: list[str], call_index: int = 0,
                 barge_in_after: float | None = None,
                 save_audio: bool = True):
        self.url = url
        self.utterances = utterances
        self.texts = utterance_texts
        self.call_index = call_index
        self.barge_in_after = barge_in_after
        self.save_audio = save_audio

        self.stream_sid = f"SIM{call_index:04d}{int(time.time())}"
        self.result = CallResult(call_index=call_index)

        self._speaking = False           # we are sending caller speech
        self._last_user_frame_at = None
        self._first_bot_audio_at = None  # since last user turn end
        self._bot_audio_started = asyncio.Event()
        self._got_clear = asyncio.Event()
        self._clear_requested_at = None
        self._received_mulaw = bytearray()
        self._playback_end = 0.0         # virtual playhead (monotonic)
        self._pending_marks: list[tuple[float, str]] = []
        self._bot_last_media_at = 0.0

    # ── receive side ─────────────────────────────────────────────────────
    async def _receiver(self, ws):
        async for raw in ws:
            msg = json.loads(raw)
            event = msg.get("event")
            now = time.monotonic()
            if event == "media":
                payload = base64.b64decode(msg["media"]["payload"])
                self._received_mulaw.extend(payload)
                self._bot_last_media_at = now
                duration = len(payload) / 8000.0
                self._playback_end = max(self._playback_end, now) + duration
                if not self._bot_audio_started.is_set():
                    self._first_bot_audio_at = now
                    self._bot_audio_started.set()
            elif event == "mark":
                name = msg.get("mark", {}).get("name", "")
                self._pending_marks.append((self._playback_end, name))
            elif event == "clear":
                if self._clear_requested_at is not None:
                    self.result.barge_in_clear_ms = round(
                        (now - self._clear_requested_at) * 1000.0, 1)
                self._playback_end = now
                self._pending_marks.clear()   # cleared marks never played
                self._got_clear.set()

    async def _mark_echoer(self, ws):
        """Echo marks back once the virtual playhead passes them."""
        while True:
            await asyncio.sleep(0.05)
            now = time.monotonic()
            due = [m for m in self._pending_marks if m[0] <= now]
            self._pending_marks = [m for m in self._pending_marks if m[0] > now]
            for _, name in due:
                await ws.send(json.dumps({
                    "event": "mark",
                    "streamSid": self.stream_sid,
                    "mark": {"name": name},
                }))

    # ── send side ────────────────────────────────────────────────────────
    async def _send_frames(self, ws, mulaw: bytes):
        """Send an utterance as paced 20 ms frames."""
        self._speaking = True
        start = time.monotonic()
        for i in range(0, len(mulaw), FRAME_BYTES_20MS):
            frame = mulaw[i:i + FRAME_BYTES_20MS]
            await ws.send(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": base64.b64encode(frame).decode()},
            }))
            # pace to real time
            target = start + (i // FRAME_BYTES_20MS + 1) * FRAME_SECONDS
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        self._last_user_frame_at = time.monotonic()
        self._speaking = False

    async def _silence_keeper(self, ws):
        """Continuous silence when not speaking — like a live phone line."""
        while True:
            if not self._speaking:
                await ws.send(json.dumps({
                    "event": "media",
                    "streamSid": self.stream_sid,
                    "media": {"payload": base64.b64encode(SILENCE_FRAME).decode()},
                }))
            await asyncio.sleep(FRAME_SECONDS)

    async def _wait_bot_quiet(self, quiet_s: float = 1.2, timeout: float = 30.0):
        """Wait until the bot stops sending audio (turn finished)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            if (self._bot_audio_started.is_set()
                    and time.monotonic() - self._bot_last_media_at > quiet_s
                    and time.monotonic() > self._playback_end):
                return True
        return False

    # ── the scenario ─────────────────────────────────────────────────────
    async def run(self) -> CallResult:
        try:
            async with websockets.connect(self.url, open_timeout=10) as ws:
                await ws.send(json.dumps({"event": "connected",
                                          "protocol": "Call", "version": "1.0.0"}))
                await ws.send(json.dumps({
                    "event": "start", "sequenceNumber": "1",
                    "start": {
                        "accountSid": "SIMULATED",
                        "streamSid": self.stream_sid,
                        "callSid": f"CA_SIM_{self.call_index}",
                        "tracks": ["inbound"],
                        "mediaFormat": {"encoding": "audio/x-mulaw",
                                        "sampleRate": 8000, "channels": 1},
                    },
                }))

                receiver = asyncio.create_task(self._receiver(ws))
                echoer = asyncio.create_task(self._mark_echoer(ws))
                silence = asyncio.create_task(self._silence_keeper(ws))

                try:
                    # Let the greeting play out first.
                    await self._wait_bot_quiet(timeout=20.0)

                    for i, utterance in enumerate(self.utterances):
                        self._bot_audio_started.clear()
                        self._first_bot_audio_at = None

                        await self._send_frames(ws, utterance)
                        turn = TurnResult(text=self.texts[i])

                        if self.barge_in_after is not None and i == len(self.utterances) - 1:
                            # Wait for the bot to start talking, let it run,
                            # then interrupt with the barge-in line.
                            try:
                                await asyncio.wait_for(
                                    self._bot_audio_started.wait(), timeout=15)
                            except asyncio.TimeoutError:
                                pass
                            await asyncio.sleep(self.barge_in_after)
                            self._clear_requested_at = time.monotonic()
                            interrupt = await self._interrupt_audio()
                            await self._send_frames(ws, interrupt)
                            try:
                                await asyncio.wait_for(
                                    self._got_clear.wait(), timeout=10)
                            except asyncio.TimeoutError:
                                pass

                        try:
                            await asyncio.wait_for(
                                self._bot_audio_started.wait(), timeout=20)
                            turn.response_ms = round(
                                (self._first_bot_audio_at
                                 - self._last_user_frame_at) * 1000.0, 1)
                        except asyncio.TimeoutError:
                            turn.response_ms = None

                        await self._wait_bot_quiet()
                        turn.bot_audio_ms = round(
                            len(self._received_mulaw) / 8.0, 1)
                        self.result.turns.append(turn)

                    await ws.send(json.dumps({"event": "stop",
                                              "streamSid": self.stream_sid}))
                finally:
                    for task in (receiver, echoer, silence):
                        task.cancel()

        except websockets.exceptions.ConnectionClosed:
            # Server hung up (agent end_call / transfer / STT failure) —
            # a normal way for a call to end; keep whatever we measured.
            pass
        except Exception as e:
            self.result.error = f"{type(e).__name__}: {e}"

        if self.save_audio and self._received_mulaw:
            OUT_DIR.mkdir(exist_ok=True)
            out = OUT_DIR / f"call{self.call_index}_bot.wav"
            pcm = mulaw_to_pcm16(bytes(self._received_mulaw))
            with wave.open(str(out), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(pcm.tobytes())
        return self.result

    async def _interrupt_audio(self) -> bytes:
        key = os.getenv("DEEPGRAM_API_KEY", "")
        if key:
            return await synthesize_caller_line(
                "Wait, wait, stop — I have a question.",
                key, OUT_DIR / "cache")
        # keyless fallback: loud 400 Hz tone (VAD barge-in mode only)
        import numpy as np
        t = np.arange(0, 1.0, 1 / 8000.0)
        tone = (np.sin(2 * np.pi * 400 * t) * 20000).astype(np.int16)
        return pcm16_to_mulaw(tone)


# ─────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────
def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 1)


def print_report(results: list[CallResult], barge_in: bool):
    print("\n" + "=" * 64)
    print("CALL SIMULATION REPORT")
    print("=" * 64)
    all_latencies = []
    for res in results:
        label = f"call {res.call_index}"
        if res.error:
            print(f"{label}: ERROR {res.error}")
            continue
        for i, turn in enumerate(res.turns):
            lat = f"{turn.response_ms} ms" if turn.response_ms is not None else "NO RESPONSE"
            print(f"{label} turn {i + 1}: {lat:>12}  ({turn.text[:48]!r})")
            if turn.response_ms is not None:
                all_latencies.append(turn.response_ms)
        if barge_in:
            if res.barge_in_clear_ms is not None:
                print(f"{label} barge-in: clear received after "
                      f"{res.barge_in_clear_ms} ms  ✓")
            else:
                print(f"{label} barge-in: NO clear received  ✗")
    if all_latencies:
        print("-" * 64)
        print(f"turns: {len(all_latencies)}   "
              f"p50: {percentile(all_latencies, 50)} ms   "
              f"p95: {percentile(all_latencies, 95)} ms   "
              f"max: {max(all_latencies)} ms")
    print("=" * 64)
    return {
        "turns": len(all_latencies),
        "p50_ms": percentile(all_latencies, 50),
        "p95_ms": percentile(all_latencies, 95),
        "calls": [
            {"call": r.call_index, "error": r.error,
             "barge_in_clear_ms": r.barge_in_clear_ms,
             "turns": [t.__dict__ for t in r.turns]}
            for r in results
        ],
    }


async def main():
    parser = argparse.ArgumentParser(description="Twilio media-stream call simulator")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/twilio-media-stream"
                        "?caller_number=%2B15550100&to_number=%2B15550199")
    parser.add_argument("--say", nargs="+", help="caller lines (Deepgram TTS)")
    parser.add_argument("--wav", nargs="+", help="caller lines as wav files")
    parser.add_argument("--calls", type=int, default=1, help="concurrent calls")
    parser.add_argument("--barge-in-after", type=float, default=None,
                        help="interrupt the bot N seconds into its last reply")
    parser.add_argument("--report", default=None, help="write JSON report here")
    args = parser.parse_args()

    utterances: list[bytes] = []
    texts: list[str] = []
    if args.wav:
        for path in args.wav:
            utterances.append(wav_to_mulaw(path))
            texts.append(Path(path).name)
    elif args.say:
        key = os.getenv("DEEPGRAM_API_KEY", "")
        if not key:
            print("--say needs DEEPGRAM_API_KEY (caller speech is synthesized); "
                  "use --wav otherwise.")
            sys.exit(1)
        for line in args.say:
            utterances.append(await synthesize_caller_line(
                line, key, OUT_DIR / "cache"))
            texts.append(line)
    else:
        parser.error("provide --say or --wav")

    calls = [
        SimulatedCall(args.url, utterances, texts, call_index=i,
                      barge_in_after=args.barge_in_after,
                      save_audio=(args.calls == 1))
        for i in range(args.calls)
    ]
    results = await asyncio.gather(*(c.run() for c in calls))
    report = print_report(list(results), barge_in=args.barge_in_after is not None)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"report written to {args.report}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    asyncio.run(main())
