"""
STT accuracy evaluation: word error rate against a reference set.

Two ways to feed it:

  1. Real recordings (the honest benchmark — accents, phone-band audio):
       python tools/stt_eval.py --pairs evals/stt_set.jsonl
     where each JSONL line is {"wav": "path.wav", "text": "reference"}.

  2. Synthetic closed loop (no recordings needed; checks the recognition
     chain and keyterm boosting, NOT accent robustness — say so when
     reporting numbers):
       python tools/stt_eval.py --synth "one chicken burger please" \
           "do you deliver to gulberg"

Audio is downsampled to 8 kHz mu-law first, so results reflect
telephone-band quality — the same audio the production pipeline sees,
not studio-quality wideband.

Options mirror the production STT config: --model, --language,
--keyterms "term1,term2" to measure how much keyterm prompting helps
(run once with and once without and compare).

Output: per-utterance WER + aggregate, JSON report via --report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import wave
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from voice.audio import mulaw_to_pcm16, pcm16_to_mulaw, resample_pcm16

DEEPGRAM_LISTEN = "https://api.deepgram.com/v1/listen"
DEEPGRAM_SPEAK = "https://api.deepgram.com/v1/speak"


# ─────────────────────────────────────────────────────────────────────────
# WER
# ─────────────────────────────────────────────────────────────────────────
def normalize(text: str) -> list[str]:
    import re
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return text.split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words / reference length."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, 1):
            cost = 0 if ref_word == hyp_word else 1
            cur[j] = min(prev[j] + 1,        # deletion
                         cur[j - 1] + 1,     # insertion
                         prev[j - 1] + cost) # substitution
        prev = cur
    return round(prev[-1] / len(ref), 4)


# ─────────────────────────────────────────────────────────────────────────
# audio plumbing
# ─────────────────────────────────────────────────────────────────────────
def wav_to_telephony_mulaw(path: str) -> bytes:
    with wave.open(path, "rb") as wav:
        assert wav.getsampwidth() == 2, f"{path}: need 16-bit PCM"
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if wav.getnchannels() == 2:
            pcm = pcm[::2]
    return pcm16_to_mulaw(resample_pcm16(pcm, rate, 8000))


async def synthesize(text: str, api_key: str) -> bytes:
    params = {"model": "aura-2-orion-en", "encoding": "mulaw",
              "sample_rate": "8000", "container": "none"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(DEEPGRAM_SPEAK, params=params,
                                 headers={"Authorization": f"Token {api_key}"},
                                 json={"text": text})
        resp.raise_for_status()
        return resp.content


async def transcribe(mulaw: bytes, api_key: str, model: str,
                     language: str, keyterms: list[str]) -> str:
    params = [("model", model), ("smart_format", "false"),
              ("punctuate", "false"), ("encoding", "mulaw"),
              ("sample_rate", "8000")]
    if language:
        params.append(("language", language))
    for term in keyterms:
        params.append(("keyterm", term))
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            DEEPGRAM_LISTEN, params=params,
            headers={"Authorization": f"Token {api_key}",
                     "Content-Type": "audio/mulaw"},
            content=mulaw)
        resp.raise_for_status()
        alts = resp.json()["results"]["channels"][0]["alternatives"]
        return alts[0]["transcript"] if alts else ""


# ─────────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="WER eval against Deepgram")
    parser.add_argument("--pairs", help="JSONL of {wav, text} reference pairs")
    parser.add_argument("--synth", nargs="+",
                        help="reference sentences; audio synthesized via Aura")
    parser.add_argument("--model", default=os.getenv("STT_MODEL", "nova-3"))
    parser.add_argument("--language", default="")
    parser.add_argument("--keyterms", default="",
                        help="comma-separated boost terms")
    parser.add_argument("--report", help="write JSON report here")
    args = parser.parse_args()

    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        print("DEEPGRAM_API_KEY required.")
        sys.exit(1)

    cases: list[tuple[str, bytes]] = []  # (reference, mulaw audio)
    if args.pairs:
        for line in Path(args.pairs).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append((row["text"], wav_to_telephony_mulaw(row["wav"])))
        source = "recordings"
    elif args.synth:
        for text in args.synth:
            cases.append((text, await synthesize(text, api_key)))
        source = "synthetic (Aura voice — measures chain, not accents)"
    else:
        parser.error("provide --pairs or --synth")

    keyterms = [t.strip() for t in args.keyterms.split(",") if t.strip()]
    results = []
    for reference, audio in cases:
        hypothesis = await transcribe(audio, api_key, args.model,
                                      args.language, keyterms)
        wer = word_error_rate(reference, hypothesis)
        results.append({"reference": reference, "hypothesis": hypothesis,
                        "wer": wer})
        flag = "  <-- check" if wer > 0.15 else ""
        print(f"WER {wer:5.2f}  ref: {reference!r}")
        print(f"           hyp: {hypothesis!r}{flag}")

    aggregate = round(sum(r["wer"] for r in results) / len(results), 4)
    print("-" * 60)
    print(f"utterances: {len(results)}   mean WER: {aggregate}   "
          f"model: {args.model}   keyterms: {len(keyterms)}   source: {source}")

    if args.report:
        Path(args.report).write_text(json.dumps({
            "model": args.model, "language": args.language,
            "keyterms": keyterms, "source": source,
            "mean_wer": aggregate, "utterances": results,
        }, indent=2, ensure_ascii=False))
        print(f"report written to {args.report}")


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    asyncio.run(main())
