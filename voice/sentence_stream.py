"""
Incremental sentence chunking for token streams.

TTS latency is dominated by "when do we send the first synthesis request",
so we cut the LLM token stream into speakable chunks as early as possible:
a chunk is emitted the moment a sentence boundary appears, and a maximum
buffer forces a cut at the last clause boundary for run-on sentences.

Handles Urdu/Arabic sentence marks (۔ ؟ ،) alongside Latin punctuation so
mixed-language agents chunk correctly, and strips markdown artifacts that
LLMs emit but TTS should never pronounce.
"""

from __future__ import annotations

import re
from typing import Iterator

# Sentence-final punctuation (Latin + Urdu/Arabic full stop & question mark)
_SENT_END = ".!?؟۔"
# Clause boundaries usable as a forced-cut fallback
_CLAUSE = ",;:،"

_MD_PATTERNS = [
    (re.compile(r"[*_`#>]+"), ""),          # markdown emphasis/heading chars
    (re.compile(r"\[(.*?)\]\(.*?\)"), r"\1"),  # [text](url) -> text
    (re.compile(r"\s{2,}"), " "),
]


def clean_for_tts(text: str) -> str:
    for pattern, repl in _MD_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


class SentenceStream:
    """
    Feed tokens in with push(); it yields complete speakable chunks.
    Call flush() at end-of-stream for the trailing partial sentence.

        stream = SentenceStream()
        for token in llm_tokens:
            for sentence in stream.push(token):
                tts(sentence)
        for sentence in stream.flush():
            tts(sentence)
    """

    def __init__(self, min_chars: int = 3, max_chars: int = 240):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def push(self, token: str) -> Iterator[str]:
        self._buf += token
        while True:
            chunk = self._extract()
            if chunk is None:
                return
            yield chunk

    def flush(self) -> Iterator[str]:
        chunk = clean_for_tts(self._buf)
        self._buf = ""
        if len(chunk) >= 1:
            yield chunk

    # ── internals ────────────────────────────────────────────────────────
    def _extract(self) -> str | None:
        # Sentence boundary: sentence-final char followed by whitespace/EOL.
        # (Avoids splitting decimals like "3.99" or "Rs. 250".)
        for i, ch in enumerate(self._buf):
            if ch in _SENT_END:
                nxt = self._buf[i + 1] if i + 1 < len(self._buf) else None
                prev = self._buf[i - 1] if i > 0 else ""
                if ch == "." and prev.isdigit() and nxt is not None and nxt.isdigit():
                    continue  # decimal number
                if nxt is None:
                    # Boundary is at the very end of the buffer; wait for the
                    # next token to confirm (could be "..." or "?!").
                    return None
                if nxt.isspace():
                    candidate = clean_for_tts(self._buf[:i + 1])
                    self._buf = self._buf[i + 1:].lstrip()
                    if len(candidate) >= self.min_chars:
                        return candidate
                    # Too short to speak alone ("Ok."), keep accumulating.
                    self._buf = candidate + " " + self._buf
                    return None

        # Forced cut: buffer too long, break at the last clause boundary or
        # space that falls WITHIN the max-chars window.
        if len(self._buf) >= self.max_chars:
            window = self._buf[:self.max_chars]
            cut = -1
            for i in range(len(window) - 1, 0, -1):
                if window[i] in _CLAUSE:
                    cut = i + 1
                    break
            if cut == -1:
                cut = window.rfind(" ")
            if cut <= 0:
                cut = self.max_chars
            candidate = clean_for_tts(self._buf[:cut])
            self._buf = self._buf[cut:].lstrip()
            if candidate:
                return candidate
        return None
