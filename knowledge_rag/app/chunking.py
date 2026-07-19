"""
Simple, dependency-free sliding-window text chunker.

Menu/profile documents are short (a few hundred to a few thousand words),
so a plain character-based chunker with overlap is enough — no need for a
heavier semantic chunking library for this phase.
"""
from typing import List


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks
