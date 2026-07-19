"""
Pure-python/numpy mu-law codec and framing helpers.

The streaming pipeline itself never transcodes (Twilio, Deepgram and the
TTS providers all speak mu-law 8 kHz), so these helpers are only needed at
the edges: the call simulator, wav export of call audio, and the legacy
local-model path. Implemented without `audioop`, which was removed from
the stdlib in Python 3.13.
"""

from __future__ import annotations

import numpy as np

MULAW_BIAS = 0x84  # 132
MULAW_CLIP = 32635
SAMPLE_RATE_TELEPHONY = 8000
# 20 ms of mu-law @ 8 kHz — the frame size Twilio Media Streams uses.
FRAME_BYTES_20MS = 160


def pcm16_to_mulaw(pcm: np.ndarray) -> bytes:
    """Encode int16 PCM samples to 8-bit mu-law (G.711 mu-law)."""
    pcm = np.asarray(pcm, dtype=np.int16).astype(np.int32)
    sign = np.where(pcm < 0, 0x80, 0)
    magnitude = np.clip(np.abs(pcm), 0, MULAW_CLIP) + MULAW_BIAS

    # exponent = position of the highest set bit above bit 7 (0..7)
    exponent = (np.floor(np.log2(magnitude)) - 7).astype(np.int32)
    exponent = np.clip(exponent, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    encoded = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return encoded.astype(np.uint8).tobytes()


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    """Decode 8-bit mu-law bytes to int16 PCM samples."""
    u = ~np.frombuffer(data, dtype=np.uint8) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = ((mantissa.astype(np.int32) << 3) + MULAW_BIAS) << exponent
    magnitude -= MULAW_BIAS
    pcm = np.where(sign, -magnitude, magnitude)
    return np.clip(pcm, -32768, 32767).astype(np.int16)


def resample_pcm16(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resampler; fine for telephony-band speech."""
    if src_rate == dst_rate or len(pcm) == 0:
        return np.asarray(pcm, dtype=np.int16)
    duration = len(pcm) / src_rate
    n_out = int(round(duration * dst_rate))
    x_src = np.linspace(0.0, duration, num=len(pcm), endpoint=False)
    x_dst = np.linspace(0.0, duration, num=n_out, endpoint=False)
    out = np.interp(x_dst, x_src, pcm.astype(np.float64))
    return np.clip(out, -32768, 32767).astype(np.int16)


def iter_frames(data: bytes, frame_bytes: int = FRAME_BYTES_20MS):
    """Yield fixed-size frames from a byte buffer (last frame may be short)."""
    for i in range(0, len(data), frame_bytes):
        yield data[i:i + frame_bytes]
