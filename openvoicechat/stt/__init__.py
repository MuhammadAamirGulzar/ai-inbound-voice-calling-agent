from .base import BaseEar as BaseEar

# Optional STT backends — guarded to avoid import errors
try:
    from .stt_deepgram import Ear_deepgram as Ear_deepgram
except ImportError:
    pass

try:
    from .stt_hf import Ear_hf as Ear_hf
except ImportError:
    pass

# from .stt_vosk import Ear_vosk as Ear_vosk