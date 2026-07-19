from .base import BaseMouth as BaseMouth
from .tts_elevenlabs import Mouth_elevenlabs as Mouth_elevenlabs
from .tts_piper import Mouth_piper as Mouth_piper

# Optional TTS backends — guarded to avoid import errors when their
# heavy dependencies (TTS/Coqui, parler_tts, transformers pipelines)
# are not installed.
try:
    from .tts_hf import Mouth_hf as Mouth_hf
except ImportError:
    pass

try:
    from .tts_parler import Mouth_parler as Mouth_parler
except ImportError:
    pass

# from .tts_tortoise import Mouth as Mouth_tortoise

try:
    from .tts_xtts import Mouth_xtts as Mouth_xtts
except ImportError:
    pass
