import sounddevice as sd
import numpy as np

if __name__ == "__main__":
    from base import BaseMouth
else:
    from .base import BaseMouth


class Mouth_xtts(BaseMouth):
    def __init__(
        self,
        model_id="tts_models/en/jenny/jenny",
        device="cpu",
        player=sd,
        speaker=None,
        wait=True,
    ):
        from TTS.api import TTS

        self.model = TTS(model_id)
        self.device = device
        self.model.to(device)
        self.speaker = speaker
        super().__init__(
            sample_rate=self.model.synthesizer.output_sample_rate,
            player=player,
            wait=wait,
        )

    def run_tts(self, text):
        output = self.model.tts(
            text=text,
            split_sentences=False,
            speaker=self.speaker,
            language="en" if self.model.is_multi_lingual else None,
        )
        return np.array(output)
