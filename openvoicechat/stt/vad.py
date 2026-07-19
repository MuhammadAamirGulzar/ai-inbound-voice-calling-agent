import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from utils import record_user
else:
    from .utils import record_user


class VoiceActivityDetection:
    def __init__(self, sampling_rate=16000):
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils
        self.sampling_rate = sampling_rate

    def contains_speech(self, audio):
        frames = np.frombuffer(b"".join(audio), dtype=np.int16)

        # Normalization: https://discuss.pytorch.org/t/torchaudio-load-normalization-question/71470
        frames = frames / (1 << 15)

        audio = torch.tensor(frames.astype(np.float32))
        speech_timestamps = self.get_speech_timestamps(
            audio, self.model, sampling_rate=self.sampling_rate, threshold=0.5
        )  # threshold=0.5
        return len(speech_timestamps) > 0


if __name__ == "__main__":
    from transformers import pipeline
    import torch
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the pipeline for automatic speech recognition
    pipe = pipeline(
        "automatic-speech-recognition", model="openai/whisper-base.en", device=device
    )

    # Assume your audio is sampled at 16000 Hz
    SAMPLING_RATE = 16000

    def transcribe(audio):
        from torch import no_grad

        # Extract input features from the audio and pass the sampling rate
        input_features = pipe.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        ).input_features.to(device)

        # Create the attention mask (1s where there's actual input data, 0s where there's padding)
        attention_mask = torch.ones(input_features.shape[:2], dtype=torch.long).to(
            device
        )

        with no_grad():
            # Generate transcription using the model, passing input features and attention mask
            transcription = pipe.model.generate(
                input_features=input_features, attention_mask=attention_mask
            )

        # Decode the transcription tokens into text
        transcription_text = pipe.tokenizer.batch_decode(
            transcription, skip_special_tokens=True
        )
        return transcription_text[0].strip()

    # Voice activity detection and recording
    vad = VoiceActivityDetection()
    audio = record_user(2, vad)

    # Transcribe the audio
    text = transcribe(np.array(audio))
    print(text)
