import sounddevice as sd
import numpy as np
if __name__ == '__main__':
    from base import BaseMouth
else:
    from .base import BaseMouth

class Mouth_piper(BaseMouth):
    def __init__(self, device='cpu', model_path='models/en_US-ryan-high.onnx',
                 config_path='models/en_US-ryan-high.onnx.json',
                 player=sd):
        import piper
        import os
        import urllib.request

        # If the default linux path is passed, or if the path doesn't exist, use local workspace path
        if "/home/salman" in model_path or not os.path.exists(model_path):
            workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            models_dir = os.path.join(workspace_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            # Local paths
            local_model_path = os.path.join(models_dir, "en_US-ryan-high.onnx")
            local_config_path = os.path.join(models_dir, "en_US-ryan-high.onnx.json")
            
            # URLs to download from
            model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
            config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
            
            if not os.path.exists(local_model_path):
                print(f"Downloading Piper model to {local_model_path}...")
                urllib.request.urlretrieve(model_url, local_model_path)
                print("Model download completed.")
                
            if not os.path.exists(local_config_path):
                print(f"Downloading Piper config to {local_config_path}...")
                urllib.request.urlretrieve(config_url, local_config_path)
                print("Config download completed.")
                
            model_path = local_model_path
            config_path = local_config_path

        self.model = piper.PiperVoice.load(model_path=model_path,
                                           config_path=config_path,
                                           use_cuda=True if device == 'cuda' else False)
        super().__init__(sample_rate=self.model.config.sample_rate, player=player)

    def run_tts(self, text):
        audio = b''
        for chunk in self.model.synthesize(text):
            audio += chunk.audio_int16_bytes
        return np.frombuffer(audio, dtype=np.int16)


if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mouth = Mouth_piper(device=device, model_path='models/en_US-ryan-high.onnx',
                        config_path='models/en_US-ryan-high.onnx.json')

    text = ("If there's one thing that makes me nervous about the future of self-driving cars, it's that they'll "
            "replace human drivers.\nI think there's a huge opportunity to make human-driven cars safer and more "
            "efficient. There's no reason why we can't combine the benefits of self-driving cars with the ease of use "
            "of human-driven cars.")
    print(text)
    mouth.say_multiple(text, lambda x: False)
    sd.wait()
