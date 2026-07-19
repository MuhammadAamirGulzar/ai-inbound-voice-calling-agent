import base64
import audioop
import numpy as np
import librosa

def decode_sip_to_stt(binary_payload: bytes, input_sample_rate: int) -> bytes:
    """
    Takes binary raw linear16 PCM audio from FreeSWITCH mod_audio_stream
    and converts/resamples it to 16-bit PCM at 16kHz for Faster Whisper STT.

    :param binary_payload: The raw bytes (L16 PCM) received from mod_audio_stream.
    :param input_sample_rate: The sample rate of the incoming audio (typically 16000 or 8000).
    :return: Raw 16-bit PCM bytes at 16kHz.
    """
    # L16 is 16-bit linear PCM, which is sample width of 2 bytes.
    if input_sample_rate == 16000:
        return binary_payload

    # Resample using audioop.ratecv (low latency, no floating-point conversion needed)
    # Parameters: (data, width, nchannels, inrate, outrate, state)
    pcm_bytes_16k, _state = audioop.ratecv(binary_payload, 2, 1, input_sample_rate, 16000, None)
    return pcm_bytes_16k

def encode_tts_to_sip(audio_array: np.ndarray, input_sample_rate: int, target_sample_rate: int = 16000) -> str:
    """
    Takes a 1D int16 numpy array from a TTS engine, resamples it to the target
    sample rate (e.g. 16000 or 8000 Hz), and returns a base64 encoded string
    of the raw PCM bytes suitable for mod_audio_stream.

    :param audio_array: 1D numpy array containing int16 audio data.
    :param input_sample_rate: The sample rate of the incoming audio_array (e.g., 22050 for Piper).
    :param target_sample_rate: The sample rate required by FreeSWITCH (typically 16000 or 8000).
    :return: Base64 encoded string of target_sample_rate L16 PCM audio.
    """
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()

    if input_sample_rate != target_sample_rate:
        # Convert to float32 between -1.0 and 1.0 for librosa.resample
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        audio_resampled_float = librosa.resample(
            audio_float, 
            orig_sr=input_sample_rate, 
            target_sr=target_sample_rate
        )
        
        # Convert back to int16 bytes safely
        audio_resampled_int16 = np.clip((audio_resampled_float * 32767.0), -32768, 32767).astype(np.int16)
        pcm_bytes = audio_resampled_int16.tobytes()
    else:
        pcm_bytes = audio_array.astype(np.int16).tobytes()

    return base64.b64encode(pcm_bytes).decode("utf-8")
