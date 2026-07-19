import base64
import audioop
import numpy as np
import librosa

def decode_twilio_to_stt(base64_payload: str) -> bytes:
    """
    Decodes a base64 encoded mu-law audio payload from Twilio (8kHz),
    and converts it to 16-bit PCM at 16kHz for Faster Whisper STT.

    :param base64_payload: The base64 string from Twilio's "media" event payload.
    :return: Raw 16-bit PCM bytes at 16kHz.
    """
    # 1. Base64 decode to raw mu-law bytes
    mulaw_bytes = base64.b64decode(base64_payload)
    
    # 2. Convert mu-law (8-bit) to linear PCM (16-bit)
    # The '2' denotes the sample width in bytes for the output (16-bit)
    pcm_bytes_8k = audioop.ulaw2lin(mulaw_bytes, 2)
    
    # 3. Resample from 8kHz to 16kHz using audioop for ultra-low latency
    # Parameters: (data, width, nchannels, inrate, outrate, state)
    pcm_bytes_16k, _state = audioop.ratecv(pcm_bytes_8k, 2, 1, 8000, 16000, None)
    
    return pcm_bytes_16k

def encode_tts_to_twilio(audio_array: np.ndarray, input_sample_rate: int) -> str:
    """
    Takes an int16 numpy array from TTS at a given sample rate, resamples it to 8kHz, 
    encodes to mu-law, and returns a base64 string ready for Twilio.

    :param audio_array: 1D numpy array containing int16 audio data.
    :param input_sample_rate: The sample rate of the incoming audio_array (e.g., 22050 for Piper).
    :return: Base64 encoded string of 8kHz mu-law audio.
    """
    # Ensure it's a 1D array
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()

    # 1. Resample to 8kHz if necessary
    if input_sample_rate != 8000:
        # librosa expects float32 between -1.0 and 1.0
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        audio_8k_float = librosa.resample(
            audio_float, 
            orig_sr=input_sample_rate, 
            target_sr=8000
        )
        
        # Convert back to int16 bytes safely
        audio_8k_int16 = np.clip((audio_8k_float * 32767.0), -32768, 32767).astype(np.int16)
        pcm_bytes_8k = audio_8k_int16.tobytes()
    else:
        pcm_bytes_8k = audio_array.astype(np.int16).tobytes()
    
    # 2. Convert linear PCM (16-bit) to mu-law (8-bit)
    # The '2' denotes the sample width in bytes of the input
    mulaw_bytes = audioop.lin2ulaw(pcm_bytes_8k, 2)
    
    # 3. Base64 encode to string for Twilio JSON payload
    base64_payload = base64.b64encode(mulaw_bytes).decode("utf-8")
    
    return base64_payload

def chunk_tts_payload(base64_payload: str, max_chunk_size: int = 4000) -> list[str]:
    """
    (Optional helper) Splits a large base64 payload into smaller chunks.
    Streaming audio smoothly back to Twilio is often better in small increments 
    (e.g., ~20-50ms chunks) rather than dumping 10 seconds of audio all at once.
    
    :param base64_payload: The full base64 audio string.
    :param max_chunk_size: Approximate string size of each chunk.
    :return: List of base64 chunks.
    """
    return [base64_payload[i:i+max_chunk_size] for i in range(0, len(base64_payload), max_chunk_size)]
