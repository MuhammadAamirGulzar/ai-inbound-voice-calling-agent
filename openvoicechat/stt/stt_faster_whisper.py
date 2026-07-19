# stt_faster_whisper.py

import os
import time
import numpy as np
import torch
from .base import BaseEar

from utils.logger import log_response_time, print_info, print_error, print_warning

class Ear_faster_whisper(BaseEar):
    def __init__(
        self,
        model_size="large-v3",
        device="cuda",
        compute_type="float16",
        silence_seconds=2,
        beam_size=5,
        language="en",
        condition_on_previous_text=False,
        word_timestamps=False,
        listener=None,
        stream=True,  # Default to stream mode for better timing measurements
        player=None,  # Added player parameter to match BaseEar
    ):
        super().__init__(silence_seconds=silence_seconds, listener=listener, stream=stream, player=player)  
        
        from faster_whisper import WhisperModel
        
        # Initialize the faster whisper model
        self.model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=2
        )
        
        self.beam_size = beam_size
        self.language = language
        self.condition_on_previous_text = condition_on_previous_text
        self.word_timestamps = word_timestamps
        
        print(f"Initialized Faster Whisper with model {model_size} on {device}")
    
    def transcribe(self, audio):
        """
        Transcribe audio using Faster Whisper
        
        :param audio: Audio data as numpy array
        :return: Transcribed text
        """
        # Start timing
        start_time = time.time()
        
        # Convert audio to appropriate format if needed
        # Faster Whisper expects audio in 16kHz, float32
        if isinstance(audio, np.ndarray):
            # If audio is already a numpy array, use it directly
            audio_data = audio
        else:
            # Otherwise convert to numpy array
            audio_data = np.frombuffer(audio, np.float32)
        
        # Transcribe using faster whisper
        segments, info = self.model.transcribe(
            audio_data,
            beam_size=self.beam_size,
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=self.word_timestamps
        )
        
        # Gather all segments
        all_segments = list(segments)
        
        # Extract transcript
        transcript = " ".join(segment.text for segment in all_segments).strip()
        
        # Record timing for this transcription
        end_time = time.time()
        self.last_transcription_time = end_time - start_time
        
        print(f"Transcription completed in {self.last_transcription_time:.3f}s")
        return transcript
    
    def transcribe_stream(self, audio_queue, transcription_queue):
        """
        Transcribe audio stream using Faster Whisper
        
        :param audio_queue: Queue containing audio chunks
        :param transcription_queue: Queue to put transcriptions into
        """
        # Accumulate audio chunks
        audio_data = bytearray()
        while True:
            chunk = audio_queue.get()
            if chunk is None:
                break
            audio_data.extend(chunk)
        
        # Convert to numpy array for processing
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, info = self.model.transcribe(
            audio_np,
            beam_size=self.beam_size,
            language=self.language,
            condition_on_previous_text=self.condition_on_previous_text,
            word_timestamps=self.word_timestamps
        )
        start_time = time.time()
        # Process segments
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        
        # Record timing for this transcription
        end_time = time.time()

        log_response_time("STT Transcribed in Time", end_time - start_time)
        self.last_transcription_time = end_time - start_time
        
        # Put the result in the transcription queue
        transcription_queue.put(transcript.strip())
        transcription_queue.put(None)  # Signal end of transcription