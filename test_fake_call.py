import asyncio
import json
import base64
import wave
import audioop
import sys
import os
import math
import websockets

async def test_fake_call(audio_file="test.wav"):
    uri = "ws://127.0.0.1:8000/twilio-media-stream"
    stream_sid = "fake_stream_sid_12345"
    
    # Fallback: Generate a simple 1-second sine wave if the file isn't found
    if not os.path.exists(audio_file):
        print(f"Warning: '{audio_file}' not found. Generating a 1-second 440Hz sine wave to test connection.")
        with wave.open(audio_file, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            
            samples = []
            for i in range(8000): # 1 second at 8kHz
                sample = int(32767.0 * math.sin(2 * math.pi * 440 * (i / 8000.0)))
                samples.append(sample)
                
            import struct
            wav.writeframes(struct.pack('<' + 'h'*8000, *samples))
            
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected.")
            
            # 1. Send Connected Event
            connected_msg = {
                "event": "connected",
                "protocol": "Call",
                "version": "1.0.0"
            }
            await websocket.send(json.dumps(connected_msg))
            
            # 2. Send Start Event
            start_msg = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "accountSid": "fake_account_sid",
                    "streamSid": stream_sid,
                    "callSid": "fake_call_sid",
                    "tracks": ["inbound"],
                    "mediaFormat": {
                        "encoding": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "channels": 1
                    }
                }
            }
            await websocket.send(json.dumps(start_msg))
            print("Sent 'start' event.")
            
            # Spin up a task to continually receive and decode the agent's audio
            server_audio_buffer = bytearray()
            agent_finished_speaking = asyncio.Event()
            receive_task = asyncio.create_task(receive_audio(websocket, agent_finished_speaking, server_audio_buffer))
            
            # Read and process the local WAV file
            with wave.open(audio_file, "rb") as wf:
                sample_rate = wf.getframerate()
                n_channels = wf.getnchannels()
                samp_width = wf.getsampwidth()
                
                pcm_data = wf.readframes(wf.getnframes())
                
                # Convert formats gracefully using audioop
                if n_channels == 2:
                    pcm_data = audioop.tomono(pcm_data, samp_width, 0.5, 0.5)
                    
                if sample_rate != 8000:
                    pcm_data, _ = audioop.ratecv(pcm_data, samp_width, 1, sample_rate, 8000, None)
                    
                # Convert 16-bit linear PCM to 8-bit mu-law
                mulaw_data = audioop.lin2ulaw(pcm_data, 2)
                
            print(f"Loaded '{audio_file}', converted to 8kHz mu-law. Total bytes: {len(mulaw_data)}")
            
            # Dynamically wait for the agent's greeting to finish
            print("Waiting for the agent to finish its greeting...")
            await agent_finished_speaking.wait()
            
            # Reset the event so we can dynamically detect the LLM's response burst later if needed
            agent_finished_speaking.clear()
            
            # 3. Stream 'media' events in paced ~20ms chunks (160 bytes per chunk at 8kHz)
            chunk_size = 160
            seq_num = 1
            
            print("Streaming audio to server...")
            for i in range(0, len(mulaw_data), chunk_size):
                chunk = mulaw_data[i:i+chunk_size]
                b64_chunk = base64.b64encode(chunk).decode("utf-8")
                
                media_msg = {
                    "event": "media",
                    "sequenceNumber": str(seq_num),
                    "streamSid": stream_sid,
                    "media": {
                        "track": "inbound",
                        "chunk": str(seq_num),
                        "timestamp": str(seq_num * 20),
                        "payload": b64_chunk
                    }
                }
                await websocket.send(json.dumps(media_msg))
                seq_num += 1
                
                # Pace the websocket exactly like a real Twilio call
                await asyncio.sleep(0.02)
                
            print("Finished sending local audio.")
            
            print("Sending continuous silence so the agent's VAD detects the end of speech...")
            silence_pcm = b'\x00' * 320
            silence_mulaw = audioop.lin2ulaw(silence_pcm, 2)
            b64_silence = base64.b64encode(silence_mulaw).decode("utf-8")
            
            async def keep_sending_silence():
                nonlocal seq_num
                try:
                    while True:
                        media_msg = {
                            "event": "media",
                            "sequenceNumber": str(seq_num),
                            "streamSid": stream_sid,
                            "media": {
                                "track": "inbound",
                                "chunk": str(seq_num),
                                "timestamp": str(seq_num * 20),
                                "payload": b64_silence
                            }
                        }
                        await websocket.send(json.dumps(media_msg))
                        seq_num += 1
                        await asyncio.sleep(0.02)
                except websockets.exceptions.ConnectionClosed:
                    pass

            silence_task = asyncio.create_task(keep_sending_silence())
            
            print("Agent is now processing and will speak back...")
            print("Press Ctrl+C to hang up the call when you are done listening.")
            
            # Keep the connection alive indefinitely until the user manually hangs up with Ctrl+C.
            while True:
                await asyncio.sleep(1)
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket connection closed by server.")
    except Exception as e:
        import traceback
        print("\n--- Exception Traceback ---")
        traceback.print_exc()
        print("---------------------------\n")
        print(f"Error: {e}")


async def receive_audio(websocket, agent_finished_speaking, server_audio_buffer):
    """
    Continually consumes websocket messages, parsing out base64 mu-law chunks,
    converting them back to PCM, and eventually saving them into response.wav.
    """
    try:
        received_any_media = False
        while True:
            try:
                # Wait for a message, but timeout after 1 second of silence
                data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                message = json.loads(data)
                
                if message.get("event") == "media":
                    received_any_media = True
                    payload = message["media"]["payload"]
                    mulaw_chunk = base64.b64decode(payload)
                    
                    # Convert mu-law back to 16-bit PCM to make it playable
                    pcm_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
                    server_audio_buffer.extend(pcm_chunk)
                    
                elif message.get("event") == "stop":
                    print("\nServer sent a 'stop' event.")
                elif message.get("event") == "mark":
                    print("\nServer sent a 'mark' event.")
                    
            except asyncio.TimeoutError:
                # If we timed out AND we previously received media, the burst is over!
                if received_any_media and not agent_finished_speaking.is_set():
                    print("--> Agent finished speaking! Server is now in listening mode.")
                    agent_finished_speaking.set()
                continue
                
    except websockets.exceptions.ConnectionClosed:
        pass
    except asyncio.CancelledError:
        pass
    finally:
        # Save the agent's accumulated reply to a playable WAV file
        if len(server_audio_buffer) > 0:
            output_file = "response.wav"
            with wave.open(output_file, 'wb') as wav:
                wav.setnchannels(1)       # Mono
                wav.setsampwidth(2)       # 16-bit
                wav.setframerate(8000)    # 8kHz
                wav.writeframes(server_audio_buffer)
            print(f"\nSaved agent's response to {output_file} ({len(server_audio_buffer)} bytes).")
        else:
            print("\nNo audio received from the agent.")


if __name__ == "__main__":
    audio_filename = sys.argv[1] if len(sys.argv) > 1 else "test.wav"
    try:
        asyncio.run(test_fake_call(audio_filename))
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
