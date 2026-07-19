import os
import json
import queue
import asyncio
import threading
import time
import torch
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from twilio.twiml.voice_response import VoiceResponse, Connect

from openvoicechat.stt.stt_faster_whisper import Ear_faster_whisper as Ear
from openvoicechat.tts.tts_piper import Mouth_piper
from openvoicechat.tts.tts_elevenlabs import Mouth_elevenlabs
from openvoicechat.llm.llm_ollama_rag import Chatbot_OllamaRAG as Chatbot
from twilio_audio_utils import decode_twilio_to_stt, encode_tts_to_twilio, chunk_tts_payload

from sql.database import SessionLocal
from sql import crud, schemas

# Global singletons removed per user request
router = APIRouter()

class TwilioListener:
    def __init__(self, input_queue):
        self.input_queue = input_queue
        self.listening = True
        self.call_active = True
        
        # Required by the VAD algorithm in utils.py (record_user_stream)
        self.CHUNK = 320  # 320 frames per 20ms at 16kHz (640 bytes)
        self.RATE = 16000

    def read(self, chunk_size):
        if not self.call_active:
            raise EOFError("Twilio stream disconnected")
        # We block and wait for the websocket to push 16kHz PCM chunks
        chunk = self.input_queue.get()
        if chunk is None or not self.call_active:
            raise EOFError("Twilio stream disconnected")
        return chunk

    def make_stream(self):
        if not self.call_active:
            raise EOFError("Twilio stream disconnected")
        self.listening = True
        self.input_queue.queue.clear()
        return self

    def close(self):
        # Called by record_user_stream when VAD detects the end of speech
        pass
        
    def stop_call(self):
        self.call_active = False
        self.listening = False
        self.input_queue.put(None)

class TwilioPlayer:
    def __init__(self, output_queue):
        self.output_queue = output_queue
        self.playing = False
        self.interrupted = False

    def play(self, audio_array, samplerate):
        self.playing = True
        self.interrupted = False
        
        # 1. Encode the raw TTS numpy array to Twilio's base64 8kHz mu-law format
        base64_payload = encode_tts_to_twilio(audio_array, samplerate)
        
        # 2. Chunk it to stream over the WebSocket
        chunks = chunk_tts_payload(base64_payload, 4000)
        
        for chunk in chunks:
            if self.interrupted:
                break
            self.output_queue.put(chunk)

    def stop(self):
        self.playing = False
        self.interrupted = True
        
    def wait(self):
        # Media Streaming pushes rapidly, we don't strictly block here for Twilio
        pass


class FallbackMouth:
    def __init__(self, player, device, voice_engine="urdu-female"):
        self.player = player
        self.device = device
        self.use_eleven = False
        self.eleven_mouth = None
        self.piper_mouth = None

        api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
        if api_key and api_key != "replace-with-provider-key":
            print(f"ElevenLabs API Key provided. Initializing ElevenLabs Mouth with voice engine: {voice_engine}...")
            voice_ids = {
                "urdu-female": "IKne3meq5aSn9XLyUdCD",
                "urdu-male": "pNInz6obpgq5qcGbe8x8",
                "english-us": "21m00Tcm4TlvDq8ikWAM"
            }
            selected_voice_id = voice_ids.get(voice_engine, "IKne3meq5aSn9XLyUdCD")
            try:
                self.eleven_mouth = Mouth_elevenlabs(api_key=api_key, voice_id=selected_voice_id, player=player)
                self.use_eleven = (voice_engine != "local-piper")
                print(f"ElevenLabs Mouth initialized successfully with voice_id {selected_voice_id}.")
            except Exception as e:
                print(f"Failed to initialize ElevenLabs Mouth: {e}. Falling back to default (Piper).")
                self.eleven_mouth = None
                self.use_eleven = False
        else:
            print("ElevenLabs API Key not provided or placeholder found. Using default (Piper).")

        # Always initialize Piper mouth as the default fallback
        print("Initializing default Piper Mouth...")
        try:
            self.piper_mouth = Mouth_piper(player=player, device=device)
            print("Piper Mouth initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Piper Mouth: {e}")
            raise e

    def say_text(self, text: str):
        if self.use_eleven and self.eleven_mouth:
            try:
                print(f"Synthesizing via ElevenLabs: '{text[:50]}...'")
                start = time.monotonic()
                output = self.eleven_mouth.run_tts(text)
                if output is not None and len(output) > 0:
                    end = time.monotonic()
                    duration = end - start
                    from utils.logger import log_response_time
                    log_response_time("ElevenLabs TTS Synthesized in Time", duration)
                    self.player.play(output, samplerate=self.eleven_mouth.sample_rate)
                    return
                else:
                    print("ElevenLabs TTS returned empty/None audio output. Falling back to default (Piper).")
            except Exception as e:
                print(f"ElevenLabs TTS generation error: {e}. Falling back to default (Piper).")
        
        # Default Piper TTS
        if self.piper_mouth:
            print(f"Synthesizing via default Piper: '{text[:50]}...'")
            start = time.monotonic()
            output = self.piper_mouth.run_tts(text)
            end = time.monotonic()
            duration = end - start
            from utils.logger import log_response_time
            log_response_time("Piper TTS Synthesized in Time", duration)
            self.player.play(output, samplerate=self.piper_mouth.sample_rate)
        else:
            print("Error: Piper mouth is not initialized.")


def resolve_restaurants_for_call(to_number: str, db) -> list:
    from sql.models import Restaurant
    if not to_number:
        return []
    return db.query(Restaurant).filter(Restaurant.order_phone_number == to_number).all()


@router.post("/voice")
async def voice(request: Request):
    """
    Twilio Webhook for incoming calls.
    Returns TwiML that connects the call to a WebSocket Media Stream.
    """
    form_data = await request.form()
    to_number = form_data.get("To", "")
    caller_number = form_data.get("From", "")
    
    from sql.database import SessionLocal
    
    db = SessionLocal()
    restaurants = []
    try:
        restaurants = resolve_restaurants_for_call(to_number, db)
        if not restaurants:
            print(f"WARNING: resolve_restaurants_for_call('{to_number}') returned zero matches in voice webhook.")
    except Exception as e:
        print(f"Error querying restaurants in voice webhook: {e}")
    finally:
        db.close()
        
    response = VoiceResponse()
    
    # Check if ALL matching restaurants are suspended. If so, reject the call.
    if restaurants and all(r.is_suspended for r in restaurants):
        print("Call rejected: All matching restaurants are suspended.")
        response.say("We are sorry, this service is temporarily unavailable. Goodbye.")
        return HTMLResponse(content=str(response), media_type="application/xml")
        
    # Check if the AI voice agent is manually paused
    if restaurants and restaurants[0].agent_configuration and not restaurants[0].agent_configuration.is_active:
        print("Agent is paused. Hanging up call.")
        response.say("Thank you for calling. The restaurant's AI assistant is currently offline. Goodbye.")
        response.hangup()
        return HTMLResponse(content=str(response), media_type="application/xml")
        
    connect = Connect()
    
    host = request.headers.get("host", "localhost:8000")
    scheme = "wss" if request.headers.get("x-forwarded-proto") == "https" or request.url.scheme == "https" else "ws"
    
    import urllib.parse
    caller_number_encoded = urllib.parse.quote(caller_number or "")
    to_number_encoded = urllib.parse.quote(to_number or "")
    ws_url = f"{scheme}://{host}/twilio-media-stream?caller_number={caller_number_encoded}&to_number={to_number_encoded}"
    
    print(f"Connecting Twilio call to WebSocket URL: {ws_url}")
    
    connect.stream(url=ws_url)
    response.append(connect)
    
    return HTMLResponse(content=str(response), media_type="application/xml")


@router.websocket("/twilio-media-stream")
async def twilio_media_stream(
    websocket: WebSocket,
    caller_number: Optional[str] = None,
    to_number: Optional[str] = None
):
    await websocket.accept()
    print(f"Twilio Media Stream WebSocket Connected (Caller: {caller_number}, To: {to_number}).")
    
    from sql.database import SessionLocal
    from sql import models
    
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    listener = TwilioListener(input_queue)
    player = TwilioPlayer(output_queue)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    stream_sid = None
    call_start_time = None
    
    chatbot_messages = []
    call_status = "completed"
    
    def conversation_loop():
        nonlocal chatbot_messages, call_status
        print("Initializing AI models in background thread...")
        
        ear = Ear(
            model_size=os.getenv("STT_MODEL_SIZE", "small"),
            device=device,
            compute_type=compute_type,
            silence_seconds=2.0,
            listener=listener,
            stream=True,
            player=player
        )
        
        # Load agent configuration from database
        db = SessionLocal()
        sys_prompt = "You are the friendly AI voice-agent taking orders. Speak in Roman Urdu/Urdu-English mix."
        greeting = "Assalam-o-Alaikum! Aaj aap kya order karna pasand farmayenge?"
        voice_engine = "urdu-female"
        
        try:
            from sql import models
            # to_number is the Twilio number called (e.g. restaurant.order_phone_number)
            # Find the restaurant matching this phone number
            restaurant = db.query(models.Restaurant).filter(models.Restaurant.order_phone_number == to_number).first()
            if restaurant:
                agent_config = restaurant.agent_configuration
                if not agent_config:
                    agent_config = models.AgentConfiguration(
                        restaurant_id=restaurant.id,
                        voice_engine="urdu-female",
                        system_prompt="You are the friendly AI voice-agent taking orders for Khyber Shinwari Restaurant. Speak in Roman Urdu/Urdu-English mix. Reference the menu database for pricing. Do not offer discounts exceeding PKR 100. Do not repeat greetings (like Assalam-o-Alaikum) in subsequent turns. Be extremely brief, using under 30 words per conversation bubble. Make sure to collect the customer's delivery address before concluding the order."
                    )
                    db.add(agent_config)
                    db.commit()
                    db.refresh(agent_config)
                
                sys_prompt = agent_config.system_prompt
                voice_engine = agent_config.voice_engine
                
                # Fetch greeting directly from RAG or store default
                try:
                    import requests
                    rag_business_id = str(restaurant.id)
                    rag_url = os.getenv("RAG_UPSTREAM", "http://127.0.0.1:8001")
                    r = requests.get(f"{rag_url}/businesses/{rag_business_id}/profile", timeout=5.0)
                    if r.status_code == 200:
                        profile = r.json()
                        persona = profile.get("persona") or {}
                        if persona.get("greeting_script"):
                            greeting = persona["greeting_script"]
                        else:
                            greeting = f"Assalam-o-Alaikum! {restaurant.name} se AI assistant bol rahi hoon. Aaj aap kya order karna pasand farmayenge?"
                            requests.patch(f"{rag_url}/businesses/{rag_business_id}/profile", json={"persona": {"greeting_script": greeting}}, timeout=5.0)
                except Exception as e:
                    print(f"Error fetching greeting from RAG: {e}")
                    greeting = f"Assalam-o-Alaikum! {restaurant.name} se AI assistant bol rahi hoon. Aaj aap kya order karna pasand farmayenge?"
        except Exception as e:
            print(f"Error loading agent configuration in conversation_loop: {e}")
        finally:
            db.close()

        mouth = FallbackMouth(player=player, device=device, voice_engine=voice_engine)

        # Resolve RAG business_id from the restaurant record (slug: restaurant name
        # lowercased + underscored) or fall back to the env var.
        rag_business_id = os.getenv("RAG_BUSINESS_ID", "")
        if not rag_business_id:
            try:
                db2 = SessionLocal()
                rest = db2.query(models.Restaurant).filter(
                    models.Restaurant.order_phone_number == to_number
                ).first()
                if rest:
                    rag_business_id = str(rest.id)
                db2.close()
            except Exception:
                pass

        chatbot = Chatbot(
            sys_prompt=sys_prompt,
            business_id=rag_business_id,
        )
        
        time.sleep(0.5)
        
        print("Speaking greeting...")
        mouth.say_text(greeting)
        chatbot.messages.append({"role": "assistant", "content": greeting})
        chatbot_messages = chatbot.messages
        
        while True:
            try:
                if not listener.call_active:
                    raise EOFError("Twilio stream disconnected")
                print("Listening for user...")
                user_text = ear.listen()
                if not listener.call_active:
                    raise EOFError("Twilio stream disconnected")
                
                if not user_text or not user_text.strip():
                    continue
                
                print(f"User said: {user_text.strip()}")
                print("Generating LLM response...")
                
                llm_response = ""
                for chunk in chatbot.run(user_text):
                    llm_response += chunk
                
                print(f"LLM said: {llm_response}")
                chatbot.post_process(llm_response)
                chatbot_messages = chatbot.messages
                
                print("Speaking LLM response...")
                mouth.say_text(llm_response)
            except EOFError:
                print("Caller hung up. Ending conversation loop gracefully.")
                call_status = "completed"
                break
            except Exception as e:
                print(f"Conversation loop ended or error: {e}")
                call_status = "failed"
                break

    loop_thread = threading.Thread(target=conversation_loop, daemon=True)
    
    async def send_audio_task():
        while True:
            try:
                chunk = await asyncio.to_thread(output_queue.get)
                if chunk is None:
                    break
                if stream_sid:
                    media_msg = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": chunk}
                    }
                    await websocket.send_json(media_msg)
            except Exception as e:
                print(f"Send audio error: {e}")
                break

    send_task = asyncio.create_task(send_audio_task())
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            event = message.get("event")
            
            if event == "connected":
                print("Twilio Stream connected event received.")
            elif event == "start":
                stream_sid = message["start"]["streamSid"]
                print(f"Twilio Stream started. StreamSid: {stream_sid}")
                from websocket_registry import active_connections
                active_connections[stream_sid] = websocket
                call_start_time = time.time()
                
                db = SessionLocal()
                try:
                    restaurants = resolve_restaurants_for_call(to_number, db)
                    if not restaurants:
                        print(f"WARNING: resolve_restaurants_for_call('{to_number}') returned zero matches on call start.")
                    for r in restaurants:
                        new_log = models.ChatHistory(
                            session_id=stream_sid,
                            restaurant_id=r.id,
                            caller_number=caller_number,
                            chat_data=[],
                            response_time=0.0,
                            status="in_progress",
                            duration_seconds=None,
                            recording_url=None
                        )
                        db.add(new_log)
                    db.commit()
                    print(f"Created initial call_logs in_progress rows for {len(restaurants)} restaurants.")
                except Exception as db_e:
                    print(f"Error creating call_logs rows on start: {db_e}")
                    db.rollback()
                finally:
                    db.close()
                
                loop_thread.start()
                
            elif event == "media":
                chunk = message["media"]["payload"]
                pcm_bytes = decode_twilio_to_stt(chunk)
                if listener.listening:
                    input_queue.put(pcm_bytes)
            elif event == "stop":
                print("Twilio Stream stop event received.")
                break
            else:
                pass
    except WebSocketDisconnect:
        print("Twilio Media Stream WebSocket Disconnected cleanly.")
    except Exception as e:
        print(f"Twilio Media Stream error: {e}")
    finally:
        print("Closing Twilio Media Stream connection.")
        from websocket_registry import active_connections
        if stream_sid:
            active_connections.pop(stream_sid, None)
        output_queue.put(None)
        listener.stop_call()
        player.stop()
        
        if call_start_time is not None:
            call_end_time = time.time()
            duration_seconds = int(call_end_time - call_start_time)
            duration_minutes = duration_seconds / 60.0
            print(f"Call ended. Duration: {duration_seconds} seconds ({duration_minutes:.2f} minutes).")
            
            db = SessionLocal()
            try:
                restaurants = resolve_restaurants_for_call(to_number, db)
                
                for r in restaurants:
                    r.used_minutes += duration_minutes
                    if r.used_minutes >= r.assigned_minutes:
                        r.is_suspended = True
                        print(f"Restaurant '{r.name}' has exceeded its quota ({r.used_minutes:.2f}/{r.assigned_minutes} minutes). Auto-suspending.")
                
                if stream_sid:
                    user_spoke = False
                    if chatbot_messages:
                        user_spoke = any(msg.get("role") == "user" for msg in chatbot_messages)
                    
                    final_status = call_status
                    if final_status == "completed" and not user_spoke:
                        final_status = "missed"
                        
                    logs = db.query(models.ChatHistory).filter(models.ChatHistory.session_id == stream_sid).all()
                    for log in logs:
                        log.chat_data = [msg for msg in chatbot_messages if msg.get("role") != "system"]
                        log.duration_seconds = duration_seconds
                        log.status = final_status
                        
                        # Auto-extract order from transcript if the call was completed
                        if final_status == "completed":
                            from utils.order_extractor import extract_order_from_transcript
                            try:
                                extract_order_from_transcript(log, db)
                            except Exception as parse_err:
                                print(f"Error auto-extracting order from call transcript: {parse_err}")
                
                db.commit()
                print(f"Updated call_logs and minutes for {len(restaurants)} restaurants.")
            except Exception as db_err:
                print(f"Error updating call logs/minutes: {db_err}")
                db.rollback()
            finally:
                db.close()

        try:
            await websocket.close()
        except RuntimeError:
            pass
