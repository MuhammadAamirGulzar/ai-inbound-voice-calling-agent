import os
import json
import queue
import asyncio
import threading
import time
import torch
import uuid
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
from openvoicechat.stt.stt_faster_whisper import Ear_faster_whisper as Ear
from openvoicechat.tts.tts_piper import Mouth_piper
from openvoicechat.tts.tts_elevenlabs import Mouth_elevenlabs
from openvoicechat.llm.llm_ollama_rag import Chatbot_OllamaRAG as Chatbot

from telephony.sip_audio import decode_sip_to_stt, encode_tts_to_sip
from sql.database import SessionLocal
from sql import models, crud, schemas

router = APIRouter()

class SIPListener:
    def __init__(self, input_queue):
        self.input_queue = input_queue
        self.listening = True
        self.call_active = True
        
        # Required by the VAD algorithm in utils.py (record_user_stream)
        self.CHUNK = 320  # 320 frames per 20ms at 16kHz (640 bytes)
        self.RATE = 16000

    def read(self, chunk_size):
        if not self.call_active:
            raise EOFError("SIP stream disconnected")
        chunk = self.input_queue.get()
        if chunk is None or not self.call_active:
            raise EOFError("SIP stream disconnected")
        return chunk

    def make_stream(self):
        if not self.call_active:
            raise EOFError("SIP stream disconnected")
        self.listening = True
        self.input_queue.queue.clear()
        return self

    def close(self):
        pass
        
    def stop_call(self):
        self.call_active = False
        self.listening = False
        self.input_queue.put(None)


class SIPPlayer:
    def __init__(self, output_queue, target_sample_rate=16000):
        self.output_queue = output_queue
        self.playing = False
        self.interrupted = False
        self.target_sample_rate = target_sample_rate

    def play(self, audio_array, samplerate):
        self.playing = True
        self.interrupted = False
        
        # Encode raw TTS numpy array to base64 L16 PCM
        base64_payload = encode_tts_to_sip(audio_array, samplerate, self.target_sample_rate)
        
        # Construct message format for mod_audio_stream
        payload = {
            "type": "streamAudio",
            "data": {
                "audioDataType": "raw",
                "sampleRate": self.target_sample_rate,
                "audioData": base64_payload
            }
        }
        
        if not self.interrupted:
            self.output_queue.put(payload)

    def stop(self):
        self.playing = False
        self.interrupted = True
        
    def wait(self):
        pass


class SIPFallbackMouth:
    def __init__(self, player, device, voice_engine="urdu-female"):
        self.player = player
        self.piper_mouth = None
        self.eleven_mouth = None
        self.use_eleven = False

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


@router.websocket("/sip-media-stream")
async def sip_media_stream(
    websocket: WebSocket,
    caller_number: Optional[str] = None,
    to_number: Optional[str] = None
):
    await websocket.accept()
    print(f"SIP Media Stream WebSocket Connected (Caller: {caller_number}, To: {to_number}).")
    
    # Check if SIP is enabled
    sip_enabled = os.getenv("SIP_ENABLED", "false").lower() == "true"
    if not sip_enabled:
        print("SIP pathway is disabled via SIP_ENABLED env var. Closing connection.")
        await websocket.close()
        return

    from sql.database import SessionLocal
    from sql import models
    
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    
    sip_sample_rate = int(os.getenv("SIP_STREAM_SAMPLE_RATE", "16000"))
    
    listener = SIPListener(input_queue)
    player = SIPPlayer(output_queue, target_sample_rate=sip_sample_rate)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    session_id = None
    call_start_time = None
    
    chatbot_messages = []
    call_status = "completed"
    
    c_num = caller_number
    t_num = to_number
    
    loop_thread = None
    loop_thread_started = False
    
    def conversation_loop():
        nonlocal chatbot_messages, call_status
        print("Initializing AI models for SIP loop in background thread...")
        
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
            restaurant = db.query(models.Restaurant).filter(models.Restaurant.order_phone_number == t_num).first()
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
            print(f"Error loading agent configuration in SIP conversation_loop: {e}")
        finally:
            db.close()

        mouth = SIPFallbackMouth(player=player, device=device, voice_engine=voice_engine)

        # Resolve RAG business_id from the restaurant or env var.
        rag_business_id = os.getenv("RAG_BUSINESS_ID", "")
        if not rag_business_id:
            try:
                db2 = SessionLocal()
                rest = db2.query(models.Restaurant).filter(
                    models.Restaurant.order_phone_number == t_num
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
        
        print("Speaking greeting over SIP...")
        mouth.say_text(greeting)
        chatbot.messages.append({"role": "assistant", "content": greeting})
        chatbot_messages = chatbot.messages
        
        while True:
            try:
                if not listener.call_active:
                    raise EOFError("SIP stream disconnected")
                print("Listening for SIP user...")
                user_text = ear.listen()
                if not listener.call_active:
                    raise EOFError("SIP stream disconnected")
                
                if not user_text or not user_text.strip():
                    continue
                
                print(f"SIP User said: {user_text.strip()}")
                print("Generating LLM response for SIP call...")
                
                llm_response = ""
                for chunk in chatbot.run(user_text):
                    llm_response += chunk
                
                print(f"SIP LLM said: {llm_response}")
                chatbot.post_process(llm_response)
                chatbot_messages = chatbot.messages
                
                print("Speaking LLM response over SIP...")
                mouth.say_text(llm_response)
            except EOFError:
                print("SIP Caller hung up. Ending conversation loop gracefully.")
                call_status = "completed"
                break
            except Exception as e:
                print(f"SIP Conversation loop ended or error: {e}")
                call_status = "failed"
                break

    async def send_audio_task():
        while True:
            try:
                payload = await asyncio.to_thread(output_queue.get)
                if payload is None:
                    break
                await websocket.send_json(payload)
            except Exception as e:
                print(f"SIP Send audio error: {e}")
                break

    send_task = asyncio.create_task(send_audio_task())
    
    try:
        while True:
            data = await websocket.receive()
            
            # Handle text frame (metadata/control message)
            if "text" in data:
                text_msg = data["text"]
                try:
                    message = json.loads(text_msg)
                    status = message.get("status")
                    event = message.get("event")
                    
                    if status == "connected" or event == "connect" or "channel_uuid" in message:
                        print(f"SIP Metadata/Connect frame received: {text_msg}")
                        session_id = message.get("channel_uuid") or message.get("uuid") or str(uuid.uuid4())
                        
                        if not c_num:
                            c_num = message.get("caller_number") or message.get("caller_id") or message.get("from")
                        if not t_num:
                            t_num = message.get("to_number") or message.get("to")
                            
                        if not session_id:
                            session_id = str(uuid.uuid4())
                            
                        if not call_start_time:
                            call_start_time = time.time()
                            
                        db = SessionLocal()
                        try:
                            restaurants = resolve_restaurants_for_call(t_num, db)
                            if not restaurants:
                                print(f"WARNING: resolve_restaurants_for_call('{t_num}') returned zero matches on SIP call start.")
                            for r in restaurants:
                                new_log = models.ChatHistory(
                                    session_id=session_id,
                                    restaurant_id=r.id,
                                    caller_number=c_num,
                                    chat_data=[],
                                    response_time=0.0,
                                    status="in_progress",
                                    duration_seconds=None,
                                    recording_url=None,
                                    transport="sip"
                                )
                                db.add(new_log)
                            db.commit()
                            print(f"Created initial call_logs rows for {len(restaurants)} restaurants with transport='sip'.")
                        except Exception as db_e:
                            print(f"Error creating call_logs rows on SIP start: {db_e}")
                            db.rollback()
                        finally:
                            db.close()
                            
                        if not loop_thread_started:
                            loop_thread = threading.Thread(target=conversation_loop, daemon=True)
                            loop_thread.start()
                            loop_thread_started = True
                            
                    elif status == "disconnected" or event == "disconnect":
                        print("SIP Stream disconnect event received.")
                        break
                except json.JSONDecodeError:
                    print(f"Received raw text frame (ignored): {text_msg}")
                    
            # Handle binary PCM audio frame
            elif "bytes" in data:
                pcm_bytes = data["bytes"]
                
                if not loop_thread_started:
                    if not session_id:
                        session_id = str(uuid.uuid4())
                    if not call_start_time:
                        call_start_time = time.time()
                        
                    db = SessionLocal()
                    try:
                        restaurants = resolve_restaurants_for_call(t_num, db)
                        for r in restaurants:
                            new_log = models.ChatHistory(
                                session_id=session_id,
                                restaurant_id=r.id,
                                caller_number=c_num,
                                chat_data=[],
                                response_time=0.0,
                                status="in_progress",
                                duration_seconds=None,
                                recording_url=None,
                                transport="sip"
                            )
                            db.add(new_log)
                        db.commit()
                        print(f"Lazy started SIP call_logs with transport='sip' for {len(restaurants)} restaurants.")
                    except Exception as db_e:
                        print(f"Error creating call_logs rows on lazy start: {db_e}")
                        db.rollback()
                    finally:
                        db.close()
                        
                    loop_thread = threading.Thread(target=conversation_loop, daemon=True)
                    loop_thread.start()
                    loop_thread_started = True
                
                pcm_bytes_16k = decode_sip_to_stt(pcm_bytes, sip_sample_rate)
                if listener.listening:
                    input_queue.put(pcm_bytes_16k)
                    
    except WebSocketDisconnect:
        print("SIP Media Stream WebSocket Disconnected cleanly.")
    except Exception as e:
        print(f"SIP Media Stream error: {e}")
    finally:
        print("Closing SIP Media Stream connection.")
        output_queue.put(None)
        listener.stop_call()
        player.stop()
        
        if call_start_time is not None:
            call_end_time = time.time()
            duration_seconds = int(call_end_time - call_start_time)
            duration_minutes = duration_seconds / 60.0
            print(f"SIP Call ended. Duration: {duration_seconds} seconds ({duration_minutes:.2f} minutes).")
            
            db = SessionLocal()
            try:
                restaurants = resolve_restaurants_for_call(t_num, db)
                
                for r in restaurants:
                    r.used_minutes += duration_minutes
                    if r.used_minutes >= r.assigned_minutes:
                        r.is_suspended = True
                        print(f"Restaurant '{r.name}' has exceeded its quota ({r.used_minutes:.2f}/{r.assigned_minutes} minutes). Auto-suspending.")
                
                if session_id:
                    user_spoke = False
                    if chatbot_messages:
                        user_spoke = any(msg.get("role") == "user" for msg in chatbot_messages)
                    
                    final_status = call_status
                    if final_status == "completed" and not user_spoke:
                        final_status = "missed"
                        
                    logs = db.query(models.ChatHistory).filter(models.ChatHistory.session_id == session_id).all()
                    for log in logs:
                        log.chat_data = chatbot_messages
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
                print(f"Updated call_logs and minutes for {len(restaurants)} restaurants over SIP.")
            except Exception as db_err:
                print(f"Error updating SIP call logs/minutes: {db_err}")
                db.rollback()
            finally:
                db.close()

        try:
            await websocket.close()
        except RuntimeError:
            pass
