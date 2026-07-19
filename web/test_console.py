import os
import json
import queue
import asyncio
import threading
import time
import torch
import base64
import numpy as np
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from openvoicechat.stt.stt_faster_whisper import Ear_faster_whisper as Ear
from openvoicechat.llm.llm_ollama_rag import Chatbot_OllamaRAG as Chatbot

from sql.database import SessionLocal
from sql import models
from telephony.legacy import FallbackMouth
from telephony.twilio_routes import resolve_restaurants_for_call

router = APIRouter()

class TestListener:
    def __init__(self, input_queue):
        self.input_queue = input_queue
        self.listening = True
        self.call_active = True
        self.CHUNK = 320
        self.RATE = 16000

    def read(self, chunk_size):
        if not self.call_active:
            raise EOFError("Call disconnected")
        chunk = self.input_queue.get()
        if chunk is None or not self.call_active:
            raise EOFError("Call disconnected")
        return chunk

    def make_stream(self):
        if not self.call_active:
            raise EOFError("Call disconnected")
        self.listening = True
        self.input_queue.queue.clear()
        return self

    def close(self):
        pass
        
    def stop_call(self):
        self.call_active = False
        self.listening = False
        self.input_queue.put(None)

class TestPlayer:
    def __init__(self, websocket, event_loop):
        self.websocket = websocket
        self.event_loop = event_loop
        self.playing = False
        self.interrupted = False

    def play(self, audio_array, samplerate):
        self.playing = True
        self.interrupted = False
        
        # Convert numpy array to int16 bytes
        pcm_bytes = audio_array.astype(np.int16).tobytes()
        base64_payload = base64.b64encode(pcm_bytes).decode("utf-8")
        
        # Send to browser via websocket
        asyncio.run_coroutine_threadsafe(
            self.websocket.send_json({
                "event": "media",
                "media": {
                    "payload": base64_payload,
                    "sampleRate": samplerate
                }
            }),
            self.event_loop
        )

    def stop(self):
        self.playing = False
        self.interrupted = True
        # Send clear event to browser
        asyncio.run_coroutine_threadsafe(
            self.websocket.send_json({
                "event": "clear"
            }),
            self.event_loop
        )
        
    def wait(self):
        pass


@router.get("/test-call", response_class=HTMLResponse)
async def test_call_page(request: Request):
    db = SessionLocal()
    try:
        restaurants = db.query(models.Restaurant).filter(models.Restaurant.is_suspended == False).all()
        if not restaurants:
            restaurants = db.query(models.Restaurant).all()
    finally:
        db.close()

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Voice Agent Web Tester</title>
        <!-- Tailwind CSS -->
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        colors: {
                            darkBg: '#0b0f19',
                            cardBg: '#111827',
                            accentPurp: '#6366f1',
                        }
                    }
                }
            }
        </script>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Outfit', sans-serif;
                background-color: #0b0f19;
            }
            .glow-btn:hover {
                box-shadow: 0 0 15px rgba(99, 102, 241, 0.6);
            }
            .glow-btn-green:hover {
                box-shadow: 0 0 15px rgba(34, 197, 94, 0.6);
            }
            .glow-btn-red:hover {
                box-shadow: 0 0 15px rgba(239, 68, 68, 0.6);
            }
        </style>
    </head>
    <body class="text-slate-100 min-h-screen flex flex-col items-center justify-center p-4">
        <div class="max-w-4xl w-full grid grid-cols-1 md:grid-cols-3 gap-6">
            <!-- Configuration Side Panel -->
            <div class="bg-cardBg p-6 rounded-2xl border border-slate-800 shadow-2xl flex flex-col justify-between">
                <div>
                    <div class="flex items-center gap-2 mb-6">
                        <span class="w-3 h-3 rounded-full bg-indigo-500 animate-ping"></span>
                        <h2 class="text-xl font-bold text-slate-100 tracking-wide">Call Simulation</h2>
                    </div>
                    
                    <div class="space-y-4">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Select Restaurant</label>
                            <select id="restaurantSelect" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 transition-colors" onchange="updateToNumber()">
                                <option value="">-- Select Restaurant --</option>
    """
    
    for r in restaurants:
        html_content += f'<option value="{r.order_phone_number}">{r.name} ({r.order_phone_number})</option>'

    html_content += """
                            </select>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Recipient (To Number)</label>
                            <input id="toNum" type="text" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-400 focus:outline-none cursor-not-allowed" placeholder="Select a restaurant" readonly>
                        </div>

                        <div>
                            <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Caller (From Number)</label>
                            <input id="fromNum" type="text" inputmode="tel" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-4 py-3 text-slate-100 focus:outline-none focus:border-indigo-500 transition-colors" placeholder="+198765432" value="+198765432">
                        </div>
                    </div>
                </div>

                <div class="mt-6 space-y-4">
                    <button id="callBtn" onclick="toggleCall()" class="w-full bg-green-600 text-white font-semibold py-3 px-6 rounded-xl glow-btn-green transition-all duration-300 flex items-center justify-center gap-2">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.94.725l.548 2.2a1 1 0 01-.321.988l-1.305.98a10.582 10.582 0 004.872 4.872l.98-1.305a1 1 0 01.988-.321l2.2.548a1 1 0 01.725.94V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path></svg>
                        Start Call
                    </button>
                    <div id="statusIndicator" class="text-center text-sm font-semibold text-slate-400">Status: Disconnected</div>
                </div>
            </div>

            <!-- Main Live Interface & Transcript -->
            <div class="md:col-span-2 bg-cardBg rounded-2xl border border-slate-800 shadow-2xl overflow-hidden flex flex-col h-[550px]">
                <!-- Header -->
                <div class="px-6 py-4 border-b border-slate-800 flex justify-between items-center bg-slate-900 bg-opacity-50">
                    <h1 class="text-lg font-bold text-slate-200">Interactive Call Session</h1>
                    <div id="callTimer" class="text-sm font-semibold text-indigo-400 hidden">00:00</div>
                </div>

                <!-- Waveform Canvas -->
                <div class="h-20 bg-slate-950 flex items-center justify-center border-b border-slate-900 relative">
                    <canvas id="visualizer" class="w-full h-full"></canvas>
                    <div id="wavePlaceholder" class="absolute text-xs text-slate-500 font-medium">Visualizer inactive. Call to start.</div>
                </div>

                <!-- Chat Transcript -->
                <div id="transcriptBox" class="flex-1 p-6 overflow-y-auto space-y-4 bg-slate-900 bg-opacity-20">
                    <!-- Transcript bubbles populate here -->
                    <div class="flex items-center justify-center h-full text-slate-500 text-sm">
                        No active call conversation. Select a restaurant and click "Start Call" to begin.
                    </div>
                </div>
            </div>
        </div>

        <!-- Event Log Panel -->
        <div class="max-w-4xl w-full mt-6 bg-cardBg p-4 rounded-xl border border-slate-800 shadow-xl">
            <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">System Events</h3>
            <div id="eventLogs" class="font-mono text-xs text-indigo-300 h-28 overflow-y-auto space-y-1 bg-slate-950 p-3 rounded-lg border border-slate-900">
                [System] Loaded AI Web Tester. Ready to simulate call.
            </div>
        </div>

        <script>
            let socket;
            let audioCtx;
            let micSource;
            let scriptNode;
            let isCalling = false;
            let nextStartTime = 0;
            let activeSources = [];
            let callTimerInterval;
            let callStartTime;
            
            let analyser;
            let dataArray;
            let animationFrameId;

            function updateToNumber() {
                const select = document.getElementById('restaurantSelect');
                document.getElementById('toNum').value = select.value;
            }

            function logEvent(txt) {
                const box = document.getElementById('eventLogs');
                const div = document.createElement('div');
                div.textContent = `[${new Date().toLocaleTimeString()}] ${txt}`;
                box.appendChild(div);
                box.scrollTop = box.scrollHeight;
            }

            function updateUI(calling) {
                const callBtn = document.getElementById('callBtn');
                const status = document.getElementById('statusIndicator');
                const timer = document.getElementById('callTimer');
                const placeholder = document.getElementById('wavePlaceholder');

                if (calling) {
                    callBtn.innerHTML = `
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636"></path></svg>
                        Hang Up
                    `;
                    callBtn.className = "w-full bg-red-600 text-white font-semibold py-3 px-6 rounded-xl glow-btn-red transition-all duration-300 flex items-center justify-center gap-2";
                    status.textContent = "Status: Call Connected";
                    status.className = "text-center text-sm font-semibold text-green-500";
                    timer.classList.remove('hidden');
                    placeholder.classList.add('hidden');
                    
                    // Clear chat
                    document.getElementById('transcriptBox').innerHTML = '';
                } else {
                    callBtn.innerHTML = `
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.94.725l.548 2.2a1 1 0 01-.321.988l-1.305.98a10.582 10.582 0 004.872 4.872l.98-1.305a1 1 0 01.988-.321l2.2.548a1 1 0 01.725.94V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z"></path></svg>
                        Start Call
                    `;
                    callBtn.className = "w-full bg-green-600 text-white font-semibold py-3 px-6 rounded-xl glow-btn-green transition-all duration-300 flex items-center justify-center gap-2";
                    status.textContent = "Status: Disconnected";
                    status.className = "text-center text-sm font-semibold text-slate-400";
                    timer.classList.add('hidden');
                    placeholder.classList.remove('hidden');
                }
            }

            function startTimer() {
                callStartTime = Date.now();
                const timerEl = document.getElementById('callTimer');
                timerEl.textContent = "00:00";
                callTimerInterval = setInterval(() => {
                    const diff = Date.now() - callStartTime;
                    const secs = Math.floor(diff / 1000) % 60;
                    const mins = Math.floor(diff / 60000);
                    timerEl.textContent = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
                }, 1000);
            }

            function stopTimer() {
                clearInterval(callTimerInterval);
            }

            function toggleCall() {
                if (isCalling) {
                    stopCall();
                } else {
                    startCall();
                }
            }

            async function startCall() {
                const toNum = document.getElementById('toNum').value.trim();
                const fromNum = document.getElementById('fromNum').value.trim();

                if (!toNum) {
                    alert("Please select a restaurant or enter a 'To' number.");
                    return;
                }

                logEvent("Initiating connection...");
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/test-media-stream-socket?caller_number=${encodeURIComponent(fromNum)}&to_number=${encodeURIComponent(toNum)}`;
                
                socket = new WebSocket(wsUrl);
                socket.binaryType = 'arraybuffer';

                socket.onopen = async () => {
                    logEvent("WebSocket connected. Starting media transmission...");
                    
                    socket.send(JSON.stringify({
                        event: "start",
                        start: {
                            streamSid: "test_web_" + Math.random().toString(36).substring(7)
                        }
                    }));

                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                        micSource = audioCtx.createMediaStreamSource(stream);
                        
                        scriptNode = audioCtx.createScriptProcessor(2048, 1, 1);
                        scriptNode.onaudioprocess = (e) => {
                            if (!isCalling) return;
                            const inputData = e.inputBuffer.getChannelData(0);
                            const pcmBuffer = floatTo16BitPCM(inputData);
                            if (socket.readyState === WebSocket.OPEN) {
                                socket.send(pcmBuffer);
                            }
                        };

                        micSource.connect(scriptNode);
                        scriptNode.connect(audioCtx.destination);
                        
                        isCalling = true;
                        updateUI(true);
                        startTimer();
                        startVisualizer();
                        logEvent("Call started successfully.");
                    } catch (err) {
                        logEvent("Error obtaining mic stream: " + err.message);
                        stopCall();
                    }
                };

                socket.onmessage = (e) => {
                    if (typeof e.data === 'string') {
                        const msg = JSON.parse(e.data);
                        if (msg.event === 'media') {
                            playAudioChunk(msg.media.payload, msg.media.sampleRate);
                        } else if (msg.event === 'clear') {
                            clearAudio();
                        } else if (msg.event === 'transcript') {
                            addTranscriptBubble(msg.role, msg.content);
                        }
                    }
                };

                socket.onclose = () => {
                    logEvent("Connection closed by server.");
                    stopCall();
                };

                socket.onerror = (err) => {
                    logEvent("WebSocket encountered error: " + err.message);
                };
            }

            function stopCall() {
                if (socket) {
                    if (socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify({ event: "stop" }));
                    }
                    socket.close();
                }

                isCalling = false;
                stopTimer();
                clearAudio();
                updateUI(false);
                
                if (scriptNode) {
                    scriptNode.disconnect();
                    scriptNode = null;
                }
                if (micSource) {
                    micSource.disconnect();
                    micSource = null;
                }
                if (audioCtx) {
                    audioCtx.close();
                    audioCtx = null;
                }
                if (animationFrameId) {
                    cancelAnimationFrame(animationFrameId);
                }
                
                const canvas = document.getElementById('visualizer');
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }

            function floatTo16BitPCM(input) {
                const output = new Int16Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    let s = Math.max(-1, Math.min(1, input[i]));
                    output[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                return output.buffer;
            }

            function playAudioChunk(base64Payload, sampleRate) {
                if (!audioCtx) return;

                const binaryString = window.atob(base64Payload);
                const len = binaryString.length;
                const bytes = new Uint8Array(len);
                for (let i = 0; i < len; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                const int16Data = new Int16Array(bytes.buffer);
                const float32Data = new Float32Array(int16Data.length);
                for (let i = 0; i < int16Data.length; i++) {
                    float32Data[i] = int16Data[i] / 32768.0;
                }

                const audioBuffer = audioCtx.createBuffer(1, float32Data.length, sampleRate);
                audioBuffer.copyToChannel(float32Data, 0);

                const source = audioCtx.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioCtx.destination);

                const now = audioCtx.currentTime;
                if (nextStartTime < now) {
                    nextStartTime = now;
                }
                source.start(nextStartTime);
                nextStartTime += audioBuffer.duration;

                activeSources.push(source);
                source.onended = () => {
                    const idx = activeSources.indexOf(source);
                    if (idx > -1) activeSources.splice(idx, 1);
                };
            }

            function clearAudio() {
                logEvent("Interrupted playback (barge-in detected).");
                activeSources.forEach(src => {
                    try { src.stop(); } catch(e) {}
                });
                activeSources = [];
                nextStartTime = 0;
            }

            function addTranscriptBubble(role, content) {
                const box = document.getElementById('transcriptBox');
                
                const wrapper = document.createElement('div');
                wrapper.className = `flex ${role === 'user' ? 'justify-end' : 'justify-start'}`;

                const bubble = document.createElement('div');
                bubble.className = `max-w-md rounded-2xl px-4 py-3 text-sm shadow-md leading-relaxed ${
                    role === 'user' 
                        ? 'bg-indigo-600 text-white rounded-tr-none' 
                        : 'bg-slate-800 text-slate-100 border border-slate-700 rounded-tl-none'
                }`;
                
                const label = document.createElement('div');
                label.className = `text-[10px] uppercase font-bold tracking-wider mb-1 ${
                    role === 'user' ? 'text-indigo-200 text-right' : 'text-slate-400'
                }`;
                label.textContent = role === 'user' ? 'Customer' : 'AI Assistant';

                const text = document.createElement('div');
                text.textContent = content;

                bubble.appendChild(label);
                bubble.appendChild(text);
                wrapper.appendChild(bubble);
                box.appendChild(wrapper);
                
                box.scrollTop = box.scrollHeight;
            }

            // Restrict fromNum input to only digits and '+'
            document.getElementById('fromNum').addEventListener('input', (e) => {
                e.target.value = e.target.value.replace(/[^\d+]/g, '');
            });

            function startVisualizer() {
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 64;
                const bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);
                
                micSource.connect(analyser);

                const canvas = document.getElementById('visualizer');
                const ctx = canvas.getContext('2d');

                function draw() {
                    if (!isCalling) return;
                    animationFrameId = requestAnimationFrame(draw);
                    analyser.getByteFrequencyData(dataArray);

                    ctx.fillStyle = '#0f172a';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);

                    const barWidth = (canvas.width / bufferLength) * 2;
                    let barHeight;
                    let x = 0;

                    for (let i = 0; i < bufferLength; i++) {
                        barHeight = dataArray[i] / 4;
                        ctx.fillStyle = `rgb(99, 102, 241)`;
                        ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                        x += barWidth;
                    }
                }
                draw();
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.websocket("/test-media-stream-socket")
async def test_media_stream_socket(
    websocket: WebSocket,
    caller_number: Optional[str] = None,
    to_number: Optional[str] = None
):
    await websocket.accept()
    print(f"[Test Call Web] connected (Caller: {caller_number}, To: {to_number})")

    input_queue = queue.Queue()
    listener = TestListener(input_queue)
    event_loop = asyncio.get_running_loop()
    player = TestPlayer(websocket, event_loop)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    stream_sid = None
    call_start_time = None
    chatbot_messages = []
    call_status = "completed"

    def conversation_loop():
        nonlocal chatbot_messages, call_status
        print("[Test Call Web] Initializing AI models in background thread...")
        
        ear = Ear(
            model_size=os.getenv("STT_MODEL_SIZE", "small"),
            device=device,
            compute_type=compute_type,
            silence_seconds=2.0,
            listener=listener,
            stream=True,
            player=player
        )
        
        db = SessionLocal()
        sys_prompt = "You are the friendly AI voice-agent taking orders. Speak in Roman Urdu/Urdu-English mix."
        greeting = "Assalam-o-Alaikum! Aaj aap kya order karna pasand farmayenge?"
        voice_engine = "urdu-female"
        
        try:
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
                    print(f"[Test Call Web] Error fetching greeting from RAG: {e}")
                    greeting = f"Assalam-o-Alaikum! {restaurant.name} se AI assistant bol rahi hoon. Aaj aap kya order karna pasand farmayenge?"
        except Exception as e:
            print(f"[Test Call Web] Error loading agent config: {e}")
        finally:
            db.close()

        mouth = FallbackMouth(player=player, device=device, voice_engine=voice_engine)

        rag_business_id = os.getenv("RAG_BUSINESS_ID", "")
        if not rag_business_id:
            try:
                db2 = SessionLocal()
                rest = db2.query(models.Restaurant).filter(models.Restaurant.order_phone_number == to_number).first()
                if rest:
                    rag_business_id = str(rest.id)
                db2.close()
            except Exception:
                pass

        chatbot = Chatbot(sys_prompt=sys_prompt, business_id=rag_business_id)
        
        time.sleep(0.5)
        
        print("[Test Call Web] Speaking greeting...")
        mouth.say_text(greeting)
        
        asyncio.run_coroutine_threadsafe(
            websocket.send_json({
                "event": "transcript",
                "role": "assistant",
                "content": greeting
            }),
            event_loop
        )
        
        chatbot.messages.append({"role": "assistant", "content": greeting})
        chatbot_messages = chatbot.messages
        
        while True:
            try:
                if not listener.call_active:
                    raise EOFError("Call disconnected")
                print("[Test Call Web] Listening for user...")
                user_text = ear.listen()
                if not listener.call_active:
                    raise EOFError("Call disconnected")
                
                if not user_text or not user_text.strip():
                    continue
                
                print(f"[Test Call Web] User: {user_text.strip()}")
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({
                        "event": "transcript",
                        "role": "user",
                        "content": user_text.strip()
                    }),
                    event_loop
                )
                
                llm_response = ""
                for chunk in chatbot.run(user_text):
                    llm_response += chunk
                
                print(f"[Test Call Web] Agent: {llm_response}")
                chatbot.post_process(llm_response)
                chatbot_messages = chatbot.messages
                
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({
                        "event": "transcript",
                        "role": "assistant",
                        "content": llm_response
                    }),
                    event_loop
                )
                
                mouth.say_text(llm_response)
            except EOFError:
                print("[Test Call Web] Call ended gracefully.")
                call_status = "completed"
                break
            except Exception as e:
                print(f"[Test Call Web] Error in loop: {e}")
                call_status = "failed"
                break

    loop_thread = threading.Thread(target=conversation_loop, daemon=True)

    try:
        while True:
            result = await websocket.receive()
            if "bytes" in result:
                # Raw binary 16kHz 16-bit PCM chunk from browser
                pcm_bytes = result["bytes"]
                if listener.listening:
                    input_queue.put(pcm_bytes)
            elif "text" in result:
                message = json.loads(result["text"])
                event = message.get("event")
                if event == "start":
                    stream_sid = message["start"]["streamSid"]
                    from telephony.registry import active_connections
                    active_connections[stream_sid] = websocket
                    call_start_time = time.time()
                    
                    db = SessionLocal()
                    try:
                        restaurants = resolve_restaurants_for_call(to_number, db)
                        for r in restaurants:
                            new_log = models.ChatHistory(
                                session_id=stream_sid,
                                restaurant_id=r.id,
                                caller_number=caller_number,
                                chat_data=[],
                                response_time=0.0,
                                status="in_progress",
                                duration_seconds=None,
                                recording_url=None,
                                transport="test_web"
                            )
                            db.add(new_log)
                        db.commit()
                        print(f"[Test Call Web] Log rows created for {len(restaurants)} restaurants.")
                    except Exception as db_e:
                        print(f"[Test Call Web] Error logging call start: {db_e}")
                        db.rollback()
                    finally:
                        db.close()
                        
                    loop_thread.start()
                elif event == "stop":
                    break
    except WebSocketDisconnect:
        print("[Test Call Web] WS Disconnected.")
    except Exception as e:
        print(f"[Test Call Web] WS Error: {e}")
    finally:
        from telephony.registry import active_connections
        if stream_sid:
            active_connections.pop(stream_sid, None)
        listener.stop_call()
        player.stop()
        
        if call_start_time is not None:
            duration_seconds = int(time.time() - call_start_time)
            duration_minutes = duration_seconds / 60.0
            
            db = SessionLocal()
            try:
                restaurants = resolve_restaurants_for_call(to_number, db)
                for r in restaurants:
                    r.used_minutes += duration_minutes
                    if r.used_minutes >= r.assigned_minutes:
                        r.is_suspended = True
                        print(f"[Test Call Web] Quota exceeded. Auto-suspending restaurant '{r.name}'")
                
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
                                print(f"[Test Call Web] Error auto-extracting order from call transcript: {parse_err}")
                
                db.commit()
                print(f"[Test Call Web] Call stats and logs updated.")
            except Exception as db_err:
                print(f"[Test Call Web] Error updating call stats/logs: {db_err}")
                db.rollback()
            finally:
                db.close()
        
        try:
            await websocket.close()
        except RuntimeError:
            pass
