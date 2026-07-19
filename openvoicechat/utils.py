import multiprocessing
import os
import queue
import random
import threading
import time
import librosa
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

TIMING = int(os.environ.get("TIMING", 0))
LOGGING = int(os.environ.get("LOGGING", 0))

timing_path = os.environ.get("TIMING_PATH", "times.csv")


def log_to_file(file_path, text):
    with open(file_path, "a") as file:
        file.write(text + "\n")

        
def clean_text_for_tts(text):
    """
    Clean text from markdown and special characters before sending to TTS
    """
    import re
    # Remove markdown formatting
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold text
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)    # Italic text
    
    # Remove bullet points and numbering
    cleaned = re.sub(r'^\s*[\*\-•]\s+', '', cleaned, flags=re.MULTILINE)  # Bullet points
    cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)    # Numbered lists
    
    # Replace special characters commonly found in markdown
    cleaned = cleaned.replace('**', '')
    cleaned = cleaned.replace('*', '')
    cleaned = cleaned.replace('`', '')
    cleaned = cleaned.replace('#', '')
    cleaned = cleaned.replace('>', '')
    
    return cleaned

def run_chat_original(
    mouth,
    ear,
    chatbot,
    minibot_args=None,
    verbose=True,
    stopping_criteria=lambda x: False,
    starting_message=True,
    logging_path="chat_log.txt",
):
    """
    Runs a chat session between a user and a bot.

    Parameters: mouth (object): An object responsible for the bot's speech output. ear (object): An object
    responsible for listening to the user's input. chatbot (object): An object responsible for generating the bot's
    responses. verbose (bool, optional): If True, prints the user's input and the bot's responses. Defaults to True.
    stopping_criteria (function, optional): A function that determines when the chat should stop. It takes the bot's
    response as input and returns a boolean. Defaults to a function that always returns False.

    The function works by continuously listening to the user's input and generating the bot's responses in separate
    threads. If the user interrupts the bot's speech, the remaining part of the bot's response is saved and prepended
    to the user's next input. The chat stops when the stopping_criteria function returns True for a bot's response.
    """
    if TIMING:
        pd.DataFrame(columns=["Model", "Time Taken"]).to_csv(timing_path, index=False)

    if starting_message:
        agent_name = minibot_args.get("agent_name")
        organization_name = minibot_args.get("organization_name")
        initial_message = random.choice(
            [
                f"Hello, I am {agent_name} from {organization_name}. How can I help you?",
                f"Hello! This is {agent_name} from {organization_name}. How can I be of assistance?",
                f"Hi! This is {organization_name}'s representative {agent_name}. What can I do for you?",
                f"Hi! You're speaking with {agent_name} from {organization_name}. What can I do for you?",
                f"Hey, there! You're speaking with {agent_name} from {organization_name}. How can I assist you?",
            ]
        )
        mouth.say_text(initial_message)

    pre_interruption_text = ""
    while True:
        user_input = pre_interruption_text + " " + ear.listen()

        if verbose:
            print("USER: ", user_input)
        if LOGGING:
            log_to_file(logging_path, "USER: " + user_input)

        llm_output_queue = queue.Queue()
        interrupt_queue = queue.Queue()
        llm_thread = threading.Thread(
            target=chatbot.generate_response_stream,
            args=(user_input, llm_output_queue, interrupt_queue, minibot_args),
        )
        tts_thread = threading.Thread(
            target=mouth.say_multiple_stream,
            args=(llm_output_queue, ear.interrupt_listen, interrupt_queue),
        )

        llm_thread.start()
        tts_thread.start()

        llm_thread.join()
        tts_thread.join()
        if not interrupt_queue.empty():
            pre_interruption_text = interrupt_queue.get()
        else:
            pre_interruption_text = ""

        res = llm_output_queue.get()
        if stopping_criteria(res):
            break
        if verbose:
            print("BOT: ", res)
        if LOGGING:
            log_to_file(logging_path, "BOT: " + res)


def run_chat(
    mouth,
    ear,
    chatbot,
    minibot_args=None,
    verbose=True,
    stopping_criteria=lambda x: False,
    starting_message=True,
    logging_path="chat_log.txt",
):
    """
    Wrapper function that decides whether to use LangChain or the original implementation.
    """
    use_langchain = True  # Set to False to disable LangChain for regular chats
    
    if use_langchain:
        try:
            run_chat_langchain(mouth, ear, chatbot, minibot_args, verbose, stopping_criteria, starting_message, logging_path)
        except Exception as e:
            print(f"Error in LangChain implementation: {e}")
            print("Falling back to original implementation")
            run_chat_original(mouth, ear, chatbot, minibot_args, verbose, stopping_criteria, starting_message, logging_path)
    else:
        run_chat_original(mouth, ear, chatbot, minibot_args, verbose, stopping_criteria, starting_message, logging_path)


def run_chat_langchain(
    mouth,
    ear,
    chatbot,
    minibot_args=None,
    verbose=True,
    stopping_criteria=lambda x: False,
    starting_message=True,
    logging_path="chat_log.txt",
):
    """
    Enhanced run_chat function that uses LangChain for all conversations.
    This is a direct replacement for the standard run_chat function.
    """
    import os
    import random
    import queue
    import threading
    import pandas as pd
    from langchain_openai import ChatOpenAI
    from langchain.schema.messages import SystemMessage, HumanMessage
    from langchain.memory import ConversationBufferMemory
    
    if TIMING:
        pd.DataFrame(columns=["Model", "Time Taken"]).to_csv(timing_path, index=False)

    # Extract agent and organization info
    agent_name = minibot_args.get("agent_name", "Agent")
    organization_name = minibot_args.get("organization_name", "Our Company")
    
    try:
        # Set up the OpenAI API key
        api_key = os.getenv("LLM_EC2_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        # Get the original system prompt from chatbot
        original_prompt = chatbot.messages[0]['content'] if chatbot.messages else ""

        print("The original prompt is:", original_prompt)


        
        # Create LLM
        llm = ChatOpenAI(
            temperature=0.7,
            api_key=api_key,
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            base_url="https://api.groq.com/openai/v1",
            streaming=True
        )
        
        # Create conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        
        # Initial message
        if starting_message:
            initial_message = random.choice([
                f"Hello, I am {agent_name} from {organization_name}. How can I help you?",
                f"Hello! This is {agent_name} from {organization_name}. How can I be of assistance?",
                f"Hi! This is {organization_name}'s representative {agent_name}. What can I do for you?",
                f"Hi! You're speaking with {agent_name} from {organization_name}. What can I do for you?",
                f"Hey, there! You're speaking with {agent_name} from {organization_name}. How can I assist you?",
            ])
            mouth.say_text(initial_message)

        pre_interruption_text = ""
        
        # Add system message to memory
        memory.chat_memory.add_message(SystemMessage(content=original_prompt))
        
        while True:
            user_input = pre_interruption_text + " " + ear.listen()

            if verbose:
                print("USER: ", user_input)
            if LOGGING:
                log_to_file(logging_path, "USER: " + user_input)
            
            # Process with LangChain
            try:
                # Add user message to memory
                memory.chat_memory.add_message(HumanMessage(content=user_input))
                
                # Get chat history
                messages = memory.chat_memory.messages
                
                # Generate response directly with the LLM (instead of using agent)
                response = llm.invoke(messages)
                agent_response = response.content
                
                # Update chatbot context to maintain state
                chatbot.messages.append({"role": "user", "content": user_input})
                chatbot.messages.append({"role": "assistant", "content": agent_response})
                
                # Add assistant message to memory
                memory.chat_memory.add_ai_message(agent_response)
                
                # Send response to TTS
                mouth.say_text(agent_response)
                
                if verbose:
                    print("BOT: ", agent_response)
                if LOGGING:
                    log_to_file(logging_path, "BOT: " + agent_response)
                
                if stopping_criteria(agent_response):
                    break
                
            except Exception as e:
                print(f"LangChain error: {e}")
                # Fall back to original chatbot implementation
                llm_output_queue = queue.Queue()
                interrupt_queue = queue.Queue()
                llm_thread = threading.Thread(
                    target=chatbot.generate_response_stream,
                    args=(user_input, llm_output_queue, interrupt_queue, minibot_args),
                )
                tts_thread = threading.Thread(
                    target=mouth.say_multiple_stream,
                    args=(llm_output_queue, ear.interrupt_listen, interrupt_queue),
                )

                llm_thread.start()
                tts_thread.start()

                llm_thread.join()
                tts_thread.join()
                if not interrupt_queue.empty():
                    pre_interruption_text = interrupt_queue.get()
                else:
                    pre_interruption_text = ""

                res = llm_output_queue.get()
                if stopping_criteria(res):
                    break
                if verbose:
                    print("BOT: ", res)
                if LOGGING:
                    log_to_file(logging_path, "BOT: " + res)
    
    except Exception as e:
        print(f"Error in LangChain setup: {e}")
        print("Falling back to standard run_chat")
        # Fall back to original run_chat implementation
        run_chat_original(mouth, ear, chatbot, minibot_args, verbose, stopping_criteria, starting_message, logging_path)


class Player_ws:
    def __init__(self, q):
        self.output_queue = q
        self.playing = False
        self._timer_thread = None
        self.interrupted = False

    def play(self, audio_array, samplerate):
        self.playing = True
        self.interrupted = False
        duration = len(audio_array) / samplerate
        if audio_array.dtype == np.int16:
            audio_array = audio_array / (1 << 15)
        audio_array = audio_array.astype(np.float32)
        audio_array = librosa.resample(
            y=audio_array, orig_sr=samplerate, target_sr=44100
        )
        audio_array = audio_array.tobytes()
        
        if not self.interrupted:
            self.output_queue.put(audio_array)
            
        if self._timer_thread is not None:
            if self._timer_thread.is_alive():
                self._timer_thread.terminate()
        self._timer_thread = multiprocessing.Process(
            target=time.sleep, args=(duration,)
        )
        self._timer_thread.start()

    def stop(self):
        self.playing = False
        self.interrupted = True
        self.output_queue.queue.clear()
        self.output_queue.put("stop".encode())
        if self._timer_thread and self._timer_thread.is_alive():
            self._timer_thread.terminate()

    def wait(self):
        if self._timer_thread:
            self._timer_thread.join()
        self.playing = False

        
class Listener_ws:
    def __init__(self, q):
        self.input_queue = q
        self.listening = False
        self.CHUNK = 5945
        self.RATE = 16_000

    def read(self, x):
        data = self.input_queue.get()
        data = np.frombuffer(data, dtype=np.float32)
        data = librosa.resample(y=data, orig_sr=44100, target_sr=16_000)
        data = data * (1 << 15)
        data = data.astype(np.int16)
        data = data.tobytes()
        return data

    def close(self):
        pass

    def make_stream(self):
        self.listening = True
        self.input_queue.queue.clear()
        return self
