from .base import BaseChatbot as BaseChatbot
from .llm_EC2 import Chatbot_LLM as Chatbot_LLM

# Optional: llama_cpp is a heavy native dependency not always installed
try:
    from .llm_llama import Chatbot_llama as Chatbot_llama
except ImportError:
    pass

# from .llm_hf import Chatbot as Chatbot_hf