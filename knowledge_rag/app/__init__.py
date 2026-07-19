import os
from dotenv import load_dotenv

# Load .env from the knowledge_rag directory explicitly, overriding any vars
# already inherited from the parent process (e.g. when auto-started by app.py).
_this_dir = os.path.dirname(os.path.abspath(__file__))
_rag_root = os.path.abspath(os.path.join(_this_dir, ".."))
load_dotenv(os.path.join(_rag_root, ".env"), override=True)

# Resolve the root directory of the repository (d:\AIColdCaller)
# to store all cache files on the D drive under the .cache folder.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "../.."))
_cache_dir = os.path.join(_project_root, ".cache")

# Unconditionally force Hugging Face and Torch cache directories
# to be located in the project root's D drive directory.
os.environ["HF_HOME"] = os.path.join(_cache_dir, "huggingface")
os.environ["HF_HUB_CACHE"] = os.path.join(_cache_dir, "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(_cache_dir, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(_cache_dir, "torch")
