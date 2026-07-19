import os
import secrets
import urllib.request
import sys

def create_env_file():
    env_path = ".env"
    example_path = ".env.example"
    
    if os.path.exists(env_path):
        print("[*] .env file already exists. Skipping creation.")
        return
        
    if not os.path.exists(example_path):
        print("[-] .env.example not found. Cannot create .env.")
        return
        
    print("[*] Creating .env file from template...")
    with open(example_path, "r") as f:
        content = f.read()
        
    # Generate secure random secrets
    content = content.replace("replace-with-long-random-secret", secrets.token_hex(32), 1) # SESSION_MIDDLEWARE_SECRET_KEY
    content = content.replace("replace-with-long-random-secret", secrets.token_hex(32), 1) # JWT_SECRET_KEY
    content = content.replace("replace-with-long-random-secret", secrets.token_hex(32), 1) # JWT_REFRESH_SECRET_KEY
    
    with open(env_path, "w") as f:
        f.write(content)
    print("[+] Created .env file with secure session and JWT secret keys.")
    print("[!] Action Required: Open the .env file and set your LLM_EC2_KEY (OpenAI API Key).")

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent:.1f}% ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading: {read_so_far / (1024*1024):.2f} MB")
        sys.stdout.flush()

def download_piper_model():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "en_US-ryan-high.onnx")
    config_path = os.path.join(models_dir, "en_US-ryan-high.onnx.json")
    
    model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx"
    config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json"
    
    if not os.path.exists(model_path):
        print(f"\n[*] Downloading Piper Voice Model (en_US-ryan-high.onnx)...")
        try:
            urllib.request.urlretrieve(model_url, model_path, download_progress)
            print("\n[+] Model downloaded successfully.")
        except Exception as e:
            print(f"\n[-] Failed to download voice model: {e}")
            print("Please download it manually from:")
            print(model_url)
            print(f"and place it at: {model_path}")
    else:
        print("[*] Piper Voice Model already exists.")
        
    if not os.path.exists(config_path):
        print(f"\n[*] Downloading Piper Voice Config (en_US-ryan-high.onnx.json)...")
        try:
            urllib.request.urlretrieve(config_url, config_path, download_progress)
            print("\n[+] Config downloaded successfully.")
        except Exception as e:
            print(f"\n[-] Failed to download config: {e}")
            print("Please download it manually from:")
            print(config_url)
            print(f"and place it at: {config_path}")
    else:
        print("[*] Piper Voice Config already exists.")

if __name__ == "__main__":
    print("=== AI Voice Agent Platform Developer Setup ===")
    create_env_file()
    download_piper_model()
    print("\n[+] Setup complete! Follow the developer guide to start your application.")
