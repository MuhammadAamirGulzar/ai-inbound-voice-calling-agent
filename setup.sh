#!/bin/bash

# Ensure conda is initialized
eval "$(conda shell.bash hook)"

# Deactivate any currently active conda environment
conda deactivate

# Update and upgrade packages
apt-get update
apt-get upgrade -y

# Install portaudio19-dev
apt install portaudio19-dev -y

# Install CUDA Lib
apt install lshw -y

# Create and activate a conda environment
conda create --name autoengage python=3.10.6 -y
conda activate autoengage

# Install Python dependencies inside the conda environment
pip install -r requirements.txt

# Install pyaudio using conda
conda install -c anaconda pyaudio=0.2.14 -y

# Install Ollama and Llama 3
curl -fsSL https://ollama.com/install.sh | sh
ollama serve & ollama pull mxbai-embed-large
