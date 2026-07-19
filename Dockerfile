FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install required system dependencies
# - python3.11 and python3-pip
# - build-essential
# - portaudio19-dev (for PyAudio)
# - ffmpeg (for audio processing)
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Copy requirements files
COPY requirements-gpu.txt .
COPY knowledge_rag/requirements.txt ./knowledge_rag/requirements.txt

# Install python dependencies using the CUDA 12.1 index
RUN pip install --no-cache-dir -r requirements-gpu.txt -r knowledge_rag/requirements.txt

# Copy application source code
COPY . .

EXPOSE 8000

# Start Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
