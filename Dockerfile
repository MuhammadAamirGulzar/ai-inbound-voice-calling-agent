# Streaming deployment image: cloud STT/LLM/TTS, no local models.
# A few hundred MB instead of multi-GB; one instance holds many
# concurrent calls.
#
# The offline legacy pipeline (local whisper/ollama/piper) is a
# development fallback, not something this image ships — install
# requirements-cpu.txt on a host with the models for that.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-streaming.txt .
RUN pip install --no-cache-dir -r requirements-streaming.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    RAG_AUTOSTART=false \
    VOICE_PIPELINE=streaming

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
