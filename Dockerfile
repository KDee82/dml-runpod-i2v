# DML Motion Studio — Runpod serverless image-to-video worker
# Lean image: torch wheels carry their own CUDA runtime, so no CUDA base needed.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
        "diffusers>=0.35.0" \
        "transformers>=4.46.0" \
        "accelerate>=0.34.0" \
        sentencepiece protobuf ftfy imageio imageio-ffmpeg \
        "huggingface_hub[hf_transfer]" safetensors pillow numpy \
        runpod

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
