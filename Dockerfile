# DML Motion Studio — Runpod serverless image-to-video worker
# Real GPU rendering (Wan 2.2 TI2V-5B) speaking the studio's existing contract.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu121 || true

RUN pip install --no-cache-dir \
        "diffusers>=0.35.0" \
        "transformers>=4.46.0" \
        accelerate \
        sentencepiece \
        protobuf \
        ftfy \
        imageio \
        imageio-ffmpeg \
        "huggingface_hub[hf_transfer]" \
        safetensors \
        pillow \
        numpy

WORKDIR /app
COPY handler.py /app/handler.py

# Pre-pull the *config* only; weights stream to the network volume on first job
# so the image stays small enough to build quickly.
RUN python - <<'PY' || true
import os
os.environ.setdefault("HF_HOME", "/tmp/hf")
from huggingface_hub import snapshot_download
snapshot_download("Wan-AI/Wan2.2-TI2V-5B-Diffusers", allow_patterns=["*.json", "*.txt", "*.md", "**/*.json"])
PY

CMD ["python3", "-u", "/app/handler.py"]
