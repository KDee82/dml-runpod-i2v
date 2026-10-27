# DML Motion Studio — Runpod serverless image-to-video worker
# Lean image: torch wheels carry their own CUDA runtime, so no CUDA base needed.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    MODEL_ID=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    FACE_RESTORE=1 \
    GFPGAN_MODEL_PATH=/opt/gfpgan/GFPGANv1.4.pth \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
        "diffusers>=0.35.0" \
        "transformers>=4.46.0" \
        "accelerate>=0.34.0" \
        sentencepiece protobuf ftfy imageio imageio-ffmpeg \
        "huggingface_hub[hf_transfer]" safetensors pillow numpy \
        runpod

# --- face restoration ------------------------------------------------------
# GFPGAN repairs faces after generation; basicsr (its dependency) builds
# against the installed torch, so it needs --no-build-isolation.
RUN pip install "numpy<2" "opencv-python-headless==4.10.0.84" && \
    pip install --no-build-isolation "basicsr==1.4.2" && \
    pip install "gfpgan==1.3.8" "facexlib==0.3.0"

# Bake the model + detector weights so a cold start never downloads them.
RUN mkdir -p /opt/gfpgan && \
    curl -fsSL -o /opt/gfpgan/GFPGANv1.4.pth \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth && \
    python - <<'PY'
import sys, types
import torchvision.transforms.functional as F

# basicsr imports a module torchvision dropped in 0.17; shim it for the warmup.
shim = types.ModuleType("torchvision.transforms.functional_tensor")
shim.rgb_to_grayscale = F.rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = shim

from gfpgan import GFPGANer
GFPGANer(model_path="/opt/gfpgan/GFPGANv1.4.pth", upscale=2, arch="clean",
         channel_multiplier=2, bg_upsampler=None)
print("face restore weights cached")
PY

WORKDIR /app
COPY handler.py /app/handler.py

# Fail the build here rather than at the first render.
RUN python -c "import ast; ast.parse(open('/app/handler.py').read()); print('handler.py parses')"

CMD ["python3", "-u", "/app/handler.py"]
