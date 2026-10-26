"""
Runpod serverless handler — real image-to-video, contract-compatible with
DML Motion Studio's services/api/runpod_worker.py.

IN  : {image_b64, positive_prompt, negative_prompt, seed, duration_seconds,
       aspect_ratio, fps, model}
OUT : {video_b64, meta}   (or {error} on failure — never a silent success)

Model: Wan 2.2 TI2V-5B (Apache-2.0, ~10 GB fp16) — fits a 24 GB RTX 4090 at
480p. The pipeline is loaded once per worker and reused across jobs.
"""
import base64
import io
import os
import tempfile
import time

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
from diffusers.utils import export_to_video
from PIL import Image
import runpod

MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
HF_HOME = os.environ.get("HF_HOME", "/runpod-volume/hf")
os.environ.setdefault("HF_HOME", HF_HOME)

_PIPE = None
_DTYPE = torch.bfloat16


def get_pipe():
    """Load once, reuse. Cold start downloads weights to the network volume."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    print(f"[worker] loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    # TI2V-5B ships a single transformer usable for both T2V and I2V
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=_DTYPE)
    except Exception as e:  # noqa: BLE001 - fall back to text-to-video
        print(f"[worker] I2V loader failed ({e}); trying T2V", flush=True)
        pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=_DTYPE)
    pipe.to("cuda")
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001
        pass
    _PIPE = pipe
    print(f"[worker] ready in {time.time() - t0:.1f}s", flush=True)
    return _PIPE


def _dims(aspect, base=704):
    """Wan likes multiples of 16; keep both edges on a 16 grid."""
    a = (aspect or "16:9").replace(" ", "")
    table = {
        "16:9": (704, 1280), "9:16": (1280, 704), "1:1": (768, 768),
        "4:5": (1280, 1024), "21:9": (544, 1280),
    }
    w, h = table.get(a, (704, 1280))
    return (w // 16) * 16, (h // 16) * 16


def _frames(duration, fps):
    """4n+1 frames (Wan requirement)."""
    n = int(round(float(duration or 5) * float(fps or 24)))
    n = max(17, min(n, 121))
    return ((n - 1) // 4) * 4 + 1


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        b64 = inp.get("image_b64") or ""
        if not b64:
            raise ValueError("image_b64 is required (image-to-video endpoint)")
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        width, height = _dims(inp.get("aspect_ratio"))
        nframes = _frames(inp.get("duration_seconds"), inp.get("fps"))
        seed = int(inp.get("seed") or 0)
        prompt = (inp.get("positive_prompt") or "cinematic motion").strip()
        neg = (inp.get("negative_prompt") or
               "blurry, distorted, low quality, static, watermark, text").strip()

        pipe = get_pipe()
        gen = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # fit the reference to the target frame, centre-crop (no letterboxing)
        tw, th = width, height
        scale = max(tw / img.width, th / img.height)
        img = img.resize((max(1, int(img.width * scale + 0.5)),
                          max(1, int(img.height * scale + 0.5))), Image.LANCZOS)
        left = (img.width - tw) // 2
        top = (img.height - th) // 2
        img = img.crop((left, top, left + tw, top + th))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=nframes,
            num_inference_steps=int(inp.get("steps") or 30),
            guidance_scale=float(inp.get("guidance") or 5.0),
        )
        if gen is not None:
            kwargs["generator"] = gen

        if isinstance(pipe, WanImageToVideoPipeline):
            kwargs["image"] = img
        out = pipe(**kwargs).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        export_to_video(out, path, fps=int(inp.get("fps") or 24))
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        os.unlink(path)

        return {
            "video_b64": video_b64,
            "meta": {
                "model": MODEL_ID,
                "width": width, "height": height, "frames": nframes,
                "fps": int(inp.get("fps") or 24), "seed": seed,
                "seconds": round(time.time() - t0, 1),
                "mode": "i2v" if isinstance(pipe, WanImageToVideoPipeline) else "t2v",
            },
        }
    except Exception as e:  # noqa: BLE001 - surface the real message, never swallow
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
