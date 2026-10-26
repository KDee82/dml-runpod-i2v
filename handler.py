"""
Runpod serverless handler — real image-to-video, contract-compatible with
DML Motion Studio's services/api/runpod_worker.py.

IN  : {image_b64, positive_prompt, negative_prompt, seed, duration_seconds,
       aspect_ratio, fps, model}
OUT : {video_b64, meta}   (or {error} on failure — never a silent success)

Model: Wan 2.2 TI2V-5B (Apache-2.0). Memory strategy matters on a 24 GB card:
the pipeline is CPU-offloaded with VAE tiling + slicing and the frame area is
capped, otherwise 720p overruns 24 GB ("CUDA out of memory").
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
os.environ.setdefault("HF_HOME", "/runpod-volume/hf")

# 24 GB card: keep total pixels <= ~480p. Raise MAX_PIXELS on a 48 GB+ GPU.
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(480 * 832)))

_PIPE = None


def get_pipe():
    """Load once per worker, reuse across jobs."""
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    print(f"[worker] loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    try:
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)
    except Exception as e:  # noqa: BLE001 - fall back to text-to-video
        print(f"[worker] I2V loader failed ({e}); trying T2V", flush=True)
        pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

    # 24 GB budget: stream weights from CPU instead of holding them all on the GPU
    try:
        pipe.enable_model_cpu_offload()
        print("[worker] cpu offload enabled", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[worker] offload failed ({e}); moving to cuda", flush=True)
        pipe.to("cuda")
    for fn in ("enable_tiling", "enable_slicing"):
        try:
            getattr(pipe.vae, fn)()
        except Exception:  # noqa: BLE001
            pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:  # noqa: BLE001
        pass
    _PIPE = pipe
    print(f"[worker] ready in {time.time() - t0:.1f}s", flush=True)
    return _PIPE


def _dims(aspect):
    """Return (width, height) on a 16-px grid, capped by MAX_PIXELS."""
    a = (aspect or "16:9").replace(" ", "")
    table = {
        "16:9": (832, 480), "9:16": (480, 832), "1:1": (624, 624),
        "4:5": (480, 608), "21:9": (960, 416),
    }
    w, h = table.get(a, (832, 480))
    if MAX_PIXELS > w * h:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    return w, h


def _frames(duration, fps):
    """Wan needs 4n+1 frames; length is capped because memory grows with frames."""
    n = int(round(float(duration or 5) * float(fps or 16)))
    n = max(17, min(n, 81))
    return ((n - 1) // 4) * 4 + 1


def handler(job):
    inp = job.get("input") or {}
    t0 = time.time()
    try:
        b64 = inp.get("image_b64") or ""
        if not b64:
            raise ValueError("image_b64 is required (image-to-video endpoint)")
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

        width, height = _dims(inp.get("aspect_ratio"))
        nframes = _frames(inp.get("duration_seconds"), inp.get("fps"))
        seed = int(inp.get("seed") or 0)
        prompt = (inp.get("positive_prompt") or "the subject moves gently").strip()
        neg = (inp.get("negative_prompt") or
               "blurry, distorted, warped, morphing, flicker, text, watermark").strip()

        pipe = get_pipe()
        gen = torch.Generator(device="cuda").manual_seed(seed) if seed else None

        # cover-crop the reference to the target frame (no letterboxing)
        scale = max(width / img.width, height / img.height)
        img = img.resize((max(1, int(img.width * scale + 0.5)),
                          max(1, int(img.height * scale + 0.5))), Image.LANCZOS)
        left, top = (img.width - width) // 2, (img.height - height) // 2
        img = img.crop((left, top, left + width, top + height))

        kwargs = dict(
            prompt=prompt,
            negative_prompt=neg,
            width=width,
            height=height,
            num_frames=nframes,
            num_inference_steps=int(inp.get("steps") or 24),
            guidance_scale=float(inp.get("guidance") or 5.0),
        )
        if gen is not None:
            kwargs["generator"] = gen
        if isinstance(pipe, WanImageToVideoPipeline):
            kwargs["image"] = img

        out = pipe(**kwargs).frames[0]

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        export_to_video(out, path, fps=int(inp.get("fps") or 16))
        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()
        os.unlink(path)

        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return {
            "video_b64": video_b64,
            "meta": {
                "model": MODEL_ID,
                "width": width, "height": height, "frames": nframes,
                "fps": int(inp.get("fps") or 16), "seed": seed,
                "seconds": round(time.time() - t0, 1),
                "mode": "i2v" if isinstance(pipe, WanImageToVideoPipeline) else "t2v",
            },
        }
    except Exception as e:  # noqa: BLE001 - surface the real message, never swallow
        import traceback
        msg = f"{type(e).__name__}: {e}"
        print("[worker] FAILED", msg, flush=True)
        return {"error": msg[:600], "traceback": traceback.format_exc()[-1200:]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
