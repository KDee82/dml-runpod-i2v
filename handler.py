"""
Runpod serverless handler — real image-to-video, contract-compatible with
DML Motion Studio's services/api/runpod_worker.py.

IN  : {image_b64, positive_prompt, negative_prompt, seed, duration_seconds,
       aspect_ratio, fps, model}
OUT : {video_b64, meta}   (or {error} on failure — never a silent success)

Model: Wan 2.2 TI2V-5B (Apache-2.0). Memory strategy matters on a 24 GB card:
the pipeline is CPU-offloaded with VAE tiling + slicing and the frame area is
capped, otherwise 720p overruns 24 GB ("CUDA out of memory").

Face pass (optional): a 5B model at 480p has no facial conditioning, so faces
drift and melt over a shot. After generation each frame's faces are detected,
restored with GFPGAN and pasted back, which repairs structure without touching
motion, identity or the rest of the frame. Disable with FACE_RESTORE=0.
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

# --- face restoration -------------------------------------------------------
# A face smaller than this many pixels cannot be repaired, only smeared, so
# tiny faces are left alone and the frame is passed through untouched.
FACE_RESTORE = os.environ.get("FACE_RESTORE", "1").lower() in ("1", "true", "yes", "on")
FACE_MIN_PX = int(os.environ.get("FACE_MIN_PX", "40"))
# 1.0 = full restoration. Lower it if faces start looking too smoothed/plastic.
FACE_STRENGTH = float(os.environ.get("FACE_STRENGTH", "1.0"))
GFPGAN_MODEL_PATH = os.environ.get(
    "GFPGAN_MODEL_PATH",
    "/opt/gfpgan/GFPGANv1.4.pth")
# GFPGAN internally averages over frames only if a sequence is supplied; we
# restore per frame, so keep one instance (fixed weights = deterministic).

_PIPE = None
_RESTORER = None
_RESTORE_ERR = None


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


def get_restorer():
    """Load GFPGAN once per worker, or None when the face pass is off/broken."""
    global _RESTORER, _RESTORE_ERR
    if not FACE_RESTORE:
        return None
    if _RESTORER is not None:
        return _RESTORER
    if _RESTORE_ERR is not None:
        return None
    try:
        # basicsr (a GFPGAN dependency) imports this module, which torchvision
        # removed in 0.17. Shim it before the import or the whole pass dies.
        import sys as _sys
        import types as _types
        try:
            import torchvision.transforms.functional as _F
            if "torchvision.transforms.functional_tensor" not in _sys.modules:
                shim = _types.ModuleType("torchvision.transforms.functional_tensor")
                shim.rgb_to_grayscale = _F.rgb_to_grayscale
                _sys.modules["torchvision.transforms.functional_tensor"] = shim
        except Exception:  # noqa: BLE001 - shim is best-effort
            pass
        from gfpgan import GFPGANer
        model = GFPGAN_MODEL_PATH if os.path.exists(GFPGAN_MODEL_PATH) else None
        t0 = time.time()
        _RESTORER = GFPGANer(model_path=model, upscale=2, arch="clean",
                             channel_multiplier=2, bg_upsampler=None)
        print(f"[worker] gfpgan ready in {time.time() - t0:.1f}s", flush=True)
        return _RESTORER
    except Exception as e:  # noqa: BLE001 - a face pass must never fail a render
        _RESTORE_ERR = f"{type(e).__name__}: {e}"
        print(f"[worker] face restore unavailable: {_RESTORE_ERR}", flush=True)
        return None


def restore_faces(frames):
    """Repair faces in generated frames. Returns (frames, stats).

    Frames with no face, or only faces below FACE_MIN_PX, come back untouched,
    so the rest of the shot is never altered by this pass.
    """
    stats = {"enabled": FACE_RESTORE, "restored": 0, "frames": len(frames)}
    if not frames or not FACE_RESTORE:
        return frames, stats
    restorer = get_restorer()
    if restorer is None:
        stats["enabled"] = False
        stats["error"] = _RESTORE_ERR or "disabled"
        return frames, stats

    import cv2
    import numpy as np
    from PIL import ImageDraw, ImageFilter

    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    out = []
    for im in frames:
        rgb = np.array(im.convert("RGB"))
        try:
            _, _, restored = restorer.enhance(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                has_aligned=False, only_center_face=False, paste_back=True)
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the job
            stats["errors"] = stats.get("errors", 0) + 1
            if stats["errors"] == 1:
                print(f"[worker] face restore frame failed: {type(e).__name__}: {e}",
                      flush=True)
            out.append(im)
            continue

        boxes = [b for b in getattr(restorer.face_helper, "face_bboxes", [])
                 if (b[2] - b[0]) >= FACE_MIN_PX and (b[3] - b[1]) >= FACE_MIN_PX]
        if restored is None or not boxes:
            out.append(im)  # no usable face: keep the generated frame as-is
            continue

        fixed = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        # Always composite through a face mask: only the detected face boxes may
        # change, whatever the restorer did to the rest of the frame.
        mask = Image.new("L", im.size, 0)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2 in boxes:
            pad = int(0.25 * max(x2 - x1, y2 - y1))
            draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad],
                           fill=int(255 * FACE_STRENGTH))
        mask = mask.filter(ImageFilter.GaussianBlur(max(2, FACE_MIN_PX // 8)))
        out.append(Image.composite(fixed, im.convert("RGB"), mask))
        stats["restored"] += 1

    return out, stats


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

        face_t0 = time.time()
        out, face_stats = restore_faces(out)
        if face_stats.get("restored"):
            face_stats["seconds"] = round(time.time() - face_t0, 1)
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

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
                "face_restore": face_stats,
            },
        }
    except Exception as e:  # noqa: BLE001 - surface the real message, never swallow
        import traceback
        msg = f"{type(e).__name__}: {e}"
        print("[worker] FAILED", msg, flush=True)
        return {"error": msg[:600], "traceback": traceback.format_exc()[-1200:]}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
