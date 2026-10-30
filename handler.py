"""
Runpod serverless handler — real image-to-video, contract-compatible with
DML Motion Studio's services/api/runpod_worker.py.

IN  : {image_b64, positive_prompt, negative_prompt, seed, duration_seconds,
       aspect_ratio, fps, model, face_anchor, face_strength}
OUT : {video_b64, meta}   (or {error} on failure — never a silent success)

Model: Wan 2.2 TI2V-5B (Apache-2.0). Memory strategy matters on a 24 GB card:
the pipeline is CPU-offloaded with VAE tiling + slicing and the frame area is
capped, otherwise 720p overruns 24 GB ("CUDA out of memory").

Face pass (optional): a 5B model at 480p has no facial conditioning, so faces
drift and melt over a shot. After generation each frame's faces are detected,
restored with GFPGAN and pasted back, which repairs structure without touching
motion, identity or the rest of the frame. On top of that, every frame after
the first usable one is pulled toward that frame's canonical aligned face
(temporal identity anchor), so GFPGAN's per-frame "average face" cannot invent
a different person late in the shot. Disable with FACE_RESTORE=0; anchor
weight with FACE_ANCHOR / per-request face_anchor.
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

# T4 / older GPUs have no native bfloat16: BF16 kernels either fall back to
# fp32 (slow) or error out. TORCH_DTYPE=float16 makes the pipeline T4-safe.
# Values: bfloat16 (default, for A40/A100/4090), float16, float32.
_TORCH_DTYPE = os.environ.get("TORCH_DTYPE", "").strip().lower()
TORCH_DTYPE = (getattr(torch, _TORCH_DTYPE, torch.bfloat16)
               if _TORCH_DTYPE in ("bfloat16", "float16", "fp16", "float32", "fp32")
               else torch.bfloat16)

# 24 GB card: keep total pixels <= ~480p. Raise MAX_PIXELS on a 48 GB+ GPU.
MAX_PIXELS = int(os.environ.get("MAX_PIXELS", str(480 * 832)))

# --- face restoration -------------------------------------------------------
# A face smaller than this many pixels cannot be repaired, only smeared, so
# tiny faces are left alone and the frame is passed through untouched.
FACE_RESTORE = os.environ.get("FACE_RESTORE", "1").lower() in ("1", "true", "yes", "on")
FACE_MIN_PX = int(os.environ.get("FACE_MIN_PX", "40"))
# 1.0 = full restoration. Lower it if faces start looking too smoothed/plastic.
FACE_STRENGTH = float(os.environ.get("FACE_STRENGTH", "1.0"))
# Weight of the temporal identity anchor: how strongly every frame after the
# first usable one is pulled onto that first canonical face. 0 = off (each
# frame keeps GFPGAN's own per-frame result); 0.65 is the studio default.
FACE_ANCHOR = float(os.environ.get("FACE_ANCHOR", "0.65"))
GFPGAN_MODEL_PATH = os.environ.get(
    "GFPGAN_MODEL_PATH",
    "/opt/gfpgan/GFPGANv1.4.pth")

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
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=TORCH_DTYPE)
    except Exception as e:  # noqa: BLE001 - fall back to text-to-video
        print(f"[worker] I2V loader failed ({e}); trying T2V", flush=True)
        pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=TORCH_DTYPE)

    # 24 GB budget: stream weights from CPU instead of holding them all on the GPU
    try:
        pipe.enable_model_cpu_offload()
        print("[worker] cpu offload enabled", flush=True)
    except Exception as e:  # noqa: BLE001 - no accelerate, or missing dep
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


def _resolve_model_path():
    """GFPGAN_MODEL_PATH doubles as an http(s) URL; download once, reuse cache."""
    src = (GFPGAN_MODEL_PATH or "").strip()
    if not src.startswith(("http://", "https://")):
        return src if os.path.exists(src) else ""
    base = os.environ.get("GFPGAN_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or tempfile.gettempdir(), "gfpgan")
    local = os.path.join(base, (src.split("?")[0].rstrip("/").split("/")[-1]
                                or "GFPGANv1.4.pth"))
    if os.path.exists(local) and os.path.getsize(local) > 1024:
        return local
    os.makedirs(base, exist_ok=True)
    import urllib.request
    print(f"[worker] downloading gfpgan weights: {src}", flush=True)
    tmp = local + ".part"
    urllib.request.urlretrieve(src, tmp)
    os.replace(tmp, local)
    return local


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
        model = _resolve_model_path()
        t0 = time.time()
        # upscale=1: paste back at source resolution. GFPGAN's default 2x hands
        # back a doubled frame and the composite lands misaligned.
        _RESTORER = GFPGANer(model_path=model or None, upscale=1, arch="clean",
                             channel_multiplier=2, bg_upsampler=None)
        print(f"[worker] gfpgan ready in {time.time() - t0:.1f}s", flush=True)
        return _RESTORER
    except Exception as e:  # noqa: BLE001 - a face pass must never fail a render
        _RESTORE_ERR = f"{type(e).__name__}: {e}"
        print(f"[worker] face restore unavailable: {_RESTORE_ERR}", flush=True)
        return None


def _as_pil_frames(frames):
    """Normalise pipeline output to a list of PIL RGB images.

    diffusers hands back either a list of PIL images or an (n, h, w, 3) numpy
    array, depending on pipeline and version — Wan returns the array, and
    treating that as a list-of-images is what broke the first live render.
    """
    import numpy as np
    from PIL import Image

    def one(f):
        if not isinstance(f, np.ndarray):
            return f
        if f.dtype != np.uint8:
            f = f.astype(np.float32)
            if f.max() <= 1.0:  # postprocess_video returns floats in [0, 1]
                f = f * 255.0
            f = np.clip(f, 0, 255).round().astype(np.uint8)
        return Image.fromarray(f)

    if isinstance(frames, np.ndarray):
        return [one(f) for f in (frames[None] if frames.ndim == 3 else frames)]
    return [one(f) for f in frames]


def _detect_boxes(helper):
    """Face boxes for one enhance() call, from facexlib's real attributes.

    facexlib's FaceRestoreHelper exposes det_faces (N x 5: x1, y1, x2, y2,
    score) and affine_matrices; cropped_faces / restored_faces pair with them
    positionally. No facexlib release defines face_bboxes; it is read only as
    a legacy fallback for exotic builds. Detection coordinates are floats.
    """
    boxes = []
    det = getattr(helper, "det_faces", None)
    if det is not None and len(det):
        for row in det:
            vals = row[:4].tolist() if hasattr(row, "tolist") else list(row)[:4]
            boxes.append([int(round(float(v))) for v in vals])
        return boxes
    legacy = getattr(helper, "face_bboxes", None)
    if legacy:
        boxes = [[int(round(float(b[0]))), int(round(float(b[1]))),
                  int(round(float(b[2]))), int(round(float(b[3])))] for b in legacy]
    return boxes


def _feathered_rect_mask(size, boxes, strength, feather):
    """Old paste-back mask: padded rectangle per box, feather-blurred."""
    from PIL import Image, ImageDraw, ImageFilter
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for x1, y1, x2, y2 in boxes:
        pad = int(0.25 * max(x2 - x1, y2 - y1))
        draw.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad],
                       fill=int(255 * strength))
    return mask.filter(ImageFilter.GaussianBlur(max(2, feather)))


def _feathered_oval_mask(size, box, feather_px):
    """Anchor mask: feathered face oval, tightened around the detection box."""
    from PIL import Image, ImageDraw, ImageFilter
    w, h = size
    x1, y1, x2, y2 = box
    bw, bh, cx, cy = x2 - x1, y2 - y1, (x1 + x2) / 2, (y1 + y2) / 2
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse(
        [cx - bw * 0.55, cy - bh * 0.62, cx + bw * 0.55, cy + bh * 0.62], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(max(2, int(feather_px))))


def restore_faces(frames, anchor=None, strength=None):
    """Repair faces in generated frames. Returns (frames, stats).

    Frames with no face, or only faces below FACE_MIN_PX, come back untouched,
    so the rest of the shot is never altered by this pass.

    anchor: 0-1 weight pulling every frame after the first usable one onto
    that frame's canonical aligned face (0 = each frame keeps GFPGAN's own
    per-frame result). strength: 0-1 blend of the restored face inside the
    face mask. Per-request values override the env defaults (FACE_ANCHOR /
    FACE_STRENGTH) so the studio can turn the knob per job.
    """
    frames = _as_pil_frames(frames)
    anchor_w = FACE_ANCHOR if anchor is None else min(1.0, max(0.0, float(anchor)))
    strength_w = FACE_STRENGTH if strength is None else min(1.0, max(0.0, float(strength)))
    stats = {"enabled": FACE_RESTORE, "restored": 0, "anchored": 0,
             "anchor_frame": None, "anchor_weight": anchor_w, "errors": 0,
             "frames": len(frames)}
    if not frames or not FACE_RESTORE:
        stats["enabled"] = False
        return frames, stats
    restorer = get_restorer()
    if restorer is None:
        stats["error"] = _RESTORE_ERR or "disabled"
        return frames, stats

    import cv2
    import numpy as np

    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    out = []
    anchor_crop = None          # aligned restored BGR crop of the first usable face
    anchor_frame = None
    feather = max(2, FACE_MIN_PX // 8)

    for i, im in enumerate(frames):
        rgb = np.array(im.convert("RGB"))
        try:
            _, aligned_restored, restored = restorer.enhance(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                has_aligned=False, only_center_face=False, paste_back=True)
        except Exception as e:  # noqa: BLE001 - one bad frame must not kill the job
            stats["errors"] += 1
            if stats["errors"] == 1:
                print(f"[worker] face restore frame failed: {type(e).__name__}: {e}",
                      flush=True)
            out.append(im)
            continue

        helper = getattr(restorer, "face_helper", None)
        boxes = _detect_boxes(helper) if helper is not None else []
        affines = list(getattr(helper, "affine_matrices", None) or [])
        usable = []
        for j, aligned in enumerate(aligned_restored or []):
            if aligned is None or j >= len(boxes):
                break  # boxes must pair with the aligned crops
            b = boxes[j]
            if (b[2] - b[0]) >= FACE_MIN_PX and (b[3] - b[1]) >= FACE_MIN_PX:
                affine = (np.asarray(affines[j], dtype=np.float32).reshape(2, 3)
                          if j < len(affines) else None)
                usable.append((b, affine, np.asarray(aligned)))
        if restored is None or not usable:
            out.append(im)  # no usable face: keep the generated frame as-is
            continue

        # GFPGAN paste-back (upscale=1 -> same size as the input frame), shown
        # only inside the padded rectangles so the rest of the frame survives.
        fixed = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        result = Image.composite(
            fixed, im.convert("RGB"),
            _feathered_rect_mask(im.size, [b for b, _, _ in usable], strength_w, feather))

        if anchor_crop is None:
            # First usable frame: it IS the canonical identity for this job.
            anchor_crop, anchor_frame = usable[0][2], i
        elif anchor_w > 0:
            # Later frames: warp the canonical face back into every usable
            # face position through that frame's affine, blend at anchor_w
            # inside a feathered oval. Legacy helpers without affines still
            # get the GFPGAN paste-back above — the anchor just can't warp.
            base_rgb = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB).astype(np.float32)
            blended = False
            for box, affine, _ in usable:
                if affine is None:
                    continue
                back = cv2.invertAffineTransform(affine)
                warped = cv2.warpAffine(anchor_crop, back,
                                        (im.width, im.height),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
                mix = (cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32)
                       * anchor_w + base_rgb * (1.0 - anchor_w))
                anchored_frame = Image.fromarray(np.clip(mix + 0.5, 0, 255).astype(np.uint8))
                oval = _feathered_oval_mask(
                    im.size, box, max(2, int(min(box[2] - box[0], box[3] - box[1]) * 0.18)))
                result = Image.composite(anchored_frame, result, oval)
                blended = True
            if blended:
                stats["anchored"] += 1

        out.append(result)
        stats["restored"] += 1

    stats["anchor_frame"] = anchor_frame
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
        out, face_stats = restore_faces(
            out, anchor=inp.get("face_anchor"), strength=inp.get("face_strength"))
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
