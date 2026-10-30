"""Offline logic test for the face pass in handler.py.

Stubs torch/diffusers/runpod so the real handler module imports on a laptop,
then exercises restore_faces() against a fake restorer that follows
facexlib/GFPGAN's real contract:

  face_helper.det_faces      N x 5 rows (x1, y1, x2, y2, score)
  face_helper.affine_matrices  frame -> aligned-crop 2x3 matrices
  enhance() returns          (bg, aligned_restored_crops, full_frame_bgr)

Covers detection read, legacy attribute fallback, anchor on/off/partial,
per-request override, per-frame failures, numpy batches and the URL model
path. No GPU, no weights, no network beyond localhost.
"""
import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


# ---- stub the heavy imports -------------------------------------------------
torch = types.ModuleType("torch")


class _Gen:
    def __init__(self, device=None):
        pass

    def manual_seed(self, s):
        return self


torch.Generator = _Gen
torch.float32 = "float32"
torch.float16 = "float16"
torch.bfloat16 = "bfloat16"
torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, is_available=lambda: False)
torch.backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)))
sys.modules["torch"] = torch

diffusers = types.ModuleType("diffusers")
diffusers.AutoencoderKLWan = object
diffusers.WanImageToVideoPipeline = type("WanImageToVideoPipeline", (), {})
diffusers.WanPipeline = type("WanPipeline", (), {})
d_utils = types.ModuleType("diffusers.utils")
d_utils.export_to_video = lambda *a, **k: None
diffusers.utils = d_utils
sys.modules["diffusers"] = diffusers
sys.modules["diffusers.utils"] = d_utils

runpod = types.ModuleType("runpod")
runpod.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
sys.modules["runpod"] = runpod

# gfpgan is deliberately NOT installed: that is the graceful-degradation path.
sys.modules.pop("gfpgan", None)

spec = importlib.util.spec_from_file_location("handler", os.path.join(REPO, "handler.py"))
handler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler)


def frame(size=(832, 480), colour=(128, 128, 128)):
    return Image.new("RGB", size, colour)


def frame_to_crop_affine(box, crop=64):
    """facexlib contract: affine maps FRAME coords -> aligned crop coords.

    M = [[s, 0, -x1*s], [0, s, -y1*s]] with s = crop / box width.
    """
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    s = float(crop) / (x2 - x1)
    return np.array([[s, 0, -x1 * s], [0, s, -y1 * s]], dtype=np.float32)


class FakeFaceHelper:
    def __init__(self, det_faces=None, affines=None, legacy_boxes=None):
        self.det_faces = det_faces
        self.affine_matrices = affines
        self.face_bboxes = legacy_boxes


class FakeRestorer:
    """enhance(): full frame becomes GREEN; aligned crops become RED (BGR)."""

    CROP = 64

    def __init__(self, det_faces=None, affines=None, legacy_boxes=None):
        self.face_helper = FakeFaceHelper(det_faces, affines, legacy_boxes)

    def enhance(self, bgr, **kw):
        out = bgr.copy()
        out[:] = (0, 255, 0)  # BGR green
        anchored_crop = np.zeros((self.CROP, self.CROP, 3), dtype=np.uint8)
        anchored_crop[:] = (0, 0, 255)  # BGR red
        n = len(self.face_helper.affine_matrices) if self.face_helper.affine_matrices \
            else len(self.face_helper.det_faces or self.face_helper.face_bboxes or [])
        return None, [anchored_crop] * n, out


def setup(det_faces=None, affines=None, legacy=None):
    handler.FACE_RESTORE = True
    handler.FACE_STRENGTH = 1.0
    handler.FACE_ANCHOR = 0.65
    handler._RESTORE_ERR = None
    handler._RESTORER = FakeRestorer(det_faces, affines, legacy)


print("face pass: disabled")
handler.FACE_RESTORE = False
h_in = frame()
out, stats = handler.restore_faces([h_in])
check("disabled returns same frames", out[0] is h_in)
check("disabled reports enabled=False", stats["enabled"] is False)
check("disabled keeps stats shape", stats.get("frames") == 1 and "anchor_weight" in stats, str(stats))

print("face pass: library missing")
handler.FACE_RESTORE = True
handler._RESTORER = None
handler._RESTORE_ERR = None
h_in = frame()
out, stats = handler.restore_faces([h_in])
check("missing gfpgan keeps frames", out[0] is h_in)
check("missing gfpgan reports error", "ModuleNotFoundError" in (stats.get("error") or ""), stats.get("error"))
check("missing gfpgan does not raise", True)

print("face pass: real attribute path (det_faces + affines)")
BOX = [100, 100, 220, 240]
setup(det_faces=[BOX + [0.99]], affines=[frame_to_crop_affine(BOX)])
out, stats = handler.restore_faces([frame()])
check("face area restored (GFPGAN green)",
      tuple(np.array(out[0])[170, 160]) == (0, 255, 0), str(tuple(np.array(out[0])[170, 160])))
check("frame outside face untouched", tuple(np.array(out[0])[10, 10]) == (128, 128, 128))
check("stats count restored frames", stats["restored"] == 1, str(stats))
check("first usable frame is the anchor", stats["anchor_frame"] == 0 and stats["anchored"] == 0, str(stats))
check("det_faces score column ignored", True)

print("face pass: temporal identity anchor")
out, stats = handler.restore_faces([frame(), frame(), frame()])
check("first frame keeps GFPGAN face",
      tuple(np.array(out[0])[170, 160]) == (0, 255, 0), str(tuple(np.array(out[0])[170, 160])))
px = np.array(out[1])[170, 160]
check("later frame pulled onto anchor (anchor=0.65 blend)",
      abs(int(px[0]) - 166) <= 6 and abs(int(px[1]) - 89) <= 6 and int(px[2]) <= 6, str(tuple(px)))
check("later frames counted as anchored", stats["anchored"] == 2 and stats["restored"] == 3, str(stats))
check("anchor frame recorded once", stats["anchor_frame"] == 0, str(stats))
check("outside oval keeps GFPGAN result",
      tuple(np.array(out[1])[300, 400]) == (128, 128, 128), str(tuple(np.array(out[1])[300, 400])))

print("face pass: anchor off / partial")
out0, stats0 = handler.restore_faces([frame(), frame()], anchor=0.0)
check("anchor 0 keeps per-frame GFPGAN face",
      tuple(np.array(out0[1])[170, 160]) == (0, 255, 0), str(tuple(np.array(out0[1])[170, 160])))
check("anchor 0 anchored nothing", stats0["anchored"] == 0 and stats0["anchor_weight"] == 0.0, str(stats0))

out_half, stats_half = handler.restore_faces([frame(), frame()], anchor=0.5)
px = np.array(out_half[1])[170, 160]
check("anchor 0.5 blends half red/half green", abs(int(px[0]) - 128) <= 6 and abs(int(px[1]) - 128) <= 6
      and int(px[2]) <= 6, str(tuple(px)))
check("per-request anchor overrides env", stats_half["anchor_weight"] == 0.5, str(stats_half))
check("anchor clamped to [0,1]",
      handler.restore_faces([frame(), frame()], anchor=9)[1]["anchor_weight"] == 1.0)

print("face pass: strength blend")
setup(det_faces=[BOX + [0.9]], affines=[frame_to_crop_affine(BOX)])
handler.FACE_ANCHOR = 0.0
handler.FACE_STRENGTH = 1.0
out, _ = handler.restore_faces([frame()])
check("strength 1 full restore", tuple(np.array(out[0])[170, 160]) == (0, 255, 0))
handler.FACE_STRENGTH = 0.5
out, _ = handler.restore_faces([frame()])
px = np.array(out[0])[170, 160]
check("strength 0.5 blends green/grey", 60 <= px[0] <= 68 and 186 <= px[1] <= 194, str(tuple(px)))
handler.FACE_STRENGTH = 0.0
out, _ = handler.restore_faces([frame()])
check("strength 0 leaves frame unchanged", tuple(np.array(out[0])[170, 160]) == (128, 128, 128))
out, stats_s = handler.restore_faces([frame()], strength=0.75)
px = np.array(out[0])[170, 160]
check("per-request strength overrides env", abs(int(px[1]) - 224) <= 6, str(tuple(px)) + str(stats_s))

print("face pass: face too small to repair")
setup(det_faces=[[10, 10, 30, 30, 0.9]], affines=[frame_to_crop_affine([10, 10, 30, 30])])
out_small, stats_small = handler.restore_faces([frame()])
check("tiny face left alone", tuple(np.array(out_small[0])[20, 20]) == (128, 128, 128))
check("tiny face not counted", stats_small["restored"] == 0)

print("face pass: alignment mismatch fails open")
setup(det_faces=[], affines=[frame_to_crop_affine(BOX)])  # crops pair with no boxes
out_mm, stats_mm = handler.restore_faces([frame()])
check("unpaired crops keep frames", tuple(np.array(out_mm[0])[170, 160]) == (128, 128, 128))
check("unpaired crops not counted", stats_mm["restored"] == 0 and stats_mm["anchor_frame"] is None,
      str(stats_mm))

print("face pass: legacy face_bboxes fallback (no affines)")
setup(legacy=[BOX])
out_leg, stats_leg = handler.restore_faces([frame(), frame()])
check("legacy boxes still restore the face",
      tuple(np.array(out_leg[0])[170, 160]) == (0, 255, 0), str(tuple(np.array(out_leg[0])[170, 160])))
check("legacy has no anchor warp", tuple(np.array(out_leg[1])[170, 160]) == (0, 255, 0)
      and stats_leg["anchored"] == 0, str(tuple(np.array(out_leg[1])[170, 160])))

print("face pass: restorer raising on a frame")


class Boom(FakeRestorer):
    def enhance(self, arr, **kw):
        raise RuntimeError("CUDA out of memory")


setup(det_faces=[BOX + [0.9]], affines=[frame_to_crop_affine(BOX)])
handler._RESTORER = Boom(det_faces=[BOX + [0.9]], affines=[frame_to_crop_affine(BOX)])
out_boom, stats_boom = handler.restore_faces([frame(), frame(), frame()],
                                             anchor=0.0)
check("per-frame failure keeps frames",
      all(tuple(np.array(f)[170, 160]) == (128, 128, 128) for f in out_boom))
check("per-frame failures counted", stats_boom["errors"] == 3, str(stats_boom))

print("face pass: pipeline returns a numpy batch")
setup(det_faces=[], affines=[])
arr = np.zeros((2, 48, 64, 3), dtype=np.float32)
arr[..., 0] = 1.0
out_np, stats_np = handler.restore_faces(arr)
check("numpy batch -> PIL frames", len(out_np) == 2 and isinstance(out_np[0], Image.Image),
      str(type(out_np[0]) if out_np else None))
check("float [0,1] scaled to 0..255", tuple(np.array(out_np[0])[5, 5]) == (255, 0, 0),
      str(tuple(np.array(out_np[0])[5, 5])))
check("numpy batch reports frame count", stats_np["frames"] == 2, str(stats_np))

setup(det_faces=[[4, 4, 44, 44, 0.9]], affines=[frame_to_crop_affine([4, 4, 44, 44])])
out_u8, _ = handler.restore_faces(np.full((1, 64, 64, 3), 128, dtype=np.uint8))
px_corner = np.array(out_u8[0])[62, 62].astype(int)
check("uint8 batch keeps scale (outside face)", int(abs(px_corner - 128).max()) <= 3, str(tuple(px_corner)))
px_mid = np.array(out_u8[0])[20, 24].astype(int)
check("uint8 batch face restored", abs(int(px_mid[0])) <= 40 and int(px_mid[1]) >= 215, str(tuple(px_mid)))

print("face pass: gfpgan weights from an http(s) URL")
served = []
weights = b"K" * 4096


class _Srv(BaseHTTPRequestHandler):
    def do_GET(self):
        served.append(1)
        self.send_response(200)
        self.send_header("Content-Length", str(len(weights)))
        self.end_headers()
        self.wfile.write(weights)

    def log_message(self, *a):
        pass


httpd = HTTPServer(("127.0.0.1", 0), _Srv)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()
cache_dir = tempfile.mkdtemp(prefix="dml-gfpgan-test-")
handler.GFPGAN_MODEL_PATH = f"http://127.0.0.1:{port}/GFPGANv1.4.pth"
os.environ["GFPGAN_CACHE"] = cache_dir
try:
    local = handler._resolve_model_path()
    check("url model downloaded once", served == [1] and os.path.isfile(local), f"served={served} local={local}")
    with open(local, "rb") as f:
        check("downloaded bytes intact", f.read() == weights)
    local2 = handler._resolve_model_path()
    check("second call reuses cache (no re-download)", served == [1] and local2 == local, f"served={served}")
finally:
    httpd.shutdown()
    handler.GFPGAN_MODEL_PATH = "/opt/gfpgan/GFPGANv1.4.pth"
    os.environ.pop("GFPGAN_CACHE", None)
missing = handler._resolve_model_path()
check("missing local path -> empty (default weights)", missing == "", repr(missing))

print("face pass: pipeline dtype env (T4 has no native bfloat16)")
dt_env = dict(os.environ)
for want in ("float16", "bogus"):
    os.environ["TORCH_DTYPE"] = want
    spec_p = importlib.util.spec_from_file_location(
        "handler_dtype_" + want, os.path.join(REPO, "handler.py"))
    mod = importlib.util.module_from_spec(spec_p)
    sys.modules.pop("handler", None)
    spec_p.loader.exec_module(mod)
    check(f"TORCH_DTYPE={want} -> {mod.TORCH_DTYPE}",
          mod.TORCH_DTYPE == ("float16" if want == "float16" else "bfloat16"),
          str(mod.TORCH_DTYPE))
os.environ.pop("TORCH_DTYPE", None)

meta_ok = {"meta-model": True}
try:
    handler._RESTORER = Boom(det_faces=[BOX + [0.9]], affines=[])
    job = {"input": {"image_b64": "not-an-image", "face_anchor": 0.9, "face_strength": 0.7}}
    res = handler.handler(job)
    check("handler surfaces bad input as error, never silently",
          isinstance(res, dict) and bool(res.get("error")), str(res)[:120])
except Exception as e:  # noqa: BLE001 - handler must not raise past its boundary
    check("handler surfaces bad input as error, never silently", False, repr(e))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
