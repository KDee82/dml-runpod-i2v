"""Offline logic test for the face pass in handler.py.

Stubs torch/diffusers/runpod so the real handler module imports on a laptop,
then exercises restore_faces() against a fake restorer. No GPU, no model.
"""
import importlib.util
import os
import sys
import types

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  ok   " if cond else "  FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


# ---- stub the heavy imports ------------------------------------------------
torch = types.ModuleType("torch")


class _Gen:
    def __init__(self, device=None):
        pass

    def manual_seed(self, s):
        return self


torch.Generator = _Gen
torch.float32 = "float32"
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


print("face pass: disabled")
handler.FACE_RESTORE = False
h_in = frame()
out, stats = handler.restore_faces([h_in])
check("disabled returns same frames", out[0] is h_in)
check("disabled reports enabled=False", stats["enabled"] is False)

print("face pass: library missing")
handler.FACE_RESTORE = True
handler._RESTORER = None
handler._RESTORE_ERR = None
h_in = frame()
out, stats = handler.restore_faces([h_in])
check("missing gfpgan keeps frames", out[0] is h_in)
check("missing gfpgan reports error", "ModuleNotFoundError" in (stats.get("error") or ""), stats.get("error"))
check("missing gfpgan does not raise", True)


class FakeFaceHelper:
    def __init__(self, boxes):
        self.face_bboxes = boxes


class FakeRestorer:
    def __init__(self, boxes):
        self.face_helper = FakeFaceHelper(boxes)

    def enhance(self, arr, **kw):
        out = arr.copy()
        out[:] = (0, 0, 255)  # BGR pure red
        return None, None, out


print("face pass: usable face")
handler._RESTORE_ERR = None
handler.FACE_STRENGTH = 1.0
handler._RESTORER = FakeRestorer([[100, 100, 220, 240]])
out, stats = handler.restore_faces([frame()])
px = np.array(out[0])[170, 160]
check("face area restored", tuple(px) == (255, 0, 0), str(tuple(px)))
check("frame outside face untouched", tuple(np.array(out[0])[10, 10]) == (128, 128, 128))
check("stats count restored frames", stats["restored"] == 1, str(stats))

print("face pass: face too small to repair")
handler._RESTORER = FakeRestorer([[10, 10, 30, 30]])
out_small, stats_small = handler.restore_faces([frame()])
check("tiny face left alone", tuple(np.array(out_small[0])[20, 20]) == (128, 128, 128))
check("tiny face not counted", stats_small["restored"] == 0)

print("face pass: partial strength")
handler._RESTORER = FakeRestorer([[100, 100, 220, 240]])
handler.FACE_STRENGTH = 0.5
out_half, _ = handler.restore_faces([frame()])
px = np.array(out_half[0])[170, 160]
check("strength 0.5 blends", 180 <= px[0] <= 210 and 45 <= px[1] <= 80, str(tuple(px)))

print("face pass: restorer raising on a frame")
class Boom(FakeRestorer):
    def enhance(self, arr, **kw):
        raise RuntimeError("CUDA out of memory")


handler._RESTORER = Boom([[100, 100, 220, 240]])
handler.FACE_STRENGTH = 1.0
out_boom, stats_boom = handler.restore_faces([frame(), frame()])
check("per-frame failure keeps frames", all(tuple(np.array(f)[170, 160]) == (128, 128, 128) for f in out_boom))
check("per-frame failures counted", stats_boom.get("errors") == 2, str(stats_boom))

print("face pass: pipeline returns a numpy batch")
# Wan returns (n, h, w, 3) floats in [0, 1] — not a list of PIL images.
handler.FACE_RESTORE = True
handler.FACE_STRENGTH = 1.0
handler._RESTORE_ERR = None
handler._RESTORER = FakeRestorer([])  # no faces: must pass through untouched
arr = np.zeros((2, 48, 64, 3), dtype=np.float32)
arr[..., 0] = 1.0
out_np, stats_np = handler.restore_faces(arr)
check("numpy batch -> PIL frames", len(out_np) == 2 and isinstance(out_np[0], Image.Image),
      str(type(out_np[0]) if out_np else None))
check("float [0,1] scaled to 0..255", tuple(np.array(out_np[0])[5, 5]) == (255, 0, 0),
      str(tuple(np.array(out_np[0])[5, 5])))
check("numpy batch reports frame count", stats_np["frames"] == 2, str(stats_np))

handler._RESTORER = FakeRestorer([[4, 4, 44, 44]])  # 40px face: at FACE_MIN_PX
out_u8, _ = handler.restore_faces(np.full((1, 64, 64, 3), 128, dtype=np.uint8))
# the feathered mask bleeds a hair at the corner, so allow 3/255 of tint
px_corner = np.array(out_u8[0])[62, 62].astype(int)
check("uint8 batch keeps scale (outside face)", int(abs(px_corner - 128).max()) <= 3,
      str(tuple(px_corner)))
check("uint8 batch face restored", tuple(np.array(out_u8[0])[24, 24]) == (255, 0, 0),
      str(tuple(np.array(out_u8[0])[24, 24])))

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
