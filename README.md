# dml-runpod-i2v

Runpod **serverless image-to-video** worker for [DML Motion Studio](../..).

Real GPU rendering with **Wan 2.2 TI2V-5B** (Apache-2.0), speaking the exact
contract the studio's `services/api/runpod_worker.py` already sends.

## Contract

Request (`POST /v2/<endpoint>/run`):

```json
{
  "input": {
    "image_b64": "<base64 png/jpeg>",
    "positive_prompt": "slow push in, cinematic",
    "negative_prompt": "blurry, distorted",
    "seed": 42,
    "duration_seconds": 5,
    "aspect_ratio": "16:9",
    "fps": 24,
    "model": "wan-2.2-ti2v-5b"
  }
}
```

Response:

```json
{ "video_b64": "<base64 mp4>", "meta": { "width": 704, "height": 1280, "frames": 121, "seconds": 214.3 } }
```

Failures return `{"error": "..."}` — the studio marks the job failed with the
message verbatim. No silent successes.

## Notes

- `num_frames` is snapped to the Wan requirement of `4n+1` (max 121 ≈ 5 s @ 24 fps).
- Dimensions snap to a 16-px grid.
- Weights download to `HF_HOME` (`/runpod-volume/hf`) on first job — attach a
  network volume to make cold starts fast; without one, expect a long first pull.
- The pipeline is cached per worker and reused across jobs.

## Face restoration

A 5B model at 480p has no facial conditioning, so faces drift and melt across a
shot. After generation, each frame's faces are detected with facexlib, restored
with GFPGAN and pasted back inside a feathered mask — only the face boxes change,
the rest of the frame is byte-identical. Motion, identity and framing are
untouched; this repairs structure, it does not add resolution.

| Env | Default | Meaning |
| --- | --- | --- |
| `FACE_RESTORE` | `1` | `0` disables the pass entirely |
| `FACE_MIN_PX` | `40` | faces smaller than this are skipped (too small to repair) |
| `FACE_STRENGTH` | `1.0` | 0–1 blend inside the face mask; lower it if faces look plastic |
| `GFPGAN_MODEL_PATH` | `/opt/gfpgan/GFPGANv1.4.pth` | baked into the image |

The pass is fail-open: a missing library, an OOM or a bad frame leaves the
generated frame untouched and reports it in `meta.face_restore` instead of
failing the render.

`python3 tests/test_face_pass.py` runs the logic offline (no GPU, no weights).

## Deploy

Runpod builds this repository directly. Create a **serverless** template from
this repo, GPU **RTX 4090**, `workersMin: 0` / `workersMax: 1` (idle time is free),
container disk 25 GB, then copy the endpoint id into the studio's `.env`:

```
RUNPOD_ENDPOINT_ID=<new endpoint id>
RUNPOD_INVOKE_URL=https://api.runpod.ai/v2
```

Keep `workersMin` at 0. The studio's kill switch (`services/api/killswitch.py`,
`scripts/killswitch.sh`, or Settings → GPU kill switch) scales the endpoint to
zero and purges the queue in one call.

## Live deployment (verified)

| Thing | Value |
|---|---|
| Image | `ghcr.io/kdee82/dml-runpod-i2v:1.1.0` (torch 2.6.0+cu124, diffusers 0.42) |
| Template | `ryrxfhm69b` — boots `handler.py` from this repo at container start |
| Endpoint | `re69tvql9fj1m4` (`DML-Motion-Studio-I2V-v4`), RTX 4090, `workersMin: 0` |
| Registry auth | Runpod GraphQL `saveRegistryAuth` → `ghcr-kdee82` (REST has no such route) |

**Verified render:** 832×480, 57 frames, 1.9 s, 114 s GPU execution.

Hot-loading the handler means prompt/model/memory changes need only a
`git push` — the next cold start picks them up. Only dependency or torch
changes require a rebuild.

## Traps that cost real debugging time

- **`diffusers>=0.35` requires `torch>=2.6`** (`torch.accelerator`). On torch
  2.4 the handler dies on import, the worker still reports "ready", and every
  job sits in `IN_QUEUE` forever. The Dockerfile now imports the stack at build
  time so this fails the build instead.
- **Cloudflare fingerprint-blocks Python's TLS stack.** `urllib` gets
  `404 page not found` on `/run` for the same URL that returns `200` from curl.
  Send requests with curl.
- **Never pass image base64 as an argv entry** — it exceeds ARG_MAX
  ("Argument list too long"). Pipe the JSON on stdin (`curl --data-binary @-`).
- **24 GB is the limit at 720p.** Use `enable_model_cpu_offload()` + VAE
  tiling/slicing and cap to ~480p; default `MAX_PIXELS` reflects that.
- **Template edits do not reach a running endpoint.** Create a new endpoint
  (or bump versions) after changing image or start command.
- **No network volume means a ~32 GB re-download per cold start.** Attach a
  volume to make warm starts fast and stop paying GPU time for downloads.

