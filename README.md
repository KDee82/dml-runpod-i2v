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
