#!/usr/bin/env python3
"""Create the serverless template + endpoint for the I2V worker, then report ids."""
import json, sys, urllib.request, urllib.error
from pathlib import Path

ROOT = Path("/root/DML Motion Studio")
env = {}
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
KEY = env["RUNPOD_API_KEY"]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
REST = "https://rest.runpod.io/v1"
IMAGE = sys.argv[1] if len(sys.argv) > 1 else "ghcr.io/kdee82/dml-runpod-i2v:1.0.0"


def call(path, body=None, method="GET"):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(REST + path, data=data, method=method, headers={
        "Authorization": f"Bearer {KEY}", "User-Agent": UA, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            raw = r.read().decode(errors="replace")
            return r.status, json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:600]
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


tpl_body = {
    "name": "DML-I2V-Wan22",
    "imageName": IMAGE,
    "isServerless": True,
    "isPublic": False,
    "containerDiskInGb": 40,
    "env": {"MODEL_ID": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            "HF_HOME": "/runpod-volume/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1"},
    "ports": ["8888/http"],
    "volumeInGb": 0,
    "volumeMountPath": "/workspace",
}

code, tpl = call("/templates", body=tpl_body, method="POST")
print("template:", code, json.dumps(tpl)[:400] if isinstance(tpl, (dict, list)) else tpl)
if code != 200 or not isinstance(tpl, dict):
    sys.exit(1)

tid = tpl.get("id")
print("templateId:", tid)

ep_body = {
    "name": "DML-Motion-Studio-I2V",
    "templateId": tid,
    "computeType": "GPU",
    "gpuTypeIds": ["NVIDIA GeForce RTX 4090"],
    "gpuCount": 1,
    "workersMin": 0,
    "workersMax": 1,
    "idleTimeout": 5,
    "scalerType": "QUEUE_DELAY",
    "scalerValue": 4,
    "flashboot": True,
}

code, ep = call("/endpoints", body=ep_body, method="POST")
print("endpoint:", code, json.dumps(ep)[:600] if isinstance(ep, (dict, list)) else ep)
if code == 200 and isinstance(ep, dict):
    print("\nNEW_ENDPOINT_ID:", ep.get("id"))
