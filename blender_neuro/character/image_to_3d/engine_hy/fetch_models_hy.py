"""Download Hunyuan3D 2.1 and its helper models, with a progress bar the
installer passes on. Files already on disk are skipped."""
import os
import sys
import threading
import time
import urllib.request

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
M = os.path.join(ROOT, "models")
LO, HI = 0.30, 0.99
api = HfApi()
jobs = []
for f in api.list_repo_tree("tencent/Hunyuan3D-2.1", recursive=True):
    p = f.path
    if getattr(f, "size", None) is None:
        continue
    if p.startswith(("hunyuan3d-dit-v2-1/", "hunyuan3d-vae-v2-1/",
                     "hunyuan3d-paintpbr-v2-1/")):
        jobs.append(("tencent/Hunyuan3D-2.1", p, f.size,
                     os.path.join(M, "tencent", "Hunyuan3D-2.1")))
for f in api.list_repo_tree("facebook/dinov2-giant"):
    if f.path.endswith((".json", ".safetensors")):
        jobs.append(("facebook/dinov2-giant", f.path, f.size,
                     os.path.join(M, "facebook", "dinov2-giant")))
total = sum(j[2] for j in jobs) or 1
done = [sum(j[2] for j in jobs if os.path.exists(os.path.join(j[3], j[1])))]
todo = [j for j in jobs if not os.path.exists(os.path.join(j[3], j[1]))]
left = [len(todo)]


def watch():
    while left[0]:
        got = done[0]
        for dp, dn, fn in os.walk(M):
            if ".cache" in dp:
                for f in fn:
                    if f.endswith(".incomplete"):
                        try:
                            got += os.path.getsize(os.path.join(dp, f))
                        except OSError:
                            pass
        frac = min(1.0, got / float(total))
        print("PROGRESS %.3f Downloading Hunyuan  %.1f / %.1f GB"
              % (LO + (HI - LO) * frac, got / 1e9, total / 1e9), flush=True)
        time.sleep(1.0)


threading.Thread(target=watch, daemon=True).start()
for repo, path, size, local in todo:
    for attempt in range(5):
        try:
            hf_hub_download(repo, path, local_dir=local)
            break
        except Exception as exc:
            print("retry %s %s" % (path, exc), flush=True)
            time.sleep(3)
    else:
        print("ERROR could not download %s" % path, flush=True)
        sys.exit(1)
    done[0] += size
    left[0] -= 1
esr = os.path.join(M, "realesrgan", "RealESRGAN_x4plus.pth")
if not os.path.exists(esr):
    os.makedirs(os.path.dirname(esr), exist_ok=True)
    data = urllib.request.urlopen(urllib.request.Request(
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
        "RealESRGAN_x4plus.pth", headers={"User-Agent": "img3d"}),
        timeout=300).read()
    open(esr, "wb").write(data)
print("models ok", flush=True)
