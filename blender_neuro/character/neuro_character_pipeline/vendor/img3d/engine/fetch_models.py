"""Download the model files the engine needs, with a progress bar the
installer passes on. Files already on disk are skipped."""

import os
import sys
import threading
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
M = os.path.join(ROOT, "models")
LO, HI = 0.27, 0.99

api = HfApi()
jobs = []
for f in api.list_repo_tree("microsoft/TRELLIS.2-4B", recursive=True):
    if getattr(f, "size", None) is not None and not f.path.startswith("."):
        jobs.append(("microsoft/TRELLIS.2-4B", f.path, f.size,
                     os.path.join(M, "microsoft", "TRELLIS.2-4B")))
for f in api.list_repo_tree("visualbruno/dinov3-vitl16-pretrain-lvd1689m"):
    if f.path in ("model.safetensors", "config.json",
                  "preprocessor_config.json"):
        jobs.append(("visualbruno/dinov3-vitl16-pretrain-lvd1689m", f.path,
                     f.size, os.path.join(M, "facebook",
                                          "dinov3-vitl16-pretrain-lvd1689m")))
for f in api.list_repo_tree("microsoft/TRELLIS-image-large", path_in_repo="ckpts"):
    if os.path.basename(f.path).startswith("ss_dec_conv3d_16l8_fp16"):
        jobs.append(("microsoft/TRELLIS-image-large", f.path, f.size,
                     os.path.join(M, "microsoft", "TRELLIS-image-large")))

total = sum(j[2] for j in jobs) or 1
done = [sum(j[2] for j in jobs if os.path.exists(os.path.join(j[3], j[1])))]
todo = [j for j in jobs if not os.path.exists(os.path.join(j[3], j[1]))]


def watch():
    while todo_left[0]:
        got = done[0]
        for dp, dn, fn in os.walk(os.path.join(M)):
            if ".cache" in dp:
                for f in fn:
                    if f.endswith(".incomplete"):
                        try:
                            got += os.path.getsize(os.path.join(dp, f))
                        except OSError:
                            pass
        frac = min(1.0, got / float(total))
        print("PROGRESS %.3f Downloading AI model  %.1f / %.1f GB"
              % (LO + (HI - LO) * frac, got / 1e9, total / 1e9), flush=True)
        time.sleep(1.0)


todo_left = [len(todo)]
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
    todo_left[0] -= 1
print("models ok", flush=True)
