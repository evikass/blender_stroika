"""Model downloads for Image to 3D.

Every model comes from its maker's official page (Hugging Face / GitHub):
the add-on either downloads it there itself (Download) or shows the official
page so people can fetch it by hand (Official Page) into the engine folder.
Each file is checked against the size and SHA-256 its maker published.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time

import bpy

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(ADDON_DIR, "models_manifest.json")
NOWINDOW = 0x08000000 if os.name == "nt" else 0

_M = {}
_CACHE = {"t": 0.0, "root": None, "status": {}}
DL = {"p": None, "q": None, "groups": [], "frac": 0.0, "text": "",
      "error": "", "done": "", "t0": 0.0}


def manifest():
    if not _M:
        with open(MANIFEST, "r", encoding="utf-8") as f:
            _M.update(json.load(f))
    return _M


def groups():
    return manifest()["groups"]


def group(gid):
    for g in groups():
        if g["id"] == gid:
            return g
    return None


def hf_cache():
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(k):
            return os.environ[k]
    home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def models_folder(root):
    return os.path.join(root, "models")


def _target(comp, f, root):
    if comp["dest"] == "hfcache":
        repo = "models--" + comp["repo"].replace("/", "--")
        return os.path.join(hf_cache(), repo, "snapshots", comp["revision"],
                            *f["path"].split("/"))
    return os.path.join(root, *comp["dest"].split("/"), *f["path"].split("/"))


def _file_ok(comp, f, root):
    p = _target(comp, f, root)
    if f.get("patched"):
        # replaced by our fixed copy after the download
        return os.path.isfile(p)
    try:
        if os.path.getsize(p) == f["size"]:
            return True
    except OSError:
        pass
    conv = f.get("convert_to")
    if conv:
        c = _target(comp, {"path": conv}, root)
        return os.path.isfile(c) and os.path.isfile(c + ".ok")
    return False


def _scan(root):
    """{group id: (have bytes, total bytes, missing files)}"""
    m = manifest()
    out = {}
    comp_state = {}
    for cid, comp in m["components"].items():
        have = total = 0
        miss = []
        for f in comp["files"]:
            total += f["size"]
            if _file_ok(comp, f, root):
                have += f["size"]
            else:
                p = _target(comp, f, root)
                if os.path.isfile(p + ".part"):
                    have += os.path.getsize(p + ".part")
                miss.append(f["path"])
        pend = os.path.isfile(os.path.join(root, *comp["dest"].split("/"),
                                           ".post_pending")) \
            if comp["dest"] != "hfcache" else False
        comp_state[cid] = (have, total, miss, pend)
    for g in m["groups"]:
        have = sum(comp_state[c][0] for c in g["comps"])
        total = sum(comp_state[c][1] for c in g["comps"])
        miss = [(c, p) for c in g["comps"] for p in comp_state[c][2]]
        pend = any(comp_state[c][3] for c in g["comps"])
        out[g["id"]] = (have, total, miss, pend)
    return out


def status(root, gid, fresh=False):
    """(state, have, total, missing) - state is 'ready', 'missing',
    'partial' or 'pending' (downloaded, finishing after the engine
    install). Looked up from disk at most every few seconds."""
    now = time.time()
    if fresh or _CACHE["root"] != root or now - _CACHE["t"] > 4.0:
        try:
            _CACHE["status"] = _scan(root) if root else {}
            if root:
                _fix_refs(root)
        except Exception:
            _CACHE["status"] = {}
        _CACHE.update(t=now, root=root)
    have, total, miss, pend = _CACHE["status"].get(gid, (0, 1, ["?"], False))
    if not miss:
        st = "pending" if pend else "ready"
    elif have > 0:
        st = "partial"
    else:
        st = "missing"
    return st, have, total, miss


def ready(root, gid):
    return status(root, gid)[0] == "ready"


def size_gb(gid):
    g = group(gid)
    return g["size"] / float(1 << 30) if g else 0.0


def needed_for(engine, root):
    """The download groups an engine needs that are not on disk yet."""
    need = {"TRELLIS": ["TRELLIS"], "PIXAL3D": ["PIXAL3D"],
            "HUNYUAN": ["HUNYUAN"], "HI3D": ["HUNYUAN", "HI3D"],
            "KLEIN": ["KLEIN"]}.get(engine, [engine])
    return [g for g in need if not ready(root, g)]


def official_page(gid):
    """The maker's page of the group's main model."""
    g = group(gid)
    comp = manifest()["components"][g["comps"][0]]
    return comp.get("page", "")


# -- running the downloader ---------------------------------------------------

def busy():
    return DL["p"] is not None


def _reader(p, q):
    for raw in iter(p.stdout.readline, b""):
        q.put(raw.decode("utf-8", "replace").rstrip())
    q.put(None)


def start(root, gids, verify=False, redraw=None):
    if busy():
        return False
    os.makedirs(models_folder(root), exist_ok=True)
    args = [sys.executable, "-u", os.path.join(ADDON_DIR, "model_downloader.py"),
            MANIFEST, root, hf_cache(), os.path.join(ADDON_DIR, "model_patches")]
    if verify:
        args.append("--verify")
    args += list(gids)
    q = queue.Queue()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(args, cwd=root, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=NOWINDOW, env=env)
    threading.Thread(target=_reader, args=(p, q), daemon=True).start()
    DL.update(p=p, q=q, groups=list(gids), frac=0.0, error="", done="",
              text="Checking files" if verify else "Starting download",
              t0=time.time(), verify=verify, redraw=redraw, speed=0.0,
              left=-1.0)
    bpy.app.timers.register(_poll, first_interval=0.3)
    return True


def cancel():
    p = DL["p"]
    if p is not None:
        try:
            p.kill()
        except Exception:
            pass
    DL.update(p=None, q=None, text="", error="Download paused - press "
              "Download to continue where it stopped", groups=[])
    _CACHE["t"] = 0.0


def _poll():
    q = DL["q"]
    if q is None:
        return None
    end = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            end = True
            break
        if line.startswith("PROGRESS "):
            parts = line.split(" ", 2)
            try:
                DL["frac"] = float(parts[1])
            except ValueError:
                pass
            DL["text"] = parts[2] if len(parts) > 2 else ""
        elif line.startswith("SPEED "):
            try:
                _, sp, left = line.split()
                DL["speed"] = float(sp)
                DL["left"] = float(left)
            except ValueError:
                pass
        elif line.startswith("ERROR "):
            DL["error"] = line[6:]
        elif line.startswith("DONE"):
            DL["done"] = line[5:]
    if DL.get("redraw"):
        try:
            DL["redraw"]()
        except Exception:
            pass
    if end:
        p = DL["p"]
        try:
            p.wait(timeout=5)
        except Exception:
            pass
        DL.update(p=None, q=None, groups=[])
        _CACHE["t"] = 0.0
        if DL.get("redraw"):
            try:
                DL["redraw"]()
            except Exception:
                pass
        return None
    return 0.3


def open_folder(root):
    d = models_folder(root)
    os.makedirs(d, exist_ok=True)
    try:
        os.startfile(d)
    except Exception:
        try:
            bpy.ops.wm.path_open(filepath=d)
        except Exception:
            pass


def speed_text():
    """'9.8 MB/s' for the bar, and 'about 25 min left' for the line."""
    sp = DL.get("speed", 0.0)
    left = DL.get("left", -1.0)
    spd = ("%.1f MB/s" % (sp / 1e6)) if sp >= 1e5 else (
        "%.0f KB/s" % (sp / 1e3) if sp > 0 else "")
    if left < 0:
        eta = "estimating time left"
    elif left < 60:
        eta = "less than a minute left"
    elif left < 3600:
        eta = "about %d min left" % round(left / 60.0)
    else:
        eta = "about %d h %02d min left" % (left // 3600, (left % 3600) // 60)
    return spd, eta


NAMES = {"trellis2": "TRELLIS.2 (main model)",
         "trellis_ssdec": "TRELLIS shape decoder",
         "dinov3": "DINOv3 (reads the picture)",
         "birefnet": "BiRefNet (removes the background)",
         "pixal3d": "Pixal3D (main model)",
         "moge": "MoGe (reads the camera)",
         "hunyuan21": "Hunyuan3D 2.1 (main model)",
         "dinov2g": "DINOv2 giant (reads the picture)",
         "realesrgan": "Real-ESRGAN (sharpens the texture)",
         "hunyuan2mv": "Hunyuan3D 2mv (more views)",
         "hi3d": "Hi3D (main model)",
         "dinov2l": "DINOv2 large",
         "clipl": "CLIP large",
         "klein": "FLUX.2 Klein 4B",
         "marigold": "Marigold (takes the shine out of the texture)"}


def part_folder(cid, root):
    """The exact folder one part's files go into."""
    comp = manifest()["components"][cid]
    if comp["dest"] == "hfcache":
        return os.path.join(hf_cache(), "models--" + comp["repo"].replace(
            "/", "--"), "snapshots", comp["revision"])
    return os.path.join(root, *comp["dest"].split("/"))


def parts(gid, root):
    """[(id, name, page, folder, files, ready)] for the hand-download list"""
    out = []
    m = manifest()
    for cid in group(gid)["comps"]:
        comp = m["components"][cid]
        ok = all(_file_ok(comp, f, root) for f in comp["files"])
        out.append((cid, NAMES.get(cid, cid), comp.get("page", ""),
                    part_folder(cid, root),
                    [f["path"] for f in comp["files"] if not f.get("patched")
                     or True], ok))
    return out


def _fix_refs(root):
    """Files put into the Hugging Face cache by hand need its 'main' note."""
    for cid, comp in manifest()["components"].items():
        if comp["dest"] != "hfcache":
            continue
        if all(_file_ok(comp, f, root) for f in comp["files"]):
            ref = os.path.join(hf_cache(), "models--" + comp["repo"].replace(
                "/", "--"), "refs", "main")
            if not os.path.isfile(ref):
                try:
                    os.makedirs(os.path.dirname(ref), exist_ok=True)
                    open(ref, "w").write(comp["revision"])
                except OSError:
                    pass
