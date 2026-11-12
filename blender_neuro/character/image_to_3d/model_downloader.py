"""Downloads the model files of one or more parts of Image to 3D straight
from their makers' official pages (Hugging Face / GitHub), checks every file
against the size and SHA-256 its maker published, and puts it where the
engines look for it. Nothing comes from our own servers.

Run by the add-on with Blender's own Python (only the standard library):

    model_downloader.py <manifest> <engine folder> <hf cache> <patch dir>
                        [--verify] GROUP [GROUP ...]

Prints PROGRESS <0..1> <text>, then DONE or ERROR <message>. Safe to stop at
any time: a half-downloaded file is kept as <name>.part and continued next
time.
"""

import hashlib
import threading
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shutil
import subprocess
import sys
import time
import urllib.request

CHUNK = 1 << 20
LOCK = threading.Lock()
WORKERS = 4          # files fetched at the same time


def say(frac, text):
    print("PROGRESS %.4f %s" % (max(0.0, min(1.0, frac)), text), flush=True)


def gb(n):
    return n / float(1 << 30)


def target(comp, f, root, cache):
    """Where one file lives on disk."""
    if comp["dest"] == "hfcache":
        repo = "models--" + comp["repo"].replace("/", "--")
        return os.path.join(cache, repo, "snapshots", comp["revision"],
                            *f["path"].split("/"))
    return os.path.join(root, *comp["dest"].split("/"), *f["path"].split("/"))


def present(comp, f, root, cache):
    """True when the file (or what it was converted into) is on disk."""
    p = target(comp, f, root, cache)
    if f.get("patched") and os.path.isfile(p):
        return True
    if os.path.isfile(p) and os.path.getsize(p) == f["size"]:
        return True
    conv = f.get("convert_to")
    if conv:
        c = target(comp, {"path": conv}, root, cache)
        return os.path.isfile(c) and os.path.isfile(c + ".ok")
    return False


def url_of(comp, f):
    if f.get("url"):
        return f["url"]
    return "https://huggingface.co/%s/resolve/%s/%s" % (
        comp["repo"], comp["revision"], f["path"])


def sha256_of(path, done_cb=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK * 8)
            if not b:
                break
            h.update(b)
            if done_cb:
                done_cb(len(b))
    return h.hexdigest()


class Progress:
    """Bytes done, the speed over the last few seconds and the time left."""

    def __init__(self, total, label):
        self.total = max(1, total)
        self.done = 0
        self.label = label
        self.last = 0.0
        self.hist = []          # (time, bytes done) over the last seconds

    def add(self, n, force=False):
        with LOCK:
            self.done += n
        now = time.time()
        if force or now - self.last > 0.5:
            self.last = now
            self.hist.append((now, self.done))
            while len(self.hist) > 2 and now - self.hist[0][0] > 8.0:
                self.hist.pop(0)
            t0, d0 = self.hist[0]
            speed = (self.done - d0) / (now - t0) if now - t0 > 0.5 else 0.0
            left = (self.total - self.done) / speed if speed > 1 else -1
            print("SPEED %.0f %.0f" % (speed, left), flush=True)
            say(self.done / float(self.total), "%s%.1f / %.1f GB" % (
                self.label, gb(self.done), gb(self.total)))


def fetch(url, path, size, prog):
    """Download url to path, continuing a .part file if there is one."""
    part = path + ".part"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    have = os.path.getsize(part) if os.path.isfile(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    prog.add(have)
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "3DRedbox-ImageTo3D"})
            if have:
                req.add_header("Range", "bytes=%d-" % have)
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:
                    # the server ignored the range: start this file again
                    prog.add(-have)
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                with open(part, mode) as out:
                    while True:
                        b = r.read(CHUNK)
                        if not b:
                            break
                        out.write(b)
                        have += len(b)
                        prog.add(len(b))
            if have == size:
                return part
            raise IOError("got %d of %d bytes" % (have, size))
        except Exception as exc:
            print("[download] retry %s: %s" % (os.path.basename(path), exc),
                  flush=True)
            time.sleep(min(30, 3 * (attempt + 1)))
    raise SystemExit("ERROR Could not download %s - check the internet "
                     "connection and press Download again" %
                     os.path.basename(path))


def patch(comp, root, patch_dir):
    """The add-on's small edits to a downloaded file (hy_patches.py): no
    changed copy of anyone's file is shipped with the add-on."""
    import hy_patches
    for key, dst in comp.get("patches", []):
        d = os.path.join(root, *comp["dest"].split("/"), *dst.split("/"))
        if os.path.isfile(d):
            hy_patches.apply(key, d)


def post(comp, root):
    """Steps that need the engine's own Python (it has PyTorch)."""
    steps = comp.get("post", [])
    if not steps:
        return
    py = os.path.join(root, "env_hy" if "convert_bins" in steps else "env",
                      "Scripts", "python.exe")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "model_post.py")
    folder = os.path.join(root, *comp["dest"].split("/"))
    flag = os.path.join(folder, ".post_pending")
    if not os.path.isfile(py):
        # the engine is not installed yet: its installer finishes this
        open(flag, "w").write(",".join(steps))
        return
    for step in steps:
        say(0.995, "Preparing files")
        r = subprocess.run([py, "-u", script, step, folder], capture_output=True,
                           text=True, creationflags=0x08000000
                           if os.name == "nt" else 0)
        sys.stdout.write(r.stdout[-2000:])
        if r.returncode != 0:
            raise SystemExit("ERROR Could not prepare %s: %s" % (
                comp.get("repo", "files"), (r.stdout + r.stderr)
                .strip().splitlines()[-1:] or ["unknown"]))
    if os.path.isfile(flag):
        os.remove(flag)


_HELD = []


def hold_lock(root):
    """One downloader at a time per engine folder: Install Engine fetches
    the models too, and two downloads writing the same file break both."""
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    fh = open(os.path.join(root, "models", ".download.lock"), "a+")
    told = False
    while True:
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _HELD.append(fh)
            return
        except OSError:
            if not told:
                say(0.0, "Waiting for the other download to finish")
                told = True
            time.sleep(2.0)


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    verify = "--verify" in sys.argv
    manifest, root, cache, patch_dir = a[:4]
    hold_lock(root)
    groups = a[4:]
    m = json.load(open(manifest, encoding="utf-8"))
    by_id = {g["id"]: g for g in m["groups"]}
    comps = []
    for gid in groups:
        for c in by_id[gid]["comps"]:
            if c not in comps:
                comps.append(c)
    name = " + ".join(by_id[g]["name"] for g in groups)

    if verify:
        todo = [(c, f) for c in comps for f in m["components"][c]["files"]
                if f.get("sha256") and not f.get("patched") and os.path.isfile(
                    target(m["components"][c], f, root, cache))]
        prog = Progress(sum(f["size"] for _, f in todo), "Checking files  ")
        bad = []
        for c, f in todo:
            p = target(m["components"][c], f, root, cache)
            if os.path.getsize(p) != f["size"] or \
                    sha256_of(p, prog.add) != f["sha256"]:
                bad.append(p)
                os.replace(p, p + ".bad")
        prog.add(0, True)
        if bad:
            print("ERROR %d file(s) were damaged and were removed - press "
                  "Download to get them again" % len(bad), flush=True)
            sys.exit(1)
        print("DONE verified", flush=True)
        return

    jobs = []
    for c in comps:
        comp = m["components"][c]
        for f in comp["files"]:
            if not present(comp, f, root, cache):
                jobs.append((c, comp, f))
    total = sum(f["size"] for _, _, f in jobs)
    prog = Progress(total, "")
    say(0.0, "0 / %.1f GB" % gb(total))
    # biggest first, several at a time: one connection alone is slow
    jobs.sort(key=lambda j: -j[2]["size"])
    errors = []

    def one(job):
        c, comp, f = job
        path = target(comp, f, root, cache)
        part = fetch(url_of(comp, f), path, f["size"], prog)
        if f.get("sha256") and sha256_of(part) != f["sha256"]:
            os.remove(part)
            raise SystemExit("ERROR %s did not match the official file - "
                             "press Download to try again" %
                             os.path.basename(path))
        os.replace(part, path)

    def worker():
        while not errors:
            with LOCK:
                if not jobs:
                    return
                job = jobs.pop(0)
            try:
                one(job)
            except SystemExit as e:
                errors.append(str(e.code))
            except Exception as e:
                errors.append("ERROR %s" % e)

    threads = [threading.Thread(target=worker, daemon=True)
               for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise SystemExit(errors[0])
    for c in comps:
        comp = m["components"][c]
        if comp["dest"] == "hfcache":
            ref = os.path.join(cache, "models--" + comp["repo"].replace(
                "/", "--"), "refs")
            os.makedirs(ref, exist_ok=True)
            open(os.path.join(ref, "main"), "w").write(comp["revision"])
        patch(comp, root, patch_dir)
        post(comp, root)
    say(1.0, "Downloaded")
    print("DONE " + name, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            msg = str(e.code)
            print(msg if msg.startswith("ERROR") else "ERROR " + msg,
                  flush=True)
            sys.exit(1)
    except Exception as e:
        print("ERROR %s" % e, flush=True)
        sys.exit(1)
