"""One-time install of the Image to 3D engine.

Run by the add-on with Blender's own Python:
    setup_engine.py <engine folder> <add-on folder> [--hunyuan | --extras]

    (nothing)    TRELLIS and Pixal3D
    --hunyuan    Hunyuan3D, then the Enhance tools (Enhance Texture, Enhance
                 Detail) that run next to it
    --extras     only the Enhance tools (for installs made before they existed)

Prints PROGRESS lines the add-on turns into its bar. Safe to run again: every
step that is already done is skipped, so an interrupted install picks up where
it stopped.
"""

import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = os.path.abspath(sys.argv[1])
ADDON = os.path.abspath(sys.argv[2])
HUNYUAN = "--hunyuan" in sys.argv
EXTRAS = "--extras" in sys.argv
NOWINDOW = 0x08000000 if os.name == "nt" else 0

UV_URL = ("https://github.com/astral-sh/uv/releases/download/0.9.5/"
          "uv-x86_64-pc-windows-msvc.zip")
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
FLASH_ATTN = ("https://github.com/mjun0812/flash-attention-prebuild-wheels/"
              "releases/download/v0.7.13/"
              "flash_attn-2.8.3%2Bcu130torch2.10-cp313-cp313-win_amd64.whl")
# Enhance Detail: Hi3D v1.0 on Python 3.10 / torch 2.7.0 cu128 (cu124 builds
# have no kernels for RTX 50 cards)
HI3D_TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
HI3D_BUILD = "cu128"
# the official Hi3D v1.0 inference code, pinned to the release checked here
HI3D_SRC = ("https://github.com/hi3d-official/Hi3Dv1.0/archive/"
            "13657ad09afaa92e350b461a1cd66399e44a59c4.zip")


def lock(name):
    """The exact package versions this environment was tested with."""
    return os.path.join(ADDON, "locks", name + ".txt")


def say(frac, text):
    print("PROGRESS %.3f %s" % (frac, text), flush=True)


def log(text):
    print("[setup] " + text, flush=True)


_CERTS = []


def cert_bundle():
    """Every certificate Windows trusts, roots and the in-between ones it has
    cached, in one file uv can read. Some networks and security programs put
    their own certificate in front of HTTPS sites; Windows knows it, uv on its
    own may not."""
    if _CERTS:
        return _CERTS[0]
    path = ""
    try:
        import ssl
        pems = []
        for store in ("ROOT", "CA"):
            try:
                for der, enc, _trust in ssl.enum_certificates(store):
                    if enc == "x509_asn":
                        pems.append(ssl.DER_cert_to_PEM_cert(der))
            except Exception:
                pass
        if pems:
            path = os.path.join(ROOT, "certs.pem")
            with open(path, "w") as f:
                f.write("".join(pems))
    except Exception as exc:
        log("certificates: %s" % exc)
    _CERTS.append(path)
    return path


def tls_probe(host):
    """Who signed the certificate this PC sees for a site: shows in the report
    when something on the network stands in front of HTTPS."""
    try:
        import socket
        import ssl
        ctx = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=15) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ss:
                iss = dict(x[0] for x in ss.getpeercert()["issuer"])
        log("TLS %s issued by %s / %s" % (host, iss.get("organizationName", "?"),
                                          iss.get("commonName", "?")))
    except Exception as exc:
        log("TLS %s: %s" % (host, str(exc)[:160]))


def env():
    e = dict(os.environ)
    certs = cert_bundle()
    if certs:
        e["SSL_CERT_FILE"] = certs
    e["UV_PYTHON_INSTALL_DIR"] = os.path.join(ROOT, "python")
    e["UV_CACHE_DIR"] = os.path.join(ROOT, "uv_cache")
    e["UV_LINK_MODE"] = "copy"
    e["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # use the certificates Windows trusts: an antivirus or company network
    # that checks HTTPS adds its own, which uv's built-in list doesn't know
    e["UV_NATIVE_TLS"] = "1"
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def run(cmd):
    log(" ".join(os.path.basename(c) if i == 0 else c
                 for i, c in enumerate(cmd))[:120])
    p = subprocess.Popen(cmd, env=env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, creationflags=NOWINDOW)
    for raw in iter(p.stdout.readline, b""):
        line = raw.decode("utf-8", "replace").strip()
        if line.startswith("PROGRESS "):
            print(line, flush=True)
        elif line:
            log(line[:100])
    if p.wait() != 0:
        raise SystemExit("ERROR install step failed: %s" %
                         os.path.basename(cmd[0]))


def hf_cache():
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(k):
            return os.environ[k]
    home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface")
    return os.path.join(home, "hub")


def download_models(group):
    """The models come from their makers' official pages, checked file by
    file (the same downloader as the add-on's Downloads section)."""
    run([sys.executable, "-u", os.path.join(ADDON, "model_downloader.py"),
         os.path.join(ADDON, "models_manifest.json"), ROOT, hf_cache(),
         os.path.join(ADDON, "model_patches"), group])


def has_nvidia():
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           text=True, creationflags=NOWINDOW, timeout=30)
        return "GPU" in r.stdout
    except Exception:
        return False


def venv(uv, name, version):
    py = os.path.join(ROOT, name, "Scripts", "python.exe")
    if not os.path.exists(py):
        run([uv, "python", "install", version])
        run([uv, "venv", os.path.join(ROOT, name), "--python", version])
    return py


def copy_engine(name, ignore=("__pycache__",)):
    dst = os.path.join(ROOT, name)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(ADDON, name), dst,
                    ignore=shutil.ignore_patterns(*ignore))


def install_klein(uv, f0, f1):
    """Enhance Texture (FLUX.2 [klein]) in its own environment."""
    say(f0, "Installing Enhance Texture")
    py = venv(uv, "env_klein", "3.13")

    def pip(*args):
        run([uv, "pip", "install", "--python", py, *args])

    pip("torch==2.10.0", "torchvision==0.25.0", "--index-url", TORCH_INDEX)
    say(f0 + (f1 - f0) * 0.6, "Installing Enhance Texture")
    pip("--no-deps", "-r", lock("klein"))
    open(os.path.join(ROOT, "INSTALLED_KLEIN"), "w").write("1")


def install_hi3d(uv, f0, f1):
    """Enhance Detail (Hi3D v1.0) in its own environment."""
    say(f0, "Installing Enhance Detail")
    py = venv(uv, "env_hi3d", "3.10")

    def pip(*args):
        run([uv, "pip", "install", "--python", py, *args])

    pip("torch==2.7.0", "torchvision==0.22.0", "--index-url",
        HI3D_TORCH_INDEX)
    say(f0 + (f1 - f0) * 0.4, "Installing Enhance Detail")
    pip("--no-deps", "-r", lock("hi3d"))
    say(f0 + (f1 - f0) * 0.7, "Installing Enhance Detail")
    src = os.path.join(ROOT, "engine_hi3d_src")
    pkg = os.path.join(src, "Hi3Dv1.0-" + HI3D_SRC.rsplit("/", 1)[1][:-4])
    if not os.path.isfile(os.path.join(pkg, "pyproject.toml")):
        data = urllib.request.urlopen(urllib.request.Request(
            HI3D_SRC, headers={"User-Agent": "img3d"}), timeout=600).read()
        os.makedirs(src, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(src)
    pip("--no-deps", "--reinstall-package", "sparc3d", pkg)
    copy_engine("engine_hi3d")
    open(os.path.join(ROOT, "INSTALLED_HI3D"), "w").write(HI3D_BUILD)


def install_extras(uv, f0, f1):
    """The two Enhance tools. A failure here leaves Hunyuan working; the
    Enhance section then offers to install them again."""
    mid = f0 + (f1 - f0) * 0.4
    ok = True
    for fn, a, b in ((install_klein, f0, mid), (install_hi3d, mid, f1)):
        try:
            fn(uv, a, b)
        except SystemExit as exc:
            ok = False
            log("could not install %s: %s" % (fn.__name__[8:], exc))
    return ok


def install_hunyuan(uv):
    py = os.path.join(ROOT, "env_hy", "Scripts", "python.exe")
    say(0.03, "Installing Python")
    if not os.path.exists(py):
        run([uv, "python", "install", "3.13"])
        run([uv, "venv", os.path.join(ROOT, "env_hy"), "--python", "3.13"])

    def pip(*args):
        run([uv, "pip", "install", "--python", py, *args])

    say(0.06, "Installing PyTorch")
    pip("torch==2.10.0", "torchvision==0.25.0", "--index-url", TORCH_INDEX)
    say(0.15, "Installing libraries")
    pip("--no-deps", "-r", lock("hunyuan"))
    site = os.path.join(ROOT, "env_hy", "Lib", "site-packages",
                        "custom_rasterizer")
    shutil.rmtree(site, ignore_errors=True)
    shutil.copytree(os.path.join(ADDON, "engine_hy", "_shim",
                                 "custom_rasterizer"), site)

    say(0.20, "Copying engine")
    copy_engine("engine_hy", ("__pycache__", "_shim"))
    # Tencent's Hunyuan3D-2.1 code is not in the add-on: it comes straight
    # from Tencent's repository, and the add-on makes its edits to it
    say(0.18, "Downloading Hunyuan code from Tencent")
    sys.path.insert(0, ADDON)
    import hy_patches
    hy_patches.fetch_code(os.path.join(ROOT, "engine_hy"),
                          own_dir=os.path.join(ADDON, "engine_hy", "hy_own"))
    open(os.path.join(ROOT, "INSTALLED_HY"), "w").write("1")

    extras_ok = install_extras(uv, 0.22, 0.45)

    say(0.45, "Downloading Hunyuan")
    download_models("HUNYUAN")
    say(1.0, "Installed")
    if not extras_ok:
        print("ERROR Hunyuan is installed, but the Enhance tools could not "
              "be set up - check the internet connection and press Install "
              "Enhance Tools", flush=True)
        sys.exit(1)
    print("DONE " + ROOT, flush=True)


def main():
    os.makedirs(ROOT, exist_ok=True)
    if not has_nvidia():
        print("ERROR An NVIDIA graphics card is needed", flush=True)
        sys.exit(1)

    say(0.01, "Getting installer")
    for host in ("github.com", "pypi.org"):
        tls_probe(host)
    uv = os.path.join(ROOT, "uv", "uv.exe")
    if not os.path.exists(uv):
        data = urllib.request.urlopen(urllib.request.Request(
            UV_URL, headers={"User-Agent": "img3d"}), timeout=300).read()
        zipfile.ZipFile(io.BytesIO(data)).extractall(os.path.join(ROOT, "uv"))
    if EXTRAS:
        ok = install_extras(uv, 0.03, 0.99)
        say(1.0, "Installed")
        if not ok:
            print("ERROR the Enhance tools could not be set up - check the "
                  "internet connection and try again", flush=True)
            sys.exit(1)
        print("DONE " + ROOT, flush=True)
        return
    if HUNYUAN:
        install_hunyuan(uv)
        return

    py = os.path.join(ROOT, "env", "Scripts", "python.exe")
    say(0.03, "Installing Python")
    if not os.path.exists(py):
        run([uv, "python", "install", "3.13"])
        run([uv, "venv", os.path.join(ROOT, "env"), "--python", "3.13"])

    def pip(*args):
        run([uv, "pip", "install", "--python", py, *args])

    say(0.06, "Installing PyTorch")
    pip("torch==2.10.0", "torchvision==0.25.0", "--index-url", TORCH_INDEX)
    say(0.18, "Installing libraries")
    pip("--no-deps", FLASH_ATTN)
    pip("--no-deps", "-r", lock("trellis"))
    wheels = os.path.join(ADDON, "engine", "wheels")
    whl = sorted(os.path.join(wheels, f) for f in os.listdir(wheels)
                 if f.endswith(".whl"))
    if whl:
        pip("--no-deps", *whl)

    say(0.25, "Copying engine")
    copy_engine("engine", ("wheels", "__pycache__"))

    say(0.27, "Downloading AI model")
    download_models("TRELLIS")

    open(os.path.join(ROOT, "INSTALLED"), "w").write("1")
    say(1.0, "Installed")
    print("DONE " + ROOT, flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            msg = str(e.code)
            print(msg if msg.startswith("ERROR") else "ERROR " + msg,
                  flush=True)
        raise
    except Exception as e:
        print("ERROR %s" % e, flush=True)
        sys.exit(1)
