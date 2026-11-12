bl_info = {
    "name": "Image to 3D 2",
    "author": "Milad Kambari - 3DRedbox Studio",
    "version": (2, 0, 8),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > Image to 3D 2",
    "description": "Turn one picture into a textured 3D model with TRELLIS.2, "
                   "running on your own GPU",
    "category": "3D View",
}

import os
import sys
import time
import queue
import threading
import subprocess
import importlib

import bpy
from bpy.props import (StringProperty, EnumProperty, IntProperty,
                       FloatProperty, BoolProperty)

from . import card
importlib.reload(card)
from . import models as dl
importlib.reload(dl)
from . import updates
importlib.reload(updates)
from . import canvas
importlib.reload(canvas)

VERSION = "v%d.%d.%d" % bl_info["version"]
ARTSTATION_URL = "https://www.artstation.com/milad_kambari/store"
SUPERHIVE_URL = "https://superhivemarket.com/creators/3dredbox"
DISCORD_URL = "https://discord.gg/EC8d7nPsYj"

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))
NOWINDOW = 0x08000000 if os.name == "nt" else 0

CARD_JOB = None      # the generation running now, as the card shows it
SETUP_JOB = None     # the engine install running now
_PROC = {"p": None, "q": None, "kind": None, "out": None, "lines": []}


# -- where the engine lives ---------------------------------------------------

# All-in-One: у вшитой копии нет своего ключа в preferences.addons
# (bl_idname обязан совпадать с id расширения) — движок ищем в стандартном
# пути 3DRedbox, а настройки берём у мастер-аддона.
_DEFAULT_ENGINE_DIR = os.path.join(os.path.expanduser("~"), "3DRedbox",
                                   "ImageTo3D")

# вшитая копия внутри neuro_character_pipeline (не самостоятельный аддон)
VENDORED = "neuro_character_pipeline" in (__package__ or "")

def _prefs():
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except Exception:
        pass
    if VENDORED:
        # All-in-One: настройки движка живут в мастер-аддоне
        # (Neuro Character Pipeline -> Preferences -> Engine Folder)
        try:
            addons = bpy.context.preferences.addons
            for k in addons.keys():
                if k.rsplit(".", 1)[-1] == "neuro_character_pipeline":
                    return addons[k].preferences
        except Exception:
            pass
    return None


def engine_dir():
    p = _prefs()
    d = getattr(p, "engine_dir", "") if p is not None else ""
    return bpy.path.abspath(d) if d else _DEFAULT_ENGINE_DIR


def engine_python(engine="TRELLIS"):
    env = {"HUNYUAN": "env_hy", "HI3D": "env_hi3d",
           "KLEIN": "env_klein", "CANVAS": "env_qwen"}.get(engine, "env")
    return os.path.join(engine_dir(), env, "Scripts", "python.exe")


def engine_script(engine="TRELLIS"):
    if engine == "HI3D":
        return os.path.join(engine_dir(), "engine_hi3d", "hi3d_server.py")
    if engine == "HUNYUAN":
        return os.path.join(engine_dir(), "engine_hy", "generate_hy.py")
    return os.path.join(engine_dir(), "engine", "generate.py")


def hy_mv_ready():
    d = engine_dir()
    return bool(d) and dl.ready(d, "HUNYUAN_MV")


def pixal_mv_ready():
    d = engine_dir()
    ck = os.path.join(d, "models", "TencentARC", "Pixal3D", "ckpts")
    need = ("ss_flow_img_dit_1_3B_64_bf16_mv",
            "slat_flow_img2shape_dit_1_3B_512_bf16_mv",
            "slat_flow_img2shape_dit_1_3B_1024_bf16_mv",
            "slat_flow_imgshape2tex_dit_1_3B_1024_bf16_mv")
    return bool(d) and all(os.path.isfile(os.path.join(ck, n + ".safetensors"))
                           for n in need)


_GPU_CC = {}


def gpu_compute_cap():
    """The graphics card's generation, e.g. 8.9 for RTX 40, 12.0 for RTX 50."""
    if "cc" not in _GPU_CC:
        cc = 0.0
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap",
                                "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=15, creationflags=NOWINDOW)
            cc = max(float(x) for x in r.stdout.split() if x.strip())
        except Exception:
            pass
        _GPU_CC["cc"] = cc
    return _GPU_CC["cc"]


def _hi3d_outdated(d):
    """Enhance Detail was first installed with a PyTorch that has no code for
    RTX 50 cards: on those it must be installed again (the newer one)."""
    try:
        build = open(os.path.join(d, "INSTALLED_HI3D")).read().strip()
    except Exception:
        return False
    return build != "cu128" and gpu_compute_cap() >= 10.0


def software_ready(engine="TRELLIS"):
    """The engine's own Python and scripts are installed."""
    d = engine_dir()
    if not d:
        return False
    if engine == "HI3D":
        # Hunyuan makes the first shape and paints the refined one
        return (software_ready("HUNYUAN") and
                os.path.isfile(engine_python("HI3D")) and
                os.path.isfile(engine_script("HI3D")) and
                os.path.isfile(os.path.join(d, "INSTALLED_HI3D")) and
                not _hi3d_outdated(d))
    if engine == "KLEIN":
        # Enhance Texture runs next to Hunyuan, in its own environment
        return (software_ready("HUNYUAN") and
                os.path.isfile(engine_python("KLEIN")) and
                os.path.isfile(os.path.join(d, "INSTALLED_KLEIN")))
    if engine == "CANVAS":
        # the Canvas picture models (Qwen) in their own environment, and
        # Klein, which draws the parts
        return (os.path.isfile(engine_python("CANVAS")) and
                os.path.isfile(os.path.join(d, "engine_qwen", "render.py")) and
                os.path.isfile(os.path.join(d, "INSTALLED_CANVAS")) and
                os.path.isfile(engine_python("KLEIN")) and
                os.path.isfile(os.path.join(d, "INSTALLED_KLEIN")))
    if engine == "PIXAL3D":
        return software_ready("TRELLIS")
    if engine == "HUNYUAN":
        need = (engine_python(engine), engine_script(engine),
                os.path.join(d, "INSTALLED_HY"))
    else:
        need = (engine_python(), engine_script(), os.path.join(d, "INSTALLED"))
    return all(os.path.exists(p) for p in need)


def models_missing(engine="TRELLIS"):
    """The download groups (Downloads section) this engine still needs."""
    d = engine_dir()
    if not d:
        return []
    try:
        return dl.needed_for(engine, d)
    except Exception as exc:
        print("[Image to 3D] model check:", exc)
        return []


def engine_ready(engine="TRELLIS"):
    return software_ready(engine) and not models_missing(engine)


def klein_ready():
    """Enhance Texture is installed and its models are downloaded."""
    d = engine_dir()
    return bool(d) and software_ready("KLEIN") and dl.ready(d, "KLEIN")


def canvas_ready():
    """The Canvas tools are installed and their models downloaded."""
    d = engine_dir()
    return bool(d) and software_ready("CANVAS") and not \
        dl.needed_for("CANVAS", d)


def extras_missing():
    """Hunyuan is installed but one of the Enhance tools is not."""
    return software_ready("HUNYUAN") and not (software_ready("KLEIN") and
                                              software_ready("HI3D"))


def show_plain_panel():
    print("[Image to 3D] card failed to draw")


# -- settings -----------------------------------------------------------------

def _redraw():
    try:
        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()
    except Exception:
        pass


class IMG3D_Settings(bpy.types.PropertyGroup):
    engine: EnumProperty(
        name="Engine",
        items=(("TRELLIS", "TRELLIS", "Microsoft TRELLIS.2"),
               ("HUNYUAN", "Hunyuan", "Tencent Hunyuan3D 2.1"),
               ("PIXAL3D", "Pixal3D", "TencentARC Pixal3D"),
               ("HI3D", "Hi3D", "Hunyuan shape refined by Hi3D (Hitem3D v1)")),
        default="TRELLIS", update=lambda s, c: _redraw())
    faces_hy: IntProperty(name="Faces", default=40000, min=1000,
                          max=50000000,
                          description="How many faces the finished model has")
    image_path: StringProperty(name="Image", subtype="FILE_PATH",
                               update=lambda s, c: _redraw())
    quality: EnumProperty(
        name="Quality",
        items=(("FAST", "Fast", "512 voxels: quick preview of the shape"),
               ("HIGH", "High", "1024 voxels: the normal setting"),
               ("ULTRA", "Ultra", "1536 voxels: finest detail, slowest, "
                                  "needs the most GPU memory")),
        default="HIGH")
    faces: IntProperty(name="Faces", default=200000, min=1000, max=50000000,
                       description="How many faces the finished model has")
    view_1: StringProperty(default="", subtype="FILE_PATH")
    view_2: StringProperty(default="", subtype="FILE_PATH")
    view_3: StringProperty(default="", subtype="FILE_PATH")
    use_texture: EnumProperty(
        name="Texture",
        items=(("ON", "On", "Paint and bake a texture onto the model"),
               ("OFF", "Off", "Shape only, no texture: much faster")),
        default="ON")
    auto_detail: EnumProperty(
        name="Enhance 3D Model",
        items=(("ON", "On", "After the shape is made, Hi3D rebuilds it with "
                            "sharp edges and fine detail (about 5 minutes)"),
               ("OFF", "Off", "Keep the shape as the engine made it")),
        default="OFF", update=lambda s, c: _redraw())
    auto_enhance: EnumProperty(
        name="Enhance Texture",
        items=(("ON", "On", "Enhance the texture right after it is painted"),
               ("OFF", "Off", "Keep the painted texture as it is")),
        default="OFF", update=lambda s, c: _redraw())
    detail_tex: EnumProperty(
        name="Repaint Texture",
        items=(("ON", "On", "Paint a new texture on the detailed model"),
               ("OFF", "Off", "Only the detailed shape, without texture")),
        default="ON", update=lambda s, c: _redraw())
    creativity: FloatProperty(
        name="Creativity", default=0.0, min=0.0, max=1.0,
        description="How much new surface detail the enhancer invents: 0 "
                    "stays close to the painted texture, 1 adds heavy wear, "
                    "scratches and weathering")
    enhance_mode: EnumProperty(
        name="Enhance",
        items=(("FAST", "Fast", "Each view repainted in one go: about half "
                                "a minute"),
               ("HIGH", "High", "Each view repainted in tiles: the most "
                                "detail, about two minutes")),
        default="HIGH")
    tex_from: EnumProperty(
        name="Paint With",
        items=(("HUNYUAN", "Hunyuan", "Hunyuan paints the texture from the "
                                      "front picture"),
               ("PIXAL3D", "Pixal3D", "Pixal3D paints the Hunyuan model from "
                                      "every picture (front, back, sides)")),
        default="HUNYUAN")
    texture: EnumProperty(
        name="Texture",
        items=(("1024", "1K", ""), ("2048", "2K", ""), ("4096", "4K", "")),
        default="2048")
    seed: IntProperty(name="Seed", default=0, min=0, max=2147483647,
                      description="Another number gives another take on the "
                                  "same picture")
    status: StringProperty(default="")


class IMG3D_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    engine_dir: StringProperty(
        name="Engine Folder", subtype="DIR_PATH",
        default=os.path.join(os.path.expanduser("~"), "3DRedbox",
                             "ImageTo3D"),
        description="Where the AI engine and its model files live "
                    "(about 25 GB)",
        update=lambda s, c: _save_config())
    classic_panel: BoolProperty(name="Classic Panel", default=False)
    accent_hue: FloatProperty(name="Accent Hue", default=0.0901, min=0.0,
                              max=1.0, subtype="FACTOR")
    accent_sat: FloatProperty(name="Accent Saturation", default=0.922,
                              min=0.0, max=1.0, subtype="FACTOR")
    corner: FloatProperty(name="Corner Rounding", default=8.0, min=0.0,
                          max=16.0)
    ui_bright: FloatProperty(name="Card Brightness", default=1.0, min=0.4,
                             max=2.0, update=lambda s, c: card.retint())
    welcomed: BoolProperty(name="Welcome Shown", default=False,
                           description="The hello has been said once already",
                           update=lambda s, c: _save_config())
    gentle_download: BoolProperty(
        name="Gentle Download", default=False,
        description="For older computers: download one file at a time and "
                    "a bit slower, so the computer stays smooth",
        update=lambda s, c: _save_config())
    auto_update: BoolProperty(
        name="Auto Update", default=False,
        description="When Blender starts, install a newer version of "
                    "Image to 3D from Superhive by itself",
        update=lambda s, c: _save_config())

    def draw(self, context):
        col = self.layout.column()
        row = col.row()
        row.operator("img3d.welcome", icon="INFO")
        row.operator("img3d.save_report", icon="FILE_TEXT")
        col.prop(self, "engine_dir")
        col.prop(self, "gentle_download")
        col.label(text="Engine ready" if engine_ready()
                  else "Engine not installed", icon="CHECKMARK"
                  if engine_ready() else "ERROR")
        col.separator()
        col.prop(self, "accent_hue", slider=True)
        col.prop(self, "accent_sat", slider=True)
        col.prop(self, "corner")
        col.prop(self, "ui_bright", slider=True)


class IMG3D_OT_welcome(bpy.types.Operator):
    """The hello, said once when the add-on is first switched on. Drawn with
    the card's own paint: one panel in the middle of the viewport, one
    button on it, and a cross to close it"""
    bl_idname = "img3d.welcome"
    bl_label = "Welcome"
    bl_description = "Show the welcome message again"

    def invoke(self, context, event):
        if context.area is None or context.area.type != "VIEW_3D":
            self.report({"INFO"}, "Image to 3D: thank you. Feedback is very "
                                  "welcome on Discord: " + DISCORD_URL)
            return {"CANCELLED"}
        if not card.welcome_show(context):
            return {"CANCELLED"}
        context.window_manager.modal_handler_add(self)
        _redraw()
        return {"RUNNING_MODAL"}

    def _end(self, context):
        card.welcome_hide()
        _redraw()

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self._end(context)
            return {"CANCELLED"}
        if event.type == "MOUSEMOVE":
            hit = card.welcome_hit(context, event)
            hot = hit if hit in {"btn", "close"} else None
            if hot != card.WELCOME["hover"]:
                card.WELCOME["hover"] = hot
                _redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            hit = card.welcome_hit(context, event)
            if hit == "btn":
                self._end(context)
                try:
                    bpy.ops.wm.url_open(url=DISCORD_URL)
                except Exception:
                    pass
                return {"FINISHED"}
            if hit in {"close", None}:
                self._end(context)
                return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def execute(self, context):
        return self.invoke(context, None)


def _welcome_due(_tries=[0]):
    """Say hello once, a moment after the add-on comes up - when there is a
    window with a 3D view to paint in; quietly gives up if one never comes."""
    try:
        if bpy.app.background:
            return None
        prefs = bpy.context.preferences.addons[__package__].preferences
        if prefs.welcomed:
            return None
        win = area = region = None
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type != "VIEW_3D":
                    continue
                for r in a.regions:
                    if r.type == "WINDOW":
                        win, area, region = w, a, r
                        break
                if region is not None:
                    break
            if region is not None:
                break
        if region is None:
            _tries[0] += 1
            return 1.5 if _tries[0] < 8 else None
        prefs.welcomed = True
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            pass
        with bpy.context.temp_override(window=win, area=area, region=region):
            bpy.ops.img3d.welcome("INVOKE_DEFAULT")
    except Exception:
        pass
    return None


# -- running something in the engine ----------------------------------------

_SYNCED = set()


def sync_engine_code():
    """The engine's scripts are copied into the engine folder when it is
    installed. After an add-on update, bring the new ones over too, so a fix
    in the engine reaches people without installing the engine again."""
    d = engine_dir()
    if not d:
        return
    import filecmp
    import shutil
    for name in ("engine", "engine_hy", "engine_hi3d", "engine_qwen"):
        src = os.path.join(ADDON_DIR, name)
        dst = os.path.join(d, name)
        if name in _SYNCED or not (os.path.isdir(src) and os.path.isdir(dst)):
            continue
        n = 0
        for r, ds, fs in os.walk(src):
            ds[:] = [x for x in ds if x not in ("__pycache__", "wheels",
                                                "_shim")]
            for f in fs:
                if not f.endswith((".py", ".json", ".yaml", ".txt")):
                    continue
                a = os.path.join(r, f)
                b = os.path.join(dst, os.path.relpath(a, src))
                try:
                    if os.path.isfile(b) and filecmp.cmp(a, b, shallow=False):
                        continue
                    os.makedirs(os.path.dirname(b), exist_ok=True)
                    shutil.copyfile(a, b)
                    n += 1
                except Exception as exc:
                    print("[Image to 3D] engine update %s: %s" % (f, exc))
        if n:
            print("[Image to 3D] updated %d engine file(s) in %s" % (n, name))
        _SYNCED.add(name)


def retire_nvdiffrast():
    """Older versions installed NVIDIA's nvdiffrast and nvdiffrec, whose
    licence allows only non-commercial use, and MeshLib, which needs a paid
    licence for commercial use (and was never called). The engines now use the add-on's
    own rasterizer: take those packages out of the engine folder, and give
    Hunyuan the custom_rasterizer that no longer needs them."""
    d = engine_dir()
    if not d:
        return
    import shutil
    for env in ("env", "env_hy"):
        sp = os.path.join(d, env, "Lib", "site-packages")
        if not os.path.isdir(sp):
            continue
        for f in os.listdir(sp):
            if f.lower().startswith(("nvdiffrast", "nvdiffrec_render",
                                      "meshlib")):
                p = os.path.join(sp, f)
                try:
                    if os.path.isdir(p):
                        shutil.rmtree(p)
                    else:
                        os.remove(p)
                    print("[Image to 3D] removed", f, "from", env)
                except Exception as exc:
                    # in use by a running engine: again on the next start
                    print("[Image to 3D] could not remove %s: %s" % (f, exc))
    src = os.path.join(ADDON_DIR, "engine_hy", "_shim", "custom_rasterizer")
    dst = os.path.join(d, "env_hy", "Lib", "site-packages",
                       "custom_rasterizer")
    if os.path.isdir(src) and os.path.isdir(os.path.dirname(dst)):
        import filecmp
        for f in os.listdir(src):
            a = os.path.join(src, f)
            b = os.path.join(dst, f)
            if not f.endswith(".py"):
                continue
            try:
                if not (os.path.isfile(b) and filecmp.cmp(a, b,
                                                          shallow=False)):
                    os.makedirs(dst, exist_ok=True)
                    shutil.copyfile(a, b)
            except Exception as exc:
                print("[Image to 3D] custom_rasterizer %s: %s" % (f, exc))


def update_engine_binaries():
    """The engine's compiled parts come from the wheels in engine/wheels and
    are put in place when the engine is installed. When an add-on update
    brings rebuilt ones (for example with support for more graphics cards),
    swap them in, so nobody has to install the engine again."""
    d = engine_dir()
    src = os.path.join(ADDON_DIR, "engine", "wheels")
    if not d or not os.path.isdir(src):
        return
    import zipfile
    import hashlib
    whls = sorted(f for f in os.listdir(src) if f.endswith(".whl"))
    stamp = ";".join("%s:%d" % (f, os.path.getsize(os.path.join(src, f)))
                     for f in whls)
    flag = os.path.join(d, "engine", ".binaries")
    try:
        if open(flag, encoding="utf-8").read() == stamp:
            return
    except Exception:
        pass
    ok = True
    n = 0
    for f in whls:
        try:
            z = zipfile.ZipFile(os.path.join(src, f))
        except Exception:
            continue
        for name in z.namelist():
            if not name.endswith(".pyd"):
                continue
            data = None
            for env in ("env", "env_hy"):
                t = os.path.join(d, env, "Lib", "site-packages",
                                 *name.split("/"))
                if not os.path.isfile(t):
                    continue
                if data is None:
                    data = z.read(name)
                    new_h = hashlib.sha256(data).digest()
                try:
                    with open(t, "rb") as fh:
                        if hashlib.sha256(fh.read()).digest() == new_h:
                            continue
                    tmp = t + ".new"
                    with open(tmp, "wb") as fh:
                        fh.write(data)
                    os.replace(tmp, t)
                    n += 1
                except Exception as exc:
                    # in use by a running engine: try again next start
                    ok = False
                    print("[Image to 3D] engine part %s: %s" % (
                        os.path.basename(t), exc))
                    try:
                        os.remove(t + ".new")
                    except Exception:
                        pass
    if n:
        print("[Image to 3D] updated %d engine part(s)" % n)
    if ok:
        try:
            os.makedirs(os.path.dirname(flag), exist_ok=True)
            with open(flag, "w", encoding="utf-8") as fh:
                fh.write(stamp)
        except Exception:
            pass


def child_env():
    """The engines speak UTF-8 whatever the Windows language is, so a file
    or folder name in Japanese, Chinese, Persian... reaches Blender intact."""
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    e["PYTHONIOENCODING"] = "utf-8"
    return e


def safe_name(text, fallback="model"):
    """A file name with only plain letters, digits, - and _ ."""
    import re
    t = re.sub(r"[^A-Za-z0-9_-]+", "_", text or "").strip("_")
    return t[:60] or fallback


def log_path():
    """The engine's own words, kept on disk for the support report."""
    d = engine_dir() or ADDON_DIR
    return os.path.join(d, "logs", "engine.log")


def _log_open(path):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path) > 4 * 2**20:
            old = path[:-4] + ".old.log"
            try:
                os.replace(path, old)
            except Exception:
                pass
        return open(path, "a", encoding="utf-8", errors="replace")
    except Exception:
        return None


def addon_log(msg):
    f = _log_open(log_path())
    if f is not None:
        try:
            f.write("%s [addon] %s\n" % (time.strftime("%H:%M:%S"), msg))
        finally:
            f.close()


def _reader(proc, q, log=None):
    f = _log_open(log) if log else None
    if f is not None:
        f.write("\n===== %s  engine started =====\n"
                % time.strftime("%Y-%m-%d %H:%M:%S"))
        f.flush()
    for raw in iter(proc.stdout.readline, b""):
        try:
            line = raw.decode("utf-8", "replace").rstrip()
            q.put(line)
            if f is not None and not line.startswith("PROGRESS "):
                f.write(line + "\n")
                f.flush()
        except Exception:
            pass
    if f is not None:
        try:
            f.write("===== engine closed (code %s) =====\n" % proc.poll())
            f.close()
        except Exception:
            pass
    q.put(None)


def _start(kind, args, cwd):
    q = queue.Queue()
    p = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=NOWINDOW, env=child_env())
    threading.Thread(target=_reader, args=(p, q, log_path()),
                     daemon=True).start()
    _PROC.update(p=p, q=q, kind=kind, lines=[])
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=0.25)


def _job():
    return CARD_JOB if _PROC["kind"] == "gen" else SETUP_JOB


def _finish(ok, text):
    global CARD_JOB, SETUP_JOB
    kind = _PROC["kind"]
    _PROC.update(p=None, q=None, kind=None)
    st = getattr(bpy.context.scene, "img3d", None)
    if kind == "gen":
        if ok:
            _import_result(text)
        elif st is not None:
            st.status = text
        CARD_JOB = None
    else:
        SETUP_JOB = None
        if st is not None:
            st.status = "" if ok else text
    _redraw()


def _poll():
    p, q = _PROC["p"], _PROC["q"]
    if p is None or q is None:
        return None
    job = _job()
    ended = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            ended = True
            break
        _PROC["lines"].append(line)
        del _PROC["lines"][:-60]
        if line.startswith("PROGRESS "):
            parts = line.split(" ", 2)
            try:
                job["frac"] = float(parts[1])
            except Exception:
                pass
            job["stage"] = parts[2] if len(parts) > 2 else job.get("stage")
            job["line"] = ""
        elif line.startswith("DONE "):
            job["result"] = line[5:].strip()
        elif line.startswith("ERROR "):
            job["error"] = line[6:].strip()
        elif line.startswith("[setup]"):
            job["line"] = line[8:][:40]
        elif line.strip() and not line.startswith(("TIME ", "  ")):
            job["line"] = line.strip()[:40]
    if job is not None and job.get("t0"):
        el = int(time.time() - job["t0"])
        job["clock"] = "%d:%02d" % (el // 60, el % 60)
    _redraw()
    if ended or p.poll() is not None:
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            if line is None:
                break
            _PROC["lines"].append(line)
            if line.startswith("DONE "):
                job["result"] = line[5:].strip()
            elif line.startswith("ERROR "):
                job["error"] = line[6:].strip()
        if job.get("cancelled"):
            _finish(False, "Cancelled")
        elif job.get("result") and os.path.exists(job["result"]):
            _finish(True, job["result"])
        else:
            msg = job.get("error")
            if not msg:
                tail = [l for l in _PROC["lines"] if l.strip()]
                msg = tail[-1][:120] if tail else "The engine stopped"
            print("[Image to 3D] engine output:\n" + "\n".join(_PROC["lines"]))
            _finish(False, msg)
        return None
    return 0.25


def _fit_to(meshes, new, tgt, pre=None):
    """Put the new model exactly where the old one was: same place, turn
    and size (the engines give back a model in their own unit box). pre
    undoes a turn the model was sent to the engine with."""
    from mathutils import Matrix, Vector

    def box(objs):
        pts = []
        for o in objs:
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
        lo = Vector([min(p[i] for p in pts) for i in range(3)])
        hi = Vector([max(p[i] for p in pts) for i in range(3)])
        return (lo + hi) * 0.5, max(hi - lo)
    for o in new:
        if o.parent is None:
            o.matrix_world = Matrix(pre) if pre is not None else \
                Matrix.Identity(4)
    bpy.context.view_layer.update()
    cn, sn = box(meshes)
    # the old model's box in its own space, then carried by its transform
    lo = Vector([min(c[i] for c in tgt.bound_box) for i in range(3)])
    hi = Vector([max(c[i] for c in tgt.bound_box) for i in range(3)])
    ct, stt = (lo + hi) * 0.5, max(hi - lo)
    k = stt / sn if sn > 0 else 1.0
    fit = tgt.matrix_world @ Matrix.Translation(ct) @ \
        Matrix.Scale(k, 4) @ Matrix.Translation(-cn)
    for o in new:
        if o.parent is None:
            o.matrix_world = fit @ o.matrix_world


def _tunnels(me):
    """How many holes run through the shape (its genus), from
    vertices - edges + faces and the number of separate pieces."""
    import numpy as np
    n = len(me.vertices)
    e = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    # label propagation with pointer jumping until every piece has one label
    lab = np.arange(n)
    while True:
        m = np.minimum(lab[e[:, 0]], lab[e[:, 1]])
        new = lab.copy()
        np.minimum.at(new, e[:, 0], m)
        np.minimum.at(new, e[:, 1], m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    pieces = len(np.unique(lab))
    chi = n - len(me.edges) + len(me.polygons)
    return max(0, (2 * pieces - chi) // 2)


def _smooth_shape(obj, res=(220, 190, 160, 130)):
    """What Hi3D is made for: a smooth, closed rough shape without holes
    through it, like the one Hunyuan hands over. TRELLIS and Pixal3D models
    carry fine scales, cracks and thin see-through spots; Hi3D then builds
    a porous surface (about 4,500 holes on a test creature, about 100 with
    this). A voxel remesh of a temporary copy keeps the whole form (horns,
    spikes, tail) and drops only the fine detail; if holes still run
    through it, a coarser remesh closes them. The model in the scene is
    not touched."""
    dims = [d / (s or 1.0) for d, s in zip(obj.dimensions, obj.scale)]
    best = None
    for r in res:
        tmp = obj.copy()
        tmp.data = obj.data.copy()
        tmp.modifiers.clear()
        bpy.context.scene.collection.objects.link(tmp)
        try:
            md = tmp.modifiers.new("smooth", "REMESH")
            md.mode = "VOXEL"
            md.voxel_size = max(max(dims) / float(r), 1e-6)
            md.adaptivity = 0.0
            dg = bpy.context.evaluated_depsgraph_get()
            dg.update()
            me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
        except Exception as exc:
            addon_log("smooth shape failed: %s" % exc)
            me = None
        finally:
            data = tmp.data
            bpy.data.objects.remove(tmp)
            bpy.data.meshes.remove(data)
        if me is None:
            continue
        holes = _tunnels(me)
        addon_log("smooth shape %d: %d faces, %d holes through it" % (
            r, len(me.polygons), holes))
        if best is not None:
            bpy.data.meshes.remove(best)
        best = me
        if holes <= 20:
            break
    return best if best is not None else obj.data


def _export_for_detail(obj, path):
    """The model's own shape as a .ply in the engines' axes (Y up), turned
    the way it stands in the scene: a Pixal3D model is stood up by its
    object turn, and the engine must see it standing like the picture."""
    import numpy as np
    me = _smooth_shape(obj)
    me.calc_loop_triangles()
    n = len(me.vertices)
    v = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", v)
    v = v.reshape(-1, 3)
    rot = np.array(obj.matrix_world.to_quaternion().to_matrix(), np.float32)
    v = v @ rot.T
    v = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1).astype("<f4")
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    t = t.reshape(-1, 3)
    # weld: a generated model is split at every UV seam, so the shape sent
    # to Hi3D was full of open cracks; Hi3D then can't tell inside from
    # outside there and builds flipped, see-through patches
    span = float(np.abs(v).max()) or 1.0
    key = np.round(v / (span * 1e-6)).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True,
                               return_inverse=True)
    inv = inv.ravel()
    v = v[first]
    t = inv[t].astype(np.int32)
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    n = len(v)
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
            "property float x\nproperty float y\nproperty float z\n"
            "element face %d\nproperty list uchar int vertex_indices\n"
            "end_header\n" % (n, len(t))).encode("ascii")
    faces = np.zeros(len(t), dtype=[("c", "u1"), ("i", "<i4", (3,))])
    faces["c"] = 3
    faces["i"] = t
    with open(path, "wb") as f:
        f.write(head)
        f.write(v.tobytes())
        f.write(faces.tobytes())
    if me is not obj.data:
        bpy.data.meshes.remove(me)
    return path


def _weld(obj):
    """The engines hand back a model split at every UV seam: the same corner
    stored two or three times, so the surface is full of invisible cracks
    (sculpting, smoothing and modifiers tear it open there). Join the
    corners that sit on the same spot; the UVs stay as they are, because
    Blender keeps them per face corner."""
    import bmesh
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    span = max(obj.dimensions) or 1.0
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=span * 1e-6)
    after = len(bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.update()
    addon_log("welded %s: %d -> %d vertices" % (obj.name, before, after))


def _import_result(path):
    st = bpy.context.scene.img3d
    before = set(bpy.data.objects)
    try:
        if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    win, area, region = card.hot_view()
    try:
        if area is not None:
            with bpy.context.temp_override(window=win, area=area,
                                           region=region):
                bpy.ops.import_scene.gltf(filepath=path)
        else:
            bpy.ops.import_scene.gltf(filepath=path)
    except Exception as exc:
        st.status = "Import failed: %s" % exc
        addon_log("import failed: %s" % exc)
        return
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == "MESH"]
    try:
        addon_log("imported %s: %d objects, %d meshes, %d faces, %d bytes" % (
            path, len(new), len(meshes),
            sum(len(o.data.polygons) for o in meshes), os.path.getsize(path)))
    except Exception:
        pass
    for o in meshes:
        try:
            _weld(o)
        except Exception as exc:
            addon_log("weld failed: %s" % exc)
    name = os.path.splitext(os.path.basename(st.image_path))[0] or "Generated"
    cursor = bpy.context.scene.cursor.location.copy()
    turn = (CARD_JOB or {}).get("engine") == "PIXAL3D"
    for o in new:
        if o.parent is None:
            o.location = cursor
            if turn:
                # Pixal3D builds in the picture's camera frame, lying on its
                # back: stand it up, facing the front like the others
                import math
                o.rotation_mode = "XYZ"
                o.rotation_euler.x += math.radians(90.0)
    job_ = CARD_JOB or {}
    tgt = bpy.data.objects.get(job_.get("replace") or "")
    if tgt is not None:
        name = tgt.name.split(" Detail")[0] + " Detail"
        _fit_to(meshes, new, tgt, job_.get("pre"))
    for k, o in enumerate(meshes):
        o.name = name if k == 0 else "%s.%d" % (name, k)
        o["img3d_image"] = job_.get("image") or st.image_path
        if job_.get("fine"):
            o["img3d_fine"] = job_["fine"]
        if job_.get("gray"):
            o["img3d_gray"] = 1
    if tgt is not None:
        # the old model stays in the file, hidden, in case it is wanted back
        tgt.hide_set(True)
        tgt.hide_render = True
        for o in meshes:
            o["img3d_prev"] = tgt.name
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in meshes:
        o.select_set(True)
    if meshes:
        bpy.context.view_layer.objects.active = meshes[0]
    work = None
    for l_ in reversed(_SRV.get("lines", [])):
        if l_.startswith("WORK "):
            work = l_[5:].strip()
            break
    if work and os.path.isdir(os.path.join(work, "views")):
        for o in meshes:
            o["img3d_work"] = work
            o["img3d_tex"] = int(st.texture)
    faces = sum(len(o.data.polygons) for o in meshes)
    job = CARD_JOB or {}
    secs = int(time.time() - job.get("t0", time.time()))
    st.status = "Done in %d:%02d  -  %s faces" % (secs // 60, secs % 60,
                                                 f"{faces:,}")
    card.say("Model added to the scene")


# -- the engine server --------------------------------------------------------
# One engine process stays up while Blender is open, so the networks it has
# loaded stay in memory and the next model skips all the loading.

_SRV = {"p": None, "q": None, "lines": [], "engine": None}


def _server_alive(engine=None):
    p = _SRV["p"]
    alive = p is not None and p.poll() is None
    if alive and engine is not None and _SRV["engine"] != engine:
        return False
    return alive


# -- memory: what is free, and the test mode that pretends to be a small PC --

_MEMTEST = {"job": None, "limit": 0}


def _win_mem():
    import ctypes
    from ctypes import wintypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD),
                    ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30


class _JELI(object):
    pass


def _job_struct():
    import ctypes
    from ctypes import wintypes

    class IOC(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BLI(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class ELI(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BLI),
                    ("IoInfo", IOC),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]
    return ELI


def memtest_on(ram_gb=9.5, vram_gb=7.0, stat_dir=None):
    """Engines started from now on may use at most ram_gb of main memory
    together and vram_gb of the graphics card each (a 16 GB / 8 GB PC once
    Windows, Blender and the screen have taken theirs)."""
    import ctypes
    k = ctypes.windll.kernel32
    k.CreateJobObjectW.restype = ctypes.c_void_p
    job = k.CreateJobObjectW(None, None)
    ELI = _job_struct()
    info = ELI()
    info.BasicLimitInformation.LimitFlags = 0x200   # JOB_MEMORY limit
    info.JobMemoryLimit = int(ram_gb * 2**30)
    k.SetInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_void_p, ctypes.c_uint]
    # no hard limit: Windows would count memory CUDA only reserves; the
    # real use (working set) is recorded per process instead
    _MEMTEST.update(job=job, limit=ram_gb)
    os.environ["IMG3D_VRAM_GB"] = str(vram_gb)
    os.environ["IMG3D_MEMSTAT_DIR"] = stat_dir or os.path.join(
        engine_dir(), "_diag", "memstat")
    server_stop()


def memtest_off():
    _MEMTEST.update(job=None, limit=0)
    os.environ.pop("IMG3D_VRAM_GB", None)
    os.environ.pop("IMG3D_MEMSTAT_DIR", None)
    server_stop()


def memtest_peak():
    """Most main memory the engines held together since the test began."""
    import ctypes
    job = _MEMTEST.get("job")
    if not job:
        return 0.0
    ELI = _job_struct()
    info = ELI()
    k = ctypes.windll.kernel32
    k.QueryInformationJobObject.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_uint,
                                            ctypes.c_void_p]
    k.QueryInformationJobObject(ctypes.c_void_p(job), 9, ctypes.byref(info),
                                ctypes.sizeof(info), None)
    return info.PeakJobMemoryUsed / 2**30


def _job_add(p):
    job = _MEMTEST.get("job")
    if not job:
        return
    import ctypes
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    h = k.OpenProcess(0x1F0FFF, False, p.pid)
    k.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k.AssignProcessToJobObject(ctypes.c_void_p(job), ctypes.c_void_p(h))
    k.CloseHandle(ctypes.c_void_p(h))


def _ram_room_gb():
    """Main memory still free for the engines (the test limit counts too)."""
    try:
        free = _win_mem()[1]
    except Exception:
        free = 0.0
    if _MEMTEST.get("limit"):
        try:
            used = sum(_proc_private_gb(x["p"].pid) for x in _POOL.values())
            if _SRV["p"] is not None:
                used += _proc_private_gb(_SRV["p"].pid)
            free = min(free, _MEMTEST["limit"] - used)
        except Exception:
            pass
    return free


def _proc_private_gb(pid):
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("a", ctypes.c_size_t), ("b", ctypes.c_size_t),
                    ("c", ctypes.c_size_t), ("d", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    k = ctypes.windll.kernel32
    k.OpenProcess.restype = ctypes.c_void_p
    h = k.OpenProcess(0x1000, False, pid)
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC),
                                          wintypes.DWORD]
    k.K32GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(c), c.cb)
    k.CloseHandle(ctypes.c_void_p(h))
    return c.WorkingSetSize / 2**30


# Hunyuan and Hi3D take turns on the graphics card. The one not working
# parks its networks in main memory and stays up, so switching back costs
# seconds instead of loading everything from disk again.
_POOL = {}
_PARKS = ("HUNYUAN", "HI3D")


def _park_current(timeout=90.0):
    p = _SRV["p"]
    eng = _SRV.get("engine")
    if p is None or p.poll() is not None or eng not in _PARKS:
        return False
    try:
        p.stdin.write(b'{"task": "park"}\n')
        p.stdin.flush()
    except Exception:
        return False
    q = _SRV["q"]
    t_end = time.time() + timeout
    while time.time() < t_end:
        try:
            line = q.get(timeout=0.5)
        except queue.Empty:
            if p.poll() is not None:
                return False
            continue
        if line is None:
            return False
        if line.startswith("PARKED"):
            _POOL[eng] = dict(p=p, q=q, lines=[])
            return True
    return False


def _server_start(engine="TRELLIS"):
    cur = _SRV.get("engine")
    alive = _SRV["p"] is not None and _SRV["p"].poll() is None
    # keep the idle engine in main memory only when there is room for it
    # next to the one that is starting (about 8 GB), else close it
    room = _ram_room_gb()
    held_ = _POOL.get(engine)
    need = 8.0 if held_ is None else 0.5
    if _small_pc():
        # a 16 GB PC: the next engine needs all of it, nothing waits
        room = 0.0
    if alive and cur != engine and engine in _PARKS and cur in _PARKS and \
            room > need and \
            _park_current():
        _SRV.update(p=None, q=None)
    else:
        # one engine at a time: two would not fit on the GPU together
        server_stop(keep_pool=True)
        if room < need:
            # short of memory: nothing else waits in the background
            for k_ in list(_POOL):
                if k_ != engine:
                    _kill(_POOL.pop(k_)["p"])
    held = _POOL.pop(engine, None)
    if held is not None and held["p"].poll() is None:
        _SRV.update(p=held["p"], q=held["q"], lines=[], engine=engine)
        return
    q = queue.Queue()
    try:
        sync_engine_code()
    except Exception as exc:
        print("[Image to 3D] engine update:", exc)
    try:
        update_engine_binaries()
    except Exception as exc:
        print("[Image to 3D] engine parts update:", exc)
    try:
        retire_nvdiffrast()
    except Exception as exc:
        print("[Image to 3D] rasterizer update:", exc)
    script = engine_script(engine)
    p = subprocess.Popen(
        [engine_python(engine), "-u", script, "--serve"],
        cwd=os.path.dirname(script), stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=NOWINDOW, env=child_env())
    _job_add(p)
    threading.Thread(target=_reader, args=(p, q, log_path()),
                     daemon=True).start()
    _SRV.update(p=p, q=q, lines=[], engine=engine)


def _kill(p):
    if p is not None and p.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                           capture_output=True, creationflags=NOWINDOW)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


def server_stop(keep_pool=False):
    p = _SRV["p"]
    _SRV.update(p=None, q=None)
    _kill(p)
    if not keep_pool:
        for k in list(_POOL):
            _kill(_POOL.pop(k)["p"])


def _send(job):
    import json
    p = _SRV["p"]
    p.stdin.write((json.dumps(job) + "\n").encode("utf-8"))
    p.stdin.flush()


def bake_ao(obj, size=2048, samples=16, strength=0.7):
    """Ambient occlusion baked onto the model (Cycles, on the graphics card)
    and multiplied into its colour: creases, seams and the undersides of
    plates get the soft darkening that makes a surface read as real."""
    import time as _t
    t0 = _t.time()
    sc = bpy.context.scene
    mat = obj.active_material
    if mat is None or not mat.use_nodes:
        return
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    base_in = bsdf.inputs["Base Color"]
    if not base_in.is_linked:
        return
    src = base_in.links[0].from_socket
    img = bpy.data.images.new(obj.name + "_AO", size, size, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    node = nt.nodes.new("ShaderNodeTexImage")
    node.image = img
    node.location = (bsdf.location.x - 600, bsdf.location.y - 400)
    for n in nt.nodes:
        n.select = False
    node.select = True
    nt.nodes.active = node
    keep = (sc.render.engine, sc.cycles.samples, sc.cycles.device)
    try:
        sc.render.engine = "CYCLES"
        sc.cycles.samples = samples
        try:
            sc.cycles.device = "GPU"
        except Exception:
            pass
        for o in bpy.context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        dim = max(obj.dimensions) or 1.0
        sc.world.light_settings.distance = dim * 0.15 if sc.world else 1.0
        bpy.ops.object.bake(type="AO", margin=8, use_clear=True)
    finally:
        sc.render.engine, sc.cycles.samples, sc.cycles.device = keep
    img.pack()
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = strength
    mix.location = (bsdf.location.x - 250, bsdf.location.y + 100)
    nt.links.new(src, mix.inputs["A"])
    nt.links.new(node.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], base_in)
    # the colour map is the one the viewport shows in Solid > Texture, not
    # the grey occlusion the bake had to make active
    if getattr(src, "node", None) is not None and src.node.type == "TEX_IMAGE":
        nt.nodes.active = src.node
    print("[Image to 3D] AO baked in %.1f s" % (_t.time() - t0))


def _swap_texture(path, target):
    """Put the enhanced texture on the model already in the scene (same
    mesh and UVs), instead of adding a second model."""
    st = bpy.context.scene.img3d
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    mat = None
    for o in new:
        if o.type == "MESH" and o.active_material is not None:
            mat = o.active_material
    tgt = bpy.data.objects.get(target)
    if mat is not None and tgt is not None:
        base_ = (tgt.active_material.name if tgt.active_material
                 else tgt.name)
        while base_.endswith(" Enhanced"):
            base_ = base_[:-9]
        mat.name = base_ + " Enhanced"
        for i_ in range(max(1, len(tgt.material_slots))):
            if tgt.material_slots:
                tgt.material_slots[i_].material = mat
            else:
                tgt.data.materials.append(mat)
    for o in new:
        me = o.data if o.type == "MESH" else None
        bpy.data.objects.remove(o)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
    if tgt is not None:
        for o in bpy.context.selected_objects:
            o.select_set(False)
        tgt.select_set(True)
        bpy.context.view_layer.objects.active = tgt
    if tgt is not None:
        try:
            bake_ao(tgt)
        except Exception as exc:
            print("[Image to 3D] no AO: %s" % exc)
    job = CARD_JOB or {}
    secs = int(time.time() - job.get("t0", time.time()))
    st.status = "Texture enhanced in %d:%02d" % (secs // 60, secs % 60)


def _swap_images(stem, target):
    """Enhance result as plain pictures: a fresh material on the same model
    (same mesh and UVs), so nothing has to be imported. The occlusion of
    the model is baked once and used again on later enhances."""
    import re as _re
    st = bpy.context.scene.img3d
    tgt = bpy.data.objects.get(target)
    if tgt is None:
        return
    def load(p, data):
        if not os.path.isfile(p):
            return None
        im = bpy.data.images.load(p, check_existing=False)
        im.colorspace_settings.name = "Non-Color" if data else "sRGB"
        im.pack()
        return im
    base_ = (tgt.active_material.name if tgt.active_material else tgt.name)
    base_ = _re.sub(r"( Enhanced(\.\d+)?)+$", "", base_)
    old = tgt.active_material
    mat = bpy.data.materials.new(base_ + " Enhanced")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    x, y = bsdf.location.x, bsdf.location.y
    col = nt.nodes.new("ShaderNodeTexImage")
    col.image = load(stem + ".jpg", False)
    col.location = (x - 700, y + 200)
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    # roughness and metal: the model's own maps are kept when it has them
    # (re-baking them only smooths them and makes the surface shinier)
    _c, met_, rou_, gltf_ = _mat_images(tgt) if old is not None else \
        (None, None, None, None)
    if gltf_ is not None:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = gltf_
        n.location = (x - 700, y - 100)
        sp_ = nt.nodes.new("ShaderNodeSeparateColor")
        sp_.location = (x - 400, y - 100)
        nt.links.new(n.outputs["Color"], sp_.inputs["Color"])
        nt.links.new(sp_.outputs["Green"], bsdf.inputs["Roughness"])
        nt.links.new(sp_.outputs["Blue"], bsdf.inputs["Metallic"])
    for suf, sock, dy, keep_ in (("_roughness", "Roughness", -100, rou_),
                                 ("_metallic", "Metallic", 0, met_)):
        if gltf_ is not None:
            break
        im = keep_ if keep_ is not None else load(stem + suf + ".jpg", True)
        if im is not None:
            n = nt.nodes.new("ShaderNodeTexImage")
            n.image = im
            n.location = (x - 400, y + dy)
            nt.links.new(n.outputs["Color"], bsdf.inputs[sock])
    nim = load(stem + "_normal.png", True)
    if nim is not None:
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = nim
        n.location = (x - 700, y - 400)
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (x - 250, y - 400)
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    for i_ in range(max(1, len(tgt.material_slots))):
        if tgt.material_slots:
            tgt.material_slots[i_].material = mat
        else:
            tgt.data.materials.append(mat)
    # the occlusion only depends on the shape: use the last one again
    ao = None
    if old is not None and old.use_nodes:
        for n in old.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image is not None and \
                    n.image.name.startswith(tgt.name + "_AO"):
                ao = n.image
    if old is not None:
        # kept (not deleted) so the enhance can be undone
        old.use_fake_user = True
        prev = tgt.get("img3d_prev_mat")
        if prev and prev != old.name:
            # only one step back is kept; the one before goes
            pm = bpy.data.materials.get(prev)
            if pm is not None and pm.users <= 1 and pm is not old:
                pm.use_fake_user = False
                if pm.users == 0:
                    bpy.data.materials.remove(pm)
        tgt["img3d_prev_mat"] = old.name
    for o in bpy.context.selected_objects:
        o.select_set(False)
    tgt.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    try:
        if ao is not None:
            node = nt.nodes.new("ShaderNodeTexImage")
            node.image = ao
            node.location = (x - 600, y - 700)
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 0.7
            mix.location = (x - 250, y + 100)
            nt.links.new(col.outputs["Color"], mix.inputs["A"])
            nt.links.new(node.outputs["Color"], mix.inputs["B"])
            nt.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
        else:
            bake_ao(tgt)
    except Exception as exc:
        print("[Image to 3D] no AO: %s" % exc)
    try:
        nt.nodes.active = col
    except Exception:
        pass
    # pictures of earlier enhances nobody uses any more
    for im in list(bpy.data.images):
        if im.users == 0 and im.name.startswith(
                ("textured", "Image_", tgt.name + "_AO")):
            bpy.data.images.remove(im)
    job = CARD_JOB or {}
    secs = int(time.time() - job.get("t0", time.time()))
    st.status = "Texture enhanced in %d:%02d" % (secs // 60, secs % 60)


_SMALL = {}


def _small_pc():
    """16 GB of main memory or an 8 GB card (or the test mode)"""
    if _MEMTEST.get("limit"):
        return True
    if "v" not in _SMALL:
        small = False
        try:
            small = _win_mem()[0] < 28.0
        except Exception:
            pass
        if not small:
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, creationflags=NOWINDOW).stdout
                small = float(o.split()[0]) < 10.5 * 1024
            except Exception:
                pass
        _SMALL["v"] = small
    return _SMALL["v"]


def _gen_done(ok, text):
    global CARD_JOB
    try:
        addon_log(("DONE " if ok else "FAILED ") + str(text))
    except Exception:
        pass
    st = getattr(bpy.context.scene, "img3d", None)
    if _small_pc():
        # a small PC: the engine leaves memory after every job, so the next
        # one (or Blender) starts with all of it
        server_stop()
    if ok and (CARD_JOB or {}).get("enhance"):
        sw_ = CARD_JOB.get("set_work")
        tg_ = bpy.data.objects.get(CARD_JOB["enhance"])
        if sw_ and tg_ is not None and \
                os.path.isdir(os.path.join(sw_, "views")):
            tg_["img3d_work"] = sw_
        try:
            if text.lower().endswith(".jpg"):
                _swap_images(os.path.splitext(text)[0], CARD_JOB["enhance"])
            else:
                _swap_texture(text, CARD_JOB["enhance"])
        except Exception as exc:
            st.status = "Could not apply the texture: %s" % exc
        CARD_JOB = None
        _redraw()
        return
    if ok:
        _import_result(text)
    elif st is not None:
        st.status = text
    job_ = CARD_JOB or {}
    CARD_JOB = None
    if ok and job_.get("then_detail"):
        try:
            bpy.ops.img3d.enhance_detail(
                no_tex=bool(job_.get("detail_notex")))
            if CARD_JOB is not None:
                CARD_JOB["t0"] = job_.get("t0", CARD_JOB["t0"])
                if job_.get("then_enhance"):
                    CARD_JOB["then_enhance"] = True
                _redraw()
                return
        except Exception as exc:
            if st is not None:
                st.status = "Could not add the detail: %s" % exc
    if ok and job_.get("then_enhance"):
        try:
            ctx = bpy.context
            if enhance_target(ctx) is not None:
                bpy.ops.img3d.enhance()
                if CARD_JOB is not None:
                    # one clock for the whole run
                    CARD_JOB["t0"] = job_.get("t0", CARD_JOB["t0"])
        except Exception as exc:
            if st is not None:
                st.status = "Could not enhance the texture: %s" % exc
    _redraw()


def _poll_gen():
    job = CARD_JOB
    q = _SRV["q"]
    if job is None:
        return None
    if q is None:
        _gen_done(False, "The engine stopped")
        return None
    ended = False
    while True:
        try:
            line = q.get_nowait()
        except queue.Empty:
            break
        if line is None:
            ended = True
            break
        _SRV["lines"].append(line)
        del _SRV["lines"][:-60]
        if line.startswith("PROGRESS "):
            parts = line.split(" ", 2)
            try:
                lo_, hi_ = job.get("span", (0.0, 1.0))
                job["frac"] = max(job.get("frac", 0.0),
                                  lo_ + (hi_ - lo_) * float(parts[1]))
            except Exception:
                pass
            job["stage"] = parts[2] if len(parts) > 2 else job.get("stage")
            job["line"] = ""
        elif line.startswith("DONE "):
            path = line[5:].strip()
            if os.path.exists(path) and job.get("chain"):
                # shape is ready: hand it to the next engine
                ch = job["chain"]
                if isinstance(ch, list):
                    nxt = dict(ch.pop(0))
                    if not ch:
                        job.pop("chain")
                else:
                    nxt = job.pop("chain")
                srv_ = nxt.pop("server", "TRELLIS")
                job["span"] = tuple(nxt.pop("span", (0.45, 1.0)))
                job["stage"] = nxt.pop("stage", "Painting texture")
                job["frac"] = job["span"][0]
                nxt["mesh"] = path
                try:
                    _server_start(srv_)
                    _send(nxt)
                except Exception as exc:
                    _gen_done(False, "Could not start Pixal3D: %s" % exc)
                    return None
                _redraw()
                return 0.25
            if os.path.exists(path):
                _gen_done(True, path)
            else:
                _gen_done(False, "The model file was not written")
            # a follow-up job (Enhance Texture) may have started
            return 0.25 if CARD_JOB is not None else None
        elif line.startswith("ERROR "):
            _gen_done(False, line[6:].strip())
            return None
        elif line.startswith(("READY", "TIME ", "PROGRESS")):
            pass
        elif line.strip() and not line.startswith(("  ", "\r")):
            job["line"] = line.strip()[:40]
    el = int(time.time() - job["t0"])
    job["clock"] = "%d:%02d" % (el // 60, el % 60)
    _redraw()
    p = _SRV["p"]
    if ended or p is None or p.poll() is not None:
        _SRV.update(p=None, q=None)
        if job.get("cancelled"):
            _gen_done(False, "Cancelled")
        else:
            tail = [l for l in _SRV["lines"] if l.strip()]
            print("[Image to 3D] engine output:\n" + "\n".join(_SRV["lines"]))
            _gen_done(False, tail[-1][:120] if tail else "The engine stopped")
        return None
    return 0.25


# -- operators ------------------------------------------------------------

class IMG3D_OT_pick_image(bpy.types.Operator):
    """Choose the picture to turn into a 3D model"""
    bl_idname = "img3d.pick_image"
    bl_label = "Choose Image"

    filepath: StringProperty(subtype="FILE_PATH")
    filter_image: BoolProperty(default=True, options={"HIDDEN"})
    filter_folder: BoolProperty(default=True, options={"HIDDEN"})

    def execute(self, context):
        if self.filepath and os.path.isfile(self.filepath):
            context.scene.img3d.image_path = self.filepath
            context.scene.img3d.status = ""
        _redraw()
        return {"FINISHED"}

    def invoke(self, context, event):
        cur = context.scene.img3d.image_path
        if cur:
            self.filepath = bpy.path.abspath(cur)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}


class IMG3D_OT_clear_image(bpy.types.Operator):
    """Take the picture away"""
    bl_idname = "img3d.clear_image"
    bl_label = "Remove Image"

    def execute(self, context):
        context.scene.img3d.image_path = ""
        context.scene.img3d.status = ""
        _redraw()
        return {"FINISHED"}


def _view_ops():
    """Pick / clear for the extra views (the card calls operators by name
    only, so each slot has its own pair)."""
    made = []

    def make(n):
        key = "view_%d" % n

        def pexec(self, context):
            if self.filepath and os.path.isfile(self.filepath):
                setattr(context.scene.img3d, key, self.filepath)
            _redraw()
            return {"FINISHED"}

        def pinvoke(self, context, event):
            cur = getattr(context.scene.img3d, key) or \
                context.scene.img3d.image_path
            if cur:
                self.filepath = bpy.path.abspath(cur)
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}

        def cexec(self, context):
            setattr(context.scene.img3d, key, "")
            _redraw()
            return {"FINISHED"}
        pick = type("IMG3D_OT_pick_view_%d" % n, (bpy.types.Operator,), {
            "__doc__": "Add a picture of the same object from the %s" % (
                ("back", "left", "right")[n - 1]),
            "bl_idname": "img3d.pick_view_%d" % n,
            "bl_label": "Add View",
            "__annotations__": {
                "filepath": StringProperty(subtype="FILE_PATH"),
                "filter_image": BoolProperty(default=True,
                                             options={"HIDDEN"}),
                "filter_folder": BoolProperty(default=True,
                                              options={"HIDDEN"})},
            "execute": pexec, "invoke": pinvoke})
        clear = type("IMG3D_OT_clear_view_%d" % n, (bpy.types.Operator,), {
            "__doc__": "Remove this view",
            "bl_idname": "img3d.clear_view_%d" % n,
            "bl_label": "Remove View",
            "execute": cexec})
        return pick, clear
    for n in (1, 2, 3):
        made.extend(make(n))
    return tuple(made)


VIEW_OPS = _view_ops()


class IMG3D_OT_generate(bpy.types.Operator):
    """Build a textured 3D model from the chosen image"""
    bl_idname = "img3d.generate"
    bl_label = "Generate 3D"

    @classmethod
    def poll(cls, context):
        st = getattr(context.scene, "img3d", None)
        return (st is not None and st.image_path
                and os.path.isfile(bpy.path.abspath(st.image_path))
                and CARD_JOB is None and _PROC["p"] is None
                and engine_ready(st.engine))

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        _free_viewport_gpu()
        global CARD_JOB
        st = context.scene.img3d
        img = bpy.path.abspath(st.image_path)
        outdir = os.path.join(engine_dir(), "output")
        os.makedirs(outdir, exist_ok=True)
        base = safe_name(os.path.splitext(os.path.basename(img))[0])
        out = os.path.join(outdir, "%s_%s.glb" % (base, time.strftime(
            "%Y%m%d_%H%M%S")))
        hi3d = st.engine == "HI3D" or (
            st.engine == "HUNYUAN" and st.auto_detail == "ON" and
            engine_ready("HI3D"))
        hy = st.engine in ("HUNYUAN", "HI3D")
        job = {"image": img, "out": out, "quality": st.quality,
               "faces": int(st.faces_hy if hy else st.faces),
               "texture": int(st.texture) if st.use_texture == "ON" else 0,
               "seed": int(st.seed)}
        if st.engine == "PIXAL3D":
            job["engine"] = "PIXAL3D"
        if hy:
            vs = {}
            for side, v in (("back", st.view_1), ("left", st.view_2),
                            ("right", st.view_3)):
                v = bpy.path.abspath(v) if v else ""
                if v and os.path.isfile(v):
                    vs[side] = v
            if vs and not hy_mv_ready():
                st.status = ("Hunyuan multi-view files are not installed - "
                             "remove the extra views")
                _redraw()
                return {"FINISHED"}
            if vs:
                job["views"] = vs
        chain = None
        job_tex_ = job["texture"]
        if hi3d:
            # Hunyuan: first shape only -> Hi3D: real detail -> Hunyuan:
            # paints the refined shape
            shape_out = out[:-4] + "_coarse.glb"
            fine_out = out[:-4] + "_hi3d.glb"
            chain = [{"server": "HI3D", "span": (0.25, 0.75),
                      "stage": "Adding detail (Hi3D)", "image": img,
                      "out": fine_out,
                      "resolution": 512 if st.quality == "FAST" else 1024,
                      "seed": job["seed"]}]
            if job["texture"] > 0:
                chain.append({"server": "HUNYUAN", "span": (0.75, 1.0),
                              "stage": "Painting texture", "image": img,
                              "out": out, "quality": st.quality,
                              "faces": job["faces"],
                              "texture": job["texture"],
                              "seed": job["seed"]})
            else:
                fine_out = out
                chain[0]["out"] = out
            job["texture"] = 0
            job["faces"] = 10000000
            job["out"] = shape_out
        elif hy and job.get("views") and job["texture"] > 0 and \
                False:  # painting is always Hunyuan now
            if not (engine_ready("PIXAL3D") and pixal_mv_ready()):
                st.status = ("Pixal3D multi-view files are not installed - "
                             "paint with Hunyuan instead")
                _redraw()
                return {"FINISHED"}
            # Hunyuan builds the shape only; Pixal3D then paints it from
            # every picture
            chain = {"task": "texture_mesh", "image": img, "out": out,
                     "views": job["views"], "texture": job["texture"],
                     "seed": job["seed"]}
            job["texture"] = 0
            job["out"] = out[:-4] + "_shape.glb"
        CARD_JOB = {"busy": True, "frac": 0.0, "stage": "Starting",
                    "line": "", "t0": time.time(), "engine": st.engine}
        if chain:
            CARD_JOB["chain"] = chain
            CARD_JOB["span"] = (0.0, 0.25) if hi3d else (0.0, 0.45)
        if job_tex_ > 0 and st.auto_enhance == "ON" and \
                klein_ready() and engine_ready("HUNYUAN") and \
                not (chain and not hi3d):
            # the texture is enhanced as soon as the model is in the scene
            CARD_JOB["then_enhance"] = True
        if not hy and st.auto_detail == "ON" and engine_ready("HI3D") and \
                engine_ready("HUNYUAN"):
            # TRELLIS / Pixal3D: their model gets Hi3D detail (and a fresh
            # texture) straight after, like Enhance Detail by hand
            CARD_JOB["then_detail"] = True
            if job_tex_ <= 0:
                CARD_JOB["detail_notex"] = True
        st.status = ""
        try:
            srv = "HUNYUAN" if hy else "TRELLIS"
            if not _server_alive(srv):
                _server_start(srv)
            _send(job)
        except Exception as exc:
            CARD_JOB = None
            server_stop()
            st.status = "Could not start the engine: %s" % exc
            _redraw()
            return {"FINISHED"}
        if not bpy.app.timers.is_registered(_poll_gen):
            bpy.app.timers.register(_poll_gen, first_interval=0.25)
        _redraw()
        return {"FINISHED"}



def _free_viewport_gpu():
    """A viewport left in Rendered mode keeps the graphics card busy (Cycles
    or EEVEE redrawing) while the engine needs all of it: drop those
    viewports to Solid. Returns how many were switched."""
    n = 0
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for sp in area.spaces:
                    if sp.type == "VIEW_3D" and sp.shading.type == "RENDERED":
                        sp.shading.type = "SOLID"
                        n += 1
                area.tag_redraw()
    except Exception:
        pass
    if n:
        try:
            card.say("Viewport set to Solid so the graphics card is free")
        except Exception:
            pass
    return n


def _mat_images(o):
    """(base colour image, metallic image, roughness image, packed glTF
    metal/rough image) of the model's first material, following the links
    back from the shader's inputs."""
    m = o.active_material
    if m is None or not m.use_nodes:
        return None, None, None, None
    bsdf = next((n for n in m.node_tree.nodes
                 if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None, None, None, None

    def src(sock, seen=None):
        seen = seen or set()
        for l_ in sock.links:
            n = l_.from_node
            if n in seen:
                continue
            seen.add(n)
            if n.type == "TEX_IMAGE" and n.image is not None:
                return n.image, l_.from_socket.name
            for i_ in n.inputs:
                r_ = src(i_, seen)
                if r_[0] is not None:
                    return r_
        return None, None
    col, _ = src(bsdf.inputs["Base Color"])
    met, _ = src(bsdf.inputs["Metallic"])
    rou, _ = src(bsdf.inputs["Roughness"])
    gltf = met if (met is not None and met is rou) else None
    return col, met, rou, gltf


def enhance_target(context):
    """The model Enhance Texture can work on: one this add-on painted (its
    painted views are kept), or any textured model - TRELLIS, Pixal3D or
    one from elsewhere - whose own texture then becomes the views."""
    o = context.view_layer.objects.active
    if o is None or o.type != "MESH":
        return None
    w = o.get("img3d_work")
    if w and os.path.isdir(os.path.join(w, "views")):
        return o
    if o.data.uv_layers and _mat_images(o)[0] is not None:
        return o
    return None


def _export_textured(o, work):
    """The model as the engine reads it (its own UVs, engine axes) plus its
    colour and metal/rough pictures, for Enhance Texture on any model."""
    import numpy as np
    os.makedirs(work, exist_ok=True)
    me = o.data
    me.calc_loop_triangles()
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    co = np.stack([co[:, 0], co[:, 2], -co[:, 1]], 1)
    nt = len(me.loop_triangles)
    tv = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tv)
    tl = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("loops", tl)
    uvl = me.uv_layers.active or me.uv_layers[0]
    uv = np.empty(len(me.loops) * 2, np.float32)
    uvl.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)[tl]
    key = np.concatenate([tv[:, None].astype(np.float64),
                          np.round(uv * 1e6)], 1)
    _, first, inv = np.unique(key, axis=0, return_index=True,
                              return_inverse=True)
    np.savez(os.path.join(work, "mesh.npz"), v=co[tv[first]],
             f=inv.reshape(-1, 3).astype(np.int32), uv=uv[first])
    col, met, rou, gltf = _mat_images(o)

    def save(img, path):
        c_ = img.copy()
        c_.filepath_raw = path
        c_.file_format = "PNG"
        c_.save()
        bpy.data.images.remove(c_)
        return path
    alb = save(col, os.path.join(work, "albedo_src.png"))
    mr = None
    try:
        ref = gltf or rou or met
        if ref is not None:
            w_, h_ = ref.size
            def px(img):
                if img is None or tuple(img.size) != (w_, h_):
                    return None
                a = np.empty(w_ * h_ * 4, np.float32)
                img.pixels.foreach_get(a)
                return a.reshape(h_, w_, 4)
            out = np.zeros((h_, w_, 4), np.float32)
            out[..., 3] = 1.0
            if gltf is not None:
                g = px(gltf)
                out[..., 0] = g[..., 2]          # metal (glTF blue)
                out[..., 1] = g[..., 1]          # rough (glTF green)
            else:
                r_, m_ = px(rou), px(met)
                out[..., 1] = r_[..., 0] if r_ is not None else 0.5
                out[..., 0] = m_[..., 0] if m_ is not None else 0.0
            im = bpy.data.images.new("img3d_mr_tmp", w_, h_, alpha=True)
            im.pixels.foreach_set(out.ravel())
            mr = os.path.join(work, "mr_src.png")
            im.filepath_raw = mr
            im.file_format = "PNG"
            im.save()
            bpy.data.images.remove(im)
    except Exception as exc:
        print("[Image to 3D] no metal/rough picture: %s" % exc)
        mr = None
    tex = max(col.size[0], col.size[1]) if col is not None else 2048
    tex = 4096 if tex > 3000 else 2048 if tex > 1500 else 1024
    return os.path.join(work, "mesh.npz"), alb, mr, tex


class IMG3D_OT_enhance(bpy.types.Operator):
    """Sharpen the texture of the selected generated model: its painted
    views are upscaled again and baked onto it (no new model, about a
    minute)"""
    bl_idname = "img3d.enhance"
    bl_label = "Enhance Texture"

    def execute(self, context):
        _free_viewport_gpu()
        global CARD_JOB
        st = context.scene.img3d
        o = enhance_target(context)
        if o is None:
            st.status = "Select a textured model to enhance it"
            _redraw()
            return {"FINISHED"}
        if CARD_JOB is not None:
            return {"FINISHED"}
        if not software_ready("KLEIN"):
            st.status = ("Enhance Texture is not installed yet - press "
                         "Install Enhance Tools in the Enhance section")
            _redraw()
            return {"FINISHED"}
        if not klein_ready():
            st.status = ("Enhance Texture is not downloaded yet - get it in "
                         "the Downloads section")
            card.focus_downloads("KLEIN")
            _redraw()
            return {"FINISHED"}
        outdir = os.path.join(engine_dir(), "output")
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, "%s_enhanced_%s.glb" % (
            safe_name(o.name), time.strftime("%Y%m%d_%H%M%S")))
        if not engine_ready("HUNYUAN"):
            st.status = ("Enhance Texture runs on the Hunyuan engine - "
                         "install Hunyuan first")
            _redraw()
            return {"FINISHED"}
        w_ = o.get("img3d_work")
        if w_ and os.path.isdir(os.path.join(w_, "views")):
            job = {"task": "enhance", "work": w_, "out": out,
                   "texture": int(o.get("img3d_tex", int(st.texture)))}
            new_work = None
        else:
            # a model this add-on did not paint: its own texture is used
            new_work = os.path.join(outdir, "_enhance", "%s_%s" % (
                bpy.path.clean_name(o.name), time.strftime("%Y%m%d_%H%M%S")))
            try:
                mesh_, alb_, mr_, tex_ = _export_textured(o, new_work)
            except Exception as exc:
                st.status = "Could not read the model's texture: %s" % exc
                _redraw()
                return {"FINISHED"}
            job = {"task": "enhance_any", "work": new_work, "mesh": mesh_,
                   "albedo": alb_, "mr": mr_, "texture": tex_}
        job.update({"mode": st.enhance_mode,
                    "creativity": round(float(st.creativity), 2)})
        CARD_JOB = {"busy": True, "frac": 0.0, "stage": "Enhancing texture",
                    "line": "", "t0": time.time(), "engine": "HUNYUAN",
                    "enhance": o.name, "set_work": new_work}
        st.status = ""
        try:
            if not _server_alive("HUNYUAN"):
                _server_start("HUNYUAN")
            _send(job)
        except Exception as exc:
            CARD_JOB = None
            server_stop()
            st.status = "Could not start the engine: %s" % exc
            _redraw()
            return {"FINISHED"}
        if not bpy.app.timers.is_registered(_poll_gen):
            bpy.app.timers.register(_poll_gen, first_interval=0.25)
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_cancel(bpy.types.Operator):
    """Stop what the engine is doing"""
    bl_idname = "img3d.cancel"
    bl_label = "Cancel"

    def execute(self, context):
        if CARD_JOB is not None:
            CARD_JOB["cancelled"] = True
            server_stop()
            _gen_done(False, "Cancelled")
            return {"FINISHED"}
        p = _PROC["p"]
        job = _job()
        if job is not None:
            job["cancelled"] = True
        if p is not None and p.poll() is None:
            try:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               capture_output=True, creationflags=NOWINDOW)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_install_engine(bpy.types.Operator):
    """Download and set up the AI engine and its model files (one time)"""
    bl_idname = "img3d.install_engine"
    bl_label = "Install Engine"
    target: StringProperty(default="")

    @classmethod
    def poll(cls, context):
        return bool(engine_dir())

    def invoke(self, context, event):
        return self.execute(context)

    def execute(self, context):
        global SETUP_JOB
        p_ = _PROC["p"]
        if p_ is not None:
            if p_.poll() is None:
                context.scene.img3d.status = (
                    "Wait for the job that is running to finish, then press "
                    "Install again")
                _redraw()
                return {"FINISHED"}
            _PROC["p"] = None        # an earlier job that already ended
        d = engine_dir()
        os.makedirs(d, exist_ok=True)
        server_stop()
        SETUP_JOB = {"busy": True, "frac": 0.0, "stage": "Installing",
                     "line": "", "t0": time.time()}
        args = [sys.executable, "-u", os.path.join(ADDON_DIR, "setup_engine.py"),
                d, ADDON_DIR]
        if self.target == "EXTRAS":
            args.append("--extras")
        elif self.target == "CANVAS":
            args.append("--canvas")
        elif self.target == "HUNYUAN" or (
                not self.target and context.scene.img3d.engine == "HUNYUAN"):
            args.append("--hunyuan")
        try:
            _start("setup", args, d)
        except Exception as exc:
            SETUP_JOB = None
            context.scene.img3d.status = "Could not start the install: %s" % exc
        _redraw()
        return {"FINISHED"}


def detail_faces(st):
    """How many faces the detailed model keeps: the Faces setting of the
    engine in use (the Hunyuan one only when Hunyuan made the model), so a
    1.4M TRELLIS model is not cut down to Hunyuan's 40K."""
    return int(st.faces_hy if st.engine in ("HUNYUAN", "HI3D") else st.faces)


def detail_target(context):
    """The selected generated model the Detail Enhancer can work on."""
    o = context.view_layer.objects.active
    if o is None or o.type != "MESH" or not o.select_get():
        return None
    img = o.get("img3d_image") or context.scene.img3d.image_path
    img = bpy.path.abspath(img) if img else ""
    return o if img and os.path.isfile(img) else None


def _start_chain(context, first_srv, first_job, chain, extra):
    global CARD_JOB
    st = context.scene.img3d
    CARD_JOB = {"busy": True, "frac": 0.0, "stage": "Starting", "line": "",
                "t0": time.time(), "engine": "HI3D"}
    CARD_JOB.update(extra)
    if chain:
        CARD_JOB["chain"] = chain
    st.status = ""
    try:
        _server_start(first_srv)
        _send(first_job)
    except Exception as exc:
        CARD_JOB = None
        server_stop()
        st.status = "Could not start the engine: %s" % exc
        _redraw()
        return
    if not bpy.app.timers.is_registered(_poll_gen):
        bpy.app.timers.register(_poll_gen, first_interval=0.25)
    _redraw()


class IMG3D_OT_enhance_detail(bpy.types.Operator):
    """Rebuild the selected model with sharp edges and fine detail (Hi3D),
    keeping its shape, then paint it again. The old model is kept hidden"""
    bl_idname = "img3d.enhance_detail"
    bl_label = "Enhance Detail"

    no_tex: BoolProperty(default=False, options={"SKIP_SAVE", "HIDDEN"})

    def execute(self, context):
        _free_viewport_gpu()
        st = context.scene.img3d
        o = detail_target(context)
        if o is None:
            st.status = "Select a generated model to enhance its detail"
            _redraw()
            return {"FINISHED"}
        if CARD_JOB is not None or not engine_ready("HI3D"):
            return {"FINISHED"}
        img = bpy.path.abspath(o.get("img3d_image") or st.image_path)
        outdir = os.path.join(engine_dir(), "output")
        os.makedirs(outdir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = os.path.join(outdir, "%s_detail_%s" % (
            safe_name(o.name), stamp))
        # the shape as the engine made it when it is still there,
        # otherwise the model in the scene
        work = o.get("img3d_work") or ""
        coarse = os.path.join(work, "white.glb") if work else ""
        pre = None
        if not (coarse and os.path.isfile(coarse)):
            coarse = _export_for_detail(o, base + "_in.ply")
            # the result comes back turned like the scene: undo that turn
            # before it is put in the old model's place
            pre = [list(r) for r in o.matrix_world.to_quaternion()
                   .inverted().to_matrix().to_4x4()]
        fine = base + "_hi3d.glb"
        # no texture when the user turned Texture off for this run
        tex = st.use_texture == "ON" and not self.no_tex
        out = base + ".glb"
        first = {"image": img, "mesh": coarse,
                 "out": fine if tex else out,
                 "resolution": 512 if st.quality == "FAST" else 1024,
                 "seed": int(st.seed)}
        chain = None
        extra = {"replace": o.name, "image": img,
                 "fine": fine if tex else out, "span": (0.0, 1.0),
                 "stage": "Adding detail", "pre": pre}
        if tex:
            extra["span"] = (0.0, 0.5)
            chain = [{"server": "HUNYUAN", "span": (0.5, 1.0),
                      "stage": "Painting texture", "image": img, "out": out,
                      "quality": st.quality,
                      "faces": detail_faces(st),
                      "texture": int(st.texture), "seed": int(st.seed)}]
        else:
            extra["gray"] = True
        _start_chain(context, "HI3D", first, chain, extra)
        return {"FINISHED"}


class IMG3D_OT_paint_detail(bpy.types.Operator):
    """Paint a texture on the selected detailed model (no new shape)"""
    bl_idname = "img3d.paint_detail"
    bl_label = "Paint Texture"

    def execute(self, context):
        _free_viewport_gpu()
        st = context.scene.img3d
        o = context.view_layer.objects.active
        if o is None or not o.get("img3d_gray") or CARD_JOB is not None:
            return {"FINISHED"}
        fine = o.get("img3d_fine") or ""
        img = bpy.path.abspath(o.get("img3d_image") or st.image_path)
        if not os.path.isfile(fine):
            st.status = "The detailed shape file is gone - run Enhance Detail"
            _redraw()
            return {"FINISHED"}
        out = fine[:-4] + "_painted_%s.glb" % time.strftime("%H%M%S")
        job = {"image": img, "mesh": fine, "out": out, "quality": st.quality,
               "faces": detail_faces(st), "texture": int(st.texture),
               "seed": int(st.seed)}
        _start_chain(context, "HUNYUAN", job, None,
                     {"replace": o.name, "image": img, "fine": fine,
                      "stage": "Painting texture"})
        return {"FINISHED"}


def _droplog(msg):
    return
    try:
        with open(os.path.join(os.path.dirname(__file__), "drop.log"),
                  "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    except Exception:
        pass


class IMG3D_OT_drop_image(bpy.types.Operator):
    """Use a picture dragged onto the card: on the big picture it becomes
    the front image, on Back / Left / Right it fills that view"""
    bl_idname = "img3d.drop_image"
    bl_label = "Drop Image"
    bl_options = {"INTERNAL"}

    filepath: StringProperty(subtype="FILE_PATH", options={"SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"SKIP_SAVE"})
    files: bpy.props.CollectionProperty(type=bpy.types.OperatorFileListElement,
                                        options={"SKIP_SAVE"})

    def _paths(self):
        out = []
        if self.directory and len(self.files):
            for f in self.files:
                p = os.path.join(self.directory, f.name)
                if os.path.isfile(p):
                    out.append(p)
        elif self.filepath and os.path.isfile(self.filepath):
            out.append(self.filepath)
        return out

    def invoke(self, context, event):
        paths = self._paths()
        _droplog("drop invoke: %s region=%s mouse=%d,%d" % (
            paths, getattr(context.region, "type", None),
            event.mouse_x, event.mouse_y))
        if not paths:
            return {"CANCELLED"}
        st = context.scene.img3d
        # where on the card it landed: the card is painted on the 3D view's
        # main region, whichever region (sidebar, header) took the drop
        hit = card.card_point(context, event.mouse_x, event.mouse_y)
        _droplog("card point: %s" % (hit,))
        if hit is None:
            # dropped on the scene, not on the card: what Blender would do
            try:
                win, area, region = card.hot_view(context)
                rv = None
                if area is not None:
                    rv = area.spaces.active.region_3d
                if rv is not None and rv.view_perspective == "CAMERA":
                    with context.temp_override(window=win, area=area,
                                               region=region):
                        bpy.ops.view3d.camera_background_image_add(
                            filepath=paths[0])
                elif area is not None:
                    with context.temp_override(window=win, area=area,
                                               region=region):
                        bpy.ops.object.empty_image_add(filepath=paths[0],
                                                       align="VIEW")
            except Exception as exc:
                _droplog("scene drop failed: %s" % exc)
            return {"FINISHED"}
        slot = None
        try:
            ws = card._state.get("widgets") or []
            i = card._hit(ws, hit[0], hit[1])
            tip = ws[i].get("tip", "") if i is not None else ""
            if tip in ("view1", "view2", "view3"):
                slot = int(tip[-1])
        except Exception:
            slot = None
        if slot is not None:
            setattr(st, "view_%d" % slot, paths[0])
        else:
            st.image_path = paths[0]
            # more pictures dropped at once fill Back, Left, Right
            for n, p in enumerate(paths[1:4], 1):
                setattr(st, "view_%d" % n, p)
        st.status = ""
        _droplog("set slot=%s" % slot)
        _redraw()
        return {"FINISHED"}

    def execute(self, context):
        paths = self._paths()
        if paths:
            context.scene.img3d.image_path = paths[0]
            _redraw()
        return {"FINISHED"}


class IMG3D_FH_image(bpy.types.FileHandler):
    bl_idname = "IMG3D_FH_image"
    bl_label = "Image to 3D picture"
    bl_import_operator = "img3d.drop_image"
    bl_file_extensions = ".png;.jpg;.jpeg;.webp;.bmp;.tif;.tiff"

    @classmethod
    def poll_drop(cls, context):
        a = context.area
        return a is not None and a.type == "VIEW_3D"


# Blender's own picture drops in the 3D view would compete with this one and
# make Blender ask which to do. They are set aside while the add-on is on
# (at register time only, never during a drag) and the drop operator does
# their job itself when the picture lands outside the card.
_FH_OFF = []


def _fh_patch():
    for name in ("VIEW3D_FH_empty_image", "VIEW3D_FH_camera_background_image"):
        cls = getattr(bpy.types, name, None)
        if cls is None:
            continue
        try:
            bpy.utils.unregister_class(cls)
            _FH_OFF.append(cls)
        except Exception:
            pass


def _fh_unpatch():
    while _FH_OFF:
        cls = _FH_OFF.pop()
        try:
            bpy.utils.register_class(cls)
        except Exception:
            pass


class IMG3D_OT_undo_texture(bpy.types.Operator):
    """Put back the texture the model had before the last Enhance Texture"""
    bl_idname = "img3d.undo_texture"
    bl_label = "Undo Enhance Texture"

    def execute(self, context):
        o = context.view_layer.objects.active
        st = context.scene.img3d
        if o is None or not o.get("img3d_prev_mat"):
            return {"FINISHED"}
        prev = bpy.data.materials.get(o["img3d_prev_mat"])
        if prev is None:
            st.status = "The earlier texture is no longer in the file"
            del o["img3d_prev_mat"]
            _redraw()
            return {"FINISHED"}
        cur = o.active_material
        for i_ in range(max(1, len(o.material_slots))):
            if o.material_slots:
                o.material_slots[i_].material = prev
        prev.use_fake_user = False
        del o["img3d_prev_mat"]
        if cur is not None and cur is not prev and cur.users == 0:
            bpy.data.materials.remove(cur)
        st.status = "Earlier texture back"
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_undo_detail(bpy.types.Operator):
    """Go back to the model as it was before Enhance Detail (the detailed
    one is removed)"""
    bl_idname = "img3d.undo_detail"
    bl_label = "Back to Previous Model"

    def execute(self, context):
        o = context.view_layer.objects.active
        st = context.scene.img3d
        if o is None or not o.get("img3d_prev"):
            return {"FINISHED"}
        prev = bpy.data.objects.get(o["img3d_prev"])
        if prev is None:
            st.status = "The earlier model is no longer in the file"
            del o["img3d_prev"]
            _redraw()
            return {"FINISHED"}
        prev.hide_set(False)
        prev.hide_render = False
        me = o.data
        bpy.data.objects.remove(o)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
        for x in context.selected_objects:
            x.select_set(False)
        prev.select_set(True)
        context.view_layer.objects.active = prev
        st.status = "Earlier model back"
        _redraw()
        return {"FINISHED"}


# -- model downloads ------------------------------------------------------------

class IMG3D_OT_dl_get(bpy.types.Operator):
    """Download this part's model files from their makers' official pages
    (Hugging Face / GitHub). Every file is checked against the official
    one. You can stop at any time and continue later"""
    bl_idname = "img3d.dl_get"
    bl_label = "Download"
    group: StringProperty()
    verify: BoolProperty(default=False, options={"SKIP_SAVE"})

    def execute(self, context):
        st = context.scene.img3d
        root = engine_dir()
        if not root:
            st.status = "Set the Engine Folder in the add-on preferences first"
            _redraw()
            return {"FINISHED"}
        g = dl.group(self.group)
        if g is None:
            return {"FINISHED"}
        gids = [self.group]
        need = g.get("needs")
        if need and not self.verify and not dl.have(root, need):
            gids.insert(0, need)
        if not dl.start(root, gids, verify=self.verify, redraw=_redraw):
            st.status = "Another download is running - wait for it or stop it"
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_dl_cancel(bpy.types.Operator):
    """Stop the download. What was already downloaded is kept and the next
    Download continues from there"""
    bl_idname = "img3d.dl_cancel"
    bl_label = "Stop Download"

    def execute(self, context):
        dl.cancel()
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_dl_page(bpy.types.Operator):
    """Download the files yourself: lists every part this model needs, with
    its official page and the exact folder its files go into"""
    bl_idname = "img3d.dl_page"
    bl_label = "Download by Hand"
    group: StringProperty()

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=560)

    def draw(self, context):
        lay = self.layout
        g = dl.group(self.group)
        root = engine_dir()
        lay.label(text="%s - download by hand" % g["name"], icon="IMPORT")
        col = lay.column(align=True)
        col.label(text="1. Open Page: download all the files on that page.")
        col.label(text="2. Open Folder: put them in that folder, keeping the "
                       "same sub-folders (for example ckpts).")
        col.label(text="3. Press Check Files when every part is there.")
        for cid, name, page, folder, files, ok in dl.parts(self.group, root):
            box = lay.box()
            row = box.row()
            row.label(text=name, icon="CHECKMARK" if ok else "BLANK1")
            op = row.operator("img3d.dl_open", text="Open Page", icon="URL")
            op.url = page
            op.folder = ""
            op = row.operator("img3d.dl_open", text="Open Folder",
                              icon="FILE_FOLDER")
            op.url = ""
            op.folder = folder
            box.label(text=folder)
            if len(files) <= 6:
                box.label(text="Files: " + ", ".join(
                    os.path.basename(f) for f in files)[:110])
            else:
                box.label(text="%d files" % len(files))
        op = lay.operator("img3d.dl_get", text="Check Files",
                          icon="CHECKMARK")
        op.group = self.group
        op.verify = True

    def execute(self, context):
        return {"FINISHED"}


class IMG3D_OT_dl_open(bpy.types.Operator):
    """Open a model's official page, or the folder its files go into"""
    bl_idname = "img3d.dl_open"
    bl_label = "Open"
    url: StringProperty(options={"SKIP_SAVE"})
    folder: StringProperty(options={"SKIP_SAVE"})

    def execute(self, context):
        if self.url:
            bpy.ops.wm.url_open(url=self.url)
        elif self.folder:
            os.makedirs(self.folder, exist_ok=True)
            try:
                os.startfile(self.folder)
            except Exception:
                bpy.ops.wm.path_open(filepath=self.folder)
        return {"FINISHED"}


class IMG3D_OT_dl_folder(bpy.types.Operator):
    """Open the folder the models go into"""
    bl_idname = "img3d.dl_folder"
    bl_label = "Open Models Folder"

    def execute(self, context):
        root = engine_dir()
        if root:
            dl.open_folder(root)
        return {"FINISHED"}


class IMG3D_OT_go_downloads(bpy.types.Operator):
    """Open the Downloads section at this model"""
    bl_idname = "img3d.go_downloads"
    bl_label = "Go to Downloads"
    group: StringProperty()

    def execute(self, context):
        card.focus_downloads(self.group)
        _redraw()
        return {"FINISHED"}


class IMG3D_OT_pick_engine_dir(bpy.types.Operator):
    """Choose the folder, on any drive, that the engine and its model files
    are installed and downloaded into"""
    bl_idname = "img3d.pick_engine_dir"
    bl_label = "Choose Install Folder"
    directory: StringProperty(subtype="DIR_PATH")

    @classmethod
    def poll(cls, context):
        return _PROC["p"] is None and CARD_JOB is None and not dl.busy()

    def invoke(self, context, event):
        cur = engine_dir()
        self.directory = (cur if cur and os.path.isdir(cur)
                          else os.path.expanduser("~")) + os.sep
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        new = bpy.path.abspath(self.directory or "").rstrip("\\/")
        if not new:
            return {"CANCELLED"}
        # a bare drive (F:) gets a folder of its own on that drive
        if os.path.splitdrive(new)[1] in ("", "\\", "/"):
            new = os.path.join(new + os.sep, "3DRedbox", "ImageTo3D")
        p = _prefs()
        if p is None:
            return {"CANCELLED"}
        server_stop()
        p.engine_dir = new
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            pass
        try:
            card.say("Install folder: " + new)
        except Exception:
            pass
        _redraw()
        return {"FINISHED"}


_FREE = {"t": 0.0, "k": None, "v": None}


def drive_free_gb(path):
    """Free space on the drive a folder is on (the folder need not exist
    yet). Asked at most every few seconds."""
    if not path:
        return None
    now = time.time()
    if _FREE["k"] == path and now - _FREE["t"] < 5.0:
        return _FREE["v"]
    p = path
    while p and not os.path.exists(p):
        up = os.path.dirname(p)
        if up == p:
            break
        p = up
    try:
        import shutil as _sh
        v = _sh.disk_usage(p).free / 2**30
    except Exception:
        v = None
    _FREE.update(t=now, k=path, v=v)
    return v


def _run(args, timeout=60):
    try:
        r = subprocess.run(args, capture_output=True, timeout=timeout,
                           creationflags=NOWINDOW)
        return (r.stdout + r.stderr).decode("utf-8", "replace").strip()
    except Exception as exc:
        return "(could not run: %s)" % exc


_ENV_CHECK = r"""
import sys, platform
print("python", sys.version.split()[0])
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    ok = torch.cuda.is_available()
    print("cuda available", ok)
    if ok:
        p = torch.cuda.get_device_properties(0)
        print("device", p.name, "sm_%d%d" % (p.major, p.minor),
              "%.1f GB" % (p.total_memory / 2**30))
        print("arch list", torch.cuda.get_arch_list())
        x = torch.ones(1024, 1024, device="cuda")
        print("matmul ok", float((x @ x).sum()) > 0)
except Exception as e:
    print("torch ERROR", repr(e))
for m in MODS:
    try:
        __import__(m)
        print("import", m, "ok")
    except Exception as e:
        print("import", m, "ERROR", repr(e)[:300])
"""

_ENV_MODS = {
    "env": ["o_voxel", "cumesh", "flex_gemm", "flash_attn"],
    "env_hy": ["custom_rasterizer"],
    "env_hi3d": [],
    "env_klein": ["diffusers"],
}


def _desktop():
    home = os.path.expanduser("~")
    for d in (os.path.join(home, "Desktop"),
              os.path.join(home, "OneDrive", "Desktop")):
        if os.path.isdir(d):
            return d
    return home


def build_report():
    import platform
    L = []
    add = L.append
    add("Image to 3D 2 support report  %s" % VERSION)
    add("Made " + time.strftime("%Y-%m-%d %H:%M:%S"))
    add("")
    add("== System ==")
    add("Blender " + bpy.app.version_string)
    add("OS " + platform.platform())
    add("CPU " + (platform.processor() or "?"))
    try:
        tot, free = _win_mem()
        add("RAM %.1f GB total, %.1f GB free" % (tot, free))
    except Exception as exc:
        add("RAM ? (%s)" % exc)
    add("GPU " + _run(["nvidia-smi", "--query-gpu=name,memory.total,"
                       "memory.used,driver_version",
                       "--format=csv,noheader"], 20))
    smi = _run(["nvidia-smi"], 20)
    for l_ in smi.splitlines():
        if "CUDA Version" in l_:
            add("Driver line: " + l_.strip(" |"))
            break
    d = engine_dir()
    add("")
    add("== Engine ==")
    add("Folder " + (d or "(not set)"))
    if d and os.path.isdir(d):
        try:
            import shutil
            u = shutil.disk_usage(d)
            add("Disk free %.1f GB" % (u.free / 2**30))
        except Exception:
            pass
        for env, mods in _ENV_MODS.items():
            py = os.path.join(d, env, "Scripts", "python.exe")
            if not os.path.exists(py):
                add("[%s] not installed" % env)
                continue
            code = "MODS = %r\n" % mods + _ENV_CHECK
            add("[%s]" % env)
            for l_ in _run([py, "-c", code], 180).splitlines()[-30:]:
                add("    " + l_)
        mdir = os.path.join(d, "models")
        if os.path.isdir(mdir):
            add("Models:")
            for org in sorted(os.listdir(mdir)):
                p_ = os.path.join(mdir, org)
                if os.path.isdir(p_):
                    add("    %s: %s" % (org, ", ".join(sorted(os.listdir(p_)))))
        try:
            add("Downloads:")
            for g in dl.groups():
                st_, have, total, miss = dl.status(d, g["id"], fresh=True)
                add("    %s: %s  %.1f / %.1f GB%s" % (
                    g["id"], st_, have / 2**30, total / 2**30,
                    ("  missing: " + ", ".join("%s/%s" % m_ for m_ in miss[:8]))
                    if miss and st_ != "missing" else ""))
            for f_ in sorted(os.listdir(d)):
                if f_.startswith("INSTALLED"):
                    add("    flag " + f_)
        except Exception as exc:
            add("Downloads: ? (%s)" % exc)
    add("")
    add("== Settings ==")
    st = getattr(bpy.context.scene, "img3d", None)
    if st is not None:
        for k in st.bl_rna.properties.keys():
            if k in ("rna_type", "name"):
                continue
            try:
                v = getattr(st, k)
                if isinstance(v, (int, float, str, bool)):
                    add("%s = %s" % (k, str(v)[:200]))
            except Exception:
                pass
    add("")
    add("== Canvas logs ==")
    try:
        import glob as _g
        cdir = os.path.join(engine_dir(), "canvas", "canvas")
        for f_ in sorted(_g.glob(os.path.join(cdir, "*.log")),
                         key=os.path.getmtime)[-6:]:
            add("--- %s" % os.path.basename(f_))
            L.extend(open(f_, encoding="utf-8", errors="replace")
                     .read().splitlines()[-120:])
    except Exception as exc:
        add("(no Canvas logs: %s)" % exc)
    add("")
    add("== Last engine output ==")
    lp = log_path()
    for f_ in (lp[:-4] + ".old.log", lp):
        if os.path.exists(f_):
            try:
                lines = open(f_, encoding="utf-8",
                             errors="replace").read().splitlines()
                L.extend(lines[-800:] if f_ == lp else lines[-200:])
            except Exception as exc:
                add("(could not read %s: %s)" % (f_, exc))
    if _SRV.get("lines"):
        add("")
        add("== Engine output still in memory ==")
        L.extend(_SRV["lines"])
    return "\n".join(L)


class IMG3D_OT_save_report(bpy.types.Operator):
    """Save a report of this PC and the last run to the Desktop, to send
    when a model will not generate"""
    bl_idname = "img3d.save_report"
    bl_label = "Save Report"
    bl_description = ("Something not working? Saves a report of your system "
                      "and the last run on your Desktop - send it to Milad")

    def execute(self, context):
        try:
            card.say("Making the report...")
            _redraw()
        except Exception:
            pass
        try:
            text = build_report()
        except Exception as exc:
            import traceback
            text = "Report failed: %s\n%s" % (exc, traceback.format_exc())
        path = os.path.join(_desktop(), "Image_to_3D_Report_%s.txt"
                            % time.strftime("%Y-%m-%d_%H-%M"))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as exc:
            self.report({"ERROR"}, "Could not save the report: %s" % exc)
            return {"CANCELLED"}
        try:
            subprocess.Popen(["explorer", "/select,", path])
        except Exception:
            pass
        st = getattr(context.scene, "img3d", None)
        if st is not None:
            st.status = "Report saved on the Desktop - send it to Milad"
        try:
            card.say("Report saved on your Desktop")
        except Exception:
            pass
        self.report({"INFO"}, "Report saved: " + path)
        return {"FINISHED"}


CLASSES = (IMG3D_Settings, IMG3D_OT_pick_image,
           IMG3D_OT_clear_image,
           IMG3D_OT_generate, IMG3D_OT_enhance, IMG3D_OT_cancel,
           IMG3D_OT_install_engine, IMG3D_OT_enhance_detail,
           IMG3D_OT_paint_detail, IMG3D_OT_drop_image,
           IMG3D_FH_image, IMG3D_OT_undo_texture,
           IMG3D_OT_undo_detail, IMG3D_OT_dl_get, IMG3D_OT_dl_cancel,
           IMG3D_OT_dl_page, IMG3D_OT_dl_open, IMG3D_OT_dl_folder,
           IMG3D_OT_go_downloads, IMG3D_OT_pick_engine_dir,
           IMG3D_OT_save_report) + VIEW_OPS


# -- system meters: graphics memory, main memory and processor load ------------

_STATS = {"cpu": None, "gpu": None, "ram": None, "vram": None, "run": False}


def _cpu_times():
    import ctypes
    from ctypes import wintypes
    i_, k_, u_ = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(i_), ctypes.byref(k_),
                                          ctypes.byref(u_))

    def f(t):
        return (t.dwHighDateTime << 32) | t.dwLowDateTime
    # kernel time already includes idle time
    return f(i_), f(k_) + f(u_)


_NVML = {"lib": None, "h": None, "tried": False, "mi": None, "ut": None}


def _nvml():
    """NVIDIA's own driver library, read in place: no program is started,
    so asking costs next to nothing."""
    if _NVML["tried"]:
        return _NVML["lib"]
    _NVML["tried"] = True
    import ctypes
    if os.name == "nt":
        names = ["nvml.dll",
                 os.path.join(os.environ.get("WINDIR", r"C:\Windows"),
                              "System32", "nvml.dll"),
                 r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll"]
    else:
        names = ["libnvidia-ml.so.1", "libnvidia-ml.so"]
    lib = None
    for n_ in names:
        try:
            lib = ctypes.CDLL(n_)
            break
        except Exception:
            lib = None
    if lib is None:
        return None
    try:
        if lib.nvmlInit_v2() != 0:
            return None
        h = ctypes.c_void_p()
        if lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) != 0:
            return None
    except Exception:
        return None

    class MI(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong),
                    ("used", ctypes.c_ulonglong)]

    class UT(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]
    _NVML.update(lib=lib, h=h, mi=MI, ut=UT)
    return lib


def _read_vram():
    import ctypes
    lib = _nvml()
    if lib is None:
        return None
    try:
        mi = _NVML["mi"]()
        if lib.nvmlDeviceGetMemoryInfo(_NVML["h"], ctypes.byref(mi)) != 0:
            return None
        ut = _NVML["ut"]()
        if lib.nvmlDeviceGetUtilizationRates(_NVML["h"],
                                             ctypes.byref(ut)) == 0:
            _STATS["gpu"] = ut.gpu / 100.0
        return mi.used / 2**30, mi.total / 2**30
    except Exception:
        return None


def _stats_loop():
    prev = None
    n = 0
    while _STATS["run"]:
        try:
            if os.name == "nt":
                idle, tot = _cpu_times()
                if prev is not None and tot > prev[1]:
                    _STATS["cpu"] = max(0.0, min(1.0, 1.0 - (idle - prev[0]) /
                                                 float(tot - prev[1])))
                prev = (idle, tot)
                t_, a_ = _win_mem()
                _STATS["ram"] = (t_ - a_, t_)
            else:
                try:
                    import psutil
                    _STATS["cpu"] = psutil.cpu_percent() / 100.0
                    vm = psutil.virtual_memory()
                    _STATS["ram"] = ((vm.total - vm.available) / 2**30,
                                     vm.total / 2**30)
                except Exception:
                    pass
            _STATS["vram"] = _read_vram()
        except Exception:
            pass
        n += 1
        time.sleep(2.0)


def _stats_tick():
    if not _STATS["run"]:
        return None
    try:
        for w in bpy.context.window_manager.windows:
            for a in w.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()
    except Exception:
        pass
    return 2.0


def _stats_start():
    if _STATS["run"]:
        return
    _STATS["run"] = True
    threading.Thread(target=_stats_loop, daemon=True).start()
    if not bpy.app.timers.is_registered(_stats_tick):
        bpy.app.timers.register(_stats_tick, first_interval=1.0,
                                persistent=True)


def _stats_stop():
    _STATS["run"] = False
    try:
        if bpy.app.timers.is_registered(_stats_tick):
            bpy.app.timers.unregister(_stats_tick)
    except Exception:
        pass


class IMG3D_OT_clear_cache(bpy.types.Operator):
    bl_idname = "img3d.clear_cache"
    bl_label = "Clear Cache"
    bl_description = ("Close the engine that stays loaded between models, so "
                      "the memory it holds on the graphics card and in RAM is "
                      "freed. The next model loads it again")

    @classmethod
    def poll(cls, context):
        return _PROC["p"] is None and CARD_JOB is None

    def execute(self, context):
        v0 = _STATS.get("vram")
        server_stop()
        import gc
        gc.collect()
        try:
            card.say("Cache cleared")
        except Exception:
            pass
        _STATS["vram"] = None
        self.report({"INFO"}, "Cache cleared")
        return {"FINISHED"}


CONFIG = os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"),
                      "3DRedbox", "image_to_3d.json")
_PREF_KEYS = ("engine_dir", "classic_panel", "accent_hue", "accent_sat",
              "corner", "ui_bright", "welcomed", "auto_update",
              "gentle_download")


def _save_config():
    """The settings also live outside Blender's preferences, so they survive
    moving from the old add-on to the extension (a new preferences entry)."""
    p = _prefs()
    if p is None:
        return
    import json
    try:
        os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
        data = {k: getattr(p, k) for k in _PREF_KEYS if hasattr(p, k)}
        with open(CONFIG, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
    except Exception:
        pass


def _find_engine():
    """An engine already installed in the usual places."""
    import string
    cands = [os.path.join(os.path.expanduser("~"), "3DRedbox", "ImageTo3D")]
    for d in string.ascii_uppercase:
        root = d + ":\\"
        if os.path.exists(root):
            cands.append(os.path.join(root, "3DRedbox", "ImageTo3D"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "INSTALLED")):
            return c
    return ""


def _restore_prefs(old=None):
    """Bring the settings over: from the old add-on's preferences when they
    were read before it was turned off, else from the config file, else find
    an engine that is already installed."""
    p = _prefs()
    if p is None:
        return
    import json
    data = dict(old or {})
    if not data:
        try:
            with open(CONFIG, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
    default_dir = os.path.join(os.path.expanduser("~"), "3DRedbox",
                               "ImageTo3D")
    fresh = (p.engine_dir in ("", default_dir) and
             not os.path.isfile(os.path.join(bpy.path.abspath(p.engine_dir),
                                             "INSTALLED")))
    if not fresh:
        return
    if not data.get("engine_dir"):
        found = _find_engine()
        if found:
            data["engine_dir"] = found
            data.setdefault("welcomed", True)
    for k, v in data.items():
        if k in _PREF_KEYS and hasattr(p, k):
            try:
                setattr(p, k, v)
            except Exception:
                pass
    if data:
        print("[Image to 3D] settings restored, engine folder:",
              p.engine_dir)


def _legacy_prefs(key):
    try:
        lp = bpy.context.preferences.addons[key].preferences
        return {k: getattr(lp, k) for k in _PREF_KEYS if hasattr(lp, k)}
    except Exception:
        return {}


def _other_copies():
    """Image to 3D installed twice: the old way (an add-on folder named
    image_to_3d) and as an extension (bl_ext.<repo>.image_to_3d)."""
    keys = list(bpy.context.preferences.addons.keys())
    return [k for k in keys if k != __package__ and
            (k == "image_to_3d" or k.endswith(".image_to_3d"))]


def _disable_self():
    try:
        import addon_utils
        addon_utils.disable(__package__, default_set=True)
        print("[Image to 3D] the extension version is installed: this "
              "older copy was turned off")
    except Exception as exc:
        print("[Image to 3D] could not turn off the old copy:", exc)
    return None


def _remove_legacy():
    """Delete the old add-on folder (scripts/addons/image_to_3d) once the
    extension version runs. Only the add-on's own files: the engine, the
    downloaded models and the settings live in the Engine Folder."""
    import shutil
    import stat
    import addon_utils
    path = None
    for mod in addon_utils.modules(refresh=False):
        if mod.__name__ == "image_to_3d":
            path = os.path.dirname(getattr(mod, "__file__", "") or "")
    if not path or os.path.abspath(path) == os.path.abspath(ADDON_DIR) or \
            not os.path.isfile(os.path.join(path, "__init__.py")):
        return None

    def onerr(fn, p, _e):
        try:
            os.chmod(p, stat.S_IWRITE)
            fn(p)
        except Exception:
            pass
    try:
        shutil.rmtree(path, onerror=onerr)
        addon_utils.modules(refresh=True)
        print("[Image to 3D] removed the old copy:", path)
    except Exception as exc:
        print("[Image to 3D] could not remove the old copy:", exc)
    return None


_REGISTERED = [False]


def _settings_due(tries=[0]):
    """After Blender has made this add-on's preferences entry."""
    if _prefs() is None and tries[0] < 20:
        tries[0] += 1
        return 0.3
    tries[0] = 0
    try:
        _restore_prefs(_OLD_PREFS)
        _save_config()
    except Exception as exc:
        print("[Image to 3D] settings:", exc)
    # after an update the old copy, going down, can take the card's drawing
    # with it: put it back up when it is missing
    try:
        if card._state.get("handle") is None and not getattr(
                _prefs(), "classic_panel", False):
            card.register_draw()
    except Exception as exc:
        print("[Image to 3D] card:", exc)
    _redraw()
    return None

_OLD_PREFS = {}


def register():
    # All-in-One: мастер-аддон сам глушит раздельные копии — миграция
    # "старый аддон vs расширение" к вшитой копии не относится
    others = [] if VENDORED else _other_copies()
    if others:
        if __package__.startswith("bl_ext."):
            # the extension wins: turn the old add-on off, so the two do
            # not fight over the same buttons, then remove its files
            import addon_utils
            for k in others:
                if not k.startswith("bl_ext."):
                    _OLD_PREFS.update(_legacy_prefs(k))
                    try:
                        addon_utils.disable(k, default_set=True)
                        print("[Image to 3D] turned off the old copy (%s)" % k)
                    except Exception as exc:
                        print("[Image to 3D] old copy:", exc)
            bpy.app.timers.register(_remove_legacy, first_interval=1.0)
        else:
            # this is the old copy and the extension is already there
            bpy.app.timers.register(_disable_self, first_interval=0.5)
            return
    for c in CLASSES:
        bpy.utils.register_class(c)
    updates.register()
    _REGISTERED[0] = True
    bpy.app.timers.register(_settings_due, first_interval=0.3)
    if __package__.startswith("bl_ext."):
        bpy.app.timers.register(_remove_legacy, first_interval=1.0)
        bpy.app.timers.register(updates.auto_update_due, first_interval=8.0)
    bpy.types.Scene.img3d = bpy.props.PointerProperty(type=IMG3D_Settings)
    card.register()
    try:
        # the Canvas used to be a separate test add-on: it is built in now
        import addon_utils
        if "img3d_canvas" in bpy.context.preferences.addons:
            addon_utils.disable("img3d_canvas", default_set=True)
            print("[Image to 3D 2] turned off the old separate Canvas add-on")
    except Exception as exc:
        print("[Image to 3D 2] old canvas:", exc)
    try:
        canvas.register()
    except Exception as exc:
        print("[Image to 3D 2] canvas:", exc)
    bpy.utils.register_class(IMG3D_OT_clear_cache)
    bpy.utils.register_class(IMG3D_OT_welcome)
    if not VENDORED and not bpy.app.timers.is_registered(_welcome_due):
        # All-in-One обходится без приветственной карточки Image to 3D
        bpy.app.timers.register(_welcome_due, first_interval=1.5)
    _stats_start()
    try:
        _fh_patch()
    except Exception as exc:
        print("[Image to 3D] drop patch:", exc)
    import atexit
    atexit.register(server_stop)


def unregister():
    if not _REGISTERED[0]:
        # this copy never registered (another copy owns the buttons)
        return
    _REGISTERED[0] = False
    try:
        dl.cancel()
    except Exception:
        pass
    updates.unregister()
    try:
        canvas.unregister()
    except Exception:
        pass
    try:
        _fh_unpatch()
    except Exception:
        pass
    _stats_stop()
    if bpy.app.timers.is_registered(_welcome_due):
        bpy.app.timers.unregister(_welcome_due)
    try:
        card.welcome_hide()
        bpy.utils.unregister_class(IMG3D_OT_welcome)
    except Exception:
        pass
    try:
        bpy.utils.unregister_class(IMG3D_OT_clear_cache)
    except Exception:
        pass
    server_stop()
    try:
        card.unregister()
    except Exception:
        pass
    p = _PROC.get("p")
    if p is not None and p.poll() is None:
        try:
            p.kill()
        except Exception:
            pass
    try:
        del bpy.types.Scene.img3d
    except Exception:
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
