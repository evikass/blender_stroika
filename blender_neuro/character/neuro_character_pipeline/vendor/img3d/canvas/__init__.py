"""Image to 3D 2 Canvas.

A node canvas in the add-on's own card style. Pick nodes from the bar on the
left, connect them, delete them, the way Blender's node editors work:

  Prompt       a sentence -> a picture (with the "3D Ready" preset)
  Upload       a picture from disk
  Image        holds a picture; every producer puts its result in one
  Find Parts   lists the parts of a picture, then makes a picture of each
  Multi View   the back, left and right of a picture
  Generate 3D  builds the model with the chosen engine, beside the others
"""

bl_info = {
    "name": "Image to 3D Canvas (dev)",
    "author": "Milad Kambari - 3DRedbox Studio",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "Image to 3D card > Open Canvas",
    "description": "Node canvas: pictures, parts, views and 3D models",
    "category": "3D View",
}

import json
import os
import queue
import subprocess
import sys
import threading
import time

import bpy
from bpy.props import StringProperty

from . import ui
import importlib as _il
_il.reload(ui)      # an add-on update brings the new drawing code too

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = "Img3dCanvasTree"
NOWIN = 0x08000000 if os.name == "nt" else 0
ENGINES = (("TRELLIS", "TRELLIS"), ("HUNYUAN", "Hunyuan"),
           ("PIXAL3D", "Pixal3D"))

# type -> (title, width, inputs, outputs)
TYPES = {
    "prompt": ("Prompt", 300, [], ["Image"]),
    "upload": ("Upload", 260, [], ["Image"]),
    "image": ("Image", 270, ["Image"], ["Image"]),
    "parts": ("Find Parts", 330, ["Image"], ["Parts"]),
    "views": ("Multi View", 250, ["Image"], ["Front", "Back", "Left",
                                             "Right"]),
    "gen3d": ("Generate 3D", 290, ["Image", "Back", "Left", "Right"],
              ["Model"]),
    "detail": ("Enhance Detail", 270, ["Model"], ["Model"]),
    "etex": ("Enhance Texture", 270, ["Model"], ["Model"]),
    "upscale": ("Upscale", 260, ["Image"], ["Image"]),
}
PALETTE = ("prompt", "upload", "image", "upscale", "parts", "views",
           "gen3d", "detail", "etex")
SCALES = (("2", "2x"), ("4", "4x"))
# picture shapes for the Prompt node: width x height
# which Qwen paints the Prompt picture: the plain picture model or the
# editing one painting onto the empty backdrop
PMODELS = (("image", "Qwen-Image"), ("edit", "Qwen-Image-Light"))
ASPECTS = (("2x3", "2:3"), ("1x1", "1:1"), ("3x2", "3:2"),
           ("9x16", "9:16"), ("16x9", "16:9"))
QUALITY = (("FAST", "Fast"), ("HIGH", "High"), ("ULTRA", "Ultra"))
FACES = {"TRELLIS": (("50000", "50K"), ("100000", "100K"),
                     ("200000", "200K"), ("500000", "500K")),
         "HUNYUAN": (("20000", "20K"), ("40000", "40K"), ("80000", "80K"),
                     ("150000", "150K")),
         "PIXAL3D": (("50000", "50K"), ("100000", "100K"),
                     ("200000", "200K"), ("500000", "500K"))}
TEXSIZE = (("1024", "1K"), ("2048", "2K"), ("4096", "4K"))


def ins_of(n):
    """The inputs a node shows: the three side views only for Hunyuan."""
    ins = TYPES[n["type"]][2]
    if n["type"] == "gen3d" and n.get("engine") != "HUNYUAN":
        return ins[:1]
    return ins


def out_kind(n, i):
    nm = TYPES[n["type"]][3][i]
    return "model" if nm == "Model" else ("parts" if nm == "Parts" else
                                          "image")


def fits(fk, t, i, tnode=None):
    """Can an output of kind fk go into input i of a node of type t?"""
    nm = TYPES[t][2][i]
    if fk == "model":
        return nm == "Model"
    if nm == "Model":
        return False
    if fk == "parts":
        return t == "image"
    return True


MENU = {}            # the "which connection?" list after dropping a wire


def drop_menu(fr, mx, my, target=None):
    """Released a wire away from a dot: on a node, the inputs it can go
    into; on the empty canvas, the nodes that can take it."""
    MENU.clear()
    src = N(fr[0])
    if not src:
        return
    fk = out_kind(src, fr[1])
    opts = []
    if target:
        t = N(target)
        for i, nm in enumerate(ins_of(t)):
            if fits(fk, t["type"], i, t):
                opts.append((nm + ("" if i == 0 or t["type"] != "gen3d"
                                   else " view"), target, i))
        if len(opts) == 1:
            link(fr[0], fr[1], target, opts[0][2])
            return
    else:
        for k in PALETTE:
            ins = TYPES[k][2]
            if ins and fits(fk, k, 0):
                opts.append((TYPES[k][0], "new:" + k, 0))
    if opts:
        MENU.update(fr=fr, pos=(mx, my), opts=opts,
                    title=("Connect to " + TYPES[N(target)["type"]][0])
                    if target else "Add and connect")


def menu_pick(k):
    st = state()
    fr, (mx, my) = MENU["fr"], MENU["pos"]
    _, tgt, i = MENU["opts"][k]
    MENU.clear()
    if tgt.startswith("new:"):
        z, v = Z(), st["view"]
        t = add_node(tgt[4:], (mx - v[0]) / z + 10, (my - v[1]) / z + 30)
        tgt = t["id"]
        st["sel"], st["selset"] = tgt, [tgt]
    link(fr[0], fr[1], tgt, i)


def _rect(n, gap=0):
    w, h = nw(n), node_h(n)
    return (n["x"] - gap, n["y"] - h - gap, n["x"] + w + gap, n["y"] + gap)


def _clash(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def free_spot(n, others, gap=40):
    """Move n down (and, when a column is full, to the right) until it
    sits on its own with some room around it."""
    for _ in range(200):
        r = _rect(n, gap / 2)
        hit = [o for o in others if o is not n and _clash(r, _rect(o, gap / 2))]
        if not hit:
            return
        n["y"] = min(o["y"] - node_h(o) for o in hit) - gap


def add_to_selected(t):
    """With nodes selected, Add gives each of them its own new node of
    that type, connected and laid out so none overlap."""
    st = state()
    ins = TYPES[t][2]
    if not ins:
        return False
    src = []
    for nid in selected():
        n = N(nid)
        if not n:
            continue
        for i in range(len(TYPES[n["type"]][3])):
            if fits(out_kind(n, i), t, 0):
                src.append((n, i))
                break
    if not src:
        return False
    # top to bottom, each source with its new node beside it on one row
    src.sort(key=lambda p: (-p[0]["y"], p[0]["x"]))
    made = []
    for n, i in src:
        m = add_node(t, n["x"], n["y"])
        link(n["id"], i, m["id"], 0)
        made.append(m)
    lay_rows([n for n, _ in src], made)
    st["selset"] = [m["id"] for m in made]
    st["sel"] = made[-1]["id"]
    return True


def lay_rows(left, right, gap_x=90, gap_y=50):
    """Two tidy columns: every left node with its right node on the same
    row, rows stacked from the top down, nothing over anything else."""
    x0 = min(n["x"] for n in left)
    wl = max(nw(n) for n in left)
    x1 = x0 + wl + gap_x
    y = max(n["y"] for n in left)
    for a, b in zip(left, right):
        a["x"], a["y"] = x0, y
        b["x"], b["y"] = x1, y
        y -= max(node_h(a), node_h(b)) + gap_y
    # the sources stay where they are: only the new column steps right,
    # past anything already there (an Image node a source made, say)
    block = right
    ids = set(id(n) for n in left + right)
    others = [o for o in state()["nodes"].values() if id(o) not in ids]
    for _ in range(400):
        if not any(_clash(_rect(b_, 20), _rect(o, 20))
                   for b_ in block for o in others):
            return
        for b_ in block:
            b_["x"] += 40


def _menu(z):
    if not MENU:
        return
    d = dpi()
    mx, my = MENU["pos"]
    w = 210 * d
    rows = len(MENU["opts"])
    h = (34 + rows * 34 + 8) * d
    x, y = mx + 8 * d, my - h + 12 * d
    reg = _REG[0]
    if reg:
        x = min(x, reg.width - w - 8 * d)
        y = max(8 * d, y)
    ui.outline(x, y, w, h, 12 * d, (0, 0, 0, .5), 2)
    ui.grad(x, y, w, h, 12 * d, ui.CARD_BOT, ui.CARD_TOP)
    ui.hit(x, y, w, h, "menu")
    ui.fill(x + 12 * d, y + h - 22 * d, 7 * d, 7 * d, 3.5 * d, ui.ACCENT)
    ui.text(MENU["title"], x + 26 * d, y + h - 23 * d, 11 * d, ui.TEXT,
            bold=True, room=w - 34 * d)
    cy = y + h - 34 * d
    for k, (label, _t, _i) in enumerate(MENU["opts"]):
        cy -= 34 * d
        ui.button(x + 8 * d, cy + 3 * d, w - 16 * d, 28 * d, label,
                  "mpick:%d" % k, z=d, size=11.5)


# ---------------------------------------------------------------- helpers

def main_addon():
    for name, mod in list(sys.modules.items()):
        if name.endswith("image_to_3d") and hasattr(mod, "CARD_JOB") \
                and hasattr(mod, "engine_dir"):
            return mod
    return None


def engine_dir():
    m = main_addon()
    try:
        d = m.engine_dir() if m else ""
    except Exception:
        d = ""
    return d or ""


def vlm_root():
    """The engine folder when the Canvas picture models (Qwen) are
    installed in it: env_qwen, engine_qwen and models/."""
    d = engine_dir()
    if d and os.path.isfile(os.path.join(d, "env_qwen", "Scripts",
                                         "python.exe")) and \
            os.path.isdir(os.path.join(d, "models",
                                       "Qwen-Image-Edit-2509")):
        return d
    return ""


CV_NAMES = {"CV_QWEN_EDIT": "Qwen-Image-Edit", "CV_QWEN_EDIT_4BIT":
            "Qwen-Image-Edit 4-bit", "CV_QWEN_IMAGE": "Qwen-Image 4-bit",
            "CV_SAM3": "SAM 3", "CV_KLEIN": "FLUX.2 Klein",
            "CV_BIREFNET": "BiRefNet", "CV_ESRGAN": "Real-ESRGAN"}


def needs(n, *gids):
    """True when every model this node uses is downloaded and the Canvas
    tools are installed; otherwise the node says what to get."""
    m = main_addon()
    d = engine_dir()
    if m is None or not d:
        n["status"] = "Image to 3D 2 is not set up"
        return False
    if not m.software_ready("CANVAS"):
        n["status"] = "Install Canvas Tools on the Image to 3D 2 card"
        return False
    miss = [g for g in gids if not m.dl.ready(d, g)]
    if miss:
        n["status"] = "Download in Downloads > Canvas: " + ", ".join(
            CV_NAMES.get(g, g) for g in miss)
        return False
    return True


QWEN_CARD = ("The Canvas picture model needs an RTX 30 card or newer "
             "(this card can't run it)")


def qwen_card_ok(n):
    """Qwen-Image-Edit runs in bfloat16, which RTX 20 / GTX 16 cards don't
    have; in float16 its numbers overflow and the picture comes out black."""
    m = main_addon()
    try:
        if m is not None and m.dl.gpu_cc() < 8.0:
            n["status"] = QWEN_CARD
            return False
    except Exception:
        pass
    return True


def has_model(gid):
    m = main_addon()
    d = engine_dir()
    return bool(m and d and m.dl.ready(d, gid))


def work_dir():
    d = os.path.join(engine_dir(), "canvas", "canvas")
    os.makedirs(d, exist_ok=True)
    return d


def redraw():
    for w in bpy.context.window_manager.windows:
        for a in w.screen.areas:
            if a.type in ("NODE_EDITOR", "VIEW_3D"):
                a.tag_redraw()


def dpi():
    try:
        return bpy.context.preferences.system.ui_scale
    except Exception:
        return 1.0


class Proc:
    def __init__(self, cmd, log, stdin=False):
        self.q = queue.Queue()
        self.log = open(log, "w", encoding="utf-8")
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace",
                                  stdin=subprocess.PIPE if stdin else None,
                                  creationflags=NOWIN)
        threading.Thread(target=self._read, daemon=True).start()

    def alive(self):
        return self.p.poll() is None

    def send(self, obj):
        self.p.stdin.write((obj if isinstance(obj, str) else
                            json.dumps(obj)) + "\n")
        self.p.stdin.flush()

    def _read(self):
        for line in self.p.stdout:
            self.log.write(line)
            self.log.flush()
            self.q.put(line.rstrip("\n"))
        self.q.put(None)

    def lines(self):
        while True:
            try:
                yield self.q.get_nowait()
            except queue.Empty:
                return


# ------------------------------------------------------------------ graph

S = {}
JOB = {}


def tree():
    for t in bpy.data.node_groups:
        if t.bl_idname == TREE:
            return t
    t = bpy.data.node_groups.new("Image to 3D Canvas", TREE)
    t.use_fake_user = True
    return t


def state():
    if not S:
        try:
            d = json.loads(tree().get("cv2", ""))
        except Exception:
            d = {}
        if not d.get("nodes"):
            d = {"view": [200.0, 700.0, 1.0], "nodes": {}, "links": [],
                 "sel": None, "seq": 0, "queue": []}
        S.update(d)
    return S


def save():
    try:
        tree()["cv2"] = json.dumps(S)
    except Exception:
        pass
    redraw()


def N(nid):
    return state()["nodes"].get(nid)


def add_node(kind, x, y, **kw):
    st = state()
    st["seq"] += 1
    nid = "%s%d" % (kind, st["seq"])
    n = {"id": nid, "type": kind, "x": x, "y": y, "status": ""}
    if kind == "prompt":
        n.update(prompt="", ready=True, aspect="2x3", pmodel="image")
    elif kind == "upscale":
        n.update(scale="2", image="")
    elif kind == "upload":
        n.update(image="")
    elif kind == "image":
        n.update(image="", label="Image", meta=None)
    elif kind == "parts":
        n.update(found=False, items=[], body=True)
    elif kind == "gen3d":
        n.update(engine="TRELLIS", texture=True, frac=0.0, obj="",
                 quality="HIGH", faces="200000", texsize="2048", seed=0)
    elif kind == "detail":
        n.update(repaint=True, quality="HIGH", frac=0.0, obj="")
    elif kind == "etex":
        n.update(mode="HIGH", creativity="0.0", frac=0.0, obj="")
    n.update(kw)
    st["nodes"][nid] = n
    st["sel"] = nid
    st["selset"] = [nid]
    return n


def selected():
    st = state()
    ss = [i for i in st.get("selset", []) if i in st["nodes"]]
    if st.get("sel") in st["nodes"] and st["sel"] not in ss:
        ss.append(st["sel"])
    return ss


def delete_node(nid):
    st = state()
    st["nodes"].pop(nid, None)
    st["links"] = [l for l in st["links"]
                   if l["from"][0] != nid and l["to"][0] != nid]
    if st["sel"] == nid:
        st["sel"] = None
    st["selset"] = [i for i in st.get("selset", []) if i != nid]


def link(a, ai, b, bi):
    st = state()
    st["links"] = [l for l in st["links"] if l["to"] != [b, bi]]
    if a != b:
        st["links"].append({"from": [a, ai], "to": [b, bi]})


def source(nid, i=0):
    """The node feeding input i, or None."""
    for l in state()["links"]:
        if l["to"] == [nid, i]:
            return N(l["from"][0]), l["from"][1]
    return None, 0


def image_of(nid, i=0):
    """The picture arriving at input i of a node."""
    n, oi = source(nid, i)
    if n is None:
        return ""
    if n["type"] in ("image", "upload", "upscale"):
        return n.get("image", "")
    if n["type"] == "views":
        return (n.get("views") or {}).get(str(oi), "")
    return ""


def deliver(nid, oi, path, label=None, meta=None, dy=0.0):
    """Put a picture into the Image node(s) on output oi; make one if no
    Image node is connected there yet."""
    st = state()
    src = N(nid)
    targets = [N(l["to"][0]) for l in st["links"]
               if l["from"] == [nid, oi] and N(l["to"][0]) and
               N(l["to"][0])["type"] == "image"]
    if not targets:
        w = nw(src)
        t = add_node("image", src["x"] + w + 70, src["y"] - dy)
        link(nid, oi, t["id"], 0)
        targets = [t]
    for t in targets:
        t["image"] = path
        if label:
            t["label"] = label
        if meta is not None:
            t["meta"] = meta
        t["status"] = ""
    return targets


# ------------------------------------------------------------------ jobs

def busy():
    return bool(JOB)


def _py(env):
    return os.path.join(engine_dir(), env, "Scripts", "python.exe")


def run_prompt(n):
    # the picture comes from Qwen-Image-Edit (20B): it follows the sentence
    # far better than Klein
    gids = ["CV_QWEN_EDIT", "CV_QWEN_EDIT_4BIT"]
    if n.get("pmodel", "image") == "image":
        gids.append("CV_QWEN_IMAGE")
    if n.get("ready", True):
        gids.append("CV_BIREFNET")
    if not needs(n, *gids):
        return
    if not qwen_card_ok(n):
        return
    if n.get("ready", True) and not os.path.isfile(_py("env")):
        n["status"] = ("3D Ready needs the main 3D engine: install it on the "
                       "Image to 3D 2 card, or turn 3D Ready off")
        return
    stop_servers(keep=("qwen", "clean"))
    free_engine()
    out = os.path.join(work_dir(), "prompt_%d.png" % int(time.time()))
    kl = server("qwen")
    for _ in kl.lines():     # anything left from before
        pass
    ready = n.get("ready", True)
    kl.send({"t2i": {"text": n["prompt"], "out": out,
                     "seed": int(time.time()) % 10000, "preset": ready,
                     "aspect": n.get("aspect", "2x3"),
                     "model": n.get("pmodel", "image"),
                     "plate": os.path.join(HERE, "media", "plate.png")}})
    if ready:
        server("clean")      # BiRefNet loads while the picture is made
    JOB.update({"kind": "t2i", "node": n["id"], "out": out, "srv": True,
                "t": time.time(), "frac": 0.0, "proc": kl})
    n["status"] = "Loading the model..."
    _timer()


def run_upscale(n):
    """Real-ESRGAN x4 (the weights Image to 3D keeps for its textures):
    the picture 2 or 4 times bigger and sharper, in a few seconds."""
    src = image_of(n["id"])
    if not src:
        n["status"] = "Connect a picture first"
        return
    w_ = os.path.join(engine_dir(), "models", "realesrgan",
                      "RealESRGAN_x4plus.pth")
    if not os.path.isfile(w_):
        n["status"] = "Download in Downloads > Canvas: Real-ESRGAN"
        return
    stop_servers()
    free_engine()
    sc = n.get("scale", "2")
    out = os.path.join(work_dir(), "%s_x%s_%d.png" % (
        n["id"], sc, int(time.time())))
    proc = Proc([_py("env"), "-u", os.path.join(HERE, "workers",
                                               "upscale.py"),
                 src, out, sc, os.path.join(engine_dir(), "models")],
                os.path.join(work_dir(), "upscale.log"))
    JOB.update({"kind": "ups", "node": n["id"], "out": out,
                "t": time.time(), "frac": 0.0, "proc": proc})
    n["status"] = "Upscaling %sx..." % sc
    _timer()


# the vision model, SAM 3 and Klein, kept loaded between jobs; kept where an
# add-on reload cannot lose them, or the old ones would hold the card
SERVERS = bpy.app.driver_namespace.setdefault("img3d_canvas_servers", {})


def server(kind):
    sv = SERVERS.get(kind)
    if sv is not None and sv.alive():
        return sv
    if kind == "klein":
        cmd = [_py("env_klein"), "-u", os.path.join(HERE, "workers",
                                                   "parts_server.py"),
               engine_dir(), vlm_root() or ""]
    elif kind == "clean":
        cmd = [_py("env"), "-u", os.path.join(HERE, "workers",
                                             "clean_bg.py"), "--serve",
               os.path.join(engine_dir(), "models")]
    elif kind == "qwen":
        root = vlm_root()
        cmd = [os.path.join(root, "env_qwen", "Scripts", "python.exe"),
               "-u", os.path.join(HERE, "workers", "views_qwen.py"), root]
    elif kind == "vlm":
        root = vlm_root()
        cmd = [os.path.join(root, "env_qwen", "Scripts", "python.exe"),
               "-u", os.path.join(HERE, "workers", "find_parts.py"),
               "--serve",
               os.path.join(root, "models")]
    else:
        cmd = [_py("env"), "-u", os.path.join(HERE, "workers",
                                             "segment_parts.py"),
               "--serve", os.path.join(engine_dir(), "models")]
    sv = SERVERS[kind] = Proc(cmd, os.path.join(work_dir(), kind + ".log"),
                              stdin=True)
    return sv


def free_engine():
    """The main add-on keeps its 3D engine loaded after a model; that holds
    several GB of the card, and Klein then crawls. Park it in main memory
    (or close it when it cannot wait there); the next model loads it."""
    m = main_addon()
    try:
        if m and m._server_alive():
            if m._SRV.get("engine") in m._PARKS and m._park_current():
                m._SRV.update(p=None, q=None)
            else:
                m.server_stop(keep_pool=True)
    except Exception as exc:
        print("[Canvas] engine not freed:", exc)


def stop_servers(keep=()):
    """Free the graphics card for Klein and the 3D engines. The part maker
    only steps aside into main memory: it is back in a second or two."""
    for k, sv in list(SERVERS.items()):
        if k in keep:
            continue
        try:
            if sv.alive():
                sv.send("PARK" if k in ("klein", "qwen") else "QUIT")
                if k in ("klein", "qwen"):
                    continue
        except Exception:
            pass
        SERVERS.pop(k, None)


def run_find(n):
    img = image_of(n["id"])
    root = vlm_root()
    if not img:
        n["status"] = "Connect a picture first"
        return
    if not needs(n, "CV_QWEN_EDIT"):
        return
    out = os.path.join(work_dir(), "parts_%s.json" % n["id"])
    stop_servers(keep=("vlm", "sam"))
    free_engine()
    vlm = server("vlm")
    vlm.send({"image": img, "out": out})
    sam_ok = has_model("CV_SAM3")
    if sam_ok:
        # SAM 3 looks at the picture and runs its checklist meanwhile
        server("sam").send({"prep": img})
    JOB.update({"kind": "find", "node": n["id"], "out": out,
                "t": time.time(), "proc": vlm, "img": img,
                "sam": sam_ok, "sam_pending": 1 if sam_ok else 0})
    n["status"] = "Loading the vision model..."
    n["items"] = []
    n["found"] = False
    n.pop("body_desc", None)
    JOB["frac"] = 0.02
    _timer()


def run_extract(owner, plan, src, seed=0):
    stop_servers(keep=("klein",))
    free_engine()
    """plan: [{"name","description","bbox","kind","out","target"}]"""
    for p in plan:
        if os.path.isfile(p["out"]):
            try:
                os.remove(p["out"])
            except Exception:
                pass
    jp = os.path.join(work_dir(), "extract.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump({"image": src, "parts": plan, "seed": seed}, fh, indent=1)
    kl = server("klein")
    for _ in kl.lines():     # anything left from before
        pass
    kl.send({"jobs": jp})
    JOB.update({"kind": "extract", "node": owner, "plan": plan,
                "t": time.time(), "proc": kl})
    _timer()


def run_start(n):
    """One Image node per checked part, on the Parts output."""
    if not needs(n, "CV_KLEIN"):
        return
    src = image_of(n["id"])
    items = [dict(i) for i in n["items"] if i["on"]]
    if n.get("body"):
        items.append({"name": "Base Body", "description":
                      n.get("body_desc", ""), "bbox": None, "body": True})
    hair = any("hair" in i["name"].lower() for i in items)
    st = state()
    # the old part pictures of this node go
    for l in [l for l in st["links"] if l["from"] == [n["id"], 0]]:
        t = N(l["to"][0])
        if t and t["type"] == "image" and t.get("meta"):
            delete_node(t["id"])
    plan = []
    w = nw(n)
    for k, it in enumerate(items):
        kind = "body" if it.get("body") else (
            "head_bald" if hair and it["name"].lower().strip() in
            ("head", "face", "head and face") else "part")
        t = add_node("image", n["x"] + w + 70 + (k % 3) * 300,
                     n["y"] - (k // 3) * 420, label=it["name"])
        t["status"] = "Waiting"
        link(n["id"], 0, t["id"], 0)
        base = bpy.path.clean_name(it["name"]) or "Part"
        meta = {"name": it["name"], "description":
                it.get("description", ""), "bbox": it.get("bbox"),
                "kind": kind, "src": src, "mirror": bool(it.get("mirror")),
                "mask": it.get("mask"), "iso": bool(n.get("iso", True))}
        t["meta"] = meta
        plan.append(dict(meta, out=os.path.join(
            work_dir(), "%s_%s.png" % (t["id"], base)), target=t["id"]))
    st["sel"] = n["id"]
    n["status"] = "Making the part pictures..."
    run_extract(n["id"], plan, src)


def run_views(n):
    src = image_of(n["id"])
    if not src:
        n["status"] = "Connect a picture first"
        return
    n["views"] = {"0": src}
    targets = {}
    for oi, side in ((1, "back"), (2, "left"), (3, "right")):
        ts = [l["to"][0] for l in state()["links"]
              if l["from"] == [n["id"], oi]]
        if not ts:
            w = nw(n)
            t = add_node("image", n["x"] + w + 70,
                         n["y"] - (oi - 1) * 420, label=side.title())
            link(n["id"], oi, t["id"], 0)
            ts = [t["id"]]
        for tid in ts:
            N(tid)["status"] = "Waiting"
        targets[oi] = ts
    plan = [{"name": side, "description": "", "bbox": None,
             "kind": "view_" + side, "src": src, "out": os.path.join(
                 work_dir(), "%s_%s.png" % (n["id"], side)),
             "target": targets[oi][0], "views": [n["id"], oi]}
            for oi, side in ((1, "back"), (2, "left"), (3, "right"))]
    # the sides are drawn by Qwen-Image-Edit from the picture alone (no 3D
    # model): it understands the turn around the figure, so what hangs on
    # the back and which way each side faces come out right
    if not needs(n, "CV_QWEN_EDIT", "CV_QWEN_EDIT_4BIT") or \
            not qwen_card_ok(n):
        return
    stop_servers(keep=("qwen",))
    free_engine()
    for p in plan:
        if os.path.isfile(p["out"]):
            try:
                os.remove(p["out"])
            except Exception:
                pass
    jp = os.path.join(work_dir(), "views.json")
    with open(jp, "w", encoding="utf-8") as fh:
        json.dump({"image": src, "parts": plan, "seed": 0}, fh, indent=1)
    qw = server("qwen")
    for _ in qw.lines():
        pass
    qw.send({"views": jp})
    JOB.update({"kind": "extract", "node": n["id"], "plan": plan,
                "t": time.time(), "proc": qw, "frac": 0.0, "fbase": 0.0,
                "done": 0, "half": 0})
    for p in plan:
        t = N(p["target"])
        if t:
            t["status"] = "Waiting"
    t = N(plan[0]["target"])
    if t:
        t["status"] = "Making the picture..."
    n["status"] = "Making the views..."
    _timer()


def render_views(objs, plan):
    """Photograph the model from behind and from both sides, on the studio
    gray, framed like the front picture."""
    from mathutils import Vector
    import math
    meshes = [o for o in objs if o.type == "MESH"]
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts),
                 min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts),
                 max(p.z for p in pts)))
    mid = (lo + hi) / 2
    size = hi - lo
    sc = bpy.data.scenes.new("img3d_views")
    try:
        for o in objs:
            sc.collection.objects.link(o)
        cam = bpy.data.objects.new("vcam", bpy.data.cameras.new("vcam"))
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.data.type = "ORTHO"
        sc.render.resolution_x = 1024
        sc.render.resolution_y = 1536
        try:
            sc.render.engine = "BLENDER_EEVEE"
        except Exception:
            sc.render.engine = "BLENDER_EEVEE_NEXT"
        sc.view_settings.view_transform = "Standard"
        w = bpy.data.worlds.new("vworld")
        sc.world = w
        w.use_nodes = True
        bg = w.node_tree.nodes.get("Background")
        bg.inputs[0].default_value = (0.42, 0.42, 0.43, 1.0)
        bg.inputs[1].default_value = 1.6
        sun = bpy.data.objects.new("vsun", bpy.data.lights.new(
            "vsun", "SUN"))
        sun.data.energy = 1.2
        sc.collection.objects.link(sun)
        dist = max(size) * 4 + 1
        for p in plan:
            side = p["kind"][5:]
            # TRELLIS models face -Y: the back is seen from +Y, the
            # character's left side from +X
            d = {"back": Vector((0, 1, 0)), "left": Vector((1, 0, 0)),
                 "right": Vector((-1, 0, 0))}[side]
            cam.location = mid + d * dist
            cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
            sun.rotation_euler = cam.rotation_euler
            span_w = size.x if side == "back" else size.y
            # the character fills the frame like in the front picture
            cam.data.ortho_scale = max(size.z * 1.12,
                                       span_w * 1.15 * 1536 / 1024)
            sc.render.filepath = p["out"]
            bpy.ops.render.render(write_still=True, scene=sc.name)
    finally:
        for o in list(sc.collection.objects):
            if o.name.startswith(("vcam", "vsun")):
                bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.scenes.remove(sc)


def render_preview(objs, path):
    """A small picture of the finished model, three-quarter from the front,
    for its node: the result is seen without leaving the Canvas."""
    from mathutils import Vector
    meshes = [o for o in objs if o.type == "MESH"]
    if not meshes:
        return ""
    bpy.context.view_layer.update()
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts),
                 min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts),
                 max(p.z for p in pts)))
    mid = (lo + hi) / 2
    size = hi - lo
    sc = bpy.data.scenes.new("img3d_preview")
    try:
        for o in meshes:
            sc.collection.objects.link(o)
        cam = bpy.data.objects.new("pcam", bpy.data.cameras.new("pcam"))
        sc.collection.objects.link(cam)
        sc.camera = cam
        cam.data.type = "ORTHO"
        sc.render.resolution_x = sc.render.resolution_y = 512
        try:
            sc.render.engine = "BLENDER_EEVEE"
        except Exception:
            sc.render.engine = "BLENDER_EEVEE_NEXT"
        sc.view_settings.view_transform = "Standard"
        w = bpy.data.worlds.new("pworld")
        sc.world = w
        w.use_nodes = True
        bg = w.node_tree.nodes.get("Background")
        # a dark backdrop, little fill and one strong key light from the
        # upper left: a plain grey model keeps its shape instead of
        # dissolving into a white blur
        bg.inputs[0].default_value = (0.20, 0.20, 0.21, 1.0)
        bg.inputs[1].default_value = 0.5
        sun = bpy.data.objects.new("psun", bpy.data.lights.new(
            "psun", "SUN"))
        sun.data.energy = 3.2
        sc.collection.objects.link(sun)
        # the models face -Y: a three-quarter view from the front, a bit
        # from above
        d = Vector((0.65, -1.0, 0.45)).normalized()
        cam.location = mid + d * (max(size) * 4 + 1)
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        ld = Vector((-0.5, -0.8, 1.0)).normalized()
        sun.rotation_euler = (-ld).to_track_quat("-Z", "Y").to_euler()
        cam.data.ortho_scale = size.length * 1.05
        sc.render.filepath = path
        bpy.ops.render.render(write_still=True, scene=sc.name)
    finally:
        for o in list(sc.collection.objects):
            if o.name.startswith(("pcam", "psun")):
                bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.scenes.remove(sc)
    return path if os.path.isfile(path) else ""


def model_objects(name):
    """The object named on a node with everything under it."""
    o = bpy.data.objects.get(name or "")
    if o is None:
        return []
    return [o] + list(o.children_recursive)


def preview_of(n):
    try:
        objs = model_objects(n.get("obj"))
        path = os.path.join(work_dir(), "%s_model_%d.png" % (
            n["id"], int(time.time())))
        n["preview"] = render_preview(objs, path)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("[Canvas] no model picture:", exc)


def run_gen(n):
    stop_servers()
    m = main_addon()
    sc = bpy.context.scene
    s3 = getattr(sc, "img3d", None)
    img = image_of(n["id"], 0)
    if not img:
        n["status"] = "Connect a picture first"
        return False
    if m is None or s3 is None:
        n["status"] = "Image to 3D is not enabled"
        return False
    if m.CARD_JOB is not None:
        n["status"] = "Another model is being made"
        return False
    s3.engine = n["engine"]
    s3.use_texture = "ON" if n["texture"] else "OFF"
    s3.image_path = img
    for prop, val in (("quality", n.get("quality", "HIGH")),
                      ("texture", n.get("texsize", "2048")),
                      ("seed", int(n.get("seed", 0))),
                      ("auto_detail", "OFF"), ("auto_enhance", "OFF")):
        try:
            setattr(s3, prop, val)
        except Exception:
            pass
    try:
        if n["engine"] == "HUNYUAN":
            s3.faces_hy = int(n.get("faces", 40000))
        else:
            s3.faces = int(n.get("faces", 200000))
    except Exception:
        pass
    for k, prop in ((1, "view_1"), (2, "view_2"), (3, "view_3")):
        try:
            setattr(s3, prop, image_of(n["id"], k) if
                    n["engine"] == "HUNYUAN" else "")
        except Exception:
            pass
    before = set(o.name for o in bpy.data.objects)
    try:
        r = bpy.ops.img3d.generate()
    except Exception as exc:
        n["status"] = "Could not start: %s" % str(exc)[:50]
        return False
    if "FINISHED" not in r:
        n["status"] = "Could not start (engine installed?)"
        return False
    src, _ = source(n["id"], 0)
    JOB.update({"kind": "gen", "node": n["id"], "before": list(before),
                "mirror": bool(((src or {}).get("meta") or {}).get(
                    "mirror")),
                "label": (src or {}).get("label") or "Model",
                "t": time.time()})
    n["status"] = "Starting"
    n["frac"] = 0.0
    _timer()
    return True


def mirror_twin(root, new):
    """The other piece of a mirrored pair: a mirror copy beside the first,
    both under one empty so they move as one part."""
    from mathutils import Vector
    bpy.context.view_layer.update()
    meshes = [o for o in new if o.type == "MESH"]
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    x0 = min(p.x for p in pts)
    x1 = max(p.x for p in pts)
    width = x1 - x0
    coll = root.users_collection[0] if root.users_collection else \
        bpy.context.scene.collection
    twin_objs = {}
    for o in [root] + list(root.children_recursive):
        c = o.copy()
        if o.data is not None:
            c.data = o.data          # same mesh, no extra memory
        coll.objects.link(c)
        twin_objs[o] = c
    for o, c in twin_objs.items():
        if o.parent in twin_objs:
            c.parent = twin_objs[o.parent]
    twin = twin_objs[root]
    base = root.name
    root.name = base + "_L"
    twin.name = base + "_R"
    # mirror across the pair's middle, then step it over beside the first
    mid = (x0 + x1) / 2.0
    twin.scale.x *= -1.0
    twin.location.x = 2 * mid - root.location.x + width + 0.06
    holder = bpy.data.objects.new(base, None)
    coll.objects.link(holder)
    holder.location = root.location.copy()
    for o in (root, twin):
        mw = o.matrix_world.copy()
        o.parent = holder
        o.matrix_parent_inverse = holder.matrix_world.inverted()
        o.matrix_world = mw
    bpy.context.view_layer.update()
    allnew = list(new) + list(twin_objs.values()) + [holder]
    return allnew, [holder]


def model_object(name):
    """The mesh to enhance for a model made here (the first of a pair)."""
    o = bpy.data.objects.get(name or "")
    if o is None:
        return None
    if o.type == "MESH":
        return o
    for c in o.children_recursive:
        if c.type == "MESH":
            return c
    return None


def run_enh(n):
    stop_servers()
    m = main_addon()
    s3 = getattr(bpy.context.scene, "img3d", None)
    src, _ = source(n["id"], 0)
    o = model_object((src or {}).get("obj"))
    if o is None:
        n["status"] = "Connect a model first"
        return False
    if m is None or s3 is None or m.CARD_JOB is not None:
        n["status"] = "Another model is being made"
        return False
    for ob in bpy.context.view_layer.objects:
        ob.select_set(False)
    o.select_set(True)
    bpy.context.view_layer.objects.active = o
    before = set(x.name for x in bpy.data.objects)
    try:
        if n["type"] == "detail":
            s3.quality = n.get("quality", "HIGH")
            s3.detail_tex = "ON" if n.get("repaint", True) else "OFF"
            r = bpy.ops.img3d.enhance_detail(
                no_tex=not n.get("repaint", True))
        else:
            s3.enhance_mode = n.get("mode", "HIGH")
            s3.creativity = float(n.get("creativity", "0.0"))
            r = bpy.ops.img3d.enhance()
    except Exception as exc:
        n["status"] = "Could not start: %s" % str(exc)[:50]
        return False
    if m.CARD_JOB is None:
        n["status"] = (s3.status or "Could not start")[:60]
        return False
    JOB.update({"kind": "enh", "node": n["id"], "before": list(before),
                "src_obj": o.name, "t": time.time()})
    n["status"] = "Starting"
    n["frac"] = 0.0
    _timer()
    return True


def cancel_job(nid):
    """Stop the node's work now. A model server is closed (the next job
    loads it again); the 3D engines stop through the main add-on."""
    st = state()
    q = st.get("queue") or []
    if nid in q and JOB.get("node") != nid:
        q.remove(nid)
        n = N(nid)
        if n:
            n["status"] = "Cancelled"
        return
    if JOB.get("node") != nid:
        return
    k = JOB.get("kind")
    if k in ("gen", "enh"):
        try:
            bpy.ops.img3d.cancel()
        except Exception as exc:
            print("[Canvas] cancel:", exc)
    procs = [JOB.get("proc")]
    for key in ("sam",):
        if JOB.get(key):
            procs.append(SERVERS.get("sam"))
    for pr in procs:
        if pr is None:
            continue
        for kk, sv in list(SERVERS.items()):
            if sv is pr:
                SERVERS.pop(kk, None)
        try:
            if pr.alive():
                subprocess.run(["taskkill", "/PID", str(pr.p.pid), "/T",
                                "/F"], capture_output=True,
                               creationflags=NOWIN) if os.name == "nt" \
                    else pr.p.kill()
        except Exception:
            try:
                pr.p.kill()
            except Exception:
                pass
    n = N(nid)
    if n:
        n["status"] = "Cancelled"
        n["frac"] = 0.0
    JOB.clear()
    if AUTO["on"]:
        auto_stop("Auto Run cancelled")


def run_button(x, y, w, h, label, action, nid, z, enabled=True):
    """The node's main button; while that node works it is Cancel."""
    queued = nid in (state().get("queue") or [])
    if JOB.get("node") == nid or queued:
        ui.button(x, y, w, h, "Cancel", "cancel:" + nid, z=z, size=13)
    else:
        ui.button(x, y, w, h, label, action, kind="accent", z=z, size=13,
                  enabled=enabled)


# ------------------------------------------------------------------ auto run

AUTO = {"on": False, "order": [], "i": 0, "cur": None, "phase": "",
        "sig": None, "msg": ""}
RUNNABLE = ("prompt", "upscale", "parts", "views", "gen3d", "detail",
            "etex")
FAILS = ("cancel", "not installed", "connect", "could not", "fail",
         "error", "another model", "stopped", "not enabled",
         "not downloaded")


def _topo():
    """Every node that makes something, in the order the wires ask for."""
    st = state()
    ids = list(st["nodes"])
    deps = {i: set() for i in ids}
    for l in st["links"]:
        a, b = l["from"][0], l["to"][0]
        if a in deps and b in deps and a != b:
            deps[b].add(a)
    order, done = [], set()
    while len(done) < len(ids):
        ready = sorted([i for i in ids if i not in done and deps[i] <= done],
                       key=lambda i: (N(i)["y"] * -1, N(i)["x"]))
        if not ready:          # a loop: take the rest as they are
            ready = [i for i in ids if i not in done]
        for i in ready:
            done.add(i)
            order.append(i)
    return [i for i in order if N(i)["type"] in RUNNABLE]


def _sig(n):
    """What the node has made so far, to tell a new result from the old."""
    t = n["type"]
    if t == "prompt":
        return tuple(N(l["to"][0]).get("image", "") for l in state()["links"]
                     if l["from"][0] == n["id"] and N(l["to"][0]))
    if t == "upscale":
        return n.get("image", "")
    if t == "views":
        return json.dumps(n.get("views") or {}, sort_keys=True)
    if t == "parts":
        return (n.get("found"), tuple(
            N(l["to"][0]).get("image", "") for l in state()["links"]
            if l["from"][0] == n["id"] and N(l["to"][0])))
    return n.get("obj", "") + "|" + str(n.get("preview", ""))


def _has_input(n):
    if n["type"] == "prompt":
        return bool(n.get("prompt", "").strip())
    if n["type"] in ("detail", "etex"):
        src, _ = source(n["id"], 0)
        return bool(src)
    return bool(image_of(n["id"]) or source(n["id"], 0)[0])


def _auto_start(n):
    t = n["type"]
    AUTO["sig"] = _sig(n)
    if t == "prompt":
        run_prompt(n)
    elif t == "upscale":
        run_upscale(n)
    elif t == "views":
        run_views(n)
    elif t == "parts":
        AUTO["phase"] = "find"
        run_find(n)
    elif t == "gen3d":
        run_gen(n)
    else:
        run_enh(n)


def auto_stop(msg="Auto Run stopped"):
    AUTO.update(on=False, cur=None, phase="", msg=msg)


def chain_from(root):
    """The root and every node its wires lead to."""
    seen, todo = {root}, [root]
    links = state()["links"]
    while todo:
        cur = todo.pop()
        for l in links:
            if l["from"][0] == cur and l["to"][0] not in seen:
                seen.add(l["to"][0])
                todo.append(l["to"][0])
    return seen


def chain_roots():
    """Nodes that start a chain: nothing feeds them, and something after
    them (or they themselves) makes a result."""
    st = state()
    fed = set(l["to"][0] for l in st["links"])
    out = []
    for nid, n in st["nodes"].items():
        if nid in fed:
            continue
        ch = chain_from(nid)
        if any(N(i) and N(i)["type"] in RUNNABLE for i in ch) and \
                (len(ch) > 1 or n["type"] == "prompt"):
            out.append(nid)
    return out


def auto_begin(root=""):
    if AUTO["on"]:
        auto_stop()
        return
    if busy():
        AUTO.update(msg="Wait for the running node to finish", root=root)
        return
    keep = chain_from(root) if root else None
    order = [i for i in _topo() if keep is None or i in keep]
    AUTO["root"] = root
    if not order:
        AUTO["msg"] = "Add nodes that make something first"
        return
    AUTO.update(on=True, order=order, i=0, cur=None, phase="", msg="")
    # not _timer(): that marks a job as running, and none is yet
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.3)


def auto_step():
    """Called by the timer when nothing is running: check the node that
    just ended, then start the next one."""
    if not AUTO["on"] or JOB:
        return
    cur = N(AUTO["cur"]) if AUTO["cur"] else None
    if cur is not None:
        stt = (cur.get("status") or "").lower()
        if cur["type"] == "parts" and AUTO["phase"] == "find":
            if cur.get("found") and cur.get("items") and \
                    not any(f in stt for f in FAILS):
                AUTO["phase"] = "parts"
                run_start(cur)
                return
            auto_stop("Auto Run stopped at Find Parts")
            return
        if any(f in stt for f in FAILS) or _sig(cur) == AUTO["sig"]:
            auto_stop("Auto Run stopped at %s" % TYPES[cur["type"]][0])
            return
        AUTO["cur"] = None
    while AUTO["i"] < len(AUTO["order"]):
        n = N(AUTO["order"][AUTO["i"]])
        AUTO["i"] += 1
        if n is None or not _has_input(n):
            continue
        AUTO.update(cur=n["id"], phase="")
        state()["sel"] = n["id"]
        _auto_start(n)
        if not JOB:
            # it could not start: the node says why
            auto_stop("Auto Run stopped at %s" % TYPES[n["type"]][0])
        return
    k = AUTO["i"]
    auto_stop("Auto Run finished")


def place(n, roots, new):
    """Stand the new model to the right of the ones made here before."""
    from mathutils import Vector
    meshes = [o for o in new if o.type == "MESH"]
    if not meshes:
        return
    bpy.context.view_layer.update()

    def bounds(objs):
        pts = [o.matrix_world @ Vector(c) for o in objs for c in o.bound_box]
        return (min(p.x for p in pts), max(p.x for p in pts),
                min(p.z for p in pts))
    others = []
    for m_ in state()["nodes"].values():
        if m_["id"] != n["id"] and m_.get("obj"):
            o = bpy.data.objects.get(m_["obj"])
            if o is not None:
                others += [c for c in [o] + list(o.children_recursive)
                           if c.type == "MESH"]
    x0, x1, z0 = bounds(meshes)
    right = bounds(others)[1] + 0.15 if others else x0
    for r in roots:
        r.location.x += right - x0
        r.location.z -= z0
    bpy.context.view_layer.update()


def clock(nid, y, z, x=None, right=None):
    """How long the running job of this node has taken, m:ss."""
    if JOB.get("node") != nid or "t0" not in JOB:
        return
    sec = int(time.time() - JOB["t0"])
    ui.text("%d:%02d" % (sec // 60, sec % 60), x or 0, y, 10.5 * z,
            ui.TEXT, right=right)


def _timer():
    JOB.setdefault("t0", time.time())
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.3)


IDLE_FREE = 20.0      # seconds without work before the models let go
_BUSY = [0.0]


def _idle_free():
    """Klein and Qwen stay loaded only while work keeps coming: once the
    Canvas has been idle a short while they close, and the card and main
    memory are the user's again. The next job loads them anew."""
    try:
        if JOB or state().get("queue") or AUTO["on"]:
            _BUSY[0] = time.time()
        elif SERVERS and time.time() - _BUSY[0] > IDLE_FREE:
            for k, sv in list(SERVERS.items()):
                try:
                    if sv.alive():
                        sv.send("QUIT")
                except Exception:
                    pass
                SERVERS.pop(k, None)
            print("[Canvas] idle: models closed, card freed")
    except Exception as exc:
        print("[Canvas] idle check:", exc)
    return 5.0


def _tick():
    _BUSY[0] = time.time()
    if JOB:
        JOB.setdefault("t0", time.time())
    try:
        _step()
        auto_step()
    except Exception:
        import traceback
        traceback.print_exc()
        AUTO["err"] = traceback.format_exc()[-1500:]
        JOB.clear()
        auto_stop("Auto Run stopped")
    redraw()
    return 0.4 if (JOB or state().get("queue") or AUTO["on"]) else None


def _step():
    st = state()
    if not JOB:
        q = st.get("queue") or []
        while q:
            n = N(q.pop(0))
            if n and run_gen(n):
                break
        save()
        return
    n = N(JOB.get("node", ""))
    k = JOB["kind"]
    if k == "ups":
        for line in JOB["proc"].lines():
            if line is None or line.startswith("ERROR"):
                if n:
                    n["status"] = (line or "ERROR stopped")[6:]
                JOB.clear()
                save()
                return
            if line.startswith("STEP"):
                JOB["frac"] = float(line.split()[1])
            if line.startswith("LOG ") and " in " in line:
                JOB["size"] = line.split()[1].replace("x", " x ")
            if line.startswith("DONE"):
                path = line[5:].strip()
                if n:
                    n["image"] = path
                    deliver(n["id"], 0, path, "Upscaled")
                    n["status"] = "Ready %s in %.0f s" % (
                        JOB.get("size", ""), time.time() - JOB["t"])
                JOB.clear()
                save()
                return
        return
    if k in ("t2i", "clean"):
        for line in JOB["proc"].lines():
            if line is None or line.startswith("ERROR"):
                if k == "clean" and n:
                    deliver(n["id"], 0, JOB["raw"], "Picture")
                if n:
                    n["status"] = (line or "ERROR stopped")[6:]
                JOB.clear()
                save()
                return
            if line.startswith("STEP"):
                JOB["frac"] = float(line.split()[1])
                JOB["drawing"] = True
            if line.startswith("DONE"):
                t0 = JOB["t"]
                if JOB.get("srv") and k == "t2i" and n and \
                        n.get("ready", True):
                    # onto the studio backdrop, by the BiRefNet that waits
                    raw = JOB["out"]
                    clean = raw[:-4] + "_ready.png"
                    cs = server("clean")
                    for _ in cs.lines():
                        pass
                    cs.send({"in": raw, "out": clean, "plate": os.path.join(
                        HERE, "media", "plate.png")})
                    JOB.clear()
                    JOB.update({"kind": "clean", "node": n["id"],
                                "out": clean, "raw": raw, "t": t0,
                                "proc": cs, "srv": True})
                    n["status"] = "Cleaning the background..."
                    return
                if JOB.get("srv") and k == "clean":
                    if n:
                        deliver(n["id"], 0, JOB["out"], "Picture")
                        n["status"] = "Picture ready in %.0f s" % (
                            time.time() - t0)
                    JOB.clear()
                    save()
                    return
                if JOB.get("srv"):
                    if n:
                        deliver(n["id"], 0, line[5:].strip(), "Picture")
                        n["status"] = "Picture ready in %.0f s" % (
                            time.time() - t0)
                    JOB.clear()
                    save()
                    return
                if k == "t2i" and n and n.get("ready", True):
                    raw = JOB["out"]
                    clean = raw[:-4] + "_ready.png"
                    JOB.clear()
                    JOB.update({"kind": "clean", "node": n["id"],
                                "out": clean, "raw": raw, "t": t0,
                                "proc": Proc([
                                    _py("env"), "-u",
                                    os.path.join(HERE, "workers",
                                                 "clean_bg.py"),
                                    raw, clean, os.path.join(
                                        HERE, "media", "plate.png"),
                                    os.path.join(engine_dir(), "models")],
                                    os.path.join(work_dir(),
                                                 "clean.log"))})
                    n["status"] = "Cleaning the background..."
                    return
                if n:
                    deliver(n["id"], 0, JOB["out"], "Picture")
                    n["status"] = "Picture ready in %.0f s" % (
                        time.time() - t0)
                JOB.clear()
                save()
                return
        if n and k == "t2i":
            n["status"] = ("Making the picture... %.0f s"
                           if JOB.get("drawing") else
                           "Loading the model... %.0f s") % (
                time.time() - JOB["t"])
    elif k in ("find", "seg"):
        if k == "find" and JOB.get("sam"):
            for line in server("sam").lines():
                if line == "JOBEND":
                    JOB["sam_pending"] -= 1
        for line in JOB["proc"].lines():
            if line and line.startswith("DONE"):
                continue        # a server's job ends at its JOBEND
            if line == "JOBEND":
                if k == "find":
                    line = "DONE " + JOB["out"]
                else:
                    if JOB.get("sam_pending", 0) > 0:
                        JOB["sam_pending"] -= 1
                        continue
                    line = "DONE"
            if line and line.startswith("PROGRESS "):
                _, f_, stage = (line.split(" ", 2) + [""])[:3]
                f_ = float(f_)
                JOB["frac"] = f_ * 0.7 if k == "find" else 0.7 + f_ * 0.3
                if n:
                    n["status"] = stage
                continue
            if line and line.startswith("ITEMS ") and n:
                try:
                    lst = json.loads(line[6:])
                except Exception:
                    continue
                old = {i["name"].lower(): i for i in n.get("items", [])}
                n["items"] = [{"name": f["name"], "description":
                               f.get("description", ""), "bbox":
                               f.get("bbox") or old.get(
                                   f["name"].lower(), {}).get("bbox"),
                               "mirror": bool(f.get("mirror")),
                               "found": f.get("found", True),
                               "on": f.get("found", True)} for f in lst]
                n["found"] = True
                continue
            if line and line.startswith("BODYDESC ") and n:
                n["body_desc"] = line[9:].strip()
                continue
            if line and line.startswith("SEG ") and n:
                _, i_, sc_ = line.split()
                i_ = int(i_)
                if i_ < len(n.get("items", [])) and float(sc_) == 0:
                    n["items"][i_]["found"] = False
                    n["items"][i_]["on"] = False
                continue
            if line is None or line.startswith("ERROR"):
                if n:
                    n["status"] = (line or "ERROR the AI stopped")[6:]
                JOB.clear()
                save()
                return
            if line.startswith("DONE") and k == "find" and JOB.get("sam"):
                # SAM 3 checks every part and cuts its exact pixels out
                outp, t0, img = JOB["out"], JOB["t"], JOB["img"]
                pend = JOB.get("sam_pending", 0)
                JOB.clear()
                sam = server("sam")
                sam.send({"parts": outp, "image": img, "dir": os.path.join(
                    work_dir(), "masks_" + n["id"])})
                JOB.update({"kind": "seg", "node": n["id"], "out": outp,
                            "t": t0, "proc": sam, "sam_pending": pend})
                n["status"] = "Cutting out the parts..."
                return
            if line == "DONE" and k == "seg":
                line = "DONE " + JOB["out"]
            if line.startswith("DONE") and k == "seg" and \
                    JOB.get("sam_pending", 0) > 0:
                continue
            if line.startswith("DONE"):
                found = json.load(open(JOB["out"], encoding="utf-8"))
                if n:
                    n["items"] = [{"name": f["name"], "description":
                                   f["description"], "bbox": f["bbox"],
                                   "mirror": bool(f.get("mirror")),
                                   "mask": f.get("mask"),
                                   "found": f.get("found", True),
                                   "on": f.get("found", True)}
                                  for f in found]
                    n["found"] = True
                    n["status"] = "%d parts found. Check them, then " \
                                  "Start" % len(found)
                JOB.clear()
                save()
                return

    elif k == "extract":
        plan = JOB["plan"]
        for line in JOB["proc"].lines():
            if line is None or line.startswith(("ALLPARTS", "ERROR")):
                for p in plan:
                    t = N(p["target"])
                    if t and not os.path.isfile(p["out"]):
                        t["status"] = "No picture"
                if n and not (n["type"] == "image" and line == "ALLPARTS"):
                    n["status"] = ("Views ready" if plan and plan[0].get(
                        "views") else "Pictures ready") \
                        if line == "ALLPARTS" else (line or "stopped")[6:]
                JOB.clear()
                save()
                return
            if line.startswith("STATE "):
                _, i_, st_ = line.split()
                t = N(plan[int(i_)]["target"])
                if t:
                    t["status"] = {"checking": "Checking the picture...",
                                   "ok": ("Checked" if t.get("status", "")
                                          .startswith(("Checking", "Not"))
                                          else ""),
                                   "redo": "Not clean, making it again...",
                                   "failed": "Could not make a clean picture"
                                   }.get(st_, "")
                continue
            if line.startswith("PROGRESS ") and "half" in JOB:
                if n and not JOB.get("done") and JOB.get("frac", 0) == 0:
                    n["status"] = "Loading Qwen (first time ~1 min)..."
                continue
            if line.startswith("STEP ") and "half" in JOB:
                if n and n["status"].startswith("Loading"):
                    n["status"] = "Making the views..."

                _, i_, f_ = line.split()
                JOB["frac"] = (int(i_) + float(f_)) / float(max(1, len(plan)))
                continue
            if line.startswith("PREVIEW "):
                _, i_, pv_ = line.split(" ", 2)
                p = plan[int(i_)]
                t = N(p["target"])
                if t:
                    t["image"] = pv_
                    t["status"] = "Cleaning up..."
                JOB["half"] = JOB.get("half", 0) + 1
                JOB["frac"] = (JOB.get("done", 0) + 0.5 * JOB["half"]) / \
                    float(max(1, len(plan)))
                continue
            if line.startswith(("PART ", "FAIL ")):
                i = int(line.split()[1])
                p = plan[i]
                t = N(p["target"])
                if "fbase" in JOB:
                    JOB["done"] = JOB.get("done", 0) + 1
                    if "half" in JOB:
                        JOB["half"] = max(0, JOB["half"] - 1)
                        JOB["frac"] = (JOB["done"] + 0.5 * JOB["half"]) / \
                            float(max(1, len(plan)))
                    else:
                        JOB["frac"] = JOB["fbase"] + (1 - JOB["fbase"]) * \
                            JOB["done"] / float(max(1, len(plan)))
                    if n:
                        n["status"] = "Making the views... %d of %d" % (
                            JOB["done"], len(plan))
                if line.startswith("PART"):
                    if p.get("views"):
                        v = N(p["views"][0])
                        if v:
                            v.setdefault("views", {})[
                                str(p["views"][1])] = p["out"]
                            for l in state()["links"]:
                                if l["from"] == p["views"]:
                                    tt = N(l["to"][0])
                                    if tt and tt["type"] == "image":
                                        tt["image"] = p["out"]
                                        tt["status"] = ""
                    if t:
                        t["image"] = p["out"]
                        t["status"] = ""
                elif t:
                    t["status"] = "Failed"
        for p in plan:
            t = N(p["target"])
            if t and t["status"] == "Waiting":
                t["status"] = "Making the picture..."
                break
    elif k == "views3d":
        m = main_addon()
        cj = m.CARD_JOB if m else None
        if cj is not None:
            if n:
                n["status"] = "3D model: " + cj.get("stage", "Working")
                JOB["frac"] = float(cj.get("frac", 0.0)) * 0.6
            return
        before = set(JOB["before"])
        new = [o for o in bpy.data.objects if o.name not in before]
        plan = JOB["plan"]
        if not new:
            if n:
                n["status"] = "The quick 3D model failed"
            JOB.clear()
            save()
            return
        ok = False
        try:
            render_views(new, plan)
            ok = True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            if n:
                n["status"] = "Could not render: %s" % str(exc)[:50]
        # the quick model was only a stand-in
        for o in new:
            try:
                bpy.data.objects.remove(o, do_unlink=True)
            except Exception:
                pass
        node = JOB["node"]
        JOB.clear()
        if ok:
            # the renders are redrawn in the look of the front picture
            jp = os.path.join(work_dir(), "views_refine.json")
            with open(jp, "w", encoding="utf-8") as fh:
                json.dump({"image": plan[0]["src"], "parts": plan,
                           "seed": 0}, fh, indent=1)
            free_engine()
            kl = server("klein")
            for _ in kl.lines():
                pass
            kl.send({"refine": jp})
            JOB.update({"kind": "extract", "node": node, "plan": plan,
                        "t": time.time(), "proc": kl, "frac": 0.62,
                        "fbase": 0.62, "done": 0})
            if n:
                n["status"] = "Cleaning up the views..."
            # the rough renders show at once; each is replaced by its clean
            # picture the moment that is ready
            import shutil
            for p in plan:
                t = N(p["target"])
                pv = p["out"][:-4] + "_preview.png"
                try:
                    shutil.copyfile(p["out"], pv)
                except Exception:
                    pv = ""
                if t:
                    if pv:
                        t["image"] = pv
                    t["status"] = "Making the picture..."
        save()
    elif k == "enh":
        m = main_addon()
        cj = m.CARD_JOB if m else None
        if cj is not None:
            if n:
                n["status"] = cj.get("stage", "Working")
                n["frac"] = float(cj.get("frac", 0.0))
            return
        before = set(JOB["before"])
        new = [o for o in bpy.data.objects if o.name not in before and
               o.type == "MESH"]
        if n:
            tgt = new[0].name if new else JOB["src_obj"]
            n["obj"] = tgt
            n["status"] = "In the scene: " + tgt
            n["frac"] = 1.0
            preview_of(n)
        JOB.clear()
        save()
    elif k == "gen":
        m = main_addon()
        cj = m.CARD_JOB if m else None
        if cj is not None:
            if n:
                n["status"] = cj.get("stage", "Working")
                n["frac"] = float(cj.get("frac", 0.0))
            return
        before = set(JOB["before"])
        new = [o for o in bpy.data.objects if o.name not in before]
        roots = [o for o in new if o.parent is None]
        if n:
            if roots:
                roots[0].name = JOB["label"]
                for o in new:
                    if o.type == "MESH" and o.parent is not None:
                        o.name = JOB["label"] + "_mesh"
                if JOB.get("mirror"):
                    new, roots = mirror_twin(roots[0], new)
                place(n, roots, new)
                n["obj"] = roots[0].name
                n["status"] = "In the scene: " + roots[0].name
                n["frac"] = 1.0
                n["preview"] = render_preview(new, os.path.join(
                    work_dir(), "%s_model_%d.png" % (n["id"],
                                                     int(time.time()))))
            else:
                s3 = getattr(bpy.context.scene, "img3d", None)
                n["status"] = (s3.status if s3 and s3.status
                               else "Stopped")[:60]
        JOB.clear()
        save()


# ------------------------------------------------------------- drawing

PAL_W = 170.0
PREVIEW_H = 230.0      # the model's picture on Generate / Enhance nodes
SOCK = {}            # ("in"/"out", nid, i) -> (x, y) region pixels
BOX = {}             # nid -> (x, y, w, h)
EDIT = {"on": None, "caret": 0, "field": "prompt", "anchor": None,
        "undo": [], "scroll": 0}       # node id whose prompt is typed in
DRAG = {}            # what the mouse drags now (for drawing the new link)


def Z():
    return state()["view"][2] * dpi()


def frame(reg, sel_only=False):
    """Fit every node (or the selected ones) in the view."""
    st = state()
    v = st["view"]
    ns = [N(i) for i in selected()] if sel_only and selected() else \
        list(st["nodes"].values())
    if not ns:
        return
    x0 = min(n["x"] for n in ns)
    x1 = max(n["x"] + nw(n) for n in ns)
    y0 = min(n["y"] - node_h(n) for n in ns)
    y1 = max(n["y"] for n in ns)
    left = PAL_W * dpi() + 40
    zz = max(0.04 * dpi(), min(1.5 * dpi(), min(
        (reg.width - left - 40) / max(1.0, x1 - x0),
        (reg.height - 80) / max(1.0, y1 - y0))))
    v[2] = zz / dpi()
    v[0] = left + ((reg.width - left) - (x1 - x0) * zz) / 2 - x0 * zz
    v[1] = (reg.height - (y1 - y0) * zz) / 2 - y0 * zz


def to_screen(x, y):
    v = state()["view"]
    z = Z()
    return v[0] + x * z, v[1] + y * z


STRETCH = ("prompt", "image", "gen3d", "detail", "etex")


def nw(n):
    """A node's width: its own size, or wider when it was pulled out."""
    return max(TYPES[n["type"]][1], float(n.get("w", 0) or 0))


def eh(n):
    """Extra height a node was pulled out by: it goes to its text box or
    its picture, so a bigger node shows more."""
    return max(0.0, float(n.get("eh", 0) or 0)) if n["type"] in STRETCH \
        else 0.0


def node_h(n):
    t = n["type"]
    if t == "prompt":
        return 38 + 150 + eh(n) + 8 + 30 + 18 + 30 + 18 + 46 + 8 + 40 + 30
    if t == "upload":
        return 38 + 36 + 8 + 40 + 30
    if t == "image":
        return 38 + 300 + eh(n) + (8 + 30 if n.get("meta") else 0) + 30
    if t == "parts":
        h = 38 + 40 + 26 + (12 if JOB.get("node") == n["id"] and
                            JOB.get("kind") in ("find", "seg") else 0)
        if n["found"]:
            h += len(n["items"]) * 48 + 8 + 30 + 10 + 40
        return h + 16
    if t == "views":
        return 38 + 40 + 4 * 34 + 30
    if t == "upscale":
        return 38 + 30 + 18 + 28 + 8 + 40 + 14 + 30
    # room for the model's picture is always kept, so a finished model
    # never pushes its node over the ones below
    pic = PREVIEW_H + eh(n) + 10
    if t == "gen3d":
        rows = 4 if n.get("engine") == "HUNYUAN" else 1
        return 38 + rows * 30 + 10 + 6 * (18 + 28 + 6) + 10 + 40 + 40 + pic
    if t == "detail":
        return 38 + 30 + 10 + 2 * (18 + 28 + 6) + 10 + 40 + 40 + pic
    if t == "etex":
        return 38 + 30 + 10 + 2 * (18 + 28 + 6) + 10 + 40 + 40 + pic
    return 120


def draw():
    sp = bpy.context.space_data
    if getattr(sp, "tree_type", "") != TREE:
        return
    st = state()
    reg = bpy.context.region
    z = Z()
    ox, oy, _ = st["view"]
    ui.HITS.clear()
    SOCK.clear()
    BOX.clear()
    import gpu
    gpu.state.blend_set("ALPHA")
    ui.fill(0, 0, reg.width, reg.height, 0, ui.BG)
    _grid(reg, ox, oy, 40 * z)
    ui.hit(0, 0, reg.width, reg.height, "bg")
    for n in st["nodes"].values():
        w, h = nw(n) * z, node_h(n) * z
        x, y = to_screen(n["x"], n["y"])
        BOX[n["id"]] = (x, y - h, w, h)
        _sockets(n, z)
    for l in st["links"]:
        a = SOCK.get(("out", l["from"][0], l["from"][1]))
        b = SOCK.get(("in", l["to"][0], l["to"][1]))
        if a and b:
            ui.bezier(a, b, (1, .576, .078, .75), 2.2 * z)
    snap = None
    if DRAG.get("kind") == "link":
        # the nearest free input pulls the wire in, like a magnet
        px, py = DRAG["pos"]
        best = 46.0 * z
        for (io, nid, i), (sx, sy) in SOCK.items():
            if io != "in" or nid == DRAG["from"][0]:
                continue
            # only inputs that can take this output pull the wire in
            _s, _t = N(DRAG["from"][0]), N(nid)
            if _s and _t and not fits(out_kind(_s, DRAG["from"][1]),
                                      _t["type"], i, _t):
                continue
            dd = ((sx - px) ** 2 + (sy - py) ** 2) ** 0.5
            if dd < best:
                best, snap = dd, (nid, i)
        DRAG["snap"] = snap
        a = SOCK.get(("out",) + tuple(DRAG["from"]))
        end = SOCK[("in",) + snap] if snap else DRAG["pos"]
        if a:
            ui.bezier(a, end, (1, .8, .5, 1.0) if snap else
                      (1, .8, .5, .6), (3.0 if snap else 2.2) * z)
    _auto_tabs(z)
    ss = set(selected())
    for n in st["nodes"].values():
        _draw_node(n, z, n["id"] in ss)
    if DRAG.get("kind") == "box":
        (ax, ay), (bx, by) = DRAG["a"], DRAG["pos"]
        bx0, by0 = min(ax, bx), min(ay, by)
        bw, bh = abs(bx - ax), abs(by - ay)
        ui.fill(bx0, by0, bw, bh, 4, (1.0, 0.576, 0.078, 0.10))
        ui.line([(bx0, by0), (bx0 + bw, by0), (bx0 + bw, by0 + bh),
                 (bx0, by0 + bh), (bx0, by0)], (1.0, 0.576, 0.078, 0.9),
                1.5)
    dragging = DRAG.get("kind") == "link"
    pulse = 0.5 + 0.5 * __import__("math").sin(time.time() * 8.0)
    for (io, nid, i), (sx, sy) in SOCK.items():
        r = 6.5 * z
        hot = dragging and snap == (nid, i) and io == "in"
        free_in = dragging and io == "in" and nid != DRAG["from"][0]
        hov = ui.HOVER[0] == "sock:%s:%s:%d" % (io, nid, i)
        if hot:
            g = (13 + 3 * pulse) * z
            ui.fill(sx - g, sy - g, 2 * g, 2 * g, g, (1, .576, .078, .28))
            r = 9.5 * z
        elif free_in:
            g = 10 * z
            ui.fill(sx - g, sy - g, 2 * g, 2 * g, g, (1, .576, .078, .12))
        elif hov:
            r = 8.5 * z
        ui.fill(sx - r - 1.5, sy - r - 1.5, 2 * r + 3, 2 * r + 3, r + 1.5,
                (0.05, 0.05, 0.06, 1))
        ui.fill(sx - r, sy - r, 2 * r, 2 * r, r,
                (1.0, 0.8, 0.5, 1.0) if hot else ui.ACCENT)
        ui.hit(sx - r * 1.8, sy - r * 1.8, r * 3.6, r * 3.6,
               "sock:%s:%s:%d" % (io, nid, i))
    _palette(reg)
    _menu(z)
    ui.text("Drag on the background to select several  \u00b7  Shift+click "
            "adds to the selection  \u00b7  Drag from a dot to connect  "
            "\u00b7  Drag a connected input away to cut it  \u00b7  X or "
            "Delete removes the selected  \u00b7  Middle mouse moves, wheel "
            "zooms  \u00b7  Home shows everything",
            PAL_W * dpi() + 30, 12, 11 * dpi(), ui.TEXT_DIM)


def _auto_tabs(z):
    """An Auto Run tab on top of the first node of every chain, tucked
    under the node's top edge like a part of it."""
    for nid in chain_roots():
        if nid not in BOX:
            continue
        x, y, w, h = BOX[nid]
        top = y + h
        tw, th = 132 * z, 34 * z
        mine = AUTO.get("root") == nid
        if AUTO["on"] and mine:
            ui.button(x + 14 * z, top - 8 * z, tw, th, "Stop Auto Run",
                      "autorun:" + nid, z=z, size=12)
            cur = N(AUTO["cur"]) if AUTO["cur"] else None
            msg = "Running %d of %d%s" % (
                max(AUTO["i"], 1), len(AUTO["order"]),
                (": " + TYPES[cur["type"]][0]) if cur else "")
        else:
            ui.button(x + 14 * z, top - 8 * z, tw, th, "Auto Run",
                      "autorun:" + nid, kind="accent", z=z, size=12,
                      enabled=not busy() or AUTO["on"])
            msg = AUTO.get("msg", "") if mine else ""
        if msg:
            ui.text(msg, x + tw + 24 * z, top + 4 * z, 11 * z,
                    ui.ACCENT if AUTO["on"] and mine else ui.TEXT_DIM,
                    room=max(w - tw - 30 * z, 60 * z))


def _grid(reg, ox, oy, step):
    if step < 8:
        return
    pts = []
    y = oy % step
    while y < reg.height:
        x = ox % step
        while x < reg.width:
            pts += [(x, y), (x + 1.5, y), (x + 1.5, y + 1.5),
                    (x, y), (x + 1.5, y + 1.5), (x, y + 1.5)]
            x += step
        y += step
    if pts:
        from gpu_extras.batch import batch_for_shader
        sh = ui._sh()
        b = batch_for_shader(sh, "TRIS", {"pos": pts})
        sh.bind()
        sh.uniform_float("color", ui.GRID)
        b.draw(sh)


def _sockets(n, z):
    x, y, w, h = BOX[n["id"]]
    _title, _w, _ins, outs = TYPES[n["type"]]
    ins = ins_of(n)
    top = y + h - 60 * z
    t = n["type"]
    for i, _nm in enumerate(ins):
        yy = y + h - (38 + 15 + 30 * i) * z if t in (
            "gen3d", "detail", "etex") else top
        SOCK[("in", n["id"], i)] = (x, yy)
    for i, _nm in enumerate(outs):
        yy = y + h - (38 + 40 + 23 + 34 * i) * z if t == "views" else (
            y + h - 53 * z if t in ("gen3d", "detail", "etex") else top)
        SOCK[("out", n["id"], i)] = (x + w, yy)


def _palette(reg):
    d = dpi()
    w = PAL_W * d
    h = (52 + len(PALETTE) * 40 + 10) * d
    x, y = 12 * d, reg.height - h - 12 * d
    ui.outline(x, y, w, h, 14 * d, (0, 0, 0, .4), 2)
    ui.grad(x, y, w, h, 14 * d, ui.CARD_BOT, ui.CARD_TOP)
    ui.fill(x + 14 * d, y + h - 26 * d, 8 * d, 8 * d, 4 * d, ui.ACCENT)
    ui.text("ADD NODE", x + 30 * d, y + h - 26 * d, 11.5 * d, ui.TEXT,
            bold=True)
    ui.hit(x, y, w, h, "pal")
    cy = y + h - 52 * d
    for k in PALETTE:
        cy -= 40 * d
        ui.button(x + 12 * d, cy + 4 * d, w - 24 * d, 32 * d,
                  TYPES[k][0], "add:" + k, z=d, size=12)


def _draw_node(n, z, sel):
    x, y, w, h = BOX[n["id"]]
    t = n["type"]
    title = TYPES[t][0]
    if t == "image" and n.get("label") and n["label"] != "Image":
        title = n["label"]
        if (n.get("meta") or {}).get("mirror"):
            title += "  (one of a pair)"
    done = t in ("gen3d", "detail", "etex") and bool(n.get("obj"))
    ui.card(x, y, w, h, title, z,
            icon_col=(0.35, 0.85, 0.45, 1) if done else ui.ACCENT,
            selected=sel)
    ui.HITS[-1] = ui.HITS[-1][:4] + ("node:" + n["id"],)
    ui.hit(x, y, w, h - 38 * z, "body:" + n["id"])
    pad = 14 * z
    cy = y + h - 38 * z
    iw = w - 2 * pad
    nid = n["id"]
    if t == "prompt":
        tbh = (150 + eh(n)) * z
        cy -= tbh
        ed_ = EDIT["on"] == nid and EDIT.get("field") == "prompt"
        ui.text_box(x + pad, cy, iw, tbh, n["prompt"],
                    "Click here and describe the picture you want...",
                    ed_, EDIT["caret"], z, "type:" + nid,
                    anchor=EDIT.get("anchor"), scroll=EDIT.get("scroll"))
        if ed_:
            EDIT["scroll"] = ui.TB["type:" + nid]["first"]
        cy -= 8 * z + 30 * z
        ui.button(x + pad, cy, iw, 30 * z,
                  "3D Ready: plain studio background" if n.get("ready")
                  else "3D Ready: off (prompt as written)",
                  "ready:" + nid, kind="picked" if n.get("ready") else "ctl",
                  z=z, size=11)
        cy -= 18 * z
        ui.text("Model", x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
        cy -= 30 * z
        ui.chips(x + pad, cy, iw, 30 * z, PMODELS,
                 n.get("pmodel", "image"), "pmodel:" + nid, z)
        cy -= 18 * z
        ui.text("Aspect", x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
        cy -= 46 * z
        ui.shape_chips(x + pad, cy, iw, 46 * z, ASPECTS,
                       n.get("aspect", "2x3"), "aspect:" + nid, z)
        cy -= 8 * z + 40 * z
        run_button(x + pad, cy, iw, 40 * z, "Generate Picture",
                   "t2i:" + nid, nid, z,
                   enabled=bool(n["prompt"]) and not busy())
        if JOB.get("node") == nid:
            ui.bar(x + pad, cy - 12 * z, iw, 5 * z, JOB.get("frac", 0.0)
                   if JOB.get("kind") == "t2i" else 1.0, z)
    elif t == "upload":
        cy -= 36 * z
        ui.text(os.path.basename(n["image"]) or "No picture chosen",
                x + pad, cy + 12 * z, 11.5 * z,
                ui.TEXT if n["image"] else ui.TEXT_DIM, room=iw - 30 * z)
        if n["image"]:
            ui.clear_btn(x + pad + iw - 12 * z, cy + 17 * z, z,
                         "clr:" + nid)
        cy -= 8 * z + 40 * z
        ui.button(x + pad, cy, iw, 40 * z, "Choose Image", "pick:" + nid,
                  kind="accent", z=z, size=13)
    elif t == "image":
        ph_ = (300 + eh(n)) * z
        cy -= ph_
        has = ui.picture(n["image"], x + pad, cy, iw, ph_)
        if has:
            ui.clear_btn(x + pad + iw - 16 * z, cy + ph_ - 16 * z, z,
                         "clr:" + nid)
            ui.save_btn(x + pad + iw - 44 * z, cy + ph_ - 16 * z, z,
                        "savepic:" + nid)
        if not has:
            ui.text(n["status"] or "Connect something that makes a "
                    "picture", 0, cy + ph_ * .5, 11 * z, ui.TEXT_DIM,
                    mid=x + w * .5, room=iw - 10 * z)
        if has and n.get("status"):
            busy_ = n["status"].startswith(("Checking", "Not", "Making"))
            col = (0.45, 0.9, 0.55, 1) if n["status"] == "Checked" else (
                ui.ACCENT if busy_ else ui.TEXT_DIM)
            import blf as _blf
            _blf.size(0, 10.5 * z)
            tw_ = _blf.dimensions(0, n["status"])[0]
            ui.fill(x + pad + 2 * z, cy + 3 * z, tw_ + 30 * z, 20 * z,
                    10 * z, (0.06, 0.07, 0.08, 0.85))
            ui.fill(x + pad + 8 * z, cy + 10 * z, 7 * z, 7 * z, 3.5 * z, col)
            ui.text(n["status"], x + pad + 22 * z, cy + 8 * z, 10.5 * z, col,
                    room=iw - 30 * z)
        if n.get("meta"):
            cy -= 8 * z + 30 * z
            half = (iw - 6 * z) / 2
            ui.button(x + pad, cy, half, 30 * z, "New Picture",
                      "redo:" + nid, z=z, size=11,
                      enabled=not busy() and bool(n["meta"].get("src")))
            ui.button(x + pad + half + 6 * z, cy, half, 30 * z,
                      "Mirror Pair", "mirror:" + nid, z=z, size=11,
                      kind="picked" if n["meta"].get("mirror") else "ctl")
    elif t == "parts":
        cy -= 40 * z + 6 * z
        run_button(x + pad, cy, iw, 40 * z,
                   "Find Parts" if not n["found"] else "Find Again",
                   "find:" + nid, nid, z,
                   enabled=not busy() and bool(image_of(nid)))
        if JOB.get("node") == nid and JOB.get("kind") in ("find", "seg"):
            cy -= 12 * z
            ui.bar(x + pad, cy, iw, 6 * z, JOB.get("frac", 0.0), z)
            ui.text("%d%%" % int(JOB.get("frac", 0.0) * 100), 0, cy + 10 * z,
                    10 * z, ui.TEXT_DIM, right=x + w - pad)
        cy -= 20 * z
        ui.text(n["status"] or "Connect a picture, then Find Parts",
                x + pad, cy, 11 * z, ui.TEXT_DIM, room=iw)
        if n["found"]:
            cy -= 6 * z
            for k, it in enumerate(n["items"]):
                cy -= 48 * z
                ui.fill(x + pad, cy, iw, 44 * z, 8 * z,
                        ui.PANEL_TOP if it["on"] else (0.08, .09, .1, 1))
                ui.check(x + pad + 8 * z, cy + 22 * z, 16 * z, it["on"],
                         "pon:%s:%d" % (nid, k), z)
                ui.text(it["name"] + ("" if it.get("found", True) else
                                      "   (not seen in the picture)"),
                        x + pad + 32 * z, cy + 26 * z,
                        12.5 * z, ui.TEXT if it["on"] else ui.TEXT_DIM,
                        bold=True, room=iw - 70 * z)
                ui.hit(x + pad + 30 * z, cy + 22 * z, iw - 70 * z, 20 * z,
                       "pname:%s:%d" % (nid, k))
                ui.text(it.get("description", ""), x + pad + 32 * z,
                        cy + 8 * z, 10 * z, ui.TEXT_DIM, room=iw - 44 * z)
                ui.hit(x + pad + 30 * z, cy + 2 * z, iw - 44 * z, 18 * z,
                       "pdesc:%s:%d" % (nid, k))
                ui.button(x + w - pad - 28 * z, cy + 22 * z, 22 * z,
                          18 * z, "\u00d7", "pdel:%s:%d" % (nid, k), z=z,
                          size=13)
            cy -= 8 * z + 30 * z
            third = (iw - 12 * z) / 3
            ui.button(x + pad, cy, third, 30 * z, "+ Add Part",
                      "padd:" + nid, z=z, size=11)
            ui.button(x + pad + third + 6 * z, cy, third, 30 * z,
                      "Base Body", "pbody:" + nid,
                      kind="picked" if n["body"] else "ctl", z=z, size=11)
            ui.button(x + pad + 2 * (third + 6 * z), cy, third, 30 * z,
                      "3/4 View", "piso:" + nid,
                      kind="picked" if n.get("iso", True) else "ctl", z=z,
                      size=11)
            cy -= 10 * z + 40 * z
            ui.button(x + pad, cy, iw, 40 * z, "Start", "start:" + nid,
                      kind="accent", z=z, size=13,
                      enabled=not busy() and (n["body"] or any(
                          i["on"] for i in n["items"])))
    elif t == "views":
        cy -= 40 * z + 6 * z
        run_button(x + pad, cy, iw, 40 * z, "Make Views", "views:" + nid,
                   nid, z, enabled=not busy() and bool(image_of(nid)))
        if JOB.get("node") == nid and JOB.get("kind") in ("views3d",
                                                          "extract"):
            ui.bar(x + pad, cy - 12 * z, iw, 6 * z, JOB.get("frac", 0.0), z)
            ui.text("%d%%" % int(JOB.get("frac", 0.0) * 100), x + pad,
                    cy - 28 * z, 10 * z, ui.TEXT_DIM)
        for i, nm in enumerate(TYPES["views"][3]):
            sy = SOCK[("out", nid, i)][1]
            ui.text(nm, 0, sy - 5 * z, 11.5 * z, ui.TEXT, right=x + w - pad)
    elif t == "gen3d":
        for i, nm in enumerate(ins_of(n)):
            sy = SOCK[("in", nid, i)][1]
            linked = bool(image_of(nid, i))
            ui.text(nm + ("" if i == 0 else " view"), x + pad + 4 * z,
                    sy - 5 * z, 11.5 * z, ui.TEXT if linked else ui.TEXT_DIM)
        ui.text("Model", 0, SOCK[("out", nid, 0)][1] - 5 * z, 11.5 * z,
                ui.TEXT if n.get("obj") else ui.TEXT_DIM, right=x + w - pad)
        rows = len(ins_of(n))
        cy = y + h - (38 + rows * 30 + 10) * z

        def row(label, items, cur, action):
            nonlocal cy
            cy -= 18 * z
            ui.text(label, x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
            cy -= 28 * z
            ui.chips(x + pad, cy, iw, 28 * z, items, cur, action, z)
            cy -= 6 * z
        row("Engine", ENGINES, n["engine"], "eng:" + nid)
        row("Quality", QUALITY, n.get("quality", "HIGH"), "qual:" + nid)
        cy -= 18 * z
        ui.text("Faces", x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
        cy -= 28 * z
        ed = EDIT["on"] == nid and EDIT.get("field") == "faces"
        ui.outline(x + pad, cy, iw, 28 * z, 8 * z,
                   ui.ACCENT if ed else ui.PANEL_EDGE, 1.5 if ed else 1.0)
        ui.fill(x + pad, cy, iw, 28 * z, 8 * z, (0.045, 0.05, 0.055, 1))
        fs_ = str(n.get("faces", ""))
        shown = fs_ if ed else "{:,}".format(int(fs_ or 0))
        if ed and EDIT.get("anchor") is not None and \
                EDIT["anchor"] != EDIT["caret"]:
            import blf as _b
            _b.size(0, 12 * z)
            a_, b_ = sorted((EDIT["anchor"], EDIT["caret"]))
            x0_ = x + pad + 10 * z + _b.dimensions(0, fs_[:a_])[0]
            x1_ = x + pad + 10 * z + _b.dimensions(0, fs_[:b_])[0]
            ui.fill(x0_, cy + 5 * z, x1_ - x0_, 18 * z, 0,
                    (1.0, 0.576, 0.078, 0.38))
        tw_ = ui.text(shown, x + pad + 10 * z, cy + 9 * z, 12 * z, ui.TEXT)
        if ed and (time.time() * 2) % 2 < 1.3:
            import blf as _b
            _b.size(0, 12 * z)
            cx_ = x + pad + 10 * z + _b.dimensions(0, fs_[:EDIT["caret"]])[0]
            ui.fill(cx_, cy + 5 * z, 1.6 * z, 18 * z, 0, ui.ACCENT)
        ui.text("click to type", 0, cy + 9 * z, 10 * z, ui.TEXT_DIM,
                right=x + w - pad - 8 * z)
        ui.hit(x + pad, cy, iw, 28 * z, "facesbox:" + nid)
        cy -= 6 * z
        row("Texture", (("on", "On"), ("off", "Off")),
            "on" if n["texture"] else "off", "texon:" + nid)
        row("Texture Size", TEXSIZE, n.get("texsize", "2048"),
            "tsize:" + nid)
        cy -= 18 * z
        ui.text("Seed", x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
        cy -= 28 * z
        q4 = (iw - 18 * z) / 4
        ui.button(x + pad, cy, q4, 28 * z, "\u2212", "seed-:" + nid, z=z)
        ui.fill(x + pad + q4 + 6 * z, cy, q4 * 2, 28 * z, 8 * z,
                (0.05, .055, .06, 1))
        ui.text(str(n.get("seed", 0)), 0, cy + 9 * z, 12 * z, ui.TEXT,
                mid=x + pad + q4 * 2 + 6 * z)
        ui.button(x + pad + q4 * 3 + 12 * z, cy, q4, 28 * z, "+",
                  "seed+:" + nid, z=z)
        cy -= 6 * z
        cy -= 10 * z + 40 * z
        queued = nid in (state().get("queue") or [])
        run_button(x + pad, cy, iw, 40 * z,
                   "Generate Again" if n.get("obj") else "Generate 3D",
                   "gen:" + nid, nid, z,
                   enabled=bool(image_of(nid)) and not queued and not (
                       busy() and JOB.get("node") == nid))
        if JOB.get("kind") == "gen" and JOB.get("node") == nid:
            ui.bar(x + pad, cy - 12 * z, iw, 5 * z, n.get("frac", 0.0), z)
    elif t in ("detail", "etex"):
        src, _ = source(nid, 0)
        has_in = bool(src and src.get("obj"))
        ui.text("Model", x + pad + 4 * z, SOCK[("in", nid, 0)][1] - 5 * z,
                11.5 * z, ui.TEXT if has_in else ui.TEXT_DIM)
        ui.text("Model", 0, SOCK[("out", nid, 0)][1] - 5 * z, 11.5 * z,
                ui.TEXT if n.get("obj") else ui.TEXT_DIM, right=x + w - pad)
        cy = y + h - (38 + 30 + 10) * z

        def row2(label, items, cur, action):
            nonlocal cy
            cy -= 18 * z
            ui.text(label, x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
            cy -= 28 * z
            ui.chips(x + pad, cy, iw, 28 * z, items, cur, action, z)
            cy -= 6 * z
        if t == "detail":
            row2("Quality", QUALITY[:2], n.get("quality", "HIGH"),
                 "dqual:" + nid)
            row2("Texture", (("on", "Repaint"), ("off", "Shape Only")),
                 "on" if n.get("repaint", True) else "off",
                 "drep:" + nid)
            label = "Enhance Detail"
        else:
            row2("Enhance", (("FAST", "Fast"), ("HIGH", "High")),
                 n.get("mode", "HIGH"), "emode:" + nid)
            row2("Creativity", (("0.0", "0"), ("0.3", "0.3"),
                                ("0.6", "0.6"), ("1.0", "1")),
                 n.get("creativity", "0.0"), "ecre:" + nid)
            label = "Enhance Texture"
        cy -= 10 * z + 40 * z
        run_button(x + pad, cy, iw, 40 * z, label, "enh:" + nid, nid, z,
                   enabled=has_in and not busy())
        if JOB.get("kind") == "enh" and JOB.get("node") == nid:
            ui.bar(x + pad, cy - 12 * z, iw, 5 * z, n.get("frac", 0.0), z)
    if t == "upscale":
        cy -= 30 * z
        src_ = image_of(nid)
        ui.text("Image", x + pad + 4 * z, SOCK[("in", nid, 0)][1] - 5 * z,
                11.5 * z, ui.TEXT if src_ else ui.TEXT_DIM)
        ui.text("Image", 0, SOCK[("out", nid, 0)][1] - 5 * z, 11.5 * z,
                ui.TEXT if n.get("image") else ui.TEXT_DIM,
                right=x + w - pad)
        cy -= 18 * z
        ui.text("Scale", x + pad, cy + 4 * z, 10.5 * z, ui.TEXT_DIM)
        cy -= 28 * z
        ui.chips(x + pad, cy, iw, 28 * z, SCALES, n.get("scale", "2"),
                 "scale:" + nid, z)
        cy -= 8 * z + 40 * z
        run_button(x + pad, cy, iw, 40 * z, "Upscale", "ups:" + nid, nid,
                   z, enabled=bool(src_) and not busy())
        if JOB.get("kind") == "ups" and JOB.get("node") == nid:
            ui.bar(x + pad, cy - 12 * z, iw, 5 * z, JOB.get("frac", 0.0), z)
    if t in ("gen3d", "detail", "etex"):
        pv_ = (PREVIEW_H + eh(n)) * z
        if n.get("preview") and ui.picture(
                n["preview"], x + pad, y + 30 * z, iw, pv_, 8 * z):
            ui.clear_btn(x + pad + iw - 16 * z,
                         y + 30 * z + pv_ - 16 * z, z,
                         "clr:" + nid)
        else:
            ui.fill(x + pad, y + 30 * z, iw, pv_, 8 * z,
                    (0.05, 0.055, 0.06, 1.0))
            ui.text("The model appears here", 0,
                    y + 30 * z + pv_ / 2 - 4 * z, 10.5 * z,
                    ui.TEXT_DIM, mid=x + w / 2)
    running = JOB.get("node") == nid and "t0" in JOB
    if running:
        # how long it has been working, at the end of the status line
        clock(nid, y + 12 * z, z, right=x + w - pad)
    if n.get("status") and t not in ("image", "parts"):
        ui.text(n["status"], x + pad, y + 12 * z, 10.5 * z,
                (0.45, 0.9, 0.55, 1) if done else ui.TEXT_DIM,
                room=iw - (40 * z if running else 0) - 14 * z)
    # the corner grip: drag it to make the node bigger
    g = 12 * z
    col_ = ui.ACCENT if ui.HOVER[0] == "rsz:" + nid else ui.TEXT_DIM
    for k in range(3):
        o = (k + 1) * 3.5 * z
        ui.line([(x + w - 3 * z - o, y + 3 * z),
                 (x + w - 3 * z, y + 3 * z + o)], col_, 1.2 * z)
    ui.hit(x + w - g - 4 * z, y, g + 4 * z, g + 4 * z, "rsz:" + nid)


# --------------------------------------------------------------- actions

def _pulse():
    def tick():
        redraw()
        return 0.04 if DRAG.get("kind") == "link" else None
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=0.04)


def _blink():
    def tick():
        redraw()
        return 0.5 if EDIT["on"] else None
    if not bpy.app.timers.is_registered(tick):
        bpy.app.timers.register(tick, first_interval=0.5)


def _end_edit():
    n = N(EDIT["on"])
    if n is not None and EDIT.get("field") == "faces":
        try:
            n["faces"] = str(max(1000, min(50000000, int(n.get("faces")
                                                          or 0) or 200000)))
        except Exception:
            n["faces"] = "200000"
    EDIT["on"] = None
    EDIT["anchor"] = None
    save()


def _word_left(s, c):
    while c > 0 and s[c - 1] == " ":
        c -= 1
    while c > 0 and s[c - 1] != " ":
        c -= 1
    return c


def _word_right(s, c):
    while c < len(s) and s[c] == " ":
        c += 1
    while c < len(s) and s[c] != " ":
        c += 1
    return c


def word_at(s, c):
    """The word around position c, for a double click."""
    a = b = max(0, min(c, len(s)))
    while a > 0 and s[a - 1] != " ":
        a -= 1
    while b < len(s) and s[b] != " ":
        b += 1
    return a, b


def type_key(event):
    """Typing in a text field, like any text box: a selection is replaced,
    Shift extends it, Ctrl moves by words; copy, cut, paste and undo."""
    n = N(EDIT["on"])
    if n is None:
        EDIT["on"] = None
        return
    field = EDIT.get("field", "prompt")
    s = str(n.get(field, ""))
    c = max(0, min(EDIT["caret"], len(s)))
    a = EDIT.get("anchor")
    a = c if a is None else max(0, min(a, len(s)))
    lo, hi = min(a, c), max(a, c)
    has_sel = lo != hi
    t = event.type
    ctrl = event.ctrl or event.oskey
    shift = event.shift
    act_ = "TB:type:" + EDIT["on"] if field == "prompt" else None
    before = s
    wm = bpy.context.window_manager

    def cut():
        return s[:lo] + s[hi:], lo

    move = None
    if t in ("ESC", "RET", "NUMPAD_ENTER"):
        _end_edit()
        redraw()
        return
    elif ctrl and t == "A":
        a, c = 0, len(s)
    elif ctrl and t in ("C", "X"):
        if has_sel:
            wm.clipboard = s[lo:hi]
            if t == "X":
                s, c = cut()
                a = c
    elif ctrl and t == "V":
        clip = wm.clipboard.replace("\r", " ").replace("\n", " ")
        if field == "faces":
            clip = "".join(ch for ch in clip if ch.isdigit())
        if has_sel:
            s, c = cut()
        s, c = s[:c] + clip + s[c:], c + len(clip)
        a = c
    elif ctrl and t == "Z":
        if EDIT["undo"]:
            s, c = EDIT["undo"].pop()
            a = c
            n[field] = s
            EDIT["caret"], EDIT["anchor"] = c, a
            redraw()
            return
    elif t == "BACK_SPACE":
        if has_sel:
            s, c = cut()
        elif ctrl:
            j = _word_left(s, c)
            s, c = s[:j] + s[c:], j
        elif c > 0:
            s, c = s[:c - 1] + s[c:], c - 1
        a = c
    elif t == "DEL":
        if has_sel:
            s, c = cut()
        elif ctrl:
            s = s[:c] + s[_word_right(s, c):]
        else:
            s = s[:c] + s[c + 1:]
        a = c
    elif t == "LEFT_ARROW":
        move = _word_left(s, c) if ctrl else (
            lo if has_sel and not shift else max(0, c - 1))
    elif t == "RIGHT_ARROW":
        move = _word_right(s, c) if ctrl else (
            hi if has_sel and not shift else min(len(s), c + 1))
    elif t in ("UP_ARROW", "DOWN_ARROW"):
        if field == "prompt":
            move = ui.tb_line_move("type:" + EDIT["on"], c,
                                   -1 if t == "UP_ARROW" else 1)
        else:
            move = 0 if t == "UP_ARROW" else len(s)
    elif t == "HOME":
        move = 0
    elif t == "END":
        move = len(s)
    elif event.unicode and event.unicode.isprintable() and not ctrl and (
            field != "faces" or event.unicode.isdigit()):
        if has_sel:
            s, c = cut()
        s, c = s[:c] + event.unicode + s[c:], c + len(event.unicode)
        a = c
    else:
        return
    if move is not None:
        c = move
        if not shift:
            a = c
    if s != before:
        # one undo step per word, not per letter
        u = EDIT["undo"]
        if not u or event.unicode in (" ", "") or len(u) == 0 or \
                abs(len(s) - len(before)) > 1:
            u.append((before, EDIT["caret"]))
        del u[:-100]
    if field == "faces":
        n["faces_typed"] = True
        s = "".join(ch for ch in s if ch.isdigit())[:9]
        c = min(c, len(s))
        a = min(a, len(s))
    n[field] = s
    EDIT["caret"], EDIT["anchor"] = c, a
    redraw()


_REG = [None]     # the canvas region, for actions run from the modal


def act(a, context):
    st = state()
    parts = a.split(":")
    kind = parts[0]
    nid = parts[1] if len(parts) > 1 else ""
    n = N(nid)
    if kind not in ("type", "facesbox") and EDIT["on"]:
        _end_edit()
    if kind == "add" and add_to_selected(nid):
        pass
    elif kind == "add":
        reg = _REG[0] or context.region
        z = Z()
        v = st["view"]
        wx = (reg.width * .45 - v[0]) / z
        wy = (reg.height * .75 - v[1]) / z
        k = len(st["nodes"]) % 5
        add_node(nid, wx + k * 30, wy - k * 30)
    elif kind == "autorun":
        auto_begin(nid)
    elif n is None:
        return
    elif kind == "type":
        if EDIT["on"] != nid or EDIT.get("field") != "prompt":
            EDIT.update(on=nid, field="prompt", caret=len(n["prompt"]),
                        anchor=None, undo=[], scroll=0)
        _blink()
    elif kind == "facesbox":
        if EDIT["on"] != nid or EDIT.get("field") != "faces":
            fs = str(n.get("faces", ""))
            # the whole number selected: typing replaces it
            EDIT.update(on=nid, field="faces", caret=len(fs), anchor=0,
                        undo=[], scroll=0)
        _blink()
    elif kind == "ready":
        n["ready"] = not n.get("ready", True)
    elif kind == "aspect":
        n["aspect"] = parts[2]
    elif kind == "pmodel":
        n["pmodel"] = parts[2]
    elif kind == "scale":
        n["scale"] = parts[2]
    elif kind == "ups":
        run_upscale(n)
    elif kind == "t2i":
        run_prompt(n)
    elif kind == "pick":
        bpy.ops.img3d_canvas.pick_image("INVOKE_DEFAULT", node=nid)
    elif kind == "find":
        run_find(n)
    elif kind == "pon":
        it = n["items"][int(parts[2])]
        it["on"] = not it["on"]
    elif kind in ("pname", "pdesc"):
        k = int(parts[2])
        bpy.ops.img3d_canvas.edit_text(
            "INVOKE_DEFAULT", target="%s:%s:%d" % (kind, nid, k),
            value=n["items"][k]["name" if kind == "pname" else
                                "description"])
    elif kind == "pdel":
        n["items"].pop(int(parts[2]))
    elif kind == "padd":
        bpy.ops.img3d_canvas.add_part("INVOKE_DEFAULT", node=nid)
    elif kind == "savepic":
        if n.get("image") and os.path.isfile(n["image"]):
            bpy.ops.img3d_canvas.save_image("INVOKE_DEFAULT", node=nid)
    elif kind == "clr":
        # the picture goes; the node stays empty until something new
        # arrives (the model itself stays in the scene)
        if n["type"] in ("image", "upload"):
            n["image"] = ""
            if n["type"] == "image":
                n["status"] = ""
        else:
            n["preview"] = ""
    elif kind == "pbody":
        n["body"] = not n["body"]
    elif kind == "piso":
        n["iso"] = not n.get("iso", True)
    elif kind == "start":
        run_start(n)
    elif kind == "redo":
        m = n["meta"]
        run_extract(nid, [dict(m, out=os.path.join(
            work_dir(), "%s_%d.png" % (nid, int(time.time()))),
            target=nid)], m["src"], seed=int(time.time()) % 1000)
        n["status"] = "Making a new picture..."
        n["image"] = ""
    elif kind == "mirror":
        n["meta"]["mirror"] = not n["meta"].get("mirror")
    elif kind == "views":
        run_views(n)
    elif kind == "eng":
        if not n.get("faces_typed"):
            n["faces"] = "40000" if parts[2] == "HUNYUAN" else "200000"
        n["engine"] = parts[2]
        if parts[2] != "HUNYUAN":
            st["links"] = [l for l in st["links"]
                           if not (l["to"][0] == nid and l["to"][1] > 0)]
    elif kind == "qual":
        n["quality"] = parts[2]
    elif kind == "faces":
        n["faces"] = parts[2]
    elif kind == "texon":
        n["texture"] = parts[2] == "on"
    elif kind == "tsize":
        n["texsize"] = parts[2]
    elif kind in ("seed-", "seed+"):
        n["seed"] = max(0, int(n.get("seed", 0)) + (1 if kind == "seed+"
                                                     else -1))
    elif kind == "adet":
        n["auto_detail"] = not n.get("auto_detail")
    elif kind == "aenh":
        n["auto_enh"] = not n.get("auto_enh")
    elif kind == "dqual":
        n["quality"] = parts[2]
    elif kind == "drep":
        n["repaint"] = parts[2] == "on"
    elif kind == "emode":
        n["mode"] = parts[2]
    elif kind == "ecre":
        n["creativity"] = parts[2]
    elif kind == "enh":
        run_enh(n)
    elif kind == "tex":
        n["texture"] = not n["texture"]
    elif kind == "cancel":
        cancel_job(nid)
    elif kind == "gen":
        if busy():
            st.setdefault("queue", []).append(nid)
            _timer()
        else:
            run_gen(n)
    save()


UI_TOKENS = bpy.app.driver_namespace.setdefault("img3d_canvas_ui", {})


class IMG3DCANVAS_OT_ui(bpy.types.Operator):
    """Clicks, dragging, connecting and zooming on the canvas"""
    bl_idname = "img3d_canvas.ui"
    bl_label = "Canvas"

    def invoke(self, context, event):
        self.area = context.area
        self.drag = None
        # one listener per canvas: a newer one (after an add-on reload, or
        # opened again) retires the older, which would otherwise keep
        # catching the clicks with stale code
        self._tok = time.time()
        UI_TOKENS[self.area.as_pointer()] = self._tok
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _region(self):
        try:
            for r in self.area.regions:
                if r.type == "WINDOW":
                    return r
        except ReferenceError:
            return None
        return None

    def _hit(self, mx, my):
        for x, y, w, h, a in reversed(ui.HITS):
            if x <= mx <= x + w and y <= my <= y + h:
                return a
        return None

    def modal(self, context, event):
        try:
            if self.area.spaces.active.tree_type != TREE:
                return {"CANCELLED"}
            if UI_TOKENS.get(self.area.as_pointer()) != self._tok:
                return {"CANCELLED"}
        except (ReferenceError, AttributeError):
            return {"CANCELLED"}
        reg = self._region()
        if reg is None:
            return {"CANCELLED"}
        mx, my = event.mouse_x - reg.x, event.mouse_y - reg.y
        inside = 0 <= mx < reg.width and 0 <= my < reg.height
        _REG[0] = reg
        st = state()
        v = st["view"]
        if self.drag:
            d = self.drag
            if event.type in ("MOUSEMOVE", "INBETWEEN_MOUSEMOVE"):
                dx, dy = mx - d["mx"], my - d["my"]
                d["mx"], d["my"] = mx, my
                if d["kind"] == "pan":
                    v[0] += dx
                    v[1] += dy
                elif d["kind"] == "node":
                    for i in d["ids"]:
                        n = N(i)
                        if n:
                            n["x"] += dx / Z()
                            n["y"] += dy / Z()
                elif d["kind"] == "rsz":
                    n = N(d["nid"])
                    if n:
                        base = TYPES[n["type"]][1]
                        n["w"] = max(base, nw(n) + dx / Z())
                        if n["type"] in STRETCH:
                            n["eh"] = max(0.0, eh(n) - dy / Z())
                elif d["kind"] == "box":
                    DRAG["pos"] = (mx, my)
                elif d["kind"] == "link":
                    DRAG["pos"] = (mx, my)
                    _pulse()
                elif d["kind"] == "tsel":
                    k_ = ui.tb_index(d["act"], mx, my)
                    if k_ is not None:
                        EDIT["caret"] = k_
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
            if event.value == "RELEASE" and event.type in (
                    "LEFTMOUSE", "MIDDLEMOUSE"):
                if d["kind"] == "box":
                    (ax, ay) = DRAG["a"]
                    x0, x1 = sorted((ax, mx))
                    y0, y1 = sorted((ay, my))
                    hits = [nid for nid, (bx, by, bw, bh) in BOX.items()
                            if bx < x1 and bx + bw > x0 and by < y1 and
                            by + bh > y0]
                    if abs(mx - ax) + abs(my - ay) < 4:
                        hits = []
                    st["selset"] = list(dict.fromkeys(
                        (st.get("selset", []) if d.get("shift") else [])
                        + hits))
                    st["sel"] = hits[-1] if hits else (
                        st["sel"] if d.get("shift") else None)
                    DRAG.clear()
                if d["kind"] == "link":
                    sn = DRAG.get("snap")
                    if sn:
                        link(d["from"][0], d["from"][1], sn[0], int(sn[1]))
                    elif abs(mx - d["x0"]) + abs(my - d["y0"]) > 12:
                        on = [nid for nid, (bx, by, bw, bh) in BOX.items()
                              if bx <= mx <= bx + bw and by <= my <= by + bh
                              and nid != d["from"][0]]
                        drop_menu(d["from"], mx, my, on[-1] if on else None)
                    DRAG.clear()
                self.drag = None
                save()
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}
        if EDIT["on"] and event.value == "PRESS" and event.type not in (
                "LEFTMOUSE", "MIDDLEMOUSE", "RIGHTMOUSE", "WHEELUPMOUSE",
                "WHEELDOWNMOUSE", "MOUSEMOVE", "INBETWEEN_MOUSEMOVE",
                "LEFT_CTRL", "RIGHT_CTRL", "LEFT_SHIFT", "RIGHT_SHIFT",
                "LEFT_ALT", "RIGHT_ALT", "OSKEY"):
            type_key(event)
            return {"RUNNING_MODAL"}
        if not inside:
            return {"PASS_THROUGH"}
        if event.type == "MOUSEMOVE":
            h = self._hit(mx, my)
            if h != ui.HOVER[0]:
                ui.HOVER[0] = h
                self.area.tag_redraw()
            return {"PASS_THROUGH"}
        if event.type in ("WHEELUPMOUSE", "WHEELDOWNMOUSE"):
            f = 1.1 if event.type == "WHEELUPMOUSE" else 1 / 1.1
            nz = max(0.04, min(2.5, v[2] * f))
            f = nz / v[2]
            v[0] = mx - (mx - v[0]) * f
            v[1] = my - (my - v[1]) * f
            v[2] = nz
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type in ("X", "DEL") and event.value == "PRESS" and \
                selected() and not EDIT["on"]:
            for nid in selected():
                delete_node(nid)
            st["sel"] = None
            st["selset"] = []
            save()
            return {"RUNNING_MODAL"}
        if event.type in ("HOME", "NUMPAD_PERIOD") and \
                event.value == "PRESS" and not EDIT["on"] and st["nodes"]:
            frame(reg, event.type == "NUMPAD_PERIOD")
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "A" and event.value == "PRESS" and \
                not EDIT["on"]:
            st["selset"] = [] if event.alt else list(st["nodes"])
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "MIDDLEMOUSE" and event.value == "PRESS":
            self.drag = {"kind": "pan", "mx": mx, "my": my}
            return {"RUNNING_MODAL"}
        if MENU and event.value == "PRESS" and event.type in (
                "ESC", "RIGHTMOUSE", "LEFTMOUSE"):
            h = self._hit(mx, my) if event.type == "LEFTMOUSE" else None
            if h and h.startswith("mpick:"):
                menu_pick(int(h.split(":")[1]))
                save()
            elif h != "menu":
                MENU.clear()
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "DOUBLE_CLICK" \
                and EDIT["on"]:
            h = self._hit(mx, my) or ""
            if h == "type:" + EDIT["on"]:
                n_ = N(EDIT["on"])
                k_ = ui.tb_index(h, mx, my)
                if n_ and k_ is not None:
                    a_, b_ = word_at(n_["prompt"], k_)
                    EDIT["anchor"], EDIT["caret"] = a_, b_
                self.drag = None
                self.area.tag_redraw()
                return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            h = self._hit(mx, my) or "bg"
            if EDIT["on"] and h not in ("type:" + EDIT["on"],
                                        "facesbox:" + EDIT["on"]):
                _end_edit()
                save()
            if h == "bg":
                # drag a box to select several nodes, like Blender
                self.drag = {"kind": "box", "mx": mx, "my": my,
                             "shift": event.shift}
                DRAG.update({"kind": "box", "a": (mx, my), "pos": (mx, my)})
            elif h == "pal":
                pass
            elif h.startswith("type:"):
                # click places the caret, Shift+click and dragging select
                nid = h.split(":", 1)[1]
                was = EDIT["on"] == nid and EDIT.get("field") == "prompt"
                st["sel"] = nid
                act(h, context)
                k_ = ui.tb_index(h, mx, my)
                if k_ is not None:
                    if not (event.shift and was):
                        EDIT["anchor"] = k_
                    elif EDIT.get("anchor") is None:
                        EDIT["anchor"] = EDIT["caret"]
                    EDIT["caret"] = k_
                self.drag = {"kind": "tsel", "act": h, "mx": mx, "my": my}
            elif h.startswith("rsz:"):
                nid = h.split(":", 1)[1]
                self.drag = {"kind": "rsz", "nid": nid, "mx": mx, "my": my}
            elif h.startswith("node:") or h.startswith("body:"):
                nid = h.split(":", 1)[1]
                ss = selected()
                if event.shift:
                    if nid in ss:
                        ss.remove(nid)
                    else:
                        ss.append(nid)
                elif nid not in ss:
                    ss = [nid]
                st["selset"] = ss
                st["sel"] = nid if nid in ss else (ss[-1] if ss else None)
                for i in ss:        # to the front
                    st["nodes"][i] = st["nodes"].pop(i)
                self.drag = {"kind": "node", "ids": list(ss) or [nid],
                             "mx": mx, "my": my}
            elif h.startswith("sock:out:"):
                _, _, nid, i = h.split(":")
                self.drag = {"kind": "link", "from": [nid, int(i)],
                             "mx": mx, "my": my, "x0": mx, "y0": my}
                DRAG.update({"kind": "link", "from": [nid, int(i)],
                             "pos": (mx, my)})
            elif h.startswith("sock:in:"):
                _, _, nid, i = h.split(":")
                old = [l for l in st["links"] if l["to"] == [nid, int(i)]]
                if old:
                    # pick the connection up again, like Blender
                    st["links"].remove(old[0])
                    fr = old[0]["from"]
                    self.drag = {"kind": "link", "from": fr, "mx": mx,
                                 "my": my, "x0": mx, "y0": my}
                    DRAG.update({"kind": "link", "from": fr,
                                 "pos": (mx, my)})
            else:
                h2 = h.split(":")
                if len(h2) > 1 and N(h2[1]):
                    st["sel"] = h2[1]
                act(h, context)
            self.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type in ("LEFTMOUSE", "RIGHTMOUSE") and \
                event.value in ("RELEASE", "CLICK"):
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}


class IMG3DCANVAS_OT_pick_image(bpy.types.Operator):
    """Choose a picture"""
    bl_idname = "img3d_canvas.pick_image"
    bl_label = "Choose Image"

    node: StringProperty(options={"HIDDEN"})
    filepath: StringProperty(subtype="FILE_PATH")
    filter_image: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    filter_folder: bpy.props.BoolProperty(default=True, options={"HIDDEN"})

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        n = N(self.node)
        if n and os.path.isfile(self.filepath):
            n["image"] = self.filepath
            deliver(n["id"], 0, self.filepath, "Picture")
            save()
        return {"FINISHED"}


class IMG3DCANVAS_OT_save_image(bpy.types.Operator):
    """Save this picture to a folder of your choice"""
    bl_idname = "img3d_canvas.save_image"
    bl_label = "Save Picture"

    node: StringProperty(options={"HIDDEN"})
    filepath: StringProperty(subtype="FILE_PATH")
    check_existing: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    filter_image: bpy.props.BoolProperty(default=True, options={"HIDDEN"})
    filter_folder: bpy.props.BoolProperty(default=True, options={"HIDDEN"})

    def invoke(self, context, event):
        n = N(self.node)
        src = (n or {}).get("image", "")
        name = os.path.basename(src) or "picture.png"
        pics = os.path.join(os.path.expanduser("~"), "Pictures")
        self.filepath = os.path.join(pics if os.path.isdir(pics) else
                                     os.path.expanduser("~"), name)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        import shutil
        n = N(self.node)
        src = (n or {}).get("image", "")
        if not src or not os.path.isfile(src):
            self.report({"ERROR"}, "The picture is no longer there")
            return {"CANCELLED"}
        dst = bpy.path.abspath(self.filepath)
        ext = os.path.splitext(src)[1].lower() or ".png"
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        if not os.path.splitext(dst)[1]:
            dst += ext
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
        except Exception as exc:
            self.report({"ERROR"}, "Could not save: %s" % exc)
            return {"CANCELLED"}
        self.report({"INFO"}, "Saved " + dst)
        if n:
            n["status"] = "Saved to " + os.path.basename(dst)
        save()
        return {"FINISHED"}


class IMG3DCANVAS_OT_edit_text(bpy.types.Operator):
    """Edit"""
    bl_idname = "img3d_canvas.edit_text"
    bl_label = "Edit"

    target: StringProperty(options={"HIDDEN"})
    value: StringProperty(name="")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        self.layout.label(text="Part name" if self.target.startswith(
            "pname") else "What this part looks like")
        self.layout.prop(self, "value", text="")

    def execute(self, context):
        kind, nid, k = self.target.split(":")
        n = N(nid)
        if n:
            it = n["items"][int(k)]
            if kind == "pname":
                it["name"] = self.value.strip() or "Part"
            else:
                it["description"] = self.value.strip()
        save()
        return {"FINISHED"}


class IMG3DCANVAS_OT_add_part(bpy.types.Operator):
    """Add a part the AI did not list"""
    bl_idname = "img3d_canvas.add_part"
    bl_label = "Add Part"

    node: StringProperty(options={"HIDDEN"})
    name: StringProperty(name="Name", default="")
    description: StringProperty(name="Looks like", default="")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=460)

    def execute(self, context):
        n = N(self.node)
        if n:
            n["items"].append({"name": self.name.strip() or "Part",
                               "description": self.description.strip(),
                               "bbox": None, "on": True})
        save()
        return {"FINISHED"}


class IMG3DCANVAS_OT_open(bpy.types.Operator):
    """Open the Image to 3D Canvas in a new window"""
    bl_idname = "img3d_canvas.open"
    bl_label = "Open Canvas"

    def execute(self, context):
        st = state()
        bpy.ops.wm.window_new()
        win = context.window_manager.windows[-1]
        area = win.screen.areas[0]
        area.ui_type = TREE
        area.spaces.active.node_tree = tree()
        try:
            area.spaces.active.show_region_ui = False
            area.spaces.active.show_region_toolbar = False
        except Exception:
            pass
        if not st["nodes"]:
            add_node("prompt", 0.0, 0.0)
            st["view"] = [PAL_W * dpi() + 60, area.height - 80.0, 1.0]
        save()
        with context.temp_override(window=win, area=area):
            bpy.ops.img3d_canvas.ui("INVOKE_DEFAULT")
        return {"FINISHED"}


class Img3dCanvasTree(bpy.types.NodeTree):
    """Split a picture into parts and build each one in 3D"""
    bl_idname = TREE
    bl_label = "Image to 3D Canvas"
    bl_icon = "IMAGE_DATA"


class IMG3DCANVAS_PT_view3d(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    # its own tab: the Image to 3D tab is painted over by that add-on's
    # card, and a panel there is never seen
    bl_category = "Canvas"
    bl_label = "Image to 3D Canvas"

    def draw(self, context):
        r = self.layout.row()
        r.scale_y = 1.5
        r.operator("img3d_canvas.open", icon="NODETREE")


CLASSES = (Img3dCanvasTree, IMG3DCANVAS_OT_ui, IMG3DCANVAS_OT_pick_image,
           IMG3DCANVAS_OT_edit_text, IMG3DCANVAS_OT_add_part,
           IMG3DCANVAS_OT_open,
           IMG3DCANVAS_OT_save_image)
_HANDLE = [None]


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    if not bpy.app.timers.is_registered(_idle_free):
        bpy.app.timers.register(_idle_free, first_interval=5.0,
                                persistent=True)
    _HANDLE[0] = bpy.types.SpaceNodeEditor.draw_handler_add(
        draw, (), "WINDOW", "POST_PIXEL")


def unregister():
    if bpy.app.timers.is_registered(_idle_free):
        bpy.app.timers.unregister(_idle_free)
    if _HANDLE[0] is not None:
        bpy.types.SpaceNodeEditor.draw_handler_remove(_HANDLE[0], "WINDOW")
        _HANDLE[0] = None
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
    S.clear()
