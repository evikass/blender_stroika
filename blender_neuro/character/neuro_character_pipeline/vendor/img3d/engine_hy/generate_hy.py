"""Image to 3D, Hunyuan3D 2.1 engine. Same protocol as generate.py:

  generate_hy.py --serve        one JSON job per stdin line
  PROGRESS <0..1> <stage> / DONE <glb> / ERROR <msg> / READY
"""

import json
import os
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

sys.path.insert(0, os.path.join(HERE, "hy3dshape"))
sys.path.insert(0, os.path.join(HERE, "hy3dpaint"))
sys.path.insert(0, HERE)

os.environ["HY3DGEN_MODELS"] = MODELS
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["U2NET_HOME"] = os.path.join(MODELS, "u2net")

_T0 = [time.time(), ""]


def say(frac, text):
    if text != _T0[1]:
        print("STAGE %.1f %s" % (time.time() - _T0[0], text), flush=True)
        _T0[1] = text
    print("PROGRESS %.3f %s" % (max(0.0, min(1.0, frac)), text), flush=True)


class Stage:
    lo, hi, name = 0.0, 1.0, ""


def stage(lo, hi, name):
    Stage.lo, Stage.hi, Stage.name = lo, hi, name
    say(lo, name)


def smooth_progress():
    import tqdm as _tq
    import tqdm.auto as _tqa
    base = _tq.tqdm

    class Quiet(base):
        def __init__(self, *a, **k):
            k["disable"] = False
            k["file"] = open(os.devnull, "w")
            super().__init__(*a, **k)

        def update(self, n=1):
            r = super().update(n)
            if self.total:
                f = min(1.0, self.n / float(self.total))
                say(Stage.lo + (Stage.hi - Stage.lo) * f, Stage.name)
            return r

    _tq.tqdm = Quiet
    _tqa.tqdm = Quiet


class no_init:
    """Build networks without filling them with random numbers first; the
    checkpoint overwrites every weight straight after."""
    NAMES = ["uniform_", "normal_", "trunc_normal_", "kaiming_uniform_",
             "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
             "zeros_", "ones_", "constant_", "orthogonal_"]

    def __enter__(self):
        import torch.nn as nn
        self.nn = nn
        self.saved = {n: getattr(nn.init, n) for n in self.NAMES}
        self.classes = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                        nn.Embedding, nn.LayerNorm, nn.GroupNorm]
        self.resets = [(c, c.reset_parameters) for c in self.classes]
        for n in self.NAMES:
            setattr(nn.init, n, lambda t, *a, **k: t)
        for c in self.classes:
            c.reset_parameters = lambda self: None
        return self

    def __exit__(self, *exc):
        for n, fn in self.saved.items():
            setattr(self.nn.init, n, fn)
        for c, fn in self.resets:
            c.reset_parameters = fn
        return False


QUALITY = {"FAST": (25, 256, 6, 512),
           "HIGH": (50, 384, 6, 512),
           "ULTRA": (50, 512, 8, 512)}

_SESSION = {}


def _birefnet():
    """BiRefNet on the graphics card (half precision): the same matting as
    rembg's birefnet-general, but the CPU version of that needs 8 to 11 GB
    of main memory for one picture, more than a 16 GB PC has to spare."""
    if "m" in _SESSION:
        return _SESSION["m"]
    import torch
    from transformers import AutoModelForImageSegmentation
    path = os.path.join(MODELS, "ZhengPeng7", "BiRefNet")
    m = AutoModelForImageSegmentation.from_pretrained(
        path, trust_remote_code=True, local_files_only=True)
    m = m.eval().half().to("cuda")
    _SESSION["m"] = m
    return m


def cut_out(img):
    import numpy as np
    if img.mode == "RGBA":
        a = np.array(img)[:, :, 3]
        if not np.all(a == 255):
            return img
    rgb = img.convert("RGB")
    try:
        import torch
        from PIL import Image
        m = _birefnet()
        x = np.asarray(rgb.resize((1024, 1024), Image.Resampling.BILINEAR),
                       np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
            np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy(x).permute(2, 0, 1)[None].half().cuda()
        with torch.no_grad():
            pred = m(t)[-1].sigmoid().float().cpu()[0, 0].numpy()
        a = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8))
        a = a.resize(rgb.size, Image.Resampling.BILINEAR)
        out = rgb.copy()
        out.putalpha(a)
        return out
    except Exception as exc:
        print("GPU cut-out failed (%s), using rembg" % exc, flush=True)
        _SESSION.pop("m", None)
    from rembg import remove, new_session
    if "s" not in _SESSION:
        _SESSION["s"] = new_session("birefnet-general")
    return remove(rgb, session=_SESSION["s"])


def release_cut_out():
    """the matting model sits on the graphics card; let it go once every
    picture is cut out, so the 3D models have the whole card"""
    if _SESSION:
        _SESSION.clear()
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def to_glb(obj_path, glb_path):
    import numpy as np
    import trimesh
    from PIL import Image
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    folder = os.path.dirname(obj_path)
    stem = os.path.splitext(os.path.basename(obj_path))[0]

    def find(*names):
        for n in names:
            for ext in (".jpg", ".png"):
                p = os.path.join(folder, n + ext)
                if os.path.exists(p):
                    return p
        return None

    albedo = find(stem, stem + "_albedo")
    metal = find(stem + "_metallic")
    rough = find(stem + "_roughness")
    base = Image.open(albedo).convert("RGB") if albedo else None
    mr = None
    if metal or rough:
        size = (base or Image.open(metal or rough)).size
        m = np.array(Image.open(metal).convert("L").resize(size)) if metal \
            else np.zeros(size[::-1], np.uint8)
        r = np.array(Image.open(rough).convert("L").resize(size)) if rough \
            else np.full(size[::-1], 200, np.uint8)
        mr = Image.fromarray(np.stack([np.zeros_like(r), r, m], -1))
    nrm = find(stem + "_normal")
    mat = trimesh.visual.material.PBRMaterial(
        baseColorTexture=base, metallicRoughnessTexture=mr,
        normalTexture=Image.open(nrm).convert("RGB") if nrm else None,
        metallicFactor=1.0 if mr is not None else 0.0,
        roughnessFactor=1.0)
    uv = mesh.visual.uv if hasattr(mesh.visual, "uv") else None
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    mesh.export(glb_path)


def make_conf(views, size, tex):
    from textureGenPipeline import Hunyuan3DPaintConfig
    conf = Hunyuan3DPaintConfig(views, size)
    conf.multiview_pretrained_path = os.path.join(
        MODELS, "tencent", "Hunyuan3D-2.1")
    conf.dino_ckpt_path = os.path.join(MODELS, "facebook", "dinov2-giant")
    # a sharper 4x upscaler for the painted views when it is there
    us = os.path.join(MODELS, "realesrgan", "4x-UltraSharp.safetensors")
    conf.realesrgan_ckpt_path = us if os.path.isfile(us) else \
        os.path.join(MODELS, "realesrgan", "RealESRGAN_x4plus.pth")
    conf.multiview_cfg_path = os.path.join(
        HERE, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
    conf.custom_pipeline = os.path.join(HERE, "hy3dpaint", "hunyuanpaintpbr")
    conf.texture_size = int(tex)
    # bake the views at the texture's own size (the original always baked
    # at 2048, so a 4K texture held only 2K of detail)
    conf.render_size = max(2048, int(tex))
    return conf


def lean_shape_pipeline(model_dir):
    """Hunyuan3D 2.1 shape pipeline that stays small: the weights are read
    straight from the file on disk (no copy in main memory) and each part
    goes on the graphics card only while it works - picture reader, then the
    shape network for the sampling, then the decoder. The stock loader holds
    up to 20 GB of main memory and puts all 7 GB on the card at once."""
    import torch
    import yaml
    from hy3dshape.utils import instantiate_from_config
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline as HP
    cfg = yaml.safe_load(open(os.path.join(model_dir, "config.yaml")))
    ck = torch.load(os.path.join(model_dir, "model.fp16.ckpt"),
                    map_location="cpu", weights_only=True, mmap=True)
    mods = {}
    for name in ("model", "conditioner"):
        with torch.device("meta"):
            m = instantiate_from_config(cfg[name])
        m.load_state_dict(ck[name], strict=True, assign=True)
        mods[name] = m.eval()
    with no_init():
        vae = instantiate_from_config(cfg["vae"])
    vae.load_state_dict(ck["vae"], strict=False)
    mods["vae"] = vae.half().eval()
    pipe = HP.__new__(HP)
    pipe.vae = mods["vae"]
    pipe.model = mods["model"]
    pipe.conditioner = mods["conditioner"]
    pipe.scheduler = instantiate_from_config(cfg["scheduler"])
    pipe.image_processor = instantiate_from_config(cfg["image_processor"])
    pipe.kwargs = {}
    pipe.dtype = torch.float16
    pipe.device = torch.device("cuda")
    home = {"model": ck["model"], "conditioner": ck["conditioner"]}

    def to_card(name):
        m = getattr(pipe, name)
        if next(m.parameters()).device.type != "cuda":
            m.to("cuda")

    def to_home(name):
        # back onto the file on disk: frees the card, costs no memory
        m = getattr(pipe, name)
        if next(m.parameters()).device.type == "cuda":
            m.load_state_dict(home[name], strict=True, assign=True)
            torch.cuda.empty_cache()

    enc = pipe.encode_cond
    exp = pipe._export

    def encode_cond(*a, **k):
        to_home("model")
        to_card("conditioner")
        try:
            return enc(*a, **k)
        finally:
            to_home("conditioner")

    def export(*a, **k):
        to_home("model")
        pipe.vae.to("cuda")
        try:
            return exp(*a, **k)
        finally:
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()

    def model_pre(mod, args):
        if next(mod.parameters()).device.type != "cuda":
            mod.to("cuda")
    pipe.encode_cond = encode_cond
    pipe._export = export
    pipe.model.register_forward_pre_hook(model_pre)
    pipe.vae.to("cpu")

    def to(device=None, dtype=None):
        # park / wake from the engine: parking sends the networks home
        if device is not None and torch.device(device).type == "cpu":
            to_home("model")
            to_home("conditioner")
            pipe.vae.to("cpu")
            torch.cuda.empty_cache()
    pipe.to = to
    return pipe



def no_gpu_reason():
    """Why CUDA is missing, in words the user can act on: most often an
    NVIDIA driver too old for this CUDA (13.0 needs driver 580 or newer)."""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=15,
                           creationflags=0x08000000 if os.name == "nt" else 0)
        line = (r.stdout or "").strip().splitlines()[0]
        name, ver = [x.strip() for x in line.split(",")[:2]]
        if float(ver.split(".")[0]) < 580:
            return ("Your NVIDIA driver (%s) is too old for the %s. Update "
                    "it to version 580 or newer from nvidia.com, restart the "
                    "PC and try again" % (ver, name))
        return ("The %s cannot be used right now (driver %s). Restart the "
                "PC and try again" % (name, ver))
    except Exception:
        return "No NVIDIA GPU found"

def mv_names():
    """Hunyuan3D-2mv's settings file names the 2.0 code (hy3dgen...), the
    engine runs the 2.1 code (hy3dshape...). Answer the old names with the
    new classes, so the downloaded file stays exactly as Tencent made it."""
    import types
    if "hy3dgen.shapegen.models" in sys.modules:
        return
    from hy3dshape.models.denoisers.hunyuan3ddit import Hunyuan3DDiT
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.models.conditioner import SingleImageEncoder
    from hy3dshape.schedulers import FlowMatchEulerDiscreteScheduler
    from hy3dshape.preprocessors import MVImageProcessorV2
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    names = {
        "hy3dgen": {},
        "hy3dgen.shapegen": {},
        "hy3dgen.shapegen.models": {
            "Hunyuan3DDiT": Hunyuan3DDiT, "ShapeVAE": ShapeVAE,
            "SingleImageEncoder": SingleImageEncoder},
        "hy3dgen.shapegen.schedulers": {
            "FlowMatchEulerDiscreteScheduler":
                FlowMatchEulerDiscreteScheduler},
        "hy3dgen.shapegen.preprocessors": {
            "MVImageProcessorV2": MVImageProcessorV2},
        "hy3dgen.shapegen.pipelines": {
            "Hunyuan3DDiTFlowMatchingPipeline":
                Hunyuan3DDiTFlowMatchingPipeline},
    }
    for name, attrs in names.items():
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.__dict__.update(attrs)
        sys.modules[name] = mod


class Engine:
    def __init__(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(no_gpu_reason())
        smooth_progress()
        try:
            from torchvision_fix import apply_fix
            apply_fix()
        except Exception:
            pass
        self.torch = torch
        self.shape = None
        self.shape_mv = None
        self.paint = None
        self.paint_key = None

    def load_shape(self, mv=False):
        """mv: the multi-view shape model (Hunyuan3D-2mv), which reads
        front / left / back / right pictures together."""
        self.park_shape()
        attr = "shape_mv" if mv else "shape"
        pipe = getattr(self, attr, None)
        if pipe is None:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
            with no_init():
                if mv:
                    mv_names()
                    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        "tencent/Hunyuan3D-2mv",
                        subfolder="hunyuan3d-dit-v2-mv",
                        use_safetensors=True, variant="fp16")
                else:
                    pipe = lean_shape_pipeline(os.path.join(
                        MODELS, "tencent", "Hunyuan3D-2.1",
                        "hunyuan3d-dit-v2-1"))
            setattr(self, attr, pipe)
        else:
            pipe.to("cuda")
        return pipe

    def park(self):
        self.park_shape()
        if self.paint is not None:
            try:
                import hy_fast
                hy_fast.gpu_park(self.paint, "cpu")
            except Exception:
                pass
            for m_ in getattr(self.paint, "models", {}).values():
                if m_ is None:
                    continue
                for a_ in ("dino_v2",):
                    d_ = getattr(m_, a_, None)
                    if d_ is not None:
                        try:
                            d_.to("cpu")
                        except Exception:
                            pass
        kp = getattr(self, "klein", None)
        if kp is not None and kp.poll() is None:
            pass   # Klein parks itself after every enhance
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()

    def park_shape(self):
        for attr in ("shape", "shape_mv"):
            p = getattr(self, attr, None)
            if p is not None:
                p.to("cpu")
        self.torch.cuda.empty_cache()

    def load_paint(self, views, size, tex):
        key = (views, size, tex)
        if self.paint is not None and self.paint_key == key and \
                not getattr(self.paint, "_spent", False):
            return self.paint
        self.paint = None
        self.torch.cuda.empty_cache()
        from textureGenPipeline import Hunyuan3DPaintPipeline
        conf = make_conf(views, size, tex)
        with no_init():
            self.paint = Hunyuan3DPaintPipeline(conf)
        import textureGenPipeline as tg
        import hy_fast
        hy_fast.install(tg)
        # an 8 GB card / 16 GB PC: the painter's big weights in 8 bits
        try:
            if hy_fast.low_memory():
                un_ = self.paint.models["multiview_model"].pipeline.unet
                print("8-bit paint weights: %d layers" %
                      hy_fast.fp8_weights(un_), flush=True)
        except Exception as exc:
            print("no 8-bit weights: %s" % exc, flush=True)
        # everything waits in main memory until its turn (the pipeline puts
        # all of it on the card when it is made, 6.6 GB)
        try:
            hy_fast.gpu_park(self.paint, "cpu")
            mv_ = self.paint.models["multiview_model"]
            hy_fast._offload(mv_)
            d_ = getattr(mv_, "dino_v2", None)
            if d_ is not None:
                d_.to("cpu")
            self.torch.cuda.empty_cache()
        except Exception as exc:
            print("paint park: %s" % exc, flush=True)
        self.paint_key = key
        return self.paint

    def enhance(self, work, out, texture=4096, passes=1, sharpen=0.0,
                ai=True, mode="HIGH", creativity=0.0, **_):
        """Better texture for a model already made: bake its kept views
        again, upscaled and sharpened more."""
        t0 = time.time()
        _T0[0] = t0
        stage(0.02, 0.05, "Enhancing texture")
        # the painting networks are not needed: free the card for the bake
        self.paint = None
        self.park_shape()
        self.torch.cuda.empty_cache()
        import hy_fast
        vdir = os.path.join(work, "views")
        # Enhance Texture runs in its own environment (newer diffusers)
        air_py = os.path.join(os.path.dirname(HERE), "env_klein", "Scripts",
                              "python.exe")
        if not os.path.isfile(air_py) and os.environ.get("AIR_ROOT"):
            air_py = os.path.join(os.environ["AIR_ROOT"], "env", "Scripts",
                                  "python.exe")
        def _fresh():
            # the views were already enhanced (same picture, same seed):
            # use them again instead of repainting
            try:
                c_ = json.load(open(os.path.join(vdir, "cams.json")))
                mp_ = os.path.join(vdir, "albedo_ai_mode.txt")
                if not os.path.isfile(mp_) or \
                        open(mp_).read().strip() != "%s %.2f" % (
                            mode, float(creativity)):
                    return False
                for i_ in range(c_["n"]):
                    a_ = os.path.join(vdir, "albedo_ai_%d.png" % i_)
                    b_ = os.path.join(vdir, "albedo_%d.png" % i_)
                    if not os.path.isfile(a_) or \
                            os.path.getmtime(a_) < os.path.getmtime(b_):
                        return False
                return True
            except Exception:
                return False
        if ai and os.path.isfile(air_py) and not _fresh():
            # Klein repaints every side view with real detail; it is loaded
            # once and waits in main memory between enhances
            import subprocess
            cams_ = json.load(open(os.path.join(vdir, "cams.json")))
            n = cams_["n"]
            stage(0.05, 0.75, "Enhancing views")
            kp = getattr(self, "klein", None)
            if kp is None or kp.poll() is not None:
                kp = subprocess.Popen(
                    [air_py, "-u", os.path.join(HERE, "klein_views.py"),
                     "--serve"],
                    cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, creationflags=0x08000000)
                self.klein = kp
                for line in kp.stdout:
                    if line.startswith("READY"):
                        break
            kp.stdin.write("%s\t%s\t%.2f\n" % (vdir, mode,
                                                float(creativity)))
            kp.stdin.flush()
            done = 0
            err = None
            for line in kp.stdout:
                line = line.rstrip()
                if line.startswith("VIEW "):
                    done += 1
                    print(line, flush=True)
                    say(0.05 + 0.70 * done / max(1, n), "Enhancing views")
                elif line.startswith("LOADED"):
                    print(line, flush=True)
                elif line.startswith("ERROR"):
                    err = line
                    print(line, flush=True)
                elif line.startswith("KLEIN DONE"):
                    break
            if err or done == 0:
                raise RuntimeError("Klein could not enhance the views")
        stage(0.75, 0.92, "Baking texture")
        conf = make_conf(6, 512, texture)
        # sharper bake: each texel takes its colour mostly from the view
        # that faces it best, instead of an average that blurs fine detail
        conf.bake_exp = 12
        obj = os.path.join(work, "enhanced", "textured.obj")
        os.makedirs(os.path.dirname(obj), exist_ok=True)
        hy_fast.enhance(conf, vdir, obj, passes=passes, sharpen=sharpen,
                        ai=ai, tex_only=True)
        stage(0.92, 0.99, "Saving")
        # the model in the scene keeps its mesh: only the pictures go back
        out = os.path.splitext(obj)[0] + ".jpg"
        print("WORK " + work, flush=True)
        print("TIME %.1f" % (time.time() - t0), flush=True)
        print("DONE " + os.path.abspath(out), flush=True)

    def enhance_any(self, work, mesh, albedo, mr=None, texture=4096, **kw):
        """Enhance Texture for a model this engine did not paint (TRELLIS,
        Pixal3D, ...): its own texture becomes the views, then the normal
        enhance runs."""
        _T0[0] = time.time()
        stage(0.0, 0.05, "Reading the model")
        self.paint = None
        self.park_shape()
        self.torch.cuda.empty_cache()
        import hy_fast
        conf = make_conf(6, 512, texture)
        hy_fast.prep_views(conf, os.path.join(work, "views"), mesh, albedo,
                           mr, size=min(2048, conf.render_size))
        self.enhance(work=work, out=None, texture=texture, **kw)

    def run(self, image, out, quality="HIGH", faces=40000, texture=2048,
            seed=0, views=None, mesh=None, **_):
        torch = self.torch
        given = mesh
        from PIL import Image
        t0 = time.time()
        _T0[0] = t0
        extra = views
        steps, octree, views, vsize = QUALITY.get(quality, QUALITY["HIGH"])

        stage(0.02, 0.05, "Removing background")
        img = Image.open(image)
        img.load()
        img = cut_out(img).convert("RGBA")

        # more pictures of the same object: {"back"/"left"/"right": path}
        sides = {}
        for side, pth in (extra or {}).items() if isinstance(extra, dict) \
                else []:
            try:
                im = Image.open(pth)
                im.load()
                sides[side] = cut_out(im).convert("RGBA")
            except Exception as exc:
                print("view skipped: %s (%s)" % (pth, exc), flush=True)

        release_cut_out()
        if given:
            # the shape was made elsewhere (Hi3D): only paint it
            import trimesh as _tm
            mesh = _tm.load(given, force="mesh", process=False)
            print("PAINT ONLY %d faces" % len(mesh.faces), flush=True)
        else:
            mesh = self._make_shape(img, sides, quality, steps, octree, seed)
        if mesh is None:
            return
        return self._finish(mesh, img, out, faces, texture, views, vsize, t0)

    def _make_shape(self, img, sides, quality, steps, octree, seed):
        torch = self.torch
        stage(0.05, 0.10, "Loading shape model")
        shape = self.load_shape(mv=bool(sides))
        stage(0.10, 0.40, "Building shape")
        gen = torch.Generator("cuda").manual_seed(int(seed))
        with torch.no_grad():
            if sides:
                print("VIEWS %d (Hunyuan multi-view: front, %s)" % (
                    1 + len(sides), ", ".join(sides)), flush=True)
                mv_steps = {"FAST": 20, "HIGH": 30, "ULTRA": 50}.get(
                    quality, 30)
                mesh = shape(image=dict(front=img, **sides),
                             num_inference_steps=mv_steps,
                             octree_resolution=octree, num_chunks=20000,
                             generator=gen)[0]
            else:
                mesh = shape(image=img, num_inference_steps=steps,
                             octree_resolution=octree, generator=gen)[0]
        try:
            from hy3dshape.postprocessors import FloaterRemover, \
                DegenerateFaceRemover
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
        except Exception:
            pass
        self.park_shape()
        return mesh

    def _finish(self, mesh, img, out, faces, texture, views, vsize, t0):
        torch = self.torch
        if int(texture) <= 0:
            # shape only: no texture painted
            stage(0.90, 0.99, "Saving")
            if len(mesh.faces) > int(faces):
                mesh = mesh.simplify_quadric_decimation(face_count=int(faces))
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            mesh.export(out)
            print("TIME %.1f" % (time.time() - t0), flush=True)
            print("DONE " + os.path.abspath(out), flush=True)
            return

        work = tempfile.mkdtemp(prefix="hy3d_")
        white = os.path.join(work, "white.glb")
        mesh.export(white)
        style = os.path.join(work, "style.png")
        img.save(style)

        stage(0.42, 0.50, "Loading texture model")
        paint = self.load_paint(views, vsize, texture)
        import trimesh

        def simplify(i, o, target_count=40000):
            m = trimesh.load(i, force="mesh")
            if len(m.faces) > int(faces):
                m = m.simplify_quadric_decimation(face_count=int(faces))
            m.export(o)
        import textureGenPipeline as tg
        tg.remesh_mesh = lambda i, o: simplify(i, o)

        stage(0.50, 0.92, "Painting texture")
        obj = os.path.join(work, "textured.obj")
        with torch.no_grad():
            paint(mesh_path=white, image_path=style, output_mesh_path=obj,
                  use_remesh=True, save_glb=False)
        torch.cuda.empty_cache()

        stage(0.94, 0.99, "Saving")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        to_glb(obj, out)
        print("WORK " + work, flush=True)
        print("TIME %.1f" % (time.time() - t0), flush=True)
        print("DONE " + os.path.abspath(out), flush=True)


def nice(exc):
    msg = str(exc).strip().splitlines()[-1] if str(exc).strip() \
        else type(exc).__name__
    if "out of memory" in msg.lower():
        msg = "Not enough GPU memory - try Quality: Fast"
    return msg[:200]


def serve():
    eng = None
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        try:
            job = json.loads(line)
            if job.get("task") == "park":
                # another engine needs the graphics card: everything waits
                # in main memory, ready for the next job
                if eng is not None:
                    eng.park()
                print("PARKED", flush=True)
                continue
            if eng is None:
                say(0.01, "Starting")
                eng = Engine()
            task_ = job.pop("task", None)
            if task_ == "enhance":
                eng.enhance(**job)
            elif task_ == "enhance_any":
                eng.enhance_any(**job)
            else:
                eng.run(**job)
        except Exception as exc:
            traceback.print_exc()
            print("ERROR " + nice(exc), flush=True)
            try:
                eng.torch.cuda.empty_cache()
            except Exception:
                pass
        print("READY", flush=True)


if __name__ == "__main__":
    try:
        if "--serve" in sys.argv:
            serve()
        else:
            import argparse
            ap = argparse.ArgumentParser()
            ap.add_argument("--image")
            ap.add_argument("--out")
            ap.add_argument("--quality", default="HIGH")
            ap.add_argument("--faces", type=int, default=40000)
            ap.add_argument("--texture", type=int, default=2048)
            ap.add_argument("--seed", type=int, default=0)
            a = ap.parse_args()
            say(0.01, "Starting")
            Engine().run(a.image, a.out, a.quality, a.faces, a.texture,
                         a.seed)
    except Exception as exc:
        traceback.print_exc()
        print("ERROR " + nice(exc), flush=True)
        sys.exit(1)
