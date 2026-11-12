"""Image to 3D engine.

Started by the Blender add-on as its own process, so Blender never loads
torch and a Cancel is just ending this process.

Two ways to run it:
  generate.py --image a.png --out a.glb ...   one model, then exit
  generate.py --serve                          stay up; one JSON job per
                                               stdin line, models kept in
                                               RAM between jobs

Progress goes to stdout as lines the add-on reads:
    PROGRESS <0..1> <stage words>
    DONE <path to glb>
    ERROR <message>
    READY            (serve mode: waiting for the next job)
"""

import argparse
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

sys.path.insert(0, os.path.join(HERE, "stubs"))
sys.path.insert(0, HERE)

# o_voxel (TRELLIS.2) asks for nvdiffrast, whose licence forbids commercial
# use: it gets the add-on's own rasterizer instead, nvdiffrast is not used
import img3d_raster  # noqa: E402
img3d_raster.install_as_nvdiffrast()

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _gpu_cc():
    try:
        import subprocess as _sp
        r = _sp.run(["nvidia-smi", "--query-gpu=compute_cap",
                     "--format=csv,noheader"], capture_output=True, text=True,
                    timeout=15, creationflags=0x08000000 if os.name == "nt" else 0)
        return max(float(x) for x in r.stdout.split() if x.strip())
    except Exception:
        return 99.0


# flash-attn runs on RTX 30 and newer only; older cards (RTX 20, GTX 16)
# use PyTorch's own attention, a little slower, same result
_ATT = "flash_attn" if (_gpu_cc() >= 8.0 and not os.environ.get("IMG3D_NO_FLASH")) \
    else "sdpa"
os.environ["ATTN_BACKEND"] = _ATT
os.environ["SPARSE_ATTN_BACKEND"] = _ATT
os.environ["SPARSE_CONV_BACKEND"] = "flex_gemm"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["U2NET_HOME"] = os.path.join(MODELS, "u2net")
os.environ["TRITON_CACHE_DIR"] = os.path.join(ROOT, "triton_cache")


def _triton_works():
    """Triton builds a small helper for each kernel with its own C compiler
    (TinyCC). Windows with "Mandatory ASLR" forced on refuses to load those
    helpers (WinError 623). Try one tiny kernel; when it fails, the sparse
    convolutions run on PyTorch's own matrix code instead: same result, no
    compiler needed, a little slower."""
    if os.environ.get("IMG3D_NO_TRITON"):
        print("Triton turned off: sparse convolution on PyTorch", flush=True)
        return False
    try:
        import torch
        import triton
        import triton.language as tl

        @triton.jit
        def _probe(x_ptr, n, B: tl.constexpr):
            i = tl.program_id(0) * B + tl.arange(0, B)
            m = i < n
            tl.store(x_ptr + i, tl.load(x_ptr + i, mask=m) + 1, mask=m)

        x = torch.zeros(16, device="cuda")
        _probe[(1,)](x, 16, B=16)
        torch.cuda.synchronize()
        return bool((x == 1).all())
    except Exception as exc:
        print("Triton can't run here (%s): sparse convolution on PyTorch"
              % str(exc).splitlines()[0][:200] if str(exc) else exc, flush=True)
        return False


_TRITON = None


def _grid_sample_without_triton():
    """The texture bake reads colours out of the voxels through one more
    Triton kernel (a weighted sum of 8 neighbours). Where Triton can't run,
    the same sum is done here with PyTorch: identical numbers, no compiler."""
    import torch
    import flex_gemm.kernels.triton as fk

    if getattr(fk, "_img3d_torch_sum", False):
        return

    def indice_weighed_sum_fwd(input, indices, weight):
        M, V = indices.shape
        C = input.shape[1]
        out = torch.empty((M, C), device=input.device, dtype=input.dtype)
        step = max(1, 16_000_000 // max(1, V * C))
        for a in range(0, M, step):
            idx = indices[a:a + step].long()
            bad = (idx < 0) | (idx == 0xffffffff)
            idx = idx.masked_fill(bad, 0)
            w = weight[a:a + step].float().masked_fill(bad, 0.0)
            f = input.index_select(0, idx.reshape(-1)).float()
            out[a:a + step] = (f.view(-1, V, C) * w.unsqueeze(-1)) \
                .sum(1).to(input.dtype)
        return out

    fk.indice_weighed_sum_fwd = indice_weighed_sum_fwd
    fk._img3d_torch_sum = True

import folder_paths  # noqa: E402  (the stub)
folder_paths.models_dir = MODELS


_T0 = [time.time(), ""]


_LAST = [0.0]


def say(frac, text):
    # the bar only moves forward within a model
    frac = max(float(frac), _LAST[0])
    _LAST[0] = frac
    if text != _T0[1]:
        print("STAGE %.1f %s" % (time.time() - _T0[0], text), flush=True)
        _T0[1] = text
    print("PROGRESS %.3f %s" % (max(0.0, min(1.0, frac)), text), flush=True)


QUALITY = {"FAST": "512", "HIGH": "1024_cascade", "ULTRA": "1536_cascade"}



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

class Stage:
    lo = 0.0
    hi = 1.0
    name = ""
    queue = []      # (name, lo, hi) for each sampling loop still to come


class Bar:
    """Turns the pipeline's step count into a bar for the add-on."""

    def __init__(self, lo, hi, total, names):
        self.lo, self.hi, self.total, self.n = lo, hi, total, 0
        self.names = names

    def update(self, k=1):
        self.n += k
        name = self.names[min(self.n, len(self.names) - 1)]
        step = (self.hi - self.lo) / float(self.total)
        Stage.lo = self.lo + step * self.n
        Stage.hi = Stage.lo + step
        Stage.name = name
        say(Stage.lo, name)


def smooth_progress():
    """Every sampler step moves the bar, not only the end of a stage."""
    import tqdm as _tq
    import tqdm.auto as _tqa

    base = _tq.tqdm

    class Quiet(base):
        def __init__(self, *a, **k):
            # a new sampling loop: it takes the next part of the bar, or
            # carries on from where the bar is instead of starting over
            if Stage.queue:
                Stage.name, Stage.lo, Stage.hi = Stage.queue.pop(0)
            else:
                Stage.lo = min(max(Stage.lo, _LAST[0]), Stage.hi)
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


def ram_gb():
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
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
        return m.ullTotalPhys / 1e9
    except Exception:
        return 0.0


class Engine:
    """The pipeline, built once. With `keep` on, every network that has been
    loaded stays in system RAM between runs and only visits the GPU for its
    own stage, so a second model skips all the loading."""

    def __init__(self, keep):
        import torch
        import torch.nn as nn
        if not torch.cuda.is_available():
            raise RuntimeError(no_gpu_reason())
        smooth_progress()
        from trellis2.modules.attention import config as attn_config
        from trellis2.modules.sparse import config as sparse_config
        attn_config.set_backend(_ATT)
        sparse_config.set_attn_backend(_ATT)
        sparse_config.set_conv_backend("flex_gemm")
        global _TRITON
        if _TRITON is None:
            _TRITON = _triton_works()
        if not _TRITON:
            from trellis2.modules.sparse.conv import config as conv_config
            conv_config.FLEX_GEMM_ALGO = "explicit_gemm"
            _grid_sample_without_triton()
        from trellis2 import models
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        self.keep = keep
        self.cache = {}
        names = ["uniform_", "normal_", "trunc_normal_", "kaiming_uniform_",
                 "kaiming_normal_", "xavier_uniform_", "xavier_normal_",
                 "zeros_", "ones_", "constant_", "orthogonal_"]
        classes = [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding,
                   nn.LayerNorm, nn.GroupNorm]
        orig = models.from_pretrained
        cache = self.cache

        def fast(path, **kw):
            if keep and path in cache:
                return cache[path]
            saved = {n: getattr(nn.init, n) for n in names}
            resets = [(c, c.reset_parameters) for c in classes]
            for n in names:
                setattr(nn.init, n, lambda t, *a, **k: t)
            for c in classes:
                c.reset_parameters = lambda self: None
            try:
                # straight onto the graphics card, one weight at a time
                # (faster on every PC, and half the main memory)
                # (the one-weight-at-a-time load gave worse textures on
                # Pixal3D, so it is off unless asked for)
                m = _lean_model(path, kw) if os.environ.get("IMG3D_LEAN") \
                    else None
                if m is None:
                    m = orig(path, **kw)
            finally:
                for n, fn in saved.items():
                    setattr(nn.init, n, fn)
                for c, fn in resets:
                    c.reset_parameters = fn
            if keep:
                cache[path] = m
            return m

        models.from_pretrained = fast
        self.pipes = {}
        self.Pipe = Trellis2ImageTo3DPipeline
        self.torch = torch

    def pipe(self, fp8):
        p = self.pipes.get(fp8)
        if p is not None:
            return p
        if fp8:
            p = self.Pipe.from_pretrained(
                os.path.join(MODELS, "visualbruno", "TRELLIS.2-4B-FP8"),
                keep_models_loaded=False, use_fp8=True)
        else:
            p = self.Pipe.from_pretrained(
                os.path.join(MODELS, "microsoft", "TRELLIS.2-4B"),
                keep_models_loaded=False)
        p.low_vram = True
        p.cuda()
        self._lean_decode(p)
        if self.keep:
            self._keep_cond(p)
        self.pipes[fp8] = p
        return p

    def _keep_cond(self, p):
        torch = self.torch
        cache = self.cache
        held = {"cond": None}
        load_cond, clean = p.load_image_cond_model, p._cleanup_cuda

        def load_image_cond_model():
            if p.image_cond_model is None and held["cond"] is not None:
                p.image_cond_model = held["cond"]
                p.image_cond_model.to(p._device)
                return
            load_cond()
            held["cond"] = p.image_cond_model

        def cleanup():
            live = set(id(m) for m in p.models.values() if m is not None)
            if p.image_cond_model is not None:
                live.add(id(p.image_cond_model))
            for m in list(cache.values()) + ([held["cond"]] if held["cond"] else []):
                if id(m) not in live:
                    try:
                        m.to("cpu")
                    except Exception:
                        pass
            clean()

        p.load_image_cond_model = load_image_cond_model
        p._cleanup_cuda = cleanup

    def pixal(self, mv=False):
        """Pixal3D: TRELLIS.2 trained to keep the picture's exact view (it
        reads the camera from the picture first). mv: its multi-view
        networks, for several pictures of the object from known sides."""
        key = "pixal_mv" if mv else "pixal"
        p = self.pipes.get(key)
        if p is not None:
            return p
        _natten_patch()
        p = self.Pipe.from_pretrained(
            os.path.join(MODELS, "TencentARC", "Pixal3D"),
            keep_models_loaded=False, isPixal3D=True, isPixal3DMV=bool(mv))
        p.low_vram = True
        p.cuda()
        # the 1024 stages read the picture as a 1024 x 1024 feature map with
        # 1024 channels (4 GB on its own), which overflows a 12 GB card -
        # worse with several views. Built in tiles instead: same resolution,
        # a quarter of the peak memory
        # single picture: the original 768 map - tiles here blur the texture
        try:
            p.PIXAL3D_IMAGE_COND_CONFIGS["tex_1024"]["naf_target_size"] = 768
        except Exception:
            pass
        single_ = ("load_pixal3d_image_cond_tex_1024",
                   "load_pixal3d_image_cond_shape_1024",
                   "load_pixal3d_image_cond_shape_512")
        for nm in (single_ if os.environ.get("IMG3D_TILE") else ()) + (
                   "load_pixal3d_mv_image_cond_shape_512",
                   "load_pixal3d_mv_image_cond_tex_1024",
                   "load_pixal3d_mv_image_cond_shape_1024"):
            orig = getattr(p, nm, None)
            if orig is None:
                continue

            def tiled(*a, _o=orig, _n=nm, **k):
                mdl = _o(*a, **k)
                try:
                    # several views at 1024 need smaller tiles
                    f_ = (8 if _n == "load_pixal3d_mv_image_cond_tex_1024"
                          else 4 if ("_mv_" in _n and "1024" in _n) else 2)
                    if "1024" in _n:
                        # smaller tiles: bigger ones overflow even a 12 GB
                        # card into system memory, which crawls
                        f_ *= 2
                    mdl.naf_tile_factor = f_
                except Exception:
                    pass
                return mdl
            setattr(p, nm, tiled)
        if self.keep:
            # networks kept in RAM between models must also leave the GPU
            # when a stage is done, or they pile up there
            self._keep_cond(p)
        self._lean_decode(p)
        if mv:
            make_pixal_mv(p)
            _half_mv_features(p)
        self.pipes[key] = p
        return p

    def _lean_decode(self, p):
        """Decoding a big model holds its whole surface (tens of millions
        of faces) on the GPU while the texture decoder runs, which overflows
        a 12 GB card into system memory and crawls. Park the surface in RAM
        while the texture is decoded."""
        import types
        from trellis2.representations import MeshWithVoxel
        eng = self

        def decode_latent(self_, shape_slat, tex_slat, resolution,
                          use_tiled=True):
            import torch as _t
            # the card holds only the shape decoder and the shape latent
            if tex_slat is not None:
                tex_slat = tex_slat.cpu()
            eng.free_gpu(self_)
            print("MEM before shape decode %.2f GB" % (
                _t.cuda.memory_allocated() / 2**30), flush=True)
            meshes, subs = self_.decode_shape_slat(shape_slat, resolution,
                                                   use_tiled=use_tiled)
            if tex_slat is not None:
                tex_slat = tex_slat.cuda()
            if tex_slat is None:
                out = []
                for m in meshes:
                    m.fill_holes()
                    out.append(MeshWithVoxel(
                        m.vertices, m.faces, origin=[-0.5, -0.5, -0.5],
                        voxel_size=1 / resolution, coords=None, attrs=None,
                        voxel_shape=None, layout=self_.pbr_attr_layout))
                return out
            meshes = [m.cpu() for m in meshes]
            eng.free_gpu(self_)
            tex_voxels = self_.decode_tex_slat(tex_slat, subs)
            del subs
            eng.free_gpu(self_)
            out = []
            for m, v in zip(meshes, tex_voxels):
                m = m.cuda()
                m.fill_holes()
                out.append(MeshWithVoxel(
                    m.vertices, m.faces, origin=[-0.5, -0.5, -0.5],
                    voxel_size=1 / resolution, coords=v.coords[:, 1:],
                    attrs=v.feats,
                    voxel_shape=torch_size(v),
                    layout=self_.pbr_attr_layout))
            return out

        def torch_size(v):
            import torch
            return torch.Size([*v.shape, *v.spatial_shape])
        p.decode_latent = types.MethodType(decode_latent, p)
        if small_pc() or _TRITON is False:
            try:
                _slab_conv_patch()
            except Exception as exc_:
                print("slab conv patch failed: %s" % exc_, flush=True)
            # small card: every block of the decoders works in pieces
            for nm_ in ("load_shape_slat_decoder", "load_tex_slat_decoder"):
                orig_ = getattr(p, nm_, None)
                if orig_ is None:
                    continue

                def lean_load(*a, _o=orig_, _n=nm_, **k):
                    r_ = _o(*a, **k)
                    key = "shape_slat_decoder" if "shape" in _n else \
                        "tex_slat_decoder"
                    mdl = p.models.get(key)
                    if mdl is not None:
                        for sub in mdl.modules():
                            if hasattr(sub, "low_vram"):
                                sub.low_vram = True
                            if hasattr(sub, "chunk_size"):
                                sub.chunk_size = 16384
                    return r_
                setattr(p, nm_, lean_load)

    def free_gpu(self, pipe):
        import gc
        torch = self.torch
        mods = [m for m in getattr(pipe, "models", {}).values() if m is not None]
        if getattr(pipe, "image_cond_model", None) is not None:
            mods.append(pipe.image_cond_model)
            # it is loaded (and put on the GPU) again when next needed
            pipe.image_cond_model = None
        mods += list(self.cache.values())
        for m in mods:
            try:
                m.to("cpu")
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    def texture_mesh(self, mesh, image, out, views=None, texture=2048, seed=0,
                     steps=8, **_):
        """Paint a model made elsewhere (Hunyuan) with Pixal3D, reading the
        colours from every picture (front + back / left / right)."""
        import math
        import numpy as np
        import trimesh
        from PIL import Image
        t0 = time.time()
        _T0[0] = t0
        _LAST[0] = 0.0
        Stage.queue = []
        say(0.02, "Reading pictures")
        m = trimesh.load(mesh, force="mesh", process=False)
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        ext = m.bounds[1] - m.bounds[0]
        tall = float(ext[1] / ext.max())
        # every picture is scaled so the object is as tall in it as it is in
        # the model, and centred; the cameras then see the model exactly the
        # way the pictures show it
        C = 1024
        P_ = C / 1.1

        def frame(p):
            im = cut_out(Image.open(p)).convert("RGBA")
            a = np.array(im)[:, :, 3]
            ys, xs = np.nonzero(a > 128)
            im = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            k = tall * P_ / im.height
            im = im.resize((max(1, round(im.width * k)),
                            max(1, round(im.height * k))),
                           Image.Resampling.LANCZOS)
            cv = Image.new("RGBA", (C, C), (0, 0, 0, 0))
            cv.paste(im, ((C - im.width) // 2, (C - im.height) // 2), im)
            return cv

        az_of = {"back": 180.0, "left": 90.0, "right": 270.0}
        ims, azs = [frame(image)], [0.0]
        for side, pth in (views or {}).items():
            try:
                ims.append(frame(pth))
                azs.append(az_of.get(side, 180.0))
            except Exception as exc:
                print("view skipped: %s (%s)" % (pth, exc), flush=True)
        # the matting model sits on the graphics card: let it go before the
        # big networks come in
        _SESSION.clear()
        import gc
        gc.collect()
        self.torch.cuda.empty_cache()
        say(0.08, "Loading model")
        for k_, q in self.pipes.items():
            if k_ != "pixal_mv":
                self.free_gpu(q)
        pipe = self.pixal(mv=True)
        from trellis2.utils.mv_camera import (build_views_from_angles,
                                              camera_distance_for_extent)
        fov = math.radians(20.0)
        dist = camera_distance_for_extent(fov, (512 / 1.1) / 2.0, 1.0, 512)
        bundle = build_views_from_angles(ims, azs, [0.0] * len(ims),
                                         camera_angle_x=fov, mesh_scale=1.0,
                                         distance=dist)
        Stage.lo, Stage.hi, Stage.name = 0.10, 0.85, "Painting texture"
        say(0.10, "Painting texture")
        with self.torch.no_grad():
            res = pipe.texture_mesh_pixal3d(
                m, views=bundle, seed=int(seed), resolution=1024,
                texture_size=int(texture), mesh_orientation="auto",
                tex_slat_sampler_params={"steps": int(steps)})
        om = res[0] if isinstance(res, (list, tuple)) else res
        self.free_gpu(pipe)
        say(0.95, "Saving")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        om.export(out)
        print("TIME %.1f" % (time.time() - t0), flush=True)
        print("DONE " + os.path.abspath(out), flush=True)

    def run(self, image, out, quality="HIGH", faces=200000, texture=2048,
            seed=0, steps=8, fp8=False, views=None, engine="TRELLIS",
            mv_swap=False, mv_framing="crop"):
        torch = self.torch
        from PIL import Image
        t0 = time.time()
        _T0[0] = t0
        _LAST[0] = 0.0
        Stage.queue = []
        if quality == "ULTRA" and small_pc():
            # Ultra's last decoding step alone needs more than an 8 GB card
            quality = "HIGH"
            print("NOTE Ultra needs a 12 GB graphics card - made with High",
                  flush=True)
            say(0.01, "Ultra needs 12 GB VRAM - using High")
        say(0.02, "Removing background")
        img = Image.open(image)
        img.load()
        img = cut_out(img)

        say(0.08, "Loading model")
        pix = engine == "PIXAL3D"
        # views: {"back": path, "left": path, "right": path} (or a plain list)
        named = views if isinstance(views, dict) else {}
        if isinstance(views, dict):
            views = list(views.values())
        pix_mv = pix and bool(named)
        for q in self.pipes.values():
            self.free_gpu(q)
        if not fp8 and small_pc() and os.path.isdir(os.path.join(
                MODELS, "visualbruno", "TRELLIS.2-4B-FP8")):
            # a small PC: the 8-bit TRELLIS networks (half the memory)
            fp8 = True
        if pix:
            pipe = self.pixal(mv=pix_mv)
        else:
            pipe = self.pipe(bool(fp8))
        self.free_gpu(pipe)
        front_rgba = img
        img = pipe.preprocess_image(img)
        mv_views = None
        if pix_mv:
            from trellis2.utils.mv_camera import build_views_from_angles
            az_of = {"back": 180.0, "left": 90.0, "right": 270.0}
            if mv_swap:
                az_of = {"back": 180.0, "left": 270.0, "right": 90.0}
            raw, azs = [], [0.0]
            for side, pth in named.items():
                try:
                    im = Image.open(pth)
                    im.load()
                    raw.append(cut_out(im))
                    azs.append(az_of.get(side, 180.0))
                except Exception as exc:
                    print("view skipped: %s (%s)" % (pth, exc), flush=True)
            if mv_framing == "crop":
                ims = crop_views_same_scale(front_rgba, raw)
            else:
                ims = [pad_square(front_rgba)] + [pad_square(r_) for r_ in raw]
            try:
                ims[0].save(os.path.join(ROOT, "_diag", "mv_in_front.png"))
                for n_, im_ in enumerate(ims[1:]):
                    im_.save(os.path.join(ROOT, "_diag", "mv_in_%d.png" % n_))
            except Exception:
                pass
            print("VIEWS %d (Pixal3D multi-view)" % len(ims), flush=True)
            mv_views = (ims, azs)
        if pix:
            views = None
        # more pictures of the same object from other sides: the model reads
        # them all together
        # TRELLIS: named sides go through its multi-view sampler, which
        # blends each picture into the side of the model it shows (a plain
        # list of pictures was only averaged and mostly ignored)
        sides = {}
        if not pix:
            for side, v in (named.items() if named else
                            zip(("back", "left", "right"), (views or [])[:3])):
                try:
                    im = Image.open(v)
                    im.load()
                    sides[side] = pipe.preprocess_image(cut_out(im))
                except Exception as exc:
                    print("view skipped: %s (%s)" % (v, exc), flush=True)
        if sides:
            print("VIEWS %d (TRELLIS multi-view: %s)" % (
                1 + len(sides), ", ".join(sides)), flush=True)

        ptype = QUALITY.get(quality, "1024_cascade")
        # Pixal3D has no 512-only networks: it always decodes at 1024 (its
        # Fast works like High), so the texture must be read at 1024 too
        names = ["Reading image", "Building shape", "Shaping detail",
                 "Painting texture", "Decoding", "Decoding"]
        bar = Bar(0.12, 0.80, 5, names)
        Stage.lo, Stage.hi, Stage.name = 0.12, 0.12 + 0.68 / 5, "Reading image"
        say(0.12, "Reading image")
        if pix:
            say(0.13, "Reading the camera")
            pipe.load_moge_model()
            cam = pipe.get_moge_camera_config(img)
            pipe.unload_moge_model()
            Stage.lo, Stage.hi, Stage.name = 0.15, 0.30, "Building shape"
            Stage.queue = []
            tex_on = int(texture) > 0
            # the stage on the card follows the network actually loaded
            marks = {"load_sparse_structure_model": ("Building shape", 0.15,
                                                     0.25),
                     "load_shape_slat_flow_model_512": ("Building shape", 0.25,
                                                        0.38),
                     "load_shape_slat_flow_model_1024": (
                         "Shaping detail", 0.38, 0.62 if tex_on else 0.80),
                     "load_tex_slat_flow_model_1024": ("Painting texture",
                                                       0.62, 0.80)}
            if not getattr(pipe, "_marked", False):
                for nm, (lab, lo_, hi_) in marks.items():
                    orig = getattr(pipe, nm)

                    def wrapped(*a, _o=orig, _l=lab, _lo=lo_, _hi=hi_, **k):
                        Stage.name, Stage.lo, Stage.hi = _l, _lo, _hi
                        say(_lo, _l)
                        return _o(*a, **k)
                    setattr(pipe, nm, wrapped)
                pipe._marked = True
            say(0.15, "Building shape")
            if mv_views is not None:
                from trellis2.utils.mv_camera import build_views_from_angles
                dist_mv = float(cam["distance"])
                if mv_framing != "crop":
                    # the views keep their framing: place the cameras so the
                    # object's largest extent in any view fills that much of
                    # the frame (how the multi-view rig was trained)
                    from trellis2.utils.mv_camera import (
                        measure_object_fill, camera_distance_for_extent)
                    fill = measure_object_fill(mv_views[0])
                    dist_mv = camera_distance_for_extent(
                        float(cam["camera_angle_x"]), fill * 512 / 2.0,
                        float(cam.get("mesh_scale", 1.0)), 512)
                    print("[MV] fill %.3f distance %.3f" % (fill, dist_mv),
                          flush=True)
                bundle = build_views_from_angles(
                    mv_views[0], mv_views[1], [0.0] * len(mv_views[1]),
                    camera_angle_x=float(cam["camera_angle_x"]),
                    mesh_scale=float(cam.get("mesh_scale", 1.0)),
                    distance=dist_mv)
                run_fn = lambda **kw: pipe.run_pixal3d_mv(views=bundle, **kw)
            else:
                run_fn = pipe.run_pixal3d
            with torch.no_grad():
                mesh = run_fn(
                    image=img, camera_params=cam, seed=int(seed),
                    pipeline_type=ptype,
                    sparse_structure_sampler_params={"steps": int(steps)},
                    shape_slat_sampler_params={"steps": int(steps)},
                    tex_slat_sampler_params={"steps": int(steps)},
                    max_num_tokens=_tokens(),
                    generate_texture_slat=bool(int(texture) > 0))[0]
        elif sides:
          with torch.no_grad():
            mesh = pipe.run_multiview(
                front=img, **sides, seed=int(seed), pipeline_type=ptype,
                sparse_structure_sampler_params={"steps": int(steps)},
                shape_slat_sampler_params={"steps": int(steps)},
                tex_slat_sampler_params={"steps": int(steps)},
                max_num_tokens=_tokens(), sparse_structure_resolution=32,
                generate_texture_slat=bool(int(texture) > 0), use_tiled=True,
                pbar=bar, sampler="euler", fill_holes=True,
                hole_iterations=1, hole_fill_algorithm="flood_fill",
                keep_only_shell=True)[0]
        else:
          with torch.no_grad():
            mesh = pipe.run(
                image=img, seed=int(seed), pipeline_type=ptype,
                sparse_structure_sampler_params={"steps": int(steps)},
                shape_slat_sampler_params={"steps": int(steps)},
                tex_slat_sampler_params={"steps": int(steps)},
                max_num_tokens=_tokens(), sparse_structure_resolution=32,
                generate_texture_slat=bool(int(texture) > 0), use_tiled=True,
                pbar=bar,
                sampler="euler", fill_holes=True, hole_iterations=1,
                hole_fill_algorithm="flood_fill", keep_only_shell=True)[0]

        if int(texture) <= 0:
            # shape only: no texture painted or baked
            Stage.lo, Stage.hi, Stage.name = 0.82, 0.96, "Cleaning mesh"
            say(0.82, "Cleaning mesh")
            pipe._cleanup_cuda()
            self.free_gpu(pipe)
            tm = shape_only(mesh, _grid_of(mesh, ptype, pix), int(faces))
            del mesh
            say(0.97, "Saving")
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            tm.export(out)
            print("TIME %.1f" % (time.time() - t0), flush=True)
            print("DONE " + os.path.abspath(out), flush=True)
            return
        Stage.lo, Stage.hi, Stage.name = 0.82, 0.96, "Baking texture"
        say(0.82, "Baking texture")
        import o_voxel
        pipe._cleanup_cuda()
        # the networks are done: all of them leave the GPU, so the bake has
        # the whole card (otherwise it spills into system memory and crawls,
        # or fails outright)
        self.free_gpu(pipe)
        print("VRAM before bake %.2f GB" % (torch.cuda.memory_reserved() / 1e9),
              flush=True)
        pv, pf = pre_simplify(mesh.vertices, mesh.faces, faces)
        import o_voxel.postprocess as _pp
        if not isinstance(_pp.cv2, _FastInpaint):
            _pp.cv2 = _FastInpaint(_pp.cv2)
        tb = time.time()
        with rebuild_res(faces):
          glb = o_voxel.postprocess.to_glb(
            vertices=pv, faces=pf, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout,
            grid_size=_grid_of(mesh, ptype, pix),
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=int(faces), texture_size=int(texture),
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=False)
        print("bake %.1fs" % (time.time() - tb), flush=True)
        del mesh
        say(0.97, "Saving")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        glb.export(out)
        pipe._cleanup_cuda()
        print("TIME %.1f" % (time.time() - t0), flush=True)
        print("DONE " + os.path.abspath(out), flush=True)


def _grid_of(mesh, ptype, pix):
    """The voxel grid the mesh and its colours were decoded on: 512, 1024,
    or 1536 for Ultra. Baking with the wrong one samples the colours in the
    wrong places, and the texture comes out black."""
    try:
        vs = float(mesh.voxel_size)
        if vs > 0:
            return int(round(1.0 / vs))
    except Exception:
        pass
    return 1024 if (ptype != "512" or pix) else 512


def _lean_model(path, kw):
    """Small PC: build the network empty and fill it on the graphics card
    one weight at a time. The usual way holds the whole network twice in
    main memory (Pixal3D keeps its weights in 32-bit: 5 GB a network)."""
    import json as _j
    import torch
    from safetensors import safe_open
    from trellis2 import models
    cfg_f, st_f = path + ".json", path + ".safetensors"
    if not (os.path.exists(cfg_f) and os.path.exists(st_f)):
        return None
    cfg = _j.load(open(cfg_f))
    try:
        with torch.device("meta"):
            m = getattr(models, cfg["name"])(**cfg["args"], **kw)
        with safe_open(st_f, "pt") as f:
            keys = set(f.keys())
            sd = m.state_dict()
            if any(k not in keys for k in sd):
                return None
            m = m.to_empty(device="cuda")
            sd = m.state_dict()
            with torch.no_grad():
                for k, t in sd.items():
                    t.copy_(f.get_tensor(k).to("cuda", non_blocking=False))
        # plain tensors kept on a layer (not weights) were made empty too
        for mod in m.modules():
            for an, av in list(vars(mod).items()):
                if isinstance(av, torch.Tensor) and av.is_meta:
                    if an == "freqs" and hasattr(mod, "rope_freq"):
                        fr = torch.arange(mod.freq_dim, dtype=torch.float32) \
                            / mod.freq_dim
                        setattr(mod, an, (mod.rope_freq[0] /
                                          (mod.rope_freq[1] ** fr)).cuda())
                    else:
                        print("lean load: %s.%s left empty, normal load"
                              % (type(mod).__name__, an), flush=True)
                        del m
                        torch.cuda.empty_cache()
                        return None
        return m
    except Exception as exc:
        print("lean load failed (%s), normal load" % exc, flush=True)
        torch.cuda.empty_cache()
        return None


def _slab_conv_patch(limit=2_000_000, piece=1_200_000):
    """On a small card the last decoder layers work on millions of voxels
    and the neighbour table of one convolution alone does not fit. A 3x3x3
    convolution only looks one voxel away, so cutting the voxels into slabs
    along X (each with a one-voxel border) gives exactly the same result
    while only one slab's table is on the card at a time."""
    import torch
    from trellis2.modules.sparse.conv import conv_flex_gemm as cf
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d
    import flex_gemm
    if getattr(cf, "_slab_done", False):
        return
    orig = cf.sparse_conv3d_forward

    def forward(self, x):
        n = x.feats.shape[0]
        Co, Kd, Kh, Kw, Ci = self.weight.shape
        key = f'SubMConv3d_neighbor_cache_{Kw}x{Kh}x{Kd}_dilation{self.dilation}'
        lim_, piece_ = (limit, piece) if _TRITON is not False else \
            (limit // 4, piece // 4)
        if n <= lim_ or x.get_spatial_cache(key) is not None:
            return orig(self, x)
        flex_gemm.ops.spconv.set_algorithm(cf.config.FLEX_GEMM_ALGO)
        flex_gemm.ops.spconv.set_hashmap_ratio(cf.config.FLEX_GEMM_HASHMAP_RATIO)
        size = torch.Size([*x.shape, *x.spatial_shape])
        halo = max(self.dilation) * (max(Kd, Kh, Kw) // 2)
        cx = x.coords[:, 1].long()
        hist = torch.bincount(cx)
        cum = torch.cumsum(hist, 0)
        cuts = [0]
        while cuts[-1] < hist.numel():
            base = int(cum[cuts[-1] - 1]) if cuts[-1] > 0 else 0
            nxt = int(torch.searchsorted(cum, base + piece_, right=True))
            cuts.append(min(max(nxt, cuts[-1] + 1), hist.numel()))
        out = torch.empty((n, Co), dtype=x.feats.dtype, device=x.feats.device)
        for a, b in zip(cuts[:-1], cuts[1:]):
            sel = ((cx >= a - halo) & (cx < b + halo)).nonzero().squeeze(1)
            inner = (cx[sel] >= a) & (cx[sel] < b)
            o, _ = sparse_submanifold_conv3d(
                x.feats[sel], x.coords[sel], size, self.weight, self.bias,
                None, self.dilation)
            out[sel[inner]] = o[inner]
            del o, sel, inner, _
            torch.cuda.empty_cache()
        return x.replace(out)

    cf.sparse_conv3d_forward = forward
    import trellis2.modules.sparse.conv.conv as cc
    for nm in dir(cc):
        obj = getattr(cc, nm)
        if isinstance(obj, type) and getattr(obj, "forward", None) is orig:
            obj.forward = forward
    cf._slab_done = True


def pre_simplify(v, f, faces):
    """The decoder's surface can have tens of millions of faces; bring it
    down first (a few seconds) so the clean-up after it is quick."""
    import cumesh
    target = max(3000000, int(faces) * 10)
    if f.shape[0] <= target:
        return v, f
    cm = cumesh.CuMesh()
    cm.init(v.cuda(), f.cuda())
    cm.simplify(target, verbose=False)
    return cm.read()


class _FastInpaint:
    """cv2 for o_voxel's bake, with a quicker inpaint: the bake fills every
    empty texel of the atlas (most of a 4K image) one by one, which takes
    minutes. Only a thin band around the UV islands needs real filling (so
    seams do not bleed); the rest gets the island's mean colour."""

    def __init__(self, cv2):
        self._cv2 = cv2

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def inpaint(self, img, mask, radius, flags):
        import numpy as np
        cv2 = self._cv2
        t = time.time()
        band = cv2.dilate((mask == 0).astype(np.uint8),
                          np.ones((3, 3), np.uint8), iterations=8)
        near = (band > 0) & (mask > 0)
        out = cv2.inpaint(img, near.astype(np.uint8), radius, flags)
        far = (mask > 0) & ~near
        if far.any():
            valid = mask == 0
            if valid.any():
                out[far] = img[valid].mean(axis=0).astype(img.dtype)
        print("inpaint %.1fs" % (time.time() - t), flush=True)
        return out


class rebuild_res:
    """Rebuild the surface on a coarser grid when the finished model is not
    huge: the same look at the face count asked for, many times faster."""

    def __init__(self, faces):
        # 512 gives ~13 million faces, plenty to reduce to anything up to a
        # few million; the full 1024 grid gives ~50 million and runs a 12 GB
        # card out of memory
        self.res = 512 if int(faces) <= 3000000 else 768

    def __enter__(self):
        import cumesh.remeshing as R
        self.R, self.orig = R, R.remesh_narrow_band_dc
        res_new = self.res
        orig = self.orig
        if res_new is None:
            return self

        def dc(vertices, faces, center, scale, resolution, band=1, **kw):
            base = scale * resolution / (resolution + 3.0 * band)
            if resolution > res_new:
                scale = (res_new + 3.0 * band) / res_new * base
                resolution = res_new
            return orig(vertices, faces, center=center, scale=scale,
                        resolution=resolution, band=band, **kw)
        R.remesh_narrow_band_dc = dc
        return self

    def __exit__(self, *a):
        self.R.remesh_narrow_band_dc = self.orig


def shape_only(mesh, res, faces):
    """The same clean-up the textured path does (rebuilt surface, reduced to
    the face count), without the UVs and texture."""
    import torch
    import trimesh
    import cumesh
    v, f = pre_simplify(mesh.vertices, mesh.faces, faces)
    v = v.cuda()
    f = f.cuda()
    res = min(res, 512 if int(faces) <= 3000000 else 768)
    cm = cumesh.CuMesh()
    cm.init(v, f)
    cm.fill_holes(max_hole_perimeter=3e-2)
    v, f = cm.read()
    bvh = cumesh.cuBVH(v, f)
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                        dtype=torch.float32, device="cuda")
    scale = 1.0
    cm.init(*cumesh.remeshing.remesh_narrow_band_dc(
        v, f, center=aabb.mean(dim=0), scale=(res + 3.0) / res * scale,
        resolution=res, band=1, project_back=0, verbose=False, bvh=bvh))
    cm.simplify(int(faces), verbose=False)
    v, f = cm.read()
    vn = v.cpu().numpy().copy()
    fn = f.cpu().numpy()
    vn[:, 1], vn[:, 2] = vn[:, 2].copy(), -vn[:, 1].copy()
    return trimesh.Trimesh(vertices=vn, faces=fn, process=False)


def _natten_patch():
    """Pixal3D's neighbourhood attention, run in tiles so it fits the card
    (from gen_pixal3d.py)."""
    global _NATTEN_DONE
    if globals().get('_NATTEN_DONE'):
        return
    import torch
    import natten
    from natten import allow_flex_compile
    allow_flex_compile()
    _na2d = natten.na2d
    def _na2d_flex(q, kk, v, *a, **k):
        with torch.autocast("cuda", enabled=False):
            return _na2d_flex_inner(q, kk, v, *a, **k)
    def _na2d_flex_inner(q, kk, v, *a, **k):
        k["backend"]="flex-fna"; k["torch_compile"]=True; k.pop("q_tile_shape",None); k.pop("kv_tile_shape",None)
        od=q.dtype
        q, kk, v = q.half().contiguous(), kk.half().contiguous(), v.half().contiguous()
        print("NA2D dtype", od, q.shape, v.shape, flush=True) if not getattr(_na2d_flex_inner, "_s", 0) else None
        _na2d_flex_inner._s = 1
        d=q.shape[-1]
        if v.shape[-1]==d:
            return _na2d(q, kk, v, *a, **k).to(od)
        pad=(-v.shape[-1])%d
        if pad: v=torch.nn.functional.pad(v,(0,pad))
        outs=[_na2d(q, kk, v[...,i:i+d].contiguous(), *a, **k) for i in range(0, v.shape[-1], d)]
        o=torch.cat(outs,-1)
        o = o[..., :o.shape[-1]-pad] if pad else o
        return o.to(od)
    TILE=192
    def _pair(x):
        return (int(x),int(x)) if not isinstance(x,(tuple,list)) else (int(x[0]),int(x[1]))
    def _na2d_tiled(q, kk, v, *a, **k):
        ks=_pair(k.get("kernel_size", a[0] if a else 3)); dl=_pair(k.get("dilation",1) or 1)
        H,W=q.shape[1],q.shape[2]
        if H*W<=TILE*TILE*2:
            return _na2d_flex(q,kk,v,*a,**k)
        out=torch.empty(q.shape[:-1]+(v.shape[-1],),dtype=q.dtype,device=q.device)
        def spans(N,r,dil):
            T=min(N,TILE+2*r)
            core=T-2*r
            res=[]; s0=0
            while s0<N:
                e0=min(N,s0+core)
                ts=min(max(0,s0-r),max(0,N-T))
                te=min(N,ts+T)
                res.append((s0,e0,ts,te)); s0=e0
            return res
        rh=(ks[0]//2)*dl[0]+dl[0]; rw=(ks[1]//2)*dl[1]+dl[1]
        for (a0,a1,t0_,t1_) in spans(H,rh,dl[0]):
            for (b0,b1,u0,u1) in spans(W,rw,dl[1]):
                o=_na2d_flex(q[:,t0_:t1_,u0:u1].contiguous(),kk[:,t0_:t1_,u0:u1].contiguous(),v[:,t0_:t1_,u0:u1].contiguous(),*a,**dict(k))
                out[:,a0:a1,b0:b1]=o[:,a0-t0_:a1-t0_,b0-u0:b1-u0]
        return out
    natten.na2d = _na2d_tiled
    import natten.functional as _nf
    if hasattr(_nf, "na2d"): _nf.na2d = _na2d_tiled
    _NATTEN_DONE = True


def _alpha_box(img):
    import numpy as np
    a = np.array(img.convert("RGBA"))[:, :, 3]
    yx = np.argwhere(a > 0.8 * 255)
    return yx[:, 1].min(), yx[:, 0].min(), yx[:, 1].max(), yx[:, 0].max()


def crop_views_same_scale(front, others):
    """Crop every view so the object has the same size in all of them.
    The views are taken from the same distance around the object, so its
    height is the same in each picture while its width is not (a bust is
    wide from the front, narrow from the side). Cropping each view tight
    to itself, as for one picture, gave every view a different scale and
    the sides no longer lined up with the front."""
    from PIL import Image
    imgs = []
    for im in [front] + list(others):
        im = im.convert("RGBA")
        m = max(im.size)
        if m > 1024:
            f = 1024.0 / m
            im = im.resize((int(im.width * f), int(im.height * f)),
                           Image.Resampling.LANCZOS)
        imgs.append(im)
    x0, y0, x1, y1 = _alpha_box(imgs[0])
    side = float(max(x1 - x0, y1 - y0))
    k = side / max(1.0, float(y1 - y0))      # square size per unit of height
    out = []
    for n, im in enumerate(imgs):
        x0, y0, x1, y1 = _alpha_box(im)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sz = side if n == 0 else k * (y1 - y0)
        h = sz / 2.0
        out.append(im.crop((int(round(cx - h)), int(round(cy - h)),
                            int(round(cx + h)), int(round(cy + h)))))
    return out


def pad_square(img):
    """Keep the picture's own framing (all views of a set are usually shot
    or generated the same way); only pad it to a square."""
    from PIL import Image
    img = img.convert("RGBA")
    m = max(img.size)
    sq = Image.new("RGBA", (m, m), (0, 0, 0, 0))
    sq.paste(img, ((m - img.width) // 2, (m - img.height) // 2))
    if m > 1024:
        sq = sq.resize((1024, 1024), Image.Resampling.LANCZOS)
    return sq


def crop_rgba(img):
    """The single-view crop (square around the object, as preprocess_image
    does) but keeping the alpha, which the multi-view path needs as the mask."""
    import numpy as np
    from PIL import Image
    img = img.convert("RGBA")
    m = max(img.size)
    if m > 1024:
        f = 1024.0 / m
        img = img.resize((int(img.width * f), int(img.height * f)),
                         Image.Resampling.LANCZOS)
    a = np.array(img)[:, :, 3]
    yx = np.argwhere(a > 0.8 * 255)
    x0, y0, x1, y1 = yx[:, 1].min(), yx[:, 0].min(), yx[:, 1].max(), yx[:, 0].max()
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sz = max(x1 - x0, y1 - y0)
    return img.crop((cx - sz // 2, cy - sz // 2, cx + sz // 2, cy + sz // 2))


def make_pixal_mv(pipe):
    """run_pixal3d with the multi-view conditioning (the repo has the MV
    networks and the MV feature code but no generation entry point)."""
    import inspect
    import re
    import textwrap
    import types
    src = textwrap.dedent(inspect.getsource(type(pipe).run_pixal3d))
    src = src.replace("def run_pixal3d(", "def run_pixal3d_mv(", 1)
    src = re.sub(r"^@.*\n", "", src, flags=re.M)
    src = src.replace("camera_params: dict,", "camera_params: dict,\n        views: dict = None,", 1)
    src, n1 = re.subn(r"self\.get_proj_cond_ss\(\s*\[image\],.*?image_cond_model=image_cond_model\s*\)",
                      "self.get_proj_cond_ss_mv(views, image_cond_model)", src, flags=re.S)
    src, n2 = re.subn(r"self\.get_proj_cond_shape\(\s*image_cond_model,\s*\[image\],\s*([\w\.]+),\s*"
                      r"camera_angle_x=camera_angle_x,\s*distance=distance,\s*mesh_scale=mesh_scale,\s*"
                      r"(grid_resolution_override=\w+,)?\s*\)",
                      r"self.get_proj_cond_shape_mv(image_cond_model, views, \1, \2)", src, flags=re.S)
    src = src.replace("load_pixal3d_image_cond_", "load_pixal3d_mv_image_cond_")
    if n1 != 1 or n2 != 3:
        raise RuntimeError("multi-view patch did not fit (%d, %d)" % (n1, n2))
    mod = inspect.getmodule(type(pipe))
    ns = dict(vars(mod))
    exec(compile(src, "<pixal_mv>", "exec"), ns)
    pipe.run_pixal3d_mv = types.MethodType(ns["run_pixal3d_mv"], pipe)


def _half_mv_features(pipe):
    """With several views the feature grid (64^3 x 2048) is held twice while
    the views are averaged: 4 GB in 32-bit, which spills off a 12 GB card.
    Keep it in 16-bit (the networks run in 16-bit anyway) and hand the
    denoiser 32-bit features again only for the voxels it uses."""
    import types
    from trellis2.trainers.flow_matching.mixins import image_conditioned_proj as icp
    from trellis2.modules.sparse import SparseTensor
    cls = icp.DinoV3ProjMultiViewFeatureExtractor
    if not getattr(cls, "_half_patched", False):
        orig = cls._project_single_view

        def proj_half(self, *a, **k):
            return orig(self, *a, **k).half()
        cls._project_single_view = proj_half
        cls._half_patched = True

    def to32(d):
        for key in ("cond", "neg_cond"):
            c = d.get(key, {})
            pr = c.get("proj")
            if isinstance(pr, SparseTensor):
                c["proj"] = SparseTensor(feats=pr.feats.float(), coords=pr.coords)
            elif pr is not None and hasattr(pr, "float"):
                c["proj"] = pr.float()
            g = c.get("global")
            if g is not None and hasattr(g, "float"):
                c["global"] = g.float()
        return d
    o_shape = pipe.get_proj_cond_shape_mv
    o_ss = pipe.get_proj_cond_ss_mv

    def fresh(fn):
        def go(*a, **k):
            import gc
            import torch
            # start the heavy feature step on an emptied card
            gc.collect()
            torch.cuda.empty_cache()
            return to32(fn(*a, **k))
        return go
    pipe.get_proj_cond_shape_mv = fresh(o_shape)
    pipe.get_proj_cond_ss_mv = fresh(o_ss)


_SESSION = {}


def _birefnet():
    """BiRefNet on the graphics card (half precision): rembg's CPU version
    needs 8 to 11 GB of main memory for one picture"""
    if "m" in _SESSION:
        return _SESSION["m"]
    from transformers import AutoModelForImageSegmentation
    path = os.path.join(MODELS, "ZhengPeng7", "BiRefNet")
    if not os.path.isdir(path):
        path = os.path.join(os.path.dirname(MODELS.rstrip("\\/")), "models",
                            "ZhengPeng7", "BiRefNet")
    m = AutoModelForImageSegmentation.from_pretrained(
        path, trust_remote_code=True, local_files_only=True)
    m = m.eval().half().to("cuda")
    _SESSION["m"] = m
    return m


def release_cut_out():
    if "m" in _SESSION:
        _SESSION.pop("m", None)
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def cut_out(img):
    import numpy as np
    if img.mode == "RGBA":
        a = np.array(img)[:, :, 3]
        if not np.all(a == 255):
            return img
    rgb = img.convert("RGB")
    if True:   # the GPU cut-out is faster on every PC (CPU one: ~20 s)
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
            release_cut_out()
            a = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8))
            a = a.resize(rgb.size, Image.Resampling.BILINEAR)
            out = rgb.copy()
            out.putalpha(a)
            return out
        except Exception as exc:
            print("GPU cut-out failed (%s), using rembg" % exc, flush=True)
            release_cut_out()
    from rembg import remove, new_session
    if "s" not in _SESSION:
        _SESSION["s"] = new_session("birefnet-general")
    return remove(rgb, session=_SESSION["s"])


def nice(exc):
    msg = str(exc).strip().splitlines()[-1] if str(exc).strip() \
        else type(exc).__name__
    if "out of memory" in msg.lower():
        msg = "Not enough GPU memory - try Quality: Fast"
    return msg[:200]


def small_pc():
    """16 GB of main memory or an 8 GB card (or the test mode)"""
    if os.environ.get("IMG3D_VRAM_GB") or os.environ.get("IMG3D_LOWMEM"):
        return True
    if ram_gb() < 28.0:
        return True
    try:
        import torch
        return torch.cuda.get_device_properties(0).total_memory < \
            10.5 * 2**30
    except Exception:
        return False


def _tokens():
    """how many detail cells the 1024 stage may use: decoding them all
    needs more than an 8 GB card holds"""
    env = os.environ.get("IMG3D_TOKENS")
    if env:
        return int(env)
    return 32768 if small_pc() else 49152


def serve(keep):
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
            if eng is None:
                say(0.01, "Starting")
                eng = Engine(keep)
            if job.pop("task", None) == "texture_mesh":
                eng.texture_mesh(**job)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--keep", type=int, default=-1)
    ap.add_argument("--image")
    ap.add_argument("--out")
    ap.add_argument("--quality", default="HIGH")
    ap.add_argument("--faces", type=int, default=200000)
    ap.add_argument("--texture", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--fp8", action="store_true")
    a = ap.parse_args()
    keep = (ram_gb() >= 48.0) if a.keep < 0 else bool(a.keep)
    if small_pc():
        keep = False
    if a.serve:
        serve(keep)
        return
    say(0.01, "Starting")
    eng = Engine(False)
    eng.run(a.image, a.out, a.quality, a.faces, a.texture, a.seed, a.steps,
            a.fp8)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        print("ERROR " + nice(exc), flush=True)
        sys.exit(1)
