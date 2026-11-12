"""Speed-ups for the Hunyuan3D 2.1 engine (same ideas as the TRELLIS one):
 - UV unwrap runs in its own process while the texture views are painted
 - the paint model leaves the graphics card before upscaling and baking,
   so nothing spills into shared memory
 - quick inpaint: only a thin band around the UV islands is really filled
"""
import os
import sys
import time
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_W = 8.0     # a picture counts this much more than a painted view


def log(msg):
    print(msg, flush=True)
    try:
        with open(os.path.join(tempfile.gettempdir(), "img3d_hy_enhance.log"), "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def _pull_push(cv2, img, mask):
    """fill the empty texels from the painted ones around them, coarse to
    fine (a second instead of ten for a 4K map)"""
    import numpy as np
    a = img.astype(np.float32)
    if a.ndim == 2:
        a = a[..., None]
    w = (mask == 0).astype(np.float32)[..., None]
    cs, ws = [a * w], [w]
    while cs[-1].shape[0] > 4:
        h_ = cs[-1].shape[0] // 2
        c_ = cv2.resize(cs[-1], (h_, h_), interpolation=cv2.INTER_AREA)
        w_ = cv2.resize(ws[-1], (h_, h_), interpolation=cv2.INTER_AREA)
        if c_.ndim == 2:
            c_ = c_[..., None]
        if w_.ndim == 2:
            w_ = w_[..., None]
        cs.append(c_)
        ws.append(w_)
    fill = cs[-1] / np.maximum(ws[-1], 1e-6)
    for c_, w_ in zip(reversed(cs[:-1]), reversed(ws[:-1])):
        H = c_.shape[0]
        up = cv2.resize(fill, (H, H), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2:
            up = up[..., None]
        k = np.clip(w_, 0, 1)
        fill = np.where(w_ > 1e-6, c_ / np.maximum(w_, 1e-6), up) * k + \
            up * (1 - k)
    out = np.where(w > 0, a, fill)
    out = np.clip(out, 0, 255).astype(img.dtype)
    return out.reshape(img.shape)


class FastInpaintCV2:
    def __init__(self, cv2):
        self._cv2 = cv2

    def __getattr__(self, name):
        return getattr(self._cv2, name)

    def inpaint(self, img, mask, radius, flags):
        import numpy as np
        cv2 = self._cv2
        if img.shape[0] > 2048:
            return _pull_push(cv2, img, mask)
        band = cv2.dilate((mask == 0).astype(np.uint8),
                          np.ones((3, 3), np.uint8), iterations=8)
        near = (band > 0) & (mask > 0)
        out = cv2.inpaint(img, near.astype(np.uint8), radius, flags)
        far = (mask > 0) & ~near
        if far.any():
            valid = mask == 0
            if valid.any():
                out[far] = img[valid].mean(axis=0).astype(img.dtype)
        return out


def start_uv(vertices, faces):
    import numpy as np
    work = tempfile.mkdtemp(prefix="hyuv_")
    src = os.path.join(work, "in.npz")
    dst = os.path.join(work, "out.npz")
    np.savez(src, v=np.asarray(vertices, np.float32),
             f=np.asarray(faces, np.int32))
    # the TRELLIS environment unwraps on the graphics card (cumesh): about a
    # minute for a million faces, where xatlas on the CPU takes many minutes
    root = os.path.dirname(HERE)
    gpu_py = os.path.join(root, "env", "Scripts", "python.exe")
    cpu = [sys.executable, os.path.join(HERE, "uv_worker.py"), src, dst]
    # only a finished TRELLIS install (the INSTALLED flag is written last)
    if os.path.isfile(gpu_py) and os.path.isfile(os.path.join(root,
                                                              "INSTALLED")):
        cmd = [gpu_py, os.path.join(HERE, "uv_worker_gpu.py"), src, dst]
    else:
        cmd = cpu
    p = subprocess.Popen(cmd, creationflags=0x08000000)
    return p, dst, (cpu if cmd is not cpu else None)


def finish_uv(job):
    import numpy as np
    p, dst = job[0], job[1]
    cpu = job[2] if len(job) > 2 else None
    p.wait()
    if (p.returncode != 0 or not os.path.exists(dst)) and cpu:
        # the graphics card unwrap could not run here: the CPU one can
        print("GPU unwrap failed - unwrapping on the CPU", flush=True)
        p = subprocess.Popen(cpu, creationflags=0x08000000)
        p.wait()
    if p.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError("UV unwrap failed")
    d = np.load(dst)
    return d["vm"], d["ix"], d["uv"]


def _visit(mod):
    """mod lives in RAM and visits the graphics card only while it runs"""
    import torch
    if getattr(mod, "_visiting", False):
        return
    mod.to("cpu")
    fwd = mod.forward

    def run(*a, **k):
        mod.to("cuda")
        try:
            return fwd(*a, **k)
        finally:
            mod.to("cpu")
            torch.cuda.empty_cache()
    mod.forward = run
    mod._visiting = True


def _offload(mv):
    """The picture reader (DINO) and the text encoder are used once at the
    start of the painting; they wait in RAM the rest of the time instead of
    filling the card next to the painter."""
    import torch
    if getattr(mv, "_offload_on", False):
        return
    d = getattr(mv, "dino_v2", None)
    if d is not None:
        _visit(d)
    mv._offload_on = True
    torch.cuda.empty_cache()


def low_memory():
    """a PC with an 8 GB card or 16 GB of main memory (or the test mode that
    pretends to be one)"""
    import torch
    if os.environ.get("IMG3D_VRAM_GB") or os.environ.get("IMG3D_LOWMEM"):
        return True
    try:
        if torch.cuda.get_device_properties(0).total_memory < 10.5 * 2**30:
            return True
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().total < 28 * 2**30
    except Exception:
        return False


def fp8_weights(model, min_numel=4096):
    """Keep the big weights in 8 bits (half the memory, on the card and in
    main memory) and widen each one to 16 bits only while its layer runs."""
    import torch
    import torch.nn as nn
    n = 0
    for mod in model.modules():
        if not isinstance(mod, (nn.Linear, nn.Conv2d)) or \
                getattr(mod, "_w8", False):
            continue
        w = mod.weight
        if w is None or w.numel() < min_numel or w.dtype != torch.float16:
            continue
        w8 = w.data.to(torch.float8_e4m3fn)
        del mod.weight
        mod.register_buffer("weight8", w8)
        mod.weight = None

        def pre(m, args):
            m.weight = m.weight8.to(torch.float16)

        def post(m, args, out):
            m.weight = None
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        mod._w8 = True
        n += 1
    return n


def gpu_park(paint, where):
    import torch
    mv = paint.models["multiview_model"]
    for name in ("unet", "vae", "text_encoder"):
        m_ = getattr(mv.pipeline, name, None)
        if m_ is not None:
            try:
                m_.to(where)
            except Exception:
                pass
    try:
        # decode the painted views one at a time: all twelve at once do
        # not fit on a 12 GB card
        mv.pipeline.vae.enable_slicing()
        mv.pipeline.vae.enable_tiling()
    except Exception:
        pass

    torch.cuda.empty_cache()


def fit_photo(photo, mask):
    """Place a cut-out photo over the model's silhouette (mask, H x W bool):
    same height, same centre. Returns (RGB image, alpha, IoU)."""
    import numpy as np
    import cv2
    from PIL import Image
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None, 0.0
    X0, X1, Y0, Y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    im = photo.convert("RGBA")
    a = np.array(im)[:, :, 3]
    py, px = np.nonzero(a > 128)
    im = im.crop((px.min(), py.min(), px.max() + 1, py.max() + 1))
    k = (Y1 - Y0) / float(im.height)
    nw, nh = max(1, round(im.width * k)), max(1, round(im.height * k))
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    cx, cy = (X0 + X1) / 2.0, (Y0 + Y1) / 2.0
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cv.paste(im, (int(round(cx - nw / 2.0)), int(round(cy - nh / 2.0))), im)
    rgba = np.array(cv)
    al = rgba[:, :, 3] > 128
    iou = float((al & mask).sum()) / max(1, float((al | mask).sum()))
    rgb = rgba[:, :, :3].copy()
    # spread the colours past the edge, so a slightly misaligned edge does
    # not bake background onto the model
    holes = (~al).astype(np.uint8)
    band = cv2.dilate(al.astype(np.uint8), np.ones((3, 3), np.uint8),
                      iterations=24) > 0
    fill = (holes > 0) & band
    if fill.any():
        rgb = cv2.inpaint(rgb, fill.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return Image.fromarray(rgb), al, iou


def install(tg):
    """Replace Hunyuan3DPaintPipeline.__call__ with the faster one."""
    import copy
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    import DifferentiableRenderer.MeshRender as MR
    if not isinstance(MR.cv2, FastInpaintCV2):
        MR.cv2 = FastInpaintCV2(MR.cv2)

    @torch.no_grad()
    def fast_call(self, mesh_path=None, image_path=None,
                  output_mesh_path=None, use_remesh=True, save_glb=True):
        if isinstance(image_path, list):
            image_prompt = image_path
        else:
            image_prompt = [Image.open(image_path)
                            if isinstance(image_path, str) else image_path]
        path = os.path.dirname(mesh_path)
        if use_remesh:
            processed = os.path.join(path, "white_mesh_remesh.obj")
            tg.remesh_mesh(mesh_path, processed)
        else:
            processed = mesh_path
        if output_mesh_path is None:
            output_mesh_path = os.path.join(path, "textured_mesh.obj")

        mesh = trimesh.load(processed, force="mesh", process=False)
        t = time.time()
        uv_job = start_uv(mesh.vertices, mesh.faces)
        _offload(self.models["multiview_model"])

        # views of the bare shape (no UVs needed for these)
        bare = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                               process=False)
        # stand-in UVs (front projection) until the real unwrap is done;
        # only the views are rendered from this
        vv = np.asarray(mesh.vertices, np.float32)
        lo, hi = vv.min(0), vv.max(0)
        fake = (vv[:, :2] - lo[:2]) / np.maximum(1e-6, (hi - lo)[:2])
        bare.visual = trimesh.visual.TextureVisuals(
            uv=(fake * 0.98 + 0.01).astype(np.float32))
        self.render.load_mesh(mesh=bare)
        log("T bare load %.1f" % (time.time() - t))
        elevs, azims, weights = self.view_processor.bake_view_selection(
            self.config.candidate_camera_elevs,
            self.config.candidate_camera_azims,
            self.config.candidate_view_weights,
            self.config.max_selected_view_num)
        normal_maps = self.view_processor.render_normal_multiview(
            elevs, azims, use_abs_coor=True)
        position_maps = self.view_processor.render_position_multiview(
            elevs, azims)

        log("T views %.1f" % (time.time() - t))
        # the views are rendered: a big model (a million faces and more) and
        # its 4K buffers would otherwise sit on the card through the painting
        # and push the paint model out of memory. Keep a tiny stand-in there
        # until the real model comes back for baking.
        tiny = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                               faces=[[0, 1, 2]], process=False)
        tiny.visual = trimesh.visual.TextureVisuals(
            uv=np.array([[0, 0], [1, 0], [0, 1]], np.float32))
        self.render.load_mesh(mesh=tiny)
        del bare
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        style = []
        for image in image_prompt:
            image = image.resize((512, 512))
            if image.mode == "RGBA":
                white = Image.new("RGB", image.size, (255, 255, 255))
                white.paste(image, mask=image.getchannel("A"))
                image = white
            style.append(image.convert("RGB"))

        # the paint model comes onto the card only now: rendering the views
        # of a big model needs the room first
        gpu_park(self, "cuda")
        f_, tot_ = torch.cuda.mem_get_info()
        log("MEM before paint: torch %.2f GB, free %.2f of %.2f GB" % (
            torch.cuda.memory_allocated() / 1e9, f_ / 1e9, tot_ / 1e9))
        t1 = time.time()
        mvs = self.models["multiview_model"](
            style, normal_maps + position_maps, prompt="high quality",
            custom_view_size=self.config.resolution, resize_input=True)
        log("T multiview %.1f" % (time.time() - t1))

        # the paint model is done: free the card for upscaling and baking
        if low_memory():
            # small PC: drop it altogether (read again next time) so the
            # bake has the main memory too
            self.models["multiview_model"] = None
            self._spent = True
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        else:
            gpu_park(self, "cpu")
        # keep the painted views, so the texture can be enhanced and baked
        # again later without painting (or building the shape) again
        try:
            import json
            vdir = os.path.join(os.path.dirname(output_mesh_path), "views")
            os.makedirs(vdir, exist_ok=True)
            for k_ in ("albedo", "mr"):
                for n_, im_ in enumerate(mvs[k_]):
                    im_.save(os.path.join(vdir, "%s_%d.png" % (k_, n_)))
            with open(os.path.join(vdir, "cams.json"), "w") as f_:
                json.dump({"elevs": [float(e) for e in elevs],
                           "azims": [float(a) for a in azims],
                           "weights": [float(w) for w in weights],
                           "n": len(mvs["albedo"])}, f_)
        except Exception as exc:
            log("views not kept: %s" % exc)

        t1 = time.time()
        enh = {"albedo": [], "mr": []}
        for k in ("albedo", "mr"):
            for im in mvs[k]:
                enh[k].append(self.models["super_model"](im).resize(
                    (self.config.render_size, self.config.render_size)))
        log("T upscale %.1f" % (time.time() - t1))

        t1 = time.time()
        vm, ix, uv = finish_uv(uv_job)
        log("T uv wait %.1f (uv total %.1f)" % (time.time() - t1,
                                              time.time() - t))
        try:
            np.savez(os.path.join(os.path.dirname(output_mesh_path), "views",
                                  "uvmesh.npz"),
                     v=np.asarray(mesh.vertices)[vm], f=ix, uv=uv)
        except Exception:
            pass
        m2 = trimesh.Trimesh(vertices=np.asarray(mesh.vertices)[vm],
                             faces=ix, process=False)
        m2.visual = trimesh.visual.TextureVisuals(uv=uv)
        self.render.load_mesh(mesh=m2)

        # the user's own pictures, far sharper than the 512-pixel painted
        # views, go straight onto the model where they show it; the painted
        # views only fill what no picture sees
        ph_views, ph_el, ph_az, ph_w = [], [], [], []
        photos = getattr(self, "_photos", None) or {}
        if photos:
            t1 = time.time()
            R = self.config.render_size
            masks = {}

            def sil(az):
                if az not in masks:
                    a_ = self.render.render_alpha(0, az, resolution=R,
                                                  return_type="np")
                    masks[az] = np.asarray(a_)[0, :, :, 0] > 0
                return masks[az]
            fixed = {"front": 0, "back": 180}
            for nm, az in fixed.items():
                if nm in photos:
                    img_, _, iou = fit_photo(photos[nm], sil(az))
                    if img_ is not None and iou > 0.5:
                        ph_views.append(img_); ph_el.append(0)
                        ph_az.append(az); ph_w.append(PHOTO_W)
                    log("photo %s az %d IoU %.0f%%" % (nm, az, iou * 100))
            sides = [n for n in ("left", "right") if n in photos]
            if sides:
                fits = {(n, az): fit_photo(photos[n], sil(az))
                        for n in sides for az in (90, 270)}
                if len(sides) == 2:
                    a1 = fits[("left", 90)][2] + fits[("right", 270)][2]
                    a2 = fits[("left", 270)][2] + fits[("right", 90)][2]
                    pick = {"left": 90, "right": 270} if a1 >= a2 else \
                        {"left": 270, "right": 90}
                else:
                    n = sides[0]
                    pick = {n: 90 if fits[(n, 90)][2] >= fits[(n, 270)][2]
                            else 270}
                for n, az in pick.items():
                    img_, _, iou = fits[(n, az)]
                    if img_ is not None and iou > 0.5:
                        ph_views.append(img_); ph_el.append(0)
                        ph_az.append(az); ph_w.append(PHOTO_W)
                    log("photo %s az %d IoU %.0f%%" % (n, az, iou * 100))
            log("T photos %.1f" % (time.time() - t1))

        t1 = time.time()
        # pictures first: the bake skips a view whose area is already
        # almost all painted, so the painted views only fill what is left
        tex, mask = self.view_processor.bake_from_multiview(
            ph_views + list(enh["albedo"]), ph_el + list(elevs),
            ph_az + list(azims), ph_w + list(weights))
        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
        tex_mr, mask_mr = self.view_processor.bake_from_multiview(
            enh["mr"], elevs, azims, weights)
        mask_mr_np = (mask_mr.squeeze(-1).cpu().numpy() * 255
                      ).astype(np.uint8)
        tex = self.view_processor.texture_inpaint(tex, mask_np)
        self.render.set_texture(tex, force_set=True)
        tex_mr = self.view_processor.texture_inpaint(tex_mr, mask_mr_np)
        self.render.set_texture_mr(tex_mr)
        log("T bake %.1f" % (time.time() - t1))
        # full size: the original halved the texture here, so a "4K" texture
        # came out 2K
        self.render.save_mesh(output_mesh_path, downsample=False)
        torch.cuda.empty_cache()
        return output_mesh_path

    tg.Hunyuan3DPaintPipeline.__call__ = fast_call


def _bake(render, views, el, az, w, exp):
    """one pass over the views, keeping only the running sum (the stock
    code merges all views again after every view and keeps every
    projection in memory)"""
    import torch
    tm = None
    for view, e, a, wt in zip(views, el, az, w):
        pt, pc, _ = render.back_project(view, e, a)
        pc = wt * (pc ** exp)
        if tm is None:
            tm = torch.zeros(render.texture_size + (pt.shape[-1],),
                             device=pt.device)
            tr = torch.zeros(render.texture_size + (1,), device=pt.device)
        vs = (pc > 0).sum()
        if vs > 0 and ((pc > 0) & (tr > 0)).sum() / vs > 0.99:
            continue
        tm += pt * pc
        tr += pc
        del pt, pc
    tm = tm / torch.clamp(tr, min=1e-8)
    return tm, tr > 1e-8


def _inpaint_small(render, tex, mask_np, size=1024):
    """fill the empty texture at low resolution and only use it where
    nothing was painted (for the smooth roughness/metal map)"""
    import cv2
    import numpy as np
    import torch
    t = tex.cpu().numpy()
    H = t.shape[0]
    ts = cv2.resize(t, (size, size), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(mask_np, (size, size), interpolation=cv2.INTER_NEAREST)
    f = render.uv_inpaint(ts, ms, vertex_inpaint=False)
    f = cv2.resize(f, (H, H), interpolation=cv2.INTER_LINEAR)
    f = f.astype(np.float32) / 255.0
    if f.ndim == 2:
        f = f[..., None]
    keep = (mask_np > 0)[..., None]
    out = np.where(keep, t, f)
    return torch.tensor(out).float().to(tex.device)


def enhance(conf, vdir, out_obj, passes=1, sharpen=0.0, ai=True,
            tex_only=False):
    """Bake the kept views again with a stronger clean-up: each view is
    upscaled (twice for passes=2, then brought back to the bake size) and
    lightly sharpened. No painting, no shape - well under a minute."""
    import json
    import numpy as np
    import torch
    import trimesh
    from PIL import Image, ImageFilter
    from textureGenPipeline import Hunyuan3DPaintPipeline
    from DifferentiableRenderer.MeshRender import MeshRender
    from utils.pipeline_utils import ViewProcessor
    from utils.image_super_utils import imageSuperNet
    import DifferentiableRenderer.MeshRender as MR
    if not isinstance(MR.cv2, FastInpaintCV2):
        MR.cv2 = FastInpaintCV2(MR.cv2)
    if not getattr(MR.meshVerticeInpaint, "_timed", False):
        _mvi = MR.meshVerticeInpaint
        def _tmvi(*a, **k):
            t_ = time.time()
            r_ = _mvi(*a, **k)
            log("T e.vertex_inpaint %.1f" % (time.time() - t_))
            return r_
        _tmvi._timed = True
        MR.meshVerticeInpaint = _tmvi
    t = time.time()
    p = Hunyuan3DPaintPipeline.__new__(Hunyuan3DPaintPipeline)
    p.config = conf
    p.models = {}
    p.stats_logs = {}
    p.render = MeshRender(default_resolution=conf.render_size,
                          texture_size=conf.texture_size,
                          bake_mode=conf.bake_mode,
                          raster_mode=conf.raster_mode)
    p.view_processor = ViewProcessor(conf, p.render)
    sr = imageSuperNet(conf)
    log("T e.init %.1f" % (time.time() - t))
    cams = json.load(open(os.path.join(vdir, "cams.json")))
    R = conf.render_size

    def up(im):
        for _ in range(max(1, int(passes))):
            # only upscale while the 4x result is not far past the bake
            # size (a 2048 view would become 8192 and fill the card)
            if im.width * 2 >= R:
                break
            im = sr(im)
        im = im.resize((R, R), Image.Resampling.LANCZOS)
        if sharpen > 0:
            im = im.filter(ImageFilter.UnsharpMask(
                radius=2, percent=int(sharpen * 150), threshold=2))
        return im
    def src(i):
        p_ = os.path.join(vdir, "albedo_ai_%d.png" % i)
        if ai and os.path.isfile(p_):
            return Image.open(p_).convert("RGB")
        return Image.open(os.path.join(vdir, "albedo_%d.png" % i))
    alb = [up(src(i)) for i in range(cams["n"])]
    # the roughness/metal views are smooth: baked from 1024 pictures
    RM = min(R, 1024)
    mr = [Image.open(os.path.join(vdir, "mr_%d.png" % i)).convert("RGB")
          .resize((RM, RM), Image.Resampling.LANCZOS) for i in range(cams["n"])]
    log("T enhance views %.1f" % (time.time() - t))
    t1 = time.time()
    d = np.load(os.path.join(vdir, "uvmesh.npz"))
    m2 = trimesh.Trimesh(vertices=d["v"], faces=d["f"], process=False)
    m2.visual = trimesh.visual.TextureVisuals(uv=d["uv"])
    p.render.load_mesh(mesh=m2)
    log("T e.mesh %.1f" % (time.time() - t1)); t1 = time.time()
    el, az, w = cams["elevs"], cams["azims"], cams["weights"]
    ex = conf.bake_exp
    def mem():
        try:
            f_, t_ = torch.cuda.mem_get_info()
            return "%.1f/%.1fG free" % (f_ / 2**30, t_ / 2**30)
        except Exception:
            return ""
    log("T e.mem " + mem())
    tex, mask = _bake(p.render, alb, el, az, w, ex)
    del alb
    mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    log("T e.bake_albedo %.1f" % (time.time() - t1)); t1 = time.time()
    tmr, mmr = _bake(p.render, mr, el, az, w, ex)
    del mr
    mmr_np = (mmr.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    log("T e.bake_mr %.1f" % (time.time() - t1)); t1 = time.time()
    tex = p.view_processor.texture_inpaint(tex, mask_np)
    log("T e.inpaint_albedo %.1f" % (time.time() - t1))
    p.render.set_texture(tex, force_set=True)
    tmr = _inpaint_small(p.render, tmr, mmr_np)
    p.render.set_texture_mr(tmr)
    log("T e.inpaint %.1f" % (time.time() - t1)); t1 = time.time()
    if tex_only:
        # only the pictures: the model in the scene keeps its mesh and UVs
        import cv2
        stem_ = os.path.splitext(out_obj)[0]
        a_ = np.clip(p.render.get_texture() * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(stem_ + ".jpg", cv2.cvtColor(a_, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        mr_ = np.clip(p.render.tex_mr.cpu().numpy() * 255.0, 0, 255
                      ).astype(np.uint8)
        cv2.imwrite(stem_ + "_metallic.jpg", mr_[:, :, 0],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(stem_ + "_roughness.jpg", mr_[:, :, 1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        p.render.save_mesh(out_obj, downsample=False)
    log("T e.save_obj %.1f" % (time.time() - t1)); t1 = time.time()
    try:
        stem = os.path.splitext(out_obj)[0]
        alb_p = stem + ".jpg"
        if os.path.isfile(alb_p):
            detail_normal(alb_p, mask_np, stem + "_normal.png")
            rp = stem + "_roughness.jpg"
            if os.path.isfile(rp):
                detail_roughness(alb_p, rp)
            log("T e.normal_rough %.1f" % (time.time() - t1))
    except Exception as exc:
        log("no normal map: %s" % exc)
    del p, sr
    torch.cuda.empty_cache()
    log("T enhance total %.1f" % (time.time() - t))
    return out_obj


def detail_normal(albedo_path, mask, out_path, strength=3.0):
    """A normal map from the fine detail of the colour texture: dark cracks,
    scratches and seams become small dents, so light catches them. Only the
    painted UV islands get it; their edges are faded out so seams stay
    flat."""
    import cv2
    import numpy as np
    a = cv2.imread(albedo_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    m = (mask > 0).astype(np.uint8) if mask is not None else \
        np.ones(g.shape, np.uint8)
    if m.shape != g.shape:
        m = cv2.resize(m, (g.shape[1], g.shape[0]),
                       interpolation=cv2.INTER_NEAREST)
    # fine detail only: the broad shading of the painted views is left out
    h = g - cv2.GaussianBlur(g, (0, 0), 6.0)
    h = cv2.GaussianBlur(h, (0, 0), 0.8)
    edge = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=6)
    fade = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), 3.0)
    h *= fade
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    nx, ny, nz = -gx * strength, gy * strength, np.ones_like(h)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nz / ln, ny / ln, nx / ln], -1)     # BGR order for cv2
    cv2.imwrite(out_path, np.clip((n * 0.5 + 0.5) * 255.0, 0, 255)
                .astype(np.uint8))
    return out_path


def detail_roughness(albedo_path, rough_path, amount=0.35):
    """Cracks, scratches and grime come out rougher (duller), clean raised
    surfaces a little smoother, following the fine detail of the colour."""
    import cv2
    import numpy as np
    a = cv2.imread(albedo_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    r = cv2.imread(rough_path, cv2.IMREAD_GRAYSCALE)
    if r is None:
        return
    g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    if r.shape != g.shape:
        r = cv2.resize(r, (g.shape[1], g.shape[0]))
    hp = g - cv2.GaussianBlur(g, (0, 0), 6.0)
    r2 = r.astype(np.float32) / 255.0 - amount * 2.0 * hp
    cv2.imwrite(rough_path, np.clip(r2 * 255.0, 0, 255).astype(np.uint8))


def _render_tex(render, tex, elev, azim, res):
    """The model with its own texture, seen from one camera, in the same
    raster as the painted views (so the bake puts it back in place)."""
    import torch
    import torch.nn.functional as F
    from DifferentiableRenderer.MeshRender import RenderConfig
    vs = render._create_view_state(RenderConfig(elev, azim, resolution=res))
    rast, _ = render.raster_rasterize(vs.pos_clip, render.pos_idx,
                                      resolution=vs.resolution)
    uv, _ = render.raster_interpolate(render.vtx_uv[None, ...], rast,
                                      render.uv_idx)
    grid = uv[..., :2] * 2.0 - 1.0
    t = tex.permute(2, 0, 1)[None, ...]
    col = F.grid_sample(t, grid, mode="bilinear", padding_mode="border",
                        align_corners=False)[0].permute(1, 2, 0)
    vis = torch.clamp(rast[0, ..., -1:], 0, 1)
    col = col * vis + (1 - vis)
    return col.clamp(0, 1)


def prep_views(conf, vdir, mesh_npz, albedo, mr=None, size=2048):
    """Views for Enhance Texture on a model that was not painted here
    (TRELLIS, Pixal3D, or any textured model): its own texture is rendered
    from the same cameras the painter uses, then the normal enhance runs on
    them. The model keeps its mesh and UVs."""
    import json
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from DifferentiableRenderer.MeshRender import MeshRender
    from utils.pipeline_utils import ViewProcessor
    t = time.time()
    os.makedirs(vdir, exist_ok=True)
    d = np.load(mesh_npz)
    v, f, uv = d["v"], d["f"], d["uv"]
    render = MeshRender(default_resolution=size, texture_size=conf.texture_size,
                        bake_mode=conf.bake_mode, raster_mode=conf.raster_mode)
    vp = ViewProcessor(conf, render)
    m = trimesh.Trimesh(vertices=v, faces=f, process=False)
    m.visual = trimesh.visual.TextureVisuals(uv=uv)
    render.load_mesh(mesh=m)
    elevs, azims, weights = vp.bake_view_selection(
        conf.candidate_camera_elevs, conf.candidate_camera_azims,
        conf.candidate_view_weights, conf.max_selected_view_num)
    dev = render.device

    def tex_of(path, default):
        if path and os.path.isfile(path):
            a = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0
        else:
            a = np.ones((8, 8, 3), np.float32) * np.array(default, np.float32)
        return torch.from_numpy(a).to(dev)
    alb = tex_of(albedo, (0.5, 0.5, 0.5))
    mrt = tex_of(mr, (0.0, 0.5, 0.0))
    for i, (e, a) in enumerate(zip(elevs, azims)):
        for k, tx in (("albedo", alb), ("mr", mrt)):
            img = _render_tex(render, tx, e, a, size if k == "albedo" else 1024)
            Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)).save(
                os.path.join(vdir, "%s_%d.png" % (k, i)))
    with open(os.path.join(vdir, "cams.json"), "w") as f_:
        json.dump({"elevs": [float(x) for x in elevs],
                   "azims": [float(x) for x in azims],
                   "weights": [float(x) for x in weights],
                   "n": len(elevs)}, f_)
    np.savez(os.path.join(vdir, "uvmesh.npz"), v=v, f=f, uv=uv)
    del render, vp, alb, mrt
    torch.cuda.empty_cache()
    log("T prep views %d in %.1f" % (len(elevs), time.time() - t))
    return len(elevs)
