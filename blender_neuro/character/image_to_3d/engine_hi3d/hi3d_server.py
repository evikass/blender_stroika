"""Hi3D (Hitem3D v1) refine server for the Image to 3D add-on.

One JSON job per stdin line: {"image", "mesh", "out", "resolution", "steps",
"seed"}. The coarse mesh is rebuilt with real high-resolution detail.

Only one network is on the graphics card at a time (image encoders, then the
512 network, the decoder, the 1024 network): the stock pipeline puts them
all there together, which fills a 12 GB card. Peak use is about 7 GB."""
import gc
import json
import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "models", "hi3d-official", "Hi3Dv1.0")
# the background remover rembg uses comes with the Hi3D download
os.environ.setdefault("U2NET_HOME", os.path.join(ROOT, "models", "u2net"))
_T0 = [time.time()]


def say(frac, text):
    print("PROGRESS %.3f %s" % (frac, text), flush=True)


def _low_memory():
    if os.environ.get("IMG3D_VRAM_GB") or os.environ.get("IMG3D_LOWMEM"):
        return True
    try:
        import psutil
        return psutil.virtual_memory().total < 28 * 2**30
    except Exception:
        return False


class Engine:
    """Each network is read from disk straight onto the graphics card when
    its turn comes. On a PC with little main memory it is dropped again
    right after (read again next time, a few seconds); with plenty of main
    memory it waits there between jobs."""

    def __init__(self):
        import torch
        from pathlib import Path
        from sparc3d import Sparc3DPipeline
        from sparc3d.pipelines import loader
        import hi3d_fix
        hi3d_fix.install_sdpa_attention()
        self.torch = torch
        say(0.02, "Loading Hi3D")
        self.root = Path(MODEL)
        self.loader = loader
        self.low = _low_memory()
        self.keep = {}
        p = Sparc3DPipeline.__new__(Sparc3DPipeline)
        p.scheduler = loader.load_scheduler(self.root)
        p.vae = None
        p.conditioner = None
        p.transformer_512 = None
        p._transformer_1024 = None
        p._transformer_1024_loader = None
        p.progress_callback = None
        p._sequential_cpu_offload = False
        p._device = torch.device("cuda")
        self.pipe = p
        print("low memory mode: %s" % self.low, flush=True)

    def get(self, name):
        """the network, on the graphics card"""
        torch = self.torch
        m = self.keep.pop(name, None)
        if m is not None:
            return m.to("cuda")
        if name == "conditioner":
            m = self.loader.load_conditioner(cache_dir=None,
                                             local_files_only=True)
            return m.to("cuda")
        if name == "vae":
            from sparc3d.models.autoencoders.config import load_sparconv_vae
            return load_sparconv_vae(self.root / "vae", device="cuda")
        from safetensors import safe_open
        from sparc3d.models.transformers import (Sparc3DTransformer,
                                                 Sparc3DTransformerConfig)
        cfg = Sparc3DTransformerConfig.from_dict(json.load(open(
            self.root / name / "config.json")))
        # an empty frame on the card, filled one weight at a time straight
        # from the file: no copy in main memory and no second copy on the
        # card (building it normally makes a 7 GB float32 copy first)
        with torch.device("meta"):
            m = Sparc3DTransformer(cfg)
        m = m.to_empty(device="cuda")
        from sparc3d.nn.rope import RotaryPositionEmbedder
        for mod in m.modules():
            if isinstance(mod, RotaryPositionEmbedder):
                fd = mod.freq_dim
                fr = torch.arange(fd, dtype=torch.float32) / fd
                mod.freqs = (1.0 / (10000 ** fr)).to("cuda")
        sd = m.state_dict()
        with safe_open(str(self.root / name / "model.safetensors"),
                       framework="pt", device="cuda") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                sd[k].copy_(t)
                del t
        del sd
        return m.eval()

    def done(self, name, m):
        """off the graphics card: dropped, or kept in main memory"""
        if self.low:
            del m
        else:
            self.keep[name] = m.to("cpu")
        self.free()

    def free(self):
        gc.collect()
        self.torch.cuda.empty_cache()

    def run(self, image, mesh, out, resolution=1024, steps=25, seed=0,
            steps2=15, **_):
        torch = self.torch
        import trimesh
        from PIL import Image
        from sparc3d.pipelines import sparc3d_pipeline as SP
        from sparc3d.utils.image import remove_background, prepare_image
        from sparc3d.sparcubes.postprocess import validate_mesh
        t0 = time.time()
        pipe = self.pipe
        res = int(resolution)
        if res not in SP.OUTPUT_RESOLUTIONS:
            res = 1024
        target = SP.OUTPUT_RESOLUTIONS.index(res)
        passes = target + 1
        say(0.03, "Reading picture")
        img = remove_background(Image.open(image))
        coarse = validate_mesh(trimesh.load(mesh, force="mesh",
                                            process=False))
        with torch.inference_mode():
            cn = self.get("conditioner")
            cond = cn(prepare_image(img)).to("cuda")
            self.done("conditioner", cn)
            del cn
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed))
            coords = pipe._adaptive_voxelize(coarse, 64, SP.PASS_BUDGETS[0])
            span = 0.9 / passes

            def prog(e):
                f = 0.06 + span * (e.pass_index + 0.8 * e.step /
                                   e.num_inference_steps)
                say(f, "Detail pass %d of %d" % (e.pass_index + 1, passes))
            pipe.progress_callback = prog
            out_mesh = None
            for li in range(target + 1):
                st = SP.STRUCTURE_RESOLUTIONS[li]
                tname = "transformer_512" if st == 64 else \
                    "transformer_1024"
                out_mesh = None
                for attempt in range(3):
                    tr = self.get(tname)
                    n_steps = int(steps) if li == 0 or not steps2 \
                        else int(steps2)
                    lat = pipe._sample_latents(tr, cond, coords, st,
                                               n_steps, gen, pass_index=li,
                                               num_passes=passes)
                    self.done(tname, tr)
                    del tr
                    say(0.06 + span * (li + 0.85),
                        "Decoding pass %d" % (li + 1))
                    fr, tot = torch.cuda.mem_get_info()
                    print("free before decode %.1f / %.1f GB" % (
                        fr / 2**30, tot / 2**30), flush=True)
                    pipe.vae = self.get("vae")
                    try:
                        dec = pipe._decode_latents(lat, st)
                        out_mesh = pipe._extract_mesh(dec)
                        del dec
                    except RuntimeError as exc:
                        # an empty decode now and then: sample this pass
                        # again with fresh noise instead of giving up
                        print("decode failed (%s), trying again" %
                              str(exc)[:80], flush=True)
                        out_mesh = None
                    del lat
                    self.done("vae", pipe.vae)
                    pipe.vae = None
                    if out_mesh is not None:
                        break
                if out_mesh is None:
                    raise RuntimeError("Hi3D could not rebuild this shape")
                print("pass %d: %d faces, peak %.1f GB" % (
                    li + 1, len(out_mesh.faces),
                    torch.cuda.max_memory_allocated() / 2**30), flush=True)
                if li < target:
                    coords = pipe._adaptive_voxelize(
                        out_mesh, SP.STRUCTURE_RESOLUTIONS[li + 1],
                        SP.PASS_BUDGETS[li + 1])
                    self.free()
            say(0.97, "Cleaning mesh")
            out_mesh = pipe._postprocess_mesh(
                out_mesh, target_face_count=SP.TARGET_FACE_COUNTS[target])
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        out_mesh.export(out)
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
                if eng is not None:
                    eng.free()
                print("PARKED", flush=True)
                continue
            job.pop("task", None)
            if eng is None:
                eng = Engine()
            eng.run(**job)
        except Exception as exc:
            traceback.print_exc()
            print("ERROR " + nice(exc), flush=True)
            try:
                eng.free()
            except Exception:
                pass
        print("READY", flush=True)


if __name__ == "__main__":
    serve()
