"""AI Render engine: Qwen-Image-Edit-2509 (Nunchaku 4-bit, Lightning 8 steps).

Run by the Blender add-on as its own process so Blender never loads torch.

  render.py --serve       one JSON job per stdin line, model stays loaded
  render.py --image a.png --out b.png --prompt "..."   one image, then exit

stdout: PROGRESS <0..1> <stage> / DONE <path> / ERROR <msg> / READY / TIME <s>
"""
import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")
BASE = os.path.join(MODELS, "Qwen-Image-Edit-2509")
QDIR = os.path.join(MODELS, "nunchaku", "lightning-251115")
TE4 = os.path.join(MODELS, "text_encoder_nf4")
IID = os.path.join(MODELS, "marigold-iid-lighting-v1-1")
MOGE = os.path.join(MODELS, "moge-2-vitl-normal")
os.environ["HF_HUB_OFFLINE"] = "1"

_T = {"t0": time.time(), "stage": ""}


def qfile():
    """FP4 for RTX 50 cards, INT4 for everything before them."""
    try:
        import torch
        major = torch.cuda.get_device_capability()[0]
    except Exception:
        major = 8
    kind = "fp4" if major >= 10 else "int4"
    return "svdq-%s_r32-qwen-image-edit-2509-lightning-8steps-251115" \
           ".safetensors" % kind


def dtype():
    """bfloat16 where the card has it (RTX 30 and newer), else float16."""
    import torch
    if os.environ.get("IMG3D_FORCE_FP16"):
        return torch.float16
    try:
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def te4_ok(path=None):
    """The 4-bit text model folder is complete: its config, index and every
    weight file it lists."""
    import json
    path = path or TE4
    try:
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(idx):
            files = set(json.load(open(idx))["weight_map"].values())
        else:
            files = {"model.safetensors"}
        return os.path.isfile(os.path.join(path, "config.json")) and all(
            os.path.getsize(os.path.join(path, f)) > 0 for f in files)
    except Exception:
        return False


def say(frac, text):
    if text != _T["stage"]:
        print("STAGE %.1f %s" % (time.time() - _T["t0"], text), flush=True)
        _T["stage"] = text
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


STYLES = {
    "PHOTO": "a photorealistic photograph with natural lighting, real "
             "materials and realistic shadows",
    "PRODUCT": "a professional studio product photograph, soft box lighting, "
               "clean background, sharp focus",
    "INTERIOR": "a photorealistic architectural interior photograph, soft "
                "daylight through the windows, realistic materials",
    "EXTERIOR": "a photorealistic architectural exterior photograph, "
                "golden hour sunlight, realistic sky and vegetation",
    "CINEMATIC": "a cinematic film still, dramatic moody lighting, shallow "
                 "depth of field, rich color grading",
    "NONE": "",
}


def build_prompt(style, extra):
    look = STYLES.get(style, STYLES["PHOTO"])
    parts = ["Render this 3D viewport screenshot as %s." % look if look
             else "Render this 3D viewport screenshot."]
    if extra and extra.strip():
        e = extra.strip()
        parts.append(e if e[-1] in ".!?" else e + ".")
    parts.append("Keep exactly the same composition, camera angle, "
                 "perspective, object shapes, proportions and layout.")
    return " ".join(parts)


class Engine:
    def __init__(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No NVIDIA GPU found")
        smooth_progress()
        self.torch = torch
        from diffusers import (QwenImageEditPlusPipeline,
                               FlowMatchEulerDiscreteScheduler)
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from transformers import Qwen2_5_VLForConditionalGeneration
        t = time.time()
        stage(0.02, 0.10, "Loading text model")
        te = self._text_encoder(Qwen2_5_VLForConditionalGeneration)
        torch.cuda.empty_cache()
        stage(0.10, 0.18, "Loading image model")
        tr = NunchakuQwenImageTransformer2DModel.from_pretrained(
            os.path.join(QDIR, qfile()))
        sched = FlowMatchEulerDiscreteScheduler.from_config({
            "base_image_seq_len": 256, "base_shift": math.log(3),
            "invert_sigmas": False, "max_image_seq_len": 8192,
            "max_shift": math.log(3), "num_train_timesteps": 1000,
            "shift": 1.0, "shift_terminal": None,
            "stochastic_sampling": False, "time_shift_type": "exponential",
            "use_beta_sigmas": False, "use_dynamic_shifting": True,
            "use_exponential_sigmas": False, "use_karras_sigmas": False})
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            BASE, transformer=tr, text_encoder=te, scheduler=sched,
            torch_dtype=dtype())
        pipe.vae.to("cuda")
        pipe.vae.enable_tiling()
        free = torch.cuda.mem_get_info()[0] / 1e9
        blocks = os.environ.get("AIR_GPU_BLOCKS")
        blocks = int(blocks) if blocks else \
            max(1, min(60, int((free - 4.0) / 0.19)))
        print("GPU free %.1f GB, %d blocks on GPU" % (free, blocks),
              flush=True)
        if blocks < 60:
            tr.set_offload(True, use_pin_memory=False,
                           num_blocks_on_gpu=blocks)
        tr.to("cuda")
        type(pipe)._execution_device = property(
            lambda self: torch.device("cuda"))
        _enc = pipe.encode_prompt

        def encode(*a, **k):
            ts = time.time()
            r = _enc(*a, **k)
            print("STAGE %.1f text encoded in %.1f s" % (
                time.time() - _T["t0"], time.time() - ts), flush=True)
            return r
        pipe.encode_prompt = encode
        self.pipe = pipe
        print("LOADED %.1f" % (time.time() - t), flush=True)

    def _text_encoder(self, cls):
        """The Qwen2.5-VL text model in 4 bit (5.5 GB), made once from the
        full one and kept, so it can stay on the GPU next to the image
        model."""
        import torch
        from transformers import BitsAndBytesConfig
        import shutil
        if te4_ok():
            return cls.from_pretrained(TE4, device_map={"": "cuda"},
                                       torch_dtype=dtype())
        if os.path.isdir(TE4):
            # a save that broke halfway (disk full, closed): make it again
            print("the small text model is incomplete: making it again",
                  flush=True)
            shutil.rmtree(TE4, ignore_errors=True)
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype())
        te = cls.from_pretrained(os.path.join(BASE, "text_encoder"),
                                 quantization_config=q,
                                 device_map={"": "cuda"},
                                 torch_dtype=dtype())
        tmp = TE4 + ".part"
        try:
            shutil.rmtree(tmp, ignore_errors=True)
            te.save_pretrained(tmp)
            if te4_ok(tmp):
                os.replace(tmp, TE4)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            print("could not keep the small text model (it is made again "
                  "next time): %s" % exc, flush=True)
        return te

    def run(self, image, out, prompt="", style="PHOTO", size="HIGH", seed=0,
            steps=8, **_):
        torch = self.torch
        from PIL import Image
        t0 = time.time()
        _T["t0"] = t0
        img = Image.open(image).convert("RGB")
        w, h = img.size
        mp = {"DRAFT": 0.5, "HIGH": 1.0, "ULTRA": 1.6}.get(size, 1.0)
        # The picture the model starts from and the one it paints must be
        # the same size and shape, or the model shifts and zooms the shot.
        import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus \
            as qe
        area = int(mp * 1024 * 1024)
        qe.VAE_IMAGE_SIZE = area
        W, H = qe.calculate_dimensions(area, w / float(h))
        img = img.resize((W, H), Image.LANCZOS)
        text = build_prompt(style, prompt)
        print("PROMPT " + text, flush=True)
        stage(0.20, 0.95, "Rendering")
        gen = torch.Generator("cpu").manual_seed(int(seed))

        def step_end(p, i, t, kw):
            print("STEP %d %.1f" % (i, time.time() - _T["t0"]), flush=True)
            return kw
        with torch.inference_mode():
            res = self.pipe(image=[img], prompt=text, negative_prompt=" ",
                            true_cfg_scale=1.0,
                            num_inference_steps=int(steps), height=H,
                            width=W, generator=gen,
                            callback_on_step_end=step_end).images[0]
        stage(0.96, 0.99, "Saving")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        res.save(out)
        print("TIME %.1f" % (time.time() - t0), flush=True)
        print("DONE " + os.path.abspath(out), flush=True)


# -- reading the light of a picture -----------------------------------------

LIGHT_PROMPT = """You are a lighting technical director. Study the lighting of
this photo-like render and answer ONLY with JSON in exactly this form:
{"key_light": {"kind": "sun|area|spot|point", "side": "left|right|front|back|top",
 "elevation": "low|medium|high", "color_hex": "#rrggbb",
 "hardness": "hard|medium|soft"},
 "through_opening": {"present": true|false, "kind": "none|window|door|blinds|leaves"},
 "fill_light": {"color_hex": "#rrggbb", "level": "very_low|low|medium|high"},
 "environment": {"floor": true|false, "back_wall": true|false, "sky_visible": true|false},
 "time_of_day": "dawn|morning|noon|afternoon|sunset|night|studio|unknown"}"""


class LightReader:
    """Takes a picture apart into what the light solver needs:
    - shading: the light and shadow alone, with the colours of things
      removed (Marigold IID Lighting)
    - depth and normals of everything in the picture, the floor and walls
      included, seen through the same camera (MoGe-2)
    - a description of the light in words (Qwen2.5-VL, already loaded)"""

    def __init__(self, engine):
        self.e = engine
        self.iid = None
        self.moge = None
        self.proc = None

    def shading(self, img):
        torch = self.e.torch
        import numpy as np
        if self.iid is None:
            from diffusers import MarigoldIntrinsicsPipeline
            self.iid = MarigoldIntrinsicsPipeline.from_pretrained(
                IID, torch_dtype=torch.float16, variant="fp16")
        self.iid.to("cuda")
        try:
            out = self.iid(img, num_inference_steps=4, ensemble_size=1,
                           processing_resolution=768,
                           generator=torch.Generator("cuda").manual_seed(0))
        finally:
            self.iid.to("cpu")
            torch.cuda.empty_cache()
        pred = np.asarray(out.prediction)
        names = list(getattr(self.iid, "target_names", None) or
                     ["albedo", "shading", "residual"])
        n = len(names)
        if pred.shape[0] % n == 0 and pred.shape[0] >= n:
            pred = pred.reshape((-1, n) + pred.shape[1:])[0]
        res = {}
        for k, name in enumerate(names):
            a = pred[k]
            if a.ndim == 3 and a.shape[0] in (1, 3):
                a = np.moveaxis(a, 0, -1)
            res[name] = np.clip(a.astype(np.float32), 0.0, None)
        return res

    def geometry(self, img, fov_x):
        torch = self.e.torch
        import numpy as np
        if self.moge is None:
            from moge.model.v2 import MoGeModel
            self.moge = MoGeModel.from_pretrained(
                os.path.join(MOGE, "model.pt"))
        self.moge.to("cuda").eval()
        try:
            t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
            t = t.permute(2, 0, 1).to("cuda")
            with torch.inference_mode():
                out = self.moge.infer(t, fov_x=float(fov_x))
        finally:
            self.moge.to("cpu")
            torch.cuda.empty_cache()
        res = {}
        for k in ("depth", "normal", "mask", "points"):
            if k in out and out[k] is not None:
                res[k] = out[k].float().cpu().numpy() \
                    if hasattr(out[k], "float") else np.asarray(out[k])
        return res

    def describe(self, img):
        torch = self.e.torch
        import json as _j
        from transformers import AutoProcessor
        if self.proc is None:
            self.proc = AutoProcessor.from_pretrained(
                os.path.join(BASE, "processor"))
        te = self.e.pipe.text_encoder
        im = img.copy()
        im.thumbnail((896, 896))
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": LIGHT_PROMPT}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[text], images=[im], return_tensors="pt").to(
            "cuda")
        with torch.inference_mode():
            out = te.generate(**inp, max_new_tokens=300, do_sample=False)
        txt = self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
        a, b = txt.find("{"), txt.rfind("}")
        try:
            return _j.loads(txt[a:b + 1]), txt
        except Exception:
            return None, txt

    def run(self, image, out_dir, fov_x=50.0, **_):
        import numpy as np
        from PIL import Image
        t0 = time.time()
        _T["t0"] = t0
        os.makedirs(out_dir, exist_ok=True)
        img = Image.open(image).convert("RGB")
        stage(0.03, 0.40, "Reading light and shadow")
        for name, a in self.shading(img).items():
            np.save(os.path.join(out_dir, name + ".npy"), a)
        stage(0.42, 0.70, "Reading the shape of the scene")
        for name, a in self.geometry(img, fov_x).items():
            np.save(os.path.join(out_dir, "geo_" + name + ".npy"), a)
        stage(0.72, 0.97, "Describing the light")
        desc, raw = self.describe(img)
        with open(os.path.join(out_dir, "light.json"), "w",
                  encoding="utf8") as f:
            json.dump({"desc": desc, "raw": raw}, f, indent=1)
        print("LIGHT " + json.dumps(desc), flush=True)
        print("TIME %.1f" % (time.time() - t0), flush=True)
        print("DONE " + os.path.abspath(out_dir), flush=True)


def nice(exc):
    msg = str(exc).strip().splitlines()[-1] if str(exc).strip() \
        else type(exc).__name__
    if "out of memory" in msg.lower():
        msg = "Not enough GPU memory - try Size: Draft"
    return msg[:200]


def serve():
    eng = None
    reader = None
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
                _T["t0"] = time.time()
                say(0.01, "Starting")
                eng = Engine()
            if job.get("cmd") == "light":
                if reader is None:
                    reader = LightReader(eng)
                reader.run(**job)
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
        ap = argparse.ArgumentParser()
        ap.add_argument("--serve", action="store_true")
        ap.add_argument("--image")
        ap.add_argument("--out")
        ap.add_argument("--prompt", default="")
        ap.add_argument("--style", default="PHOTO")
        ap.add_argument("--size", default="HIGH")
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--steps", type=int, default=8)
        a = ap.parse_args()
        if a.serve:
            serve()
        else:
            say(0.01, "Starting")
            e = Engine()
            e.run(a.image, a.out, a.prompt, a.style, a.size, a.seed, a.steps)
    except Exception as exc:
        traceback.print_exc()
        print("ERROR " + nice(exc), flush=True)
        sys.exit(1)
