"""Enhance the painted views of a model with FLUX.2 [klein] (from the AI
Render install): each side view is enlarged to 2048 and repainted in four
overlapping tiles, which are colour-matched to the original and blended
softly, so the texture gets real new detail without visible tile seams.

    klein_views.py <views dir>    writes albedo_ai_<n>.png
"""
import json, os, sys, time, subprocess
import numpy as np
from PIL import Image

AIR = os.environ.get("AIR_ROOT", "")
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT = ("Enhance this image into a sharp, high resolution, highly detailed "
          "version of itself. Keep exactly the same shapes, outlines, position, "
          "framing and colors with flat even lighting. Add fine realistic "
          "surface detail, small scratches, pores and crisp material texture. "
          "Do not add shadows, highlights, reflections or new objects.")
S = 2048          # each view is worked on at this size
T = 1280          # tile size: two per side, overlapping by 2*T - S = 512
STARTS = (0, S - T)


def describe(img_path):
    """Name the materials in the picture with the vision model that ships
    with Qwen-Image-Edit (the AI Render install), so Klein paints the right
    kind of surface detail. Returns '' if it cannot."""
    try:
        import torch
        from transformers import (Qwen2_5_VLForConditionalGeneration,
                                  AutoProcessor)
        M = os.path.join(AIR, "models")
        te = os.path.join(M, "text_encoder_nf4")
        if not os.path.isfile(os.path.join(te, "config.json")):
            return ""
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            te, device_map={"": "cuda"}, dtype=torch.bfloat16)
        proc = AutoProcessor.from_pretrained(
            os.path.join(M, "Qwen-Image-Edit-2509", "processor"))
        img = Image.open(img_path).convert("RGB").resize((512, 512))
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text":
             "List the surface materials of this object as short phrases "
             "with their colour and condition, comma separated, at most 25 "
             "words, nothing else. Example: worn blue leather armor plates, "
             "scratched bronze trim, cracked ivory bone."}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model.generate(**inp, max_new_tokens=60, do_sample=False)
        ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        del model
        torch.cuda.empty_cache()
        return " ".join(ans.replace("\n", " ").split())[:300]
    except Exception as exc:
        print("LOG no material description: %s" % exc, flush=True)
        return ""


def worker():
    p = subprocess.Popen([sys.executable, "-u",
                          os.path.join(os.path.dirname(
                              os.path.abspath(__file__)),
                              "klein_worker_hy.py")],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1,
                         encoding="utf-8", errors="replace",
                         creationflags=0x08000000)
    for l in p.stdout:
        l = l.rstrip()
        if l == "READY":
            return p
        if l.startswith("ERROR"):
            raise RuntimeError(l)
    raise RuntimeError("Klein did not start")


def paint(p, src, tmp, prompt=PROMPT):
    ip, op = tmp + "_in.png", tmp + "_out.png"
    src.save(ip)
    p.stdin.write(json.dumps({"image": ip, "out": op, "text": prompt,
                              "size": "HIGH", "seed": 0}) + "\n")
    p.stdin.flush()
    for l in p.stdout:
        l = l.rstrip()
        if l.startswith("DONE"):
            return np.asarray(Image.open(op).convert("RGB").resize(
                src.size, Image.Resampling.LANCZOS)).astype(np.float32)
        if l.startswith("ERROR"):
            raise RuntimeError(l)
    raise RuntimeError("Klein stopped")


def no_shine(res, ref, m, room=18.0):
    """Take out bright glints the model painted in: no pixel may get much
    brighter than the original around it (fine detail stays, shine goes)."""
    try:
        from PIL import ImageFilter
        lum = lambda x: x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114
        base = np.asarray(Image.fromarray(np.clip(ref, 0, 255).astype(
            np.uint8)).filter(ImageFilter.GaussianBlur(6))).astype(np.float32)
        cap = lum(base) + room
        l = lum(res)
        k = np.where(l > cap, cap / np.maximum(l, 1e-3), 1.0)
        k = np.where(m, k, 1.0)
        return res * k[..., None]
    except Exception:
        return res


def match(res, ref, m):
    """same overall colour and brightness as the original tile"""
    if m.sum() < 100:
        return res
    out = res.copy()
    for c in range(3):
        a, b = res[..., c][m], ref[..., c][m]
        sa, sb = a.std() + 1e-3, b.std() + 1e-3
        out[..., c] = (res[..., c] - a.mean()) * (sb / sa) + b.mean()
    return np.clip(out, 0, 255)


def ramp(n, ov, left, right):
    r = np.ones(n, np.float32)
    x = (np.arange(ov) + 0.5) / ov
    if left:
        r[:ov] = x
    if right:
        r[-ov:] = x[::-1]
    return r


MIN_COVER = 0.15   # a tile with less of the model than this keeps its colours


def creative(c):
    """creativity 0..1 -> (input softening, colour-match strength, prompt)"""
    c = max(0.0, min(1.0, float(c)))
    pr = PROMPT
    if c >= 0.35:
        pr += (" Add rich, intricate new surface detail: heavy wear, deep "
               "scratches, weathering and fine material grain.")
    if c >= 0.7:
        pr += (" Be bold and inventive with the detail: chipped edges, "
               "grime in the crevices, subtle colour variation and patina.")
    return 1.0 - 0.5 * c, 1.0 - 0.7 * c, pr


def soften(img, f):
    if f >= 0.999:
        return img
    w, h = img.size
    small = img.resize((max(64, int(w * f)), max(64, int(h * f))),
                       Image.Resampling.LANCZOS)
    return small.resize((w, h), Image.Resampling.BICUBIC)


def process(p, vdir, prompt=PROMPT, mode="HIGH", creativity=0.0):
    soft, keep, prompt = creative(creativity)
    cams = json.load(open(os.path.join(vdir, "cams.json")))
    side = list(range(cams["n"]))   # every painted view, top and bottom too
    ov = 2 * T - S
    tmp = os.path.join(vdir, "_klein")
    t0 = time.time()
    for k, v in enumerate(side):
        tv = time.time()
        big = Image.open(os.path.join(vdir, "albedo_%d.png" % v)).convert(
            "RGB").resize((S, S), Image.Resampling.LANCZOS)
        a = np.asarray(big).astype(np.float32)
        bg = a[4, 4].copy()
        obj = np.abs(a - bg).max(axis=2) > 6
        if mode == "FAST":
            # the whole view in one go: quicker, a little less fine detail
            res = paint(p, soften(big, soft), tmp, prompt)
            mt = match(res, a, obj)
            out = mt * keep + np.clip(res, 0, 255) * (1.0 - keep)
            out[~obj] = a[~obj]
            Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
                os.path.join(vdir, "albedo_ai_%d.png" % v))
            print("VIEW %d/%d %.1fs" % (k + 1, len(side), time.time() - tv),
                  flush=True)
            continue
        acc = np.zeros_like(a)
        wsum = np.zeros(a.shape[:2], np.float32)
        for yi, y in enumerate(STARTS):
            for xi, x in enumerate(STARTS):
                ref = a[y:y + T, x:x + T]
                m = obj[y:y + T, x:x + T]
                if m.mean() < MIN_COVER:
                    res = ref
                else:
                    res0 = paint(p, soften(Image.fromarray(
                        ref.astype(np.uint8)), soft), tmp, prompt)
                    res = match(res0, ref, m) * keep + \
                        np.clip(res0, 0, 255) * (1.0 - keep)
                wy = ramp(T, ov, yi > 0, yi < len(STARTS) - 1)
                wx = ramp(T, ov, xi > 0, xi < len(STARTS) - 1)
                w = np.outer(wy, wx)
                acc[y:y + T, x:x + T] += res * w[..., None]
                wsum[y:y + T, x:x + T] += w
        out = acc / np.maximum(wsum, 1e-6)[..., None]
        out[~obj] = a[~obj]
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(
            os.path.join(vdir, "albedo_ai_%d.png" % v))
        print("VIEW %d/%d %.1fs" % (k + 1, len(side), time.time() - tv),
              flush=True)
    for f in ("_klein_in.png", "_klein_out.png"):
        try:
            os.remove(os.path.join(vdir, f))
        except Exception:
            pass
    print("KLEIN VIEWS %.1f" % (time.time() - t0), flush=True)


def delight(vdir, views, strength=0.7):
    """Marigold (intrinsic image decomposition) separates each repainted
    view into pure colour, shading and shine; the shine the reference
    picture carried is taken out of the colour, the fine detail stays."""
    try:
        import torch
        from PIL import ImageFilter
        from diffusers import MarigoldIntrinsicsPipeline
        mp = os.path.join(ENGINE_ROOT, "models", "prs-eth",
                          "marigold-iid-lighting-v1-1")
        if not os.path.isfile(os.path.join(mp, "model_index.json")) and AIR:
            mp = os.path.join(AIR, "models", "marigold-iid-lighting-v1-1")
        if not os.path.isfile(os.path.join(mp, "model_index.json")):
            return
        t0 = time.time()
        pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            mp, torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.set_progress_bar_config(disable=True)

        def srgb(x):
            x = np.clip(x, 0, 1)
            return np.where(x <= 0.0031308, 12.92 * x,
                            1.055 * np.power(x, 1 / 2.4) - 0.055)
        for v in views:
            p_ = os.path.join(vdir, "albedo_ai_%d.png" % v)
            E = Image.open(p_).convert("RGB")
            W, H = E.size
            a = np.asarray(E).astype(np.float32) / 255.0
            bg = a[4, 4].copy()
            obj = np.abs(a - bg).max(axis=2) > 6 / 255.0
            pred = np.asarray(pipe(E, num_inference_steps=4,
                                   processing_resolution=768,
                                   output_type="np").prediction)
            alb = Image.fromarray((srgb(pred[0]) * 255).astype(np.uint8))
            alb = np.asarray(alb.resize((W, H), Image.Resampling.LANCZOS)
                             ).astype(np.float32) / 255.0
            blur = np.asarray(E.filter(ImageFilter.GaussianBlur(4))
                              ).astype(np.float32) / 255.0
            detail = np.clip(a / np.maximum(blur, 0.02), 0, 3)
            d = np.clip(alb * detail, 0, 1)
            out = a * (1 - strength) + d * strength
            out[~obj] = a[~obj]
            Image.fromarray((out * 255).astype(np.uint8)).save(p_)
        del pipe
        torch.cuda.empty_cache()
        print("LOG delit %d views in %.1f s" % (len(views), time.time() - t0),
              flush=True)
    except Exception as exc:
        print("LOG no delight: %s" % exc, flush=True)


def park(p):
    """Klein waits in main memory; False when it could not leave the
    graphics card (then it is closed, so the card is free for the rest)."""
    p.stdin.write("PARK\n")
    p.stdin.flush()
    for l in p.stdout:
        if l.startswith("PARKED"):
            try:
                return int(l.split()[1]) < 1200
            except Exception:
                return True
    return False


def main():
    t0 = time.time()
    p = worker()
    print("LOADED %.1f" % (time.time() - t0), flush=True)
    process(p, sys.argv[1])
    try:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
    except Exception:
        pass


def _small():
    if os.environ.get("IMG3D_VRAM_GB") or os.environ.get("IMG3D_LOWMEM"):
        return True
    try:
        import psutil
        if psutil.virtual_memory().total < 28 * 2**30:
            return True
    except Exception:
        pass
    try:
        import subprocess as sp_
        o = sp_.run(["nvidia-smi", "--query-gpu=memory.total",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, creationflags=0x08000000).stdout
        return float(o.split()[0]) < 10.5 * 1024
    except Exception:
        return False


def serve():
    """stay up: one views folder per stdin line; Klein is loaded once and
    waits in main memory between jobs"""
    p = None
    print("READY", flush=True)
    for line in sys.stdin:
        parts = line.rstrip("\n").split("\t")
        vdir = parts[0].strip()
        mode = parts[1].strip().upper() if len(parts) > 1 else "HIGH"
        try:
            cre = float(parts[2]) if len(parts) > 2 else 0.0
        except ValueError:
            cre = 0.0
        if not vdir:
            continue
        if vdir == "QUIT":
            break
        try:
            t0 = time.time()
            if p is None or p.poll() is not None:
                p = worker()
            print("LOADED %.1f" % (time.time() - t0), flush=True)
            process(p, vdir, mode=mode, creativity=cre)
            if _small():
                # small PC: Klein leaves main memory too (it loads again
                # from disk next time, a few seconds)
                try:
                    p.stdin.write("QUIT\n")
                    p.stdin.flush()
                    p.wait(timeout=60)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                p = None
            elif not park(p):
                try:
                    p.stdin.write("QUIT\n")
                    p.stdin.flush()
                    p.wait(timeout=60)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                p = None
            # shine out of the colour, with Klein waiting in main memory
            cams_ = json.load(open(os.path.join(vdir, "cams.json")))
            delight(vdir, list(range(cams_["n"])))
            with open(os.path.join(vdir, "albedo_ai_mode.txt"), "w") as f_:
                f_.write("%s %.2f" % (mode, cre))
            print("KLEIN DONE", flush=True)
        except Exception as exc:
            print("ERROR %s" % str(exc)[:200], flush=True)
            print("KLEIN DONE 0", flush=True)
    if p is not None:
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
        except Exception:
            pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve()
    elif len(sys.argv) > 2 and sys.argv[1] == "--describe":
        print("DESC " + describe(sys.argv[2]), flush=True)
    else:
        main()
