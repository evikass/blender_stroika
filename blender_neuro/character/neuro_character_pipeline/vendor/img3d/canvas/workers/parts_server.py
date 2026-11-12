"""Make the part pictures, check them and make the failed ones again; stays
loaded between jobs so the next Start or New Picture skips all loading.

    parts_server.py <engine dir> <vision root or "">

stdin, one per line: {"jobs": <jobs.json>} / "PARK" / "QUIT"
stdout per job: LOG / PART <i> <png> / STATE <i> checking|ok|redo|failed /
FAIL <i> <msg> / ALLPARTS / JOBEND
PARK moves everything to main memory so the card is free for other work;
the next job brings it back in a second or two.
"""
import os


def _dt(torch):
    """bfloat16 on RTX 30 and newer, float16 on older cards."""
    try:
        if not os.environ.get("IMG3D_FORCE_FP16") and \
                torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["HF_HUB_OFFLINE"] = "1"

import extract_parts as ep          # noqa: E402  (prompts and cut-outs)

ENG, VROOT = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
sys.path.insert(0, os.path.join(ENG, "engine_hy"))
import klein_worker_hy as kw        # noqa: E402

K = {}          # torch, pipe
V = {}          # the vision model that checks the pictures, and its reader


def out(*a):
    print(*a, flush=True)


def gpu_free():
    import gc
    import torch
    gc.collect()
    # the matrix library keeps a small work area on the card; it sits in the
    # middle of a big memory block and keeps all of it from being returned
    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    torch.cuda.empty_cache()


def klein():
    if "pipe" not in K:
        t = time.time()
        K["torch"], K["pipe"] = kw.build()
        out("LOG Klein ready in %.1fs" % (time.time() - t))
    pipe = K["pipe"]
    if next(pipe.transformer.parameters()).device.type != "cuda":
        t = time.time()
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        out("LOG Klein back on the card in %.1fs" % (time.time() - t))
    return K["torch"], pipe


def klein_park():
    if "pipe" in K and next(K["pipe"].transformer.parameters()).device.type \
            == "cuda":
        pipe = K["pipe"]
        pipe.transformer.to("cpu")
        pipe.vae.to("cpu")
        # small things left on the card (the last instruction read, buffers)
        # pin whole memory blocks: the card is not really freed, and the
        # next job then runs out of room and crawls
        try:
            getattr(pipe, "_pe_cache", {}).clear()
        except Exception:
            pass
        sch = getattr(pipe, "scheduler", None)
        for k in ("timesteps", "sigmas"):
            v = getattr(sch, k, None)
            if v is not None and hasattr(v, "is_cuda") and v.is_cuda:
                setattr(sch, k, v.cpu())
        for m in (pipe.transformer, pipe.vae):
            for b in m.buffers():
                if b.device.type == "cuda":
                    b.data = b.data.cpu()
        gpu_free()
        torch = K["torch"]
        left = torch.cuda.memory_reserved()
        if left > 256 * 2 ** 20:
            out("LOG still held on the card after parking: %d MB" % (
                left >> 20))


def vlm_park():
    if "m" in V and V.get("on_gpu"):
        try:
            V["m"].to("cpu")
        except Exception as exc:
            out("LOG checker cannot wait in main memory: %s" % exc)
            del V["m"]
        V["on_gpu"] = False
        gpu_free()


def _size(w, h, mp):
    area = mp * 1024 * 1024
    return (max(256, int(round(math.sqrt(area * w / float(h)) / 16)) * 16),
            max(256, int(round(math.sqrt(area * h / float(w)) / 16)) * 16))


def draw(torch, pipe, pe, inp, path, seed):
    """One picture at full size. The reference goes in at its own size (at
    most full size): blowing a small cut-out up adds no detail, only work."""
    from PIL import Image
    img = Image.open(inp).convert("RGB")
    w, h = img.size
    W, H = _size(w, h, 1.0)
    rw, rh = _size(w, h, max(0.25, min(1.0, w * h / (1024.0 * 1024.0))))
    ref = img.resize((rw, rh), Image.LANCZOS)
    g = torch.Generator("cuda").manual_seed(int(seed))
    with torch.inference_mode():
        res = pipe(prompt_embeds=pe, image=[ref], height=H, width=W,
                   num_inference_steps=4, generator=g).images[0]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    res.save(path)


def inputs(jobs, parts):
    """The picture and the instruction for every part."""
    from PIL import Image
    src = Image.open(jobs["image"]).convert("RGB")
    tmp = os.path.join(os.path.dirname(jobs["_path"]), "_crops")
    os.makedirs(tmp, exist_ok=True)
    res = []
    for i, p in enumerate(parts):
        src_p = p.get("src") or jobs["image"]
        im = src if src_p == jobs["image"] else \
            Image.open(src_p).convert("RGB")
        if p.get("mask") and os.path.isfile(p["mask"]) and \
                p["kind"] == "part":
            inp = ep.cut_out(im, p, os.path.join(tmp, "cut_%d.png" % i))
            text = ep.cut_prompt(p)
        else:
            inp, text = src_p, ep.prompt(p)
        res.append((inp, text))
    return res


def read_prompts(torch, pipe, texts):
    """All instructions read in one go: the prompt model comes onto the
    card once, not once per part."""
    import hashlib
    fresh = [t for t in texts if not os.path.isfile(os.path.join(
        kw.PE_DIR, hashlib.sha1(t.encode("utf-8")).hexdigest() + ".pt"))]
    swap = bool(fresh) and getattr(pipe, "_te_swap", False) and \
        pipe.text_encoder is not None
    if swap:
        pipe.text_encoder.to("cuda")
        pipe._te_swap = False
    try:
        return [kw.prompt_embeds(torch, pipe, t).clone() for t in texts]
    finally:
        if swap:
            pipe.text_encoder.to("cpu")
            pipe._te_swap = True
            gpu_free()


def make(jobs, idx, parts, seed, checked=()):
    """Draw the parts; each is shown the moment it is ready."""
    torch, pipe = klein()
    work = inputs(jobs, parts)
    t = time.time()
    pes = read_prompts(torch, pipe, [text for _, text in work])
    out("LOG prompts read in %.1fs" % (time.time() - t))
    done = []
    for k, (i, p) in enumerate(zip(idx, parts)):
        try:
            t = time.time()
            draw(torch, pipe, pes[k], work[k][0], p["out"], seed)
            out("LOG %s drawn in %.1fs" % (p["name"], time.time() - t))
            out("PART %d %s" % (i, p["out"]))
            out("STATE %d %s" % (i, "checking" if i in checked else "ok"))
            done.append(i)
        except Exception as exc:
            traceback.print_exc()
            out("FAIL %d %s" % (i, str(exc)[:160]))
    return done


ASK = ("Does this picture show ONLY the %s by itself, with no person, no "
       "character, no face, no head and no body anywhere in it? Answer "
       "only yes or no.")


def check(items):
    """Which pictures still show the character: [(i, path, name)] -> bad."""
    if not items or not VROOT:
        return []
    import torch
    from PIL import Image
    t = time.time()
    klein_park()
    models = os.path.join(VROOT, "models")
    if "m" not in V:
        from transformers import (AutoProcessor,
                                  Qwen2_5_VLForConditionalGeneration)
        nf4 = os.path.join(models, "text_encoder_nf4")
        full = os.path.join(models, "Qwen-Image-Edit-2509", "text_encoder")
        V["m"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            nf4 if os.path.isfile(os.path.join(nf4, "config.json")) else
            full, device_map={"": "cuda"}, dtype=_dt(torch))
        V["proc"] = AutoProcessor.from_pretrained(
            os.path.join(models, "Qwen-Image-Edit-2509", "processor"))
    elif not V.get("on_gpu"):
        V["m"].to("cuda")
    V["on_gpu"] = True
    out("LOG checker ready in %.1fs" % (time.time() - t))
    m, proc = V["m"], V["proc"]
    bad = []
    for i, path, name in items:
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((768, 768))
            q = ASK % name
            if "hair" in name.lower():
                q = ("Does this picture show ONLY hair by itself, like a "
                     "wig, with no face, no eyes, no skin and no body? "
                     "Answer only yes or no.")
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=[text], images=[img],
                       return_tensors="pt").to("cuda")
            with torch.inference_mode():
                o = m.generate(**inp, max_new_tokens=4, do_sample=False)
            a = proc.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].lower()
            ok = a.strip().startswith("yes")
            out("STATE %d %s" % (i, "ok" if ok else "redo"))
            if not ok:
                bad.append(i)
        except Exception as exc:
            out("STATE %d ok" % i)
            out("LOG check failed: %s" % exc)
    out("LOG checked in %.1fs" % (time.time() - t))
    vlm_park()
    return bad


def job(jp):
    jobs = json.load(open(jp, encoding="utf-8"))
    jobs["_path"] = jp
    parts = jobs["parts"]
    seed = int(jobs.get("seed", 0))
    t0 = time.time()
    checked = set(i for i, p in enumerate(parts) if p.get("kind") == "part") \
        if VROOT else set()
    done = make(jobs, list(range(len(parts))), parts, seed, checked)
    bad = check([(i, parts[i]["out"], parts[i]["name"]) for i in done
                 if i in checked])
    if bad:
        out("LOG again: %s" % ", ".join(parts[i]["name"] for i in bad))
        again = [dict(parts[i], strict=True) for i in bad]
        redone = make(jobs, bad, again, seed + 17)
        for i in bad:
            if i not in redone:
                out("STATE %d failed" % i)
    out("LOG all done in %.1fs" % (time.time() - t0))
    out("ALLPARTS")


B = {}          # BiRefNet, for the 3D Ready background


def birefnet_park():
    if "m" in B and B.get("on_gpu"):
        B["m"].to("cpu")
        B["on_gpu"] = False
        gpu_free()


def t2i(j):
    """A picture from a sentence; with the 3D Ready preset the subject is
    then laid on the studio backdrop."""
    import text_image as ti
    torch, pipe = klein()
    t = time.time()
    text = j["text"].strip()
    if j.get("preset"):
        text = text.rstrip(". ") + "." + ti.STYLE
    pe = read_prompts(torch, pipe, [text])[0]
    g = torch.Generator("cuda").manual_seed(int(j.get("seed", 0)))

    def step_end(p_, i, t_, kw_):
        out("STEP %.3f" % ((i + 1) / 4.0))
        return kw_
    with torch.inference_mode():
        img = pipe(prompt_embeds=pe, height=1536, width=1024,
                   num_inference_steps=4, generator=g,
                   callback_on_step_end=step_end).images[0]
    os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
    img.save(j["out"])
    out("LOG picture in %.1fs" % (time.time() - t))
    out("DONE " + j["out"])


def clean(path, c):
    import numpy as np
    import torch
    from PIL import Image
    if "m" not in B:
        from transformers import AutoModelForImageSegmentation
        B["m"] = AutoModelForImageSegmentation.from_pretrained(
            os.path.join(c["models"], "ZhengPeng7", "BiRefNet"),
            trust_remote_code=True, local_files_only=True).eval().half()
    if not B.get("on_gpu"):
        B["m"].cuda()
        B["on_gpu"] = True
    rgb = Image.open(path).convert("RGB")
    x = np.asarray(rgb.resize((1024, 1024), Image.Resampling.BILINEAR),
                   np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
        np.array([0.229, 0.224, 0.225], np.float32)
    tt = torch.from_numpy(x).permute(2, 0, 1)[None].half().cuda()
    with torch.no_grad():
        pred = B["m"](tt)[-1].sigmoid().float().cpu()[0, 0].numpy()
    a = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8)).resize(
        rgb.size, Image.Resampling.BILINEAR)
    bg = Image.open(c["plate"]).convert("RGB").resize(rgb.size,
                                                      Image.LANCZOS)
    bg.paste(rgb, (0, 0), a)
    dst = path[:-4] + "_ready.png"
    bg.save(dst)
    return dst


def refine(jp):
    """The quick 3D renders of the back and sides, redrawn in the look of
    the front picture."""
    import refine_views as rv
    from PIL import Image
    jobs = json.load(open(jp, encoding="utf-8"))
    torch, pipe = klein()
    front = Image.open(jobs["image"]).convert("RGB")
    pes = read_prompts(torch, pipe, [rv.text(p["kind"])
                                     for p in jobs["parts"]])
    t0 = time.time()
    for i, p in enumerate(jobs["parts"]):
        try:
            t = time.time()
            src = Image.open(p["out"]).convert("RGB")
            src.save(p["out"][:-4] + "_render.png")
            W, H = src.size
            W, H = W // 16 * 16, H // 16 * 16
            # the render only gives the angle and the outline, the front
            # picture the look: at full size both together overflow a
            # 12 GB card and crawl, at these sizes the result is the same
            src = src.resize(_size(W, H, 0.5), Image.LANCZOS)
            ref = front.resize(_size(front.width, front.height, 1.0),
                               Image.LANCZOS)
            g = torch.Generator("cuda").manual_seed(
                int(jobs.get("seed", 0)))
            with torch.inference_mode():
                res = pipe(prompt_embeds=pes[i], image=[src, ref],
                           height=H, width=W, num_inference_steps=4,
                           generator=g).images[0]
            if p["kind"] != "view_back":
                res = rv.facing_fixed(res, src.resize((W, H)))
            res.save(p["out"])
            out("LOG view %.1fs" % (time.time() - t))
            out("PART %d %s" % (i, p["out"]))
        except Exception as exc:
            traceback.print_exc()
            out("FAIL %d %s" % (i, str(exc)[:160]))
    out("LOG views in %.1fs" % (time.time() - t0))
    out("ALLPARTS")


SIDE = {"view_back": "seen from directly behind, showing its back",
        "view_left": "in a side profile view from its left side, facing "
                     "toward the left edge of the picture",
        "view_right": "in a side profile view from its left side, facing "
                      "toward the left edge of the picture"}


def sheet_text(kind):
    return ("This is a character turnaround sheet with two panels. The left "
            "panel shows the character from the front. The right panel is "
            "empty: fill it with the exact same character %s. Keep the left "
            "panel exactly as it is. The new view must match the front view "
            "exactly: the same body proportions, the same clothing, the same "
            "lengths of the coat, skirt and sleeves, the same bare or covered "
            "skin, the same hair, the same accessories and colors. Same "
            "size, same height, same ground line, full body from head to "
            "feet, plain light gray studio background, soft studio light."
            % SIDE[kind])


def full_text(kind):
    side = {"view_back": "seen from directly behind",
            "view_left": "seen from its left side, facing left",
            "view_right": "seen from its right side, facing right"}[kind]
    return ("The first image shows a character %s. The second image is the "
            "same character seen from the front. Redraw the first image as "
            "a clean, sharp, high quality full size picture, in exactly the "
            "art style, shading, colors and materials of the second image, "
            "with the same fine details. Keep exactly the view angle, pose, "
            "silhouette, proportions and position of the first image. Do "
            "not turn or mirror the character. Plain light gray studio "
            "background, soft studio light." % side)


def views(jp):
    """The back and both sides drawn from the front picture alone. The
    front picture is laid in the left half of a two-panel sheet and Klein
    fills the empty right half with the new view, so the new view is drawn
    right next to the real character. Klein turns figures to their left
    reliably but not to their right: the right side is drawn as the left
    side of the mirrored picture and mirrored back. The new panel is then
    drawn again at full size in the look of the front picture."""
    from PIL import Image, ImageOps
    jobs = json.load(open(jp, encoding="utf-8"))
    torch, pipe = klein()
    front = Image.open(jobs["image"]).convert("RGB")
    kinds = [p["kind"] for p in jobs["parts"]]
    pes = read_prompts(torch, pipe, [sheet_text(k) for k in kinds] +
                       [full_text(k) for k in kinds])
    n = len(kinds)
    pw, ph = _size(front.width, front.height, 0.85)
    panel = front.resize((pw, ph), Image.LANCZOS)
    bgc = front.getpixel((2, 2))
    W, H = _size(front.width, front.height, 1.0)
    full_ref = front.resize((W, H), Image.LANCZOS)
    t0 = time.time()
    for i, p in enumerate(jobs["parts"]):
        try:
            t = time.time()
            mirror = p["kind"] == "view_right"
            sheet = Image.new("RGB", (pw * 2, ph), bgc)
            sheet.paste(ImageOps.mirror(panel) if mirror else panel, (0, 0))
            g = torch.Generator("cuda").manual_seed(int(jobs.get("seed", 0)))
            with torch.inference_mode():
                res = pipe(prompt_embeds=pes[i], image=[sheet], height=ph,
                           width=pw * 2, num_inference_steps=4,
                           generator=g).images[0]
            crop = res.crop((pw, 0, pw * 2, ph))
            if mirror:
                crop = ImageOps.mirror(crop)
            pv = p["out"][:-4] + "_preview.png"
            crop.resize((W, H), Image.LANCZOS).save(pv)
            out("PREVIEW %d %s" % (i, pv))
            g = torch.Generator("cuda").manual_seed(int(jobs.get("seed", 0)))
            with torch.inference_mode():
                res = pipe(prompt_embeds=pes[n + i],
                           image=[crop, full_ref], height=H, width=W,
                           num_inference_steps=4, generator=g).images[0]
            res.save(p["out"])
            out("LOG view %s %.1fs" % (p["kind"], time.time() - t))
            out("PART %d %s" % (i, p["out"]))
        except Exception as exc:
            traceback.print_exc()
            out("FAIL %d %s" % (i, str(exc)[:160]))
    out("LOG views in %.1fs" % (time.time() - t0))
    out("ALLPARTS")


def main():
    klein()
    out("READY")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        if line == "PARK":
            klein_park()
            vlm_park()
            birefnet_park()
            out("PARKED")
            continue
        try:
            j = json.loads(line)
            if "t2i" in j:
                birefnet_park() if not j["t2i"].get("clean") else None
                t2i(j["t2i"])
            elif "views" in j:
                birefnet_park()
                views(j["views"])
            elif "refine" in j:
                birefnet_park()
                refine(j["refine"])
            else:
                birefnet_park()
                job(j["jobs"])
        except Exception as exc:
            traceback.print_exc()
            out("ERROR %s" % str(exc)[:200])
        out("JOBEND")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
