"""The back and both sides of a picture, drawn by Qwen-Image-Edit.

    views_qwen.py <Qwen root>      (the folder with engine/render.py,
                                    models/ and env/)

stdin, one per line: {"views": <views.json>} / "QUIT"
views.json: {"image": front picture, "parts": [{"kind": "view_back" |
"view_left" | "view_right", "out": png}, ...], "seed"}
stdout: READY / STEP <i> <frac> / PART <i> <png> / FAIL <i> <msg> /
ALLPARTS / JOBEND / LOG / ERROR
Qwen understands the turn around the figure far better than a plain
picture model: what hangs on the back, which way each side faces.
"""
import json
import os
import sys
import time
import traceback

ROOT = sys.argv[1]
sys.path.insert(0, os.path.join(ROOT, "engine_qwen"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["HF_HUB_OFFLINE"] = "1"

KEEP = ("Keep exactly the same framing and crop as the input picture: show "
        "only the parts that are visible in the input, do not add a body, "
        "legs or anything that is not in the input. The camera keeps the "
        "same distance and zoom, it does not move back or zoom out. Same "
        "pose, same size "
        "and position in the picture, the same clothing, hair, accessories, "
        "colors, materials and art style. Plain light gray studio "
        "background, soft studio light.")
TEXT = {
    "view_back": "Rotate the camera 180 degrees around the subject: show "
                 "the exact same subject from directly behind, a back "
                 "view, as close up and as large in the picture as in the "
                 "input, cropped at the same place. " + KEEP,
    "view_left": "Rotate the camera 90 degrees to the right around the "
                 "subject: show the exact same subject in a side profile "
                 "view from its left side, the subject faces the left edge "
                 "of the picture. " + KEEP,
    "view_right": "Rotate the camera 90 degrees to the left around the "
                  "subject: show the exact same subject in a side profile "
                  "view from its right side, the subject faces the right "
                  "edge of the picture. " + KEEP,
}
E = {}


def out(*a):
    print(*a, flush=True)


def engine():
    if "e" not in E:
        import render as R
        t = time.time()
        E["e"] = R.Engine()
        out("LOG Qwen ready in %.1fs" % (time.time() - t))
    return E["e"]


def blocks_on_card(pipe, torch, n=None):
    """How many of the image model's 60 parts sit on the card; the rest
    wait in main memory and come in step by step."""
    torch.cuda.empty_cache()
    if n is None:
        free = torch.cuda.mem_get_info()[0] / 1e9
        n = max(1, min(60, int((free - 3.0) / 0.19)))
    pipe.transformer.set_offload(True, use_pin_memory=False,
                                 num_blocks_on_gpu=n)
    torch.cuda.empty_cache()
    return n


def te_off(pipe, torch):
    """The 4-bit text model does not really leave the card when moved (its
    packed weights stay behind), so it is let go and read again from disk
    for the next job: a few seconds, against ~5 GB of card held all along."""
    if pipe.text_encoder is None:
        return
    before = torch.cuda.mem_get_info()[0] / 1e9
    pipe.text_encoder = None
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    out("LOG text model let go: %.1f -> %.1f GB free" % (
        before, torch.cuda.mem_get_info()[0] / 1e9))


def te_on(eng):
    pipe = eng.pipe
    if pipe.text_encoder is None:
        from transformers import Qwen2_5_VLForConditionalGeneration
        t = time.time()
        pipe.text_encoder = eng._text_encoder(
            Qwen2_5_VLForConditionalGeneration)
        out("LOG text model read in %.1fs" % (time.time() - t))


def park():
    """Step aside for Klein or the 3D engine: everything into main memory,
    nothing reloaded from disk next time."""
    if "e" not in E:
        return
    eng = E["e"]
    te_off(eng.pipe, eng.torch)
    if "t" in E:
        try:
            blocks_on_card(E["t"], eng.torch, 1)
        except Exception as exc:
            out("LOG Qwen-Image cannot step aside: %s" % exc)
    try:
        blocks_on_card(eng.pipe, eng.torch, 1)
        eng.pipe.vae.to("cpu")
    except Exception as exc:
        out("LOG image model cannot step aside: %s" % exc)
    eng.torch.cuda.empty_cache()


def views(jp):
    from PIL import Image
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qe
    jobs = json.load(open(jp, encoding="utf-8"))
    eng = engine()
    torch = eng.torch
    front = Image.open(jobs["image"]).convert("RGB")
    area = 1024 * 1024
    qe.VAE_IMAGE_SIZE = area
    W, H = qe.calculate_dimensions(area, front.width / float(front.height))
    img = front.resize((W, H), Image.LANCZOS)
    steps = 8
    t0 = time.time()
    pipe = eng.pipe
    # all instructions read first; then the text model steps off the card
    # and the image model gets that room (its parts no longer have to come
    # in from main memory at every step)
    cw, ch = qe.calculate_dimensions(qe.CONDITION_IMAGE_SIZE,
                                     front.width / float(front.height))
    cond = pipe.image_processor.resize(front, ch, cw)
    pes = []
    # the image model makes room, the text model reads all instructions,
    # then steps off the card and the image model takes that room
    try:
        blocks_on_card(pipe, torch, 1)
        pipe.vae.to("cuda")
        te_on(eng)
    except Exception as exc:
        out("LOG text model to the card: %s" % exc)
    for p in jobs["parts"]:
        with torch.inference_mode():
            pes.append(pipe.encode_prompt(prompt=TEXT[p["kind"]],
                                          image=[cond], device="cuda"))
    try:
        te_off(pipe, torch)
        n = blocks_on_card(pipe, torch)
        out("LOG %d of 60 image model parts on the card" % n)
    except Exception as exc:
        out("LOG text model stays on the card: %s" % exc)
    for i, p in enumerate(jobs["parts"]):
        try:
            t = time.time()

            def step_end(pp, k, tt, kw, i=i):
                out("STEP %d %.3f" % (i, (k + 1) / float(steps)))
                return kw
            gen = torch.Generator("cpu").manual_seed(int(jobs.get("seed", 0)))
            with torch.inference_mode():
                res = pipe(image=[img], prompt_embeds=pes[i][0],
                           prompt_embeds_mask=pes[i][1],
                           true_cfg_scale=1.0,
                           num_inference_steps=steps, height=H, width=W,
                           generator=gen,
                           callback_on_step_end=step_end).images[0]
            res.save(p["out"])
            out("LOG view %s %.1fs" % (p["kind"], time.time() - t))
            out("PART %d %s" % (i, p["out"]))
        except Exception as exc:
            traceback.print_exc()
            out("FAIL %d %s" % (i, str(exc)[:160]))
    out("LOG views in %.1fs" % (time.time() - t0))
    out("ALLPARTS")


def t2i_pipe():
    """Qwen-Image, the plain picture model (Apache 2.0, nunchaku 4-bit,
    Lightning 8 steps). It shares the text model and the VAE with the
    editing pipeline, so only its own image model is extra (main memory)."""
    eng = engine()
    if "t" not in E:
        import math
        from diffusers import (QwenImagePipeline,
                               FlowMatchEulerDiscreteScheduler)
        from nunchaku import NunchakuQwenImageTransformer2DModel
        from transformers import Qwen2Tokenizer
        torch = eng.torch
        t = time.time()
        mdir = os.path.join(ROOT, "models")
        tr = NunchakuQwenImageTransformer2DModel.from_pretrained(
            os.path.join(mdir, "nunchaku", "qwen-image",
                         "svdq-%s_r32-qwen-image-lightningv1.1-8steps"
                         ".safetensors" % (
                             "fp4" if torch.cuda.get_device_capability()[0]
                             >= 10 else "int4")))
        tr.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
        tr.to("cuda")
        sched = FlowMatchEulerDiscreteScheduler.from_config(
            eng.pipe.scheduler.config)
        tok = Qwen2Tokenizer.from_pretrained(
            os.path.join(mdir, "Qwen-Image", "tokenizer"))
        p = QwenImagePipeline(scheduler=sched, vae=eng.pipe.vae,
                              text_encoder=eng.pipe.text_encoder,
                              tokenizer=tok, transformer=tr)
        type(p)._execution_device = property(
            lambda self: torch.device("cuda"))
        E["t"] = p
        out("LOG Qwen-Image ready in %.1fs" % (time.time() - t))
    return E["t"]


def t2i_image(j, text):
    """The picture straight from the sentence with Qwen-Image."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qe
    eng = engine()
    torch = eng.torch
    t = time.time()
    tp = t2i_pipe()
    aw, ah = (int(v) for v in str(j.get("aspect", "2x3")).split("x"))
    W, H = qe.calculate_dimensions(1024 * 1536, aw / float(ah))
    W, H = W // 16 * 16, H // 16 * 16
    try:
        blocks_on_card(eng.pipe, torch, 1)
        blocks_on_card(tp, torch, 1)
        eng.pipe.vae.to("cuda")
        te_on(eng)
    except Exception as exc:
        out("LOG text model to the card: %s" % exc)
    tp.text_encoder = eng.pipe.text_encoder
    with torch.inference_mode():
        pe, pm = tp.encode_prompt(prompt=text, device="cuda")
    tp.text_encoder = None
    te_off(eng.pipe, torch)
    n = blocks_on_card(tp, torch)
    out("LOG %d of 60 Qwen-Image parts on the card" % n)
    steps = 8

    def step_end(pp, k, tt, kw):
        out("STEP %.3f" % ((k + 1) / float(steps)))
        return kw
    gen = torch.Generator("cpu").manual_seed(int(j.get("seed", 0)))
    with torch.inference_mode():
        img = tp(prompt_embeds=pe, prompt_embeds_mask=pm,
                 true_cfg_scale=1.0, num_inference_steps=steps,
                 height=H, width=W, generator=gen,
                 callback_on_step_end=step_end).images[0]
    blocks_on_card(tp, torch, 1)
    os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
    img.save(j["out"])
    out("LOG Qwen-Image picture %dx%d in %.1fs" % (W, H, time.time() - t))
    out("DONE " + j["out"])


def t2i(j):
    """A picture from a sentence. Qwen-Image-Edit paints onto the empty
    studio backdrop, so the subject arrives already standing on it."""
    from PIL import Image
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as qe
    import text_image as ti
    eng = engine()
    torch = eng.torch
    pipe = eng.pipe
    t = time.time()
    text = j["text"].strip()
    if j.get("preset"):
        text = text.rstrip(". ") + "." + ti.STYLE
    if j.get("model", "edit") == "image":
        return t2i_image(j, text)
    if "t" in E:
        blocks_on_card(E["t"], torch, 1)
    text = ("Paint onto this empty studio backdrop: " + text +
            " Fill the picture with the subject only, nothing else.")
    aw, ah = (int(v) for v in str(j.get("aspect", "2x3")).split("x"))
    area = 1024 * 1536
    W, H = qe.calculate_dimensions(area, aw / float(ah))
    qe.VAE_IMAGE_SIZE = W * H
    plate = j.get("plate")
    if plate and os.path.isfile(plate):
        base = Image.open(plate).convert("RGB").resize((W, H), Image.LANCZOS)
    else:
        base = Image.new("RGB", (W, H), (190, 190, 192))
    cw, ch = qe.calculate_dimensions(qe.CONDITION_IMAGE_SIZE, W / float(H))
    cond = pipe.image_processor.resize(base, ch, cw)
    try:
        blocks_on_card(pipe, torch, 1)
        pipe.vae.to("cuda")
        te_on(eng)
    except Exception as exc:
        out("LOG text model to the card: %s" % exc)
    with torch.inference_mode():
        pe = pipe.encode_prompt(prompt=text, image=[cond], device="cuda")
    te_off(pipe, torch)
    blocks_on_card(pipe, torch)
    steps = 8

    def step_end(pp, k, tt, kw):
        out("STEP %.3f" % ((k + 1) / float(steps)))
        return kw
    gen = torch.Generator("cpu").manual_seed(int(j.get("seed", 0)))
    with torch.inference_mode():
        img = pipe(image=[base], prompt_embeds=pe[0],
                   prompt_embeds_mask=pe[1], true_cfg_scale=1.0,
                   num_inference_steps=steps, height=H, width=W,
                   generator=gen, callback_on_step_end=step_end).images[0]
    os.makedirs(os.path.dirname(os.path.abspath(j["out"])), exist_ok=True)
    img.save(j["out"])
    out("LOG picture in %.1fs" % (time.time() - t))
    out("DONE " + j["out"])


def main():
    engine()
    out("READY")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        if line == "PARK":
            park()
            out("PARKED")
            continue
        try:
            j = json.loads(line)
            if "views" in j:
                views(j["views"])
            elif "t2i" in j:
                t2i(j["t2i"])
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
