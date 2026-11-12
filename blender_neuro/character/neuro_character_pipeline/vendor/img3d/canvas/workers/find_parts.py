"""Look at a picture and list the parts a 3D artist would model separately.

    find_parts.py <image> <out.json> <models dir>

Runs Qwen2.5-VL (the vision model inside Qwen-Image-Edit) and writes
[{"name", "description", "bbox": [x1,y1,x2,y2] in image pixels}, ...].
stdout: LOG <text> / DONE <out.json> / ERROR <msg>
"""
import os


def _nf4_ok(path):
    """The 4-bit copy is complete (a broken save is skipped)."""
    import json
    try:
        idx = os.path.join(path, "model.safetensors.index.json")
        files = set(json.load(open(idx))["weight_map"].values()) \
            if os.path.isfile(idx) else {"model.safetensors"}
        return os.path.isfile(os.path.join(path, "config.json")) and all(
            os.path.getsize(os.path.join(path, f)) > 0 for f in files)
    except Exception:
        return False


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
import os
import re
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"

ASK = (
    "You are a 3D artist preparing this picture to be built as separate 3D "
    "meshes for a game. Look carefully and break the subject into every "
    "separable part a 3D artist would model as its own mesh (for a "
    "character: hair, head, each clothing item, each accessory, footwear, "
    "belts, bags and pouches, gloves, jewelry, "
    "weapons or held items; a pair of matching items (two boots, two "
    "gloves) is ONE part; for an object: each distinct component). Hair "
    "and head are always separate parts. Every part must be ONE simple "
    "piece that is modeled on its own: never group several pieces into one "
    "part. Split armor into its plates (helmet, chest plate, shoulder "
    "pads, arm guards or gauntlets, belt, skirt plates, leg guards); split "
    "an outfit into its garments. A part should not cover more than about "
    "a third of the subject's height unless it truly is one continuous "
    "piece such as a cape or a long dress. Only list things you can actually "
    "see. For each part give: name (short, 1-3 words), description (at "
    "most 10 words: what it is, color, material), "
    "and bbox_2d as [x1,y1,x2,y2] pixel coordinates of that part in the "
    "picture. Answer with only a JSON list.")


def out(*a):
    print(*a, flush=True)


SPLIT = (
    "This picture shows the %s of a character (%s). A 3D artist will model "
    "it as several separate pieces. Break it into the separate pieces "
    "that are clearly VISIBLE in this picture (for armor that can be a "
    "chest plate, shoulder pads, arm guards, gauntlets, belt, skirt "
    "plates, leg guards; for clothing: each garment and each attached "
    "accessory). Never invent a piece that is not there: no helmet on a "
    "bare head, no back plate that cannot be seen from the front, and do "
    "not list the same area twice under two names. A pair "
    "of matching pieces is ONE piece. For each piece give: name (short, "
    "1-3 words), description (shape, color, material, details) and bbox_2d "
    "as [x1,y1,x2,y2] pixel coordinates in this picture. Answer with only "
    "a JSON list.")
SHAPE = (
    "Look closely at the %s of this character (%s). The other parts are: "
    "%s. Describe ONLY this piece, so that someone who cannot see the "
    "picture can draw exactly its real shape: what kind of object it "
    "really is (for example a flat panel, a curved plate, a ring, a tube, "
    "a strap, a garment), its outline and edges (pointed, rounded, "
    "straight), whether it is one piece or a pair, where it sits on the "
    "body and how it hangs, its colors, material and trims. If it could be "
    "mistaken for something else, add a sentence like 'It is a ..., not "
    "a ...'. Two or three sentences, no list.")
MISSING = (
    "These separate parts were already found on this character: %s. Look "
    "again very carefully, also at small pieces and at the layers under "
    "other pieces: an undershirt or bodysuit under armor or a jacket, "
    "gloves, knee guards, elbow guards, socks or stockings, straps, "
    "jewelry, pouches, buckles worn as separate items. List every "
    "separate piece you can clearly see that is NOT in that list yet, "
    "with name (short, 1-3 words), description and bbox_2d as "
    "[x1,y1,x2,y2]. A pair of matching pieces is ONE piece. If nothing is "
    "missing, answer []. Answer with only a JSON list.")
SHAPES = (
    "For each of these separate pieces of this character: %s. Describe "
    "exactly its real shape so someone who cannot see the picture can draw "
    "it: what kind of object it really is (flat panel, curved plate, ring, "
    "strap, garment...), its outline and edges, where it sits and how it "
    "hangs, colors, material and trims, in one or two sentences; add 'It "
    "is a ..., not a ...' if it could be mistaken for something else. Also "
    "say whether it is a pair of two mirror-image pieces on the left and "
    "right side of the body. Answer with only a JSON list of "
    "{\"name\": ..., \"shape\": ..., \"pair\": \"yes\" or \"no\"}.")
PAIR = (
    "Look at the %s of this character. Is it a pair of two matching "
    "pieces, one on the left and one on the right side of the body, that "
    "are mirror images of each other with the same design? Answer only "
    "yes or no.")
BODY = (
    "Describe the body of this character under the clothes in one short "
    "sentence, for an artist who has to model its bare base body: say first "
    "whether it is a woman, a man, a girl, a boy or a creature, then the "
    "body build (slim, athletic, muscular, heavy...), the age and the skin "
    "color. Answer with only the sentence.")
GROUP = ("suit", "armor", "armour", "outfit", "set", "costume", "clothing",
         "clothes", "uniform", "gear", "equipment")
WHOLE = ("cape", "cloak", "robe", "dress", "body", "hair", "tail", "wing",
         "head", "face", "boot", "shoe")


def items(lst):
    """The list as it stands, for the canvas to show while it grows."""
    out("ITEMS " + json.dumps([{"name": p["name"], "description":
                                p.get("description", ""), "bbox":
                                p.get("bbox"), "mirror": p.get("mirror",
                                                               False)}
                               for p in lst]))


M = {}


def load(models):
    """The vision model, once; a server keeps it for the next picture."""
    if M:
        return M["torch"], M["m"], M["proc"]
    import torch
    from transformers import (AutoProcessor,
                              Qwen2_5_VLForConditionalGeneration)
    t = time.time()
    nf4 = os.path.join(models, "text_encoder_nf4")
    full = os.path.join(models, "Qwen-Image-Edit-2509", "text_encoder")
    m = None
    if os.path.isfile(os.path.join(full, "config.json")) and \
            os.environ.get("IMG3D_VLM_8BIT"):
        # the full model in 8 bits sees small shapes much better than the
        # 4-bit copy; fall back to that one when the card is too small
        try:
            from transformers import BitsAndBytesConfig
            m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                full, device_map={"": "cuda"}, dtype=_dt(torch),
                quantization_config=BitsAndBytesConfig(load_in_8bit=True))
            out("LOG vision model: 8-bit")
        except Exception as exc:
            out("LOG 8-bit failed (%s), using 4-bit" % str(exc)[:80])
            m = None
            torch.cuda.empty_cache()
    if m is None and _nf4_ok(nf4):
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            nf4, device_map={"": "cuda"}, dtype=_dt(torch))
    if m is None:
        # no small copy yet (Prompt makes it on its first run): shrink the
        # full model to 4 bits while loading, never load it whole (16 GB)
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=_dt(torch))
        m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            full, device_map={"": "cuda"}, dtype=_dt(torch),
            quantization_config=q)
        out("LOG vision model: 4-bit made while loading")
        tmp = nf4 + ".part"
        try:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            m.save_pretrained(tmp)
            if _nf4_ok(tmp):
                shutil.rmtree(nf4, ignore_errors=True)
                os.replace(tmp, nf4)
            else:
                shutil.rmtree(tmp, ignore_errors=True)
        except Exception as exc:
            out("LOG could not keep the 4-bit copy: %s" % str(exc)[:80])
    proc = AutoProcessor.from_pretrained(
        os.path.join(models, "Qwen-Image-Edit-2509", "processor"))
    out("LOG loaded in %.1f s" % (time.time() - t))
    M.update(torch=torch, m=m, proc=proc)
    return torch, m, proc


def run(img_path, out_path, models):
    from PIL import Image
    torch, m, proc = load(models)
    out("PROGRESS 0.10 Looking at the picture")
    img = Image.open(img_path).convert("RGB")
    # the vision model reads a smaller copy: plenty to name the parts, and
    # several times quicker; boxes are scaled back to the real picture
    small = img.copy()
    small.thumbnail((640, 960))
    fx = img.width / float(small.width)
    fy = img.height / float(small.height)

    def ask(picture, question, ox=0, oy=0):
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[picture],
                   return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = m.generate(**inp, max_new_tokens=2000, do_sample=False)
        ans = proc.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        # the model answers in the resized picture's pixels
        gh = int(inp["image_grid_thw"][0][1]) * 14
        gw = int(inp["image_grid_thw"][0][2]) * 14
        sx, sy = picture.width / float(gw), picture.height / float(gh)
        try:
            parts = json.loads(re.search(r"\[.*\]", ans, re.S).group(0))
        except Exception:
            return []
        res = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            b = p.get("bbox_2d") or p.get("bbox") or [0, 0, gw, gh]
            name = str(p.get("name", "Part")).split("-")[-1].strip() or \
                "Part"
            res.append({"name": name,
                        "description": str(p.get("description", "")),
                        "bbox": [int(b[0] * sx) + ox, int(b[1] * sy) + oy,
                                 int(b[2] * sx) + ox, int(b[3] * sy) + oy]})
        return res

    res = [dict(p, bbox=[int(p["bbox"][0] * fx), int(p["bbox"][1] * fy),
                         int(p["bbox"][2] * fx), int(p["bbox"][3] * fy)])
           for p in ask(small, ASK)]
    items(res)
    out("PROGRESS 0.25 Breaking big parts into pieces")
    # big parts that are really several pieces: ask again on that piece
    if res:
        top = min(p["bbox"][1] for p in res)
        bot = max(p["bbox"][3] for p in res)
        tall = max(1, bot - top)
    final = []
    for p in res:
        nm = p["name"].lower()
        x1, y1, x2, y2 = p["bbox"]
        big = (y2 - y1) > 0.42 * tall
        dsc = p.get("description", "").lower()
        group = any(g in nm for g in GROUP) or sum(
            w in dsc for w in ("shoulder", "arm guard", "gauntlet", "chest",
                               "plate", "greave", "guard", "pad")) >= 2 or \
            " including " in dsc or nm in ("top", "bottom", "upper body",
                                           "lower body")
        whole = any(g in nm for g in WHOLE)
        if (group or big) and not whole:
            w, h = x2 - x1, y2 - y1
            cx1, cy1 = max(0, int(x1 - .08 * w)), max(0, int(y1 - .08 * h))
            cx2 = min(img.width, int(x2 + .08 * w))
            cy2 = min(img.height, int(y2 + .08 * h))
            crop = img.crop((cx1, cy1, cx2, cy2))
            cs = crop.copy()
            cs.thumbnail((640, 640))
            sx_, sy_ = crop.width / float(cs.width), crop.height / float(
                cs.height)
            sub = [dict(q, bbox=[int(q["bbox"][0] * sx_) + cx1,
                                 int(q["bbox"][1] * sy_) + cy1,
                                 int(q["bbox"][2] * sx_) + cx1,
                                 int(q["bbox"][3] * sy_) + cy1])
                   for q in ask(cs, SPLIT % (p["name"],
                                             p["description"][:200]))]
            ca = max(1, (cx2 - cx1) * (cy2 - cy1))
            # a "piece" as big as the whole item is the item again
            sub = [q for q in sub if q["name"].lower() != nm and
                   (q["bbox"][2] - q["bbox"][0]) *
                   (q["bbox"][3] - q["bbox"][1]) < 0.7 * ca]
            if len(sub) >= 2:
                out("LOG split %s into %d" % (p["name"], len(sub)))
                final += sub
                continue
        final.append(p)
    res = final
    items(res)
    # the same part listed twice (left and right boot): one part, one box
    merged = []
    for p in res:
        same = [q for q in merged if q["name"].lower() == p["name"].lower()]
        if same:
            q = same[0]
            b, c = q["bbox"], p["bbox"]
            q["bbox"] = [min(b[0], c[0]), min(b[1], c[1]),
                         max(b[2], c[2]), max(b[3], c[3])]
        else:
            merged.append(p)
    # one question for all the pieces: what each really looks like, and
    # whether it is a mirrored pair
    out("PROGRESS 0.55 Describing the pieces")
    names = [p["name"] for p in merged]
    q = SHAPES % ", ".join(names)
    try:
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[small],
                   return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = m.generate(**inp, max_new_tokens=90 * len(names) + 50,
                           do_sample=False)
        ans = proc.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        out("LOG shapes answer: %s" % " ".join(ans.split())[:300])
        got = json.loads(re.search(r"\[.*\]", ans, re.S).group(0))
        by = {str(g.get("name", "")).lower(): g for g in got
              if isinstance(g, dict)}
        for p in merged:
            g = by.get(p["name"].lower())
            if not g:
                continue
            d = " ".join(str(g.get("shape", "")).split())
            if len(d) > 8:
                p["description"] = d
            nm_ = p["name"].lower()
            p["mirror"] = str(g.get("pair", "")).lower().startswith(
                ("y", "t")) and not any(
                w in nm_ for w in ("hair", "head", "face", "body", "cape",
                                   "chest", "belt", "skirt", "cloak"))
    except Exception as exc:
        out("LOG shapes failed: %s" % str(exc)[:100])
    out("PROGRESS 0.90 Looking at the body")
    try:
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": BODY}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = proc(text=[text], images=[small],
                   return_tensors="pt").to("cuda")
        with torch.inference_mode():
            o = m.generate(**inp, max_new_tokens=80, do_sample=False)
        ans = proc.batch_decode(o[:, inp["input_ids"].shape[1]:],
                                skip_special_tokens=True)[0]
        ans = " ".join(ans.replace("\n", " ").split())[:300]
        if len(ans) > 8:
            out("BODYDESC " + ans)
    except Exception as exc:
        out("LOG body failed: %s" % str(exc)[:100])
    out("PROGRESS 0.95 Done")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=1)
    items(merged)
    out("DONE " + out_path)


def main():
    if sys.argv[1] == "--serve":
        # stay up: one {"image","out"} job per line, the model stays loaded
        models = sys.argv[2]
        load(models)
        out("READY")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "QUIT":
                break
            try:
                j = json.loads(line)
                run(j["image"], j["out"], models)
            except Exception as exc:
                traceback.print_exc()
                out("ERROR %s" % str(exc)[:200])
            out("JOBEND")
        return
    run(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % (str(exc).strip().splitlines() or [type(exc).__name__])[-1][:200])
