"""Cut every listed part out of the picture with SAM 3.

    segment_parts.py <parts.json> <image> <out dir> <models dir>

For each part name SAM 3 finds the exact pixels of that thing in the
picture (both pieces of a pair). Writes <out dir>/mask_<i>.png, and adds
"mask", "bbox" (from the mask), "score" and "found" to every part in
parts.json. A part SAM 3 cannot find at all is marked found=false.
stdout: LOG / SEG <i> <score> / DONE / ERROR <msg>
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
import os
import sys
import traceback


# things the vision model often misses: SAM 3 looks for each one itself,
# and keeps it when it is really there and not part of something listed
CHECK = (
    ("Undersuit", "bodysuit", .4), ("Undersuit", "undershirt", .4),
    ("Undersuit", "sleeves", .45), ("Gloves", "gloves", .25),
    ("Knee Guards", "knee pads", .4), ("Elbow Guards", "elbow pads", .4),
    ("Front Flap", "tabard", .4), ("Front Flap", "leather panel", .5),
    ("Hip Plates", "hip armor plates", .5), ("Socks", "socks", .4),
    ("Scarf", "scarf", .45), ("Necklace", "necklace", .45),
    ("Earrings", "earrings", .45), ("Hair Clip", "hair clip", .45),
    ("Pouch", "pouch", .45), ("Collar", "collar", .45),
    ("Pants", "pants", .45), ("Shirt", "shirt", .45),
    ("Jacket", "jacket", .45), ("Hood", "hood", .5),
    ("Helmet", "helmet", .55), ("Shield", "shield", .5),
    ("Sword", "sword", .5), ("Bracelets", "bracelets", .45),
    ("Strap", "strap", .55))


def out(*a):
    print(*a, flush=True)


S = {}        # the loaded model, and the last picture's features + checks


def load(models):
    if "model" in S:
        return
    import torch
    from transformers import Sam3Model, Sam3Processor
    path = os.path.join(models, "facebook", "sam3")
    S["proc"] = Sam3Processor.from_pretrained(path)
    S["model"] = Sam3Model.from_pretrained(
        path, dtype=_dt(torch)).to("cuda").eval()
    S["torch"] = torch
    out("LOG SAM 3 ready")


def prep(img_path, models):
    """Look at the picture once and run the checklist on it. Done while the
    vision model is still naming the parts, so it costs no waiting."""
    import numpy as np
    from PIL import Image
    load(models)
    torch, proc, model = S["torch"], S["proc"], S["model"]
    key = (img_path, os.path.getmtime(img_path))
    if S.get("key") == key:
        return
    img = Image.open(img_path).convert("RGB")
    iin = proc(images=img, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        vis = model.get_vision_features(
            pixel_values=iin["pixel_values"].to(_dt(torch)))
    S.update(key=key, img=img, vis=vis, checks=[])
    for c_, (name, text, thr) in enumerate(CHECK):
        out("PROGRESS %.2f Checking for %s" % (c_ / len(CHECK), text))
        r = seg(text, thr)
        if len(r["scores"]) == 0:
            continue
        sc = r["scores"].float().cpu().numpy()
        ms = r["masks"].cpu().numpy().astype(bool)
        m = ms[sc >= max(thr, sc.max() * 0.6)].any(0)
        if m.sum() >= 400:
            S["checks"].append((name, m, float(sc.max())))


def seg(text, thr):
    torch, proc, model = S["torch"], S["proc"], S["model"]
    img = S["img"]
    tin = proc(text=text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        res = model(vision_embeds=S["vis"], input_ids=tin["input_ids"],
                    attention_mask=tin["attention_mask"])
    return proc.post_process_instance_segmentation(
        res, threshold=thr, mask_threshold=0.5,
        target_sizes=[(img.height, img.width)])[0]


def run(pj, img_path, odir, models):
    import numpy as np
    from PIL import Image
    prep(img_path, models)
    img = S["img"]
    parts = json.load(open(pj, encoding="utf-8"))
    os.makedirs(odir, exist_ok=True)
    for i, p in enumerate(parts):
        out("PROGRESS %.2f Cutting out %s" % (i / max(1, len(parts)),
                                              p["name"]))
        best = None
        for text in _queries(p["name"]):
            r = seg(text, 0.3)
            if len(r["scores"]) == 0:
                continue
            sc = r["scores"].float().cpu().numpy()
            ms = r["masks"].cpu().numpy().astype(bool)
            # every instance close to the best one: both boots, both pads
            m = ms[sc >= max(0.3, sc.max() * 0.6)].any(0)
            s_ = float(sc.max())
            if best is None or s_ > best[0]:
                best = (s_, m)
        if best is None or best[1].sum() < 50:
            p["found"] = False
            p["score"] = 0.0
            out("SEG %d 0" % i)
            continue
        s_, m = best
        ys, xs = np.where(m)
        p["bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                     int(ys.max()) + 1]
        mp = os.path.join(odir, "mask_%02d.png" % i)
        Image.fromarray((m * 255).astype(np.uint8)).save(mp)
        p.update(mask=mp, score=round(s_, 3), found=True, area=int(m.sum()))
        out("SEG %d %.2f" % (i, s_))
    # what the vision model did not list, from the checklist already run
    seen = set(p["name"].lower() for p in parts)
    masks = {}
    for i, p in enumerate(parts):
        if p.get("mask"):
            masks[i] = np.asarray(Image.open(p["mask"])) > 127
    for name, m, score in S.get("checks", []):
        if name.lower() in seen:
            continue
        area = int(m.sum())
        # inside a listed part: a separate piece that part swallowed, when
        # it is clearly smaller; the same thing again when it is not
        carve, dup = [], False
        for j, mj in masks.items():
            ov = (m & mj).sum() / float(area)
            if ov > 0.5:
                if area < 0.6 * mj.sum():
                    carve.append(j)
                else:
                    dup = True
        if dup:
            continue
        i = len(parts)
        for j in carve:
            masks[j] = masks[j] & ~m
            Image.fromarray((masks[j] * 255).astype(np.uint8)).save(
                parts[j]["mask"])
            out("LOG %s taken out of %s" % (name, parts[j]["name"]))
        ys, xs = np.where(m)
        mp = os.path.join(odir, "mask_%02d.png" % i)
        Image.fromarray((m * 255).astype(np.uint8)).save(mp)
        masks[i] = m
        parts.append({"name": name, "description": "", "mask": mp,
                      "bbox": [int(xs.min()), int(ys.min()),
                               int(xs.max()) + 1, int(ys.max()) + 1],
                      "score": round(score, 3), "found": True,
                      "area": area, "added": True,
                      "mirror": name in ("Gloves", "Knee Guards",
                                         "Elbow Guards", "Socks",
                                         "Earrings", "Bracelets",
                                         "Hip Plates")})
        seen.add(name.lower())
        out("LOG added %s (%.2f)" % (name, score))
    out("ITEMS " + json.dumps([{"name": q["name"], "description":
                                q.get("description", ""),
                                "found": q.get("found", True),
                                "mirror": q.get("mirror", False)}
                               for q in parts]))
    with open(pj, "w", encoding="utf-8") as fh:
        json.dump(parts, fh, indent=1)


def main():
    if sys.argv[1] == "--serve":
        # stay up: {"prep": image} or {"parts", "image", "dir"} per line
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
                if "prep" in j:
                    prep(j["prep"], models)
                else:
                    run(j["parts"], j["image"], j["dir"], models)
                    out("DONE")
            except Exception as exc:
                traceback.print_exc()
                out("ERROR %s" % str(exc)[:200])
            out("JOBEND")
        return
    pj, img_path, odir, models = sys.argv[1:5]
    run(pj, img_path, odir, models)
    out("DONE")


def _queries(name):
    n = name.strip().lower()
    q = [n]
    if n.endswith("s") and not n.endswith("ss"):
        q.append(n[:-1])
    return q


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
