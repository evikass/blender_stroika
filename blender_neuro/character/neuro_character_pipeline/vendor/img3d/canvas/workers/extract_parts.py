"""Draw every chosen part of a picture on its own, with FLUX.2 [klein].

    extract_parts.py <jobs.json> <engine dir>

jobs.json: {"image": src, "parts": [{"name","description","bbox","kind",
"out"}, ...]}; kind is "part", "head_bald" or "body".
stdout: LOG / PART <index> <png> / FAIL <index> <msg> / ALLPARTS / ERROR <msg>
"""
import json
import os
import sys
import traceback


def out(*a):
    print(*a, flush=True)


def view(p):
    """Front, or a three-quarter view from a little above, which shows the
    depth, thickness and sides of the piece for the 3D model."""
    if p.get("iso"):
        return ("seen from a three-quarter angle slightly from above "
                "(an isometric-like 3/4 view, turned about 35 degrees), so "
                "its depth, thickness, sides and inner surfaces are clearly "
                "visible")
    return "front view"


def prompt(p):
    n = p["name"].lower()
    d = p.get("description", "").strip()
    if p["kind"] == "body":
        low = " " + d.lower().replace(",", " ").replace(".", " ") + " "
        fem = any(w in low for w in (" woman ", " girl ", " female ",
                                     " lady ", " women "))
        who = ("This character is: %s " % d) if d else ""
        if fem:
            sex = ("She stays a woman with her female body: the same "
                   "chest, waist, hips and female figure, do not make her "
                   "a man. ")
            wear = ("a simple plain gray sports bra and plain gray "
                    "underwear shorts")
        else:
            sex = ("Keep the same sex and the same body shape as the "
                   "character, a woman stays a woman and a man stays a "
                   "man. ")
            wear = "only simple plain gray underwear shorts"
        return ("Turn this character into its bare base body for 3D "
                "modeling: remove all hair, all clothing and all "
                "accessories and footwear. %s%sKeep exactly the same face, "
                "the same eyes with pupils and the same expression, the "
                "same skin color, the same body proportions and the same "
                "standing pose. The head is bald and smooth. The body "
                "wears %s. Bare feet. Full body, centered on a plain light "
                "gray background, same art style." % (who, sex, wear))
    if p["kind"].startswith("view_"):
        side = {"view_back": "from directly behind (the back view)",
                "view_left": "from its left side (a side profile view)",
                "view_right": "from its right side (a side profile view)"
                }[p["kind"]]
        return ("Show this exact same subject %s. Keep exactly the same "
                "design, proportions, colors, materials and every detail, "
                "the same art style, the whole subject in the picture, "
                "centered on the same plain light gray background, same "
                "soft studio light." % side)
    if p["kind"] == "head_bald":
        return ("Show only the head and face of this character with all "
                "the hair removed: a smooth bald head. Keep exactly the "
                "same face, the same eyes with pupils, eyebrows, ears and "
                "expression, the same skin color and art style. Head only, "
                "cut at the neck, centered on a plain light gray "
                "background, front view.")
    one = ""
    if p.get("mirror"):
        one = (" Show only ONE piece of the pair: the one on the character's "
               "right side (on the left of the picture), not both.")
    strict = ""
    if p.get("strict"):
        strict = (" IMPORTANT: the result must contain no person, no face, "
                  "no head, no hands and no body at all, only the %s." % n)
    if "hair" in n:
        return ("Show only the hair of this character, like a wig on an "
                "invisible head: the same hairstyle, length, color and "
                "strands, but no face, no eyes, no skin, no head and no "
                "body. %s Centered, %s, on a plain light gray "
                "background, same art style.%s" % (d, view(p), strict))
    return ("Isolate only the %s of this character: %s Keep exactly its "
            "real shape, outline, size, proportions, colors, material and "
            "trims as seen in the picture; only fill in small areas that are "
            "hidden behind other things. Do not turn it into a different "
            "kind of object. Remove everything that is not the %s (no body, "
            "no skin, no other clothing or items). Show it alone, whole, "
            "centered, %s, on a plain light gray background, in "
            "exactly the same art style.%s%s" % (n, d or ".", n, view(p),
                                                  one, strict))


def cut_prompt(p):
    n = p["name"].lower()
    d = p.get("description", "").strip()
    strict = (" The result must contain no person, no face, no head, no "
              "hands and no body." if p.get("strict") else "")
    if p.get("mirror"):
        one = (" This is ONE piece of a matching pair: show only this one "
               "piece, not two.")
        n = n[:-1] if n.endswith("s") and not n.endswith("ss") else n
    else:
        one = ""
    return ("This picture shows the %s of a character, cut out from "
            "everything else. Turn it into a clean, complete picture of "
            "only this %s: %s Keep exactly its shape, outline, size, "
            "colors, material and trims; only fill in the areas that were "
            "hidden or cut off by other things, and smooth the cut edges. "
            "Do not add a person, face, head, body or any other item.%s%s "
            "Centered, %s, on a plain light gray background, same "
            "art style." % (n, n, d or "", one, strict, view(p)))


def cut_out(src, p, path):
    """The part's own pixels on the studio gray, cropped around it."""
    import numpy as np
    from PIL import Image, ImageFilter
    m = Image.open(p["mask"]).convert("L").resize(src.size)
    a = np.asarray(m) > 127
    if p.get("mirror"):
        # one piece of the pair: the half on the left of the picture
        xs = np.where(a.any(0))[0]
        mid = (xs.min() + xs.max()) // 2
        a[:, mid:] = False
    m = Image.fromarray((a * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(1.5))
    bg = Image.new("RGB", src.size, (176, 176, 179))
    bg.paste(src, (0, 0), m)
    ys, xs = np.where(np.asarray(m) > 20)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    w, h = x2 - x1, y2 - y1
    side = max(w, h) * 1.35 + 32
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2),
           int(cy + side / 2))
    canvas = Image.new("RGB", (box[2] - box[0], box[3] - box[1]),
                       (176, 176, 179))
    canvas.paste(bg.crop((max(0, box[0]), max(0, box[1]),
                          min(src.width, box[2]), min(src.height, box[3]))),
                 (max(0, -box[0]), max(0, -box[1])))
    canvas.save(path)
    return path


def main():
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    eng = sys.argv[2]
    sys.path.insert(0, os.path.join(eng, "engine_hy"))
    import klein_worker_hy as kw
    from PIL import Image
    torch, pipe = kw.build()
    out("LOG Klein ready")
    src = Image.open(jobs["image"]).convert("RGB")
    tmp = os.path.join(os.path.dirname(sys.argv[1]), "_crops")
    os.makedirs(tmp, exist_ok=True)
    for i, p in enumerate(jobs["parts"]):
        try:
            src_p = p.get("src") or jobs["image"]
            text = prompt(p)
            if p.get("mask") and os.path.isfile(p["mask"]) and \
                    p["kind"] == "part":
                # SAM 3 found its exact pixels: Klein only completes them
                inp = cut_out(src if src_p == jobs["image"] else
                              Image.open(src_p).convert("RGB"), p,
                              os.path.join(tmp, "cut_%d.png" % i))
                text = cut_prompt(p)
            elif p["kind"] in ("body", "part", "head_bald") or \
                    p["kind"].startswith("view_"):
                inp = src_p
            else:
                x1, y1, x2, y2 = p.get("bbox") or [0, 0, src.width,
                                                   src.height]
                w, h, m = x2 - x1, y2 - y1, 0.18
                box = (int(max(0, x1 - w * m)), int(max(0, y1 - h * m)),
                       int(min(src.width, x2 + w * m)),
                       int(min(src.height, y2 + h * m)))
                if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                    box = (0, 0, src.width, src.height)
                inp = os.path.join(tmp, "crop_%d.png" % i)
                (src if src_p == jobs["image"] else
                 Image.open(src_p).convert("RGB")).crop(box).save(inp)
            # small pieces are drawn at full size, not at their size in
            # the picture
            im = Image.open(inp)
            if max(im.size) < 1024:
                k = 1024.0 / max(im.size)
                inp2 = os.path.join(tmp, "up_%d.png" % i)
                im.convert("RGB").resize((int(im.width * k),
                                          int(im.height * k)),
                                         Image.LANCZOS).save(inp2)
                inp = inp2
            kw.run(torch, pipe, {"image": inp, "out": p["out"],
                                 "text": text, "size": "HIGH",
                                 "seed": int(jobs.get("seed", 0))})
            out("PART %d %s" % (i, p["out"]))
        except Exception as exc:
            traceback.print_exc()
            out("FAIL %d %s" % (i, str(exc)[:160]))
    out("ALLPARTS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
