"""Turn the quick 3D renders of the back and sides into clean pictures in
the art style of the front picture, with FLUX.2 [klein].

    refine_views.py <jobs.json> <engine dir>

jobs.json: {"image": front picture, "parts": [{"kind": "view_back", "out":
render (overwritten)}, ...], "seed"}. The render keeps the angle, shape and
framing; the front picture gives the look.
stdout: LOG / PART <index> <png> / FAIL <index> <msg> / ALLPARTS / ERROR
"""
import json
import os
import sys
import time
import traceback

SIDE = {"view_back": "from directly behind (the back view)",
        "view_left": "from its left side (a side profile view)",
        "view_right": "from its right side (a side profile view)"}


def out(*a):
    print(*a, flush=True)


SHEET = {"view_back": "from directly behind, showing its back",
         "view_left": "in a side profile view from its left side, facing "
                      "toward the left edge of the picture",
         "view_right": "in a side profile view from its right side, facing "
                       "toward the right edge of the picture"}

FULL = {"view_back": "from directly behind (the back view)",
        "view_left": "from its left side (a side profile view, facing the "
                     "left edge of the picture)",
        "view_right": "from its right side (a side profile view, facing the "
                      "right edge of the picture)"}


def sheet_text(kind):
    """Two panels side by side: the model draws the new view next to the
    front one, so it keeps the same character, size and look."""
    return ("A character turnaround sheet: two panels side by side on a "
            "plain light gray studio background. Left panel: the character "
            "from the image, seen from the front. Right panel: the exact "
            "same character seen %s. Full body in both panels, from head to "
            "feet, same size and height, same art style, colors, materials "
            "and every detail of the outfit. Soft studio light."
            % SHEET[kind])


def full_text(kind):
    """The new view at full size, in the exact look of the front."""
    return ("The first image is a picture of a character seen %s. The "
            "second image is the same character seen from the front. Redraw "
            "the first image as a clean, sharp, high quality picture of this "
            "character %s, in exactly the art style, shading, colors and "
            "materials of the second image, with the same fine details. Keep "
            "exactly the view angle, pose, silhouette, proportions, size and "
            "position of the first image. Do not turn the character. Plain "
            "light gray studio background, soft studio light."
            % (FULL[kind], FULL[kind]))


def text(kind):
    return ("The first image is a rough 3D render of a character seen %s. "
            "The second image is the same character seen from the front. "
            "Redraw the first image as a clean, sharp, high quality "
            "picture of this character %s, in exactly the art style, "
            "shading, colors and materials of the second image, with the "
            "same fine details. Keep exactly the view angle, pose, "
            "silhouette, proportions, size and position of the first "
            "image. Do not turn the character. Plain light gray studio "
            "background, soft studio light." % (SIDE[kind], SIDE[kind]))


def shape(im):
    """The subject's outline, cropped and scaled to a fixed size."""
    import numpy as np
    from PIL import Image
    a = np.asarray(im.convert("RGB").resize((256, 384))).astype(float)
    bg = np.median(np.concatenate([a[:8].reshape(-1, 3),
                                   a[-8:].reshape(-1, 3)]), 0)
    m = np.abs(a - bg).sum(2) > 40
    ys, xs = np.where(m)
    if len(xs) < 50:
        return m.astype(float)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.asarray(Image.fromarray((m * 255).astype("uint8")).resize(
        (96, 144))) > 127


def facing_fixed(res, render):
    """The model sometimes turns a side view the other way round: when the
    mirrored picture matches the render's outline better, mirror it."""
    from PIL import ImageOps
    a, b = shape(res), shape(render)
    bf = b[:, ::-1]
    iou = (a & b).sum() / float((a | b).sum() + 1)
    iouf = (a & bf).sum() / float((a | bf).sum() + 1)
    if iouf > iou + 0.02:
        out("LOG mirrored back (%.2f vs %.2f)" % (iouf, iou))
        return ImageOps.mirror(res)
    return res


def main():
    jobs = json.load(open(sys.argv[1], encoding="utf-8"))
    sys.path.insert(0, os.path.join(sys.argv[2], "engine_hy"))
    import klein_worker_hy as kw
    from PIL import Image
    torch, pipe = kw.build()
    out("LOG Klein ready")
    front = Image.open(jobs["image"]).convert("RGB")
    for i, p in enumerate(jobs["parts"]):
        try:
            t0 = time.time()
            src = Image.open(p["out"]).convert("RGB")
            src.save(p["out"][:-4] + "_render.png")
            W, H = src.size
            W, H = W // 16 * 16, H // 16 * 16
            src = src.resize((W, H), Image.LANCZOS)
            ref = front.copy()
            ref.thumbnail((W, H), Image.LANCZOS)
            pe = kw.prompt_embeds(torch, pipe, text(p["kind"]))
            g = torch.Generator("cuda").manual_seed(
                int(jobs.get("seed", 0)))
            with torch.inference_mode():
                res = pipe(prompt_embeds=pe, image=[src, ref], height=H,
                           width=W, num_inference_steps=4,
                           generator=g).images[0]
            if p["kind"] != "view_back":
                res = facing_fixed(res, src)
            res.save(p["out"])
            out("LOG view %.1fs" % (time.time() - t0))
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
