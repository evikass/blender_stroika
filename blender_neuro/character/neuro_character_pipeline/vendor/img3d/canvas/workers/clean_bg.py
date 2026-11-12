"""Put the subject of a picture on the canvas's studio background.

    clean_bg.py <in.png> <out.png> <plate.png> <models dir>
    clean_bg.py --plate <reference.png> <plate.png> <models dir>

The subject is cut out with BiRefNet and laid on the plate: the soft gray
studio backdrop of the reference picture, so every picture made with the
"3D Ready" preset has exactly that background.
stdout: DONE <png> / ERROR <msg>
"""
import os
import sys
import traceback

import numpy as np
from PIL import Image


def out(*a):
    print(*a, flush=True)


M = {}      # BiRefNet stays loaded in --serve mode


def alpha_of(rgb, models):
    import torch
    from transformers import AutoModelForImageSegmentation
    if "m" not in M:
        M["m"] = AutoModelForImageSegmentation.from_pretrained(
            os.path.join(models, "ZhengPeng7", "BiRefNet"),
            trust_remote_code=True,
            local_files_only=True).eval().half().cuda()
    m = M["m"]
    x = np.asarray(rgb.resize((1024, 1024), Image.Resampling.BILINEAR),
                   np.float32) / 255.0
    x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
        np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1)[None].half().cuda()
    with torch.no_grad():
        pred = m(t)[-1].sigmoid().float().cpu()[0, 0].numpy()
    a = Image.fromarray((pred * 255).clip(0, 255).astype(np.uint8))
    return a.resize(rgb.size, Image.Resampling.BILINEAR)


def fill_frame(rgb, a, margin=0.06):
    """Bring the subject up to fill the picture: a crop around it with a
    thin margin, at the picture's own shape, scaled back to full size."""
    m = np.asarray(a) > 128
    if not m.any():
        return rgb, a
    ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    W, H = rgb.size
    bw, bh = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    s = min(W / float(bw), H / float(bh))
    if s <= 1.05:
        return rgb, a
    cw, ch = W / s, H / s
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    box = (int(round(cx - cw / 2)), int(round(cy - ch / 2)),
           int(round(cx + cw / 2)), int(round(cy + ch / 2)))
    rgb = rgb.crop(box).resize((W, H), Image.Resampling.LANCZOS)
    a = a.crop(box).resize((W, H), Image.Resampling.LANCZOS)
    return rgb, a


def make_plate(ref, plate, models):
    """The reference's backdrop with its subject painted out: every row is
    filled across the subject from the backdrop on its left and right."""
    rgb = Image.open(ref).convert("RGB")
    a = np.asarray(alpha_of(rgb, models), np.float32) / 255.0
    img = np.asarray(rgb, np.float32)
    mask = a > 0.02
    # grow the mask a little so no fringe is left
    from PIL import ImageFilter
    mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8))
                      .filter(ImageFilter.MaxFilter(15))) > 0
    h, w = mask.shape
    res = img.copy()
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        xs = np.where(~row)[0]
        if len(xs) < 2:
            continue
        for c in range(3):
            res[y, :, c] = np.interp(np.arange(w), xs, img[y, xs, c])
    # smooth it: the backdrop is a soft gradient with no detail
    p = Image.fromarray(res.clip(0, 255).astype(np.uint8))
    p = p.filter(ImageFilter.GaussianBlur(25))
    p.save(plate)


def main():
    if sys.argv[1] == "--serve":
        # stay up: {"in","out","plate"} per line, BiRefNet stays loaded
        models = sys.argv[2]
        out("READY")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line == "QUIT":
                break
            try:
                import json
                j = json.loads(line)
                rgb = Image.open(j["in"]).convert("RGB")
                a = alpha_of(rgb, models)
                rgb, a = fill_frame(rgb, a)
                bg = Image.open(j["plate"]).convert("RGB").resize(
                    rgb.size, Image.Resampling.LANCZOS)
                bg.paste(rgb, (0, 0), a)
                bg.save(j["out"])
                out("DONE " + j["out"])
            except Exception as exc:
                traceback.print_exc()
                out("ERROR %s" % str(exc)[:200])
            out("JOBEND")
        return
    if sys.argv[1] == "--plate":
        make_plate(sys.argv[2], sys.argv[3], sys.argv[4])
        out("DONE " + sys.argv[3])
        return
    src, dst, plate, models = sys.argv[1:5]
    rgb = Image.open(src).convert("RGB")
    a = alpha_of(rgb, models)
    rgb, a = fill_frame(rgb, a)
    bg = Image.open(plate).convert("RGB").resize(rgb.size,
                                                 Image.Resampling.LANCZOS)
    bg.paste(rgb, (0, 0), a)
    bg.save(dst)
    out("DONE " + dst)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        out("ERROR %s" % str(exc)[:200])
