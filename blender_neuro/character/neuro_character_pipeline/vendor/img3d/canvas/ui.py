"""The Canvas, drawn in the add-on's own card style.

Everything is painted with the GPU over an empty node editor: the nodes are
the add-on's dark rounded cards with the orange accent, pictures are shown as
pictures, and a modal operator handles clicks, dragging nodes, panning (middle
mouse or dragging the background) and zooming (wheel).
"""
import math
import os
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

# the add-on's palette (card.py)
CARD_TOP = (0.098, 0.111, 0.118, 0.97)
CARD_BOT = (0.072, 0.084, 0.092, 0.97)
PANEL_TOP = (0.125, 0.138, 0.145, 1.0)
PANEL_BOT = (0.090, 0.102, 0.112, 1.0)
PANEL_EDGE = (0.175, 0.190, 0.200, 1.0)
CTRL_TOP = (0.195, 0.210, 0.220, 1.0)
CTRL_BOT = (0.150, 0.163, 0.174, 1.0)
CTRL_TOP_H = (0.250, 0.265, 0.276, 1.0)
CTRL_BOT_H = (0.200, 0.214, 0.226, 1.0)
EDGE = (1.0, 1.0, 1.0, 0.09)
TEXT = (0.910, 0.910, 0.925, 1.0)
TEXT_DIM = (0.478, 0.478, 0.494, 1.0)
ACCENT = (1.0, 0.576, 0.078, 1.0)
ACCENT_TOP = (1.0, 0.655, 0.220, 1.0)
ACCENT_BOT = (0.995, 0.545, 0.082, 1.0)
ON_ACCENT = (0.110, 0.063, 0.008, 1.0)
PICKED_TOP = (0.290, 0.190, 0.070, 1.0)
PICKED_BOT = (0.200, 0.135, 0.055, 1.0)
TRACK = (0.215, 0.231, 0.243, 1.0)
GRID = (1.0, 1.0, 1.0, 0.035)
BG = (0.055, 0.060, 0.066, 1.0)

R_CARD, R_CTL = 14.0, 8.0
HITS = []            # (x, y, w, h, action) in region pixels, from last draw
HOVER = [None]
_TEX = {}


def _sh():
    return gpu.shader.from_builtin("UNIFORM_COLOR")


def _rr(x, y, w, h, r, steps=5):
    r = max(0.0, min(r, w * 0.5, h * 0.5))
    pts = []
    for cx, cy, a0 in ((x + w - r, y + h - r, 0.0), (x + r, y + h - r, math.pi * .5),
                       (x + r, y + r, math.pi), (x + w - r, y + r, math.pi * 1.5)):
        for i in range(steps + 1):
            a = a0 + (math.pi * 0.5) * (i / steps)
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    mid = (x + w * .5, y + h * .5)
    tris = []
    for i in range(len(pts)):
        tris.extend((mid, pts[i], pts[(i + 1) % len(pts)]))
    return tris


def fill(x, y, w, h, r, col):
    if w <= 0 or h <= 0:
        return
    sh = _sh()
    b = batch_for_shader(sh, "TRIS", {"pos": _rr(x, y, w, h, r)})
    # see-through colors need blending on, whatever drew before
    gpu.state.blend_set("ALPHA")
    sh.bind()
    sh.uniform_float("color", col)
    b.draw(sh)


def grad(x, y, w, h, r, cb, ct):
    if w <= 0 or h <= 0:
        return
    sh = gpu.shader.from_builtin("SMOOTH_COLOR")
    pts = _rr(x, y, w, h, r)
    cols = []
    for p in pts:
        t = max(0.0, min(1.0, (p[1] - y) / (h or 1.0)))
        cols.append(tuple(cb[i] + (ct[i] - cb[i]) * t for i in range(4)))
    b = batch_for_shader(sh, "TRIS", {"pos": pts, "color": cols})
    sh.bind()
    b.draw(sh)


def outline(x, y, w, h, r, col, width=1.0):
    fill(x - width, y - width, w + 2 * width, h + 2 * width, r + width, col)


def text(s, x, y, size, col, mid=None, right=None, bold=False, room=None):
    blf.size(0, size)
    if room is not None:
        while s and blf.dimensions(0, s)[0] > room:
            s = s[:-2] + "…" if len(s) > 2 else ""
    tw = blf.dimensions(0, s)[0]
    if mid is not None:
        x = mid - tw * 0.5
    elif right is not None:
        x = right - tw
    blf.color(0, *col)
    blf.position(0, x, y, 0)
    blf.draw(0, s)
    if bold:
        blf.position(0, x + 0.6, y, 0)
        blf.draw(0, s)
    return tw


def line(pts, col, width=2.0):
    sh = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    reg = bpy.context.region
    sh.bind()
    sh.uniform_float("viewportSize", (reg.width, reg.height))
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("color", col)
    b = batch_for_shader(sh, "LINE_STRIP", {"pos": pts})
    b.draw(sh)


def bezier(a, b, col, width):
    (x0, y0), (x1, y1) = a, b
    dx = max(40.0, abs(x1 - x0) * 0.5)
    pts = []
    for i in range(33):
        t = i / 32.0
        u = 1 - t
        px = u ** 3 * x0 + 3 * u * u * t * (x0 + dx) + 3 * u * t * t * (x1 - dx) + t ** 3 * x1
        py = u ** 3 * y0 + 3 * u * u * t * y0 + 3 * u * t * t * y1 + t ** 3 * y1
        pts.append((px, py))
    line(pts, col, width)


def texture(path):
    """GPU texture of a picture on disk, reloaded when the file changes."""
    if not path or not os.path.isfile(path):
        return None, (1, 1)
    mt = os.path.getmtime(path)
    hit = _TEX.get(path)
    if hit and hit[0] == mt:
        return hit[1], hit[2]
    try:
        key = "cvtex_" + os.path.basename(path)
        img = bpy.data.images.get(key)
        if img is None:
            img = bpy.data.images.load(path, check_existing=False)
            img.name = key
            try:
                img.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        else:
            img.filepath = path
            img.reload()
        tex = gpu.texture.from_image(img)
        size = tuple(img.size)
        _TEX[path] = (mt, tex, size)
        return tex, size
    except Exception:
        return None, (1, 1)


def picture(path, x, y, w, h, r=10.0):
    """Fit a picture into the box; a placeholder when there is none."""
    fill(x, y, w, h, r, (0.05, 0.055, 0.06, 1.0))
    tex, (iw, ih) = texture(path)
    if tex is None:
        return False
    s = min(w / max(iw, 1), h / max(ih, 1))
    dw, dh = iw * s, ih * s
    px, py = x + (w - dw) * .5, y + (h - dh) * .5
    sh = gpu.shader.from_builtin("IMAGE")
    b = batch_for_shader(sh, "TRI_FAN", {
        "pos": ((px, py), (px + dw, py), (px + dw, py + dh), (px, py + dh)),
        "texCoord": ((0, 0), (1, 0), (1, 1), (0, 1))})
    sh.bind()
    sh.uniform_sampler("image", tex)
    gpu.state.blend_set("ALPHA")
    b.draw(sh)
    return True


# -------------------------------------------------------------- controls

def hit(x, y, w, h, action):
    HITS.append((x, y, w, h, action))


def hovered(action):
    return HOVER[0] == action


def button(x, y, w, h, label, action, kind="ctl", enabled=True, z=1.0,
           size=12.0):
    if not enabled:
        grad(x, y, w, h, R_CTL * z, CTRL_BOT, CTRL_TOP)
        text(label, 0, y + h * .5 - size * z * .36, size * z, TEXT_DIM,
             mid=x + w * .5, room=w - 10)
        return
    hov = hovered(action)
    if kind == "accent":
        grad(x, y, w, h, R_CTL * z, ACCENT_BOT,
             ACCENT_TOP if not hov else (1.0, 0.72, 0.35, 1.0))
        text(label, 0, y + h * .5 - size * z * .36, size * z, ON_ACCENT,
             mid=x + w * .5, bold=True, room=w - 10)
    elif kind == "picked":
        outline(x, y, w, h, R_CTL * z, ACCENT, 1.0)
        grad(x, y, w, h, R_CTL * z, PICKED_BOT, PICKED_TOP)
        text(label, 0, y + h * .5 - size * z * .36, size * z, ACCENT,
             mid=x + w * .5, room=w - 10)
    else:
        grad(x, y, w, h, R_CTL * z, CTRL_BOT_H if hov else CTRL_BOT,
             CTRL_TOP_H if hov else CTRL_TOP)
        text(label, 0, y + h * .5 - size * z * .36, size * z, TEXT,
             mid=x + w * .5, room=w - 10)
    hit(x, y, w, h, action)


def chips(x, y, w, h, items, current, action, z):
    gap = 4 * z
    cw = (w - gap * (len(items) - 1)) / len(items)
    for i, (key, label) in enumerate(items):
        button(x + i * (cw + gap), y, cw, h, label, action + ":" + key,
               kind="picked" if key == current else "ctl", z=z, size=11.0)


def shape_chips(x, y, w, h, items, current, action, z):
    """Chips that show each picture shape as a small block of that shape
    above its numbers, so the choice is seen, not read."""
    gap = 4 * z
    cw = (w - gap * (len(items) - 1)) / len(items)
    for i, (key, label) in enumerate(items):
        bx = x + i * (cw + gap)
        picked = key == current
        button(bx, y, cw, h, "", action + ":" + key,
               kind="picked" if picked else "ctl", z=z, size=11.0)
        aw, ah = (float(v) for v in key.split("x"))
        box = 18 * z
        k = box / max(aw, ah)
        fw, fh = aw * k, ah * k
        col = ACCENT if picked else TEXT
        fill(bx + (cw - fw) * .5, y + h - 8 * z - box + (box - fh) * .5,
             fw, fh, 2 * z, col[:3] + (0.85,))
        text(label, 0, y + 6 * z, 10 * z, col if picked else TEXT_DIM,
             mid=bx + cw * .5)


def check(x, y, s, on, action, z):
    if on:
        grad(x, y, s, s, 4 * z, ACCENT_BOT, ACCENT_TOP)
        line([(x + s * .25, y + s * .52), (x + s * .43, y + s * .3),
              (x + s * .76, y + s * .72)], ON_ACCENT, 2.0 * z)
    else:
        grad(x, y, s, s, 4 * z, CTRL_BOT, CTRL_TOP)
    hit(x, y, s, s, action)


def bar(x, y, w, h, frac, z):
    fill(x, y, w, h, h * .5, TRACK)
    if frac > 0:
        grad(x, y, max(h, w * min(1.0, frac)), h, h * .5, ACCENT_BOT, ACCENT_TOP)


def card(x, y, w, h, title, z, icon_col=ACCENT, selected=False):
    """A node: the add-on's card with its title row."""
    if selected:
        outline(x, y, w, h, R_CARD * z, (1.0, 0.576, 0.078, 0.85), 1.5 * z)
    else:
        outline(x, y, w, h, R_CARD * z, (0.0, 0.0, 0.0, 0.35), 2.0 * z)
    # opaque first: the selection ring must not shine through the card
    fill(x, y, w, h, R_CARD * z, (0.06, 0.07, 0.077, 1.0))
    grad(x, y, w, h, R_CARD * z, CARD_BOT, CARD_TOP)
    fill(x + 14 * z, y + h - 26 * z, 8 * z, 8 * z, 4 * z, icon_col)
    text(title.upper(), x + 30 * z, y + h - 26 * z, 11.5 * z, TEXT,
         bold=True, room=w - 50 * z)
    hit(x, y + h - 38 * z, w, 38 * z, "drag")


def wrap(s, width, size):
    """Word-wrapped lines as (start index, text)."""
    blf.size(0, size)
    lines, start, i = [], 0, 0
    while start <= len(s):
        if blf.dimensions(0, s[start:])[0] <= width:
            lines.append((start, s[start:]))
            break
        end = start + 1
        last_space = -1
        while end <= len(s) and blf.dimensions(0, s[start:end])[0] <= width:
            if s[end - 1] == " ":
                last_space = end
            end += 1
        cut = last_space if last_space > start else max(start + 1, end - 1)
        lines.append((start, s[start:cut]))
        start = cut
    return lines or [(0, "")]


TB = {}       # each text box's layout from the last draw, for the mouse


def text_box(x, y, w, h, s, placeholder, editing, caret, z, action,
             anchor=None, scroll=None):
    """A multi-line text area: wraps the words, shows the caret and the
    selected text when typed in."""
    if editing:
        outline(x, y, w, h, 8 * z, ACCENT, 1.5)
    else:
        outline(x, y, w, h, 8 * z, PANEL_EDGE, 1.0)
    fill(x, y, w, h, 8 * z, (0.045, 0.05, 0.055, 1.0))
    size = 12.5 * z
    pad = 10 * z
    lh = size * 1.45
    lines = wrap(s, w - 2 * pad, size) if s else [(0, "")]
    top = y + h - pad - size
    if not s and not editing:
        text(placeholder, x + pad, top, size, TEXT_DIM, room=w - 2 * pad)
    maxl = max(1, int((h - 2 * pad) // lh))
    cl = 0
    for k, (st, t) in enumerate(lines):
        if st <= caret:
            cl = k
    first = 0
    if editing:
        first = scroll if scroll is not None else 0
        # keep the caret's line in view
        if cl < first:
            first = cl
        if cl >= first + maxl:
            first = cl - maxl + 1
        first = max(0, min(first, max(0, len(lines) - maxl)))
    blf.size(0, size)
    sel = None
    if editing and anchor is not None and anchor != caret:
        sel = (min(anchor, caret), max(anchor, caret))
    for k, (st, t) in enumerate(lines[first:first + maxl]):
        ly = top - k * lh
        if sel:
            a, b = max(sel[0], st), min(sel[1], st + len(t))
            if a < b or (sel[0] <= st and sel[1] > st + len(t)):
                x0 = x + pad + blf.dimensions(0, t[:a - st])[0]
                x1 = x + pad + blf.dimensions(0, t[:max(a, b) - st])[0]
                fill(x0, ly - 4 * z, max(x1 - x0, 3 * z), size + 6 * z, 0,
                     (1.0, 0.576, 0.078, 0.38))
        text(t, x + pad, ly, size, TEXT)
    if editing:
        st, t = lines[cl]
        blf.size(0, size)
        cx = x + pad + blf.dimensions(0, t[:caret - st])[0]
        cy = top - (cl - first) * lh
        if cl - first < maxl and (time.time() * 2) % 2 < 1.3:
            fill(cx, cy - 3 * z, 1.6 * z, size + 4 * z, 0, ACCENT)
    if len(lines) > first + maxl:
        text("\u2026", 0, y + 4 * z, size, TEXT_DIM, right=x + w - pad)
    TB[action] = {"x": x + pad, "top": top, "lh": lh, "size": size,
                  "lines": lines, "first": first, "maxl": maxl,
                  "box": (x, y, w, h)}
    hit(x, y, w, h, action)


def tb_index(action, mx, my):
    """The letter position under the mouse in a text box."""
    L = TB.get(action)
    if not L:
        return None
    k = L["first"] + int((L["top"] + L["size"] + L["lh"] * 0.25 - my) //
                         L["lh"])
    k = max(0, min(len(L["lines"]) - 1, k))
    st, t = L["lines"][k]
    blf.size(0, L["size"])
    best, bi = 1e9, 0
    for i in range(len(t) + 1):
        if i == len(t) and k < len(L["lines"]) - 1 and t.endswith(" "):
            break       # the end of a wrapped line is the next line's start
        d = abs(L["x"] + blf.dimensions(0, t[:i])[0] - mx)
        if d < best:
            best, bi = d, i
    return st + bi


def tb_line_move(action, caret, d):
    """The position one line up (d=-1) or down (d=1) from the caret."""
    L = TB.get(action)
    if not L:
        return caret
    lines = L["lines"]
    cl = 0
    for k, (st, t) in enumerate(lines):
        if st <= caret:
            cl = k
    nk = cl + d
    if nk < 0:
        return 0
    if nk >= len(lines):
        return lines[-1][0] + len(lines[-1][1])
    blf.size(0, L["size"])
    st, t = lines[cl]
    px = blf.dimensions(0, t[:caret - st])[0]
    st2, t2 = lines[nk]
    best, bi = 1e9, 0
    for i in range(len(t2) + 1):
        dd = abs(blf.dimensions(0, t2[:i])[0] - px)
        if dd < best:
            best, bi = dd, i
    return st2 + bi


def save_btn(x, y, z, action):
    """A small round button with a down arrow: saves that picture."""
    r = 11 * z
    hov = HOVER[0] == action
    fill(x - r, y - r, 2 * r, 2 * r, r,
         (0.95, 0.55, 0.1, 0.95) if hov else (0.06, 0.07, 0.08, 0.85))
    col = ON_ACCENT if hov else TEXT
    line([(x, y + 5 * z), (x, y - 3 * z)], col, 1.6 * z)
    line([(x - 4 * z, y + 1 * z), (x, y - 3.5 * z), (x + 4 * z, y + 1 * z)],
         col, 1.6 * z)
    line([(x - 5 * z, y - 6 * z), (x + 5 * z, y - 6 * z)], col, 1.6 * z)
    hit(x - r, y - r, 2 * r, 2 * r, action)


def clear_btn(x, y, z, action):
    """A small round x in a picture's corner: empties that picture."""
    r = 11 * z
    hov = HOVER[0] == action
    fill(x - r, y - r, 2 * r, 2 * r, r,
         (0.75, 0.2, 0.15, 0.95) if hov else (0.06, 0.07, 0.08, 0.85))
    text("\u00d7", 0, y - 5.5 * z, 15 * z, TEXT, mid=x)
    hit(x - r, y - r, 2 * r, 2 * r, action)
