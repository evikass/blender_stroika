"""A card of controls drawn into the viewport itself.

The sidebar is where every setting lives and it stays that way. This is the
same set, laid out the way Milad drew it: a titled card with the engine at the
top, the settings under headings that say what they are for, the folded extras
under those, and the run button with the tools pinned at the bottom behind a
rule, so the thing pressed most is always in the same place.

Nothing here is Blender's own UI. The card is drawn as flat rounded rectangles
through `gpu` with text through `blf`, and a modal operator decides for every
event whether the card wants it or Blender should still see it. That is the
same shape as the overlay UIs shipping in other add-ons, minus their renderer:
Skia would draw this more prettily and would also carry a quarter of a gigabyte
of wheels behind it, where `gpu` and `blf` are already in the process.

Three things here are easy to get wrong, all three learned the hard way:

* **Screen scale.** Everything is authored at 1x and multiplied by `ui_scale`
  at draw time. Reading a hit rect in unscaled pixels and drawing it in scaled
  ones puts every button a finger's width from where it looks.
* **Which corner a rectangle is given by.** Blender takes the bottom left while
  the card is laid out downwards from its top, so a row goes in at ``y - h``
  and the cursor drops by ``h + gap``. Placing it at ``y`` is what put the
  first draft's sliders through the switches above them.
* **Letting go of events.** The pointer being over the card is what makes it
  swallow a click; anywhere else the event passes straight through, so orbit,
  select and every shortcut still work with the card up.
"""

from __future__ import annotations

import math
import sys
import time

import blf
import bpy
import colorsys
import os
import gpu
from gpu_extras.batch import batch_for_shader


# -- the look ---------------------------------------------------------------

CARD = (0.106, 0.106, 0.114, 0.97)
# the card, the boxes and the buttons are all faintly lit from the top
# The design values, straight off the mock-up. What Blender finally paints
# comes out a little lighter than the number asked for, so the card carries a
# brightness of its own (Preferences) and every surface is derived from these.
CARD_TOP_D = (0.098, 0.111, 0.118)
CARD_BOT_D = (0.072, 0.084, 0.092)
PANEL_TOP_D = (0.125, 0.138, 0.145)
PANEL_BOT_D = (0.090, 0.102, 0.112)
PANEL_EDGE_D = (0.175, 0.190, 0.200)
CTRL_TOP_D = (0.195, 0.210, 0.220)
CTRL_BOT_D = (0.150, 0.163, 0.174)
TRACK_D = (0.175, 0.190, 0.200)
RULE_D = (0.155, 0.168, 0.178)
CARD_TOP = CARD_TOP_D + (0.97,)
CARD_BOT = CARD_BOT_D + (0.97,)
# The boxes and the buttons are opaque colours, not white at a low alpha:
# stacking translucent layers over the card came out far lighter on screen
# than the design, and every layer added to the error.
PANEL_TOP = PANEL_TOP_D + (1.0,)
PANEL_BOT = PANEL_BOT_D + (1.0,)
PANEL_EDGE = PANEL_EDGE_D + (1.0,)
CTRL_TOP = CTRL_TOP_D + (1.0,)
CTRL_BOT = CTRL_BOT_D + (1.0,)
CTRL_TOP_HOVER = (0.250, 0.265, 0.276, 1.0)
CTRL_BOT_HOVER = (0.200, 0.214, 0.226, 1.0)
EDGE = (1.0, 1.0, 1.0, 0.09)
CONTROL = (1.0, 1.0, 1.0, 0.055)
CONTROL_HOVER = (1.0, 1.0, 1.0, 0.10)
PICKED = (0.235, 0.157, 0.055, 1.0)
TRACK = (0.215, 0.231, 0.243, 1.0)
RULE = (0.196, 0.212, 0.224, 1.0)
TEXT = (0.910, 0.910, 0.925, 1.0)
TEXT_DIM = (0.478, 0.478, 0.494, 1.0)
ACCENT = (1.0, 0.576, 0.078, 1.0)
ACCENT_HOVER = (1.0, 0.655, 0.220, 1.0)
ON_ACCENT = (0.110, 0.063, 0.008, 1.0)
KNOB = (0.980, 0.980, 0.988, 1.0)
TEXT_CHIP = (0.720, 0.720, 0.735, 1.0)

# Orange is our colour, not necessarily the buyer's, and a colour you cannot
# change is a colour you have to put up with. Hue and saturation are kept in
# the add-on preferences so the choice outlives the file, and everything the
# card draws in the accent is derived from them rather than written down.
ACCENT_EDGE = (0.62, 0.357, 0.048, 1.0)
ACCENT_HUE = 0.0901          # the hue and saturation of #ff9314
ACCENT_SAT = 0.922


def _ac(alpha=1.0):
    """The accent colour at a given strength."""
    return (ACCENT[0], ACCENT[1], ACCENT[2], alpha)


def accent_rgb(hue, sat):
    return colorsys.hsv_to_rgb(hue % 1.0, max(0.0, min(1.0, sat)), 1.0)


def retint():
    """Rebuild every accent-derived colour from the saved hue and saturation.

    PICKED and ON_ACCENT are not free choices - one is the accent sunk to a
    background and the other is what stays readable on top of it - so they
    follow the accent rather than being picked alongside it.
    """
    global ACCENT, ACCENT_HOVER, ON_ACCENT, PICKED, ACCENT_EDGE
    hue, sat = ACCENT_HUE, ACCENT_SAT
    prefs = _prefs()
    if prefs is not None:
        hue = float(getattr(prefs, "accent_hue", hue))
        sat = float(getattr(prefs, "accent_sat", sat))
    r, g, b = accent_rgb(hue, sat)
    ACCENT = (r, g, b, 1.0)
    ACCENT_HOVER = accent_rgb(hue, sat * 0.78) + (1.0,)
    # the hairline around the two unwrap buttons: the accent, but sunk far
    # enough towards the card that it marks them out without shouting
    ACCENT_EDGE = (r * 0.62, g * 0.62, b * 0.62, 1.0)
    ON_ACCENT = (r * 0.11, g * 0.11, b * 0.11, 1.0)
    PICKED = (r * 0.235, g * 0.235, b * 0.235, 1.0)
    global ACCENT_TOP, ACCENT_BOT, PICKED_TOP, PICKED_BOT
    ACCENT_TOP = (r + (1.0 - r) * 0.10, g + (1.0 - g) * 0.22,
                  b + (1.0 - b) * 0.16, 1.0)
    ACCENT_BOT = (r * 0.995, g * 0.94, b * 0.72, 1.0)
    PICKED_TOP = (r * 0.29 + 0.02, g * 0.29 + 0.02, b * 0.30 + 0.04, 1.0)
    PICKED_BOT = (r * 0.20 + 0.015, g * 0.20 + 0.015, b * 0.22 + 0.035, 1.0)

    # How round the corners are is one number, not four. The card, the
    # buttons, the value boxes and the logo all keep the proportions they were
    # drawn with and follow it together, so square is square everywhere and
    # round is round everywhere.
    # every grey follows one brightness, so the card can be matched to a
    # screen without touching the code
    b = 1.0
    if prefs is not None:
        b = max(0.4, min(2.0, float(getattr(prefs, "ui_bright", 1.0))))
    global CARD_TOP, CARD_BOT, PANEL_TOP, PANEL_BOT, PANEL_EDGE
    global CTRL_TOP, CTRL_BOT, CTRL_TOP_HOVER, CTRL_BOT_HOVER, TRACK, RULE
    dim = lambda c, a=1.0: tuple(min(1.0, v * b) for v in c) + (a,)
    CARD_TOP, CARD_BOT = dim(CARD_TOP_D, 0.97), dim(CARD_BOT_D, 0.97)
    PANEL_TOP, PANEL_BOT = dim(PANEL_TOP_D), dim(PANEL_BOT_D)
    PANEL_EDGE = dim(PANEL_EDGE_D)
    CTRL_TOP, CTRL_BOT = dim(CTRL_TOP_D), dim(CTRL_BOT_D)
    CTRL_TOP_HOVER = dim(tuple(v * 1.28 for v in CTRL_TOP_D))
    CTRL_BOT_HOVER = dim(tuple(v * 1.33 for v in CTRL_BOT_D))
    TRACK, RULE = dim(TRACK_D), dim(RULE_D)

    global R_CARD, R_CTL, R_BOX, R_LOGO
    corner = CORNER
    if prefs is not None:
        corner = float(getattr(prefs, "corner", corner))
    R_CTL = corner
    R_CARD = corner * 1.75
    R_BOX = corner * 0.75
    R_LOGO = corner * 0.875


def _prefs():
    try:
        return bpy.context.preferences.addons[__package__].preferences
    except Exception:
        return None

ACCENT_TOP = (1.0, 0.655, 0.220, 1.0)
ACCENT_BOT = (0.995, 0.545, 0.082, 1.0)
PICKED_TOP = (0.290, 0.190, 0.070, 1.0)
PICKED_BOT = (0.200, 0.135, 0.055, 1.0)

R_CARD = 14.0
R_CTL = 8.0
R_BOX = 6.0
R_LOGO = 7.0
CORNER = 8.0                 # what the control radius is when nothing is set

# Wide enough to read, narrow enough to leave the viewport's own gizmo column
# clear. At 268 the card's left edge sat over the axis widget and the zoom and
# pan buttons under it; 234 cleared them and read as a splinter, so it sits a
# tenth wider than that.
CARD_W = 284.0
TAB_STRIP = 26.0          # the sidebar collapsed to its tabs alone
PAD = 15.0
GAP = 9.0
ROW = 34.0
TAB = "Image to 3D 2"

# Bake is hidden for now; it comes back in a later release
SHOW_BAKE = False

_state = {
    "group": "ORGANIC",
    "open": set(),
    "tip": None,          # (widget index, key, when the pointer settled)
    "ticking": False,
    "handle": None,
    "modal": False,
    "quit": False,
    "widgets": [],
    "hover": None,
    "drag": None,
    "pan": None,          # a middle button drag, for a pen with no wheel
    "note": None,         # a line the card says back, and when
    "painted": 0.0,       # last repaint we asked for
    "beat": 0.0,          # last event the modal saw
    "drawn": 0.0,         # last time the card was painted
    "asked": 0.0,         # when we last asked for a repaint, 0 if not waiting
    "cost": 0.02,         # what a repaint of this scene actually takes
    "dirty": False,       # a repaint we skipped and still owe
    "was_up": False,      # whether the card was showing last time we looked
    "typing": None,       # the number box being typed into, if any
    "scroll": 0.0,        # how far the contents are pushed up
    "over": 0.0,          # how much taller than the room it has, last draw
    "bar": None,          # a drag on the scroll bar itself
}


def register_draw():
    """Put the card's draw handler back after the classic panel had its turn."""
    if _state["handle"] is None:
        _drop_stale_card()
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_PIXEL")
        bpy.app.driver_namespace[_CARD_HANDLE] = _state["handle"]
    _state["quit"] = False


def _visible(context, area=None):
    """The card is up exactly when our own tab in the sidebar is.

    Milad's words: it should behave like the N panel itself. So it is not a
    button of its own - it follows the sidebar being open and the Smart Remesh
    tab being the one on top, and closing that tab closes the card with it.
    """
    prefs = _prefs()
    if prefs is not None and getattr(prefs, "classic_panel", False):
        return False        # the old side panel has the floor
    if area is None:
        area = getattr(context, "area", None)
    if area is None or area.type != "VIEW_3D":
        return False
    space = area.spaces.active
    if not getattr(space, "show_region_ui", False):
        return False
    for region in area.regions:
        if region.type == "UI":
            # A sidebar dragged shut is still "shown" and still remembers its
            # tab, so the flag and the category both say yes while nothing is
            # on screen. Its width is what actually tells: open it reads a few
            # hundred pixels, dragged shut it reads 26.
            if region.width < 60:
                return False
            return getattr(region, "active_panel_category", "") == TAB
    return False


# -- drawing ----------------------------------------------------------------

def _round_rect(x, y, w, h, r, steps=5):
    """Triangles for a rounded rectangle, as one fan around its middle."""
    r = max(0.0, min(r, w * 0.5, h * 0.5))
    pts = []
    corners = ((x + w - r, y + h - r, 0.0),
               (x + r, y + h - r, math.pi * 0.5),
               (x + r, y + r, math.pi),
               (x + w - r, y + r, math.pi * 1.5))
    for cx, cy, a0 in corners:
        for i in range(steps + 1):
            a = a0 + (math.pi * 0.5) * (i / steps)
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    mid = (x + w * 0.5, y + h * 0.5)
    tris = []
    for i in range(len(pts)):
        tris.extend((mid, pts[i], pts[(i + 1) % len(pts)]))
    return tris


# -- the band the contents are allowed to paint in --------------------------
#
# The card is taller than most viewports, so the rows below its foot have to
# be cut off. The GPU scissor was doing that, but its box is counted from the
# window rather than from the region, so on a screen where the 3D view does
# not start at the bottom of the window the cut landed in the wrong place and
# the last panel ran on over the timeline. The band is kept here instead and
# every fill, line and word is trimmed to it by hand - no GPU state involved,
# and it can be checked without Blender.
_CLIP = [None]          # (y0, y1) in real pixels, or None for no clipping


def _clip_y(y, h):
    """The part of a rect that may be painted, or None."""
    band = _CLIP[0]
    if band is None:
        return y, h
    y0, y1 = band
    top = min(y + h, y1)
    bot = max(y, y0)
    if top - bot <= 0.0:
        return None
    return bot, top - bot


def _inside(y, h=0.0):
    """Is this whole thing inside the band?"""
    band = _CLIP[0]
    return band is None or (y >= band[0] and y + h <= band[1])


def _fill(shader, x, y, w, h, r, color):
    if w <= 0.0 or h <= 0.0:
        return
    cut = _clip_y(y, h)
    if cut is None:
        return
    if cut != (y, h):
        # a row crossing the foot of the card: square off what is left, the
        # way a scissor would
        pts = _round_rect(x, y, w, h, r)
        y0, hh = cut
        pts = [(p[0], min(max(p[1], y0), y0 + hh)) for p in pts]
        batch = batch_for_shader(shader, "TRIS", {"pos": pts})
        shader.uniform_float("color", color)
        batch.draw(shader)
        return
    batch = batch_for_shader(shader, "TRIS",
                             {"pos": _round_rect(x, y, w, h, r)})
    shader.uniform_float("color", color)
    batch.draw(shader)


_SMOOTH = [None]


def _fill_grad(shader, x, y, w, h, r, c_bot, c_top):
    """A rounded rectangle lit from the top: c_bot at its foot, c_top at its
    head. The flat fill is the same shape, so the two can be mixed freely."""
    if _SMOOTH[0] is None:
        try:
            _SMOOTH[0] = gpu.shader.from_builtin("SMOOTH_COLOR")
        except Exception:
            _SMOOTH[0] = False
    sh = _SMOOTH[0]
    if not sh:
        _fill(shader, x, y, w, h, r, c_bot)
        return
    cut = _clip_y(y, h)
    if cut is None:
        return
    pts = _round_rect(x, y, w, h, r)
    if cut != (y, h):
        y0, ch = cut
        pts = [(p[0], min(max(p[1], y0), y0 + ch)) for p in pts]
    hh = h if h > 1e-6 else 1.0
    cols = []
    for p in pts:
        # the light still runs over the whole rect, so a row cut in half keeps
        # the shade it would have had
        t = max(0.0, min(1.0, (p[1] - y) / hh))
        cols.append(tuple(c_bot[i] + (c_top[i] - c_bot[i]) * t for i in range(4)))
    batch = batch_for_shader(sh, "TRIS", {"pos": pts, "color": cols})
    sh.bind()
    batch.draw(sh)
    shader.bind()


def _text(s, x, y, size, color, scale, right=None, mid=None, bold=False):
    # a word cannot be cut in half, so one that would hang out of the card is
    # left out until it is scrolled into view
    if not _inside(y - 1.0 * scale, size * scale * 1.35):
        return
    blf.size(0, size * scale)
    blf.color(0, *color)
    if right is not None or mid is not None:
        tw, _ = blf.dimensions(0, s)
        x = (right - tw) if right is not None else (mid - tw * 0.5)
    blf.position(0, x, y, 0)
    blf.draw(0, s)
    if bold:
        # blf has one weight, so a heavier line is the same glyphs a breath to
        # the right - enough to read as bold at this size, and cheaper than
        # shipping a second font to say one word louder.
        blf.position(0, x + max(0.6, 0.5 * scale), y, 0)
        blf.draw(0, s)


# -- what a control means, shown rather than written ------------------------
#
# A number on a slider says nothing about what it does to a mesh. Rest on one
# for a moment and a small card opens beside the panel with the idea drawn out
# and moving: the grid getting finer as the count climbs, the fold passing the
# angle and turning into an edge, the two panels coming apart to be remeshed
# on their own. The drawing is the explanation; the sentence under it is a
# reminder, not the lesson.

TIP_DELAY = 0.55
TIP_W = 244.0

TIPS = {
    "image": ("The picture",
              "One object, seen whole, on a plain or transparent background. "
              "The background is removed for you. A clear front three "
              "quarter view gives the best back side.", None),
    "folder": ("Install Folder",
               "Where the engine and every model file go. Pick any folder on "
               "any drive with room: each engine needs about 20 to 35 GB. "
               "Files already downloaded stay in the old folder.", None),
    "meters": ("Memory and processor",
               "VRAM is the memory on the graphics card, RAM the main memory, "
               "GPU how busy the graphics card is, CPU how busy the processor is. A bar turns red above 90%.",
               None),
    "report": ("Save Report",
               "A model will not generate? This saves a report of your PC "
               "and the last run on your Desktop. Send that file to Milad "
               "and he can fix it for your system.", None),
    "clearcache": ("Clear Cache",
                   "The engine stays loaded between models so the next one "
                   "starts faster. This closes it and frees the memory it "
                   "holds on the graphics card and in RAM.", None),
    "view1": ("Back view",
              "Optional. The same object seen from behind. The main picture "
              "is the Front.", None),
    "view2": ("Left view",
              "Optional. The same object seen from its own left side: stand "
              "at its left hand. In this picture it looks to the left.",
              None),
    "view3": ("Right view",
              "Optional. The same object seen from its own right side: stand "
              "at its right hand. In this picture it looks to the right.",
              None),
    "viewmap": ("Where each picture is taken",
                "Seen from above. The main picture is the Front. Walk around "
                "the object to the Back, Left and Right cameras: same object, "
                "same light, the whole object in the frame. Hunyuan only.",
                None),
    "generate": ("Generate 3D",
                 "Builds the shape and paints its texture from the picture, "
                 "on your own GPU, then adds the model at the 3D cursor.",
                 None),
    "cancel": ("Cancel", "Stops the engine. Nothing is added to the scene.",
               None),
    "install": ("Install Engine",
                "Sets up the engine on your computer, in the Install Folder "
                "shown on the Downloads tab. You only do this once.",
                None),
    "etrellis": ("TRELLIS",
                 "Microsoft's model. Fast, faithful to the picture on every "
                 "side, free for any use.", None),
    "epixal": ("Pixal3D",
               "TRELLIS.2 trained to keep the picture's exact view and "
               "proportions: it reads the camera from the picture first. "
               "Good when the result must line up with the photo.", None),
    "undotex": ("Undo Enhance Texture",
                "Puts back the texture the model had before the last "
                "Enhance Texture.", None),
    "undodet": ("Back to Previous Model",
                "Removes the detailed model and shows the one it was made "
                "from again, with its own texture.", None),
    "adetail": ("Enhance 3D Model",
                "On: right after the shape is made, Hi3D rebuilds it with "
                "sharp edges, rivets and fine detail, then the texture is "
                "painted on the detailed model. Adds about 5 minutes.", None),
    "aenhance": ("Enhance Texture",
                 "On: the texture is enhanced as soon as the model is in the "
                 "scene (same as pressing Enhance Texture). Adds about 1 to "
                 "3 minutes.", None),
    "detail": ("Detail Enhancer",
               "Rebuilds the selected model with Hi3D: sharp edges, rivets, "
               "scales and fine detail, same shape. The texture is painted "
               "again (the old one no longer fits the new surface). The old "
               "model stays in the file, hidden. About 5 minutes, or 10 with "
               "the texture.", None),
    "ehi3d": ("Hi3D",
              "Hunyuan makes the shape, then Hi3D (Hitem3D v1) rebuilds it "
              "with sharp edges and fine detail, then Hunyuan paints it. "
              "About 5 minutes more than Hunyuan alone.", None),
    "ehunyuan": ("Hunyuan 3D",
                 "Tencent's model. Richer, more realistic texture; slower "
                 "and needs more GPU memory. Check its licence for your "
                 "region before commercial use.", None),
    "qfast": ("Fast",
              "A quick look at the shape at 512 voxels. Good for trying "
              "pictures and seeds before a final run.", None),
    "qhigh": ("High",
              "1024 voxels. Sharp detail and a clean texture. The setting "
              "to use most of the time.", None),
    "qultra": ("Ultra",
               "1536 voxels: the finest detail. Slowest, and needs the most "
               "GPU memory.", None),
    "faces": ("Faces",
              "How many faces the finished model keeps. Fewer is lighter to "
              "work with; the texture keeps the small detail either way.",
              None),
    "texture": ("Texture size",
                "Resolution of the colour, roughness and metal maps baked "
                "onto the model.", None),
    "usetex": ("Texture",
               "On paints a texture onto the model. Off builds the shape "
               "only, with no texture: much faster.", None),
    "seed": ("Seed",
             "Each number is a different take on the same picture, mostly "
             "on the sides the picture does not show.", None),
    "accent": ("Accent colour", "The card's colour.", None),
    "corner": ("Corner rounding", "How round the card's corners are.", None),
}


def _text_w(text, size, scale):
    """How wide a line of text is, in unscaled pixels."""
    try:
        blf.size(0, size * scale)
        return blf.dimensions(0, text)[0] / scale
    except Exception:
        return len(text) * size * 0.55


_POLY = [None]


def _aaline(shader, pts, width, color, scale, closed=False):
    """A smooth line through the points. Blender's polyline shader draws with
    anti-aliasing, so small icons stop looking like staircases."""
    if len(pts) < 2:
        return
    if _POLY[0] is None:
        try:
            _POLY[0] = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
        except Exception:
            _POLY[0] = False
    sh = _POLY[0]
    pl = [(p[0] * scale, p[1] * scale) for p in pts]
    # an icon cut in half reads as a broken icon, so one that would hang out
    # of the card waits until it is scrolled in
    band = _CLIP[0]
    if band is not None:
        ys = [p[1] for p in pl]
        if min(ys) < band[0] - 1.0 or max(ys) > band[1] + 1.0:
            return
    if closed:
        pl = pl + [pl[0]]
    if not sh:
        for a, b in zip(pl[:-1], pl[1:]):
            _seg(shader, (a[0] / scale, a[1] / scale), (b[0] / scale, b[1] / scale),
                 width, color, scale)
        return
    try:
        vp = gpu.state.viewport_get()
        sh.bind()
        sh.uniform_float("viewportSize", (float(vp[2]), float(vp[3])))
        sh.uniform_float("lineWidth", float(width * scale))
        sh.uniform_float("color", color)
        batch = batch_for_shader(sh, "LINE_STRIP", {"pos": pl})
        batch.draw(sh)
    except Exception:
        for a, b in zip(pl[:-1], pl[1:]):
            _seg(shader, (a[0] / scale, a[1] / scale), (b[0] / scale, b[1] / scale),
                 width, color, scale)
    finally:
        shader.bind()


def _fit(text, room, size, scale, bold=False, floor=9):
    """The biggest size at or under `size` that still fits `room` pixels."""
    while size > floor and _text_w(text, size, scale) > room:
        size -= 1
    return size


def _quad(shader, a, b, c, d, color, scale):
    """Any four-cornered patch, as two triangles."""
    pts = [(p[0] * scale, p[1] * scale) for p in (a, b, c, d)]
    band = _CLIP[0]
    if band is not None:
        ys = [p[1] for p in pts]
        if min(ys) < band[0] - 1.0 or max(ys) > band[1] + 1.0:
            return
    batch = batch_for_shader(shader, "TRIS", {"pos": (
        pts[0], pts[1], pts[2], pts[0], pts[2], pts[3])})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _seg(shader, a, b, width, color, scale):
    """A straight line of a given width, at any angle."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    nx = -dy / length * width * 0.5
    ny = dx / length * width * 0.5
    _quad(shader, (a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny),
          (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny), color, scale)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _tip_grid(shader, x, y, w, h, phase, scale):
    """A square that gets finer and coarser again."""
    n = 2 + int(round(abs(math.sin(phase * math.pi)) * 5))
    side = min(w, h) - 8.0
    ox = x + (w - side) * 0.5
    oy = y + (h - side) * 0.5
    t = 1.2
    for i in range(n + 1):
        at = i * (side - t) / n
        _fill(shader, ox * scale, (oy + at) * scale, side * scale, t * scale,
              0.0, ACCENT if i in (0, n) else TRACK)
        _fill(shader, (ox + at) * scale, oy * scale, t * scale, side * scale,
              0.0, ACCENT if i in (0, n) else TRACK)


def _tip_fold(shader, x, y, w, h, phase, scale):
    """Two panes hinged at the middle; the crease lights when it is steep."""
    ang = 20.0 + abs(math.sin(phase * math.pi)) * 70.0
    side = min(w * 0.42, h - 16.0)
    cx = x + w * 0.5
    cy = y + h * 0.5
    lean = math.cos(math.radians(ang)) * side
    drop = math.sin(math.radians(ang)) * side * 0.5
    _fill(shader, (cx - lean) * scale, (cy - drop) * scale, lean * scale,
          2.0 * scale, 0.0, TRACK)
    _fill(shader, cx * scale, (cy - drop) * scale, lean * scale, 2.0 * scale,
          0.0, TRACK)
    _fill(shader, (cx - lean) * scale, (cy - drop - side * 0.45) * scale,
          2.0 * scale, side * 0.45 * scale, 0.0, TRACK)
    _fill(shader, (cx + lean - 2.0) * scale,
          (cy - drop - side * 0.45) * scale, 2.0 * scale,
          side * 0.45 * scale, 0.0, TRACK)
    hot = ACCENT if ang > 45.0 else TRACK
    _fill(shader, (cx - 2.0) * scale, (cy - drop - 3.0) * scale, 4.0 * scale,
          (side * 0.45 + 8.0) * scale, 2.0 * scale, hot)


def _cube_and_net(x, y, w, h):
    """The six faces of a cube, as (standing, opened out) corner pairs.

    Shared by the two drawings that need it: one folds a cube open to show
    where the UV islands come from, the other folds the flat layout back up to
    show where the seams are.
    """
    cell = min((w - 10.0) / 4.2, (h - 8.0) / 3.2)
    nx = x + (w - cell * 4.0) * 0.5
    ny = y + (h - cell * 3.0) * 0.5

    def net(cx, cy):
        px = nx + cx * cell
        py = ny + cy * cell
        pad = cell * 0.06
        return ((px + pad, py + pad), (px + cell - pad, py + pad),
                (px + cell - pad, py + cell - pad), (px + pad, py + cell - pad))

    side = cell * 0.92
    ux, uy = 0.866 * side, -0.5 * side
    vx, vy = -0.866 * side, -0.5 * side
    ax = x + w * 0.5
    ay = y + h * 0.5 + side * 0.95

    def at(du, dv, dd):
        return (ax + ux * du + vx * dv, ay + uy * du + vy * dv - side * dd)

    top = (at(0, 0, 0), at(1, 0, 0), at(1, 1, 0), at(0, 1, 0))
    right = (at(1, 0, 0), at(1, 0, 1), at(1, 1, 1), at(1, 1, 0))
    leftf = (at(0, 1, 0), at(1, 1, 0), at(1, 1, 1), at(0, 1, 1))
    return ((top, net(1, 2)), (leftf, net(0, 1)), (right, net(1, 1)),
            (top, net(1, 0)), (right, net(2, 1)), (leftf, net(3, 1)))


def _tip_split(shader, x, y, w, h, phase, scale):
    """A cube, opened out into its UV net, and gridded evenly all over.

    Milad's note on the first pass was that the cube was not a cube - it was
    three squares stacked, which reads as paper, not as a solid. So the faces
    are real four-cornered patches now and the cube is drawn in the ordinary
    isometric way, three faces meeting at a corner. Unfolding is then just the
    same four corners walking to where they sit on the flat net, one lerp per
    corner, and the grid rides along with them.

    Every face gets the same number of cells, because that is the truth of it:
    the pieces are remeshed apart from one another, not at different densities.
    """
    t = phase % 1.0
    open_ = 0.0 if t < 0.22 else min(1.0, (t - 0.22) / 0.34)
    open_ = open_ * open_ * (3.0 - 2.0 * open_)
    lay = 0.0 if t < 0.60 else min(1.0, (t - 0.60) / 0.28)

    faces = _cube_and_net(x, y, w, h)

    cells = 3
    line = (1.0, 1.0, 1.0, 0.20 * lay)
    for start, end in faces:
        corner = [_lerp(start[i], end[i], open_) for i in range(4)]
        _quad(shader, corner[0], corner[1], corner[2], corner[3], CONTROL,
              scale)
        edge = _ac(0.35 + 0.65 * open_)
        for i in range(4):
            _seg(shader, corner[i], corner[(i + 1) % 4], 1.4, edge, scale)
        if lay <= 0.0:
            continue
        for i in range(1, cells):
            f = i / cells
            _seg(shader, _lerp(corner[0], corner[1], f),
                 _lerp(corner[3], corner[2], f), 1.1, line, scale)
            _seg(shader, _lerp(corner[0], corner[3], f),
                 _lerp(corner[1], corner[2], f), 1.1, line, scale)


def _tip_shells(shader, x, y, w, h, phase, scale):
    """Two sheets lying on each other, welded into one."""
    close = abs(math.sin(phase * math.pi))
    side = min(w * 0.5, h - 20.0)
    off = (1.0 - close) * 18.0
    cx = x + w * 0.5
    cy = y + (h - side) * 0.5
    for sign, tone in ((-1.0, TRACK), (1.0, TRACK)):
        ox = cx - side * 0.5 + sign * off * 0.5
        oy = cy + sign * off * 0.35
        _fill(shader, ox * scale, oy * scale, side * scale, side * scale,
              3.0 * scale, ACCENT if off < 3.0 else tone)
        _fill(shader, (ox + 2.0) * scale, (oy + 2.0) * scale,
              (side - 4.0) * scale, (side - 4.0) * scale, 2.0 * scale, CARD)


def _tip_weld(shader, x, y, w, h, phase, scale):
    """Two points brought together and made into one."""
    close = abs(math.sin(phase * math.pi))
    cx = x + w * 0.5
    cy = y + h * 0.5
    gapx = (1.0 - close) * 22.0
    if gapx < 2.5:
        _fill(shader, (cx - 6.0) * scale, (cy - 6.0) * scale, 12.0 * scale,
              12.0 * scale, 6.0 * scale, ACCENT)
    else:
        for sign in (-1.0, 1.0):
            _fill(shader, (cx + sign * gapx - 4.0) * scale, (cy - 4.0) * scale,
                  8.0 * scale, 8.0 * scale, 4.0 * scale, TRACK)
    _fill(shader, (cx - 30.0) * scale, (cy - 1.0) * scale, 60.0 * scale,
          2.0 * scale, 0.0, RULE)


def _tip_meet(shader, x, y, w, h, phase, scale):
    """Two rows of points closing on each other until they touch."""
    close = abs(math.sin(phase * math.pi))
    gapx = (1.0 - close) * 26.0 + 3.0
    cx = x + w * 0.5
    for i in range(5):
        py = y + 14.0 + i * (h - 28.0) / 4.0
        for sign in (-1.0, 1.0):
            hot = ACCENT if gapx < 6.0 else TRACK
            _fill(shader, (cx + sign * gapx - 3.0) * scale, (py - 3.0) * scale,
                  6.0 * scale, 6.0 * scale, 3.0 * scale, hot)
        _fill(shader, (cx - gapx) * scale, (py - 1.0) * scale,
              gapx * 2.0 * scale, 2.0 * scale, 0.0,
              ACCENT if gapx < 6.0 else RULE)


def _tip_tear(shader, x, y, w, h, phase, scale):
    """A hole in a sheet closing up."""
    open_ = 1.0 - abs(math.sin(phase * math.pi))
    side = min(w - 40.0, h - 16.0)
    ox = x + (w - side) * 0.5
    oy = y + (h - side) * 0.5
    _fill(shader, ox * scale, oy * scale, side * scale, side * scale,
          4.0 * scale, CONTROL)
    r = 4.0 + open_ * 16.0
    _fill(shader, (ox + side * 0.5 - r) * scale, (oy + side * 0.5 - r) * scale,
          r * 2.0 * scale, r * 2.0 * scale, r * scale,
          CARD if open_ > 0.15 else ACCENT)


def _tip_swap(shader, x, y, w, h, phase, scale):
    """Two objects, one of them going, the other taking its place.

    Two drawings before this one missed it. Sliding blocks said "now there are
    two", and fading one square into another said "it changed in place" - and
    the switch is neither. There are two objects on the desk, the one carrying
    the old mesh is deleted, and the one carrying the new mesh moves into the
    place it left.
    """
    t = phase % 1.0
    fade = 0.0 if t < 0.28 else min(1.0, (t - 0.28) / 0.24)
    move = 0.0 if t < 0.58 else min(1.0, (t - 0.58) / 0.30)
    move = move * move * (3.0 - 2.0 * move)

    side = min((w - 34.0) * 0.5, h - 22.0)
    gapx = 18.0
    oy = y + (h - side) * 0.5
    lx = x + (w - side * 2.0 - gapx) * 0.5
    rx = lx + side + gapx

    def square(px, cells, alpha, hot):
        p0 = (px, oy)
        p1 = (px + side, oy)
        p2 = (px + side, oy + side)
        p3 = (px, oy + side)
        _quad(shader, p0, p1, p2, p3, (1.0, 1.0, 1.0, 0.055 * alpha), scale)
        edge = (_ac(alpha) if hot
                else (1.0, 1.0, 1.0, 0.20 * alpha))
        for i in range(4):
            _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.4, edge,
                 scale)
        line = (1.0, 1.0, 1.0, (0.22 if hot else 0.16) * alpha)
        for i in range(1, cells):
            f = i / cells
            _seg(shader, _lerp(p0, p1, f), _lerp(p3, p2, f), 1.0, line, scale)
            _seg(shader, _lerp(p0, p3, f), _lerp(p1, p2, f), 1.0, line, scale)
        if not hot:
            for i in range(cells):
                _seg(shader, _lerp(p0, p1, i / cells),
                     _lerp(p3, p2, (i + 1) / cells), 1.0, line, scale)

    # the old one, dense and triangulated, deleted as the switch goes on
    if fade < 1.0:
        square(lx, 6, 1.0 - fade, False)
    # the new one, clean quads, moving into the place the old one left
    square(rx + (lx - rx) * move, 3, 1.0, True)


def _cursor(shader, x, y, size, scale):
    """A pointer, so a drawing can say "the mouse is here"."""
    a = (x, y)
    b = (x, y - size)
    c = (x + size * 0.72, y - size * 0.72)
    _quad(shader, a, b, c, c, (1.0, 1.0, 1.0, 0.92), scale)
    _quad(shader, (a[0] + 1.0, a[1] - 1.0), (b[0] + 1.0, b[1] + 1.5),
          (c[0] - 1.0, c[1] + 0.5), (c[0] - 1.0, c[1] + 0.5),
          (0.1, 0.1, 0.12, 1.0), scale)


def _ring(shader, cx, cy, r, width, color, scale, steps=18):
    """A circle outline, for a brush."""
    import math as _m
    prev = None
    for i in range(steps + 1):
        a = 2.0 * _m.pi * i / steps
        pt = (cx + _m.cos(a) * r, cy + _m.sin(a) * r)
        if prev is not None:
            _seg(shader, prev, pt, width, color, scale)
        prev = pt


def _tip_pick(shader, x, y, w, h, phase, scale):
    """Hovering lights a piece; clicking takes that piece out on its own."""
    t = phase % 1.0
    parts = 3
    gapx = 8.0
    pw = (w - 16.0 - gapx * (parts - 1)) / parts
    ph = min(h - 26.0, pw * 1.15)
    oy = y + (h - ph) * 0.5 - 4.0
    ox = x + 8.0

    sweep = min(1.0, t / 0.62)
    under = min(parts - 1, int(sweep * parts))
    taken = t > 0.70

    for k in range(parts):
        px = ox + k * (pw + gapx)
        p0 = (px, oy)
        p1 = (px + pw, oy)
        p2 = (px + pw, oy + ph)
        p3 = (px, oy + ph)
        lit = (k == under)
        if taken and not lit:
            continue                     # the rest are gone once one is taken
        fill = _ac(0.16) if lit else (1.0, 1.0, 1.0, 0.055)
        _quad(shader, p0, p1, p2, p3, fill, scale)
        edge = (_ac(1.0) if lit
                else (1.0, 1.0, 1.0, 0.16))
        for i in range(4):
            _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.4, edge,
                 scale)
        line = (1.0, 1.0, 1.0, 0.14)
        for i in (1, 2):
            f = i / 3.0
            _seg(shader, _lerp(p0, p1, f), _lerp(p3, p2, f), 1.0, line, scale)
            _seg(shader, _lerp(p0, p3, f), _lerp(p1, p2, f), 1.0, line, scale)

    if not taken:
        cx = ox + under * (pw + gapx) + pw * 0.55
        _cursor(shader, cx, oy + ph * 0.62, 11.0, scale)


def _tip_isolate(shader, x, y, w, h, phase, scale):
    """One piece kept, the rest put away - and brought back again."""
    alone = abs(math.sin(phase * math.pi))
    parts = 3
    gapx = 8.0
    pw = (w - 16.0 - gapx * (parts - 1)) / parts
    ph = min(h - 26.0, pw * 1.15)
    oy = y + (h - ph) * 0.5
    ox = x + 8.0
    keep = 1

    for k in range(parts):
        fade = 1.0 if k == keep else 1.0 - alone
        if fade < 0.02:
            continue
        px = ox + k * (pw + gapx)
        p0 = (px, oy)
        p1 = (px + pw, oy)
        p2 = (px + pw, oy + ph)
        p3 = (px, oy + ph)
        held = (k == keep)
        _quad(shader, p0, p1, p2, p3,
              _ac(0.16) if held else (1.0, 1.0, 1.0,
                                                      0.055 * fade), scale)
        edge = (_ac(1.0) if held
                else (1.0, 1.0, 1.0, 0.16 * fade))
        for i in range(4):
            _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.4, edge,
                 scale)


def _tip_relax(shader, x, y, w, h, phase, scale):
    """A brush dragged over crumpled mesh, leaving it even behind it."""
    t = phase % 1.0
    front = x + 10.0 + (w - 20.0) * min(1.0, t / 0.85)
    cols, rows = 7, 4
    ox = x + 10.0
    oy = y + (h - 12.0 - (h - 34.0)) * 0.5 + 6.0
    gw = w - 20.0
    gh = h - 34.0

    def point(i, j):
        px = ox + gw * i / cols
        py = oy + gh * j / rows
        if px > front:
            # not reached yet: still crumpled
            px += math.sin((i * 1.7 + j * 2.3) * 1.9) * 3.4
            py += math.cos((i * 2.1 + j * 1.3) * 1.7) * 3.0
        return (px, py)

    line = (1.0, 1.0, 1.0, 0.20)
    for j in range(rows + 1):
        for i in range(cols):
            _seg(shader, point(i, j), point(i + 1, j), 1.1, line, scale)
    for i in range(cols + 1):
        for j in range(rows):
            _seg(shader, point(i, j), point(i, j + 1), 1.1, line, scale)

    _ring(shader, front, oy + gh * 0.5, 13.0, 1.6, _ac(1.0),
          scale)
    _ring(shader, front, oy + gh * 0.5, 13.0, 26.0,
          _ac(0.10), scale, steps=10)


def _checker(shader, p0, p1, p2, p3, n, alpha, scale):
    """A checkerboard laid inside any four-cornered patch."""
    for i in range(n):
        for j in range(n):
            if (i + j) % 2:
                continue
            a = _lerp(_lerp(p0, p1, i / n), _lerp(p3, p2, i / n), j / n)
            b = _lerp(_lerp(p0, p1, (i + 1) / n), _lerp(p3, p2, (i + 1) / n),
                      j / n)
            c = _lerp(_lerp(p0, p1, (i + 1) / n), _lerp(p3, p2, (i + 1) / n),
                      (j + 1) / n)
            d = _lerp(_lerp(p0, p1, i / n), _lerp(p3, p2, i / n), (j + 1) / n)
            _quad(shader, a, b, c, d, _ac(alpha), scale)


def _tip_uvkeep(shader, x, y, w, h, phase, scale):
    """The new mesh's islands landing where the old ones already were."""
    snap = abs(math.sin(phase * math.pi))
    box = min(w - 16.0, h - 12.0)
    ox = x + (w - box) * 0.5
    oy = y + (h - box) * 0.5
    spots = ((0.04, 0.52, 0.42, 0.42), (0.52, 0.52, 0.42, 0.42),
             (0.28, 0.06, 0.42, 0.40))
    stray = ((0.30, 0.30), (-0.22, 0.24), (0.16, -0.26))

    for (rx, ry, rw, rh), (dx, dy) in zip(spots, stray):
        # where the original UV already sits
        p0 = (ox + rx * box, oy + ry * box)
        p1 = (p0[0] + rw * box, p0[1])
        p2 = (p1[0], p0[1] + rh * box)
        p3 = (p0[0], p2[1])
        for i in range(4):
            _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.2,
                 (1.0, 1.0, 1.0, 0.18), scale)
        # and the new one, walking onto it
        ox2 = dx * box * (1.0 - snap)
        oy2 = dy * box * (1.0 - snap)
        q = [(pt[0] + ox2, pt[1] + oy2) for pt in (p0, p1, p2, p3)]
        _quad(shader, q[0], q[1], q[2], q[3],
              _ac(0.10 + 0.12 * snap), scale)
        for i in range(4):
            _seg(shader, q[i], q[(i + 1) % 4], 1.4,
                 _ac(0.45 + 0.55 * snap), scale)


def _tip_seams(shader, x, y, w, h, phase, scale):
    """Where the UV was cut, said the way a modeller would show it.

    A square with two lines on it said nothing about where those lines came
    from. So this runs the unfolding backwards: the flat layout first, then it
    folds up into the model, and then the edges it was cut along light up on
    the folded thing. Those edges are the seams.
    """
    t = phase % 1.0
    shut = 0.0 if t < 0.28 else min(1.0, (t - 0.28) / 0.34)
    shut = shut * shut * (3.0 - 2.0 * shut)
    glow = 0.0 if t < 0.66 else min(1.0, (t - 0.66) / 0.24)
    faces = _cube_and_net(x, y, w, h)

    for start, end in faces:
        corner = [_lerp(end[i], start[i], shut) for i in range(4)]
        _quad(shader, corner[0], corner[1], corner[2], corner[3], CONTROL,
              scale)
        edge = _ac(0.16 + 0.20 * (1.0 - shut) + 0.64 * glow)
        for i in range(4):
            _seg(shader, corner[i], corner[(i + 1) % 4], 1.3 + 1.2 * glow,
                 edge, scale)
        line = (1.0, 1.0, 1.0, 0.14)
        for i in (1, 2):
            f = i / 3.0
            _seg(shader, _lerp(corner[0], corner[1], f),
                 _lerp(corner[3], corner[2], f), 1.0, line, scale)
            _seg(shader, _lerp(corner[0], corner[3], f),
                 _lerp(corner[1], corner[2], f), 1.0, line, scale)


def _tip_bake(shader, x, y, w, h, phase, scale):
    """The original's surface carried onto the remesh."""
    t = phase % 1.0
    take = min(1.0, max(0.0, (t - 0.30) / 0.40))
    side = min((w - 40.0) * 0.5, h - 20.0)
    oy = y + (h - side) * 0.5
    lx = x + (w - side * 2.0 - 20.0) * 0.5
    rx = lx + side + 20.0

    def square(px, cells, checker):
        p0 = (px, oy)
        p1 = (px + side, oy)
        p2 = (px + side, oy + side)
        p3 = (px, oy + side)
        _quad(shader, p0, p1, p2, p3, (1.0, 1.0, 1.0, 0.055), scale)
        if checker > 0.01:
            _checker(shader, p0, p1, p2, p3, 4, 0.55 * checker, scale)
        line = (1.0, 1.0, 1.0, 0.18)
        for i in range(1, cells):
            f = i / cells
            _seg(shader, _lerp(p0, p1, f), _lerp(p3, p2, f), 1.0, line, scale)
            _seg(shader, _lerp(p0, p3, f), _lerp(p1, p2, f), 1.0, line, scale)
        for i in range(4):
            _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.3,
                 (1.0, 1.0, 1.0, 0.20), scale)

    square(lx, 7, 1.0)
    square(rx, 3, take)
    # the hand-off
    ax = lx + side + 5.0
    bx = rx - 5.0
    ay = oy + side * 0.5
    _seg(shader, (ax, ay), (bx, ay), 1.6, _ac(1.0), scale)
    _quad(shader, (bx, ay), (bx - 5.0, ay + 4.0), (bx - 5.0, ay - 4.0),
          (bx - 5.0, ay - 4.0), _ac(1.0), scale)


def _tip_pbr(shader, x, y, w, h, phase, scale):
    """One map, or the whole set of them."""
    lift = abs(math.sin(phase * math.pi))
    side = min(w * 0.42, h - 26.0)
    cx = x + w * 0.5
    cy = y + (h - side) * 0.5
    for k in range(3):
        off = k * 9.0 * lift
        px = cx - side * 0.5 - off * 0.6
        py = cy + off
        _quad(shader, (px, py), (px + side, py), (px + side, py + side),
              (px, py + side),
              (1.0, 1.0, 1.0, 0.05 + 0.03 * (2 - k)), scale)
        for i in range(4):
            a = ((px, py), (px + side, py), (px + side, py + side),
                 (px, py + side))[i]
            b = ((px + side, py), (px + side, py + side), (px, py + side),
                 (px, py))[i]
            _seg(shader, a, b, 1.2,
                 _ac(0.9 if k == 0 else 0.35), scale)


def _tip_gpu(shader, x, y, w, h, phase, scale):
    """The same bake, on the card and on the processor."""
    t = phase % 1.0
    fast = min(1.0, t * 2.4)
    slow = min(1.0, t * 0.85)
    bw = w - 24.0
    bx = x + 12.0
    for k, (frac, color) in enumerate(((fast, _ac(1.0)),
                                       (slow, (1.0, 1.0, 1.0, 0.22)))):
        by = y + h * (0.62 - k * 0.30)
        _fill(shader, bx * scale, by * scale, bw * scale, 9.0 * scale,
              4.5 * scale, (1.0, 1.0, 1.0, 0.10))
        _fill(shader, bx * scale, by * scale, bw * frac * scale, 9.0 * scale,
              4.5 * scale, color)


def _tip_sharp(shader, x, y, w, h, phase, scale):
    """A solid whose hard edges light up, and whose shading breaks on them.

    Sharp is not a line you draw on a model, it is a break in the shading: the
    two faces stop blending into each other. So the drawing shows the same
    solid twice over - soft, with the faces all one wash, then hard, with each
    face on its own tone and the edges between them lit.
    """
    on = abs(math.sin(phase * math.pi))
    side = min((w - 26.0) * 0.42, (h - 20.0) * 0.55)
    ux, uy = 0.866 * side, -0.5 * side
    vx, vy = -0.866 * side, -0.5 * side
    ax = x + w * 0.5
    ay = y + h * 0.5 + side * 0.92

    def at(du, dv, dd):
        return (ax + ux * du + vx * dv, ay + uy * du + vy * dv - side * dd)

    top = (at(0, 0, 0), at(1, 0, 0), at(1, 1, 0), at(0, 1, 0))
    right = (at(1, 0, 0), at(1, 0, 1), at(1, 1, 1), at(1, 1, 0))
    front = (at(0, 1, 0), at(1, 1, 0), at(1, 1, 1), at(0, 1, 1))

    # soft: one wash over all three. hard: each face its own tone.
    soft = 0.10
    for face, tone in ((top, 0.20), (front, 0.11), (right, 0.05)):
        v = soft + (tone - soft) * on
        _quad(shader, face[0], face[1], face[2], face[3],
              (1.0, 1.0, 1.0, v), scale)

    # the outline is always there, faint - it is the silhouette, not a mark
    for face in (top, right, front):
        for i in range(4):
            _seg(shader, face[i], face[(i + 1) % 4], 1.1,
                 (1.0, 1.0, 1.0, 0.16), scale)

    # the three edges where the faces meet: these are the sharp ones
    glow = _ac(0.12 + 0.88 * on)
    for a, b in ((at(0, 1, 0), at(1, 1, 0)),
                 (at(1, 1, 0), at(1, 0, 0)),
                 (at(1, 1, 0), at(1, 1, 1))):
        _seg(shader, a, b, 1.3 + 1.6 * on, glow, scale)


def _tip_unbake(shader, x, y, w, h, phase, scale):
    """The baked skin lifted straight off the model and let go.

    A wipe read as nothing much - it was the same square either way. This is
    the skin itself coming away: it peels up off the surface as one sheet,
    tilts as it goes, and thins out to nothing, leaving the bare model behind
    it. One object, one thing happening to it, and nothing it shares with the
    Bake drawing.
    """
    t = phase % 1.0
    lift = min(1.0, max(0.0, (t - 0.08) / 0.72))
    lift = lift * lift * (3.0 - 2.0 * lift)          # ease in and out

    side = min(w * 0.48, h - 30.0)
    ox = x + w * 0.30 - side * 0.5
    oy = y + (h - side) * 0.5 - 6.0
    p0 = (ox, oy)
    p1 = (ox + side, oy)
    p2 = (ox + side, oy + side)
    p3 = (ox, oy + side)

    # the model, always there under it
    _quad(shader, p0, p1, p2, p3, (1.0, 1.0, 1.0, 0.055), scale)
    line = (1.0, 1.0, 1.0, 0.18)
    for k in range(1, 4):
        f = k / 4.0
        _seg(shader, _lerp(p0, p1, f), _lerp(p3, p2, f), 1.0, line, scale)
        _seg(shader, _lerp(p0, p3, f), _lerp(p1, p2, f), 1.0, line, scale)
    for k in range(4):
        _seg(shader, (p0, p1, p2, p3)[k], (p1, p2, p3, p0)[k], 1.3,
             (1.0, 1.0, 1.0, 0.22), scale)

    # the baked skin, coming off it
    fade = 1.0 - min(1.0, max(0.0, (lift - 0.45) / 0.55))
    if fade <= 0.01:
        return
    dx = side * 0.72 * lift
    dy = side * 0.22 * lift
    lean = side * 0.20 * lift            # the far edge runs ahead: it tilts
    shrink = 1.0 - 0.18 * lift

    def moved(pt, extra):
        cx = ox + side * 0.5
        cy = oy + side * 0.5
        return (cx + (pt[0] - cx) * shrink + dx + extra,
                cy + (pt[1] - cy) * shrink + dy)

    s0 = moved(p0, 0.0)
    s1 = moved(p1, 0.0)
    s2 = moved(p2, lean)
    s3 = moved(p3, lean)
    _checker(shader, s0, s1, s2, s3, 4, 0.55 * fade, scale)
    edge = _ac(0.30 + 0.55 * fade)
    for k in range(4):
        _seg(shader, (s0, s1, s2, s3)[k], (s1, s2, s3, s0)[k], 1.4, edge,
             scale)


def _tip_purequad(shader, x, y, w, h, phase, scale):
    """A patch that still has triangles in it, and the same patch without.

    Quad-dominant means "quads where it can, triangles where it cannot". Pure
    quads means there is no triangle left anywhere - so the drawing is one
    piece of mesh with the diagonals showing, and then the same piece with
    them gone.
    """
    on = abs(math.sin(phase * math.pi))
    cols, rows = 4, 3
    cw = (w - 20.0) / cols
    ch = min((h - 22.0) / rows, cw)
    ox = x + (w - cw * cols) * 0.5
    oy = y + (h - ch * rows) * 0.5

    def pt(i, j):
        return (ox + i * cw, oy + j * ch)

    # the faces, tinted as they come good
    _quad(shader, pt(0, 0), pt(cols, 0), pt(cols, rows), pt(0, rows),
          _ac(0.04 + 0.10 * on), scale)

    line = (1.0, 1.0, 1.0, 0.22)
    for i in range(cols + 1):
        _seg(shader, pt(i, 0), pt(i, rows), 1.1, line, scale)
    for j in range(rows + 1):
        _seg(shader, pt(0, j), pt(cols, j), 1.1, line, scale)

    # the diagonals are the triangles. They are what "pure quads" removes.
    gone = (1.0, 0.45, 0.35, 0.85 * (1.0 - on))
    if 1.0 - on > 0.02:
        for i, j in ((0, 2), (2, 1), (3, 0)):
            _seg(shader, pt(i, j), pt(i + 1, j + 1), 1.5, gone, scale)


def _tip_reproject(shader, x, y, w, h, phase, scale):
    """The new mesh pulled back down onto the shape it came from.

    A remesher builds an even lattice first and only then asks where the old
    surface actually was. Each step drags the new points back onto it, so a
    low count leaves the result floating over the detail and a high one hugs
    it - at the price of the flow being smoothed as it goes.
    """
    steps = 4
    k = min(1.0, (phase % 1.0) / 0.85)
    k = math.floor(k * steps) / float(steps)      # it moves in steps, not flow

    x0 = x + 12.0
    x1 = x + w - 12.0
    mid = y + h * 0.5
    amp = min(h * 0.24, 22.0)

    def surf(u):
        return (x0 + (x1 - x0) * u,
                mid + math.sin(u * math.pi * 1.6 - 0.5) * amp)

    def flat(u):
        return (x0 + (x1 - x0) * u, mid + amp * 0.55)

    # the original surface
    prev = None
    for i in range(41):
        pt = surf(i / 40.0)
        if prev is not None:
            _seg(shader, prev, pt, 1.4, (1.0, 1.0, 1.0, 0.28), scale)
        prev = pt

    # the new mesh, on its way back onto it
    pts = []
    for i in range(9):
        u = i / 8.0
        a = flat(u)
        b = surf(u)
        pts.append((a[0], a[1] + (b[1] - a[1]) * k))

    if k < 0.999:
        for i, pt in enumerate(pts):
            b = surf(i / 8.0)
            _seg(shader, pt, b, 1.0, (1.0, 1.0, 1.0, 0.16), scale)

    for i in range(len(pts) - 1):
        _seg(shader, pts[i], pts[i + 1], 1.8, _ac(1.0), scale)
    for pt in pts:
        _fill(shader, (pt[0] - 2.0) * scale, (pt[1] - 2.0) * scale,
              4.0 * scale, 4.0 * scale, 2.0 * scale, _ac(1.0))


def _tip_accent(shader, x, y, w, h, phase, scale):
    """The same card, wearing a different colour."""
    t = phase % 1.0
    hue = t
    band = min(h - 34.0, 16.0)
    bw = w - 24.0
    bx = x + 12.0
    by = y + h - band - 10.0
    steps = 30
    step = bw / float(steps)
    for i in range(steps):
        _fill(shader, (bx + i * step) * scale, by * scale,
              (step + 0.8) * scale, band * scale, 0.0,
              accent_rgb(i / (steps - 1.0), 0.92) + (1.0,))
    kx = bx + bw * hue
    _fill(shader, (kx - 6.0) * scale, (by - 3.0) * scale, 12.0 * scale,
          (band + 6.0) * scale, 6.0 * scale, KNOB)

    # and what it is doing: a control, in whatever colour is under the knob
    col = accent_rgb(hue, 0.92) + (1.0,)
    ry = y + 8.0
    rh = min(h - band - 30.0, 22.0)
    _fill(shader, bx * scale, ry * scale, (bw * 0.52) * scale, rh * scale,
          8.0 * scale, col)
    _fill(shader, (bx + bw * 0.58) * scale, (ry + rh * 0.5 - 4.0) * scale,
          (bw * 0.42) * scale, 8.0 * scale, 4.0 * scale, col)


def _tip_corner(shader, x, y, w, h, phase, scale):
    """The same three controls, from square to fully rounded."""
    f = abs(math.sin(phase * math.pi))
    bw = w - 40.0
    bx = x + 20.0
    for k, (bh, col) in enumerate(((22.0, _ac(1.0)),
                                   (18.0, CONTROL_HOVER),
                                   (18.0, CONTROL))):
        by = y + h - 16.0 - k * 26.0
        _fill(shader, bx * scale, by * scale, bw * scale, bh * scale,
              (bh * 0.5) * f * scale, col)


def _patch(shader, x, y, w, h, cols, rows, scale, fill=None,
           line=(1.0, 1.0, 1.0, 0.18), border=(1.0, 1.0, 1.0, 0.22)):
    """A rectangle with a grid ruled across it - the stand-in for a mesh."""
    p0 = (x, y)
    p1 = (x + w, y)
    p2 = (x + w, y + h)
    p3 = (x, y + h)
    if fill is not None:
        _quad(shader, p0, p1, p2, p3, fill, scale)
    for i in range(1, cols):
        f = i / float(cols)
        _seg(shader, _lerp(p0, p1, f), _lerp(p3, p2, f), 1.0, line, scale)
    for j in range(1, rows):
        f = j / float(rows)
        _seg(shader, _lerp(p0, p3, f), _lerp(p1, p2, f), 1.0, line, scale)
    for i in range(4):
        _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.3, border,
             scale)


def _dot(shader, px, py, r, color, scale):
    _fill(shader, (px - r) * scale, (py - r) * scale, r * 2.0 * scale,
          r * 2.0 * scale, r * scale, color)


def _arc(shader, cx, cy, r, a0, a1, width, color, scale, steps=16):
    prev = None
    for i in range(steps + 1):
        a = a0 + (a1 - a0) * i / steps
        pt = (cx + math.cos(a) * r, cy + math.sin(a) * r)
        if prev is not None:
            _seg(shader, prev, pt, width, color, scale)
        prev = pt


def _box_edges(shader, ox, oy, pw, ph, scale, color=(1.0, 1.0, 1.0, 0.22)):
    c = ((ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph))
    for i in range(4):
        _seg(shader, c[i], c[(i + 1) % 4], 1.3, color, scale)


def _tip_remesh(shader, x, y, w, h, phase, scale):
    """Uneven mesh, then the same shape carried on even quads."""
    t = phase % 1.0
    done = min(1.0, max(0.0, (t - 0.15) / 0.55))
    side = min((w - 60.0) * 0.5, h - 20.0)
    oy = y + (h - side) * 0.5
    lx = x + (w - side * 2.0 - 18.0) * 0.5
    rx = lx + side + 18.0

    _quad(shader, (lx, oy), (lx + side, oy), (lx + side, oy + side),
          (lx, oy + side), (1.0, 1.0, 1.0, 0.05), scale)
    rows = ((0.0, 0.34, 3), (0.34, 0.62, 5), (0.62, 1.0, 2))
    for a, b, n in rows:
        ya = oy + side * a
        yb = oy + side * b
        _seg(shader, (lx, ya), (lx + side, ya), 1.0, (1.0, 1.0, 1.0, 0.20),
             scale)
        for k in range(1, n):
            f = k / float(n)
            _seg(shader, (lx + side * f, ya),
                 (lx + side * min(1.0, f + 0.10), yb), 1.0,
                 (1.0, 1.0, 1.0, 0.20), scale)
    _box_edges(shader, lx, oy, side, side, scale)

    if done > 0.02:
        _patch(shader, rx, oy, side, side, 4, 4, scale,
               fill=_ac(0.05 + 0.07 * done),
               line=(1.0, 1.0, 1.0, 0.18 * done),
               border=_ac(0.30 + 0.60 * done))
    ay = oy + side * 0.5
    _seg(shader, (lx + side + 4.0, ay), (rx - 4.0, ay), 1.6, _ac(0.9), scale)


def _tip_hardsurface(shader, x, y, w, h, phase, scale):
    """A panelled shape, re-ruled so the quads run along its own edges."""
    t = phase % 1.0
    snap = min(1.0, max(0.0, (t - 0.20) / 0.50))
    side = min(w - 30.0, h - 20.0)
    ox = x + (w - side) * 0.5
    oy = y + (h - side) * 0.5
    cut = side * 0.38

    poly = ((ox, oy), (ox + side, oy), (ox + side, oy + side - cut),
            (ox + side - cut, oy + side - cut), (ox + side - cut, oy + side),
            (ox, oy + side))
    _quad(shader, poly[0], poly[1], poly[2], poly[5], (1.0, 1.0, 1.0, 0.05),
          scale)
    _quad(shader, poly[5], poly[2], poly[3], poly[4], (1.0, 1.0, 1.0, 0.05),
          scale)

    n = 4
    sk = (1.0 - snap) * side * 0.22
    for i in range(1, n):
        f = i / float(n)
        _seg(shader, (ox + side * f + sk, oy),
             (ox + side * f - sk, oy + side), 1.0,
             (1.0, 1.0, 1.0, 0.16 + 0.10 * snap), scale)
        _seg(shader, (ox, oy + side * f - sk),
             (ox + side, oy + side * f + sk), 1.0,
             (1.0, 1.0, 1.0, 0.16 + 0.10 * snap), scale)

    for i in range(len(poly)):
        _seg(shader, poly[i], poly[(i + 1) % len(poly)], 1.5,
             _ac(0.35 + 0.65 * snap), scale)


def _tip_strictparts(shader, x, y, w, h, phase, scale):
    """Counting, which is what the switch is really about.

    The top row is the parts painted on the model, always four of them. The
    bottom row is the pieces the sheet is cut into. On, there are four, one
    under each part, joined to it. Off, the engine is free to rework them,
    and the bottom row comes back with more pieces than there were parts, or
    with fewer.
    """
    t = phase % 1.0
    keep = t < 0.48
    n_top = 4
    if keep:
        n_bot = 4
    elif t < 0.66:
        n_bot = 6
    elif t < 0.83:
        n_bot = 2
    else:
        n_bot = 5
    band = min(w - 40.0, 180.0)
    ox = x + (w - band) * 0.5
    bh = 15.0
    gap_y = 26.0
    ty = y + h * 0.5 + gap_y * 0.5 - bh * 0.5
    by = y + h * 0.5 - gap_y * 0.5 - bh * 0.5

    def row(n, oy, strong):
        gap = 5.0
        bw = (band - gap * (n - 1)) / n
        out = []
        for k in range(n):
            px = ox + k * (bw + gap)
            _fill(shader, px * scale, oy * scale, bw * scale, bh * scale,
                  3.0 * scale, _ac(0.30 + 0.16 * (k % 3)) if strong
                  else (1.0, 1.0, 1.0, 0.13))
            out.append(px + bw * 0.5)
        return out

    top = row(n_top, ty, True)
    bot = row(n_bot, by, keep)
    if keep:
        # one under each, joined, so the counting is not left to be guessed
        lit = min(n_top, int((t / 0.48) * (n_top + 1)))
        for k in range(n_top):
            col = _ac(0.95) if k < lit else _ac(0.28)
            _seg(shader, (top[k], ty), (bot[k], by + bh), 1.8, col, scale)
    else:
        # no ties at all: whatever came out, came out
        for k in range(n_bot):
            _seg(shader, (bot[k] - 3.0, by + bh * 0.5),
                 (bot[k] + 3.0, by + bh * 0.5), 1.4,
                 (1.0, 1.0, 1.0, 0.22), scale)


def _tip_uvorient(shader, x, y, w, h, phase, scale):
    """Islands turned every which way, then all brought round so the same
    direction runs up the sheet - the arrows are the grain of the texture."""
    t = phase % 1.0
    laid = _ease(min(1.0, max(0.0, (t - 0.25) / 0.45)))
    pw = (w - 46.0) / 3.0
    ph = min(h - 26.0, pw * 1.3)
    oy = y + (h - ph) * 0.5
    start = (-120.0, 165.0, -40.0)
    for k in range(3):
        ox = x + 15.0 + k * (pw + 8.0)
        _patch(shader, ox, oy, pw, ph, 2, 2, scale,
               fill=(1.0, 1.0, 1.0, 0.05), line=(1.0, 1.0, 1.0, 0.10),
               border=(1.0, 1.0, 1.0, 0.18))
        ang = math.radians(start[k] * (1.0 - laid) + 90.0 * laid)
        cx, cy = ox + pw * 0.5, oy + ph * 0.5
        L = min(pw, ph) * 0.34
        dx, dy = math.cos(ang) * L, math.sin(ang) * L
        tip = (cx + dx, cy + dy)
        col = _ac(0.45 + 0.55 * laid)
        _seg(shader, (cx - dx, cy - dy), tip, 2.0, col, scale)
        # the head of the arrow, so up is up and not just a line
        for s in (-1.0, 1.0):
            a2 = ang + s * math.radians(148.0)
            _seg(shader, tip,
                 (tip[0] + math.cos(a2) * L * 0.45,
                  tip[1] + math.sin(a2) * L * 0.45), 2.0, col, scale)


def _tip_crease(shader, x, y, w, h, phase, scale):
    """Two faces meeting, and the angle between them being measured."""
    t = phase % 1.0
    ang = math.radians(20.0 + 130.0 * abs(math.sin(t * math.pi)))
    cx = x + w * 0.5
    cy = y + h * 0.40
    L = min(w * 0.34, h * 0.50)
    a = (cx - L, cy)
    b = (cx + math.cos(math.pi - ang) * L, cy + math.sin(math.pi - ang) * L)
    hot = ang > math.radians(75.0)
    _seg(shader, a, (cx, cy), 2.0, (1.0, 1.0, 1.0, 0.30), scale)
    _seg(shader, (cx, cy), b, 2.0, (1.0, 1.0, 1.0, 0.30), scale)
    _arc(shader, cx, cy, L * 0.34, math.pi, math.pi - ang, 1.2,
         (1.0, 1.0, 1.0, 0.22), scale)
    if hot:
        _dot(shader, cx, cy, 7.0, _ac(0.20), scale)
    _dot(shader, cx, cy, 3.5 if hot else 2.2,
         _ac(1.0) if hot else (1.0, 1.0, 1.0, 0.35), scale)


def _tip_calibrate(shader, x, y, w, h, phase, scale):
    """Asked for a count, missed it, measured, and came back onto it."""
    t = phase % 1.0
    x0 = x + 16.0
    x1 = x + w - 16.0
    tx = x0 + (x1 - x0) * 0.55
    ty = y + h * 0.5
    _seg(shader, (x0, ty), (x1, ty), 1.2, (1.0, 1.0, 1.0, 0.20), scale)
    _seg(shader, (tx, ty - 9.0), (tx, ty + 9.0), 1.6, (1.0, 1.0, 1.0, 0.45),
         scale)
    if t < 0.30:
        px = x0 + (x1 - x0) * (t / 0.30)
    elif t < 0.55:
        px = x1
    elif t < 0.82:
        f = (t - 0.55) / 0.27
        px = x1 + (tx - x1) * f
    else:
        px = tx
    _dot(shader, px, ty, 5.0, _ac(1.0), scale)


def _tip_search(shader, x, y, w, h, phase, scale):
    """Several settings tried, and the one that came out best kept."""
    t = phase % 1.0
    n = 3
    gap = 8.0
    pw = (w - 24.0 - gap * (n - 1)) / n
    ph = min(h - 24.0, pw)
    oy = y + (h - ph) * 0.5
    ox = x + 12.0
    tried = min(n, int(t / 0.22))
    best = 1
    picked = t > 0.62
    for k in range(n):
        px = ox + k * (pw + gap)
        show = k < tried
        win = picked and k == best
        _patch(shader, px, oy, pw, ph, 2 + k, 2 + k, scale,
               fill=_ac(0.14) if win else (1.0, 1.0, 1.0,
                                           0.05 if show else 0.02),
               line=(1.0, 1.0, 1.0, 0.18 if show else 0.0),
               border=_ac(1.0) if win else (1.0, 1.0, 1.0,
                                            0.20 if show else 0.08))


def _tip_autothresh(shader, x, y, w, h, phase, scale):
    """The numbers found for you instead of dialled in by hand."""
    t = phase % 1.0
    bw = w - 36.0
    bx = x + 18.0
    for k, target in enumerate((0.62, 0.34)):
        by = y + h * (0.58 - k * 0.24)
        _fill(shader, bx * scale, by * scale, bw * scale, 5.0 * scale,
              2.5 * scale, (1.0, 1.0, 1.0, 0.14))
        f = min(1.0, max(0.0, (t - 0.10 - k * 0.14) / 0.45)) * target
        _fill(shader, bx * scale, by * scale, bw * f * scale, 5.0 * scale,
              2.5 * scale, _ac(1.0))
        _dot(shader, bx + bw * f, by + 2.5, 6.0, KNOB, scale)


def _tip_weldlines(shader, x, y, w, h, phase, scale):
    """Two lines almost on top of each other, made into one."""
    t = phase % 1.0
    close = min(1.0, max(0.0, (t - 0.15) / 0.55))
    x0 = x + 16.0
    x1 = x + w - 16.0
    cy = y + h * 0.5
    d = 9.0 * (1.0 - close)
    for s in (1.0, -1.0):
        prev = None
        for i in range(25):
            u = i / 24.0
            pt = (x0 + (x1 - x0) * u, cy + s * d + math.sin(u * 5.0) * 4.0)
            if prev is not None:
                _seg(shader, prev, pt, 1.6,
                     _ac(0.35 + 0.65 * close) if close > 0.75
                     else (1.0, 1.0, 1.0, 0.45),
                     scale)
            prev = pt
    if close < 0.98:
        for u in (0.2, 0.5, 0.8):
            px = x0 + (x1 - x0) * u
            py = cy + math.sin(u * 5.0) * 4.0
            _seg(shader, (px, py - d), (px, py + d), 1.0,
                 (1.0, 1.0, 1.0, 0.18), scale)


def _tip_straightloop(shader, x, y, w, h, phase, scale):
    """Loops that cross flat ground and say nothing, taken out."""
    keep = abs(math.sin(phase * math.pi))
    pw = w - 28.0
    ph = min(h - 24.0, 52.0)
    ox = x + 14.0
    oy = y + (h - ph) * 0.5
    _quad(shader, (ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph),
          (1.0, 1.0, 1.0, 0.05), scale)
    for i in range(1, 8):
        f = i / 8.0
        extra = i % 2 == 1
        if extra:
            col = _ac(0.60 * keep)
        else:
            col = (1.0, 1.0, 1.0, 0.20)
        _seg(shader, (ox + pw * f, oy), (ox + pw * f, oy + ph), 1.1, col,
             scale)
    _box_edges(shader, ox, oy, pw, ph, scale)


def _tip_minlen(shader, x, y, w, h, phase, scale):
    """Stubs too short to be worth a line, dropped."""
    keep = abs(math.sin(phase * math.pi))
    x0 = x + 16.0
    span = w - 32.0
    lens = (1.0, 0.18, 0.72, 0.10, 0.55, 0.24)
    n = len(lens)
    for i, L in enumerate(lens):
        ly = y + h - 14.0 - i * ((h - 26.0) / float(n - 1))
        short = L < 0.30
        a = 0.55 if not short else 0.55 * keep
        _seg(shader, (x0, ly), (x0 + span * L, ly), 2.0,
             _ac(a) if short else (1.0, 1.0, 1.0, a), scale)


def _tip_circle(shader, x, y, w, h, phase, scale):
    """A ring that came out of the mesh as a polygon, put back on a circle."""
    r = min(w, h) * 0.30
    cx = x + w * 0.5
    cy = y + h * 0.5
    round_ = abs(math.sin(phase * math.pi))
    n = 7
    _arc(shader, cx, cy, r, 0.0, 2.0 * math.pi, 1.0, (1.0, 1.0, 1.0, 0.16),
         scale, steps=30)
    prev = None
    for i in range(n + 1):
        a = 2.0 * math.pi * i / n
        rr = r * (1.0 - 0.18 * (1.0 - round_) * abs(math.cos(a * 1.5)))
        pt = (cx + math.cos(a) * rr, cy + math.sin(a) * rr)
        if prev is not None:
            _seg(shader, prev, pt, 1.8, _ac(0.35 + 0.65 * round_), scale)
        prev = pt


def _tip_typed(shader, x, y, w, h, phase, scale):
    """A straight border kept straight, a curved one kept curved."""
    on = abs(math.sin(phase * math.pi))
    x0 = x + 16.0
    x1 = x + w - 16.0
    top = y + h * 0.72
    bot = y + h * 0.28
    _seg(shader, (x0, top), (x1, top), 2.0, _ac(0.25 + 0.75 * on), scale)
    prev = None
    for i in range(25):
        u = i / 24.0
        pt = (x0 + (x1 - x0) * u, bot + math.sin(u * math.pi) * 16.0)
        if prev is not None:
            _seg(shader, prev, pt, 2.0, _ac(0.25 + 0.75 * on), scale)
        prev = pt
    for ys in (top, bot):
        prev = None
        for i in range(13):
            u = i / 12.0
            base = ys + (math.sin(u * math.pi) * 16.0 if ys == bot else 0.0)
            pt = (x0 + (x1 - x0) * u, base + math.sin(u * 9.0) * 3.0
                  * (1.0 - on))
            if prev is not None:
                _seg(shader, prev, pt, 1.0,
                     (1.0, 1.0, 1.0, 0.18 * (1.0 - on)), scale)
            prev = pt


def _tip_gap(shader, x, y, w, h, phase, scale):
    """A line that stops short, carried across to the one facing it."""
    t = phase % 1.0
    grow = min(1.0, max(0.0, (t - 0.20) / 0.50))
    cy = y + h * 0.5
    x0 = x + 14.0
    x1 = x + w - 14.0
    ga = x0 + (x1 - x0) * 0.30
    gb = x0 + (x1 - x0) * 0.74
    _seg(shader, (x0, cy), (ga, cy), 2.0, (1.0, 1.0, 1.0, 0.40), scale)
    _seg(shader, (gb, cy), (x1, cy), 2.0, (1.0, 1.0, 1.0, 0.40), scale)
    _dot(shader, ga, cy, 2.6, (1.0, 1.0, 1.0, 0.55), scale)
    _dot(shader, gb, cy, 2.6, (1.0, 1.0, 1.0, 0.55), scale)
    if grow > 0.01:
        _seg(shader, (ga, cy), (ga + (gb - ga) * grow, cy), 3.2, _ac(1.0),
             scale)
        _dot(shader, ga + (gb - ga) * grow, cy, 3.0, _ac(1.0), scale)



def _tip_mirror(shader, x, y, w, h, phase, scale):
    """One side built, the other side taken from it."""
    t = phase % 1.0
    over = min(1.0, max(0.0, (t - 0.20) / 0.50))
    side = min(w * 0.36, h - 22.0)
    cx = x + w * 0.5
    oy = y + (h - side) * 0.5
    _patch(shader, cx - side - 6.0, oy, side, side, 3, 3, scale,
           fill=_ac(0.10), border=_ac(0.85))
    _seg(shader, (cx, oy - 8.0), (cx, oy + side + 8.0), 1.2,
         (1.0, 1.0, 1.0, 0.30), scale)
    if over > 0.02:
        _patch(shader, cx + 6.0, oy, side * over, side, 3, 3, scale,
               fill=_ac(0.10 * over), border=_ac(0.85 * over))







def _tip_marked(shader, x, y, w, h, phase, scale):
    """The lines it found, and the lines you drew, on the same model."""
    only = abs(math.sin(phase * math.pi))
    pw = w - 28.0
    ph = min(h - 24.0, 56.0)
    ox = x + 14.0
    oy = y + (h - ph) * 0.5
    _quad(shader, (ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph),
          (1.0, 1.0, 1.0, 0.05), scale)
    found = (1.0, 1.0, 1.0, 0.32 * (1.0 - only))
    for f in (0.22, 0.50, 0.78):
        _seg(shader, (ox + pw * f, oy), (ox + pw * f, oy + ph), 1.3, found,
             scale)
    _seg(shader, (ox, oy + ph * 0.55), (ox + pw, oy + ph * 0.55), 1.3, found,
         scale)
    _seg(shader, (ox + pw * 0.12, oy + ph * 0.20),
         (ox + pw * 0.88, oy + ph * 0.80), 2.0, _ac(1.0), scale)
    _box_edges(shader, ox, oy, pw, ph, scale)


def _tip_clean(shader, x, y, w, h, phase, scale):
    """A knotted patch taken out and laid again as plain rows."""
    t = phase % 1.0
    fixed = min(1.0, max(0.0, (t - 0.20) / 0.50))
    pw = w - 28.0
    ph = min(h - 24.0, 56.0)
    ox = x + 14.0
    oy = y + (h - ph) * 0.5
    _quad(shader, (ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph),
          (1.0, 1.0, 1.0, 0.05), scale)
    mess = (1.0, 0.45, 0.35, 0.55 * (1.0 - fixed))
    if fixed < 0.98:
        knots = (((0.18, 0.10), (0.44, 0.86)), ((0.30, 0.90), (0.62, 0.14)),
                 ((0.12, 0.55), (0.78, 0.42)), ((0.55, 0.08), (0.86, 0.72)))
        for a, b in knots:
            _seg(shader, (ox + pw * a[0], oy + ph * a[1]),
                 (ox + pw * b[0], oy + ph * b[1]), 1.4, mess, scale)
    if fixed > 0.02:
        for i in range(1, 5):
            f = i / 5.0
            _seg(shader, (ox + pw * f, oy), (ox + pw * f, oy + ph), 1.1,
                 _ac(0.55 * fixed), scale)
        for j in range(1, 3):
            f = j / 3.0
            _seg(shader, (ox, oy + ph * f), (ox + pw, oy + ph * f), 1.1,
                 _ac(0.55 * fixed), scale)
    _box_edges(shader, ox, oy, pw, ph, scale)


def _tip_confine(shader, x, y, w, h, phase, scale):
    """The tidying stops at the line and does not run past it."""
    t = phase % 1.0
    grow = min(1.0, max(0.0, (t - 0.15) / 0.55))
    pw = w - 28.0
    ph = min(h - 24.0, 56.0)
    ox = x + 14.0
    oy = y + (h - ph) * 0.5
    bound = ox + pw * 0.62
    _quad(shader, (ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph),
          (1.0, 1.0, 1.0, 0.05), scale)
    _quad(shader, (ox, oy), (ox + pw * 0.62 * grow, oy),
          (ox + pw * 0.62 * grow, oy + ph), (ox, oy + ph), _ac(0.16), scale)
    _seg(shader, (bound, oy - 6.0), (bound, oy + ph + 6.0), 2.0, _ac(1.0),
         scale)
    _box_edges(shader, ox, oy, pw, ph, scale)


def _tip_useless(shader, x, y, w, h, phase, scale):
    """Loops on flat ground go, loops that hold a curve stay."""
    keep = abs(math.sin(phase * math.pi))
    x0 = x + 14.0
    x1 = x + w - 14.0
    base = y + h * 0.40

    def prof(u):
        if u < 0.5:
            return base
        return base + math.sin((u - 0.5) * math.pi * 2.0) * 20.0

    prev = None
    for i in range(41):
        u = i / 40.0
        pt = (x0 + (x1 - x0) * u, prof(u))
        if prev is not None:
            _seg(shader, prev, pt, 1.6, (1.0, 1.0, 1.0, 0.35), scale)
        prev = pt
    for i in range(1, 12):
        u = i / 12.0
        px = x0 + (x1 - x0) * u
        py = prof(u)
        flat = u < 0.5
        a = 0.55 if not flat else 0.55 * keep
        _seg(shader, (px, py - 9.0), (px, py + 9.0), 1.4,
             _ac(a) if flat else (1.0, 1.0, 1.0, a), scale)


def _tip_every(shader, x, y, w, h, phase, scale):
    """One loop out of every few kept, the rest let go."""
    keep = abs(math.sin(phase * math.pi))
    x0 = x + 14.0
    span = w - 28.0
    cy = y + h * 0.5
    n = 10
    for i in range(n):
        px = x0 + span * i / float(n - 1)
        stay = i % 3 == 0
        a = 0.60 if stay else 0.60 * (1.0 - keep)
        _seg(shader, (px, cy - 16.0), (px, cy + 16.0), 1.8,
             _ac(a) if stay else (1.0, 1.0, 1.0, a), scale)


def _tip_reset(shader, x, y, w, h, phase, scale):
    """Everything the add-on put on the model, taken back off."""
    t = phase % 1.0
    wipe = min(1.0, max(0.0, (t - 0.15) / 0.55))
    pw = w - 28.0
    ph = min(h - 24.0, 56.0)
    ox = x + 14.0
    oy = y + (h - ph) * 0.5
    _quad(shader, (ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph),
          (1.0, 1.0, 1.0, 0.05), scale)
    left = 1.0 - wipe
    for i in range(1, 6):
        f = i / 6.0
        if f > left:
            continue
        _seg(shader, (ox + pw * f, oy), (ox + pw * f, oy + ph), 1.4,
             _ac(0.75), scale)
    _seg(shader, (ox + pw * left, oy - 5.0), (ox + pw * left, oy + ph + 5.0),
         1.6, (1.0, 1.0, 1.0, 0.30), scale)
    _box_edges(shader, ox, oy, pw, ph, scale)


def _tip_merge(shader, x, y, w, h, phase, scale):
    """Points closer together than the distance, made into one."""
    t = phase % 1.0
    close = min(1.0, max(0.0, (t - 0.15) / 0.55))
    cx = x + w * 0.5
    cy = y + h * 0.5
    r = min(w, h) * 0.22
    _arc(shader, cx, cy, r, 0.0, 2.0 * math.pi, 1.0, (1.0, 1.0, 1.0, 0.20),
         scale, steps=28)
    if close > 0.9:
        _dot(shader, cx, cy, 6.0, _ac(0.25), scale)
    for dx, dy in ((-0.75, 0.35), (0.62, 0.58), (0.30, -0.70),
                   (-0.55, -0.45)):
        _dot(shader, cx + dx * r * (1.0 - close),
             cy + dy * r * (1.0 - close), 3.2, _ac(1.0), scale)


def _tip_confirm(shader, x, y, w, h, phase, scale):
    """The result kept, and carried on with as the model."""
    t = phase % 1.0
    tick = min(1.0, max(0.0, (t - 0.25) / 0.45))
    side = min(w - 40.0, h - 22.0)
    ox = x + (w - side) * 0.5
    oy = y + (h - side) * 0.5
    _patch(shader, ox, oy, side, side, 4, 4, scale, fill=_ac(0.08),
           border=_ac(0.55))
    # a tick, drawn on
    a = (ox + side * 0.22, oy + side * 0.52)
    b = (ox + side * 0.44, oy + side * 0.28)
    c = (ox + side * 0.82, oy + side * 0.76)
    if tick > 0.0:
        f = min(1.0, tick / 0.4)
        _seg(shader, a, (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f),
             3.0, _ac(1.0), scale)
    if tick > 0.4:
        f = min(1.0, (tick - 0.4) / 0.6)
        _seg(shader, b, (b[0] + (c[0] - b[0]) * f, b[1] + (c[1] - b[1]) * f),
             3.0, _ac(1.0), scale)


def _tip_restore(shader, x, y, w, h, phase, scale):
    """The remesh set aside and the model you started with brought back."""
    t = phase % 1.0
    back = min(1.0, max(0.0, (t - 0.15) / 0.55))
    side = min(w * 0.40, h - 24.0)
    cx = x + w * 0.5
    oy = y + (h - side) * 0.5
    # the remesh, sliding out to the right and fading
    if back < 0.98:
        px = cx - side * 0.5 + side * 1.3 * back
        _patch(shader, px, oy, side, side, 4, 4, scale,
               fill=_ac(0.08 * (1.0 - back)),
               line=(1.0, 1.0, 1.0, 0.16 * (1.0 - back)),
               border=_ac(0.75 * (1.0 - back)))
    # the original, coming back in from the left
    if back > 0.02:
        px = cx - side * 0.5 - side * 1.3 * (1.0 - back)
        _patch(shader, px, oy, side, side, 2, 2, scale,
               fill=(1.0, 1.0, 1.0, 0.06 * back),
               line=(1.0, 1.0, 1.0, 0.20 * back),
               border=(1.0, 1.0, 1.0, 0.30 * back))


def _tip_valence(shader, x, y, w, h, phase, scale):
    """Stray points, spurs and pass-through vertices, cleared out.

    A vertex with fewer than three edges is not holding anything: a loose
    point holds nothing at all, a spur holds one edge that goes nowhere, and
    a point sitting between two edges is a bend in a line that could be one
    line. All three go.
    """
    gone = abs(math.sin(phase * math.pi))
    left = 1.0 - gone
    pw = w - 30.0
    ph = min(h - 26.0, 54.0)
    ox = x + 15.0
    oy = y + (h - ph) * 0.5

    _quad(shader, (ox, oy), (ox + pw, oy), (ox + pw, oy + ph), (ox, oy + ph),
          (1.0, 1.0, 1.0, 0.05), scale)
    line = (1.0, 1.0, 1.0, 0.22)
    for i in range(1, 3):
        f = i / 3.0
        _seg(shader, (ox + pw * f, oy), (ox + pw * f, oy + ph), 1.1, line,
             scale)
    _seg(shader, (ox, oy + ph * 0.5), (ox + pw, oy + ph * 0.5), 1.1, line,
         scale)
    _box_edges(shader, ox, oy, pw, ph, scale)

    # a point loose in the middle of nothing
    _dot(shader, ox + pw * 0.52, oy + ph * 0.78, 3.0, _ac(left), scale)
    # a spur: one edge hanging off the border and stopping
    sx = ox + pw * 0.33
    _seg(shader, (sx, oy), (sx, oy - 13.0), 1.8, _ac(left), scale)
    _dot(shader, sx, oy - 13.0, 3.0, _ac(left), scale)
    # and a point sitting between two edges, where one line would do
    _dot(shader, ox + pw * 0.667, oy + ph * 0.5, 3.0, _ac(left), scale)


# -- tutorial posters -------------------------------------------------------
#
# A play triangle says "there is a video". The video's own thumbnail says
# which video, which is the thing worth knowing before clicking. Posters live
# beside this file in assets/ and are loaded once: Blender decodes the file,
# the pixels are copied into a texture, and the image datablock is thrown away
# again so nothing is left behind in the user's .blend.

_POSTERS = {}


def _poster(name):
    """The named thumbnail as a texture, or None if it is not shipped."""
    if name in _POSTERS:
        return _POSTERS[name]
    tex = None
    folder = os.path.join(os.path.dirname(__file__), "assets")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        path = os.path.join(folder, name + ext)
        if not os.path.exists(path):
            continue
        img = None
        try:
            img = bpy.data.images.load(path)
            # Read the file's own values, not a colour-managed version of
            # them: what is drawn here goes straight to the screen.
            img.colorspace_settings.name = "Non-Color"
            # Small, because the popup is small. A full thumbnail is two
            # million pixels to copy through Python for a 170 pixel picture.
            img.scale(384, 216)
            w, h = img.size
            buf = gpu.types.Buffer("FLOAT", w * h * 4, img.pixels[:])
            tex = gpu.types.GPUTexture((w, h), format="RGBA16F", data=buf)
        except Exception:
            tex = None
        finally:
            if img is not None:
                try:
                    bpy.data.images.remove(img)
                except Exception:
                    pass
        break
    _POSTERS[name] = tex
    return tex


def forget_posters():
    _POSTERS.clear()


def _draw_poster(tex, x, y, w, h, scale):
    """One textured rectangle. Leaves the shader unbound - the caller rebinds."""
    sh = gpu.shader.from_builtin("IMAGE")
    batch = batch_for_shader(sh, "TRI_FAN", {
        "pos": ((x * scale, y * scale), ((x + w) * scale, y * scale),
                ((x + w) * scale, (y + h) * scale),
                (x * scale, (y + h) * scale)),
        "texCoord": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))})
    sh.bind()
    sh.uniform_sampler("image", tex)
    batch.draw(sh)


def _tip_watch_cloth(shader, x, y, w, h, phase, scale):
    """The cloth lesson's own thumbnail."""
    _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_cloth")


def _tip_watch_hard(shader, x, y, w, h, phase, scale):
    """The hard surface lesson's own thumbnail."""
    _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_hardsurface")


def _tip_watch_parts(shader, x, y, w, h, phase, scale):
    """The texel density lesson's own thumbnail."""
    _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_parts")


def _tip_watch_udim(shader, x, y, w, h, phase, scale):
    """The UDIM lesson's own thumbnail."""
    _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_udim")


def _tip_watch_symmetry(shader, x, y, w, h, phase, scale):
    """The symmetry lesson's own thumbnail."""
    _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_symmetry")


def _tip_watch_stack(shader, x, y, w, h, phase, scale):
    """The stacked repeats lesson's own thumbnail."""
    _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_stack")


def _tip_watch(shader, x, y, w, h, phase, scale, poster="tutorial_overview"):
    """The tutorial's own thumbnail, with a play mark over it."""
    pulse = 0.5 + 0.5 * abs(math.sin(phase * math.pi))
    fw = min(w - 24.0, 160.0)
    fh = fw * 0.5625
    if fh > h - 16.0:
        fh = h - 16.0
        fw = fh / 0.5625
    ox = x + (w - fw) * 0.5
    oy = y + (h - fh) * 0.5

    tex = _poster(poster)
    if tex is not None:
        _draw_poster(tex, ox, oy, fw, fh, scale)
        shader.bind()                       # the tip carries on in flat colour
        # a soft scrim so the play mark reads over a bright frame
        _fill(shader, ox * scale, oy * scale, fw * scale, fh * scale, 0.0,
              (0.0, 0.0, 0.0, 0.12))
    else:
        # nothing shipped yet: the frame on its own
        _fill(shader, ox * scale, oy * scale, fw * scale, fh * scale,
              7.0 * scale, _ac(0.20 + 0.25 * pulse))
        _fill(shader, (ox + 1.5) * scale, (oy + 1.5) * scale,
              (fw - 3.0) * scale, (fh - 3.0) * scale, 6.0 * scale,
              (0.09, 0.09, 0.10, 1.0))

    side = fh * 0.24
    cx = ox + fw * 0.5
    cy = oy + fh * 0.5
    r = side * 0.92
    _fill(shader, (cx - r) * scale, (cy - r) * scale, r * 2.0 * scale,
          r * 2.0 * scale, r * scale, _ac(0.55 + 0.35 * pulse))
    bx = (cx - side * 0.28) * scale
    by = (cy - side * 0.5) * scale
    s3 = side * scale
    batch = batch_for_shader(shader, "TRIS", {"pos": (
        (bx, by), (bx, by + s3), (bx + s3 * 0.88, by + s3 * 0.5))})
    shader.uniform_float("color", ON_ACCENT)
    batch.draw(shader)


def _tip_classic(shader, x, y, w, h, phase, scale):
    """This card, or the side panel the add-on shipped with."""
    over = abs(math.sin(phase * math.pi))
    pw = min(w * 0.40, 76.0)
    ph = min(h - 20.0, 82.0)
    cx = x + w * 0.5
    oy = y + (h - ph) * 0.5

    def frame(px, lit, rounded):
        r = (7.0 if rounded else 1.0) * scale
        _fill(shader, px * scale, oy * scale, pw * scale, ph * scale, r,
              (1.0, 1.0, 1.0, 0.05 + 0.06 * lit))
        rows = 5
        for i in range(rows):
            ry = oy + ph - 14.0 - i * ((ph - 20.0) / rows)
            rw = pw - 16.0
            rr = (5.0 if rounded else 0.0) * scale
            colour = _ac(0.25 + 0.65 * lit) if i == 0 else (
                1.0, 1.0, 1.0, 0.10 + 0.12 * lit)
            _fill(shader, (px + 8.0) * scale, ry * scale, rw * scale,
                  8.0 * scale, rr, colour)

    frame(cx - pw - 8.0, 1.0 - over, True)
    frame(cx + 8.0, over, False)


def _tip_flatfan(shader, x, y, w, h, phase, scale):
    """A flat panel cut into a fan, and the same panel as one face.

    The points Milad circled sit on the long diagonal of a fan. They are not
    loose - they each hold three edges - so nothing that hunts loose points
    will ever reach them. The area has to come out and go back in whole.
    """
    clean = abs(math.sin(phase * math.pi))
    side = min(w - 30.0, h - 20.0)
    ox = x + (w - side) * 0.5
    oy = y + (h - side) * 0.5
    p0 = (ox, oy)
    p1 = (ox + side, oy)
    p2 = (ox + side, oy + side)
    p3 = (ox, oy + side)
    _quad(shader, p0, p1, p2, p3, _ac(0.05 + 0.08 * clean), scale)

    # the fan, and the points sitting along its long edge
    fade = 1.0 - clean
    if fade > 0.02:
        line = (1.0, 1.0, 1.0, 0.30 * fade)
        for f in (0.25, 0.5, 0.75, 1.0):
            _seg(shader, p0, _lerp(p3, p2, f), 1.1, line, scale)
        for f in (0.3, 0.55, 0.8):
            pt = _lerp(p0, p2, f)
            _seg(shader, pt, _lerp(p1, p2, f * 0.9), 1.0, line, scale)
            _dot(shader, pt[0], pt[1], 2.6, _ac(0.9 * fade), scale)

    for i in range(4):
        _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i],
             1.4 + 1.0 * clean, _ac(0.35 + 0.65 * clean), scale)


def _tip_flatkeep(shader, x, y, w, h, phase, scale):
    """The same flat panel, with one line on it that has to stay."""
    clean = abs(math.sin(phase * math.pi))
    side = min(w - 30.0, h - 20.0)
    ox = x + (w - side) * 0.5
    oy = y + (h - side) * 0.5
    p0 = (ox, oy)
    p1 = (ox + side, oy)
    p2 = (ox + side, oy + side)
    p3 = (ox, oy + side)
    _quad(shader, p0, p1, p2, p3, (1.0, 1.0, 1.0, 0.05), scale)
    fade = 1.0 - clean
    if fade > 0.02:
        line = (1.0, 1.0, 1.0, 0.28 * fade)
        for f in (0.25, 0.5, 0.75, 1.0):
            _seg(shader, p0, _lerp(p3, p2, f), 1.1, line, scale)
    for i in range(4):
        _seg(shader, (p0, p1, p2, p3)[i], (p1, p2, p3, p0)[i], 1.3,
             (1.0, 1.0, 1.0, 0.22), scale)
    # the line that was asked for, untouched throughout
    _seg(shader, _lerp(p0, p3, 0.45), _lerp(p1, p2, 0.45), 2.4, _ac(1.0),
         scale)



def _wrap(text, width, size, scale):
    blf.size(0, size * scale)
    out = []
    line = ""
    for word in text.split():
        probe = (line + " " + word).strip()
        if blf.dimensions(0, probe)[0] > width * scale and line:
            out.append(line)
            line = word
        else:
            line = probe
    if line:
        out.append(line)
    return out[:4]


def _tip_usethis(shader, x, y, w, h, phase, scale):
    """A piece pulled out of a result, told to stand on its own.

    The link back to the model it came from is what makes a remesh of the
    piece rebuild the whole thing, so the drawing is that link breaking.
    """
    t = phase % 1.0
    cut = min(1.0, max(0.0, (t - 0.25) / 0.45))
    big = min(w * 0.34, h - 26.0)
    small = big * 0.42
    ly = y + (h - big) * 0.5
    lx = x + w * 0.18 - big * 0.5
    rx = x + w * 0.80 - small * 0.5
    ry = y + (h - small) * 0.5

    _patch(shader, lx, ly, big, big, 3, 3, scale,
           fill=(1.0, 1.0, 1.0, 0.05),
           line=(1.0, 1.0, 1.0, 0.16),
           border=(1.0, 1.0, 1.0, 0.22 + 0.0 * cut))
    _patch(shader, rx, ry, small, small, 2, 2, scale,
           fill=_ac(0.08 + 0.10 * cut),
           line=(1.0, 1.0, 1.0, 0.18),
           border=_ac(0.35 + 0.65 * cut))

    # the tie between them, breaking in the middle
    ax = lx + big + 6.0
    bx = rx - 6.0
    cy = y + h * 0.5
    mid = (ax + bx) * 0.5
    gap = (bx - ax) * 0.5 * cut
    _seg(shader, (ax, cy), (mid - gap, cy), 1.8,
         (1.0, 1.0, 1.0, 0.35 * (1.0 - cut) + 0.12), scale)
    _seg(shader, (mid + gap, cy), (bx, cy), 1.8,
         (1.0, 1.0, 1.0, 0.35 * (1.0 - cut) + 0.12), scale)


def _tip_cage(shader, x, y, w, h, phase, scale):
    """A dense model, a small cage over it, and the cage smoothed back onto it.

    Three steps, left to right: what came in, the handful of quads laid over
    it, and what those quads become once they are subdivided with the hard
    edges creased.
    """
    t = phase % 1.0
    step = min(1.0, max(0.0, (t - 0.10) / 0.75)) * 2.0
    side = min((w - 40.0) / 3.0, h - 22.0)
    oy = y + (h - side) * 0.5
    gap = (w - side * 3.0) / 4.0

    def at(i):
        return x + gap + i * (side + gap)

    # the model as it arrives
    _patch(shader, at(0), oy, side, side, 6, 6, scale,
           fill=(1.0, 1.0, 1.0, 0.05),
           line=(1.0, 1.0, 1.0, 0.16),
           border=(1.0, 1.0, 1.0, 0.22))
    # the cage
    a = min(1.0, max(0.0, step))
    if a > 0.02:
        _patch(shader, at(1), oy, side, side, 2, 2, scale,
               fill=_ac(0.06 * a), line=_ac(0.55 * a), border=_ac(0.85 * a))
    # and what it becomes
    b = min(1.0, max(0.0, step - 1.0))
    if b > 0.02:
        _patch(shader, at(2), oy, side, side, 4, 4, scale,
               fill=_ac(0.10 * b),
               line=(1.0, 1.0, 1.0, 0.20 * b),
               border=_ac(0.9 * b))
        # the creased edge that stays sharp through the smoothing
        p0 = (at(2), oy + side * 0.5)
        p1 = (at(2) + side, oy + side * 0.5)
        _seg(shader, p0, p1, 2.4, _ac(b), scale)
    for i in (0, 1):
        if step > i:
            ax = at(i) + side + gap * 0.15
            bx = at(i + 1) - gap * 0.15
            cy = oy + side * 0.5
            _seg(shader, (ax, cy), (bx, cy), 1.6, _ac(0.8), scale)


# Which drawing goes with which explanation. Rebuilt after a careless
# regex took the whole table out with one of its entries.
# -- what the Image to 3D controls do, drawn ---------------------------------
#
# The same idea as the rest of the card: a control is explained by watching it
# happen, not by a sentence about it. These are the ones about cutting and
# flattening, which the remesh side had no need of.

def _tip_au_unwrap(shader, x, y, w, h, phase, scale):
    """A closed box, cut along the lit edges, opening out flat."""
    t = abs(math.sin(phase * math.pi))
    s = min(w * 0.22, h * 0.30)
    cx, cy = x + w * 0.5, y + h * 0.5
    _patch(shader, cx - s * 0.5, cy - s * 0.5, s, s, 3, 3, scale,
           fill=CONTROL_HOVER)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ln = s * t
        if ln < 1.0:
            continue
        if dx:
            px = cx + s * 0.5 if dx > 0 else cx - s * 0.5 - ln
            _patch(shader, px, cy - s * 0.5, ln, s, 2, 3, scale, fill=CONTROL)
        else:
            py = cy + s * 0.5 if dy > 0 else cy - s * 0.5 - ln
            _patch(shader, cx - s * 0.5, py, s, ln, 3, 2, scale, fill=CONTROL)
    # the cuts the flaps opened along
    for a, b in (((cx - s * 0.5, cy - s * 0.5), (cx + s * 0.5, cy - s * 0.5)),
                 ((cx - s * 0.5, cy + s * 0.5), (cx + s * 0.5, cy + s * 0.5)),
                 ((cx - s * 0.5, cy - s * 0.5), (cx - s * 0.5, cy + s * 0.5)),
                 ((cx + s * 0.5, cy - s * 0.5), (cx + s * 0.5, cy + s * 0.5))):
        _aaline(shader, [a, b], 1.6, _ac(0.35 + 0.65 * t), scale)


def _tip_au_parts(shader, x, y, w, h, phase, scale):
    """One model, and the pieces the add-on sees in it."""
    t = abs(math.sin(phase * math.pi))
    bw = min(w * 0.42, 86.0)
    bh = min(h - 24.0, 70.0)
    ox = x + (w - bw) * 0.5
    oy = y + (h - bh) * 0.5
    cuts = (0.0, 0.34, 0.66, 1.0)
    cols = (_ac(0.55), CONTROL_HOVER, CONTROL)
    for k in range(3):
        a, b = cuts[k], cuts[k + 1]
        off = (k - 1) * 7.0 * t
        _patch(shader, ox, oy + bh * a + off, bw, bh * (b - a) - 1.0,
               3, 2, scale, fill=cols[k])


def _prio_draw(shader, x, y, w, h, scale, n, skew, step):
    """The whole trade in one picture: the model, then the sheet.

    On the left is the model, a curved thing, with the cuts on it drawn in
    orange - those are the lines somebody has to live with. On the right is
    what comes off it: n pieces, with the checker inside them pulled about
    by however much curve each piece was left carrying. Cut it once and one
    piece swallows the lot, so its squares come out crooked. Cut it four
    times and the squares are square.

    The outline of a flat piece stays a rectangle and only its grid is
    warped: a piece that is the wrong shape reads as perspective, a square
    with crooked squares in it reads as what it is.
    """
    mw = 58.0
    mh = min(h - 30.0, 54.0)
    my = y + (h - mh) * 0.5 + 3.0
    mx = x + 8.0
    # the model: a curved shell, seen from the side
    steps = 18
    top = [(mx + mw * (i / steps),
            my + mh * (0.62 + 0.38 * math.sin(math.pi * (i / steps))))
           for i in range(steps + 1)]
    bot = [(px, my + mh * 0.10) for px, _py in top]
    _aaline(shader, top, 1.2, (1.0, 1.0, 1.0, 0.38), scale)
    _aaline(shader, bot, 1.2, (1.0, 1.0, 1.0, 0.24), scale)
    _aaline(shader, [top[0], bot[0]], 1.2, (1.0, 1.0, 1.0, 0.24), scale)
    _aaline(shader, [top[-1], bot[-1]], 1.2, (1.0, 1.0, 1.0, 0.24), scale)
    for k in range(1, n):
        f = k / float(n)
        i0 = int(round(f * steps))
        _aaline(shader, [top[i0], (top[i0][0], my + mh * 0.10)], 2.0,
                _ac(0.95), scale)
    # the way across
    ax = mx + mw + 9.0
    ay = my + mh * 0.40
    _aaline(shader, [(ax, ay), (ax + 11.0, ay)], 1.3,
            (1.0, 1.0, 1.0, 0.34), scale)
    _aaline(shader, [(ax + 6.0, ay + 4.0), (ax + 11.0, ay),
                     (ax + 6.0, ay - 4.0)], 1.3,
            (1.0, 1.0, 1.0, 0.34), scale)
    # the sheet
    bx = ax + 20.0
    bw = max(40.0, x + w - 8.0 - bx)
    bh = mh
    oy = my
    gap = 4.0
    pw = (bw - gap * (n - 1)) / n
    cols, rows = 3, 3
    for k in range(n):
        px = bx + k * (pw + gap)
        cx = px + pw * 0.5

        def P(i, j):
            fx = i / float(cols)
            fy = j / float(rows)
            bow = 0.17 * skew * math.sin(math.pi * fx) * math.sin(math.pi * fy)
            crowd = 0.15 * skew * math.sin(math.pi * fy)
            return (cx + (fx - 0.5 + bow) * pw, oy + (fy + crowd) * bh)

        for j in range(rows + 1):
            _aaline(shader, [P(i, j) for i in range(cols + 1)], 1.1,
                    (1.0, 1.0, 1.0, 0.22), scale)
        for i in range(cols + 1):
            _aaline(shader, [P(i, j) for j in range(rows + 1)], 1.1,
                    (1.0, 1.0, 1.0, 0.22), scale)
    # which of the three this drawing belongs to
    sw = 22.0
    sx = x + (w - (sw * 3.0 + 10.0)) * 0.5
    for i in range(3):
        _fill(shader, (sx + i * (sw + 5.0)) * scale, (y + 5.0) * scale,
              sw * scale, 3.0 * scale, 1.5 * scale,
              _ac(0.95) if i == step else (1.0, 1.0, 1.0, 0.16))


def _prio_settle(phase):
    """Run once and stop, rather than swing back and forth for ever.

    Swinging was the bug: Less Stretch went one piece to four and Fewer
    Seams went four to one, and a second later both were just pieces coming
    and going, which is the same drawing twice. Each one now walks to its
    own answer and stays there, so what is on screen after a moment IS the
    choice: one crooked piece, or four square ones.
    """
    t = min(1.0, max(0.0, phase * 0.9))
    return t * t * (3.0 - 2.0 * t)


def _tip_au_prio_less(shader, x, y, w, h, phase, scale):
    """Less Stretch: cut it again and again until it lies flat."""
    e = _prio_settle(phase)
    n = 1 + int(e * 3.999)
    _prio_draw(shader, x, y, w, h, scale, n, (4 - n) / 3.0, 0)


def _tip_au_prio_bal(shader, x, y, w, h, phase, scale):
    """Balanced: one cut taken, the little that is left carried."""
    e = _prio_settle(phase)
    _prio_draw(shader, x, y, w, h, scale, 1 + int(e * 1.999),
               0.62 - 0.36 * e, 1)


def _tip_au_prio_few(shader, x, y, w, h, phase, scale):
    """Fewer Seams: the cuts go away and the piece carries the pull."""
    e = _prio_settle(phase)
    n = 4 - int(e * 3.999)
    _prio_draw(shader, x, y, w, h, scale, n, (4 - n) / 3.0, 2)


def _tip_au_pack(shader, x, y, w, h, phase, scale):
    """The pieces gathered into the square, none of them touching."""
    t = abs(math.sin(phase * math.pi))
    s = min(w * 0.34, h - 20.0)
    ox = x + (w - s) * 0.5
    oy = y + (h - s) * 0.5
    _box_edges(shader, ox, oy, s, s, scale)
    iso = (((0.04, 0.52, 0.44, 0.44), (-0.34, 0.22)),
           ((0.52, 0.52, 0.44, 0.44), (0.38, 0.26)),
           ((0.04, 0.04, 0.92, 0.44), (0.06, -0.38)))
    for (fx, fy, fw, fh), (dx, dy) in iso:
        px = ox + s * (fx + dx * (1.0 - t))
        py = oy + s * (fy + dy * (1.0 - t))
        _patch(shader, px + 1.5, py + 1.5, s * fw - 3.0, s * fh - 3.0,
               2, 2, scale, fill=CONTROL_HOVER if t > 0.9 else CONTROL)


def _tip_au_margin(shader, x, y, w, h, phase, scale):
    """The gap left between two pieces, so their edges do not bleed."""
    t = abs(math.sin(phase * math.pi))
    bw = min(w * 0.26, 60.0)
    bh = min(h - 30.0, 58.0)
    gap = 4.0 + 20.0 * t
    cx, cy = x + w * 0.5, y + (h - bh) * 0.5
    _patch(shader, cx - gap * 0.5 - bw, cy, bw, bh, 2, 2, scale, fill=CONTROL)
    _patch(shader, cx + gap * 0.5, cy, bw, bh, 2, 2, scale, fill=CONTROL)
    _aaline(shader, [(cx - gap * 0.5, cy + bh * 0.5),
                     (cx + gap * 0.5, cy + bh * 0.5)], 1.6, _ac(0.9), scale)


def _tip_au_relax(shader, x, y, w, h, phase, scale):
    """A crumpled piece, evened out."""
    t = abs(math.sin(phase * math.pi))
    bw = min(w - 60.0, 104.0)
    bh = min(h - 26.0, 68.0)
    ox = x + (w - bw) * 0.5
    oy = y + (h - bh) * 0.5
    rows, cols = 4, 5

    def pt(i, j):
        fx, fy = i / float(cols), j / float(rows)
        n = math.sin((i * 2.3 + j * 3.7)) * (1.0 - t)
        edge = 0.0 if (i in (0, cols) or j in (0, rows)) else 1.0
        return (ox + bw * fx + n * 7.0 * edge, oy + bh * fy + n * 5.0 * edge)

    for j in range(rows + 1):
        _aaline(shader, [pt(i, j) for i in range(cols + 1)], 1.1,
                (1.0, 1.0, 1.0, 0.22 + 0.10 * t), scale)
    for i in range(cols + 1):
        _aaline(shader, [pt(i, j) for j in range(rows + 1)], 1.1,
                (1.0, 1.0, 1.0, 0.22 + 0.10 * t), scale)


def _tip_au_count(shader, x, y, w, h, phase, scale):
    """The same model asked for in fewer pieces, or in more."""
    t = abs(math.sin(phase * math.pi))
    bw = min(w - 60.0, 118.0)
    bh = min(h - 30.0, 60.0)
    ox = x + (w - bw) * 0.5
    oy = y + (h - bh) * 0.5
    n = 2 + int(round(t * 4.0))
    for k in range(n):
        _patch(shader, ox + bw * k / n + 1.5, oy, bw / n - 3.0, bh, 2, 2,
               scale, fill=_ac(0.5) if k % 2 == 0 else CONTROL)


def _tip_au_chunk(shader, x, y, w, h, phase, scale):
    """A piece too big to solve in one go, cut at its creases into chunks
    that are then solved side by side. A bigger number means fewer, larger
    chunks - and fewer islands at the end."""
    t = abs(math.sin(phase * math.pi))
    bw = min(w - 60.0, 126.0)
    bh = min(h - 28.0, 64.0)
    ox = x + (w - bw) * 0.5
    oy = y + (h - bh) * 0.5
    # one piece at t=0, three chunks at t=1, drawn apart by what they were cut
    n = 3
    gap = 7.0 * t
    piece = (bw - gap * (n - 1)) / n
    for k in range(n):
        px = ox + k * (piece + gap)
        _patch(shader, px, oy, piece, bh, 2, 3, scale,
               fill=CONTROL_HOVER if t > 0.5 else CONTROL)
        if k and t > 0.05:
            # the crease the cut was made along
            _aaline(shader, [(px - gap * 0.5, oy - 3.0),
                             (px - gap * 0.5, oy + bh + 3.0)],
                    1.8, _ac(0.35 + 0.65 * t), scale)
    # all of them under way at the same time
    for k in range(n):
        px = ox + k * (piece + gap)
        _fill(shader, px * scale, (oy - 12.0) * scale, piece * scale,
              4.0 * scale, 2.0 * scale, CONTROL)
        _fill(shader, px * scale, (oy - 12.0) * scale,
              piece * t * scale, 4.0 * scale, 2.0 * scale, _ac(0.85))


def _tip_au_udim(shader, x, y, w, h, phase, scale):
    """The layout laid across several tiles instead of one square."""
    t = abs(math.sin(phase * math.pi))
    s = min(w * 0.19, h - 26.0)
    gap = 5.0
    n = 3
    ox = x + (w - (s * n + gap * (n - 1))) * 0.5
    oy = y + (h - s) * 0.5
    for k in range(n):
        tx = ox + k * (s + gap)
        live = t * n > k
        _box_edges(shader, tx, oy, s, s, scale,
                   color=(1.0, 1.0, 1.0, 0.30 if live else 0.12))
        if not live:
            continue
        # about the same amount of surface on every tile
        for (fx, fy, fw, fh) in ((0.06, 0.52, 0.42, 0.42),
                                 (0.52, 0.52, 0.42, 0.42),
                                 (0.06, 0.06, 0.88, 0.42)):
            _patch(shader, tx + s * fx, oy + s * fy, s * fw, s * fh, 2, 2,
                   scale, fill=_ac(0.45))
    # the tile numbers, as ticks under them
    for k in range(n):
        tx = ox + k * (s + gap)
        _fill(shader, (tx + s * 0.3) * scale, (oy - 9.0) * scale,
              s * 0.4 * scale, 3.0 * scale, 1.5 * scale,
              _ac(0.8) if t * n > k else CONTROL)


def _udim_tiles(shader, x, y, w, h, scale, n=3):
    """Three empty tiles, and where they sit."""
    s = min(w * 0.19, h - 26.0)
    gap = 5.0
    ox = x + (w - (s * n + gap * (n - 1))) * 0.5
    oy = y + (h - s) * 0.5
    for k in range(n):
        tx = ox + k * (s + gap)
        _box_edges(shader, tx, oy, s, s, scale, color=(1.0, 1.0, 1.0, 0.30))
        _fill(shader, (tx + s * 0.3) * scale, (oy - 9.0) * scale,
              s * 0.4 * scale, 3.0 * scale, 1.5 * scale, _ac(0.55))
    return ox, oy, s, gap


def _tip_au_udim_obj(shader, x, y, w, h, phase, scale):
    """Object: each model whole on a tile of its own.

    Three objects, three tiles, and each object lands complete on one of
    them - which is how a vehicle or a set is textured, one part at a time.
    They fill their tiles by different amounts, and that is the price of
    keeping them whole.
    """
    t = abs(math.sin(phase * math.pi))
    ox, oy, s, gap = _udim_tiles(shader, x, y, w, h, scale)
    # each object is one block of its own size, arriving on its own tile
    sizes = (0.86, 0.62, 0.74)
    for k, f in enumerate(sizes):
        tx = ox + k * (s + gap)
        grow = min(1.0, max(0.0, t * 3.0 - k))
        if grow <= 0.0:
            continue
        b = s * f * grow
        _patch(shader, tx + (s - b) * 0.5, oy + (s - b) * 0.5, b, b, 2, 2,
               scale, fill=_ac(0.45))


def _tip_au_udim_bal(shader, x, y, w, h, phase, scale):
    """Balanced: the islands dealt out so every tile carries the same.

    The pieces are shared by area rather than by which object they came
    from, so all three tiles end up equally full and the texel density is
    the same across the set. One object can end up split over two tiles,
    which is what the dealing costs.
    """
    t = abs(math.sin(phase * math.pi))
    ox, oy, s, gap = _udim_tiles(shader, x, y, w, h, scale)
    deal = ((0.06, 0.52, 0.42, 0.42), (0.52, 0.52, 0.42, 0.42),
            (0.06, 0.06, 0.88, 0.42))
    for k in range(3):
        tx = ox + k * (s + gap)
        for j, (fx, fy, fw, fh) in enumerate(deal):
            # the pieces fly in from the tile next door: they belong to
            # whichever tile has room, not to the object they came from
            a = min(1.0, max(0.0, t * 4.0 - (k * 3 + j) * 0.25))
            if a <= 0.0:
                continue
            px = tx + s * fx - (1.0 - a) * (s + gap) * (1 if k else -1)
            _patch(shader, px, oy + s * fy, s * fw, s * fh, 2, 2,
                   scale, fill=_ac(0.45))


def _tip_au_smart(shader, x, y, w, h, phase, scale):
    """Blender's own projection, taken part by part rather than all at once."""
    t = abs(math.sin(phase * math.pi))
    s = min(w * 0.17, h - 28.0)
    ox = x + (w - (s * 3.0 + 12.0)) * 0.5
    oy = y + (h - s) * 0.5
    for k in range(3):
        tx = ox + k * (s + 6.0)
        done = t * 3.0 > k
        _patch(shader, tx, oy, s, s, 3, 3, scale,
               fill=_ac(0.5) if done else CONTROL)
        if done:
            # the direction this part was projected from
            _aaline(shader, [(tx + s * 0.5, oy + s + 12.0),
                             (tx + s * 0.5, oy + s + 2.0)], 1.6, _ac(0.9),
                    scale)
            _aaline(shader, [(tx + s * 0.34, oy + s + 7.0),
                             (tx + s * 0.5, oy + s + 2.0),
                             (tx + s * 0.66, oy + s + 7.0)], 1.6, _ac(0.9),
                    scale)


def _tip_au_smart_angle(shader, x, y, w, h, phase, scale):
    """How far two faces may lean apart before the projection cuts there."""
    t = abs(math.sin(phase * math.pi))
    bw = min(w * 0.22, 60.0)
    bh = min(h - 30.0, 54.0)
    cx = x + w * 0.5
    oy = y + (h - bh) * 0.5
    a = 0.15 + t * 1.0
    # one face flat, the next one leaning away by the angle
    _patch(shader, cx - bw, oy, bw, bh, 2, 2, scale, fill=CONTROL)
    p0 = (cx, oy)
    p1 = (cx + math.cos(-a) * bw, oy + math.sin(-a) * bw + bh * 0.0)
    _quad(shader, p0, p1, (p1[0], p1[1] + bh), (p0[0], p0[1] + bh),
          CONTROL_HOVER, scale)
    _aaline(shader, [p0, (p0[0], p0[1] + bh)], 2.0,
            _ac(0.3 + 0.7 * min(1.0, t * 1.6)), scale)
    _arc(shader, cx, oy + bh * 0.5, 13.0, -a, 0.0, 1.4, TEXT_DIM, scale,
         steps=10)


def _tip_au_size(shader, x, y, w, h, phase, scale):
    """The same image with more pixels in it, or fewer."""
    t = abs(math.sin(phase * math.pi))
    s = min(w * 0.30, h - 22.0)
    ox = x + (w - s) * 0.5
    oy = y + (h - s) * 0.5
    n = 2 + int(round(t * 6.0))
    cell = s / n
    for j in range(n):
        for i in range(n):
            if (i + j) % 2:
                continue
            _fill(shader, (ox + i * cell) * scale, (oy + j * cell) * scale,
                  cell * scale, cell * scale, 0.0, _ac(0.75))
    _box_edges(shader, ox, oy, s, s, scale)


def _tip_au_preview(shader, x, y, w, h, phase, scale):
    """The cuts drawn onto the model before anything is unwrapped."""
    t = abs(math.sin(phase * math.pi))
    r = min(w * 0.17, h * 0.34)
    cx, cy = x + w * 0.5, y + h * 0.5
    # a head-ish shape, so the lines read as cuts on a model
    pts = []
    for i in range(30):
        a = 2.0 * math.pi * i / 30.0
        rr = r * (1.0 + 0.16 * math.cos(a) - 0.10 * math.sin(a * 2.0))
        pts.append((cx + math.cos(a) * rr * 0.86, cy + math.sin(a) * rr))
    _quad(shader, (cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r),
          (cx - r, cy + r), (0.0, 0.0, 0.0, 0.0), scale)
    _aaline(shader, pts, 1.4, (1.0, 1.0, 1.0, 0.28), scale, closed=True)
    # the seams, drawn on in the order the planner lays them
    runs = (((0.0, -1.0), (0.0, 1.0)),
            ((-0.78, 0.18), (0.0, 0.52)),
            ((0.78, 0.18), (0.0, 0.52)))
    for k, (a, b) in enumerate(runs):
        f = min(1.0, max(0.0, t * 3.0 - k))
        if f <= 0.0:
            continue
        p0 = (cx + a[0] * r * 0.82, cy + a[1] * r * 0.92)
        p1 = (cx + b[0] * r * 0.82, cy + b[1] * r * 0.92)
        _aaline(shader, [p0, (p0[0] + (p1[0] - p0[0]) * f,
                              p0[1] + (p1[1] - p0[1]) * f)],
                2.0, _ac(0.95), scale)
    ex = x + w * 0.17
    _arc(shader, ex, cy, 7.0, 0.0, math.pi * 2.0, 1.3, TEXT_DIM, scale,
         steps=16)
    _dot(shader, ex, cy, 2.6, TEXT_DIM, scale)


def _tip_au_showparts(shader, x, y, w, h, phase, scale):
    """The pieces painted onto the model, and taken off again."""
    on = abs(math.sin(phase * math.pi))
    bw = min(w * 0.40, 84.0)
    bh = min(h - 24.0, 68.0)
    ox = x + (w - bw) * 0.5
    oy = y + (h - bh) * 0.5
    cols = (_ac(0.65 * on), (0.35, 0.60, 0.95, 0.55 * on),
            (0.40, 0.85, 0.55, 0.50 * on))
    for k in range(3):
        _patch(shader, ox, oy + bh * k / 3.0, bw, bh / 3.0 - 1.0, 3, 2, scale,
               fill=cols[k] if on > 0.05 else CONTROL)


# -- Image to 3D explainers, one per control ----------------------------------

_PARTC = ((0.36, 0.78, 0.52), (0.38, 0.58, 0.98), (0.96, 0.58, 0.22),
          (0.85, 0.42, 0.78))


def _ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _swing(phase):
    """0 -> 1, a hold, and back, over one cycle."""
    return _ease(min(1.0, max(0.0, (0.5 - 0.5 * math.cos(2.0 * math.pi * phase)) * 1.4 - 0.2)))


def _pc(i, a):
    c = _PARTC[i % len(_PARTC)]
    return (c[0], c[1], c[2], a)


def _polyline_part(shader, pts, frac, width, color, scale):
    """The first `frac` of a polyline, so a line can be drawn in."""
    if frac <= 0.0:
        return
    L = [0.0]
    for a, b in zip(pts[:-1], pts[1:]):
        L.append(L[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    stop = L[-1] * min(1.0, frac)
    for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
        if L[i] >= stop:
            break
        seg = L[i + 1] - L[i]
        f = 1.0 if L[i + 1] <= stop else (stop - L[i]) / max(seg, 1e-9)
        _seg(shader, a, _lerp(a, b, f), width, color, scale)


def _tip_augarment(shader, x, y, w, h, phase, scale):
    """A shirt, and the seams an artist would sew it from drawn onto it one
    after another: armholes, side lines, collar, hem band."""
    bh = min(h - 16.0, 130.0)
    bw = bh * 1.25
    ox = x + (w - bw) * 0.5
    oy = y + (h - bh) * 0.5

    def P(u, v):
        return (ox + u * bw, oy + v * bh)
    shape = [P(0.30, 0.0), P(0.70, 0.0), P(0.70, 0.60), P(0.93, 0.47),
             P(1.0, 0.64), P(0.78, 0.90), P(0.60, 0.96), P(0.5, 0.88),
             P(0.40, 0.96), P(0.22, 0.90), P(0.0, 0.64), P(0.07, 0.47),
             P(0.30, 0.60), P(0.30, 0.0)]
    for a, b in zip(shape[:-1], shape[1:]):
        _seg(shader, a, b, 1.4, (1.0, 1.0, 1.0, 0.30), scale)
    seams = [
        [P(0.70, 0.60), P(0.73, 0.72), P(0.76, 0.82), P(0.78, 0.90)],
        [P(0.30, 0.60), P(0.27, 0.72), P(0.24, 0.82), P(0.22, 0.90)],
        [P(0.70, 0.60), P(0.70, 0.30), P(0.70, 0.0)],
        [P(0.30, 0.60), P(0.30, 0.30), P(0.30, 0.0)],
        [P(0.41, 0.92), P(0.5, 0.84), P(0.59, 0.92)],
        [P(0.30, 0.07), P(0.70, 0.07)],
    ]
    p = min(1.0, phase * 1.3) * len(seams)
    for i, sm in enumerate(seams):
        _polyline_part(shader, sm, p - i, 2.2, _ac(0.95), scale)


def _tip_aujoin(shader, x, y, w, h, phase, scale):
    """Two islands slide together and the seam between them goes."""
    t = _swing(phase)
    s = min(h - 40.0, 70.0)
    cy = y + (h - s) * 0.5
    gap = 30.0 * (1.0 - t)
    lx = x + w * 0.5 - s - gap * 0.5
    rx = x + w * 0.5 + gap * 0.5
    _patch(shader, lx, cy, s, s, 3, 3, scale, fill=_ac(0.10 + 0.10 * t))
    _patch(shader, rx, cy, s, s, 3, 3, scale, fill=_ac(0.10 + 0.10 * t))
    # the seam on the two facing edges fades once they meet
    a = 0.9 * (1.0 - t)
    _seg(shader, (lx + s, cy), (lx + s, cy + s), 2.2, _ac(a), scale)
    _seg(shader, (rx, cy), (rx, cy + s), 2.2, _ac(a), scale)
    if t > 0.95:
        _box_edges(shader, lx, cy, 2.0 * s + gap, s, scale, _ac(0.9))


def _tip_aupickisland(shader, x, y, w, h, phase, scale):
    """The pointer clicks islands on the model; the picked ones share a colour."""
    s = min(h - 40.0, 62.0)
    gap = 8.0
    total = 3 * s + 2 * gap
    ox = x + (w - total) * 0.5
    oy = y + (h - s) * 0.5
    step = min(2, int(phase * 3.0))
    picked = {0} if step >= 1 else set()
    if step >= 2:
        picked.add(2)
    for i in range(3):
        px = ox + i * (s + gap)
        fill = _ac(0.30) if i in picked else _pc(i, 0.30)
        _patch(shader, px, oy, s, s, 3, 3, scale, fill=fill,
               border=_ac(0.95) if i in picked else (1.0, 1.0, 1.0, 0.25))
    target = (0, 0, 2)[step]
    cx = ox + target * (s + gap) + s * 0.55
    _cursor(shader, cx, oy + s * 0.55, 13.0, scale)


def _tip_auredo(shader, x, y, w, h, phase, scale):
    """Only the selected island is flattened again; the rest stay put."""
    t = _swing(phase)
    s = min(h - 40.0, 60.0)
    gap = 12.0
    ox = x + (w - (3 * s + 2 * gap)) * 0.5
    oy = y + (h - s) * 0.5
    for i in (0, 2):
        _patch(shader, ox + i * (s + gap), oy, s, s, 3, 3, scale,
               fill=(1.0, 1.0, 1.0, 0.05))
    mx = ox + s + gap
    k = 14.0 * (1.0 - t)
    p0 = (mx + k, oy - k * 0.4)
    p1 = (mx + s - k * 0.6, oy + k * 0.5)
    p2 = (mx + s + k * 0.5, oy + s - k * 0.3)
    p3 = (mx - k * 0.4, oy + s + k * 0.6)
    _quad(shader, p0, p1, p2, p3, _ac(0.16), scale)
    _checker(shader, p0, p1, p2, p3, 3, 0.28, scale)
    for a, b in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
        _seg(shader, a, b, 1.6, _ac(0.95), scale)


def _tip_austack(shader, x, y, w, h, phase, scale):
    """Two identical parts land on one spot in the square and stay there."""
    t = _swing(phase)
    sq = min(h - 20.0, 110.0)
    ox = x + (w - sq) * 0.5
    oy = y + (h - sq) * 0.5
    _box_edges(shader, ox, oy, sq, sq, scale, (1.0, 1.0, 1.0, 0.30))
    s = sq * 0.32
    home = (ox + sq * 0.12, oy + sq * 0.55)
    start_b = (ox + sq * 0.55, oy + sq * 0.12)
    b = _lerp(start_b, home, t)
    _patch(shader, home[0], home[1], s, s, 2, 2, scale, fill=_ac(0.20))
    _patch(shader, b[0], b[1], s, s, 2, 2, scale, fill=_ac(0.20),
           border=_ac(0.9))
    # the rest of the sheet, packed as usual
    _patch(shader, ox + sq * 0.55, oy + sq * 0.55, sq * 0.33, sq * 0.33, 2, 2,
           scale, fill=(1.0, 1.0, 1.0, 0.06))


def _tip_aupickmerge(shader, x, y, w, h, phase, scale):
    """Click one piece, then its neighbour: the two become one piece."""
    s = min(h - 40.0, 60.0)
    ox = x + (w - 3 * s) * 0.5
    oy = y + (h - s) * 0.5
    step = min(2, int(phase * 3.0))
    for i in range(3):
        c = _pc(i, 0.35)
        if step >= 2 and i == 1:
            c = _pc(0, 0.35)
        _patch(shader, ox + i * s, oy, s, s, 3, 3, scale, fill=c,
               border=(1.0, 1.0, 1.0, 0.25))
    # the border between the two picked ones is drawn until they merge
    if step < 2:
        _seg(shader, (ox + s, oy), (ox + s, oy + s), 2.0,
             (1.0, 1.0, 1.0, 0.7), scale)
    else:
        _box_edges(shader, ox, oy, 2 * s, s, scale, _ac(0.95))
    tx = ox + (0.5 if step == 0 else 1.5) * s
    _cursor(shader, tx, oy + s * 0.55, 13.0, scale)


def _tip_aumergesel(shader, x, y, w, h, phase, scale):
    """Everything selected becomes one piece."""
    t = _swing(phase)
    s = min(h - 36.0, 52.0)
    ox = x + (w - 4 * s) * 0.5
    oy = y + (h - s) * 0.5
    sel = (1, 2)
    for i in range(4):
        c = _pc(i, 0.35)
        if i in sel and t > 0.5:
            c = _ac(0.40)
        _patch(shader, ox + i * s, oy, s, s, 2, 2, scale, fill=c,
               border=(1.0, 1.0, 1.0, 0.25))
    if t <= 0.5:
        for i in sel:
            _box_edges(shader, ox + i * s, oy, s, s, scale, _ac(0.95))
    else:
        _box_edges(shader, ox + s, oy, 2 * s, s, scale, _ac(0.95))


def _tip_ausplit(shader, x, y, w, h, phase, scale):
    """The selected faces are cut out into a piece of their own."""
    t = _swing(phase)
    pw = min(w - 60.0, 150.0)
    ph = min(h - 40.0, 80.0)
    ox = x + (w - pw) * 0.5
    oy = y + (h - ph) * 0.5
    _patch(shader, ox, oy, pw, ph, 6, 3, scale, fill=_pc(1, 0.30))
    cw = pw / 6.0
    ch = ph / 3.0
    sx = ox + cw * 3.0
    sy = oy + ch * 1.0
    lift = 10.0 * t
    fill = _pc(2, 0.55) if t > 0.5 else _ac(0.25)
    _patch(shader, sx, sy + lift, cw * 2.0, ch * 2.0, 2, 2, scale, fill=fill,
           border=_ac(0.95))


def _tip_auaspect(shader, x, y, w, h, phase, scale):
    """On a wide image the squares stretch; Correct Aspect keeps them square."""
    t = _swing(phase)
    rh = min(h - 30.0, 70.0)
    rw = rh * 2.0
    ox = x + (w - rw) * 0.5
    oy = y + (h - rh) * 0.5
    cols = 3 if t < 0.5 else 6
    rows = 3
    cw = rw / cols
    ch = rh / rows
    for i in range(cols):
        for j in range(rows):
            if (i + j) % 2 == 0:
                _quad(shader, (ox + i * cw, oy + j * ch), (ox + (i + 1) * cw, oy + j * ch),
                      (ox + (i + 1) * cw, oy + (j + 1) * ch), (ox + i * cw, oy + (j + 1) * ch),
                      _ac(0.55), scale)
    _box_edges(shader, ox, oy, rw, rh, scale, (1.0, 1.0, 1.0, 0.35))


_TIP_ART = {
    "augarment": _tip_augarment,
    "aujoin": _tip_aujoin,
    "aupickisland": _tip_aupickisland,
    "auredo": _tip_auredo,
    "austack": _tip_austack,
    "aupickmerge": _tip_aupickmerge,
    "aumergesel": _tip_aumergesel,
    "ausplit": _tip_ausplit,
    "auaspect": _tip_auaspect,
    "auunwrap": _tip_au_unwrap,
    "ausmart": _tip_au_smart,
    "auparts": _tip_au_parts,
    "auprioless": _tip_au_prio_less,
    "aupriobal": _tip_au_prio_bal,
    "aupriofew": _tip_au_prio_few,
    "aupreview": _tip_au_preview,
    "ausize": _tip_au_size,
    "auudim": _tip_au_udim,
    "auudimobj": _tip_au_udim_obj,
    "auudimbal": _tip_au_udim_bal,
    None: _tip_au_smart,
    "ausmartangle": _tip_au_smart_angle,
    "aupack": _tip_au_pack,
    "aumargin": _tip_au_margin,
    "aurelax": _tip_au_relax,
    "aucount": _tip_au_count,
    "auchunk": _tip_au_chunk,
    "aushowparts": _tip_au_showparts,
    "cage": _tip_cage,
    "watchhard": _tip_watch_hard,
    "usethis": _tip_usethis,
    "accent": _tip_accent,
    "autothresh": _tip_autothresh,
    "bake": _tip_bake,
    "calibrate": _tip_calibrate,
    "circle": _tip_circle,
    "classic": _tip_classic,
    "clean": _tip_clean,
    "confine": _tip_confine,
    "confirm": _tip_confirm,
    "corner": _tip_corner,
    "crease": _tip_crease,
    "strictparts": _tip_strictparts,
    "uvorient": _tip_uvorient,
    "every": _tip_every,
    "flatfan": _tip_flatfan,
    "flatkeep": _tip_flatkeep,
    "fold": _tip_fold,
    "gap": _tip_gap,
    "gpu": _tip_gpu,
    "grid": _tip_grid,
    "hardsurface": _tip_hardsurface,
    "isolate": _tip_isolate,
    "marked": _tip_marked,
    "meet": _tip_meet,
    "merge": _tip_merge,
    "minlen": _tip_minlen,
    "mirror": _tip_mirror,
    "pbr": _tip_pbr,
    "pick": _tip_pick,
    "purequad": _tip_purequad,
    "relax": _tip_relax,
    "remesh": _tip_remesh,
    "reproject": _tip_reproject,
    "reset": _tip_reset,
    "restore": _tip_restore,
    "seams": _tip_seams,
    "search": _tip_search,
    "sharpedge": _tip_sharp,
    "shells": _tip_shells,
    "split": _tip_split,
    "straightloop": _tip_straightloop,
    "swap": _tip_swap,
    "tear": _tip_tear,
    "typed": _tip_typed,
    "unbake": _tip_unbake,
    "useless": _tip_useless,
    "uvkeep": _tip_uvkeep,
    "valence": _tip_valence,
    "watch": _tip_watch,
    "watchcloth": _tip_watch_cloth,
    "watchparts": _tip_watch_parts,
    "watchudim": _tip_watch_udim,
    "watchsym": _tip_watch_symmetry,
    "watchstack": _tip_watch_stack,
    "weld": _tip_weld,
    "weldlines": _tip_weldlines,
}


def _bake_state():
    """The bake in progress, as the bake operator reports it."""
    import sys
    mod = sys.modules.get(__package__ + ".lines_preview")
    return getattr(mod, "BAKE", None)


def _run_state():
    """The remesh in progress, as the Organic operator reports it."""
    import sys
    pkg = sys.modules.get(__package__)
    return getattr(pkg, "RUN", None)


def _draw_tip(shader, card, scale):
    tip = _state["tip"]
    if tip is None:
        return
    index, key, since = tip
    waited = time.time() - since
    if waited < TIP_DELAY:
        return
    found = TIPS.get(key)
    widgets = _state["widgets"]
    if not found or index >= len(widgets):
        return
    title, body, art = found
    wx, wy, ww, wh = widgets[index]["rect"]
    # blf leaves its own blend state behind, and the tip is drawn after the
    # text - without setting it back, every translucent fill here lands opaque
    # and the whole drawing came out as one white square.
    gpu.state.blend_set("ALPHA")
    shader.bind()

    lines = _wrap(body, TIP_W - 24.0, 11, scale)
    # a hint about a setting says what it says in words; the moving drawing
    # belongs to the buttons, where there is a thing happening to show
    art_h = 96.0 if art else 0.0
    h = 16.0 + 16.0 + 6.0 + len(lines) * 15.0 + (10.0 + art_h + 14.0
                                                 if art else 14.0)
    x = card[0] - TIP_W - 10.0
    y = min(max(wy + wh - h, 20.0), card[1] + card[3] - h)

    _fill(shader, (x - 1.0) * scale, (y - 1.0) * scale, (TIP_W + 2.0) * scale,
          (h + 2.0) * scale, 13.0 * scale, EDGE)
    _fill(shader, x * scale, y * scale, TIP_W * scale, h * scale,
          12.0 * scale, CARD)

    if art:
        _TIP_ART.get(art, _tip_grid)(shader, x + 12.0, y + 12.0,
                                     TIP_W - 24.0, art_h,
                                     (time.time() - since) * 0.55, scale)

    ty = y + h - 24.0
    _text(title, (x + 12.0) * scale, ty * scale, 12, TEXT, scale, bold=True)
    ty -= 17.0
    for line in lines:
        _text(line, (x + 12.0) * scale, ty * scale, 11, TEXT_DIM, scale)
        ty -= 15.0


def _settle():
    """Pay back a repaint the throttle skipped.

    Blender stops sending mouse moves the moment the pointer stops, so the
    last move of a sweep is often the one the throttle drops - and the
    highlight would then sit on the row before the one under the pointer until
    something else happened to force a paint.
    """
    if not _state["dirty"] or _state["quit"]:
        _state["dirty"] = False
        return None
    _state["dirty"] = False
    _state["painted"] = time.time()
    _state["asked"] = _state["painted"]
    for win in bpy.context.window_manager.windows:
        for a in win.screen.areas:
            if a.type == "VIEW_3D":
                a.tag_redraw()
    return None


def _tick():
    """Redraw while a tip is open, so its drawing can move.

    The animation is never worth more than about a quarter of the machine. On
    a light scene that leaves it at its full twenty frames a second; on a
    heavy one it slows itself down instead of fighting the viewport for every
    frame.
    """
    if _state["quit"] or _state["tip"] is None:
        _state["ticking"] = False
        return None
    if _state["asked"]:
        return 0.02              # the last one has not come back yet
    _state["asked"] = time.time()
    area = getattr(bpy.context, "area", None)
    if area is not None and area.type == "VIEW_3D":
        area.tag_redraw()
    else:
        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()
    return min(0.5, max(0.05, _state["cost"] * 4.0))


# What to say when an action will not run. "That one cannot run here" is true
# and useless: it says nothing about what is missing or what to do about it,
# and Milad hit exactly that pressing Confirm with nothing to confirm.

WHY = {
    "img3d.generate": "choose an image first",
    "img3d.install_engine": "choose the Install Folder on the Downloads tab",
}


# -- what the card holds ----------------------------------------------------

def _settings(context):
    return getattr(context.scene, "img3d", None)


def _job_state():
    """What the unwrap is doing, if it is running."""
    try:
        import sys
        mod = sys.modules.get(__package__)
        return getattr(mod, "CARD_JOB", None)
    except Exception:
        return None


def _bake_job():
    """What the bake is doing, if it is running."""
    try:
        import sys
        mod = sys.modules.get(__package__)
        return getattr(mod, "BAKE_JOB", None)
    except Exception:
        return None


def _pick_state():
    try:
        import sys
        return getattr(sys.modules.get(__package__), "PICK", None)
    except Exception:
        return None


def _has_other_uv(obj, st):
    """True when the model carries a uv map that is not ours."""
    try:
        name = st.uv_name
        return any(l.name != name for l in obj.data.uv_layers)
    except Exception:
        return False


def _part_preview(obj):
    try:
        import sys
        return sys.modules[__package__].part_preview_active(obj)
    except Exception:
        return False


def _stretch_preview(obj):
    try:
        import sys
        return sys.modules[__package__].stretch_preview_active(obj)
    except Exception:
        return False


def _hand_parts(obj):
    try:
        import sys
        return sys.modules[__package__].manual_parts_ready(obj)
    except Exception:
        return False


def _engine_state(engine="TRELLIS"):
    """(installed, job) for the chosen engine, as the add-on reports it."""
    try:
        import sys
        pkg = sys.modules.get(__package__)
        return bool(pkg.engine_ready(engine)), getattr(pkg, "SETUP_JOB", None)
    except Exception:
        return False, None


def _job_state():
    try:
        import sys
        return getattr(sys.modules.get(__package__), "CARD_JOB", None)
    except Exception:
        return None


_THUMBS = {}


def _thumb(path):
    """The picked image as a small texture, made once per file."""
    if not path:
        return None
    try:
        stamp = os.path.getmtime(path)
    except Exception:
        return None
    key = (path, stamp)
    if key in _THUMBS:
        return _THUMBS[key]
    if len(_THUMBS) > 6:
        _THUMBS.clear()
    tex = None
    img = None
    try:
        img = bpy.data.images.load(path, check_existing=False)
        img.colorspace_settings.name = "Non-Color"
        w, h = img.size
        if w <= 0 or h <= 0:
            raise ValueError("empty image")
        k = 256.0 / max(w, h)
        if k < 1.0:
            img.scale(max(1, int(w * k)), max(1, int(h * k)))
        w, h = img.size
        buf = gpu.types.Buffer("FLOAT", w * h * 4, img.pixels[:])
        tex = (gpu.types.GPUTexture((w, h), format="RGBA16F", data=buf), w, h)
    except Exception:
        tex = None
    finally:
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass
    _THUMBS[key] = tex
    return tex


def _build(context, region):
    """Lay the card out and return its widgets, in unscaled pixels."""
    st = _settings(context)
    if st is None:
        return [], (0.0, 0.0, 0.0, 0.0)

    scale = context.preferences.system.ui_scale or 1.0
    heads = 0.0
    area = getattr(context, "area", None)
    if area is not None:
        for r in area.regions:
            if r.type in {"HEADER", "TOOL_HEADER"}:
                heads += r.height
    x = region.width / scale - TAB_STRIP - CARD_W
    top = (region.height - heads) / scale - 6.0

    w = []
    inner = CARD_W - PAD * 2.0 - 8.0
    left = x + PAD + 4.0
    y = [top - PAD + _state["scroll"]]

    def put(kind, h, gap=GAP, **kw):
        kw["kind"] = kind
        kw["rect"] = (left, y[0] - h, inner, h)
        w.append(kw)
        y[0] -= h + gap
        return kw

    def field(text, owner, prop, lo, hi, decimals=0, suffix="", tip=None,
              save=False):
        if not hasattr(owner, prop):
            return
        val = getattr(owner, prop)
        shown = (f"{int(val):,}" if not decimals else f"{val:.{decimals}f}")
        put("field", 26.0, gap=6.0, text=text, shown=shown + suffix, tip=tip,
            save=save, act=("type", (owner, prop, lo, hi, decimals)))
        put("bar", 14.0, value=val, lo=lo, hi=hi, decimals=decimals, tip=tip,
            save=save, act=("slide", (owner, prop)))

    def enum_row(owner, prop, items, h=36.0, tip=None, tips=None):
        n = len(items)
        cw = (inner - 6.0 * (n - 1)) / n
        cur = getattr(owner, prop)
        size = 12 if n <= 2 else (11 if n == 3 else 10)
        for k, (name, code) in enumerate(items):
            w.append({"kind": "chip", "tsize": size,
                      "rect": (left + k * (cw + 6.0), y[0] - h, cw, h),
                      "text": name, "on": cur == code,
                      "tip": (tips[k] if tips else tip),
                      "act": ("enum", (owner, prop, code))})
        y[0] -= h + GAP

    def note_line(text, color=None):
        w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                  "text": text, "size": 11,
                  "color": TEXT_DIM if color is None else color})
        y[0] -= 16.0

    def links(items):
        n = len(items)
        cw = inner / n
        for k, (name, url) in enumerate(items):
            w.append({"kind": "linktext",
                      "rect": (left + k * cw, y[0] - 22.0, cw, 22.0),
                      "text": name, "act": ("url", url)})
        y[0] -= 22.0 + GAP

    def rule():
        put("rule", 1.0, gap=GAP + 2.0)

    def powered_hi3d():
        # the Hi3D licence asks every product using it to show this credit
        w.append({"kind": "linktext",
                  "rect": (left, y[0] - 18.0, inner, 18.0),
                  "text": "Powered by Hi3D 1.0",
                  "act": ("url", "https://hi3d.ai")})
        y[0] -= 18.0 + 4.0

    if not _state.get("started"):
        _state["started"] = True
        _state["open"].update({"image", "settings"})

    def panel_begin(key, title, icon):
        ph = {"kind": "panel", "rect": (left - 13.0, 0.0, inner + 26.0, 0.0),
              "top": y[0]}
        w.append(ph)
        y[0] -= 11.0
        open_ = key in _state["open"]
        put("head", 24.0, gap=12.0 if open_ else 3.0, text=title, icon=icon,
            on=open_, act=("fold", key))
        return open_, ph

    def panel_end(ph):
        bottom = y[0] + GAP - 11.0
        ph["rect"] = (left - 13.0, bottom, inner + 26.0, ph["top"] - bottom)
        y[0] = bottom - GAP - 3.0

    try:
        import sys as _s
        _m = _s.modules[__package__]
    except Exception:
        _m = None

    # header ---------------------------------------------------------------
    w.append({"kind": "logo", "rect": (left, y[0] - 30.0, 30.0, 30.0)})
    w.append({"kind": "title", "rect": (left + 40.0, y[0] - 15.0, 0, 0),
              "text": "Image to 3D 2", "room": inner - 40.0})
    w.append({"kind": "sub", "rect": (left + 40.0, y[0] - 29.0, 0, 0),
              "text": "Textured 3D model from one picture",
              "room": inner - 40.0})
    y[0] -= 30.0 + GAP + 4.0

    # tabs: making a model, and the downloads it needs ----------------------
    tab_ = _state.get("tab", "gen")
    tw_ = (inner - 6.0) / 2.0
    for k_, (name_, code_) in enumerate((("Generate", "gen"),
                                         ("Downloads", "dl"))):
        w.append({"kind": "chip", "tsize": 12,
                  "rect": (left + k_ * (tw_ + 6.0), y[0] - 32.0, tw_, 32.0),
                  "text": name_, "on": tab_ == code_, "tip": None,
                  "act": ("tab", code_)})
    y[0] -= 32.0 + GAP + 2.0

    engine = getattr(st, "engine", "TRELLIS")
    ready, setup = _engine_state(engine)
    job = _job_state()
    hyl = engine in ("HUNYUAN", "HI3D")
    label = {"HUNYUAN": "Hunyuan 3D", "PIXAL3D": "Pixal3D",
             "HI3D": "Hi3D"}.get(engine,
                                                               "TRELLIS")

    def install_folder(compact=False):
        """Where the engine and the models go, and the way to change it."""
        root_ = _m.engine_dir() if _m is not None else ""
        if not compact:
            w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                      "text": "Install Folder", "size": 12, "color": TEXT})
            y[0] -= 18.0
        put("path", 28.0, gap=6.0, text=root_ or "not chosen yet",
            tip="folder", act=("op", "img3d.pick_engine_dir"))
        free_ = _m.drive_free_gb(root_) if _m is not None else None
        if free_ is not None:
            drv_ = os.path.splitdrive(root_)[0] or "this drive"
            note_line("%d GB free on %s" % (int(free_), drv_))
        cw_ = (inner - 6.0) / 2.0
        for k_, (name_, act_) in enumerate((
                ("Change Folder", ("op", "img3d.pick_engine_dir")),
                ("Open Folder", ("call", ("img3d.dl_folder", {}))))):
            w.append({"kind": "chip", "tsize": 11,
                      "rect": (left + k_ * (cw_ + 6.0), y[0] - 26.0, cw_,
                               26.0),
                      "text": name_, "on": False, "tip": "folder",
                      "act": act_})
        y[0] -= 26.0 + GAP

    def view_grid():
        """Hunyuan reads up to four pictures of the object. Each slot shows a
        faded example of what belongs in it until a picture is picked."""
        exd = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "media", "views")
        fp = bpy.path.abspath(st.image_path) if st.image_path else ""
        fp = fp if fp and os.path.exists(fp) else ""
        sides_ = []
        for n in range(3):
            v = getattr(st, "view_%d" % (n + 1), "")
            vp = bpy.path.abspath(v) if v else ""
            sides_.append(vp if vp and os.path.exists(vp) else "")
        fh = 150.0
        w.append({"kind": "vslot", "rect": (left, y[0] - fh, inner, fh),
                  "path": fp, "ex": os.path.join(exd, "front.png"),
                  "text": "Front", "big": True, "tip": "image",
                  "act": ("op", "img3d.pick_image")})
        if fp:
            w.append({"kind": "btn", "text": "\u00d7", "over": True, "tsize": 26,
                      "tip": "image",
                      "rect": (left + inner - 30.0, y[0] - 30.0, 24.0, 24.0),
                      "act": ("op", "img3d.clear_image")})
        y[0] -= fh + 6.0
        cw = (inner - 12.0) / 3.0
        th = 84.0
        row = (("Left", 2, "left"), ("Right", 3, "right"), ("Back", 1, "back"))
        for i, (name, n, key) in enumerate(row):
            x0 = left + i * (cw + 6.0)
            w.append({"kind": "vslot", "rect": (x0, y[0] - th, cw, th),
                      "path": sides_[n - 1],
                      "ex": os.path.join(exd, key + ".png"),
                      "text": name, "tip": "view%d" % n,
                      "act": ("op", "img3d.pick_view_%d" % n)})
        y[0] -= th + 4.0
        if any(sides_):
            for i, (name, n, key) in enumerate(row):
                if sides_[n - 1]:
                    x0 = left + i * (cw + 6.0)
                    w.append({"kind": "btn", "text": "Remove",
                              "tip": "view%d" % n,
                              "rect": (x0, y[0] - 22.0, cw, 22.0),
                              "act": ("op", "img3d.clear_view_%d" % n)})
            y[0] -= 26.0
        put("info", 18.0, text="Front is needed, the others are optional",
            tip="image")

    if tab_ == "gen":
        # image ---------------------------------------------------------------
        open_, ph = panel_begin("image", "IMAGE", "eye")
        if open_ and hyl:
            view_grid()
        elif open_:
            path = bpy.path.abspath(st.image_path) if st.image_path else ""
            if path and os.path.exists(path):
                th_ = put("thumb", 150.0, path=path, tip="image",
                          act=("op", "img3d.pick_image"))
                tx_, ty_, tw_, thh_ = th_["rect"]
                # a small x in the corner takes the picture away
                w.append({"kind": "btn", "text": "\u00d7", "over": True, "tsize": 26,
                          "tip": "image",
                          "rect": (tx_ + tw_ - 30.0, ty_ + thh_ - 30.0,
                                   24.0, 24.0),
                          "act": ("op", "img3d.clear_image")})
                note_line(os.path.basename(path))
                put("btn", 30.0, text="Change Image", icon="cursor",
                    tip="image", act=("op", "img3d.pick_image"))
            else:
                put("btn", ROW, text="Choose Image", icon="cursor",
                    tip="image", act=("op", "img3d.pick_image"), lead=True)
                put("info", 18.0, text="one object, plain or no background",
                    tip="image")
        panel_end(ph)

        # system meters ---------------------------------------------------------
        _ms = sys.modules.get(__package__)
        stats = getattr(_ms, "_STATS", None) if _ms is not None else None
        if stats is not None:
            rows_ = []
            v_ = stats.get("vram")
            r_ = stats.get("ram")
            c_ = stats.get("cpu")
            if v_:
                rows_.append(("VRAM", v_[0] / v_[1], "%.1f / %.0f GB" % v_))
            if r_:
                rows_.append(("RAM", r_[0] / r_[1], "%.1f / %.0f GB" % r_))
            g_ = stats.get("gpu")
            if g_ is not None:
                rows_.append(("GPU", g_, "%d%%" % round(g_ * 100.0)))
            if c_ is not None:
                rows_.append(("CPU", c_, "%d%%" % round(c_ * 100.0)))
            for lab_, fr_, txt_ in rows_:
                put("meter", 16.0, gap=5.0, text=lab_, frac=fr_, value=txt_,
                    tip="meters")
            busy_ = bool((job is not None and job.get("busy")) or
                         (setup is not None and setup.get("busy")))
            if rows_ and not busy_:
                y[0] -= 2.0
                put("btn", 26.0, text="Clear Cache", tip="clearcache",
                    act=("op", "img3d.clear_cache"))
            if not busy_:
                y[0] -= 2.0
                put("btn", 26.0, text="Save Report", tip="report",
                    act=("op", "img3d.save_report"))
            y[0] -= 4.0

        # run -----------------------------------------------------------------
        if job is not None and job.get("busy"):
            put("running", 42.0,
                text="%s  %d%%" % (job.get("stage", "Working"),
                                   int(job.get("frac", 0.0) * 100.0)),
                frac=job.get("frac", 0.0))
            put("runline", 14.0, text="  -  ".join(
                t for t in (job.get("clock", ""), job.get("line", "")) if t))
            put("btn", 28.0, text="Cancel", tip="cancel",
                act=("op", "img3d.cancel"))
        elif setup is not None and setup.get("busy"):
            put("running", 42.0,
                text="%s  %d%%" % (setup.get("stage", "Installing"),
                                   int(setup.get("frac", 0.0) * 100.0)),
                frac=setup.get("frac", 0.0))
            put("runline", 14.0, text="  -  ".join(
                t for t in (setup.get("clock", ""), setup.get("line", "")) if t))
            put("btn", 28.0, text="Cancel", tip="cancel",
                act=("op", "img3d.cancel"))
        elif not ready and _m is not None and _m.software_ready(engine) and \
                _m.models_missing(engine):
            miss = _m.models_missing(engine)
            names = " + ".join(_m.dl.group(g)["name"] for g in miss)
            gbs = sum(_m.dl.size_gb(g) for g in miss)
            note_line("%s is not downloaded yet (%.1f GB)" % (names, gbs),
                      ACCENT)
            y[0] -= 4.0
            put("run", 44.0, text="Go to Downloads", icon="package",
                tip=None, act=("call", ("img3d.go_downloads",
                                               {"group": miss[-1]})))
        elif not ready:
            put("run", 44.0, text="Install " + label, icon="package",
                tip="install", act=("op", "img3d.install_engine"))
            put("info", 18.0, text="one time download, about %s" %
                ("35 GB" if engine == "HUNYUAN" else "25 GB"), tip="install")
            install_folder(compact=True)
        else:
            put("run", 44.0, text="Generate 3D", icon="cube", tip="generate",
                act=("op", "img3d.generate"))
            # the node Canvas: works on its own once its picture models
            # are installed; until then the card says what is missing
            _m_ = sys.modules.get(__package__)
            put("btn", 30.0, text="Open Canvas", icon="cube", tip=None,
                act=("op", "img3d_canvas.open"))
            if _m_ is not None and not _m_.software_ready("CANVAS"):
                note_line("The Canvas picture tools are not installed yet")
                put("btn", 26.0, text="Install Canvas Tools",
                    icon="package", tip=None,
                    act=("call", ("img3d.install_engine",
                                  {"target": "CANVAS"})))
            elif _m_ is not None and not _m_.canvas_ready():
                note_line("The Canvas models are not downloaded yet")
                put("btn", 26.0, text="Go to Downloads", tip=None,
                    act=("call", ("img3d.go_downloads",
                                  {"group": "CV_QWEN_EDIT"})))
            if _m_ is not None and _m_.engine_ready("HI3D"):
                put("switch", 26.0, gap=4.0, text="Enhance 3D Model",
                    on=st.auto_detail == "ON", tip="adetail",
                    act=("enum", (st, "auto_detail",
                                  "OFF" if st.auto_detail == "ON" else "ON")))
                if st.auto_detail == "ON":
                    powered_hi3d()
            if st.use_texture == "ON" and _m_ is not None and \
                    _m_.klein_ready() and _m_.engine_ready("HUNYUAN"):
                put("switch", 26.0, gap=4.0, text="Enhance Texture",
                    on=st.auto_enhance == "ON", tip="aenhance",
                    act=("enum", (st, "auto_enhance",
                                  "OFF" if st.auto_enhance == "ON" else "ON")))
            y[0] -= GAP - 4.0

    # downloads --------------------------------------------------------------
    if tab_ == "dl":
        _state["open"].add("folder")
        open_, ph = panel_begin("folder", "INSTALL FOLDER", "gear")
        install_folder(compact=True)
        panel_end(ph)
    if tab_ == "dl" and _m is not None and _m.engine_dir():
        _state["open"].add("downloads")
        _dl = _m.dl
        root_ = _m.engine_dir()
        foc = _state.get("dl_focus")
        foc_id = foc[0] if foc and time.time() - foc[1] < 12.0 else None
        open_, ph = panel_begin("downloads", "DOWNLOADS", "package")
        if open_:
            _gp = _prefs()
            if _gp is not None and hasattr(_gp, "gentle_download"):
                put("switch", 26.0, gap=4.0, text="Gentle Download",
                    on=bool(_gp.gentle_download),
                    act=("prop", (_gp, "gentle_download")))
                for _gl in _wrap("For older computers: one file at a "
                                 "time, so the computer stays smooth"
                                 if _gp.gentle_download else
                                 "Turn on if the computer freezes while "
                                 "downloading", inner, 11, scale):
                    note_line(_gl)
                y[0] -= 6.0
            run_ = _dl.DL
            cat_ = None
            for g in _dl.groups():
                gid = g["id"]
                stt, have, total, _miss = _dl.status(root_, gid)
                if g.get("cat") and g["cat"] != cat_:
                    cat_ = g["cat"]
                    y[0] -= 6.0
                    w.append({"kind": "label",
                              "rect": (left, y[0] - 12.0, 0, 0),
                              "text": cat_.upper(), "size": 13,
                              "color": ACCENT})
                    y[0] -= 20.0
                    for line in _wrap("Each Canvas node needs only its "
                                      "own models: download the ones "
                                      "for the nodes you use.", inner,
                                      11, scale):
                        note_line(line)
                    y[0] -= 6.0
                mine = _dl.busy() and gid in run_.get("groups", [])
                hot = gid == foc_id
                w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                          "text": "%s   %.1f GB" % (g["name"], total /
                                                  float(1 << 30)),
                          "size": 12, "color": ACCENT if hot else TEXT})
                y[0] -= 18.0
                if g.get("use"):
                    note_line("For " + g["use"])
                if mine:
                    spd_, eta_ = _dl.speed_text()
                    line = (run_.get("text") or "Downloading") + (
                        "  -  " + eta_ if not run_.get("verify") else "")
                elif stt == "ready":
                    line = "Installed"
                elif stt == "pending":
                    _eg = "HUNYUAN" if gid in ("HUNYUAN", "HUNYUAN_MV",
                                               "HI3D") else "TRELLIS"
                    _mm = sys.modules.get(__package__)
                    line = ("Downloaded - press Finish Setup"
                            if _mm is not None and _mm.software_ready(_eg)
                            else "Downloaded - press Install Engine to finish")
                elif stt == "partial":
                    line = "Partly downloaded (%d%%)" % int(
                        100.0 * have / max(1, total))
                else:
                    line = "Not downloaded"
                if g.get("needs") and stt != "ready" and not \
                        _dl.have(root_, g["needs"]):
                    line += " - also needs " + _dl.group(g["needs"])["name"]
                note_line(line, ACCENT if mine else None)
                if mine:
                    put("running", 30.0, text="%d%%%s" % (
                        int(run_.get("frac", 0.0) * 100.0),
                        ("     " + spd_) if spd_ else ""),
                        frac=run_.get("frac", 0.0))
                cw = (inner - 12.0) / 3.0
                page_ = ("By Hand", ("call", ("img3d.dl_page",
                                                    {"group": gid})))
                check_ = ("Check Files", ("call", ("img3d.dl_get", {
                    "group": gid, "verify": True})))
                if mine:
                    acts = (("Stop", ("call", ("img3d.dl_cancel", {}))),
                            ("Check Files", None), page_)
                elif stt == "ready":
                    # the Download button stays visible, greyed out
                    acts = (("Installed", None), check_, page_)
                elif stt == "pending":
                    # the files are here; the engine they run in is not
                    eng_ = "HUNYUAN" if gid in ("HUNYUAN", "HUNYUAN_MV",
                                                "HI3D") else "TRELLIS"
                    _mm = sys.modules.get(__package__)
                    if _mm is not None and _mm.software_ready(eng_):
                        # the engine is there: only the last step is left
                        acts = (("Finish Setup", ("call", (
                            "img3d.dl_get", {"group": gid}))),
                                check_, page_)
                    else:
                        acts = (("Install Engine", ("call", (
                            "img3d.install_engine", {"target": eng_}))),
                                check_, page_)
                else:
                    acts = (("Download", ("call", ("img3d.dl_get",
                                                   {"group": gid}))),
                            check_ if have > 0 else ("Check Files", None),
                            page_)
                for k_, (name_, act_) in enumerate(acts):
                    chip_ = {"kind": "chip", "tsize": 10,
                             "rect": (left + k_ * (cw + 6.0), y[0] - 28.0,
                                      cw, 28.0),
                             "text": name_,
                             "on": (k_ == 0 and act_ is not None and
                                    stt != "ready" and not mine),
                             "tip": None}
                    if act_ is not None:
                        chip_["act"] = act_
                    else:
                        chip_["dim"] = True
                    w.append(chip_)
                y[0] -= 28.0 + GAP + 6.0
            if run_.get("error") and not _dl.busy():
                for line in _wrap(run_["error"], inner, 11, scale):
                    note_line(line, ACCENT)
            for line in _wrap("Every model comes from its maker's official "
                              "page and is checked against the official "
                              "file. By Hand shows each part's page and "
                              "the folder its files go into.", inner, 11,
                              scale):
                note_line(line)
        panel_end(ph)

    if tab_ == "gen":
        # enhance a model already made -------------------------------------------
        _m_ = sys.modules.get(__package__)
        tex_t = (_m_.enhance_target(bpy.context)
                 if _m_ is not None else None)
        det_t = (_m_.detail_target(bpy.context)
                 if _m_ is not None and _m_.engine_ready("HI3D") else None)
        det_miss = []
        ext_miss = bool(_m_ is not None and _m_.extras_missing())
        if _m_ is not None and det_t is None and _m_.software_ready("HI3D"):
            try:
                if _m_.detail_target(bpy.context) is not None:
                    det_miss = _m_.models_missing("HI3D")
            except Exception:
                det_miss = []
        act_ = bpy.context.view_layer.objects.active
        und_t = act_ is not None and bool(act_.get("img3d_prev_mat"))
        und_d = act_ is not None and bool(act_.get("img3d_prev")) and \
            bpy.data.objects.get(act_.get("img3d_prev", "")) is not None
        if ready and (tex_t is not None or det_t is not None or und_t or
                      und_d or det_miss or (ext_miss and (
                          tex_t is not None or (_m_ is not None and
                                                _m_.detail_target(bpy.context))))) and \
                not (job is not None and job.get("busy")):
            if not _state.get("enhance_opened"):
                _state["open"].add("enhance")
                _state["enhance_opened"] = True
            open_, ph = panel_begin("enhance", "ENHANCE", "paint")
            if open_:
                if ext_miss:
                    note_line("The Enhance tools are not installed yet")
                    put("btn", 30.0, text="Install Enhance Tools", icon="package",
                        tip=None, act=("call", ("img3d.install_engine",
                                                {"target": "EXTRAS"})))
                    if not _m_.software_ready("KLEIN"):
                        tex_t = None
                if tex_t is not None and not _m_.software_ready("HUNYUAN"):
                    # Enhance Texture and Enhance Detail run next to Hunyuan
                    note_line("The Enhance tools run with the Hunyuan engine:")
                    note_line("install it first (its models can stay as they are)")
                    put("btn", 30.0, text="Install Hunyuan Engine",
                        icon="package", tip=None,
                        act=("call", ("img3d.install_engine",
                                      {"target": "HUNYUAN"})))
                    tex_t = None
                if tex_t is not None and not _m_.klein_ready():
                    note_line("Enhance Texture is not downloaded yet")
                    put("btn", 26.0, text="Go to Downloads", tip=None,
                        act=("call", ("img3d.go_downloads", {"group": "KLEIN"})))
                    tex_t = None
                if tex_t is not None:
                    w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                              "text": "Texture", "size": 12, "color": TEXT})
                    y[0] -= 20.0
                    enum_row(st, "enhance_mode", (("Fast", "FAST"),
                                                  ("High", "HIGH")),
                             tip="texture")
                    put("btn", 30.0, text="Enhance Texture", icon="paint",
                        tip="texture", act=("op", "img3d.enhance"))
                if und_t:
                    put("btn", 26.0, text="Undo Enhance Texture", tip="undotex",
                        act=("op", "img3d.undo_texture"))
                if tex_t is not None and det_t is not None:
                    rule()
                if det_miss:
                    note_line("Enhance Detail is not downloaded yet")
                    put("btn", 26.0, text="Go to Downloads", tip=None,
                        act=("call", ("img3d.go_downloads",
                                      {"group": det_miss[-1]})))
                if det_t is not None:
                    w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                              "text": "Detail", "size": 12, "color": TEXT})
                    y[0] -= 20.0
                    enum_row(st, "detail_tex", (("Repaint Texture", "ON"),
                                                ("Shape Only", "OFF")),
                             tip="detail")
                    put("btn", 30.0, text="Enhance Detail", icon="cube",
                        tip="detail", act=("op", "img3d.enhance_detail"))
                    powered_hi3d()
                    if det_t.get("img3d_gray"):
                        put("btn", 30.0, text="Paint Texture", icon="paint",
                            tip="detail", act=("op", "img3d.paint_detail"))
                if und_d:
                    put("btn", 26.0, text="Back to Previous Model",
                        tip="undodet", act=("op", "img3d.undo_detail"))
            panel_end(ph)

        # settings --------------------------------------------------------------
        open_, ph = panel_begin("settings", "SETTINGS", "gear")
        if open_:
            w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                      "text": "Engine", "size": 12, "color": TEXT})
            y[0] -= 20.0
            enum_row(st, "engine", (("TRELLIS", "TRELLIS"),
                                    ("Hunyuan", "HUNYUAN"),
                                    ("Pixal3D", "PIXAL3D")),
                     tips=("etrellis", "ehunyuan", "epixal"))
            w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                      "text": "Quality", "size": 12, "color": TEXT})
            y[0] -= 20.0
            enum_row(st, "quality", (("Fast", "FAST"), ("High", "HIGH"),
                                     ("Ultra", "ULTRA")),
                     tips=("qfast", "qhigh", "qultra"))
            if hyl:
                field("Faces", st, "faces_hy", 5000, 2000000, tip="faces")
            else:
                field("Faces", st, "faces", 5000, 2000000, tip="faces")
            w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                      "text": "Texture", "size": 12, "color": TEXT})
            y[0] -= 20.0
            enum_row(st, "use_texture", (("On", "ON"), ("Off", "OFF")),
                     tip="usetex")
            if st.use_texture == "ON":
                enum_row(st, "texture", (("1K", "1024"), ("2K", "2048"),
                                         ("4K", "4096")), tip="texture")
            field("Seed", st, "seed", 0, 9999, tip="seed")
        panel_end(ph)

        prefs = _prefs()
        if prefs is not None and hasattr(prefs, "accent_hue"):
            open_, ph = panel_begin("ui", "CUSTOMIZE", "paint")
            if open_:
                hue = float(prefs.accent_hue)
                sat = float(prefs.accent_sat)
                steps = 34
                put("ramp", 12.0, gap=6.0, value=hue, lo=0.0, hi=1.0, decimals=3,
                    tip="accent",
                    ramp=[accent_rgb(i / (steps - 1.0), max(sat, 0.25)) + (1.0,)
                          for i in range(steps)],
                    act=("slide", (prefs, "accent_hue")))
                put("ramp", 12.0, value=sat, lo=0.0, hi=1.0, decimals=3,
                    tip="accent",
                    ramp=[accent_rgb(hue, i / (steps - 1.0)) + (1.0,)
                          for i in range(steps)],
                    act=("slide", (prefs, "accent_sat")))
                field("Corner Rounding", prefs, "corner", 0.0, 16.0, 1,
                      tip="corner", save=True)
            panel_end(ph)

    # update ----------------------------------------------------------------
    _u = getattr(_m, "updates", None) if _m is not None else None
    if _u is not None:
        try:
            us_ = _u.status()
        except Exception:
            us_ = None
        if us_ is not None:
            title_ = "UPDATE AVAILABLE" if us_["newer"] else "UPDATE"
            open_, ph = panel_begin("update", title_, "package")
            if open_:
                note_line("Installed  v" + us_["installed"])
                if us_["mode"] == "legacy":
                    note_line("For one-click updates inside Blender,")
                    note_line("connect Superhive to Blender once.")
                    put("btn", 26.0, text="Connect Superhive",
                        act=("op", "img3d.update_connect"))
                else:
                    note_line(("Latest  v" + us_["latest"]) if us_["latest"]
                              else "Latest  not checked yet",
                              ACCENT if us_["newer"] else None)
                    if us_["newer"]:
                        put("run", 40.0, text="Download Update",
                            icon="package",
                            act=("op", "img3d.update_install"))
                        put("btn", 26.0, text="What's New",
                            act=("op", "img3d.update_notes"))
                    put("btn", 26.0, text="Check for Updates",
                        act=("op", "img3d.update_check"))
                    _pp = _prefs()
                    if _pp is not None and hasattr(_pp, "auto_update"):
                        put("switch", 26.0, gap=4.0, text="Auto Update",
                            on=bool(_pp.auto_update),
                            act=("prop", (_pp, "auto_update")))
                if us_.get("note"):
                    note_line(us_["note"])
            panel_end(ph)

    if st.status:
        for line in _wrap(st.status, inner, 11, scale):
            note_line(line)

    rule()
    stamp = ("  ·  " + _m.VERSION) if _m is not None else ""
    w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
              "text": "Milad Kambari - 3DRedbox Studio" + stamp, "size": 11,
              "color": TEXT_DIM})
    y[0] -= 16.0
    if _m is not None:
        links((("ArtStation", getattr(_m, "ARTSTATION_URL", "")),
               ("Superhive", getattr(_m, "SUPERHIVE_URL", "")),
               ("Discord", getattr(_m, "DISCORD_URL", ""))))

    note = _state.get("note")
    if note is not None and time.time() - note[1] < 12.0:
        for line in _wrap(note[0], inner, 11, scale):
            w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                      "text": line, "size": 11, "color": ACCENT})
            y[0] -= 15.0
        y[0] -= 3.0

    bottom = y[0] - PAD
    full = (top + PAD - bottom) + _state["scroll"]
    room = top + PAD - 6.0
    _state["over"] = max(0.0, full - room)
    if _state["scroll"] > _state["over"] + 0.5 and not _state.get("relayout"):
        _state["scroll"] = _state["over"]
        _state["relayout"] = True
        try:
            return _build(context, region)
        finally:
            _state["relayout"] = False
    shown = min(full, room)
    card = (x, top + PAD - shown, CARD_W, shown)
    return w, card


def _typing_here(it):
    """Is this the number box that is open for typing?"""
    t = _state["typing"]
    if t is None or it.get("kind") != "field":
        return False
    act = it.get("act")
    if not act or act[0] != "type":
        return False
    if act[1][1] != t["prop"]:
        return False
    # Blender hands back a fresh Python wrapper for the same settings block on
    # every access, so `is` says no even when it is the same block. Comparing
    # what they point at is the only test that holds - without it the box drew
    # as if nothing was happening and typing looked dead.
    try:
        return act[1][0].as_pointer() == t["owner"].as_pointer()
    except Exception:
        return False


def _stop_typing(context, apply):
    """Close the box, keeping what was typed or throwing it away."""
    t = _state["typing"]
    _state["typing"] = None
    if t is None:
        return
    if apply and t["text"].strip() not in ("", "-", ".", "-."):
        try:
            val = float(t["text"])
        except ValueError:
            val = None
        if val is not None:
            # The slider is deliberately narrower than the setting allows -
            # Faces runs 20 to 20,000 on the card and 8 to 200,000 in the
            # property - so a typed number is held to what the setting really
            # permits, not to what is comfortable to drag.
            lo, hi = t["lo"], t["hi"]
            try:
                rna = t["owner"].bl_rna.properties[t["prop"]]
                lo, hi = rna.hard_min, rna.hard_max
            except Exception:
                pass
            val = min(max(val, lo), hi)
            try:
                setattr(t["owner"], t["prop"],
                        int(round(val)) if not t["decimals"] else val)
            except Exception:
                pass
            if t.get("save"):
                try:
                    bpy.ops.wm.save_userpref()
                except Exception:
                    pass
    _redraw(context, getattr(context, "area", None))



def _glyph(shader, name, x, y, size, color, scale):
    """The small line drawings on the buttons and the section headers, drawn
    with smooth lines so they stay crisp at any interface scale."""
    s = size

    def P(px, py):
        return (x + px * s, y + py * s)

    def path(pts, t=1.5, closed=False):
        _aaline(shader, [P(*p) for p in pts], t, color, scale, closed=closed)

    def circle(cx, cy, r, t=1.4, steps=20):
        pts = [(cx + math.cos(2 * math.pi * i / steps) * r,
                cy + math.sin(2 * math.pi * i / steps) * r) for i in range(steps)]
        path(pts, t, closed=True)

    def dot(cx, cy, r=0.09):
        _fill(shader, (x + (cx - r) * s) * scale, (y + (cy - r) * s) * scale,
              2 * r * s * scale, 2 * r * s * scale, r * s * scale, color)

    if name == "cube":
        path([(0.5, 0.97), (0.94, 0.73), (0.94, 0.27), (0.5, 0.03),
              (0.06, 0.27), (0.06, 0.73)], 1.5, closed=True)
        path([(0.06, 0.73), (0.5, 0.5), (0.94, 0.73)], 1.4)
        path([(0.5, 0.5), (0.5, 0.03)], 1.4)
    elif name == "grid":
        # a flat projection grid: Blender's own way of doing it
        path([(0.06, 0.06), (0.94, 0.06), (0.94, 0.94), (0.06, 0.94)], 1.5,
             closed=True)
        for f in (0.36, 0.65):
            path([(0.06, f), (0.94, f)], 1.3)
            path([(f, 0.06), (f, 0.94)], 1.3)
    elif name == "paint":
        # a painter's palette: the card's own colour is chosen here
        circle(0.5, 0.52, 0.44, 1.5)
        for cx, cy in ((0.30, 0.72), (0.56, 0.80), (0.74, 0.60)):
            dot(cx, cy, 0.085)
        path([(0.30, 0.16), (0.44, 0.12), (0.56, 0.20), (0.56, 0.34),
              (0.42, 0.38), (0.30, 0.30)], 1.4, closed=True)
    elif name == "waves":
        for k in range(3):
            yy = 0.24 + k * 0.26
            path([(0.06, yy), (0.28, yy + 0.09), (0.5, yy - 0.09),
                  (0.72, yy + 0.09), (0.94, yy)], 1.5)
    elif name == "eye":
        path([(0.04, 0.5), (0.22, 0.76), (0.5, 0.84), (0.78, 0.76), (0.96, 0.5)], 1.5)
        path([(0.04, 0.5), (0.22, 0.24), (0.5, 0.16), (0.78, 0.24), (0.96, 0.5)], 1.5)
        circle(0.5, 0.5, 0.17, 1.5)
    elif name == "parts":
        # three little cubes, like Smart Segmentation's parts
        for ox, oy in ((0.04, 0.02), (0.52, 0.02), (0.28, 0.46)):
            path([(ox + 0.22, oy + 0.46), (ox + 0.44, oy + 0.33),
                  (ox + 0.44, oy + 0.11), (ox + 0.22, oy - 0.02),
                  (ox, oy + 0.11), (ox, oy + 0.33)], 1.4, closed=True)
            path([(ox, oy + 0.33), (ox + 0.22, oy + 0.2), (ox + 0.44, oy + 0.33)], 1.2)
            path([(ox + 0.22, oy + 0.2), (ox + 0.22, oy - 0.02)], 1.2)
    elif name == "cursor":
        path([(0.30, 0.96), (0.30, 0.14), (0.50, 0.34), (0.62, 0.10),
              (0.72, 0.15), (0.60, 0.39), (0.80, 0.39)], 1.6, closed=True)
    elif name == "merge":
        path([(0.1, 0.86), (0.46, 0.5), (0.9, 0.5)], 1.6)
        path([(0.1, 0.14), (0.46, 0.5)], 1.6)
        dot(0.1, 0.86); dot(0.1, 0.14); dot(0.9, 0.5)
    elif name == "split":
        path([(0.1, 0.5), (0.5, 0.5), (0.9, 0.86)], 1.6)
        path([(0.5, 0.5), (0.9, 0.14)], 1.6)
        dot(0.1, 0.5); dot(0.9, 0.86); dot(0.9, 0.14)
    elif name == "link":
        path([(0.42, 0.72), (0.24, 0.72), (0.12, 0.5), (0.24, 0.28), (0.42, 0.28)], 1.6)
        path([(0.58, 0.72), (0.76, 0.72), (0.88, 0.5), (0.76, 0.28), (0.58, 0.28)], 1.6)
        path([(0.36, 0.5), (0.64, 0.5)], 1.6)
    elif name == "gear":
        circle(0.5, 0.5, 0.19, 1.8)
        for k in range(8):
            a = k * math.pi / 4.0
            path([(0.5 + 0.30 * math.cos(a), 0.5 + 0.30 * math.sin(a)),
                  (0.5 + 0.42 * math.cos(a), 0.5 + 0.42 * math.sin(a))], 2.4)
    elif name == "package":
        path([(0.5, 0.97), (0.92, 0.74), (0.92, 0.26), (0.5, 0.03),
              (0.08, 0.26), (0.08, 0.74)], 1.5, closed=True)
        path([(0.08, 0.74), (0.5, 0.5), (0.92, 0.74)], 1.3)
        path([(0.5, 0.5), (0.5, 0.03)], 1.3)
    elif name == "bake":
        for k in range(3):
            xx = 0.26 + k * 0.24
            path([(xx, 0.08), (xx - 0.09, 0.32), (xx + 0.09, 0.6), (xx, 0.92)], 1.6)
    elif name == "logo":
        # the mark: a cube in outline, the same one the videos and the store
        # pages carry
        r_ = 0.46
        h_ = r_ * 0.87
        top_ = (0.5, 0.5 + r_)
        path([top_, (0.5 + h_, 0.5 + r_ / 2), (0.5 + h_, 0.5 - r_ / 2),
              (0.5, 0.5 - r_), (0.5 - h_, 0.5 - r_ / 2),
              (0.5 - h_, 0.5 + r_ / 2)], 2.2, closed=True)
        path([(0.5 - h_, 0.5 + r_ / 2), (0.5, 0.5), (0.5 + h_, 0.5 + r_ / 2)],
             2.2)
        path([(0.5, 0.5), (0.5, 0.5 - r_)], 2.2)
    elif name == "shirt":
        # a t-shirt: neck, shoulders, sleeves, body
        path([(0.38, 0.92), (0.5, 0.82), (0.62, 0.92), (0.95, 0.74),
              (0.84, 0.54), (0.72, 0.60), (0.72, 0.06), (0.28, 0.06),
              (0.28, 0.60), (0.16, 0.54), (0.05, 0.74)], 1.5, closed=True)
    elif name == "chev_down":
        path([(0.18, 0.64), (0.5, 0.34), (0.82, 0.64)], 1.6)
    elif name == "chev_up":
        path([(0.18, 0.36), (0.5, 0.66), (0.82, 0.36)], 1.6)
    elif name == "close":
        path([(0.2, 0.2), (0.8, 0.8)], 1.6)
        path([(0.2, 0.8), (0.8, 0.2)], 1.6)
    elif name == "info":
        circle(0.5, 0.5, 0.42, 1.3)
        path([(0.5, 0.28), (0.5, 0.56)], 1.5)
        dot(0.5, 0.71, 0.055)


def card_point(context, mouse_x, mouse_y):
    """The pointer in the card's own coordinates, or None when it is not on
    the card. Works from any region of the window."""
    try:
        card = _state.get("card")
        if not card or not is_on():
            return None
        scale = context.preferences.system.ui_scale or 1.0
        for a in context.window.screen.areas:
            if a.type != "VIEW_3D":
                continue
            for r in a.regions:
                if r.type != "WINDOW" or r.as_pointer() != _state.get("hot"):
                    continue
                mx = (mouse_x - r.x) / scale
                my = (mouse_y - r.y) / scale
                cx, cy, cw, ch = card
                if cx - 6 <= mx <= cx + cw + 6 and cy - 6 <= my <= cy + ch + 6:
                    return (mx, my)
                return None
    except Exception:
        pass
    return None


def pointer_on_card(context):
    """Is the pointer on the card right now (no event needed: used while a
    picture is being dragged in from outside Blender)?"""
    try:
        mxy = _state.get("mouse")
        card = _state.get("card")
        if not mxy or not card or not is_on():
            return False
        scale = context.preferences.system.ui_scale or 1.0
        for a in context.window.screen.areas:
            if a.type != "VIEW_3D":
                continue
            for r in a.regions:
                if r.type == "WINDOW" and \
                        r.x <= mxy[0] < r.x + r.width and \
                        r.y <= mxy[1] < r.y + r.height:
                    if r.as_pointer() != _state.get("hot"):
                        return False
                    mx = (mxy[0] - r.x) / scale
                    my = (mxy[1] - r.y) / scale
                    cx, cy, cw, ch = card
                    return (cx - 6 <= mx <= cx + cw + 6 and
                            cy - 6 <= my <= cy + ch + 6)
    except Exception:
        pass
    return False


def over_card(context, event):
    """Is the pointer on the card rather than on the model?

    The card is painted over the 3d view's own region, not beside it, so a
    modal operator that takes every click inside that region takes the
    card's buttons with it - which is what made "Unwrap Those Again" do
    nothing while the island picker was running. Anything modal asks this
    first and passes the click on when the answer is yes.
    """
    try:
        region = context.region
        if region is None:
            return False
        card = _state.get("card")
        if not card:
            return False
        scale = context.preferences.system.ui_scale or 1.0
        mx = (event.mouse_x - region.x) / scale
        my = (event.mouse_y - region.y) / scale
        cx, cy, cw, ch = card
        pad = 6.0
        return (cx - pad <= mx <= cx + cw + pad
                and cy - pad <= my <= cy + ch + pad)
    except Exception:
        return False


def _hit(widgets, mx, my):
    # small buttons laid over a picture are found before the picture
    order = [i for i, it in enumerate(widgets) if it.get("over")] + \
        [i for i, it in enumerate(widgets) if not it.get("over")]
    for i in order:
        it = widgets[i]
        # "field" is the words and the number above a slider. It is not a
        # control - it has nothing to press - but it is the part of the row
        # the pointer actually lands on, so it has to be reachable or the
        # explanation on it can never be seen.
        if it["kind"] in {"label", "logo", "title", "section", "rule",
                          "panel", "sep", "sub"}:
            continue
        x, y, w, h = it["rect"]
        if x <= mx <= x + w and y <= my <= y + h:
            # a row scrolled out of the card is not there to be pressed, even
            # where a sliver of it still sits under the pointer
            c = _state.get("card")
            if c is not None and not (c[1] - 1.0 <= my <= c[1] + c[3] + 1.0):
                continue
            return i
    return None


def _draw():
    """Paint the card. If anything in here throws, the plain sidebar panel
    is put back so the add-on never ends up with no interface at all."""
    try:
        _draw_card()
    except Exception:
        import traceback
        traceback.print_exc()
        if not _state.get("failed"):
            _state["failed"] = True
            try:
                import sys
                mod = sys.modules.get(__package__)
                clear()
                if mod is not None:
                    mod.show_plain_panel()
                print("[Image to 3D] the card could not draw, "
                      "the plain panel is up instead")
            except Exception:
                pass


def _draw_card():
    context = bpy.context
    region = context.region
    if region is None or not _visible(context):
        return
    # Preferences can be loaded after this module registers - a start-up, a
    # different user file - so the saved colour is picked up here rather than
    # only when it is changed.
    prefs = _prefs()
    if prefs is not None:
        key = (getattr(prefs, "accent_hue", None),
               getattr(prefs, "accent_sat", None),
               getattr(prefs, "corner", None))
        if _state.get("tint") != key:
            _state["tint"] = key
            retint()
    widgets, card = _build(context, region)
    if not widgets:
        return
    if _state["typing"] is not None and not any(_typing_here(w) for w in widgets):
        _state["typing"] = None
    # Every 3D view draws its own card, laid out for its own width. With the
    # viewport split in two, the second one drawn used to overwrite the first
    # one's buttons, and clicks were tested against the wrong card. So each
    # keeps its own, and the shared copy is the one the pointer is over.
    key = region.as_pointer()
    _state.setdefault("regions", {})[key] = (widgets, card)
    if _state.get("hot") in (None, key) or _state.get("hot") not in _state["regions"]:
        _state["hot"] = key
        _state["widgets"] = widgets
        _state["card"] = card
    now = time.time()
    # How long the machine took to answer the last repaint we asked for. On a
    # light scene this is a millisecond or two; on a 200k face garment it is
    # fifteen, and asking twenty times a second for that is what made the
    # pointer feel heavy. Measured rather than guessed, because it is the
    # scene that decides, not us.
    if _state["asked"]:
        _state["cost"] = _state["cost"] * 0.7 + (now - _state["asked"]) * 0.3
        _state["asked"] = 0.0
    _state["drawn"] = now

    scale = context.preferences.system.ui_scale or 1.0
    hover = _state["hover"] if _state.get("hot") == key else None
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    shader.bind()

    def box(rect, r, color):
        x, y, w, h = rect
        _fill(shader, x * scale, y * scale, w * scale, h * scale, r * scale,
              color)

    def gbox(rect, r, c_bot, c_top):
        x, y, w, h = rect
        _fill_grad(shader, x * scale, y * scale, w * scale, h * scale,
                   r * scale, c_bot, c_top)

    def lead(rect, r, lit):
        """The two buttons that actually unwrap something.

        Everything else on the card is a setting or a touch-up; these two
        are the work. So they carry a hairline of the accent to tell them
        apart from the rest, and they fill with it under the pointer - the
        colour answers "is this the one I am about to press", which it can
        only do by changing.
        """
        x, y, w, h = rect
        _fill(shader, x * scale, y * scale, w * scale, h * scale, r * scale,
              ACCENT if lit else ACCENT_EDGE)
        _fill_grad(shader, (x + 1.0) * scale, (y + 1.0) * scale,
                   (w - 2.0) * scale, (h - 2.0) * scale,
                   max(0.0, r - 1.0) * scale,
                   ACCENT if lit else CTRL_BOT,
                   ACCENT_HOVER if lit else CTRL_TOP)

    def ring(rect, r, color, t=1.0, inside=None):
        """A border, drawn as the card colour punched out of a filled rect."""
        x, y, w, h = rect
        _fill(shader, x * scale, y * scale, w * scale, h * scale, r * scale,
              color)
        _fill(shader, (x + t) * scale, (y + t) * scale, (w - t * 2) * scale,
              (h - t * 2) * scale, max(0.0, r - t) * scale,
              inside if inside is not None else CARD)

    cx, cy, cw, ch = card
    _fill(shader, (cx - 1.0) * scale, (cy - 1.0) * scale, (cw + 2.0) * scale,
          (ch + 2.0) * scale, (R_CARD + 1.0) * scale, EDGE)
    _fill_grad(shader, cx * scale, cy * scale, cw * scale, ch * scale,
               R_CARD * scale, CARD_BOT, CARD_TOP)

    # Rows scrolled past the top or the bottom must not paint over the
    # viewport. Everything drawn from here on is trimmed to the card's own
    # frame, and the band is cleared again afterwards so the tip that opens
    # beside the card is not trimmed with it.
    clipped = _state["over"] > 0.0
    _CLIP[0] = ((cy + 1.0) * scale, (cy + ch - 1.0) * scale) if clipped else None

    for i, it in enumerate(widgets):
        k = it["kind"]
        x, y, w, h = it["rect"]
        # rows scrolled right out of the card cost nothing to skip
        if clipped and (y + h < cy - 2.0 or y > cy + ch + 2.0):
            continue
        lit = (i == hover)
        if k == "logo":
            gbox(it["rect"], R_LOGO, ACCENT_BOT, ACCENT_TOP)
            _glyph(shader, "logo", x + w * 0.19, y + h * 0.19, w * 0.62,
                   ON_ACCENT, scale)
        elif k == "chip":
            if it["on"]:
                gbox(it["rect"], R_CTL, PICKED_BOT, PICKED_TOP)
                x2, y2, w2, h2 = it["rect"]
                # the accent outline, drawn as a frame around the filled chip
                _fill(shader, x2 * scale, y2 * scale, w2 * scale, h2 * scale,
                      R_CTL * scale, ACCENT)
                _fill_grad(shader, (x2 + 1.0) * scale, (y2 + 1.0) * scale,
                           (w2 - 2.0) * scale, (h2 - 2.0) * scale,
                           max(0.0, R_CTL - 1.0) * scale, PICKED_BOT, PICKED_TOP)
            elif it.get("dim"):
                # shown but not pressable (a model that is already installed)
                x2, y2, w2, h2 = it["rect"]
                _fill(shader, x2 * scale, y2 * scale, w2 * scale, h2 * scale,
                      R_CTL * scale, CTRL_BOT)
            else:
                gbox(it["rect"], R_CTL,
                     CTRL_BOT_HOVER if lit else CTRL_BOT,
                     CTRL_TOP_HOVER if lit else CTRL_TOP)
        elif k == "playbtn":
            # Quiet at rest. It is a link to a video, not something the card
            # wants pressed, so it only comes up under the pointer.
            if lit:
                box(it["rect"], R_CTL, CONTROL_HOVER)
            bx = (x + 9.5) * scale
            by = (y + 7.5) * scale
            side = 11.0 * scale
            batch = batch_for_shader(shader, "TRIS", {"pos": (
                (bx, by), (bx, by + side),
                (bx + side * 0.88, by + side * 0.5))})
            shader.uniform_float("color", ACCENT if lit else TEXT_DIM)
            batch.draw(shader)
        elif k == "info":
            # A quiet hint marker: the round i everybody knows. The words
            # themselves live in the explainer, so the card stays uncluttered.
            # The middle of the disc is named ix, iy and not cx, cy: those two
            # are the card's own corner, read on every row of the loop to
            # decide what is scrolled out of sight, and writing over them here
            # moved the card's foot up to this icon - so every row below it
            # counted as scrolled away and the card came out with its words
            # and nothing else.
            _glyph(shader, "info", x, y + (h - 14.0) * 0.5, 14.0,
                   ACCENT if lit else TEXT_DIM, scale)
        elif k == "play":
            # Same manners as the one in the header: a link, drawn quietly,
            # that only lights up when the pointer is on it.
            if lit:
                _fill(shader, (x - 4.0) * scale, y * scale,
                      (w + 8.0) * scale, h * scale, R_CTL * scale,
                      CONTROL_HOVER)
            bx = (x + 4.0) * scale
            by = (y + 8.0) * scale
            side = 10.0 * scale
            batch = batch_for_shader(shader, "TRIS", {"pos": (
                (bx, by), (bx, by + side),
                (bx + side * 0.88, by + side * 0.5))})
            shader.uniform_float("color", ACCENT if lit else TEXT_DIM)
            batch.draw(shader)
        elif k == "switch":
            _fill(shader, x * scale, y * scale, w * scale, 1.0 * scale, 0.0,
                  RULE)
            kx = (x + w - 30.0) * scale
            ky = (y + (h - 17.0) * 0.5) * scale
            if it["on"]:
                _fill_grad(shader, kx, ky, 30.0 * scale, 17.0 * scale,
                           8.5 * scale, ACCENT_BOT, ACCENT_TOP)
            else:
                _fill(shader, kx, ky, 30.0 * scale, 17.0 * scale, 8.5 * scale,
                      TRACK)
            knob = kx + (14.5 if it["on"] else 2.0) * scale
            _fill(shader, knob, ky + 2.0 * scale, 13.0 * scale, 13.0 * scale,
                  6.5 * scale, KNOB)
        elif k == "field":
            if _typing_here(it):
                # a border, not a filled block: the inside has to be opaque or
                # the accent shows straight through it and the whole box reads
                # as one solid orange slab
                ring((x + w - 84.0, y, 84.0, h), R_BOX, ACCENT, 1.5,
                     inside=CARD)
            else:
                gbox((x + w - 84.0, y, 84.0, h), R_BOX, CTRL_BOT, CTRL_TOP)
        elif k == "bar":
            ty = y + (h - 5.0) * 0.5
            _fill(shader, x * scale, ty * scale, w * scale, 5.0 * scale,
                  2.5 * scale, TRACK)
            span = float(it["hi"] - it["lo"]) or 1.0
            frac = max(0.0, min(1.0, (it["value"] - it["lo"]) / span))
            _fill(shader, x * scale, ty * scale, w * frac * scale,
                  5.0 * scale, 2.5 * scale, ACCENT)
            _fill(shader, (x + w * frac - 7.0) * scale, (ty - 4.5) * scale,
                  14.0 * scale, 14.0 * scale, 7.0 * scale, ACCENT)
        elif k == "ramp":
            ramp = it["ramp"]
            n = len(ramp)
            ty = y + (h - 9.0) * 0.5
            step = w / float(n)
            for i, c in enumerate(ramp):
                _fill(shader, (x + i * step) * scale, ty * scale,
                      (step + 0.8) * scale, 9.0 * scale, 0.0, c)
            frac = max(0.0, min(1.0, (it["value"] - it["lo"])
                                / (float(it["hi"] - it["lo"]) or 1.0)))
            kx = x + w * frac
            _fill(shader, (kx - 6.0) * scale, (ty - 3.0) * scale,
                  12.0 * scale, 15.0 * scale, 6.0 * scale, KNOB)
            _fill(shader, (kx - 4.0) * scale, (ty - 1.0) * scale,
                  8.0 * scale, 11.0 * scale, 4.0 * scale, ACCENT)
        elif k == "rule":
            _fill(shader, x * scale, y * scale, w * scale, 1.0 * scale, 0.0,
                  RULE)
        elif k == "run":
            lead(it["rect"], R_CTL, lit)
            if it.get("icon"):
                tw = _text_w(it["text"], 13, scale)
                gx = max(x + 12.0, x + w * 0.5 - tw * 0.5 - 26.0)
                _glyph(shader, it["icon"], gx, y + (h - 18.0) * 0.5, 18.0,
                       ON_ACCENT if lit else TEXT, scale)
        elif k == "running":
            # the button itself becomes the bar: the track, then the part done
            box(it["rect"], R_CTL, CONTROL)
            done = max(0.0, min(1.0, it["frac"]))
            if done > 0.0:
                gbox((x, y, max(R_CTL * 2.0, w * done), h), R_CTL,
                     ACCENT_BOT, ACCENT_TOP)
        elif k == "ghost":
            ring(it["rect"], R_CTL, ACCENT, 1.0,
                 inside=CONTROL_HOVER if lit else CARD)
        elif k in {"btn", "tool", "fold"}:
            if it.get("lead"):
                lead(it["rect"], R_CTL, lit)
            else:
                gbox(it["rect"], R_CTL,
                     CTRL_BOT_HOVER if lit else CTRL_BOT,
                     CTRL_TOP_HOVER if lit else CTRL_TOP)
            if it.get("icon"):
                tw = _text_w(it["text"], 12, scale)
                gx = max(x + 10.0, x + w * 0.5 - tw * 0.5 - 20.0)
                _glyph(shader, it["icon"], gx, y + (h - 14.0) * 0.5, 14.0,
                       ON_ACCENT if (it.get("lead") and lit) else TEXT, scale)
        elif k == "head":
            _glyph(shader, it.get("icon", "gear"), x, y + (h - 18.0) * 0.5,
                   18.0, TEXT if it["on"] else TEXT_DIM, scale)
            _glyph(shader, "chev_up" if it["on"] else "chev_down",
                   x + w - 14.0, y + (h - 12.0) * 0.5, 12.0, TEXT_DIM, scale)
        elif k == "panel":
            x2, y2, w2, h2 = it["rect"]
            _fill(shader, x2 * scale, y2 * scale, w2 * scale, h2 * scale,
                  R_CARD * 0.8 * scale, PANEL_EDGE)
            _fill_grad(shader, (x2 + 1.0) * scale, (y2 + 1.0) * scale,
                       (w2 - 2.0) * scale, (h2 - 2.0) * scale,
                       (R_CARD * 0.8 - 1.0) * scale, PANEL_BOT, PANEL_TOP)
        elif k == "thumb":
            box(it["rect"], R_BOX, CTRL_TOP_HOVER if lit else PANEL_BOT)
            t = _thumb(it["path"])
            if t is not None and _inside(y * scale, h * scale):
                tex, tw_, th_ = t
                aw, ah = w - 12.0, h - 12.0
                f = min(aw / float(tw_), ah / float(th_))
                dw, dh = tw_ * f, th_ * f
                _draw_poster(tex, x + (w - dw) * 0.5, y + (h - dh) * 0.5,
                             dw, dh, scale)
                gpu.state.blend_set("ALPHA")
                shader.bind()
        elif k == "path":
            gbox(it["rect"], R_CTL, CTRL_BOT_HOVER if lit else CTRL_BOT,
                 CTRL_TOP_HOVER if lit else CTRL_TOP)
            _glyph(shader, "package", x + 9.0, y + (h - 13.0) * 0.5, 13.0,
                   TEXT_DIM, scale)
        elif k == "meter":
            fr_ = max(0.0, min(1.0, it["frac"]))
            tx0, tw0 = x + 44.0, w - 44.0 - 92.0
            ty0 = y + (h - 6.0) * 0.5
            _fill(shader, tx0 * scale, ty0 * scale, tw0 * scale, 6.0 * scale,
                  3.0 * scale, PANEL_BOT)
            colm = (0.92, 0.33, 0.30, 1.0) if fr_ > 0.9 else _ac()
            if fr_ > 0.0:
                _fill(shader, tx0 * scale, ty0 * scale,
                      max(6.0, tw0 * fr_) * scale, 6.0 * scale, 3.0 * scale,
                      colm)
        elif k == "vslot":
            real = it["path"]
            bg = CTRL_TOP_HOVER if lit else PANEL_BOT
            if real:
                ring(it["rect"], R_BOX, _ac(0.9), 1.2, inside=bg)
            else:
                box(it["rect"], R_BOX, bg)
            t = _thumb(real or it["ex"])
            foot = 0.0 if real else 18.0
            if t is not None and _inside(y * scale, h * scale):
                tex, tw_, th_ = t
                aw, ah = w - 12.0, h - 12.0 - foot
                f = min(aw / float(tw_), ah / float(th_))
                dw, dh = tw_ * f, th_ * f
                px_, py_ = x + (w - dw) * 0.5, y + foot + 6.0 + (ah - dh) * 0.5
                gpu.state.blend_set("ALPHA")
                _draw_poster(tex, px_, py_, dw, dh, scale)
                gpu.state.blend_set("ALPHA")
                shader.bind()
            if real:
                lw_ = _text_w(it["text"], 10, scale) + 12.0
                _fill(shader, (x + 5.0) * scale, (y + 5.0) * scale,
                      lw_ * scale, 16.0 * scale, 4.0 * scale,
                      (0.0, 0.0, 0.0, 0.65))
            else:
                # dashed border that follows the rounded corners
                dc = (0.75, 0.76, 0.8, 0.55 if not lit else 0.9)
                r_ = R_BOX
                i0 = 0.75
                xa, ya, xb, yb = x + i0, y + i0, x + w - i0, y + h - i0
                path = []
                for (ccx, ccy, a0) in ((xb - r_, ya + r_, -math.pi / 2),
                                       (xb - r_, yb - r_, 0.0),
                                       (xa + r_, yb - r_, math.pi / 2),
                                       (xa + r_, ya + r_, math.pi)):
                    for q in range(7):
                        an = a0 + (math.pi / 2) * q / 6.0
                        path.append((ccx + math.cos(an) * r_,
                                     ccy + math.sin(an) * r_))
                path.append(path[0])
                # walk the outline once and draw every other piece of
                # equal length: a fixed number of steps, so it always ends
                cum = [0.0]
                for j in range(len(path) - 1):
                    cum.append(cum[-1] + math.hypot(
                        path[j + 1][0] - path[j][0],
                        path[j + 1][1] - path[j][1]))
                total = cum[-1]

                def at(dist):
                    dist = max(0.0, min(total, dist))
                    for j in range(len(path) - 1):
                        if cum[j + 1] >= dist:
                            L = cum[j + 1] - cum[j]
                            f_ = (dist - cum[j]) / L if L > 1e-9 else 0.0
                            return _lerp(path[j], path[j + 1], f_)
                    return path[-1]
                nd = max(8, min(400, int(total / 9.0)))
                period = total / nd
                for k2 in range(nd):
                    d0 = k2 * period
                    d1 = d0 + period * 0.55
                    pts_ = [at(d0)]
                    for j in range(len(path) - 1):
                        if d0 < cum[j + 1] < d1:
                            pts_.append(path[j + 1])
                    pts_.append(at(d1))
                    for q in range(len(pts_) - 1):
                        _seg(shader, pts_[q], pts_[q + 1], 1.2, dc, scale)
        elif k == "viewmap":
            _draw_viewmap(shader, x, y, w, h, it["have"], scale)
            shader.bind()
        elif k == "sep":
            _fill(shader, x * scale, y * scale, w * scale, 1.0 * scale, 0.0,
                  RULE)

    for it in widgets:
        k = it["kind"]
        x, y, w, h = it["rect"]
        if k == "title":
            size = _fit(it["text"], it.get("room", 999.0), 16, scale, bold=True, floor=11) \
                if it.get("room") else 16
            _text(it["text"], x * scale, y * scale, size, TEXT, scale, bold=True)
        elif k == "label":
            _text(it["text"], x * scale, y * scale, it["size"], it["color"],
                  scale)
        elif k == "section":
            _text(it["text"], x * scale, (y + 3.0) * scale, 10, TEXT_DIM,
                  scale)
        elif k == "chip":
            size = _fit(it["text"], w - 10.0, it.get("tsize", 12), scale)
            _text(it["text"], 0, (y + (h - 11.0) * 0.5) * scale, size,
                  TEXT if it["on"] else (TEXT_DIM if it.get("dim")
                                         else TEXT_CHIP), scale,
                  mid=(x + w * 0.5) * scale)
        elif k == "play":
            lit = (widgets.index(it) == hover)
            _text(it["text"], (x + 22.0) * scale, (y + 8.0) * scale, 12,
                  TEXT if lit else TEXT_DIM, scale)
        elif k == "path":
            room_ = w - 38.0
            t_ = it["text"]
            if _text_w(t_, 11, scale) > room_:
                while len(t_) > 4 and _text_w("..." + t_, 11, scale) > room_:
                    t_ = t_[1:]
                t_ = "..." + t_
            _text(t_, (x + 29.0) * scale, (y + (h - 10.0) * 0.5) * scale, 11,
                  TEXT, scale)
        elif k == "meter":
            _text(it["text"], x * scale, (y + 3.0) * scale, 10, TEXT_DIM,
                  scale)
            _text(it["value"], 0, (y + 3.0) * scale, 10, TEXT, scale,
                  right=(x + w) * scale)
        elif k == "vslot":
            if it["path"]:
                _text(it["text"], (x + 11.0) * scale, (y + 9.0) * scale, 10,
                      TEXT, scale)
            else:
                lit_ = (widgets.index(it) == hover)
                _text("+  " + it["text"], 0, (y + 6.0) * scale,
                      13 if it.get("big") else 11, TEXT if lit_ else TEXT_DIM,
                      scale, mid=(x + w * 0.5) * scale)
        elif k == "info":
            lit = (widgets.index(it) == hover)
            size = _fit(it["text"], w - 24.0, 11, scale, floor=8)
            _text(it["text"], (x + 20.0) * scale, (y + (h - 10.0) * 0.5) * scale,
                  size, TEXT if lit else TEXT_DIM, scale)
        elif k == "switch":
            size = _fit(it["text"], w - 40.0, 12, scale)
            _text(it["text"], x * scale, (y + (h - 11.0) * 0.5) * scale, size,
                  TEXT, scale)
        elif k == "field":
            size = _fit(it["text"], w - 92.0, 12, scale)
            _text(it["text"], x * scale, (y + 8.0) * scale, size, TEXT, scale)
            if _typing_here(it):
                caret = "|" if int(time.time() * 2) % 2 else " "
                _text(_state["typing"]["text"] + caret, 0, (y + 8.0) * scale,
                      12, ACCENT, scale, right=(x + w - 12.0) * scale)
            else:
                vsize = _fit(it["shown"], 68.0, 12, scale)
                _text(it["shown"], 0, (y + 8.0) * scale, vsize, TEXT, scale,
                      right=(x + w - 12.0) * scale)
        elif k == "fold":
            _text(">" if not it["on"] else "v", (x + 12.0) * scale,
                  (y + 9.0) * scale, 11, TEXT_DIM, scale)
            _text(it["text"], (x + 28.0) * scale, (y + 9.0) * scale,
                  _fit(it["text"], w - 38.0, 12, scale),
                  TEXT if it["on"] else TEXT_DIM, scale)
        elif k == "quietfold":
            lit = (widgets.index(it) == hover)
            tint = TEXT if (it["on"] or lit) else TEXT_DIM
            _text(">" if not it["on"] else "v", x * scale,
                  (y + 6.0) * scale, 11, tint, scale)
            _text(it["text"], (x + 14.0) * scale, (y + 6.0) * scale, 11,
                  tint, scale)
        elif k == "run":
            lit = (widgets.index(it) == hover)
            room = w - 24.0 - (28.0 if it.get("icon") else 0.0)
            size = _fit(it["text"], room, 13, scale, bold=True)
            tw = _text_w(it["text"], size, scale) if it.get("icon") else 0.0
            mid = x + w * 0.5
            if it.get("icon"):
                gx = max(x + 12.0, mid - tw * 0.5 - 26.0)
                mid = max(mid, gx + 18.0 + 8.0 + tw * 0.5)
                mid = min(mid, x + w - 10.0 - tw * 0.5)
            _text(it["text"], 0, (y + (h - 12.0) * 0.5) * scale, size,
                  ON_ACCENT if lit else TEXT, scale, mid=mid * scale,
                  bold=True)
        elif k == "running":
            _text(it["text"], 0, (y + (h - 12.0) * 0.5) * scale, 13, TEXT,
                  scale, mid=(x + w * 0.5) * scale, bold=True)
        elif k == "runline":
            _text(it["text"], 0, (y + 2.0) * scale, 10, TEXT_DIM, scale,
                  mid=(x + w * 0.5) * scale)
        elif k == "ghost":
            _text(it["text"], 0, (y + (h - 11.0) * 0.5) * scale, 12, ACCENT,
                  scale, mid=(x + w * 0.5) * scale)
        elif k == "linktext":
            lit = (widgets.index(it) == hover)
            _text(it["text"], 0, (y + (h - 11.0) * 0.5) * scale, 11,
                  ACCENT if lit else TEXT_DIM, scale,
                  mid=(x + w * 0.5) * scale)
        elif k in {"btn", "tool"}:
            lit = (widgets.index(it) == hover)
            room = w - 20.0 - (22.0 if it.get("icon") else 0.0)
            if it.get("over"):
                room = w
            size = _fit(it["text"], room, it.get("tsize", 12), scale)
            tw = _text_w(it["text"], size, scale) if it.get("icon") else 0.0
            mid = x + w * 0.5
            if it.get("icon"):
                gx = max(x + 10.0, mid - tw * 0.5 - 20.0)
                mid = max(mid, gx + 14.0 + 6.0 + tw * 0.5)
                mid = min(mid, x + w - 8.0 - tw * 0.5)
            _text(it["text"], 0, (y + (h - (14.0 if it.get("over")
                                             else 11.0)) * 0.5) * scale,
                  size, ON_ACCENT if (it.get("lead") and lit) else TEXT,
                  scale, mid=mid * scale)
        elif k == "head":
            lit = (widgets.index(it) == hover)
            size = _fit(it["text"], w - 46.0, 12, scale, bold=True)
            _text(it["text"], (x + 28.0) * scale, (y + (h - 10.0) * 0.5) * scale,
                  size, TEXT if (it["on"] or lit) else TEXT_DIM, scale, bold=True)
        elif k == "sub":
            # the header play button sits to its right: shrink, never overlap
            size = _fit(it["text"], it.get("room", 999.0), 10, scale, floor=7) \
                if it.get("room") else 10
            _text(it["text"], x * scale, y * scale, size, TEXT_DIM, scale)

    _CLIP[0] = None
    if clipped:
        # A slim bar down the right edge: it says both that there is more
        # below and how far down the card we are, so the foot is never
        # silently lost off the bottom of the frame.
        over = _state["over"]
        full = ch + over
        tx = cx + cw - 6.0
        ty, th = cy + 8.0, ch - 16.0
        _fill(shader, tx * scale, ty * scale, 5.0 * scale, th * scale,
              2.5 * scale, TRACK)
        hh = max(26.0, th * (ch / full))
        f = min(1.0, max(0.0, _state["scroll"] / over)) if over > 0.0 else 0.0
        hy = ty + (th - hh) * (1.0 - f)
        _fill(shader, tx * scale, hy * scale, 5.0 * scale, hh * scale,
              2.5 * scale, (0.52, 0.53, 0.56, 0.98))

    _draw_tip(shader, card, scale)
    gpu.state.blend_set("NONE")


# -- the operator that gives it a mouse -------------------------------------

def _redraw(context, area=None):
    """Ask for a repaint - of one area where we know which one.

    A repaint is not cheap: Blender redraws the whole 3D view, model and all,
    to move a highlight by one row. On a 197k face garment that is what the
    pointer felt like it was dragging. So only the area under the pointer is
    asked, and only when something actually changed.
    """
    hot = _state.get("hot_area")
    if hot is not None:
        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.as_pointer() == hot:
                    a.tag_redraw()
                    return
    if area is not None:
        area.tag_redraw()
        return
    for a in context.screen.areas:
        if a.type == "VIEW_3D":
            a.tag_redraw()


# Buttons that do real work on the mesh are not run from inside the card's
# own mouse handler. The card is a modal operator; starting another operator
# that changes mode, edits the mesh and calls the packer while that modal is
# still on Blender's handler stack is the one thing here that can take the
# whole program down, and Pack Again - mode in, pack, mode out - is the
# heaviest of them. They are handed to a timer instead, which runs them a
# moment later in an ordinary context, with the window and area they were
# pressed in put back around them.
_DEFER = {"img3d.generate", "img3d.install_engine"}


def say(text):
    """Put a line on the card for a few seconds.

    An operator started from one of the card's buttons runs from a timer, so
    its own error report lands in the Info log where nobody is looking. This
    is the way back to the card.
    """
    _state["note"] = (str(text), time.time())
    try:
        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.type == 'VIEW_3D':
                    a.tag_redraw()
    except Exception:
        pass


def hot_view(context=None):
    """The 3D view the pointer is actually in, as (window, area, region).

    The card is one modal operator and it is started in whichever viewport
    Blender happened to list first. Everything it runs therefore inherits
    that viewport's area and region, not the one whose card was clicked. On
    a single viewport those are the same and nothing shows; on a split
    screen - a small camera view beside the main one - a button pressed on
    the main card ran against the camera view. Pick Islands On Model is
    where it was noticed: the picker kept the wrong region, so every click
    on the model was outside the region it was watching and nothing was
    picked, while clicking in the little camera view worked.

    The modal already tracks the region under the pointer for its own
    drawing. This hands that back as real objects, so an operator can be
    run against it.
    """
    want_r = _state.get("hot")
    want_a = _state.get("hot_area")
    if want_r is None and want_a is None:
        return None, None, None
    try:
        wins = bpy.context.window_manager.windows
    except Exception:
        return None, None, None
    for win in wins:
        for a in win.screen.areas:
            if a.type != "VIEW_3D":
                continue
            if want_a is not None and a.as_pointer() != want_a:
                continue
            for r in a.regions:
                if r.type != "WINDOW":
                    continue
                if want_r is None or r.as_pointer() == want_r:
                    return win, a, r
    return None, None, None


def _run_later(idname, win, area, region):
    def go():
        mod, _, name = idname.partition(".")
        op = getattr(getattr(bpy.ops, mod, None), name, None)
        if op is None:
            return None
        try:
            if win is not None and area is not None:
                with bpy.context.temp_override(window=win, area=area,
                                               region=region):
                    op("INVOKE_DEFAULT")
            else:
                op("INVOKE_DEFAULT")
        except Exception as exc:
            _state["note"] = (str(exc)[:44], time.time())
            print("[Image to 3D] card:", exc)
        return None

    try:
        bpy.app.timers.register(go, first_interval=0.0)
    except Exception:
        go()


def _fire(context, it, mx):
    st = _settings(context)
    kind, arg = it["act"]
    if kind == "group":
        _state["group"] = arg
    elif kind in {"backend", "mode"}:
        setattr(st, kind, arg)
    elif kind == "enum":
        owner, name, code = arg
        setattr(owner, name, code)
    elif kind == "enum_menu":
        try:
            bpy.ops.wm.context_menu_enum("INVOKE_DEFAULT", data_path=arg)
        except Exception as exc:
            _state["note"] = (str(exc)[:44], time.time())
    elif kind == "prop":
        owner, name = arg
        setattr(owner, name, not getattr(owner, name))
    elif kind == "tab":
        _state["tab"] = arg
        _state["scroll"] = 0.0
    elif kind == "fold":
        if arg in _state["open"]:
            _state["open"].discard(arg)
        else:
            _state["open"].add(arg)
    elif kind == "slide":
        owner, name = arg
        x, y, w, h = it["rect"]
        frac = max(0.0, min(1.0, (mx - x) / max(1.0, w)))
        val = it["lo"] + frac * (it["hi"] - it["lo"])
        setattr(owner, name,
                int(round(val)) if not it.get("decimals") else float(val))
    elif kind == "call":
        name_, kw_ = arg
        mod_, _, n_ = name_.partition(".")
        try:
            op_ = getattr(getattr(bpy.ops, mod_), n_)
            if n_ == "dl_page":
                op_("INVOKE_DEFAULT", **kw_)
            else:
                op_(**kw_)
        except Exception as exc:
            _state["note"] = (str(exc)[:44], time.time())
    elif kind == "url":
        if arg:
            try:
                bpy.ops.wm.url_open(url=arg)
            except Exception:
                pass
    elif kind == "op":
        if arg == "__lesson_hard__":
            try:
                from . import LESSON_HARDSURFACE as url
                bpy.ops.wm.url_open(url=url)
            except Exception:
                pass
            return
        if arg in ("__lesson_parts__", "__lesson_udim__",
                   "__lesson_symmetry__", "__lesson_stack__"):
            try:
                from . import LESSONS
                found = LESSONS.get(arg[9:-2])
            except Exception:
                found = None
            if found:
                try:
                    bpy.ops.wm.url_open(url=found[1])
                except Exception:
                    pass
            return
        if arg == "__lesson__":
            try:
                from . import LESSONS
                found = LESSONS.get("cloth")
            except Exception:
                found = None
            if found:
                try:
                    bpy.ops.wm.url_open(url=found[1])
                except Exception:
                    pass
            return
        mod, _, name = arg.partition(".")
        op = getattr(getattr(bpy.ops, mod, None), name, None)
        if op is None:
            return
        # Most of these want Object Mode, and the card is reachable from
        # Sculpt Mode because Relax puts you there. Pressing Confirm from a
        # brush did nothing at all and said nothing either; now the mode is
        # left first and the action runs, and if it still cannot, the card
        # says why instead of swallowing it.
        try:
            ready = op.poll()
        except Exception:
            ready = False
        obj = context.active_object
        if not ready and obj is not None and obj.mode != "OBJECT":
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
                ready = op.poll()
            except Exception:
                ready = False
        if not ready:
            _state["note"] = (WHY.get(arg, "that one cannot run here"),
                              time.time())
            return
        # Run it against the viewport the button was pressed in, not the one
        # the card's modal happens to live in.
        hw, ha, hr = hot_view(context)
        if ha is None:
            hw = getattr(context, "window", None)
            ha = getattr(context, "area", None)
            hr = getattr(context, "region", None)
        if arg in _DEFER:
            _run_later(arg, hw, ha, hr)
            return
        try:
            if hw is not None and ha is not None and hr is not None:
                with bpy.context.temp_override(window=hw, area=ha, region=hr):
                    op("INVOKE_DEFAULT")
            else:
                op("INVOKE_DEFAULT")
        except Exception as exc:
            _state["note"] = (str(exc)[:44], time.time())
            print("[Image to 3D] card:", exc)


class IMG3D_PT_tab(bpy.types.Panel):
    """An empty panel, kept only so the sidebar has a tab to switch to.

    The card is the interface now, and it follows the sidebar's tab. A tab
    with nothing registered under it does not exist, so this holds the name
    open and draws nothing at all - the card fills that space itself.
    """

    bl_label = "Image to 3D 2"
    bl_idname = "IMG3D_PT_tab"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB
    bl_options = {"HIDE_HEADER"}

    def draw(self, context):
        # The card is painted over this region. The few lines here are only
        # seen when the card is not drawing, and then they are the way out.
        if _state.get("handle") is not None and not _state.get("failed"):
            return
        col = self.layout.column()
        col.label(text="The viewport card is off", icon='INFO')
        col.operator("img3d.ui_switch", text="Use the plain panel",
                     icon='WINDOW').classic = True


class IMG3D_OT_overlay(bpy.types.Operator):
    """Give the viewport card a mouse. Started on its own, never by hand."""

    bl_idname = "img3d.card"
    bl_label = "Viewport Card"
    bl_options = {"INTERNAL"}

    def modal(self, context, event):
        # One card, one modal. The keep-alive below starts a new one when it
        # has heard nothing for a few seconds - which is also what happens
        # when Blender simply sits in the background - and the old one was
        # still there. Found with 76 of them stacked up: every mouse move ran
        # the card's hit test 76 times, the sliders lagged behind the pointer
        # and felt dead. A copy that is not the newest now leaves quietly.
        if getattr(self, "_token", None) != _state.get("token"):
            return {"CANCELLED"}
        _state["beat"] = time.time()
        # where the pointer is, also while a file is dragged in from outside
        _state["mouse"] = (event.mouse_x, event.mouse_y)
        if _state["quit"]:
            _state["modal"] = False
            return {"CANCELLED"}
        if _dialog_open(context):
            _state["hover"] = None
            _state["tip"] = None
            return {"PASS_THROUGH"}
        # The pointer's own viewport. The modal belongs to whichever 3D view
        # it was started in, and Blender gives its coordinates relative to
        # that one - so in a split screen every click on the other card
        # landed hundreds of pixels away from where it was aimed. Held while
        # a slider is dragged or a number typed, so the pointer drifting over
        # the border does not swap cards half way.
        hot_area = None
        hot_region = None
        busy = _state["drag"] is not None or _state["typing"] is not None
        for a in context.window.screen.areas:
            if a.type != "VIEW_3D":
                continue
            for r in a.regions:
                if r.type != "WINDOW":
                    continue
                if busy:
                    if r.as_pointer() == _state.get("hot"):
                        hot_area, hot_region = a, r
                elif (r.x <= event.mouse_x < r.x + r.width
                        and r.y <= event.mouse_y < r.y + r.height):
                    hot_area, hot_region = a, r
        if hot_region is not None:
            key = hot_region.as_pointer()
            laid = _state.get("regions", {}).get(key)
            if laid is not None and key != _state.get("hot"):
                _state["hot"] = key
                _state["widgets"], _state["card"] = laid
                _state["hover"] = None
                _state["tip"] = None
            _state["hot_area"] = hot_area.as_pointer()
        if not _visible(context, hot_area):
            if _state["typing"] is not None:
                _stop_typing(context, False)
            if _state["hover"] is not None:
                _state["hover"] = None
                _state["drag"] = None
                _state["tip"] = None
            # Clicking another tab in the sidebar repaints the sidebar and
            # nothing else, so the card's own handler is never called again
            # and the last painting of it just stays on the screen. Blender's
            # panels vanish because they live in the region that was
            # repainted; the card does not, so it has to ask.
            if _state["was_up"]:
                _state["was_up"] = False
                _redraw(context, getattr(context, "area", None))
            return {"PASS_THROUGH"}
        if not _state["was_up"]:
            # Coming back to our tab repaints the sidebar and nothing else, so
            # without this the card waited for the next viewport redraw to
            # appear - a mouse move, or the timer below. That wait is what
            # read as the card opening late, or not at all until the sidebar
            # was closed and opened again.
            _state["was_up"] = True
            _redraw(context, getattr(context, "area", None))

        scale = context.preferences.system.ui_scale or 1.0
        if hot_region is not None:
            mx = (event.mouse_x - hot_region.x) / scale
            my = (event.mouse_y - hot_region.y) / scale
        else:
            mx = event.mouse_region_x / scale
            my = event.mouse_region_y / scale

        # A box open for typing owns the keyboard until it is closed, or a
        # digit would reach Blender as a shortcut and change the mode.
        if _state["typing"] is not None:
            if event.value != "PRESS":
                return {"RUNNING_MODAL"}
            if event.type in {"RET", "NUMPAD_ENTER"}:
                _stop_typing(context, True)
                return {"RUNNING_MODAL"}
            if event.type == "ESC":
                _stop_typing(context, False)
                return {"RUNNING_MODAL"}
            if event.type == "BACK_SPACE":
                t = _state["typing"]
                t["text"] = "" if t.get("fresh") else t["text"][:-1]
                t["fresh"] = False
                _redraw(context, context.area)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE":
                _stop_typing(context, True)   # clicking away keeps it
                return {"RUNNING_MODAL"}
            ch = event.ascii
            if ch and (ch.isdigit() or ch in "-."):
                # The box opens showing the current value, selected the way
                # Blender's own fields open. The first key replaces it: typing
                # 10000 into a box that read 7262 used to give 726210000,
                # which was then clamped to the maximum and looked like the
                # typing had been ignored.
                t = _state["typing"]
                if t.get("fresh"):
                    t["text"] = ""
                    t["fresh"] = False
                t["text"] += ch
                _redraw(context, context.area)
            return {"RUNNING_MODAL"}
        widgets = _state["widgets"]
        over = _hit(widgets, mx, my)

        if _state["drag"] is not None:
            if event.type == "MOUSEMOVE":
                it = (widgets[_state["drag"]]
                      if _state["drag"] < len(widgets) else None)
                if it is not None:
                    _fire(context, it, mx)
                    _redraw(context, context.area)
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                it = (widgets[_state["drag"]]
                      if _state["drag"] < len(widgets) else None)
                _state["drag"] = None
                # The colour is meant to outlast the session. Blender only
                # writes preferences out by itself when auto-save is on, so
                # the file is written here, once, when the knob is let go.
                if it is not None and (it["kind"] == "ramp"
                                      or it.get("save")):
                    try:
                        bpy.ops.wm.save_userpref()
                    except Exception:
                        pass
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}

        # A pen and tablet has no wheel, and the middle button is what a
        # pen's barrel switch is usually mapped to. Blender's own panels
        # take a middle drag as a pan, so the card does too: grab the card
        # and it moves with the pointer. Only while there is something to
        # reach, so the viewport keeps its own middle button everywhere
        # else and whenever the card already fits.
        if _state.get("pan") is not None:
            if event.type == "MOUSEMOVE":
                sy, s0 = _state["pan"]
                was = _state["scroll"]
                _state["scroll"] = min(max(0.0, s0 + (my - sy)),
                                       _state["over"])
                if _state["scroll"] != was:
                    _redraw(context, getattr(context, "area", None))
                return {"RUNNING_MODAL"}
            if event.type == "MIDDLEMOUSE" and event.value == "RELEASE":
                _state["pan"] = None
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}

        if event.type == "MIDDLEMOUSE" and event.value == "PRESS":
            card = _state.get("card")
            if (_state["over"] > 0.0 and card is not None
                    and card[0] <= mx <= card[0] + card[2] + TAB_STRIP
                    and card[1] <= my <= card[1] + card[3]):
                _state["pan"] = (my, _state["scroll"])
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        # The bar down the right edge is a handle, not only a mark. On a
        # laptop there is no wheel and no middle button - the wheel is a
        # trackpad gesture and the pan is a middle drag - so without
        # something to take hold of, the foot of a card taller than the
        # viewport could not be reached at all.
        if _state.get("bar") is not None:
            if event.type == "MOUSEMOVE":
                sy, s0, span = _state["bar"]
                was = _state["scroll"]
                _state["scroll"] = min(max(0.0, s0 - (my - sy) * span),
                                       _state["over"])
                if _state["scroll"] != was:
                    _redraw(context, getattr(context, "area", None))
                return {"RUNNING_MODAL"}
            if event.type == "LEFTMOUSE" and event.value == "RELEASE":
                _state["bar"] = None
                return {"RUNNING_MODAL"}
            return {"RUNNING_MODAL"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            card = _state.get("card")
            over_now = _state["over"]
            if (over_now > 0.0 and card is not None
                    and card[0] + card[2] - 14.0 <= mx
                    <= card[0] + card[2] + 4.0
                    and card[1] <= my <= card[1] + card[3]):
                ch = card[3]
                th = max(ch - 16.0, 1.0)
                hh = max(26.0, th * (ch / (ch + over_now)))
                span = over_now / max(th - hh, 1.0)
                _state["bar"] = (my, _state["scroll"], span)
                return {"RUNNING_MODAL"}

        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE", "TRACKPADPAN",
                          "PAGE_UP", "PAGE_DOWN"}:
            # Only when the pointer is on the card, and only when there is
            # something to reach. Anywhere else the wheel is the viewport's.
            # A trackpad sends no wheel at all - two fingers on a laptop is
            # TRACKPADPAN - so on a machine without a mouse the card could
            # not be scrolled and the foot of it simply could not be
            # reached. The tab strip counts as the card here: it is the
            # strip the card is pinned against, and a wheel there that did
            # nothing read as the card being stuck.
            card = _state.get("card")
            if (_state["over"] > 0.0 and card is not None
                    and card[0] <= mx <= card[0] + card[2] + TAB_STRIP
                    and card[1] <= my <= card[1] + card[3]):
                if event.type == "TRACKPADPAN":
                    step = float(getattr(event, "mouse_prev_y", event.mouse_y)
                                 - event.mouse_y) / scale
                elif event.type in {"PAGE_UP", "PAGE_DOWN"}:
                    if event.value != "PRESS":
                        return {"RUNNING_MODAL"}
                    step = (card[3] - 40.0) * (1.0 if event.type == "PAGE_DOWN"
                                               else -1.0)
                else:
                    step = 48.0 if event.type == "WHEELDOWNMOUSE" else -48.0
                was = _state["scroll"]
                _state["scroll"] = min(max(0.0, was + step), _state["over"])
                if _state["scroll"] != was:
                    _redraw(context, getattr(context, "area", None))
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

        if event.type == "MOUSEMOVE":
            if over != _state["hover"]:
                _state["hover"] = over
                key = None
                if over is not None:
                    key = widgets[over].get("tip")
                _state["tip"] = ((over, key, time.time())
                                 if key is not None else None)
                if _state["tip"] is not None and not _state["ticking"]:
                    _state["ticking"] = True
                    if not bpy.app.timers.is_registered(_tick):
                        bpy.app.timers.register(_tick, first_interval=TIP_DELAY)
                # Sweeping across the card crosses a row every few pixels, and
                # a frame for each of those is a frame of the whole viewport.
                # Twenty-five a second is smooth and is a third of the cost.
                now = time.time()
                wait = max(0.04, _state["cost"] * 1.5)
                if now - _state["painted"] > wait:
                    _state["painted"] = now
                    _state["asked"] = now
                    _state["dirty"] = False
                    _redraw(context, context.area)
                else:
                    _state["dirty"] = True
                    if not bpy.app.timers.is_registered(_settle):
                        bpy.app.timers.register(_settle, first_interval=wait)
            return {"PASS_THROUGH"} if over is None else {"RUNNING_MODAL"}

        if over is None:
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            it = widgets[over]
            if "act" not in it:
                return {"RUNNING_MODAL"}      # a label, not a button
            kind, arg = it["act"]
            if kind == "type":
                fx, fy, fw, fh = it["rect"]
                if mx < fx + fw - 84.0:
                    return {"RUNNING_MODAL"}  # the words, not the number
                owner, prop, lo, hi, decimals = arg
                val = getattr(owner, prop)
                _state["typing"] = {
                    "owner": owner, "prop": prop, "lo": lo, "hi": hi,
                    "decimals": decimals, "save": bool(it.get("save")),
                    "fresh": True,
                    "text": (str(int(val)) if not decimals
                             else ("%.*f" % (decimals, val))),
                }
                _redraw(context, context.area)
                return {"RUNNING_MODAL"}
            if (kind == "prop" and isinstance(arg, tuple)
                    and arg[0] is _prefs()):
                _fire(context, it, mx)
                try:
                    bpy.ops.wm.save_userpref()
                except Exception:
                    pass
                _redraw(context, context.area)
                return {"RUNNING_MODAL"}
            if it["kind"] in {"bar", "ramp"}:
                _state["drag"] = over
            _fire(context, it, mx)
            _redraw(context, context.area)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE":
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
        # Blender ends a modal on its own for plenty of reasons - a file load,
        # a window closing, an error anywhere in the chain. Without this the
        # flag stayed set, the keep-alive saw a live modal that was not there,
        # and the card went on drawing while nothing on it could be clicked.
        if getattr(self, "_token", None) == _state.get("token"):
            _state["modal"] = False

    def invoke(self, context, event):
        if _state["modal"]:
            return {"CANCELLED"}
        _state["modal"] = True
        _state["token"] = _state.get("token", 0) + 1
        self._token = _state["token"]
        _state["beat"] = time.time()
        _state["hover"] = None
        _state["drag"] = None
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}


def _showing():
    """Is the card up somewhere, without needing a draw to ask?"""
    prefs = _prefs()
    if prefs is not None and getattr(prefs, "classic_panel", False):
        return False
    try:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                space = area.spaces.active
                if not getattr(space, "show_region_ui", False):
                    continue
                for r in area.regions:
                    if (r.type == "UI" and r.width >= 60
                            and getattr(r, "active_panel_category", "") == TAB):
                        return True
    except Exception:
        pass
    return False


def _dialog_open(context):
    """A file window (export, open, choose image) or another Blender window
    is up: its clicks are its own, the card must not take them."""
    try:
        wm = context.window_manager
        main = context.window
        for win in wm.windows:
            if win == main:
                continue
            for area in win.screen.areas:
                if area.type == "FILE_BROWSER":
                    return True
    except Exception:
        pass
    return False


def _keep_alive():
    """Start the card's modal, and start it again if it ever drops.

    A modal dies with a file load or a window closing, and the card would go
    quiet with the tab still open. One cheap check a second brings it back
    without anyone noticing it went.
    """
    if _state["quit"]:
        return None
    # A modal that died without saying so. The old test asked whether the card
    # had been painted in the last two seconds, on the reasoning that a fresh
    # paint means the pointer is over it - but a dead modal is exactly what
    # stops anything asking for a paint, so the card that most needed reviving
    # was the one this could never see. An add-on update leaves one behind
    # every time. Now the only question is whether the card is up and whether
    # anything has reached it lately.
    now = time.time()
    if (_state["modal"] and _state["handle"] is not None
            and now - _state["beat"] > 5.0 and _showing()):
        _state["modal"] = False
    # The same check once a second, for the case where the pointer left the
    # viewport before the tab changed and the modal sees no events at all.
    up = _showing()
    if up != _state["was_up"]:
        _state["was_up"] = up
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    if _PLANE["handle"] is None and _state["handle"] is not None:
        try:
            _plane_up()
        except Exception:
            pass
    if not _state["modal"]:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                region = next((r for r in area.regions if r.type == "WINDOW"),
                              None)
                if region is None:
                    continue
                try:
                    with bpy.context.temp_override(window=win, area=area,
                                                   region=region):
                        bpy.ops.img3d.card("INVOKE_DEFAULT")
                except Exception:
                    pass
                break
            if _state["modal"]:
                break
    # Five times a second, not once: this is the only thing watching for a
    # tab change when the pointer is not in the viewport at all, and a whole
    # second of the card being missing is a second of it looking broken. The
    # check itself is a few attribute reads.
    return 0.2


def clear():
    """Take the card down, whoever is asking."""
    if _state["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_state["handle"],
                                                      "WINDOW")
        except Exception:
            pass
    _state["handle"] = None
    bpy.app.driver_namespace.pop(_CARD_HANDLE, None)
    _plane_down()
    _state["widgets"] = []
    _state["hover"] = None
    _state["drag"] = None
    _state["pan"] = None
    _state["bar"] = None
    _state["tip"] = None


def is_on():
    return bool(_state["handle"])


CLASSES = (IMG3D_OT_overlay, IMG3D_PT_tab)


_CARD_HANDLE = "img3d_card_draw_handle"
_PLANE_HANDLE = "img3d_plane_draw_handle"
_PLANE = {"handle": None}


def _plane_quad(obj, frame):
    """The mirror plane as four corners in world space, big enough to cut
    through the whole model and a little over."""
    import numpy as np
    centre, nrm, up = frame
    centre = np.asarray(centre, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-20)
    up = np.asarray(up, dtype=float)
    up = up - float(up @ nrm) * nrm
    n = float(np.linalg.norm(up))
    if n < 1e-9:
        up = np.array([0.0, 0.0, 1.0]) - float(nrm[2]) * nrm
        n = max(float(np.linalg.norm(up)), 1e-20)
    up = up / n
    side = np.cross(nrm, up)
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    if not len(co):
        return None
    rel = co - centre
    # how far the model reaches along the plane's own two directions
    du = float(np.abs(rel @ up).max()) * 1.06 + 1e-6
    dv = float(np.abs(rel @ side).max()) * 1.06 + 1e-6
    corners = [centre - up * du - side * dv,
               centre + up * du - side * dv,
               centre + up * du + side * dv,
               centre - up * du + side * dv]
    mw = obj.matrix_world
    from mathutils import Vector
    return [mw @ Vector((float(c[0]), float(c[1]), float(c[2])))
            for c in corners]


_PLANE_CUT = {}


def _plane_cut(obj, frame):
    """Where the plane actually meets the surface, as a set of segments.

    On a closed model the plane itself is buried inside the mesh and a
    viewer sees nothing of it. The line it cuts along the surface is the
    part that can be seen, and it is the part that means something: it is
    where the two halves meet, and where a seam on the plane would run.

    Worked out once per model and plane, not once per redraw - a draw
    handler runs on every mouse move.
    """
    import numpy as np
    me = obj.data
    centre, nrm, _up = frame
    centre = np.asarray(centre, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-20)
    key = (me.name, len(me.vertices), len(me.edges),
           tuple(np.round(centre, 5).tolist()),
           tuple(np.round(nrm, 5).tolist()))
    hit = _PLANE_CUT.get(key)
    if hit is not None:
        return hit
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    d = (co - centre) @ nrm
    a, b = ev[:, 0], ev[:, 1]
    da, db = d[a], d[b]
    cross = (da > 0) != (db > 0)
    pts = []
    if cross.any():
        t = da[cross] / np.where(np.abs(da[cross] - db[cross]) > 1e-20,
                                 da[cross] - db[cross], 1e-20)
        pa, pb = co[a[cross]], co[b[cross]]
        pts = pa + (pb - pa) * t[:, None]
    # the crossings joined up face by face: every face the plane passes
    # through has exactly two of them, and those two are one segment
    segs = []
    if len(pts):
        eidx = np.flatnonzero(cross)
        where = {int(e): i for i, e in enumerate(eidx.tolist())}
        start = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_start", start)
        count = np.empty(len(me.polygons), dtype=np.int64)
        me.polygons.foreach_get("loop_total", count)
        le = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("edge_index", le)
        for f in range(len(me.polygons)):
            s0, c0 = int(start[f]), int(count[f])
            got = [where[int(e)] for e in le[s0:s0 + c0].tolist()
                   if int(e) in where]
            for k in range(0, len(got) - 1, 2):
                segs.append((pts[got[k]], pts[got[k + 1]]))
    if len(_PLANE_CUT) > 8:
        _PLANE_CUT.clear()
    _PLANE_CUT[key] = segs
    return segs


def _draw_plane():
    """The mirror plane, drawn where symmetry is actually being taken from.

    A toggle that says "Symmetry" and nothing else leaves the user guessing
    which way the model is being folded and whether it found a plane at all.
    This draws the plane it is using, so both questions are answered by
    looking.
    """
    try:
        context = bpy.context
        st = getattr(context.scene, "img3d", None)
        # Symmetry is the master switch: with it off there is no mirror
        # being taken from anywhere, so there is no plane to draw either.
        if (st is None or not getattr(st, "sym_show", False)
                or not getattr(st, "symmetry", False)):
            return
        obj = context.active_object
        if obj is None or obj.type != 'MESH' or not len(obj.data.vertices):
            return
        import sys
        pkg = sys.modules.get(__package__)
        frame = pkg.model_frame(obj) if pkg is not None else None
        if frame is None:
            return
        quad = _plane_quad(obj, frame)
        if quad is None:
            return
    except Exception:
        return
    # The plane runs through the middle of the model, so on anything closed
    # it is behind the surface from wherever you stand and depth testing
    # hides all of it. It is a guide, not geometry: it is drawn over the
    # model, faintly, with the line where it meets the surface drawn solid
    # on top - that line is the part that says anything.
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_test_set("NONE")
    gpu.state.depth_mask_set(False)
    shader.bind()
    fill = batch_for_shader(shader, "TRIS", {"pos": [tuple(v) for v in quad]},
                            indices=[(0, 1, 2), (0, 2, 3)])
    shader.uniform_float("color", (0.20, 0.62, 1.00, 0.09))
    fill.draw(shader)
    gpu.state.line_width_set(1.5)
    edge = batch_for_shader(
        shader, "LINES",
        {"pos": [tuple(quad[i]) for i in (0, 1, 1, 2, 2, 3, 3, 0)]})
    shader.uniform_float("color", (0.35, 0.72, 1.00, 0.55))
    edge.draw(shader)
    try:
        segs = _plane_cut(obj, frame)
    except Exception:
        segs = []
    if segs:
        mw = obj.matrix_world
        from mathutils import Vector
        pos = []
        for pa, pb in segs:
            pos.append(tuple(mw @ Vector((float(pa[0]), float(pa[1]),
                                          float(pa[2])))))
            pos.append(tuple(mw @ Vector((float(pb[0]), float(pb[1]),
                                          float(pb[2])))))
        gpu.state.line_width_set(2.5)
        line = batch_for_shader(shader, "LINES", {"pos": pos})
        shader.uniform_float("color", (0.30, 0.78, 1.00, 1.0))
        line.draw(shader)
    gpu.state.line_width_set(1.0)
    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set("NONE")
    gpu.state.blend_set("NONE")


def _drop_stale_plane():
    ns = bpy.app.driver_namespace
    old = ns.pop(_PLANE_HANDLE, None)
    if old is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
        except Exception:
            pass


def _plane_up():
    _drop_stale_plane()
    _PLANE["handle"] = bpy.types.SpaceView3D.draw_handler_add(
        _draw_plane, (), "WINDOW", "POST_VIEW")
    bpy.app.driver_namespace[_PLANE_HANDLE] = _PLANE["handle"]


def _plane_down():
    if _PLANE["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_PLANE["handle"],
                                                      "WINDOW")
        except Exception:
            pass
    _PLANE["handle"] = None
    bpy.app.driver_namespace.pop(_PLANE_HANDLE, None)


def _drop_stale_card():
    """Remove a card handler left behind by an earlier load of this module."""
    ns = bpy.app.driver_namespace
    old = ns.pop(_CARD_HANDLE, None)
    if old is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
        except Exception:
            pass


def register():
    # A thumbnail that could not be found once is remembered as missing for
    # the rest of the session, which is right while the add-on sits still
    # and wrong the moment it is updated: the new poster is on disk and the
    # popup still draws an empty frame. Coming up is exactly when to forget.
    forget_posters()
    for c in CLASSES:
        bpy.utils.register_class(c)
    print("[Image to 3D] card registered, sidebar tab: %s" % TAB)
    retint()
    _state["quit"] = False
    # The only reference to a draw handler used to live in this module, so a
    # reload that lost the module lost the handler with it - and the old card
    # went on painting underneath the new one, two cards deep. The handle is
    # kept somewhere that outlives the module instead, and any handle found
    # there is taken down before a new one goes up.
    _drop_stale_card()
    _plane_up()
    _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
        _draw, (), "WINDOW", "POST_PIXEL")
    bpy.app.driver_namespace[_CARD_HANDLE] = _state["handle"]
    if not bpy.app.timers.is_registered(_keep_alive):
        bpy.app.timers.register(_keep_alive, first_interval=0.5,
                                persistent=True)


def unregister():
    # A handler left drawing after the module is gone cannot be reached to
    # switch off, so the card comes down before anything is unregistered.
    _state["quit"] = True
    try:
        if bpy.app.timers.is_registered(_keep_alive):
            bpy.app.timers.unregister(_keep_alive)
    except Exception:
        pass
    try:
        clear()
    except Exception:
        pass
    try:
        forget_posters()
    except Exception:
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The hello
#   Said once, the first time the add-on comes up. Blender's own dialog is
#   grey boxes and a Cancel button, which is not what the rest of this looks
#   like, so it is drawn here with the card's own paint: the same charcoal
#   panel, the same accent, the same rounded corners. It is one panel in the
#   middle of the viewport with one button on it.
# ---------------------------------------------------------------------------

WELCOME = {"on": False, "handle": None, "region": None, "rects": {},
           "hover": None}

WELCOME_TITLE = "Thank you for choosing Image to 3D 2"
WELCOME_LINES = (
    "Your feedback is what shapes this add-on.",
    "Any thought, question or idea is very welcome,",
    "and I read every one.",
    "",
    "If a model will not generate, press Save Report",
    "on the Image to 3D 2 panel. It saves a report file",
    "on your Desktop. Send it to me and I will fix it",
    "for your system.",
    "",
    "Come and say hello on Discord.",
)
W_W = 420.0
W_H = 356.0


def _welcome_layout(region, scale):
    """Where the panel and its two buttons sit, in unscaled pixels."""
    w = W_W
    h = W_H
    x = (region.width / scale - w) * 0.5
    y = (region.height / scale - h) * 0.5
    panel = (x, y, w, h)
    btn = (x + 24.0, y + 34.0, w - 48.0, 38.0)
    close = (x + w - 34.0, y + h - 34.0, 22.0, 22.0)
    return panel, btn, close


def _in_rect(rect, px, py):
    x, y, w, h = rect
    return x <= px <= x + w and y <= py <= y + h


def _draw_welcome():
    if not WELCOME["on"]:
        return
    context = bpy.context
    region = context.region
    if region is None or region.as_pointer() != WELCOME["region"]:
        return
    scale = context.preferences.system.ui_scale or 1.0
    panel, btn, close = _welcome_layout(region, scale)
    WELCOME["rects"] = {"panel": panel, "btn": btn, "close": close}
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    gpu.state.blend_set("ALPHA")
    shader.bind()
    px, py, pw, ph = panel
    _fill(shader, (px - 1.0) * scale, (py - 1.0) * scale, (pw + 2.0) * scale,
          (ph + 2.0) * scale, (R_CARD + 1.0) * scale, EDGE)
    # Solid, not the card's slight transparency: this one is read, not
    # glanced at, and a model showing through the words costs more than the
    # glass look is worth. Lifted off the card's near black towards the grey
    # the buttons are painted in, with the same soft light down it, so it
    # reads as one of this add-on's own surfaces rather than a black hole in
    # the viewport.
    def grey(c, k=0.85):
        return (c[0] * k, c[1] * k, c[2] * k, 1.0)
    _fill_grad(shader, px * scale, py * scale, pw * scale, ph * scale,
               R_CARD * scale, grey(CTRL_BOT), grey(CTRL_TOP))
    # the mark, so it is plainly this add-on speaking
    lx, ly = px + 24.0, py + ph - 54.0
    _fill_grad(shader, lx * scale, ly * scale, 30.0 * scale, 30.0 * scale,
               R_LOGO * scale, ACCENT_BOT, ACCENT_TOP)
    _glyph(shader, "logo", lx + 30.0 * 0.19, ly + 30.0 * 0.19, 30.0 * 0.62,
           ON_ACCENT, scale)
    # the one button
    hot = WELCOME["hover"]
    _fill_grad(shader, btn[0] * scale, btn[1] * scale, btn[2] * scale,
               btn[3] * scale, R_CTL * scale,
               ACCENT_BOT if hot != "btn" else ACCENT_TOP,
               ACCENT_TOP if hot != "btn" else ACCENT_TOP)
    # the way out
    cx, cy, cw, ch = close
    _glyph(shader, "close", cx + 4.0, cy + 4.0, 14.0,
           TEXT if hot == "close" else TEXT_DIM, scale)

    _text(WELCOME_TITLE, (px + 64.0) * scale, (py + ph - 44.0) * scale, 15,
          TEXT, scale, bold=True)
    ty = py + ph - 88.0
    for line in WELCOME_LINES:
        if line:
            _text(line, (px + 24.0) * scale, ty * scale, 12,
                  (0.74, 0.74, 0.76, 1.0), scale)
        ty -= 20.0
    _text("Join the Discord", (btn[0] + btn[2] * 0.5) * scale,
          (btn[1] + 13.0) * scale, 13, ON_ACCENT, scale,
          mid=(btn[0] + btn[2] * 0.5) * scale, bold=True)
    _text("Milad Kambari - 3DRedbox Studio", (px + 24.0) * scale,
          (py + 14.0) * scale, 11, TEXT_DIM, scale)
    gpu.state.blend_set("NONE")


def welcome_show(context):
    if WELCOME["on"]:
        return False
    region = getattr(context, "region", None)
    if region is None:
        return False
    WELCOME["region"] = region.as_pointer()
    WELCOME["hover"] = None
    WELCOME["on"] = True
    if WELCOME["handle"] is None:
        WELCOME["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw_welcome, (), "WINDOW", "POST_PIXEL")
    return True


def welcome_hide():
    WELCOME["on"] = False
    WELCOME["hover"] = None
    if WELCOME["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(WELCOME["handle"],
                                                      "WINDOW")
        except Exception:
            pass
    WELCOME["handle"] = None


def welcome_hit(context, event):
    """What the pointer is on: "btn", "close", or None."""
    region = getattr(context, "region", None)
    if region is None:
        return None
    scale = context.preferences.system.ui_scale or 1.0
    panel, btn, close = _welcome_layout(region, scale)
    px = event.mouse_region_x / scale
    py = event.mouse_region_y / scale
    if _in_rect(close, px, py):
        return "close"
    if _in_rect(btn, px, py):
        return "btn"
    if _in_rect(panel, px, py):
        return "panel"
    return None


def focus_downloads(group):
    """Open the Downloads section and light up one model's row."""
    _state["tab"] = "dl"
    _state["open"].add("downloads")
    _state["dl_focus"] = (group, time.time())

def _draw_viewmap(shader, x, y, w, h, have, scale):
    """Seen from above: the object in the middle, facing the Front camera,
    and where the Back, Left and Right pictures are taken from. A camera
    that already has its picture is drawn in the accent colour."""
    if not _inside(y * scale, h * scale):
        return
    _fill(shader, x * scale, y * scale, w * scale, h * scale, 8.0 * scale,
          PANEL_BOT)
    cx, cy = x + w * 0.5, y + h * 0.5 - 2.0
    off = (0.50, 0.51, 0.55, 1.0)
    cams = {"Front": (cx, y + h - 15.0, 0.0, -1.0),
            "Back": (cx, y + 13.0, 0.0, 1.0),
            "Left": (x + 52.0, cy, 1.0, 0.0),
            "Right": (x + w - 52.0, cy, -1.0, 0.0)}
    on = dict(zip(("Front", "Back", "Left", "Right"), have))
    for name, (px, py, dx, dy) in cams.items():
        col = _ac() if on[name] else off
        # line of sight, dashed, from the camera to the object
        line = (col[0], col[1], col[2], 0.45)
        sx, sy = px + dx * 10.0, py + dy * 8.0
        ex, ey = cx - dx * 17.0, cy - dy * 17.0
        n = 5
        for i in range(n):
            a = i / float(n)
            b = a + 0.5 / n
            _seg(shader, (sx + (ex - sx) * a, sy + (ey - sy) * a),
                 (sx + (ex - sx) * b, sy + (ey - sy) * b), 1.5, line, scale)
        # the camera: a body and a lens pointing at the object
        bw, bh = (16.0, 11.0) if dy else (11.0, 16.0)
        _fill(shader, (px - bw * 0.5) * scale, (py - bh * 0.5) * scale,
              bw * scale, bh * scale, 2.5 * scale, col)
        lw, lh = (7.0, 4.0) if dy else (4.0, 7.0)
        lx = px + dx * (bw * 0.5 + 1.5) - lw * 0.5
        ly = py + dy * (bh * 0.5 + 1.5) - lh * 0.5
        _fill(shader, lx * scale, ly * scale, lw * scale, lh * scale,
              1.0 * scale, col)
    # the object, and the way its face points (towards Front)
    _dot(shader, cx, cy, 14.0, (0.30, 0.31, 0.34, 1.0), scale)
    _ring(shader, cx, cy, 14.0, 1.5, (0.62, 0.63, 0.67, 1.0), scale)
    _seg(shader, (cx, cy + 2.0), (cx, cy + 12.0), 3.0, _ac(), scale)
    _dot(shader, cx - 5.0, cy + 6.0, 1.6, (0.85, 0.86, 0.9, 1.0), scale)
    _dot(shader, cx + 5.0, cy + 6.0, 1.6, (0.85, 0.86, 0.9, 1.0), scale)
    # names
    for name, (px, py, dx, dy) in cams.items():
        col = TEXT if on[name] else TEXT_DIM
        if name == "Front":
            _text("Front", (px + 14.0) * scale, (py - 4.0) * scale, 10, col,
                  scale)
        elif name == "Back":
            _text("Back", (px + 14.0) * scale, (py - 4.0) * scale, 10, col,
                  scale)
        elif name == "Left":
            _text("Left", 0, (py - 4.0) * scale, 10, col, scale,
                  right=(px - 10.0) * scale)
        else:
            _text("Right", (px + 10.0) * scale, (py - 4.0) * scale, 10, col,
                  scale)
    _text("seen from above", (x + 7.0) * scale, (y + h - 13.0) * scale, 9,
          TEXT_DIM, scale)
