# -*- coding: utf-8 -*-
"""GPU-карточка NCP поверх вьюпорта — тот же приём, что у карточки
Image to 3D (vendor/img3d/card.py), но под конвейер персонажа.

Пользователь sees: тёмная карточка с оранжевым акцентом (#ff9314), прибитая
к вкладке «NCP» сайдбара: статус конвейера с трекером шагов, вкладки
«Модель / Скелет / Поза», переключатель T-Поза/Анимация, запись по слоям,
рычаги. Классическая N-панель остаётся как запасной путь: карточка
выключается одной галкой в Preferences или кнопкой «панель» на самой
карточке.

Как это устроено (решения из card.py, проверенные на практике):

* ничего из Blender UI: плоские скруглённые прямоугольники через ``gpu``,
  текст через ``blf``, мыш даёт модальный оператор;
* каждое событие: карточка берёт только то, что попало в неё, остальное
  уходит Blender'у (PASS_THROUGH) — орбита, выбор и все шорткаты живут;
* масштаб: всё нарисовано в 1x и умножается на ``ui_scale`` при отрисовке,
  хит-тест читается в непомноженных пикселях;
* прямоугольники заданы от нижнего левого угла, карточка раскладывается
  сверху вниз: строка кладётся в ``y - h``, курсор съедает ``h + gap``;
* клиппинг ручной полосой (_CLIP), не GPU scissor: scissor считается от
  окна, а не от региона, и на экране, где 3D-вид не от низа окна, режет
  не там;
* отрисовка упала — карточка молча уходит, панели возвращаются: аддон не
  может остаться без интерфейса.
"""

from __future__ import annotations

import math
import sys
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader


# -- вид ---------------------------------------------------------------------

TAB = "NCP"
CARD_W = 284.0
TAB_STRIP = 26.0            # свернутый сайдбар: карточка прижимается к нему
PAD = 15.0
GAP = 9.0
R_CARD = 14.0
R_CTL = 8.0
TIP_DELAY = 0.55
TIP_W = 244.0

CARD = (0.106, 0.106, 0.114, 0.97)
CARD_TOP = (0.098, 0.111, 0.118, 0.97)
CARD_BOT = (0.072, 0.084, 0.092, 0.97)
PANEL_TOP = (0.125, 0.138, 0.145, 1.0)
PANEL_BOT = (0.090, 0.102, 0.112, 1.0)
PANEL_EDGE = (0.175, 0.190, 0.200, 1.0)
CTRL_TOP = (0.195, 0.210, 0.220, 1.0)
CTRL_BOT = (0.150, 0.163, 0.174, 1.0)
CTRL_TOP_HOVER = (0.250, 0.265, 0.276, 1.0)
CTRL_BOT_HOVER = (0.200, 0.214, 0.226, 1.0)
EDGE = (1.0, 1.0, 1.0, 0.09)
CONTROL_HOVER = (1.0, 1.0, 1.0, 0.10)
TRACK = (0.215, 0.231, 0.243, 1.0)
RULE = (0.196, 0.212, 0.224, 1.0)
TEXT = (0.910, 0.910, 0.925, 1.0)
TEXT_DIM = (0.478, 0.478, 0.494, 1.0)
TEXT_CHIP = (0.720, 0.720, 0.735, 1.0)
ACCENT = (1.0, 0.576, 0.078, 1.0)
ACCENT_HOVER = (1.0, 0.655, 0.220, 1.0)
ACCENT_EDGE = (0.62, 0.357, 0.048, 1.0)
ACCENT_TOP = (1.0, 0.655, 0.220, 1.0)
ACCENT_BOT = (0.995, 0.545, 0.082, 1.0)
ON_ACCENT = (0.110, 0.063, 0.008, 1.0)
PICKED_TOP = (0.290, 0.190, 0.070, 1.0)
PICKED_BOT = (0.200, 0.135, 0.055, 1.0)
KNOB = (0.980, 0.980, 0.988, 1.0)
GOOD = (0.55, 0.83, 0.44, 1.0)
BAD = (1.0, 0.42, 0.36, 1.0)

_CARD_HANDLE = "ncp_overlay_draw"

_state = {
    "handle": None,
    "modal": False,
    "token": 0,
    "beat": 0.0,
    "widgets": [],
    "card": None,
    "hover": None,
    "tip": None,          # (индекс, ключ, когда указатель остановился)
    "ticking": False,
    "note": None,         # строка, которую карточка говорит в ответ
    "tab": "gen",
    "scroll": 0.0,
    "over": 0.0,
    "was_up": False,
    "mouse": None,
    "regions": {},        # layout каждого 3D-вида
    "hot": None,
    "hot_area": None,
    "drag": None,
    "pan": None,
    "bar": None,
    "painted": 0.0,
    "asked": 0.0,
    "cost": 0.02,
    "dirty": False,
    "relayout": False,
    "quit": False,
    "failed": False,
}


def _pkg():
    """Модуль аддона (там JOB, STEPS, _step_flags...); None вне Blender."""
    return sys.modules.get(__package__) if __package__ else None


def _prefs():
    try:
        addons = bpy.context.preferences.addons
    except Exception:
        return None
    for key in addons.keys():
        if key.rsplit(".", 1)[-1].lower() == "neuro_character_pipeline":
            try:
                return addons[key].preferences
            except Exception:
                return None
    return None


def overlay_enabled():
    """Галка «GPU-карточка» в Preferences. False = классическая N-панель."""
    p = _prefs()
    if p is None:
        return True
    try:
        return bool(getattr(p, "overlay", True))
    except Exception:
        return True


def panels_off():
    """True, когда карточка живая и занимает вкладку: N-панели NCP прячутся."""
    return (_state["handle"] is not None and not _state["failed"]
            and overlay_enabled())


def _visible(context, area=None):
    """Карточка поднята ровно тогда, когда активна вкладка NCP сайдбара."""
    if not overlay_enabled():
        return False
    if area is None:
        area = getattr(context, "area", None)
    if area is None or area.type != "VIEW_3D":
        return False
    space = area.spaces.active
    if not getattr(space, "show_region_ui", False):
        return False
    for region in area.regions:
        if region.type == "UI":
            if region.width < 60:
                return False
            return getattr(region, "active_panel_category", "") == TAB
    return False


def register_draw():
    if _state["handle"] is None:
        _drop_stale_card()
        _state["handle"] = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), "WINDOW", "POST_PIXEL")
        try:
            bpy.app.driver_namespace[_CARD_HANDLE] = _state["handle"]
        except Exception:
            pass
    _state["quit"] = False


def _drop_stale_card():
    """Хэндл от прошлогоregisters (reload модуля) снять до нового."""
    try:
        old = bpy.app.driver_namespace.get(_CARD_HANDLE)
    except Exception:
        old = None
    if old is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(old, "WINDOW")
        except Exception:
            pass
        try:
            bpy.app.driver_namespace.pop(_CARD_HANDLE, None)
        except Exception:
            pass


def clear():
    """Снять карточку, кто бы ни звал."""
    if _state["handle"] is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_state["handle"],
                                                      "WINDOW")
        except Exception:
            pass
    _state["handle"] = None
    try:
        bpy.app.driver_namespace.pop(_CARD_HANDLE, None)
    except Exception:
        pass
    _state["widgets"] = []
    _state["hover"] = None
    _state["tip"] = None
    _state["modal"] = False
    _state["was_up"] = False


def is_on():
    return (_state["handle"] is not None and not _state["quit"]
            and overlay_enabled())


def say(text):
    """Строка на карточке на несколько секунд. Путь назад для оператора,
    запущенного таймером: его report попадает в Info, куда никто не смотрит."""
    _state["note"] = (str(text), time.time())
    try:
        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()
    except Exception:
        pass


# -- примитивы ---------------------------------------------------------------

def _round_rect(x, y, w, h, r, steps=5):
    """Треугольники скруглённого прямоугольника: один веер вокруг середины."""
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


_CLIP = [None]     # (y0, y1) в реальных пикселях, или None


def _clip_y(y, h):
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
    band = _CLIP[0]
    return band is None or (y >= band[0] and y + h <= band[1])


def _fill(shader, x, y, w, h, r, color):
    if w <= 0.0 or h <= 0.0:
        return
    cut = _clip_y(y, h)
    if cut is None:
        return
    if cut != (y, h):
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
    """Скруглённый прямоугольник, подсвеченный сверху."""
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
        y0, hh = cut
        pts = [(p[0], min(max(p[1], y0), y0 + hh)) for p in pts]
    cols = []
    for p in pts:
        k = 0.0 if h <= 1e-6 else max(0.0, min(1.0, (p[1] - y) / h))
        cols.append((c_bot[0] + (c_top[0] - c_bot[0]) * k,
                     c_bot[1] + (c_top[1] - c_bot[1]) * k,
                     c_bot[2] + (c_top[2] - c_bot[2]) * k,
                     c_bot[3] + (c_top[3] - c_bot[3]) * k))
    batch = batch_for_shader(sh, "TRIS", {"pos": pts, "color": cols})
    sh.bind()
    batch.draw(sh)
    shader.bind()


def _text(s, x, y, size, color, scale, right=None, mid=None, bold=False):
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
        blf.position(0, x + max(0.6, 0.5 * scale), y, 0)
        blf.draw(0, s)


def _text_w(text, size, scale):
    try:
        blf.size(0, size * scale)
        return blf.dimensions(0, text)[0] / scale
    except Exception:
        return len(text) * size * 0.55


def _fit(text, room, size, scale, bold=False, floor=9):
    while size > floor and _text_w(text, size, scale) > room:
        size -= 1
    return size


def _wrap(text, width, size, scale):
    words = str(text).split()
    lines, cur = [], ""
    for wd in words:
        probe = (cur + " " + wd).strip()
        if cur and _text_w(probe, size, scale) > width:
            lines.append(cur)
            cur = wd
        else:
            cur = probe
    if cur:
        lines.append(cur)
    return lines or [""]


def _seg(shader, a, b, width, color, scale):
    batch = batch_for_shader(shader, "LINE_STRIP",
                             {"pos": [(a[0] * scale, a[1] * scale),
                                      (b[0] * scale, b[1] * scale)]})
    shader.uniform_float("color", color)
    try:
        gpu.state.line_width_set(width * scale)
    except Exception:
        pass
    batch.draw(shader)
    try:
        gpu.state.line_width_set(1.0)
    except Exception:
        pass


def _dot(shader, px, py, r, color, scale, steps=14):
    pts = []
    for i in range(steps + 1):
        a = math.pi * 2.0 * i / steps
        pts.append((px + math.cos(a) * r * scale,
                    py + math.sin(a) * r * scale))
    batch = batch_for_shader(shader, "TRIS", {"pos": pts})
    shader.uniform_float("color", color)
    batch.draw(shader)


def _ring(shader, cx, cy, r, width, color, scale, steps=18):
    pts = []
    for i in range(steps + 1):
        a = math.pi * 2.0 * i / steps
        pts.append((cx + math.cos(a) * r * scale,
                    cy + math.sin(a) * r * scale))
    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": pts})
    shader.uniform_float("color", color)
    try:
        gpu.state.line_width_set(width * scale)
    except Exception:
        pass
    batch.draw(shader)
    try:
        gpu.state.line_width_set(1.0)
    except Exception:
        pass


def _cross(shader, cx, cy, s, color, scale):
    _seg(shader, (cx - s, cy - s), (cx + s, cy + s), 1.4, color, scale)
    _seg(shader, (cx - s, cy + s), (cx + s, cy - s), 1.4, color, scale)


def _logo_glyph(shader, x, y, s, color, scale):
    """Нейрон: ядро и отростки — внутри оранжевого квадрата логотипа."""
    cx, cy = x + s * 0.5, y + s * 0.5
    _dot(shader, cx, cy, s * 0.13, color, scale)
    for k in range(5):
        a = math.pi * 2.0 * k / 5.0 + 0.35
        r0, r1 = s * 0.20, s * 0.40
        _seg(shader, (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
             (cx + math.cos(a) * r1, cy + math.sin(a) * r1), 1.5, color, scale)
        _dot(shader, cx + math.cos(a) * r1, cy + math.sin(a) * r1,
             s * 0.055, color, scale)


# -- подсказки ---------------------------------------------------------------

TIPS = {
    "gen": ("FULL CYCLE",
            "Всё одной кнопкой: картинка → меш → скелет (+ мышцы, если "
            "X-Muscle включён). Каждый шаг можно запустить и отдельно.",
            None),
    "gimg": ("Generate Image",
             "Промпт уходит в workflow ComfyUI, готовая картинка "
             "запоминается как исходник для меша. Workflow задаётся в "
             "классической панели (раздел 2).", None),
    "gmesh": ("Generate Mesh",
              "Image to 3D строит меш с текстурой из картинки. Первый раз "
              "нужно установить движок и скачать модели (вкладка "
              "«Модель» классической панели или Downloads у img3d).", None),
    "pickimg": ("Pick Image",
                "Взять готовый PNG/JPG с диска вместо генерации: "
                "результат ComfyUI, рендер, фото.", None),
    "quality": ("Качество",
                "Быстро — 512, быстро для проб; Высокое — 1024, рабочий "
                "вариант; Ультра — 1536, самый мелкий-detail и самая "
                "медленная.", None),
    "texture": ("Текстура",
                "Разрешение текстуры на меше: 1K/2K/4K.", None),
    "engine": ("Движок",
               "TRELLIS — быстрый и точный; Hunyuan — богаче текстура, "
               "медленнее и требует больше видеопамяти.", None),
    "rig": ("Build Skeleton",
            "Скелет строится по геометрии меша: сам считает ноги, руки/"
            "крылья, хвост, шею и пальцы. Меш остаётся в своих "
            "пропорциях, весы назначаются автоматически.", None),
    "orient": ("Auto-orient",
               "Поставить меш сам: подъём из лежачего положения и доворот "
               "лицом к +X. Носки и конечности строятся по факту тела.",
               None),
    "autor": ("Auto-Rig",
              "Универсальный разбор тела любого плана. Выключите, чтобы "
              "вернуть старый режим: меш масштабируется под скелет 1.70 м.",
              None),
    "fingers": ("Пальцы",
                "Добавить кости пальцев: рука — 10 костей; у существ "
                "пальцы ищутся по геометрии кисти/лапы/крыла.", None),
    "marks": ("Маркеры точек",
              "Пустышки в найденных точках тела (промежность, плечи, "
              "шея) — для контроля подгонки.", None),
    "plan": ("План тела",
             "Авто определяет план сам. Человек — контракт 18 костей "
             "(мышцы и neuro-анимация работают). Остальные планы — свой "
             "скелет той же схемы: 4 ноги, крылья, птица, дракон.", None),
    "mode": ("T-Поза / Анимация",
             "T-Поза: IK выключен, поза сброшена в rest — персонаж как "
             "построен. Анимация: IK включён, тяните рычаги — персонаж "
             "двигается.", None),
    "rec": ("Автозапись",
            "Пока включено, любое изменение позы рычагов включённых "
            "слоёв само ставит кейфрейм на текущем кадре. Скролл кадров "
            "не пишет, дубли не ставит.", None),
    "layers": ("Слои записи",
               "Тело = Root/Body/Head/Tail. Руки, ноги, пальцы (оранжевые "
               "диски), крылья — рычаги каждого слоя пишутся только если "
               "слой включён.", None),
    "keyframe": ("+ Кадр",
                 "Записать текущую позу рычагов включённых слоёв кейфреймом "
                 "на текущий кадр вручную.", None),
    "ik": ("IK-рычаги",
           "Рычаги рук/ног/крыльев двигают конечности через IK. "
           "Импорт Neuro Motion выключает IK сам.", None),
    "poles": ("Полюсы коленей/локтей",
              "Рычаги-кольца у колена и локтя: направление сгиба сустава. "
              "Тяните полюс — сустав разворачивается, IK не щёлкает.",
              None),
    "fhands": ("Рычаги пальцев",
               "Оранжевый диск на каждый палец: тяните — палец сгибается "
               "за мышью. Рычаги едут с кистью и не деформируют меш.",
               None),
    "hide": ("Скрыть кости скелета",
             "В вьюпорте остаются только крупные рычаги. Они не "
             "деформируют меш и в анимации костей не участвуют.", None),
    "handles": ("Выделить / Пересобрать",
                "Выделить — все рычаги и Pose mode. Пересобрать — слой "
                "рычагов заново (например, риг от старой версии аддона).",
                None),
    "cancel": ("Отмена",
               "Остановить текущий шаг конвейера. Ничего не удаляется.",
               None),
    "classic": ("Классическая панель",
                "Вернуть обычную N-панель. Обратно — галка «GPU-карточка "
                "вьюпорта» в Preferences аддона.", None),
}

WHY = {
    "ncp.build_skeleton": "нет меша персонажа: сгенерируйте или выделите",
    "ncp.generate_image": "нужен workflow ComfyUI в API-формате (раздел 2)",
    "ncp.generate_mesh": "Image to 3D не готов: установите движок/модели",
    "ncp.set_mode": "сначала постройте скелет (вкладка «Скелет»)",
    "ncp.record_key": "сначала постройте скелет (вкладка «Скелет»)",
    "ncp.select_handles": "нет рычагов: постройте скелет или Пересобрать",
    "ncp.rebuild_handles": "сначала постройте скелет (вкладка «Скелет»)",
}


# -- раскладка ---------------------------------------------------------------

def _settings(context):
    return getattr(context.scene, "ncp", None)


def _build(context, region):
    """Разложить карточку; вернуть виджеты и её прямоугольник (в 1x-пикселях)."""
    st = _settings(context)
    if st is None:
        return [], (0.0, 0.0, 0.0, 0.0)
    pkg = _pkg()
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
    inner = CARD_W - PAD * 2.0 - 4.0
    left = x + PAD + 2.0
    y = [top - PAD + _state["scroll"]]

    def put(kind, h, gap=GAP, **kw):
        kw["kind"] = kind
        kw["rect"] = (left, y[0] - h, inner, h)
        w.append(kw)
        y[0] -= h + gap
        return kw

    def label(text, size=12, color=TEXT, gap=17.0):
        w.append({"kind": "label", "rect": (left, y[0] - 13.0, 0, 0),
                  "text": text, "size": size, "color": color})
        y[0] -= gap

    def clabel(text, col, size=11):
        w.append({"kind": "clabel", "rect": (left, y[0] - 13.0, 0, 0),
                  "text": text, "size": size, "color": col})
        y[0] -= 16.0

    def hint(text):
        for line in _wrap(text, inner, 10, scale):
            w.append({"kind": "hint", "rect": (left, y[0] - 11.0, 0, 0),
                      "text": line})
            y[0] -= 13.0
        y[0] -= 2.0

    def chip_row(items, h=26.0):
        n = len(items)
        gp = 6.0
        cw = (inner - gp * (n - 1)) / n
        for k, (name, on, tip, act) in enumerate(items):
            w.append({"kind": "chip",
                      "rect": (left + k * (cw + gp), y[0] - h, cw, h),
                      "text": name, "on": on, "tip": tip, "act": act,
                      "tsize": 11 if n >= 4 else 12})
        y[0] -= h + GAP

    def switch(text, owner, prop, tip=None):
        if owner is None or not hasattr(owner, prop):
            return
        put("switch", 24.0, gap=4.0, text=text,
            on=bool(getattr(owner, prop)),
            act=("prop", (owner, prop)), tip=tip)

    def btn(text, op_id, h=30.0, tip=None, props=None):
        put("btn", h, text=text, act=("op", op_id), tip=tip,
            props=props or {})

    def run(text, op_id, h=40.0, tip=None):
        put("run", h, text=text, act=("op", op_id), tip=tip)

    def rule():
        put("rule", 1.0, gap=GAP + 2.0)

    # ---- шапка ---------------------------------------------------------------
    w.append({"kind": "logo", "rect": (left, y[0] - 30.0, 30.0, 30.0)})
    w.append({"kind": "title", "rect": (left + 40.0, y[0] - 15.0, 0, 0),
              "text": "Neuro Character", "room": inner - 92.0})
    w.append({"kind": "sub", "rect": (left + 40.0, y[0] - 29.0, 0, 0),
              "text": "Картинка → меш → скелет → анимация",
              "room": inner - 44.0})
    w.append({"kind": "linktext",
              "rect": (left + inner - 50.0, y[0] - 16.0, 50.0, 14.0),
              "text": "панель", "act": ("classic",), "tip": "classic"})
    y[0] -= 30.0 + GAP + 2.0

    # ---- вкладки -------------------------------------------------------------
    tab_ = _state.get("tab", "gen")
    cw = (inner - 12.0) / 3.0
    for k, (name, code) in enumerate((("Модель", "gen"),
                                      ("Скелет", "rig"),
                                      ("Поза", "pose"))):
        w.append({"kind": "chip",
                  "rect": (left + k * (cw + 6.0), y[0] - 28.0, cw, 28.0),
                  "text": name, "on": tab_ == code, "tip": None,
                  "act": ("tab", code)})
    y[0] -= 28.0 + GAP

    # ---- статус конвейера ----------------------------------------------------
    job = getattr(pkg, "JOB", None) if pkg else None
    job_active = bool(job and job.get("active"))
    phase = (job.get("phase") if job else "") or ""
    phase_ru = getattr(pkg, "PHASE_RU", {}) if pkg else {}
    if st.state == "ERROR":
        state_text, state_col = "Ошибка", BAD
    elif st.state == "DONE":
        state_text, state_col = "Готово", GOOD
    elif st.state in phase_ru:
        sec = max(0, int(time.time() - (job.get("t0", time.time())
                                        if job else time.time())))
        state_text, state_col = "%s  (%d:%02d)" % (phase_ru[st.state],
                                                   sec // 60, sec % 60), ACCENT
    else:
        state_text, state_col = "Конвейер персонажа", TEXT

    lines = []
    if st.status:
        col = BAD if st.state == "ERROR" else TEXT_DIM
        lines = [(t, col) for t in _wrap(st.status, inner - 14.0, 10.5,
                                         scale)[:5]]

    steps = getattr(pkg, "STEPS", ()) if pkg else ()
    tstates = None
    if steps and pkg is not None:
        try:
            flags = pkg._step_flags(context, st)
            tstates = pkg._tracker_states(flags, phase)
        except Exception:
            tstates = None
    if tstates is None:
        tstates = [("pending", key) for key, _n, _i in steps]

    h_stat = 10.0 + 18.0 + (len(lines) * 13.0 + 3.0 if lines else 0.0) \
        + 30.0 + 8.0
    stat = put("stat", h_stat, gap=GAP, state_text=state_text,
               state_col=state_col, lines=lines, tracker=tstates)
    if job_active:
        rx, ry, rw, rh = stat["rect"]
        w.append({"kind": "xbtn", "over": True,
                  "rect": (rx + rw - 21.0, ry + rh - 23.0, 16.0, 16.0),
                  "act": ("op", "ncp.cancel"), "tip": "cancel"})

    # ошибки регистрации/вендора видны и на карточке (в классической
    # панели они на месте, но карточка занимает вкладку первой)
    try:
        _warns = []
        if pkg is not None:
            _warns.extend(list(getattr(pkg, "REGISTER_ERRORS", []))[:2])
            _warns.extend(list(pkg._vendor_warnings())[:1])
        for _t in _warns:
            if _t:
                clabel(("Внимание: " + str(_t))[:44], BAD, size=10)
    except Exception:
        pass

    # ---- содержимое вкладки --------------------------------------------------
    arm = bpy.data.objects.get("NeuroCharacter")
    if tab_ == "gen":
        run("FULL CYCLE", "ncp.run_all", tip="gen")
        hint("Всё одной кнопкой: картинка → меш → скелет "
             "(+ мышцы, если X-Muscle включён)")
        label("Картинка (ComfyUI)")
        ok = None
        if pkg is not None:
            try:
                ok = bool(pkg._comfy_alive_cached())
            except Exception:
                ok = None
        if ok is True:
            clabel("ComfyUI отвечает", GOOD)
        elif ok is False:
            clabel("ComfyUI не запущен", BAD)
        btn("Generate Image", "ncp.generate_image", tip="gimg")
        btn("Pick Image", "ncp.pick_image", h=26.0, tip="pickimg")
        label("Меш (Image to 3D)")
        btn("Generate Mesh", "ncp.generate_mesh", tip="gmesh")
        label("Качество", gap=17.0)
        chip_row([("Быстро", st.quality == "FAST", "quality",
                   ("enum", (st, "quality", "FAST"))),
                  ("Высокое", st.quality == "HIGH", "quality",
                   ("enum", (st, "quality", "HIGH"))),
                  ("Ультра", st.quality == "ULTRA", "quality",
                   ("enum", (st, "quality", "ULTRA")))])
        label("Текстура", gap=17.0)
        chip_row([("1K", st.texture == "1024", "texture",
                   ("enum", (st, "texture", "1024"))),
                  ("2K", st.texture == "2048", "texture",
                   ("enum", (st, "texture", "2048"))),
                  ("4K", st.texture == "4096", "texture",
                   ("enum", (st, "texture", "4096")))])
        label("Движок", gap=17.0)
        chip_row([("TRELLIS", st.engine == "TRELLIS", "engine",
                   ("enum", (st, "engine", "TRELLIS"))),
                  ("Hunyuan", st.engine == "HUNYUAN", "engine",
                   ("enum", (st, "engine", "HUNYUAN")))])
    elif tab_ == "rig":
        run("Build Skeleton", "ncp.build_skeleton", tip="rig")
        switch("Auto-orient", st, "orient_auto", tip="orient")
        switch("Auto-Rig (universal)", st, "auto_rig", tip="autor")
        switch("Пальцы", st, "add_fingers", tip="fingers")
        switch("Маркеры точек", st, "debug_landmarks", tip="marks")
        label("План тела", gap=17.0)
        chip_row([("Авто", st.body_plan == "AUTO", "plan",
                   ("enum", (st, "body_plan", "AUTO"))),
                  ("Человек", st.body_plan == "HUMAN", "plan",
                   ("enum", (st, "body_plan", "HUMAN"))),
                  ("4 ноги", st.body_plan == "QUADRUPED", "plan",
                   ("enum", (st, "body_plan", "QUADRUPED")))])
        chip_row([("Крылатый", st.body_plan == "WINGED_QUAD", "plan",
                   ("enum", (st, "body_plan", "WINGED_QUAD"))),
                  ("Птица", st.body_plan == "BIRD", "plan",
                   ("enum", (st, "body_plan", "BIRD"))),
                  ("Дракон", st.body_plan == "DRAGON", "plan",
                   ("enum", (st, "body_plan", "DRAGON")))])
        hint("Авто сам считает ноги, крылья и пальцы по геометрии; "
             "остальные планы отключают распознавание")
        hint("Ручной доворот Rot X/Z и Spec JSON — в классической "
             "панели (кнопка «панель» в шапке)")
    else:
        if arm is None:
            hint("Скелет ещё не построен")
            hint("Вкладка «Скелет» → Build Skeleton")
        else:
            label("Режим", gap=17.0)
            chip_row([("T-Поза", st.rig_mode == "TPOSE", "mode",
                       ("mode", "TPOSE")),
                      ("Анимация", st.rig_mode == "ANIM", "mode",
                       ("mode", "ANIM"))], h=30.0)
            hint("Анимация: тяните рычаги — персонаж двигается. "
                 "T-Поза: rest, IK выключен")
            rule()
            label("Запись по слоям", gap=17.0)
            switch("Автозапись", st, "rec_auto", tip="rec")
            chip_row([("Тело", st.rec_body, "layers",
                       ("prop", (st, "rec_body"))),
                      ("Руки", st.rec_arms, "layers",
                       ("prop", (st, "rec_arms"))),
                      ("Ноги", st.rec_legs, "layers",
                       ("prop", (st, "rec_legs")))])
            chip_row([("Пальцы", st.rec_fingers, "layers",
                       ("prop", (st, "rec_fingers"))),
                      ("Крылья", st.rec_wings, "layers",
                       ("prop", (st, "rec_wings"))),
                      ("+ Кадр", False, "keyframe",
                       ("op", "ncp.record_key"))])
            hint("Слой Тело = Root/Body/Head/Tail; импорт motion "
                 "выключает автозапись сам")
            rule()
            label("Рычаги", gap=17.0)
            switch("IK-рычаги", st, "ctrl_ik", tip="ik")
            switch("Полюсы коленей/локтей", st, "ctrl_poles", tip="poles")
            switch("Рычаги пальцев", st, "ctrl_fingers", tip="fhands")
            switch("Скрыть кости скелета", st, "ctrl_hide_deform",
                   tip="hide")
            chip_row([("Выделить", False, "handles",
                       ("op", "ncp.select_handles")),
                      ("Пересобрать", False, "handles",
                       ("op", "ncp.rebuild_handles"))], h=28.0)
            hint("Root — весь персонаж; кольца — суставы; оранжевые "
                 "диски — пальцы. Рычаги не деформируют меш")

    # ---- сказать в ответ -----------------------------------------------------
    note = _state.get("note")
    if note is not None and time.time() - note[1] < 10.0:
        for line in _wrap(note[0], inner, 11, scale)[:3]:
            w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
                      "text": line, "size": 11, "color": ACCENT})
            y[0] -= 15.0
        y[0] -= 3.0

    rule()
    ver = "2.10.0"
    try:
        if pkg is not None and getattr(pkg, "bl_info", None):
            ver = ".".join(str(v) for v in pkg.bl_info["version"])
    except Exception:
        pass
    w.append({"kind": "label", "rect": (left, y[0] - 12.0, 0, 0),
              "text": "NCP v%s · blender_neuro" % ver, "size": 10,
              "color": TEXT_DIM})
    y[0] -= 16.0

    bottom = y[0] - PAD
    full = (top + PAD - bottom) + _state["scroll"]
    room = top + PAD - 6.0
    _state["over"] = max(0.0, full - room)
    if (_state["scroll"] > _state["over"] + 0.5
            and not _state.get("relayout")):
        _state["scroll"] = _state["over"]
        _state["relayout"] = True
        try:
            return _build(context, region)
        finally:
            _state["relayout"] = False
    shown = min(full, room)
    card = (x, top + PAD - shown, CARD_W, shown)
    return w, card


def _hit(widgets, mx, my):
    # мелкие кнопки поверх блока статуса находятся раньше самого блока
    order = [i for i, it in enumerate(widgets) if it.get("over")] + \
        [i for i, it in enumerate(widgets) if not it.get("over")]
    for i in order:
        it = widgets[i]
        if it["kind"] in {"label", "logo", "title", "sub", "hint", "rule",
                          "stat", "clabel", "linktext_no"}:
            continue
        x, y, w, h = it["rect"]
        if x <= mx <= x + w and y <= my <= y + h:
            c = _state.get("card")
            if c is not None and not (c[1] - 1.0 <= my <= c[1] + c[3] + 1.0):
                continue
            return i
    return None


# -- отрисовка ---------------------------------------------------------------

def _draw():
    """Отрисовка упала — карточка уходит, панели возвращаются."""
    try:
        _draw_card()
    except Exception:
        import traceback
        traceback.print_exc()
        if not _state.get("failed"):
            _state["failed"] = True
            print("[NCP] карточка не смогла отрисоваться, "
                  "классическая панель возвращена")


def _draw_card():
    context = bpy.context
    region = context.region
    if region is None or not _visible(context):
        return
    widgets, card = _build(context, region)
    if not widgets:
        return
    key = region.as_pointer()
    _state.setdefault("regions", {})[key] = (widgets, card)
    if _state.get("hot") in (None, key) or \
            _state.get("hot") not in _state["regions"]:
        _state["hot"] = key
        _state["widgets"] = widgets
        _state["card"] = card
    now = time.time()
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
        x, y, w, h = rect
        _fill(shader, x * scale, y * scale, w * scale, h * scale, r * scale,
              ACCENT if lit else ACCENT_EDGE)
        _fill_grad(shader, (x + 1.0) * scale, (y + 1.0) * scale,
                   (w - 2.0) * scale, (h - 2.0) * scale,
                   max(0.0, r - 1.0) * scale,
                   ACCENT if lit else CTRL_BOT,
                   ACCENT_HOVER if lit else CTRL_TOP)

    cx, cy, cw, ch = card
    _fill(shader, (cx - 1.0) * scale, (cy - 1.0) * scale,
          (cw + 2.0) * scale, (ch + 2.0) * scale, (R_CARD + 1.0) * scale, EDGE)
    _fill_grad(shader, cx * scale, cy * scale, cw * scale, ch * scale,
               R_CARD * scale, CARD_BOT, CARD_TOP)

    clipped = _state["over"] > 0.0
    _CLIP[0] = ((cy + 1.0) * scale, (cy + ch - 1.0) * scale) if clipped \
        else None

    # фон
    for i, it in enumerate(widgets):
        k = it["kind"]
        x, y, w, h = it["rect"]
        if clipped and (y + h < cy - 2.0 or y > cy + ch + 2.0):
            continue
        lit = (i == hover)
        if k == "logo":
            gbox(it["rect"], 7.0, ACCENT_BOT, ACCENT_TOP)
            _logo_glyph(shader, x, y, min(w, h), ON_ACCENT, scale)
        elif k == "chip":
            if it["on"]:
                gbox(it["rect"], R_CTL, PICKED_BOT, PICKED_TOP)
                x2, y2, w2, h2 = it["rect"]
                _fill(shader, x2 * scale, y2 * scale, w2 * scale,
                      h2 * scale, R_CTL * scale, ACCENT)
                _fill_grad(shader, (x2 + 1.0) * scale, (y2 + 1.0) * scale,
                           (w2 - 2.0) * scale, (h2 - 2.0) * scale,
                           max(0.0, R_CTL - 1.0) * scale,
                           PICKED_BOT, PICKED_TOP)
            else:
                gbox(it["rect"], R_CTL,
                     CTRL_BOT_HOVER if lit else CTRL_BOT,
                     CTRL_TOP_HOVER if lit else CTRL_TOP)
        elif k == "run":
            lead(it["rect"], R_CTL, lit)
        elif k == "btn":
            gbox(it["rect"], R_CTL,
                 CTRL_BOT_HOVER if lit else CTRL_BOT,
                 CTRL_TOP_HOVER if lit else CTRL_TOP)
        elif k == "xbtn":
            if lit:
                box(it["rect"], 8.0, CONTROL_HOVER)
            _cross(shader, x + w * 0.5, y + h * 0.5, 3.0,
                   TEXT if lit else TEXT_DIM, scale)
        elif k == "switch":
            kx = (x + w - 30.0) * scale
            ky = (y + (h - 17.0) * 0.5) * scale
            if it["on"]:
                _fill_grad(shader, kx, ky, 30.0 * scale, 17.0 * scale,
                           8.5 * scale, ACCENT_BOT, ACCENT_TOP)
            else:
                _fill(shader, kx, ky, 30.0 * scale, 17.0 * scale,
                      8.5 * scale, TRACK)
            knob = kx + (14.5 if it["on"] else 2.0) * scale
            _fill(shader, knob, ky + 2.0 * scale, 13.0 * scale,
                  13.0 * scale, 6.5 * scale, KNOB)
        elif k == "stat":
            x2, y2, w2, h2 = it["rect"]
            _fill(shader, x2 * scale, y2 * scale, w2 * scale, h2 * scale,
                  R_CARD * 0.8 * scale, PANEL_EDGE)
            _fill_grad(shader, (x2 + 1.0) * scale, (y2 + 1.0) * scale,
                       (w2 - 2.0) * scale, (h2 - 2.0) * scale,
                       max(0.0, R_CARD * 0.8 - 1.0) * scale,
                       PANEL_BOT, PANEL_TOP)
            # трекер шагов
            tr = it.get("tracker") or ()
            n = max(1, len(tr))
            cell = w2 / n
            dy = y2 + 14.0
            pulse = 0.5 + 0.5 * math.sin(time.time() * 4.0)
            for k2, (tstate, _key) in enumerate(tr):
                dcx = x2 + (k2 + 0.5) * cell
                if tstate == "done":
                    _dot(shader, dcx, dy, 3.4, GOOD, scale)
                elif tstate == "active":
                    _dot(shader, dcx, dy, 3.8, ACCENT, scale)
                    _ring(shader, dcx, dy, 5.6 + pulse * 1.6, 1.2,
                          _ac_alpha(ACCENT, 0.55), scale)
                else:
                    _dot(shader, dcx, dy, 3.0, TRACK, scale)
        elif k == "rule":
            _fill(shader, x * scale, y * scale, w * scale, 1.0 * scale, 0.0,
                  RULE)

    # текст
    for i, it in enumerate(widgets):
        k = it["kind"]
        x, y, w, h = it["rect"]
        if clipped and (y + h < cy - 2.0 or y > cy + ch + 2.0):
            continue
        lit = (i == hover)
        if k == "title":
            size = _fit(it["text"], it.get("room", 999.0), 15, scale,
                        bold=True, floor=11)
            _text(it["text"], x * scale, y * scale, size, TEXT, scale,
                  bold=True)
        elif k == "sub":
            size = _fit(it["text"], it.get("room", 999.0), 10, scale,
                        floor=7)
            _text(it["text"], x * scale, y * scale, size, TEXT_DIM, scale)
        elif k == "label":
            _text(it["text"], x * scale, y * scale, it["size"],
                  it["color"], scale)
        elif k == "hint":
            _text(it["text"], x * scale, y * scale, 10, TEXT_DIM, scale)
        elif k == "clabel":
            _dot(shader, (x + 4.0) * scale, (y + 4.0) * scale, 3.0,
                 it["color"], scale)
            _text(it["text"], (x + 12.0) * scale, y * scale, it["size"],
                  it["color"], scale)
        elif k == "chip":
            size = _fit(it["text"], w - 10.0, it.get("tsize", 12), scale)
            _text(it["text"], 0, (y + (h - 11.0) * 0.5) * scale, size,
                  TEXT if it["on"] else TEXT_CHIP, scale,
                  mid=(x + w * 0.5) * scale)
        elif k == "run":
            size = _fit(it["text"], w - 24.0, 13, scale, bold=True)
            _text(it["text"], 0, (y + (h - 12.0) * 0.5) * scale, size,
                  ON_ACCENT if lit else TEXT, scale,
                  mid=(x + w * 0.5) * scale, bold=True)
        elif k == "btn":
            size = _fit(it["text"], w - 20.0, it.get("tsize", 12), scale)
            _text(it["text"], 0, (y + (h - 11.0) * 0.5) * scale, size,
                  TEXT, scale, mid=(x + w * 0.5) * scale)
        elif k == "switch":
            size = _fit(it["text"], w - 40.0, 12, scale)
            _text(it["text"], x * scale, (y + (h - 11.0) * 0.5) * scale,
                  size, TEXT, scale)
        elif k == "stat":
            x2, y2, w2, h2 = it["rect"]
            _dot(shader, (x2 + 10.0) * scale, (y2 + h2 - 17.0) * scale,
                 3.4, it.get("state_col", TEXT), scale)
            _text(it.get("state_text", ""), (x2 + 18.0) * scale,
                  (y2 + h2 - 21.0) * scale, 12, it.get("state_col", TEXT),
                  scale, bold=True)
            ly = y2 + h2 - 21.0
            for line, col in it.get("lines", ()):
                ly -= 13.0
                _text(line, (x2 + 10.0) * scale, ly * scale, 10.5, col,
                      scale)
            tr = it.get("tracker") or ()
            n = max(1, len(tr))
            cell = w2 / n
            dy = y2 + 14.0
            for k2, (_tstate, key) in enumerate(tr):
                name = ""
                for s_key, s_name, _s_icon in tr_names():
                    if s_key == key:
                        name = s_name
                        break
                _text(name, 0, (dy - 11.0) * scale, 8.5, TEXT_DIM, scale,
                      mid=(x2 + (k2 + 0.5) * cell) * scale)
        elif k == "linktext":
            _text(it["text"], 0, (y + (h - 11.0) * 0.5) * scale, 11,
                  ACCENT if lit else TEXT_DIM, scale,
                  mid=(x + w * 0.5) * scale)

    _CLIP[0] = None
    if clipped:
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


def tr_names():
    """(ключ, имя, иконка) шагов — из аддона, с запасным списком."""
    pkg = _pkg()
    steps = getattr(pkg, "STEPS", None) if pkg else None
    if steps:
        return steps
    return (("sketch", "Sketch", ""), ("image", "Image", ""),
            ("mesh", "Mesh", ""), ("rig", "Rig", ""),
            ("muscles", "Musc", ""), ("anim", "Anim", ""))


def _ac_alpha(c, a):
    return (c[0], c[1], c[2], a)


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
    title, body, _art = found
    wx, wy, ww, wh = widgets[index]["rect"]
    gpu.state.blend_set("ALPHA")
    shader.bind()
    lines = _wrap(body, TIP_W - 24.0, 11, scale)
    h = 16.0 + 16.0 + 6.0 + len(lines) * 15.0 + 14.0
    x = card[0] - TIP_W - 10.0
    if x < 2.0:
        x = card[0] + card[2] + 10.0
    y = min(max(wy + wh - h, 20.0), card[1] + card[3] - h)
    _fill(shader, (x - 1.0) * scale, (y - 1.0) * scale, (TIP_W + 2.0) * scale,
          (h + 2.0) * scale, 13.0 * scale, EDGE)
    _fill(shader, x * scale, y * scale, TIP_W * scale, h * scale,
          12.0 * scale, CARD)
    ty = y + h - 24.0
    _text(title, (x + 12.0) * scale, ty * scale, 12, TEXT, scale, bold=True)
    ty -= 17.0
    for line in lines:
        _text(line, (x + 12.0) * scale, ty * scale, 11, TEXT_DIM, scale)
        ty -= 15.0


def _redraw(context, area=None):
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


def _redraw_all():
    try:
        for win in bpy.context.window_manager.windows:
            for a in win.screen.areas:
                if a.type == "VIEW_3D":
                    a.tag_redraw()
    except Exception:
        pass


def hot_view(context=None):
    """3D-вид под указателем: (window, area, region) — кнопка должна работать
    в том виде, где её нажали, а не там, где живёт модальный оператор."""
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


def _run_later(idname, win, area, region, **props):
    """Оператор запускается таймером: менять сцену из обработчика карточки
    (модального оператора) нельзя — это путь к падению Blender."""
    def go():
        mod, _, name = idname.partition(".")
        op = getattr(getattr(bpy.ops, mod, None), name, None)
        if op is None:
            say("нет оператора %s" % idname)
            return None
        try:
            if win is not None and area is not None:
                with bpy.context.temp_override(window=win, area=area,
                                               region=region):
                    op("INVOKE_DEFAULT", **props)
            else:
                op("INVOKE_DEFAULT", **props)
        except Exception as exc:
            say(str(exc)[:60])
            print("[NCP] card:", exc)
        return None

    try:
        bpy.app.timers.register(go, first_interval=0.0)
    except Exception:
        go()


def _fire_op(context, arg, props):
    mod, _, name = arg.partition(".")
    op = getattr(getattr(bpy.ops, mod, None), name, None)
    if op is None:
        say("нет оператора %s" % arg)
        return
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
        say(WHY.get(arg, "это действие сейчас недоступно"))
        return
    hw, ha, hr = hot_view(context)
    if ha is None:
        hw = getattr(context, "window", None)
        ha = getattr(context, "area", None)
        hr = getattr(context, "region", None)
    _run_later(arg, hw, ha, hr, **props)


def _fire(context, it, mx=None):
    kind, arg = it["act"]
    if kind == "tab":
        _state["tab"] = arg
        _state["scroll"] = 0.0
    elif kind == "prop":
        owner, name = arg
        try:
            setattr(owner, name, not getattr(owner, name))
        except Exception as exc:
            say(str(exc)[:60])
    elif kind == "enum":
        owner, name, code = arg
        try:
            setattr(owner, name, code)
        except Exception as exc:
            say(str(exc)[:60])
    elif kind == "mode":
        _fire_op(context, "ncp.set_mode", {"mode": arg})
    elif kind == "op":
        _fire_op(context, arg, it.get("props") or {})
    elif kind == "classic":
        p = _prefs()
        if p is not None:
            try:
                p.overlay = False
            except Exception:
                pass
        clear_note()
        _redraw_all()


def clear_note():
    _state["note"] = None


def over_card(context, event):
    """Указатель на карточке? Другие модальные операторы спрашивают это
    первыми и пропускают клик, если да."""
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


def _settle():
    """Дорисовать то, что троттлинг пропустил."""
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
    """Перерисовка, пока открыта подсказка (задержка появления — 0.55 с)."""
    if _state["quit"] or _state["tip"] is None:
        _state["ticking"] = False
        return None
    if _state["asked"]:
        return 0.02
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


def _showing():
    """Карточка поднята где-то на экране — без обращения к отрисовке."""
    if not overlay_enabled():
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
                            and getattr(r, "active_panel_category", "")
                            == TAB):
                        return True
    except Exception:
        pass
    return False


def _keep_alive():
    """Запустить модальный оператор карточки и поднимать его, если умер.

    Модальный оператор умирает с загрузкой файла или закрытием окна —
    карточка осталась бы нарисованной, но мёртвой. Дешёвая проверка
    дважды в секунду поднимает его незаметно.
    """
    if _state["quit"]:
        return None
    now = time.time()
    if (_state["modal"] and _state["handle"] is not None
            and now - _state["beat"] > 5.0 and _showing()):
        _state["modal"] = False
    up = _showing()
    if up != _state["was_up"]:
        _state["was_up"] = up
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
    if not _state["modal"] and up:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                region = next((r for r in area.regions
                               if r.type == "WINDOW"), None)
                if region is None:
                    continue
                try:
                    with bpy.context.temp_override(window=win, area=area,
                                                   region=region):
                        bpy.ops.ncp.card("INVOKE_DEFAULT")
                except Exception:
                    pass
                break
            if _state["modal"]:
                break
    return 0.5


# -- оператор мыши -----------------------------------------------------------

class NCP_OT_overlay(bpy.types.Operator):
    """Мышь для карточки. Запускается сам (keep-alive), не по кнопке."""

    bl_idname = "ncp.card"
    bl_label = "NCP Viewport Card"
    bl_options = {"INTERNAL"}

    def modal(self, context, event):
        if getattr(self, "_token", None) != _state.get("token"):
            return {"CANCELLED"}
        _state["beat"] = time.time()
        _state["mouse"] = (event.mouse_x, event.mouse_y)
        if _state["quit"]:
            _state["modal"] = False
            return {"CANCELLED"}
        hot_area = None
        hot_region = None
        busy = _state["drag"] is not None or _state["pan"] is not None \
            or _state["bar"] is not None
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
            if _state["hover"] is not None:
                _state["hover"] = None
                _state["drag"] = None
                _state["tip"] = None
            if _state["was_up"]:
                _state["was_up"] = False
                _redraw(context, getattr(context, "area", None))
            return {"PASS_THROUGH"}
        if not _state["was_up"]:
            _state["was_up"] = True
            _redraw(context, getattr(context, "area", None))

        scale = context.preferences.system.ui_scale or 1.0
        if hot_region is not None:
            mx = (event.mouse_x - hot_region.x) / scale
            my = (event.mouse_y - hot_region.y) / scale
        else:
            mx = event.mouse_region_x / scale
            my = event.mouse_region_y / scale
        widgets = _state["widgets"]
        over = _hit(widgets, mx, my)

        if _state["pan"] is not None:
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

        if _state["bar"] is not None:
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

        if event.type == "MIDDLEMOUSE" and event.value == "PRESS":
            card = _state.get("card")
            if (_state["over"] > 0.0 and card is not None
                    and card[0] <= mx <= card[0] + card[2] + TAB_STRIP
                    and card[1] <= my <= card[1] + card[3]):
                _state["pan"] = (my, _state["scroll"])
                return {"RUNNING_MODAL"}
            return {"PASS_THROUGH"}

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
            card = _state.get("card")
            if (_state["over"] > 0.0 and card is not None
                    and card[0] <= mx <= card[0] + card[2] + TAB_STRIP
                    and card[1] <= my <= card[1] + card[3]):
                if event.type == "TRACKPADPAN":
                    step = float(
                        getattr(event, "mouse_prev_y", event.mouse_y)
                        - event.mouse_y) / scale
                elif event.type in {"PAGE_UP", "PAGE_DOWN"}:
                    if event.value != "PRESS":
                        return {"RUNNING_MODAL"}
                    step = (card[3] - 40.0) * (1.0 if event.type
                                               == "PAGE_DOWN" else -1.0)
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
                        bpy.app.timers.register(_tick,
                                                first_interval=TIP_DELAY)
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
                        bpy.app.timers.register(_settle,
                                                first_interval=wait)
            return {"PASS_THROUGH"} if over is None else {"RUNNING_MODAL"}

        if over is None:
            return {"PASS_THROUGH"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            it = widgets[over]
            if "act" not in it:
                return {"RUNNING_MODAL"}
            _fire(context, it, mx)
            _redraw(context, context.area)
            return {"RUNNING_MODAL"}
        if event.type == "LEFTMOUSE":
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def cancel(self, context):
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


class NCP_OT_ui_switch(bpy.types.Operator):
    """Переключить интерфейс: GPU-карточка <-> классическая N-панель."""

    bl_idname = "ncp.ui_switch"
    bl_label = "Viewport Card On/Off"
    bl_options = {"REGISTER", "UNDO"}

    classic: bpy.props.BoolProperty(
        name="Classic panel", default=True,
        description="True = вернуть классическую N-панель")

    def execute(self, context):
        p = _prefs()
        if p is None:
            self.report({"ERROR"}, "Preferences аддона недоступны")
            return {"CANCELLED"}
        try:
            p.overlay = not bool(self.classic)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        try:
            bpy.ops.wm.save_userpref()
        except Exception:
            pass
        msg = ("Классическая N-панель возвращена" if self.classic
               else "GPU-карточка включена — вкладка NCP")
        self.report({"INFO"}, msg)
        say(msg)
        _redraw_all()
        return {"FINISHED"}


class NCP_PT_tab(bpy.types.Panel):
    """Пустая панель: держит вкладку NCP живой, пока карточка рисует сама.
    Карточка не смогла — здесь виден выход на классическую панель."""

    bl_label = "Neuro Character"
    bl_idname = "NCP_PT_tab"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = TAB
    bl_options = {"HIDE_HEADER"}
    bl_order = 0

    def draw(self, context):
        if _state.get("handle") is not None and not _state.get("failed"):
            return
        col = self.layout.column()
        if _state.get("failed"):
            col.label(text="Карточка упала — панели возвращены", icon="ERROR")
        else:
            col.label(text="GPU-карточка выключена", icon="INFO")
        col.operator("ncp.ui_switch", text="Включить карточку",
                     icon="WINDOW").classic = False


CLASSES = (NCP_OT_overlay, NCP_OT_ui_switch, NCP_PT_tab)


def register():
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except Exception as e:
            print("[NCP] overlay class %s: %s" % (c.__name__, e))
    _state["quit"] = False
    _state["failed"] = False
    register_draw()
    if not bpy.app.timers.is_registered(_keep_alive):
        bpy.app.timers.register(_keep_alive, first_interval=0.5,
                                persistent=True)


def unregister():
    _state["quit"] = True
    try:
        if bpy.app.timers.is_registered(_keep_alive):
            bpy.app.timers.unregister(_keep_alive)
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
    except Exception:
        pass
    try:
        if bpy.app.timers.is_registered(_settle):
            bpy.app.timers.unregister(_settle)
    except Exception:
        pass
    try:
        clear()
    except Exception:
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass
