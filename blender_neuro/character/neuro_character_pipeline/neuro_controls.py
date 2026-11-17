# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.neuro_controls — слой рычагов (control handles)
поверх neuro-скелета.

Рычаги — крупные кости-ручки, за которые персонаж двигается мышью:
  CTRL_Root  — весь персонаж целиком (перенести/поставить на место),
               работает ВСЕГДА, в том числе поверх нейроанимации;
  CTRL_Body  — корпус (с ним голова/хвост/крылья едут вместе);
  CTRL_Arm_L/R, CTRL_Wing_L/R — кончик руки/крыла (IK);
  CTRL_Foot_L/R (+ front/hind у четвероногих) — нога за ступню (IK);
  CTRL_Head  — голова/шея;  CTRL_Tail — кончик хвоста;
  CTRL_Pole_Knee_*/CTRL_Pole_Elbow_* — полюсы суставов: направление
               сгиба колена/локтя (2-бонные IK-цепи; тянешь полюс —
               сустав разворачивается, конечность не щёлкает: угол
               pole_angle подбирается по факту геометрии при сборке);
  CTRL_Finger_* — отдельный рычаг на каждый палец (IK на 2 сегмента:
               тянешь кончик пальца — палец сгибается за мышью).

Правила безопасности анимации (то, ради чего слой сделан отдельным):
  1. CTRL-кости use_deform=False — в скининг и веса не попадают,
     меш они не деформируют;
  2. IK/COPY_ROTATION-констрейнты называются NCP_*: influence ставит
     set_ik() — галка «IK handles» в панели. Import Neuro Motion сам
     гасит IK, чтобы нейроанимация играла как записана;
  3. остовные (deform) кости можно спрятать одной галкой — в вьюпорте
     остаются только рычаги (кликать легко, анимировать не мешает);
  4. пальцевые рычаги попадают в отдельную коллекцию костей
     NCP_Controls_Fine — их можно скрыть отдельно от крупных рычагов.

Формы рычагов (2.11.0): пальцы — маленькие оранжевые ДИСКИ
(NCP_Shape_Disc), остальные суставы — КОЛЬЦА с полым ободом
(NCP_Shape_Ring: сустав виден сквозь кольцо), Root — кольцо,
Body — диск. Размеры ЛОКАЛЬНЫЕ: кольцо сустава — ~0.5 длины
концевой кости (ступня — кольцо размером со ступню, кисть — с
кисть), диск пальца — ~0.55 длины пальца, а не 0.09–0.17 роста
(v2.10.1: у персонажа 1.7 м кольца были 15–30 см в диаметре и
закрывали вьюпорт). Базы хранятся на арматуре (ncp_ctrl_sizes,
JSON): apply_size_mult() применяет множитель слайдера «Размер
рычагов» живьём; set_handles_visible() показывает/прячет ВСЕ
рычаги одной галкой (Bone.hide на CTRL_*/NCPK_*).
set_mode() — переключатель «T-поза / анимация»
(TPOSE: IK выкл + поза рычагов сброшена в rest; ANIM: IK вкл).
Запись по слоям: record_frame() ставит кейфрейм текущей позы рычагов
включённых слоёв (body/arms/legs/fingers/wings), record_set_enabled()
подключает depsgraph-хэндлер автозаписи — изменилась поза рычага
включённого слоя, кейфрейм на текущем кадре (остовные кости и
нейроанимация не затрагиваются).

Иерархия: CTRL_Root -> CTRL_Body -> pelvis. Root/Body/Head/Tail —
прямое parenting (двигаются всегда, никаких констрейнтов); Arm/Wing/
Foot/Pole — IK-цели и полюсы, parent CTRL_Root (при движении тела
ноги остаются приколотыми к ступням — можно присаживать персонажа);
рычаги пальцев parent к кости кисти/крыла — едут вместе с рукой.

Цель IK = ХВОСТ последней кости цепи (use_tail=True, 2.8.0: в 2.7.0
было use_tail=False — Blender исключал конечную кость из цепи и тянул
к рычагу колено/локоть вместо ступни/кисти). Кончик рычага ставится
ровно в сустав (голеностоп/запястье/кончик пальца).

IK-хелперы NCPK_* (2.8.0): у human-контракта остовные кости ком-якорные
(кость покрывает половину звена), 2-бонная цепь из них короче
конечности — IK с недостижимой целью не сгибает колено. Хелперы
натянуты сустав-в-сустав (плечо->локоть->запястье, бедро->колено->
голеностоп), IK стоит на них, а остовные кости следуют за хелперами
через COPY_TRANSFORMS с влиянием = галка IK. FK-анимация (Import
Neuro Motion) пишет pb.matrix абсолютно и influence гасит — движение
играет как записано; хелперы сбрасываются в rest при импорте.

Топология рычагов строится по ФАКТУ арматуры: цепочки конечностей
находятся обходом дерева костей (то же дерево, что построил
neuro_creature.analyze), поэтому слой работает для любого плана —
человек, четвероногий, птица, дракон, двуногий с хвостом.
"""
import json
import math
import os
import re

import bpy
from mathutils import Vector

PFX = "CTRL_"
CONS_PFX = "NCP_"
# вспомогательные IK-кости (натянуты сустав-в-сустав). У human-контракта
# остовные кости ком-якорные (кость = половина звена), 2-бонная IK-цепь
# из них короче конечности — IK с недостижимой целью не сгибает колено.
# Хелперы дают солверу настоящие рычаги; остовные кости следуют за ними
# через COPY_TRANSFORMS (влияние = галка IK, FK-анимация не затронута)
HLP_PFX = "NCPK_"

# кости-коннекторы (короткие перемычки у груди/таза) в IK-цепи не берём
_CONNECTORS = ("shoulderP_", "hipR_")

_KIND_PFX = {
    "leg": "CTRL_Foot",
    "arm": "CTRL_Arm",
    "wing": "CTRL_Wing",
    "tail": "CTRL_Tail",
    "head": "CTRL_Head",
}

_KIND_COLOR = {
    "root": (0.10, 0.85, 0.35, 0.9),
    "body": (0.25, 0.55, 1.00, 0.9),
    "head": (0.95, 0.75, 0.15, 0.9),
    "arm": (0.95, 0.45, 0.10, 0.9),
    "leg": (0.90, 0.20, 0.20, 0.9),
    "wing": (0.65, 0.35, 0.90, 0.9),
    "tail": (0.40, 0.80, 0.90, 0.9),
    "pole": (0.90, 0.25, 0.55, 0.9),
    "finger": (1.00, 0.45, 0.00, 0.9),
}

# имена человеческих пальцев (add_finger_chains: fng_thumb_1_L ...)
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


class ControlsError(RuntimeError):
    pass


# ------------------------------------------------- геометрия цепочек

def _activate(arm_obj):
    """Сделать арматуру активной (mode_set работает с активным объектом)."""
    try:
        for o in bpy.context.view_layer.objects.selected:
            o.select_set(False)
    except Exception:
        pass
    try:
        arm_obj.select_set(True)
    except Exception:
        pass
    try:
        bpy.context.view_layer.objects.active = arm_obj
    except Exception:
        pass


def _edit_mode(arm_obj):
    _activate(arm_obj)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    try:
        bpy.ops.object.mode_set(mode="EDIT")
    except RuntimeError:
        pass


def _object_mode():
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass


def _children_map(ebs):
    out = {}
    for eb in ebs:
        if eb.name.startswith(HLP_PFX):
            continue                     # хелперы прозрачны для топологии
        p = eb.parent
        if p is not None:
            out.setdefault(p.name, []).append(eb.name)
    return out


def _classify(first_name):
    if first_name.startswith("tail"):
        return "tail"
    if first_name.startswith(("neck_", "head")):
        return "head"
    if first_name.startswith("wing"):
        return "wing"
    if first_name.startswith(("arm", "uarm", "farm", "shoulderP")):
        return "arm"
    if first_name.startswith(("leg", "thigh", "shin", "hipR", "foot")):
        return "leg"
    return None


def _dot(a, b):
    """Скалярное произведение по компонентам (работает и со стабом)."""
    return a.x * b.x + a.y * b.y + a.z * b.z


def _cross(a, b):
    """Векторное произведение по компонентам (работает и со стабом).

    Blender 4+ требует единственный аргумент-последовательность:
    Vector((x, y, z)) — Vector(x, y, z) кидает TypeError
    «takes at most 1 argument» (урок v2.10.1)."""
    return Vector((a.y * b.z - a.z * b.y,
                   a.z * b.x - a.x * b.z,
                   a.x * b.y - a.y * b.x))


def _side_of(chain_names):
    for n in chain_names:
        if n.endswith("_L") or "_L_" in n:
            return "_L"
        if n.endswith("_R") or "_R_" in n:
            return "_R"
    return ""


def _qual_of(first_name):
    for tok in ("front", "hind"):
        if ("_" + tok + "_") in first_name:
            return "_" + tok
    m = re.match(r"leg(\d+)_", first_name)
    if m:
        return "_" + m.group(1)
    return ""


def find_finger_chains(arm_obj):
    """Пальцевые цепочки (fng_*) по факту дерева костей.

    Возвращает список dict(parent, side, bones): parent — кость кисти/
    крыла (родитель первого сегмента), bones — сегменты пальца от
    основания к кончику (обычно 2). Человек: 10 пальцев по 2 сегмента
    (fng_thumb_1_L...), creature — 1-5 пальцев с геометрии или std."""
    _edit_mode(arm_obj)
    try:
        ebs = arm_obj.data.edit_bones
        children = _children_map(ebs)
        out = []
        for eb in ebs:
            nm = eb.name
            if not nm.startswith("fng_"):
                continue
            par = eb.parent
            if par is not None and par.name.startswith("fng_"):
                continue                   # не основание пальца
            chain = [nm]
            while True:
                kids = [k for k in children.get(chain[-1], [])
                        if k.startswith("fng_")]
                if len(kids) != 1:
                    break
                chain.append(kids[0])
            out.append({
                "parent": par.name if par is not None else "",
                "side": _side_of(chain),
                "bones": chain,
            })
        return out
    finally:
        _object_mode()


def _finger_handle_name(first_bone, used):
    """CTRL_Finger_* — короткое имя рычага пальца.

    Человек: fng_thumb_1_L -> CTRL_Finger_thumb_L; creature:
    fng_arm_L_1_1 -> CTRL_Finger_arm_L_1, fng_arm_L_std_1_1 ->
    CTRL_Finger_arm_L_std_1. Коллизии гасятся суффиксом _2/_3..."""
    m = re.match(r"fng_(%s)_(\d+)_(L|R)$" % "|".join(_FINGER_NAMES),
                 first_bone)
    if m:
        base = "CTRL_Finger_%s_%s" % (m.group(1), m.group(3))
    else:
        m = re.match(r"fng_(.+)_std_(\d+)_(\d+)$", first_bone)
        if m:
            base = "CTRL_Finger_%s_std_%s" % (m.group(1), m.group(2))
        else:
            m = re.match(r"fng_(.+?)_(\d+)_(\d+)$", first_bone)
            if m:
                base = "CTRL_Finger_%s_%s" % (m.group(1), m.group(2))
            else:
                base = "CTRL_Finger_" + first_bone[4:]
    name = base
    i = 2
    while name in used:
        name = "%s_%d" % (base, i)
        i += 1
    used.add(name)
    return name


def find_chains(arm_obj):
    """Цепочки конечностей по факту дерева костей арматуры.

    Возвращает список dict(kind, side, qual, bones): bones — имена
    костей от корня цепочки до кончика (без пальцев). Ствол
    (pelvis->spine->chest) в цепочки не входит; цепочка = путь до
    развилки (пальцы дают развилку) или листа. Голова может быть
    цепочкой из одной кости."""
    _edit_mode(arm_obj)
    try:
        ebs = arm_obj.data.edit_bones
        if len(ebs) == 0:
            raise ControlsError("у арматуры нет костей")
        children = _children_map(ebs)
        root = None
        for eb in ebs:
            if eb.parent is None:
                root = eb.name
                break
        if root is None:
            raise ControlsError("нет корневой кости")
        # ствол: root -> spine -> chest (есть во всех планах)
        trunk = [root]
        while True:
            nxt = [c for c in children.get(trunk[-1], [])
                   if c in ("spine", "chest", "torso")]
            if not nxt:
                break
            trunk.append(nxt[0])
        chains = []
        for t in trunk:
            starts = [c for c in children.get(t, []) if c not in trunk]
            for s in starts:
                chain = [s]
                while True:
                    kids = [k for k in children.get(chain[-1], [])
                            if not k.startswith("fng_")]
                    if len(kids) != 1:
                        break
                    chain.append(kids[0])
                kind = _classify(chain[0])
                if kind is None:
                    continue
                if len(chain) < 2 and kind != "head":
                    continue          # одиночная кость — не цепочка
                chains.append({
                    "kind": kind,
                    "side": _side_of(chain),
                    "qual": _qual_of(chain[0]),
                    "bones": chain,
                })
        return chains
    finally:
        _object_mode()


# ------------------------------------------------- формы и косметика

def _circle(n=24, r=1.0, z=0.0):
    return [(r * math.cos(6.2831853 * i / n),
             r * math.sin(6.2831853 * i / n), z) for i in range(n)]


# диск: центр + полное кольцо вершин, один n-gon (в 2.8.0 последний
# сегмент терялся — лицо многоугольника было собрано неверно)
_SHAPE_DISC = (
    [(0.0, 0.0, 0.0)] + _circle(24, 1.0),
    [list(range(25))],
)
# кольцо (полый обод): видно сустав сквозь него — «кольца на суставах»,
# которые просил пользователь; диск остался на пальцах
_o = _circle(24, 1.0)
_i = _circle(24, 0.62)
_SHAPE_RING = (
    _o + _i,
    [[k, (k + 1) % 24, 24 + (k + 1) % 24, 24 + k] for k in range(24)],
)
_SHAPE_BOX = (
    [(-0.6, -0.6, -0.6), (0.6, -0.6, -0.6), (0.6, 0.6, -0.6),
     (-0.6, 0.6, -0.6), (-0.6, -0.6, 0.6), (0.6, -0.6, 0.6),
     (0.6, 0.6, 0.6), (-0.6, 0.6, 0.6)],
    [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
     (2, 3, 7, 6), (3, 0, 4, 7)],
)
SHAPES = {"disc": ("NCP_Shape_Disc", _SHAPE_DISC),
          "ring": ("NCP_Shape_Ring", _SHAPE_RING),
          "box": ("NCP_Shape_Box", _SHAPE_BOX)}


def _helpers_collection():
    """Коллекция NCP_Helpers для объектов-форм (вне вида).

    Стабы без bpy.data.collections молча пропускают (тогда формы
    остаются в scene collection, как в v2.10.1 — косметике не мешает)."""
    try:
        coll = bpy.data.collections.get("NCP_Helpers")
        if coll is None:
            coll = bpy.data.collections.new("NCP_Helpers")
        scene = bpy.context.scene
        linked = any(c.name == coll.name
                     for c in scene.collection.children)
        if not linked:
            scene.collection.children.link(coll)
        return coll
    except Exception:
        return None


def _exclude_from_view(coll):
    """Исключить коллекцию из вью-слоя (объекты не рисуются и не
    мешают в аутлайнере). Custom shapes читают МЕШ-данные напрямую,
    видимость объекта-формы на них не влияет."""
    try:
        def _walk(lc):
            if lc.collection.name == coll.name:
                lc.exclude = True
                return True
            for ch in lc.children:
                if _walk(ch):
                    return True
            return False
        _walk(bpy.context.view_layer.layer_collection)
    except Exception:
        pass


def _get_shape(kind):
    """Объект-форма для custom_shape (скрытый; None = косметика не удалась).

    v2.11.0: формы живут в коллекции NCP_Helpers, исключённой из
    вью-слоя (глаз + hide_viewport + hide_render + hide_select):
    в аутлайнере и вьюпорте их больше не видно, случайно выбрать/
    показать их нельзя; на отображение custom shapes это не влияет."""
    name, geo = SHAPES[kind]
    try:
        me = bpy.data.meshes.get(name)
        if me is None:
            me = bpy.data.meshes.new(name)
            me.from_pydata(geo[0], [], geo[1])
        ob = bpy.data.objects.get(name)
        if ob is None:
            ob = bpy.data.objects.new(name, me)
            try:
                bpy.context.scene.collection.objects.link(ob)
            except Exception:
                pass
        ob.hide_render = True
        ob.hide_select = True
        try:
            ob.hide_viewport = True
        except AttributeError:
            pass
        try:
            ob.hide_set(True)
        except Exception:
            pass
        coll = _helpers_collection()
        if coll is not None and ob.name not in coll.objects:
            for c in list(ob.users_collection):
                try:
                    c.objects.unlink(ob)
                except Exception:
                    pass
            try:
                coll.objects.link(ob)
            except Exception:
                pass
            _exclude_from_view(coll)
        return ob
    except Exception:
        return None


def _set_shape(pb, kind, size):
    ob = _get_shape(kind)
    if ob is None:
        return
    try:
        pb.custom_shape = ob
    except Exception:
        return
    try:
        pb.use_custom_shape_bone_size = False
    except Exception:
        pass
    # v2.11.0: на Blender 4+/5.x НЕТ custom_shape_scale_uniform
    # (удалён в 4.0, остался custom_shape_scale_xyz) — раньше оба
    # варианта падали в except, масштаб оставался 1.0 и кольца
    # рисовались радиусом 1 МЕТР (форма-меш R=1) — «кольца очень
    # большие» на скриншоте v2.10.1
    s = float(size)
    try:
        pb.custom_shape_scale_xyz = (s, s, s)      # Blender 4.x/5.x
    except AttributeError:
        try:
            pb.custom_shape_scale_uniform = s       # Blender 3.x
        except Exception:
            try:
                pb.custom_shape_scale = (s, s, s)   # очень старые
            except Exception:
                pass
    except Exception:
        pass


def _set_color(pb, rgba):
    try:
        pb.color.palette = "CUSTOM"
        pb.color.custom = rgba
    except Exception:
        pass


def _assign_coll(data, bone_name, coll_name):
    """Bone collections (Blender 4+); старые версии и стабы — молча нет.

    assign() принимает ОБЪЕКТ кости (Bone/EditBone); строка имени на
    Blender 4.1+/5.x кидает TypeError и кость молча оставалась вне
    коллекций (урок v2.10.1: коллекции создавались пустыми)."""
    try:
        colls = data.collections
    except AttributeError:
        return
    try:
        coll = colls.get(coll_name)
        if coll is None:
            coll = colls.new(coll_name)
        bone = data.bones.get(bone_name)
        if bone is None:
            bone = data.edit_bones.get(bone_name)
        if bone is None:
            return
        coll.assign(bone)
    except Exception:
        pass


# ----------------------------------------------------- сборка слоя

def _handle_name(kind, side, qual, used):
    base = _KIND_PFX[kind] + (qual or "") + (side or "")
    name = base
    i = 2
    while name in used:
        name = "%s_%d" % (base, i)
        i += 1
    used.add(name)
    return name


def _dir_of(eb):
    v = eb.tail - eb.head
    if v.length > 1e-6:
        return v.normalized()
    return Vector((1.0, 0.0, 0.0))          # у нейроскелета лицо +X


def _bone_kind(handle_name):
    return "pole" if handle_name.startswith("CTRL_Pole") else \
        "finger" if handle_name.startswith("CTRL_Finger") else \
        "leg" if handle_name.startswith("CTRL_Foot") else \
        "head" if handle_name.startswith("CTRL_Head") else \
        "wing" if handle_name.startswith("CTRL_Wing") else \
        "tail" if handle_name.startswith("CTRL_Tail") else "arm"


def _perp_to(v, axis):
    """Компонента v, перпендикулярная axis (или None для вырожденной)."""
    o = v - axis * _dot(v, axis)
    return o if o.length > 1e-9 else None


def _bone_axes(eb):
    """(X, Z) оси кости в пространстве арматуры (из матрицы кости).

    Нужны для pole_angle: Blender сгибает 2-бонную IK-цепь в
    направлении up(theta) = X*cos(theta) + Z*sin(theta).
    Blender 4.0+: EditBone.matrix — 4x4 (колонки 4-компонентные),
    оси берём элементами m[r][c] — работает и для 3x3 старых
    версий, и для 4x4 новых (урок v2.10.1: m.col[0] на Blender 4+
    давал 4D-вектор и ронял весь слой рычагов с AttributeError
    «vectors must have the same dimensions» — пользователь видел
    скелет без единого рычага и молчание вместо ошибки)."""
    try:
        m = eb.matrix
        ax = Vector((m[0][0], m[1][0], m[2][0]))
        az = Vector((m[0][2], m[1][2], m[2][2]))
        if ax.length < 1e-9 or az.length < 1e-9:
            return None, None
        return ax.normalized(), az.normalized()
    except Exception:
        return None, None


def _horiz_dir(v):
    """Горизонтальная проекция направления (носок вперёд)."""
    h = Vector((v.x, v.y, 0.0))
    if h.length < 1e-9:
        return None
    return h.normalized()


def _avg_dir(dirs, fallback):
    """Среднее направление (носок вперёд) или fallback (+X = лицо)."""
    acc = Vector((0.0, 0.0, 0.0))
    for d in dirs:
        acc = acc + d
    if acc.length < 1e-9:
        return fallback
    return acc.normalized()


def _pole_bend_dir(eb1, anchor, face, kind, qual):
    """Куда должен смотреть сустав (колено/локоть) — направление сгиба.

    Основной вариант — по факту геометрии: проекция первой кости цепи
    на плоскость, перпендикулярную оси цепи (сустав и так согнут,
    сохраняем его направление). Для прямой цепи — анатомический
    запасной вариант: колено вперёд (к носку; у передних лап
    четвероногих — назад, как локоть), локоть назад."""
    axis = anchor - eb1.head
    if axis.length < 1e-9:
        return None
    axis = axis * (1.0 / axis.length)
    k = _perp_to(eb1.tail - eb1.head, axis)
    if k is not None:
        return k.normalized()
    back = Vector((-face.x, -face.y, -face.z))   # назад = против носка
    if kind == "leg":
        return back if qual == "_front" else face
    return back


def _pole_angle_for(eb1, rootpos, target, bend_dir):
    """pole_angle (радианы) для 2-бонной IK-цепи с полюсом.

    Солвер Blender (intern/iksolver, ConstrainPoleVector): при
    pole_angle=theta суставная ось up(theta) = X_кости*cos(theta) +
    Z_кости*sin(theta) разворачивается на полюс — и цепь сгибается
    в этом направлении. Подбираем theta так, чтобы при включении IK
    сустав остался в текущем положении (bend_dir) — без прыжка позы;
    дальше пользователь тянет полюс — сустав следует за ним."""
    axis = target - rootpos
    if axis.length < 1e-9:
        return 0.0
    axis = axis * (1.0 / axis.length)
    ax, az = _bone_axes(eb1)
    if ax is None or az is None:
        return 0.0
    a = _perp_to(ax, axis)
    b = _perp_to(az, axis)
    k = _perp_to(bend_dir, axis)
    if a is None or b is None or k is None:
        return 0.0
    aa = _dot(_cross(a, k), axis)
    bb = _dot(_cross(b, k), axis)
    if abs(aa) + abs(bb) < 1e-12:
        return 0.0
    theta = math.atan2(-aa, bb)
    up = a * math.cos(theta) + b * math.sin(theta)
    if _dot(up, k) < 0.0:
        theta += math.pi
    return theta


def _hinge_axis(bone, bend):
    """Какая локальная ось кости сгибает сустав В сторону bend.

    bend — мировое направление сгиба (перпендикулярно кости). Ось
    шарнира = cross(Y_кости, bend); ищем локальную ось (X или Z),
    параллельную шарниру. Возвращает ("x"|"z", flip) где flip=True
    если положительный поворот вокруг локальной оси сгибает ОТ bend.
    """
    m = bone.matrix_local
    y = Vector((m[0][1], m[1][1], m[2][1]))
    if y.length < 1e-9:
        return None, False
    y = y.normalized()
    b = bend - y * bend.dot(y)
    if b.length < 1e-9:
        return None, False
    b = b.normalized()
    hinge = y.cross(b)                       # вращение вокруг hinge
    ax = Vector((m[0][0], m[1][0], m[2][0]))
    az = Vector((m[0][2], m[1][2], m[2][2]))
    dx, dz = ax.dot(hinge), az.dot(hinge)
    if abs(dx) >= abs(dz):
        return "x", dx < 0.0
    return "z", dz < 0.0


def _apply_ik_limits(arm_obj, chains, fchains):
    """Ограничители IK-суставов (v2.13.0) — use_ik_limit_* на костях
    IK-цепей. Стандартный солвер уважает их: колено/локоть/палец
    сгибаются только в анатомическую сторону и не выкручиваются
    наизнанку (жалоба 2.12 «кости сгибаются под неправильными
    углами, проворачиваются наизнанку»).

    Лимиты действуют ТОЛЬКО на решение IK (FK и нейроанимация —
    абсолютные матрицы — не ограничены), поэтому ставятся на
    вспомогательные кости: NCPK-хелперы ног/рук human-контракта,
    сами кости у creature-планов и пальцев. Остовные кости спеки
    не трогаются. Наклон плеча/бедра (первая кость цепи) не
    ограничивается — ограничен только твист."""
    for ch in chains:
        host = ch.get("ik_host") or ch.get("ik_names") or []
        if not host:
            continue
        kind = ch["kind"]
        bend = ch.get("bend_dir")
        pose = arm_obj.pose.bones
        for i, bn in enumerate(host):
            pb = pose.get(bn)
            bone = arm_obj.data.bones.get(bn)
            if pb is None or bone is None:
                continue
            try:
                # твист вокруг оси кости: ноги почти не крутятся;
                # у ПРЕДПЛЕЧЬЯ и пальцев твист-канал ЗАМКН (lock_ik_y):
                # стандартный солвер имеет систематический дрейф по
                # твисту на зеркальных (R) цепях — на bpy 5.3 farm_R
                # самопроизвольно скручивалась на 27° при включении IK
                # (жалоба «кости проворачиваются»); замок твиста
                # обнуляет дрейф, сгиб (шарнир) не затрагивает
                tlim = (0.0, 0.0)
                lock_y = False
                if kind == "leg":
                    tlim = (-25.0, 25.0) if i == 0 else (-12.0, 12.0)
                elif kind == "arm":
                    if i == 0:
                        tlim = (-45.0, 45.0)
                    else:
                        lock_y = True
                elif kind == "wing":
                    tlim = (-45.0, 45.0)
                elif kind == "tail":
                    tlim = (-50.0, 50.0)
                elif kind == "head":
                    tlim = (-85.0, 85.0)
                if lock_y:
                    try:
                        pb.lock_ik_y = True
                    except Exception:
                        pass
                elif tlim != (0.0, 0.0):
                    pb.use_ik_limit_y = True
                    pb.ik_min_y = math.radians(tlim[0])
                    pb.ik_max_y = math.radians(tlim[1])
                # шарнир (односторонний сгиб) — со второй кости цепи
                if bend is not None and i >= 1:
                    ax, flip = _hinge_axis(bone, bend)
                    if ax is None:
                        continue
                    lo, hi = (0.0, 150.0)
                    if flip:
                        lo, hi = -150.0, 0.0
                    setattr(pb, "use_ik_limit_" + ax, True)
                    setattr(pb, "ik_min_" + ax, math.radians(lo))
                    setattr(pb, "ik_max_" + ax, math.radians(hi))
                    # вторая поперечная ось: небольшой запас
                    oax = "z" if ax == "x" else "x"
                    setattr(pb, "use_ik_limit_" + oax, True)
                    setattr(pb, "ik_min_" + oax, math.radians(-12.0))
                    setattr(pb, "ik_max_" + oax, math.radians(12.0))
            except Exception:
                pass
    # пальцы: сгиб к ладони до 115°, разгибание всего 12°, твист ±12°,
    # боковой люфт ±10° — узкий конверт не даёт солверу «дрейфовать»
    # по многообразию решений при включении IK (снап пальцев 2.12)
    for fc in fchains:
        bend = None
        try:
            pb0 = arm_obj.pose.bones.get(fc["bones"][0])
            if pb0 is not None:
                m = pb0.bone.matrix_local
                y = Vector((m[0][1], m[1][1], m[2][1])).normalized()
                d = Vector((0.0, 0.0, -1.0))
                bend = d - y * d.dot(y)
                if bend.length < 1e-6:
                    bend = Vector((1.0, 0.0, 0.0))
                bend = bend.normalized()
        except Exception:
            bend = None
        for j, bn in enumerate(fc["bones"]):
            pb = arm_obj.pose.bones.get(bn)
            bone = arm_obj.data.bones.get(bn)
            if pb is None or bone is None:
                continue
            try:
                # твист пальцев замкнут (дрейф солвера на R-цепях)
                try:
                    pb.lock_ik_y = True
                except Exception:
                    pass
                if bend is not None:
                    ax, flip = _hinge_axis(bone, bend)
                    if ax is not None:
                        lo, hi = (-115.0, 12.0) if flip else (-12.0, 115.0)
                        setattr(pb, "use_ik_limit_" + ax, True)
                        setattr(pb, "ik_min_" + ax, math.radians(lo))
                        setattr(pb, "ik_max_" + ax, math.radians(hi))
                        oax = "z" if ax == "x" else "x"
                        setattr(pb, "use_ik_limit_" + oax, True)
                        setattr(pb, "ik_min_" + oax, math.radians(-10.0))
                        setattr(pb, "ik_max_" + oax, math.radians(10.0))
            except Exception:
                pass


def build_controls(arm_obj, ik_on=True, hide_deform=False, fingers=True,
                   poles=True):
    """Собрать слой рычагов. Возвращает короткий отчёт (строка).

    fingers — отдельные рычаги пальцев (CTRL_Finger_*, IK на 2
    сегмента); poles — полюсы коленей/локтей (CTRL_Pole_*). Ломать
    риг не может: каждая косметическая вещь (формы, цвета,
    коллекции) обёрнута в try/except."""
    if arm_obj is None or getattr(arm_obj, "type", "") != "ARMATURE":
        raise ControlsError("нет арматуры")
    remove_controls(arm_obj)                 # пересборка на чистую

    chains = find_chains(arm_obj)
    fchains = find_finger_chains(arm_obj) if fingers else []

    # -- сессия редактирования: CTRL-кости + пере-parenting корня
    made = []
    poles_made = []
    fingers_made = []
    lens = {}
    kind_sizes = {}
    h = 0.05
    _edit_mode(arm_obj)
    try:
        ebs = arm_obj.data.edit_bones
        root = None
        for eb in ebs:
            if eb.parent is None:
                root = eb
                break
        if root is None:
            raise ControlsError("нет корневой кости")
        zs = [v.z for eb in ebs for v in (eb.head, eb.tail)]
        h = max(max(zs) - min(zs), 0.05)
        base_z = min(zs)
        lens = {eb.name: float(eb.length) for eb in ebs}
        root_name = root.name
        rh = Vector((root.head.x, root.head.y, root.head.z))
        rd = _dir_of(root)

        def new_ctrl(name, head, tail, parent=None):
            eb = ebs.new(name)
            eb.head = Vector(head)
            eb.tail = Vector(tail)
            eb.roll = 0.0
            eb.use_deform = False
            try:
                eb.show_wire = True
            except Exception:
                pass
            if parent is not None:
                eb.parent = ebs.get(parent)
                try:
                    eb.use_connect = False
                except Exception:
                    pass
            made.append(name)
            return eb

        rz = min(rh.z - 0.06 * h, base_z + 0.01)
        new_ctrl("CTRL_Root", (rh.x, rh.y, rz), (rh.x, rh.y, rz + 0.06 * h))
        new_ctrl("CTRL_Body", (rh.x, rh.y, rh.z),
                 Vector((rh.x, rh.y, rh.z)) + rd * max(0.10 * h, 0.02),
                 parent="CTRL_Root")
        # корневая кость скелета — ребёнок CTRL_Body: тело двигается
        # всегда (без IK), CTRL_Root переносит весь персонаж целиком
        rb = ebs.get(root_name)
        if rb is not None:
            rb.parent = ebs.get("CTRL_Body")
            try:
                rb.use_connect = False
            except Exception:
                pass

        used = {"CTRL_Root", "CTRL_Body"}
        toe_dirs = []
        for ch in chains:
            ebs_c = [ebs.get(n) for n in ch["bones"]]
            if any(e is None for e in ebs_c):
                continue
            tip = ebs_c[-1]
            d = _dir_of(tip)
            # кости IK-цепи: без коннекторов; у ног ступня — вне цепи
            # (она получает COPY_ROTATION от рычага)
            ikn = [n for n in (ch["bones"][:-1]
                               if ch["kind"] == "leg" else ch["bones"])
                   if not n.startswith(_CONNECTORS)]
            if ch["kind"] == "leg":
                ikn = [n for n in ikn
                       if lens.get(n, 0.0) >= 0.07 * h]
            if not ikn:
                continue
            last_ik = ebs.get(ikn[-1])
            if last_ik is None:
                continue
            # якорь рычага: сустав, за который берутся рукой —
            # голеностоп (голова ступни) у ног, кончик у остальных
            anchor = tip.head if ch["kind"] == "leg" else tip.tail
            anchor = Vector((anchor.x, anchor.y, anchor.z))
            ch["ik_names"] = ikn
            ch["anchor"] = anchor
            name = _handle_name(ch["kind"], ch["side"], ch["qual"], used)
            par = "CTRL_Body" if ch["kind"] in ("head", "tail", "wing") \
                else "CTRL_Root"
            if ch["kind"] == "leg":
                # нога: ГОЛОВА рычага — В ГОЛЕНОСТОПЕ (это и есть точка
                # IK-цели: на Blender 4/5 use_tail=True тянет конец цепи
                # к ГОЛОВЕ рычага — проверено на bpy 5.3), хвост торчит
                # ВПЕРЕД за носок как рукоятка. Раньше голова была у
                # пятки: солвер тянул голеностоп К ПЯТКЕ (13-15 см) и
                # вся нога скручивалась при включении IK (жалоба 2.12:
                # «кости проворачиваются наизнанку»)
                hl = max(0.09 * h, min(tip.length * 0.7, 0.16 * h))
                new_ctrl(name, anchor, anchor + d * hl, parent=par)
                ch["target"] = anchor
            else:
                # ВЫНОСНОЙ РЫЧАГ (v2.12.0): голова в суставе, хвост ЗА
                # кончиком конечности (за кисть/крыло/голову). IK-цель
                # = хвост рычага; на прямой конечности цель вне
                # досягаемости не меняет позу (конечность уже вытянута
                # в её сторону), а сам рычаг торчит наружу — его видно
                # и легко схватить мышью
                hl = max(0.10 * h, min(tip.length * 0.6, 0.20 * h))
                tgt = anchor + d * hl
                new_ctrl(name, anchor, tgt, parent=par)
                ch["target"] = tgt
            ch["handle"] = name
            # v2.11.0: кольцо сустава — по ТОЛЩИНЕ СВОЕЙ конечности
            # (0.5 длины концевой кости, рамка 0.02h..0.07h), а не
            # 0.09–0.25 роста (v2.10.1: кольца 15–30 см закрывали
            # вьюпорт у персонажа 1.7 м)
            kind_sizes[name] = max(0.02 * h,
                                   min(tip.length * 0.5, 0.07 * h))
            if ch["kind"] == "leg":
                hd = _horiz_dir(d)
                if hd is not None:
                    toe_dirs.append(hd)

        # -- IK-хелперы сустав-в-сустав: у human-контракта остовные
        # кости ком-якорные (кость = половина звена: бедро — от сустава
        # до середины бедра), 2-бонная цепь из них КОРОЧЕ конечности,
        # IK с недостижимой целью не сгибает колено. Хелперы натянуты
        # по настоящим суставам; остовные кости следуют за ними через
        # COPY_TRANSFORMS (влияние = галка IK; FK-анимация пишет
        # pb.matrix абсолютно и не затронута). У creature-планов кости
        # уже сустав-в-сустав — хелперы не нужны
        helpers_made = 0
        for ch in chains:
            ikn = ch.get("ik_names") or []
            if ch["kind"] not in ("leg", "arm") or len(ikn) != 2:
                ch["ik_host"] = ikn
                continue
            b1, b2 = ebs.get(ikn[0]), ebs.get(ikn[1])
            if b1 is None or b2 is None:
                ch["ik_host"] = ikn
                continue
            # разрыв 1: между костями (com-якорный остов);
            # разрыв 2: хвост второй кости НЕ ДОХОДИТ до якоря
            # (fitted-спека 2.3.0: thigh = бедро->колено ЦЕЛИКОМ,
            #  shin = колено->середина голени ПОЛОВИНА — IK-цепь
            #  [thigh, shin] была короче ноги на пол-голени и
            #  рычаг ступни тянул колено вниз вместо сгиба;
            #  урок v2.10.1 — проверять ОБА разрыва)
            gap = float((b2.head - b1.tail).length)
            gap2 = float((ch["anchor"] - b2.tail).length)
            if gap <= 0.02 * h and gap2 <= 0.02 * h:
                ch["ik_host"] = ikn        # кости уже сустав-в-сустав
                continue
            hname = ch["handle"] or "CTRL_Limb"
            n1 = HLP_PFX + hname[len(PFX):] + "_1"
            n2 = HLP_PFX + hname[len(PFX):] + "_2"
            if n1 in used or n2 in used:
                ch["ik_host"] = ikn
                continue
            used.add(n1)
            used.add(n2)
            try:
                h1 = ebs.new(n1)
                h1.head = Vector((b1.head.x, b1.head.y, b1.head.z))
                h1.tail = Vector((b2.head.x, b2.head.y, b2.head.z))
                h1.roll = 0.0
                h1.use_deform = False
                h1.parent = b1.parent
                h2 = ebs.new(n2)
                h2.head = Vector((b2.head.x, b2.head.y, b2.head.z))
                h2.tail = Vector((ch["anchor"].x, ch["anchor"].y,
                                  ch["anchor"].z))
                h2.roll = 0.0
                h2.use_deform = False
                h2.parent = h1
                ch["ik_host"] = [n1, n2]
                ch["hlp_map"] = ((ikn[0], n1), (ikn[1], n2))
                helpers_made += 2
            except Exception as _e:
                if os.environ.get("NCP_DEBUG"):
                    print("HLP FAIL %s: %s" % (n1, _e))
                ch["ik_host"] = ikn
        if os.environ.get("NCP_DEBUG"):
            for ch in chains:
                print("DBG chain", ch["kind"], ch.get("side"),
                      "host=", ch.get("ik_host"),
                      "ikn=", ch.get("ik_names"))

        # -- направление сгиба коленей/локтей: нужно полюсам И лимитам
        # (v2.13.0: считается ВСЕГДА, не только с полюсами)
        face = _avg_dir(toe_dirs, Vector((1.0, 0.0, 0.0)))
        for ch in chains:
            host = ch.get("ik_host") or []
            ch["bend_dir"] = None
            if ch["kind"] not in ("leg", "arm") or len(host) != 2:
                continue
            eb1, eb2 = ebs.get(host[0]), ebs.get(host[1])
            if eb1 is None or eb2 is None:
                continue
            ch["bend_dir"] = _pole_bend_dir(eb1, ch["anchor"], face,
                                            ch["kind"], ch["qual"])

        # -- полюсы коленей/локтей: только 2-бонные leg/arm цепи
        if poles:
            for ch in chains:
                kdir = ch.get("bend_dir")
                host = ch.get("ik_host") or []
                if kdir is None or len(host) != 2:
                    continue          # полюс — только для 2-бонных цепей
                base = "CTRL_Pole_%s%s%s" % (
                    "Knee" if ch["kind"] == "leg" else "Elbow",
                    ch["qual"] or "", ch["side"] or "")
                pname = base
                i = 2
                while pname in used:
                    pname = "%s_%d" % (base, i)
                    i += 1
                used.add(pname)
                eb1, eb2 = ebs.get(host[0]), ebs.get(host[1])
                rootpos = Vector((eb1.head.x, eb1.head.y, eb1.head.z))
                kneepos = Vector((eb2.head.x, eb2.head.y, eb2.head.z))
                chain_len = float((ch["anchor"] - rootpos).length)
                pl = max(0.10 * h, min(0.35 * chain_len, 0.30 * h))
                new_ctrl(pname, kneepos, kneepos + kdir * pl,
                         parent="CTRL_Root")
                ch["pole"] = pname
                ch["pole_angle"] = _pole_angle_for(eb1, rootpos,
                                                   ch["anchor"], kdir)
                poles_made.append(pname)
                # v2.11.0: полюс — компактное кольцо у сустава
                kind_sizes[pname] = max(0.015 * h,
                                        min(0.06 * chain_len,
                                            0.035 * h))

        # -- рычаги пальцев: по одному на каждый пальцевую цепочку
        for fc in fchains:
            segs = [ebs.get(n) for n in fc["bones"]]
            if any(e is None for e in segs):
                continue
            if not fc["parent"] or ebs.get(fc["parent"]) is None:
                continue
            tipf = segs[-1]
            d = _dir_of(tipf)
            fname = _finger_handle_name(fc["bones"][0], used)
            hl = max(0.015 * h, min(tipf.length * 0.6, 0.04 * h))
            # рычаг пальца parent к кисти/крылу: едет вместе с рукой
            new_ctrl(fname, tipf.tail - d * hl, tipf.tail,
                     parent=fc["parent"])
            fc["handle"] = fname
            fingers_made.append(fname)
            # v2.11.0: диск пальца — по длине ПАЛЬЦА (0.55 сегмента,
            # рамка 0.006h..0.013h), а не 0.025–0.06 роста (v2.10.1:
            # диск 4–10 см на пальце 1 см толщиной)
            kind_sizes[fname] = max(0.006 * h,
                                    min(tipf.length * 0.55, 0.013 * h))
    finally:
        _object_mode()

    # -- констрейнты (influence ставит set_ik; по умолчанию из ik_on)
    for ch in chains:
        handle = ch.get("handle")
        if not handle:
            continue
        host = ch.get("ik_host") or ch.get("ik_names") or []
        if not host or any(arm_obj.pose.bones.get(n) is None
                           for n in host):
            continue
        last = arm_obj.pose.bones.get(host[-1])
        if last is not None:
            try:
                c = last.constraints.new("IK")
                c.name = CONS_PFX + "IK_" + handle
                c.target = arm_obj
                c.subtarget = handle
                c.chain_count = min(3, len(host))
                # 2.8.0: use_tail=True — конечная кость В цепи, конец
                # цепи (ступня/кисть/кончик) тянется к рычагу.
                # ВАЖНО (проверено на bpy 5.3): use_tail=True тянет
                # конец цепи к ГОЛОВЕ рычага, use_tail=False — к ХВОСТУ.
                # Все цепи, кроме пальцев: голова рычага в суставе
                # (голеностоп/запястье/кончик крыла/хвоста/макушка) ->
                # use_tail=True (и перетаскивание, и вращение живут).
                # Пальцы: голова диска на 2-4 см ДО кончика (рычаг
                # торчит назад) — цель = ХВОСТ диска (на кончике):
                # use_tail=False. В 2.12 у ног цель фактически
                # оказывалась У ПЯТКИ (голова рычага) — нога
                # скручивалась при включении IK
                c.use_tail = ch["kind"] in ("leg", "arm", "head",
                                            "tail", "wing")
                try:
                    c.use_stretch = False
                except Exception:
                    pass
                # v2.13.0: вращение рычага передаётся концу цепи
                # (кольца поворачивают кисть/стопу/голову). Rest-рамки
                # рычага и концевой кости совпадают (рычаг строится
                # вдоль той же кости с roll=0) — снапа при включении
                # IK нет (проверено: снап 0.0°, передача 1:1).
                # Голова/хвост/крылья: цель (хвост рычага) вне
                # досягаемости и солвер её игнорирует — вращение кольца
                # ЕДИНСТВЕННЫЙ способ поворачивать их из позы
                try:
                    c.use_rotation = ch["kind"] in ("arm", "head",
                                                    "tail", "wing")
                except Exception:
                    pass
            except Exception:
                pass
            # полюс сустава: направление сгиба колена/локтя
            pname = ch.get("pole")
            if pname:
                try:
                    c.pole_target = arm_obj
                    c.pole_subtarget = pname
                    c.pole_angle = float(ch.get("pole_angle", 0.0))
                except Exception:
                    pass
        # ступня смотрит туда же, куда повернули рычаг
        if ch["kind"] == "leg":
            foot_pb = arm_obj.pose.bones.get(ch["bones"][-1])
            if foot_pb is not None:
                try:
                    c = foot_pb.constraints.new("COPY_ROTATION")
                    c.name = CONS_PFX + "ROT_" + handle
                    c.target = arm_obj
                    c.subtarget = handle
                except Exception:
                    pass

    # -- остовные кости следуют за IK-хелперами (влияние = галка IK;
    # при выключенном IK кости свободны — FK-анимация играет как записана)
    for ch in chains:
        for bone_name, hlp_name in ch.get("hlp_map") or ():
            pb = arm_obj.pose.bones.get(bone_name)
            if pb is None or arm_obj.pose.bones.get(hlp_name) is None:
                continue
            try:
                c = pb.constraints.new("COPY_TRANSFORMS")
                c.name = CONS_PFX + "CT_" + bone_name
                c.target = arm_obj
                c.subtarget = hlp_name
            except Exception:
                pass

    # -- констрейнты пальцев: IK кончика пальца на рычаг
    for fc in fchains:
        handle = fc.get("handle")
        if not handle:
            continue
        tip_pb = arm_obj.pose.bones.get(fc["bones"][-1])
        if tip_pb is None:
            continue
        try:
            c = tip_pb.constraints.new("IK")
            c.name = CONS_PFX + "IK_" + handle
            c.target = arm_obj
            c.subtarget = handle
            c.chain_count = min(2, len(fc["bones"]))
            # v2.13.0: use_tail=False — цель = ХВОСТ рычага (он на
            # кончике пальца). Раньше use_tail=True ставил цель в
            # ГОЛОВУ рычага (2-4 см ДО кончика) — при включении IK
            # пальцы срывались к рычагу (снап 40-318°, жалоба «кости
            # проворачиваются наизнанку»)
            c.use_tail = False
            try:
                c.use_stretch = False
            except Exception:
                pass
            # вращение диска пальца сгибает палец (конец цепи
            # разворачивается за диском, кончик дожимается позицией)
            try:
                c.use_rotation = True
            except Exception:
                pass
        except Exception:
            pass

    # -- ограничители IK-суставов (v2.13.0): колени/локти/пальцы
    # сгибаются только в анатомическую сторону, твист ограничен
    try:
        _apply_ik_limits(arm_obj, chains, fchains)
    except Exception:
        pass

    # -- косметика: формы, цвета, коллекции, видимость
    # пальцы — оранжевые ДИСКИ; остальные суставы — КОЛЬЦА (полый обод,
    # сустав виден сквозь него); Root — кольцо, Body — диск.
    # v2.11.0: Root/Body тоже компактные (0.055h/0.045h — ранее
    # 0.17h/0.12h: кольцо Root было 58 см диаметром у роста 1.7 м);
    # базы размеров сохраняются на арматуре для слайдера «Размер
    # рычагов» (apply_size_mult)
    pose = arm_obj.pose.bones
    sizes_all = dict(kind_sizes)
    sizes_all["CTRL_Root"] = 0.055 * h
    sizes_all["CTRL_Body"] = 0.045 * h
    for name, kind in (("CTRL_Root", "root"), ("CTRL_Body", "body")):
        try:
            pb = pose.get(name)
            if pb is not None:
                _set_shape(pb, "ring" if kind == "root" else "disc",
                           sizes_all[name])
                _set_color(pb, _KIND_COLOR[kind])
                _assign_coll(arm_obj.data, name, "NCP_Controls")
        except Exception:
            pass
    for name in made[2:]:
        try:
            pb = pose.get(name)
            if pb is None:
                continue
            _set_shape(pb, "disc" if name.startswith("CTRL_Finger_")
                       else "ring", sizes_all.get(name, 0.03 * h))
            _set_color(pb, _KIND_COLOR[_bone_kind(name)])
        except Exception:
            pass
        coll = "NCP_Controls_Fine" if name.startswith("CTRL_Finger_") \
            else "NCP_Controls"
        _assign_coll(arm_obj.data, name, coll)
    for pb in pose:
        if not pb.name.startswith(PFX):
            _assign_coll(arm_obj.data, pb.name, "NCP_Deform")
    try:
        arm_obj["ncp_ctrl_sizes"] = json.dumps(
            {k: round(float(v), 5) for k, v in sizes_all.items()})
    except Exception:
        pass
    try:
        arm_obj.show_in_front = True
    except Exception:
        pass
    try:
        arm_obj["ncp_ctrl"] = True
    except Exception:
        pass

    set_ik(arm_obj, bool(ik_on))
    set_deform_hidden(arm_obj, bool(hide_deform))
    if not made:
        return ""
    main_handles = [ch.get("handle") for ch in chains
                    if ch.get("handle")]
    rep = "рычаги: %d (Root, Body%s)" % (
        2 + len(main_handles),
        ", " + ", ".join(main_handles) if main_handles else "")
    if poles_made:
        rep += "; полюсов: %d (колени/локти)" % len(poles_made)
    if fingers_made:
        rep += "; пальцев: %d" % len(fingers_made)
    if helpers_made:
        rep += "; IK-хелперов: %d" % helpers_made
    if not ik_on:
        rep += "; IK выкл"
    return rep


def remove_constraints(arm_obj):
    """Снять все констрейнты NCP_* (перед пересборкой слоя)."""
    n = 0
    try:
        for pb in arm_obj.pose.bones:
            for c in [c for c in pb.constraints
                      if c.name.startswith(CONS_PFX)]:
                pb.constraints.remove(c)
                n += 1
    except Exception:
        pass
    return n


def apply_size_mult(arm_obj, mult):
    """Множитель размера рычагов — слайдер «Размер рычагов» (v2.11.0).

    Базовые радиусы форм (в метрах) сохраняются на арматуре при
    сборке (ncp_ctrl_sizes, JSON: имя кости -> радиус); здесь scale
    = base * mult ставится всем костям из базы живьём — пересборка
    не нужна. Возвращает число обновлённых костей (0 — нет базы/
    нет рычагов).

    v2.11.0: на Blender 4+/5.x свойства custom_shape_scale_uniform
    нет (удалён в 4.0) — пишем custom_shape_scale_xyz."""
    if arm_obj is None:
        return 0
    try:
        raw = arm_obj.get("ncp_ctrl_sizes")
        sizes = json.loads(raw) if raw else {}
    except Exception:
        sizes = {}
    if not sizes:
        return 0
    n = 0
    try:
        pose = arm_obj.pose.bones
    except AttributeError:
        return 0
    for name, base in sizes.items():
        pb = pose.get(name)
        if pb is None:
            continue
        s = float(base) * float(mult)
        try:
            pb.custom_shape_scale_xyz = (s, s, s)
            n += 1
            continue
        except AttributeError:
            pass
        except Exception:
            continue
        try:
            pb.custom_shape_scale_uniform = s
            n += 1
        except Exception:
            pass
    return n


def set_handles_visible(arm_obj, visible):
    """Показать/скрыть ВСЕ рычаги одной галкой (v2.11.0).

    Скрываются CTRL_* (рычаги, включая пальцевые диски) и NCPK_*
    (IK-хелперы) через Bone.hide — остовные кости не трогаются,
    работает в Object/Pose mode. Возвращает число затронутых
    костей (0 — нет рычагов)."""
    if arm_obj is None:
        return 0
    n = 0
    try:
        bones = arm_obj.data.bones
    except AttributeError:
        return 0
    for b in bones:
        nm = getattr(b, "name", "")
        if nm.startswith(PFX) or nm.startswith(HLP_PFX):
            try:
                b.hide = not bool(visible)
                n += 1
            except Exception:
                pass
    return n


def remove_controls(arm_obj):
    """Удалить слой рычагов (кости CTRL_*/NCPK_* и констрейнты NCP_*)."""
    n = remove_constraints(arm_obj)
    _edit_mode(arm_obj)
    try:
        ebs = arm_obj.data.edit_bones
        orphan_fix = [eb for eb in ebs
                      if eb.parent is not None and
                      (eb.parent.name.startswith(PFX) or
                       eb.parent.name.startswith(HLP_PFX))]
        for eb in [eb for eb in ebs
                   if eb.name.startswith(PFX)
                   or eb.name.startswith(HLP_PFX)]:
            ebs.remove(eb)
        for eb in orphan_fix:
            try:
                eb.parent = None
            except Exception:
                pass
    except Exception:
        pass
    _object_mode()
    for key in ("ncp_ctrl", "ncp_ctrl_sizes"):
        try:
            if arm_obj.get(key) is not None:
                del arm_obj[key]
        except Exception:
            pass
    return n


# ------------------------------------------------------- состояние

def has_handles(arm_obj):
    try:
        return any(pb.name.startswith(PFX)
                   for pb in arm_obj.pose.bones)
    except Exception:
        return False


def handle_names(arm_obj):
    try:
        return [pb.name for pb in arm_obj.pose.bones
                if pb.name.startswith(PFX)]
    except Exception:
        return []


def set_ik(arm_obj, on):
    """Вкл/выкл IK-рычаги: influence всех констрейнтов NCP_*.

    Влияет только на цепочки с констрейнтами (руки/ноги/крылья/хвост);
    Root/Body — прямое parenting, двигаются всегда."""
    val = 1.0 if on else 0.0
    try:
        for pb in arm_obj.pose.bones:
            for c in pb.constraints:
                if c.name.startswith(CONS_PFX):
                    c.influence = val
    except Exception:
        pass
    try:
        arm_obj["ncp_ik"] = bool(on)
    except Exception:
        pass


def ik_on(arm_obj):
    try:
        for pb in arm_obj.pose.bones:
            for c in pb.constraints:
                if c.name.startswith(CONS_PFX) and \
                        float(getattr(c, "influence", 0.0)) > 0.01:
                    return True
    except Exception:
        pass
    return False


def set_deform_hidden(arm_obj, on):
    """Спрятать остовные кости (в вьюпорте остаются только рычаги)."""
    try:
        for b in arm_obj.data.bones:
            b.hide = bool(on) and not b.name.startswith(PFX)
    except Exception:
        pass
    try:
        arm_obj["ncp_deform_hidden"] = bool(on)
    except Exception:
        pass


def reset_helpers(arm_obj):
    """Сбросить позу IK-хелперов в rest (влияние COPY_TRANSFORMS
    при выключенном IK = 0, но хелперы могли остаться в позированной
    позиции после ручной анимации — сброс перед нейроанимацией)."""
    n = 0
    try:
        for pb in arm_obj.pose.bones:
            if not pb.name.startswith(HLP_PFX):
                continue
            for attr, val in (("location", (0.0, 0.0, 0.0)),
                              ("rotation_quaternion", (1.0, 0.0, 0.0,
                                                       0.0)),
                              ("scale", (1.0, 1.0, 1.0))):
                try:
                    setattr(pb, attr, val)
                except Exception:
                    pass
            n += 1
    except Exception:
        pass
    return n


def select_handles(context, arm_obj):
    """Выделить все рычаги и включить Pose mode."""
    _activate(arm_obj)
    try:
        bpy.ops.object.mode_set(mode="POSE")
    except RuntimeError:
        pass
    first = None
    try:
        for pb in arm_obj.pose.bones:
            pb.select = pb.name.startswith(PFX)
            if first is None and pb.name.startswith(PFX):
                first = pb
        if first is not None:
            arm_obj.pose.bones.active = first
    except Exception:
        pass
    return handle_names(arm_obj)


# --------------------------------- режим: T-поза / анимация (2.9.0)

def reset_pose(arm_obj):
    """Сброс позы в rest (T-поза): обнулить location/rotation/scale
    всех рычагов и IK-хелперов. Остовные кости не трогаем — при
    выключенном IK они свободны и уже стоят как записано.
    Возвращает число сброшенных костей."""
    n = 0
    try:
        for pb in arm_obj.pose.bones:
            if not (pb.name.startswith(PFX) or
                    pb.name.startswith(HLP_PFX)):
                continue
            for attr, val in (("location", (0.0, 0.0, 0.0)),
                              ("rotation_quaternion",
                               (1.0, 0.0, 0.0, 0.0)),
                              ("scale", (1.0, 1.0, 1.0))):
                try:
                    setattr(pb, attr, val)
                except Exception:
                    pass
            n += 1
    except Exception:
        pass
    return n


def set_mode(arm_obj, mode):
    """Понятный переключатель «T-поза / движения».

    TPOSE — IK выключен, поза рычагов сброшена в rest: персонаж стоит
           в своей T-позе, как построен;
    ANIM  — IK включён: тяните рычаги (кольца суставов, диски
           пальцев), персонаж двигается.
    Возвращает строку отчёта."""
    mode = "TPOSE" if str(mode).upper().startswith("T") else "ANIM"
    if mode == "TPOSE":
        set_ik(arm_obj, False)
        n = reset_pose(arm_obj)
        rep = "T-поза: IK выкл, рычаги сброшены (%d)" % n
    else:
        set_ik(arm_obj, True)
        rep = "Анимация: IK вкл — тяните рычаги"
    try:
        arm_obj["ncp_mode"] = mode
    except Exception:
        pass
    return rep


def get_mode(arm_obj):
    try:
        return arm_obj.get("ncp_mode") or "ANIM"
    except Exception:
        return "ANIM"


# ------------------------------------ запись движений по слоям (2.9.0)

# слои записи: какие рычаги попадают в кейфреймы
REC_LAYERS = ("body", "arms", "legs", "fingers", "wings")

_REC = {"on": False, "snap": {}}


def rec_layer_of(name):
    """Слой записи для рычага (или None — не рычаг)."""
    if name.startswith("CTRL_Finger_"):
        return "fingers"
    if name.startswith(("CTRL_Arm", "CTRL_Pole_Elbow")):
        return "arms"
    if name.startswith(("CTRL_Foot", "CTRL_Pole_Knee")):
        return "legs"
    if name.startswith("CTRL_Wing"):
        return "wings"
    if name.startswith(("CTRL_Root", "CTRL_Body", "CTRL_Head",
                        "CTRL_Tail")):
        return "body"
    return None


def rec_layers_state(arm_obj):
    """Включённые слои записи (хранятся на арматуре; по умолчанию все)."""
    st = {k: True for k in REC_LAYERS}
    try:
        raw = arm_obj.get("ncp_rec_layers")
        if raw:
            for k, v in dict(raw).items():
                if k in st:
                    st[k] = bool(v)
    except Exception:
        pass
    return st


def rec_set_state(arm_obj, state):
    """Сохранить включённые слои записи на арматуру."""
    try:
        arm_obj["ncp_rec_layers"] = {
            k: bool(v) for k, v in dict(state).items() if k in REC_LAYERS}
    except Exception:
        pass


def rec_layer_bones(arm_obj, enabled=None):
    """Рычаги включённых слоёв записи."""
    if enabled is None:
        enabled = rec_layers_state(arm_obj)
    out = []
    try:
        for pb in arm_obj.pose.bones:
            lay = rec_layer_of(pb.name)
            if lay is not None and enabled.get(lay):
                out.append(pb)
    except Exception:
        pass
    return out


def record_frame(arm_obj, enabled=None, frame=None):
    """Кейфрейм текущей позы рычагов включённых слоёв («Записать кадр»
    и автозапись). Кадр по умолчанию — текущий. Возвращает число
    закейфрейменных костей."""
    if frame is None:
        try:
            frame = int(bpy.context.scene.frame_current)
        except Exception:
            frame = 1
    if enabled is None:
        enabled = rec_layers_state(arm_obj)
    n = 0
    try:
        for pb in rec_layer_bones(arm_obj, enabled):
            ok = False
            for dp in ("location", "rotation_quaternion"):
                try:
                    pb.keyframe_insert(dp, frame=frame)
                    ok = True
                except Exception:
                    pass
            if ok:
                n += 1
    except Exception:
        pass
    return n


def _rec_snapshot(pbs):
    """Подпись позы рычагов (круг до 4 знаков) — чтобы запись шла
    только когда поза реально изменилась (скролл кадров не пишет)."""
    sig = []
    for pb in pbs:
        try:
            loc = tuple(round(float(v), 4) for v in pb.location)
        except Exception:
            loc = ()
        try:
            rot = tuple(round(float(v), 4)
                        for v in pb.rotation_quaternion)
        except Exception:
            rot = ()
        sig.append((pb.name, loc, rot))
    return tuple(sig)


def _rec_tick(scene, depsgraph):
    """depsgraph-хэндлер автозаписи: поза рычагов включённых слоёв
    изменилась — кейфрейм на текущем кадре (влияние констрейнтов
    NCP_* уже в костях, пишем итоговую позу)."""
    try:
        if not _REC.get("on"):
            return
        try:
            arms = [o for o in scene.objects
                    if getattr(o, "type", "") == "ARMATURE"]
        except Exception:
            return
        for arm_obj in arms:
            try:
                if not arm_obj.get("ncp_ctrl"):
                    continue
                enabled = rec_layers_state(arm_obj)
                pbs = rec_layer_bones(arm_obj, enabled)
                if not pbs:
                    continue
                sig = _rec_snapshot(pbs)
                if _REC["snap"].get(arm_obj.name) == sig:
                    continue
                _REC["snap"][arm_obj.name] = sig
                record_frame(arm_obj, enabled)
            except Exception:
                pass
    except Exception:
        pass


def record_is_enabled():
    return bool(_REC.get("on"))


def record_set_enabled(on):
    """Вкл/выкл автозапись по слоям (depsgraph-хэндлер). Возвращает
    фактическое состояние (False, если bpy.app.handlers недоступен)."""
    handlers = getattr(bpy.app, "handlers", None)
    lst = getattr(handlers, "depsgraph_update_post", None) \
        if handlers is not None else None
    if lst is None:
        _REC["on"] = False
        return False
    on = bool(on)
    if on and not _REC.get("on"):
        try:
            if _rec_tick not in lst:
                lst.append(_rec_tick)
        except Exception:
            pass
    elif not on and _REC.get("on"):
        try:
            if _rec_tick in lst:
                lst.remove(_rec_tick)
        except Exception:
            pass
    _REC["on"] = on
    if not on:
        _REC["snap"] = {}
    return _REC["on"]
