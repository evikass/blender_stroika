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
  CTRL_Head  — голова/шея;  CTRL_Tail — кончик хвоста.

Правила безопасности анимации (то, ради чего слой сделан отдельным):
  1. CTRL-кости use_deform=False — в скининг и веса не попадают,
     меш они не деформируют;
  2. IK/COPY_ROTATION-констрейнты называются NCP_*: influence ставит
     set_ik() — галка «IK handles» в панели. Import Neuro Motion сам
     гасит IK, чтобы нейроанимация играла как записана;
  3. остовные (deform) кости можно спрятать одной галкой — в вьюпорте
     остаются только рычаги (кликать легко, анимировать не мешает).

Иерархия: CTRL_Root -> CTRL_Body -> pelvis. Root/Body/Head/Tail —
прямое parenting (двигаются всегда, никаких констрейнтов); Arm/Wing/
Foot — IK-цели, parent CTRL_Root (при движении тела ноги остаются
приколотыми к ступням — можно присаживать персонажа).

Топология рычагов строится по ФАКТУ арматуры: цепочки конечностей
находятся обходом дерева костей (то же дерево, что построил
neuro_creature.analyze), поэтому слой работает для любого плана —
человек, четвероногий, птица, дракон, двуногий с хвостом.
"""
import math
import re

import bpy
from mathutils import Vector

PFX = "CTRL_"
CONS_PFX = "NCP_"

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
}


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

_SHAPE_DISC = (
    [(1.0, 0.0, 0.0)] + [
        (math.cos(6.2831853 * i / 24.0),
         math.sin(6.2831853 * i / 24.0), 0.0)
        for i in range(24)
    ],
    [list(range(24))],
)
_SHAPE_BOX = (
    [(-0.6, -0.6, -0.6), (0.6, -0.6, -0.6), (0.6, 0.6, -0.6),
     (-0.6, 0.6, -0.6), (-0.6, -0.6, 0.6), (0.6, -0.6, 0.6),
     (0.6, 0.6, 0.6), (-0.6, 0.6, 0.6)],
    [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
     (2, 3, 7, 6), (3, 0, 4, 7)],
)
SHAPES = {"disc": ("NCP_Shape_Disc", _SHAPE_DISC),
          "box": ("NCP_Shape_Box", _SHAPE_BOX)}


def _get_shape(kind):
    """Объект-форма для custom_shape (скрытый; None = косметика не удалась)."""
    name, geo = SHAPES[kind]
    try:
        me = bpy.data.meshes.get(name)
        if me is None:
            me = bpy.data.meshes.new(name)
            me.from_pydata(geo[0], [], geo[1])
        ob = bpy.data.objects.get(name)
        if ob is None:
            ob = bpy.data.objects.new(name, me)
            ob.hide_render = True
            ob.hide_select = True
            try:
                ob.hide_viewport = True
            except AttributeError:
                pass
            try:
                bpy.context.scene.collection.objects.link(ob)
            except Exception:
                pass
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
        pb.custom_shape_scale_uniform = size
    except Exception:
        try:
            pb.custom_shape_scale = size
        except Exception:
            pass


def _set_color(pb, rgba):
    try:
        pb.color.palette = "CUSTOM"
        pb.color.custom = rgba
    except Exception:
        pass


def _assign_coll(data, bone_name, coll_name):
    """Bone collections (Blender 4+); старые версии и стабы — молча нет."""
    try:
        colls = data.collections
    except AttributeError:
        return
    try:
        coll = colls.get(coll_name)
        if coll is None:
            coll = colls.new(coll_name)
        coll.assign(bone_name)
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
    return "leg" if handle_name.startswith("CTRL_Foot") else \
        "head" if handle_name.startswith("CTRL_Head") else \
        "wing" if handle_name.startswith("CTRL_Wing") else \
        "tail" if handle_name.startswith("CTRL_Tail") else "arm"


def build_controls(arm_obj, ik_on=True, hide_deform=False):
    """Собрать слой рычагов. Возвращает короткий отчёт (строка).

    Ломать риг не может: каждая косметическая вещь (формы, цвета,
    коллекции) обёрнута в try/except."""
    if arm_obj is None or getattr(arm_obj, "type", "") != "ARMATURE":
        raise ControlsError("нет арматуры")
    remove_controls(arm_obj)                 # пересборка на чистую

    chains = find_chains(arm_obj)

    # -- сессия редактирования: CTRL-кости + пере-parenting корня
    made = []
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
        for ch in chains:
            ebs_c = [ebs.get(n) for n in ch["bones"]]
            if any(e is None for e in ebs_c):
                continue
            tip = ebs_c[-1]
            d = _dir_of(tip)
            anchor = tip.head if ch["kind"] == "leg" else tip.tail
            name = _handle_name(ch["kind"], ch["side"], ch["qual"], used)
            par = "CTRL_Body" if ch["kind"] in ("head", "tail", "wing") \
                else "CTRL_Root"
            new_ctrl(name, (anchor.x, anchor.y, anchor.z),
                     anchor + d * max(0.10 * h, tip.length * 0.6),
                     parent=par)
            ch["handle"] = name
            kind_sizes[name] = max(0.09 * h, min(tip.length, 0.25 * h))
    finally:
        _object_mode()

    # -- констрейнты (influence ставит set_ik; по умолчанию из ik_on)
    for ch in chains:
        handle = ch.get("handle")
        if not handle:
            continue
        pbs = [arm_obj.pose.bones.get(n) for n in ch["bones"]]
        if any(pb is None for pb in pbs):
            continue
        if ch["kind"] == "leg":
            ik_names = [n for n in ch["bones"][:-1]
                        if not n.startswith(_CONNECTORS)]
            ik_names = [n for n in ik_names
                        if lens.get(n, 0.0) >= 0.07 * h]
        else:
            ik_names = [n for n in ch["bones"]
                        if not n.startswith(_CONNECTORS)]
        if not ik_names:
            continue
        last = arm_obj.pose.bones.get(ik_names[-1])
        if last is not None:
            try:
                c = last.constraints.new("IK")
                c.name = CONS_PFX + "IK_" + handle
                c.target = arm_obj
                c.subtarget = handle
                c.chain_count = min(3, len(ik_names))
                try:
                    c.use_stretch = False
                except Exception:
                    pass
                try:
                    c.use_tail = False
                except Exception:
                    pass
            except Exception:
                pass
        # ступня смотрит туда же, куда повернули рычаг
        if ch["kind"] == "leg":
            foot_pb = pbs[-1]
            try:
                c = foot_pb.constraints.new("COPY_ROTATION")
                c.name = CONS_PFX + "ROT_" + handle
                c.target = arm_obj
                c.subtarget = handle
            except Exception:
                pass

    # -- косметика: формы, цвета, коллекции, видимость
    pose = arm_obj.pose.bones
    for name, kind in (("CTRL_Root", "root"), ("CTRL_Body", "body")):
        try:
            pb = pose.get(name)
            if pb is not None:
                _set_shape(pb, "box" if kind == "root" else "disc",
                           0.16 * h if kind == "root" else 0.12 * h)
                _set_color(pb, _KIND_COLOR[kind])
        except Exception:
            pass
    for name in made[2:]:
        try:
            pb = pose.get(name)
            if pb is None:
                continue
            _set_shape(pb, "disc", kind_sizes.get(name, 0.09 * h))
            _set_color(pb, _KIND_COLOR[_bone_kind(name)])
        except Exception:
            pass
        _assign_coll(arm_obj.data, name, "NCP_Controls")
    for pb in pose:
        if not pb.name.startswith(PFX):
            _assign_coll(arm_obj.data, pb.name, "NCP_Deform")
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
    handles = ", ".join(made[2:])
    return "рычаги: %d (Root, Body%s%s)" % (
        len(made), ", " + handles if handles else "",
        "; IK выкл" if not ik_on else "")


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


def remove_controls(arm_obj):
    """Удалить слой рычагов (кости CTRL_* и констрейнты NCP_*)."""
    n = remove_constraints(arm_obj)
    _edit_mode(arm_obj)
    try:
        ebs = arm_obj.data.edit_bones
        orphan_fix = [eb for eb in ebs
                      if eb.parent is not None and
                      eb.parent.name.startswith(PFX)]
        for eb in [eb for eb in ebs if eb.name.startswith(PFX)]:
            ebs.remove(eb)
        for eb in orphan_fix:
            try:
                eb.parent = None
            except Exception:
                pass
    except Exception:
        pass
    _object_mode()
    try:
        if arm_obj.get("ncp_ctrl") is not None:
            del arm_obj["ncp_ctrl"]
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
