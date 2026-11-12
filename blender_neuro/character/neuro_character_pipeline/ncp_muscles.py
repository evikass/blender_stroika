# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.ncp_muscles — мост к X-Muscle System.

Единица мышцы у X-Muscle = пара костей (начало + крепление), создание
только в POSE-режиме с упорядоченным выделением. Мы автоматизируем это
для всех пар neuro-скелета (13 мышц-групп) и затем одним вызовом
пришиваем их к коже (apply_musculature).
"""
import bpy
import importlib
import sys


# (origin, insertion, имя мышцы, суффикс стороны)
MUSCLE_PAIRS = [
    ("pelvis",  "spine",   "Spine", ""),
    ("spine",   "chest",   "Chest", ""),
    ("chest",   "head",    "Neck",  ""),
    ("pelvis",  "thigh_L", "Hip",   ".L"),
    ("thigh_L", "shin_L",  "Knee",  ".L"),
    ("shin_L",  "foot_L",  "Ankle", ".L"),
    ("pelvis",  "thigh_R", "Hip",   ".R"),
    ("thigh_R", "shin_R",  "Knee",  ".R"),
    ("shin_R",  "foot_R",  "Ankle", ".R"),
    ("chest",   "uarm_L",  "Shldr", ".L"),
    ("uarm_L",  "farm_L",  "Elbow", ".L"),
    ("chest",   "uarm_R",  "Shldr", ".R"),
    ("uarm_R",  "farm_R",  "Elbow", ".R"),
]

_STYLE_OP = {
    "STYLIZED": "add_muscle",
    "BASIC": "add_basic_muscle",
    "STRIP": "add_strip_muscle",
}


def resolve():
    """Модуль X-Muscle System: сперва вшитая копия (vendor.xmuscle),
    затем раздельный аддон (обычный или bl_ext.*). Не путаем с
    X-Muscle Pie Menu (в его ключе тоже есть 'xmuscle')."""
    try:
        return importlib.import_module(".vendor.xmuscle", __package__)
    except Exception:
        pass
    try:
        keys = list(bpy.context.preferences.addons.keys())
    except Exception:
        return None
    for k in keys:
        lk = k.lower()
        if "xmuscle" in lk and "piemenu" not in lk:
            return sys.modules.get(k)
    return None


def installed():
    return resolve() is not None


def scene_ready(scn):
    """Scene-пропсы X-Muscle появляются после enable аддона."""
    return getattr(scn, "Muscle_Name", None) is not None


def muscle_objects():
    return [o for o in bpy.data.objects
            if getattr(o, "Muscle_XID", False)]


def create_all(context, arm_obj):
    """Авто-мышцы на все пары костей. Возвращает (создано, всего, ошибки)."""
    scn = context.scene
    st = scn.ncp
    if not scene_ready(scn):
        raise RuntimeError("X-Muscle System не активирован "
                           "(включите галочку в Preferences -> Add-ons)")
    fn = _STYLE_OP.get(st.muscle_style, "add_muscle")
    op = getattr(bpy.ops.muscle, fn)

    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    before = len(muscle_objects())
    created, errors = 0, []
    for origin, insert, name, side in MUSCLE_PAIRS:
        pb_o = arm_obj.pose.bones.get(origin)
        pb_i = arm_obj.pose.bones.get(insert)
        if pb_o is None or pb_i is None:
            errors.append("%s->%s: нет кости" % (origin, insert))
            continue
        try:
            bpy.ops.pose.select_all(action="DESELECT")
            scn.Muscle_Name = name
            scn.Prefix = "XMSL_"
            scn.Suffix = side
            scn.use_Affixes = True
            scn.Create_Type = "AUTOAIM"
            pb_o.select = True
            pb_i.select = True
            pb_o["selection_order"] = 0
            pb_i["selection_order"] = 1
            arm_obj.pose.bones.active = pb_i
            op('EXEC_DEFAULT')
            created += 1
        except Exception as e:
            errors.append("%s->%s: %s" % (origin, insert, e))
        finally:
            for pb in (pb_o, pb_i):
                try:
                    if "selection_order" in pb:
                        del pb["selection_order"]
                except Exception:
                    pass
                pb.select = False

    bpy.ops.object.mode_set(mode="OBJECT")
    return created, len(muscle_objects()) - before, errors


def attach_to_skin(context, skin):
    """Пришить все мышцы к коже: shrinkwrap + веса + драйверы."""
    if skin is None:
        raise RuntimeError("Не найден меш кожи (Character_Mesh)")
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    n = 0
    for m in muscle_objects():
        m.select_set(True)
        n += 1
    if n == 0:
        raise RuntimeError("Мышцы не созданы")
    skin.select_set(True)
    context.view_layer.objects.active = skin
    bpy.ops.muscle.apply_musculature()
    return n


def enable_dynamics():
    """Включить softbody-динамику мышц во вьюпорте и рендере."""
    on = 0
    for m in muscle_objects():
        try:
            m.Dynamics_View3D = True
            m.Dynamics_Render = True
            on += 1
        except Exception:
            pass
    return on


def bake_caches(context):
    """Пересобрать/запечь softbody-кэши всех мышц (после анимации)."""
    bpy.ops.muscle.smart_update()
