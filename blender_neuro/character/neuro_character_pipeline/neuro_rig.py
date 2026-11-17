# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.neuro_rig — neuro-скелет, подгонка меша, скининг, анимация.

Функции build_armature / _link_frames / apply_animation вендорены из
blender_neuro_importer.py (контракт имён костей = имена звеньев human.json).
Изменения: только обвязка (подгонка меша, скининг, импорт на готовую арматуру).
"""
import copy
import json
import math
import os

import bpy
from mathutils import Matrix, Quaternion, Vector

try:
    import numpy as np
except Exception:                        # numpy не мешает остальному ригу
    np = None


# --------------------------------------------------------- резервный спек

_FALLBACK_SPEC = {
    "name": "human_v1",
    "root": "pelvis",
    "links": [
        {"name": "pelvis", "parent": None, "joint": None, "com": [0, 0, 1.0],
         "shape": {"type": "capsule", "axis": "x", "radius": 0.09, "half_len": 0.06}},
        {"name": "spine", "parent": "pelvis", "com": [0, 0, 1.16],
         "joint": {"anchor": [0, 0, 1.06], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.085, "half_len": 0.09}},
        {"name": "chest", "parent": "spine", "com": [0, 0, 1.33],
         "joint": {"anchor": [0, 0, 1.24], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.105, "half_len": 0.12}},
        {"name": "head", "parent": "chest", "com": [0, 0, 1.56],
         "joint": {"anchor": [0, 0, 1.46], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.09, "half_len": 0.05}},
        {"name": "shoulderP_L", "parent": "chest", "com": [0, 0.26, 1.40],
         "joint": {"anchor": [0, 0.20, 1.40], "axis": [0, 0, 1]},
         "shape": {"type": "capsule", "axis": "y", "radius": 0.035, "half_len": 0.035}},
        {"name": "uarm_L", "parent": "shoulderP_L", "com": [0, 0.45, 1.40],
         "joint": {"anchor": [0, 0.30, 1.40], "axis": [1, 0, 0]},
         "shape": {"type": "capsule", "axis": "y", "radius": 0.048, "half_len": 0.13}},
        {"name": "farm_L", "parent": "uarm_L", "com": [0, 0.76, 1.40],
         "joint": {"anchor": [0, 0.60, 1.40], "axis": [0, 0, 1]},
         "shape": {"type": "capsule", "axis": "y", "radius": 0.042, "half_len": 0.14}},
        {"name": "hipR_L", "parent": "pelvis", "com": [0, 0.11, 0.94],
         "joint": {"anchor": [0, 0.11, 0.94], "axis": [1, 0, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.04, "half_len": 0.04}},
        {"name": "thigh_L", "parent": "hipR_L", "com": [0, 0.11, 0.72],
         "joint": {"anchor": [0, 0.11, 0.92], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.075, "half_len": 0.18}},
        {"name": "shin_L", "parent": "thigh_L", "com": [0, 0.11, 0.34],
         "joint": {"anchor": [0, 0.11, 0.54], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.058, "half_len": 0.18}},
        {"name": "foot_L", "parent": "shin_L", "com": [0.06, 0.11, 0.04],
         "joint": {"anchor": [0, 0.11, 0.16], "axis": [0, 1, 0]},
         "shape": {"type": "box", "half": [0.12, 0.05, 0.04]}},
        {"name": "shoulderP_R", "parent": "chest", "com": [0, -0.26, 1.40],
         "joint": {"anchor": [0, -0.20, 1.40], "axis": [0, 0, 1]},
         "shape": {"type": "capsule", "axis": "y", "radius": 0.035, "half_len": 0.035}},
        {"name": "uarm_R", "parent": "shoulderP_R", "com": [0, -0.45, 1.40],
         "joint": {"anchor": [0, -0.30, 1.40], "axis": [1, 0, 0]},
         "shape": {"type": "capsule", "axis": "y", "radius": 0.048, "half_len": 0.13}},
        {"name": "farm_R", "parent": "uarm_R", "com": [0, -0.76, 1.40],
         "joint": {"anchor": [0, -0.60, 1.40], "axis": [0, 0, 1]},
         "shape": {"type": "capsule", "axis": "y", "radius": 0.042, "half_len": 0.14}},
        {"name": "hipR_R", "parent": "pelvis", "com": [0, -0.11, 0.94],
         "joint": {"anchor": [0, -0.11, 0.94], "axis": [1, 0, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.04, "half_len": 0.04}},
        {"name": "thigh_R", "parent": "hipR_R", "com": [0, -0.11, 0.72],
         "joint": {"anchor": [0, -0.11, 0.92], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.075, "half_len": 0.18}},
        {"name": "shin_R", "parent": "thigh_R", "com": [0, -0.11, 0.34],
         "joint": {"anchor": [0, -0.11, 0.54], "axis": [0, 1, 0]},
         "shape": {"type": "capsule", "axis": "z", "radius": 0.058, "half_len": 0.18}},
        {"name": "foot_R", "parent": "shin_R", "com": [0.06, -0.11, 0.04],
         "joint": {"anchor": [0, -0.11, 0.16], "axis": [0, 1, 0]},
         "shape": {"type": "box", "half": [0.12, 0.05, 0.04]}},
    ],
}


def load_spec(user_path=""):
    """Спек существа: файл пользователя -> presets/human.json -> встроенный."""
    if user_path and os.path.isfile(user_path):
        with open(user_path, "r", encoding="utf-8") as f:
            return json.load(f)
    here = os.path.dirname(os.path.abspath(__file__))
    cands = (
        os.path.join(here, "..", "..", "presets", "human.json"),
        os.path.join(here, "..", "presets", "human.json"),
        os.path.join(here, "presets", "human.json"),
        os.path.join(here, "human.json"),
    )
    for c in cands:
        if os.path.isfile(c):
            with open(c, "r", encoding="utf-8") as f:
                return json.load(f)
    return copy.deepcopy(_FALLBACK_SPEC)


def spec_height(spec):
    """Рост существа по спеку (метры): для human_v1 = 1.70."""
    top, bot = -1e9, 1e9
    for l in spec["links"]:
        sh, com = l["shape"], l["com"]
        ax = sh.get("axis", "z")
        if sh["type"] == "capsule":
            ez = (sh["half_len"] + sh["radius"]) if ax == "z" else sh["radius"]
        elif sh["type"] == "box":
            ez = sh["half"][2]
        else:
            ez = sh["radius"]
        top = max(top, com[2] + ez)
        bot = min(bot, com[2] - ez)
    return top - bot


# ------------------------------------------------- арматура (вендоринг)

def build_armature(spec, name="NeuroCreature"):
    """Арматура neuro-существа. Имена костей = имена звеньев спеки."""
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for l in spec["links"]:
        eb = arm_data.edit_bones.new(l["name"])
        if l["parent"] is None:
            head = Vector(l["com"]) - Vector((0, 0, 0.12))
            tail = Vector(l["com"]) + Vector((0, 0, 0.03))
        else:
            head = Vector(l["joint"]["anchor"])
            tail = Vector(l["com"])
            if (tail - head).length < 0.02:
                tail = head + Vector((0, 0, 0.1))
        eb.head, eb.tail = head, tail
        eb.roll = 0.0
        edit_bones[l["name"]] = eb
    for l in spec["links"]:
        if l["parent"] is not None:
            edit_bones[l["name"]].parent = edit_bones[l["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")

    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm_obj


def _link_frames(spec, root_pos, root_quat, joint_angles):
    """Мировые 4x4 матрицы рамок звеньев. root_quat — [x,y,z,w] (pybullet)."""
    by_name = {l["name"]: l for l in spec["links"]}
    frames = {}
    for l in spec["links"]:
        if l["parent"] is None:
            R = Quaternion((root_quat[3], root_quat[0], root_quat[1],
                            root_quat[2])).to_matrix().to_4x4()
            frames[l["name"]] = Matrix.Translation(Vector(root_pos)) @ R
            continue
        j = l["joint"]
        par = by_name[l["parent"]]
        par_frame = (par["com"] if par["parent"] is None
                     else par["joint"]["anchor"])
        l_a = Vector(j["anchor"]) - Vector(par_frame)
        rot = Matrix.Rotation(float(joint_angles.get(l["name"], 0.0)), 4,
                              Vector(j["axis"]))
        frames[l["name"]] = (frames[par["name"]] @
                             Matrix.Translation(l_a) @ rot)
    return frames


def apply_animation(arm_obj, spec, frames, fps=48.0, frame_step=1):
    """FK-реплей движения в позы костей с кейфреймами (вендоринг импортёра)."""
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))

    rest = {pb.name: (arm_obj.matrix_world.inverted() @ pb.bone.matrix_local)
            for pb in arm_obj.pose.bones}

    act = arm_obj.animation_data_create()
    act.action = bpy.data.actions.new("neuro_motion")

    joint_names = [l["name"] for l in spec["links"] if l.get("joint")]
    n_frames = len(frames)
    for fi, fr in enumerate(frames):
        f = fi * frame_step + 1
        jd = dict(zip(joint_names, [float(a) for a in fr["joints"]]))
        frames_i = _link_frames(spec, fr["root_pos"], fr["root_quat"], jd)
        for l in spec["links"]:
            pb = arm_obj.pose.bones.get(l["name"])
            if pb is None:
                continue
            F = frames_i[l["name"]]
            R_rest = rest[l["name"]].to_3x3()
            if l["parent"] is None:
                head_local = Vector((0, 0, -0.12))
            else:
                head_local = Vector((0, 0, 0))
            pos = (F @ Matrix.Translation(head_local)).to_translation()
            M = Matrix.Translation(pos) @ (F.to_3x3() @ R_rest).to_4x4()
            pb.matrix = M
            pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("rotation_quaternion", frame=f)
    scene.frame_start = 1
    scene.frame_end = max(2, n_frames * frame_step)
    scene.frame_set(1)


def apply_motion_to_armature(arm_obj, payload, frame_step=1):
    """Neuro motion JSON -> готовая арматура. Возвращает (n_frames, fps).
    Если арматура подогнана под тело (Auto-Rig, custom prop ncp_height),
    движение ретаргетится по высоте: корень масштабируется
    fitted_height / высота спеки из JSON (кадры двигаются вместе с
    персонажем, ноги не проваливаются/не висят)."""
    spec = payload["spec"]
    frames = payload["frames"]
    fps = float(payload.get("fps", 48.0))
    step = max(1, int(frame_step))
    if step > 1:
        frames = frames[::step]
    try:
        fitted_h = float(arm_obj.get("ncp_height", 0.0) or 0.0)
    except Exception:
        fitted_h = 0.0
    if fitted_h > 0.01:
        base_h = spec_height(spec)
        if base_h > 0.01:
            frames = scale_frames_root(frames, fitted_h / base_h)
    apply_animation(arm_obj, spec, frames, fps=fps / step, frame_step=1)
    return len(frames), fps / step


def scale_frames_root(frames, k):
    """Масштаб корневой траектории кадров (ретаргет по росту)."""
    if abs(k - 1.0) < 1e-6:
        return frames
    out = []
    for fr in frames:
        fr2 = dict(fr)
        fr2["root_pos"] = [c * k for c in fr["root_pos"]]
        out.append(fr2)
    return out


# -------------------------------------------------------- меш и скининг

def _world_bbox(obj):
    pts = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts),
                 min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts),
                 max(p.z for p in pts)))
    return mn, mx


def orientation_from_extents(ex, ey, ez):
    """(rot_x_deg, rot_z_deg) по габаритам меша, чтобы он стоял (Z-up)
    и смотрел вдоль +X, как neuro-скелет.

    У T-pose гуманоида глубина (спина-грудь) — самая маленькая сторона
    bbox. Она и есть "взгляд": вдоль X — уже смотрит +X; вдоль Y —
    повернуть на +90 (Blender-фронт -Y -> +X); вдоль Z — меш лежит на
    спине (Y-up движка), поднять на +90 вокруг X и довернуть на +90.
    Ничего не знаем о знаке (лицо/затылок неразличимы по bbox) — если
    угадали назад, пользователю достаточно Mesh rot Z = -90."""
    vals = (("x", ex), ("y", ey), ("z", ez))
    depth = min(vals, key=lambda kv: kv[1])[0]
    if depth == "x":
        return 0.0, 0.0
    if depth == "y":
        return 0.0, 90.0
    return 90.0, 90.0


def guess_orientation(mesh):
    """Авто-углы для меша по его мировому bbox."""
    mn, mx = _world_bbox(mesh)
    return orientation_from_extents(mx.x - mn.x, mx.y - mn.y, mx.z - mn.z)


def fit_mesh_to_skeleton(mesh, spec, rot_x_deg=0.0, rot_z_deg=0.0,
                         do_apply=True):
    """Масштаб/позиция меша под neuro-скелет: повороты X/Z (ориентация),
    рост по спеку, низ в z=0, центр xy в (0,0). Скелет не трогаем
    (контракт анимации в метрах)."""
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    if abs(rot_x_deg) > 1e-6:
        mesh.rotation_euler.x += math.radians(rot_x_deg)
    if abs(rot_z_deg) > 1e-6:
        mesh.rotation_euler.z += math.radians(rot_z_deg)
    bpy.context.view_layer.update()

    h = spec_height(spec)
    mn, mx = _world_bbox(mesh)
    if mx.z - mn.z < 1e-6:
        raise RuntimeError("У меша нулевая высота bbox")
    s = h / (mx.z - mn.z)
    mesh.scale = tuple(c * s for c in mesh.scale)
    bpy.context.view_layer.update()

    mn, mx = _world_bbox(mesh)
    cx, cy = (mn.x + mx.x) / 2.0, (mn.y + mx.y) / 2.0
    mesh.location.x += -cx
    mesh.location.y += -cy
    mesh.location.z += -mn.z
    bpy.context.view_layer.update()

    if do_apply:
        bpy.ops.object.select_all(action="DESELECT")
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.transform_apply(location=False, rotation=True,
                                       scale=True)
    return s


# ------------------------------------------------------------- пальцы

# (имя, сдвиг в ПЛОСКОСТИ ЛАДОНИ (вперёд=+X), прогиб ВНИЗ, длина, м)
# v2.9.0: пальцы растут от ЗАПЯСТЬЯ и веером в плоскости ладони.
# Раньше база ставилась на длину предплечья + 7 см ЗА кисть, а веер
# расходился по мировому Z — ладонь выглядела повёрнутой на 90°, а
# кости пальцев висели в воздухе рядом с рукой (жалобы 2.7/2.8).
FINGER_PLAN = (
    ("thumb", 0.030, 0.012, 0.062),
    ("index", 0.014, 0.005, 0.078),
    ("middle", 0.000, 0.000, 0.084),
    ("ring", -0.014, 0.005, 0.078),
    ("pinky", -0.028, 0.010, 0.062),
)


def finger_bone_names():
    """Имена 20 добавочных костей пальцев (в спеке human.json их нет)."""
    out = []
    for side in ("L", "R"):
        for fname, _off_p, _off_n, _ln in FINGER_PLAN:
            out.append("fng_%s_1_%s" % (fname, side))
            out.append("fng_%s_2_%s" % (fname, side))
    return out


def add_finger_chains(arm_obj, scale=1.0, finger_paths=None, mesh_obj=None):
    """По 10 костей пальцев от запястий farm_L/farm_R.

    v2.11.0: приоритет — ПУТИ ИЗ ГЕОМЕТРИИ кисти (dict
    {"paths": [(p_base, p_mid, p_tip), ...], "fan": dict|None} из
    neuro_creature._human_finger_paths; принимается и старый список
    путей): достаточно 3+ путей на сторону (у механических/
    мультяшных кистей может читаться 4 пальца — веер хуже).
    Если пальцы не разделились (варежка, перчатка), но кисть найдена
    — веер рисуется по РАМКЕ КИСТИ fan: ось пальцев d (запястье->
    кончик) и ось развода u из PCA геометрии. Это чинит главный
    дефект анатомического веера: у персонажа с руками вниз пальцы
    костей смотрели вдоль предплечья «ладонью в пол», а настоящие
    пальцы меша — в сторону. И только если кисти в геометрии нет
    вовсе — старый анатомический веер вдоль farm.

    СННАП НА МЕШ: если передан mesh_obj, точки путей, отстоящие от
    поверхности кисти не дальше 0.06*h, подтягиваются к поверхности
    (внутрь на 0.002*h) — кости пальцев лежат внутри пальцев, а не
    висят в воздухе у ладони.

    Анатомический веер (последний фолбак, без геометрии):
    нейро-контракт, лицо +X, руки в T-pose вдоль ±Y, ладони вниз.
    Направление пальцев = вдоль предплечья (arm_dir); веер
    расходится поперёк руки в горизонтали, развёрнутой к лицу (+X):
    большой палец вперёд, мизинец назад; лёгкий прогиб всех пальцев
    вниз. Направления считаются от фактической кости farm_*.
    neuro-анимация (apply_animation) ведёт только кости спеки,
    пальцы просто едут вместе с рукой. scale — множитель размеров
    анатомического веера (рост персонажа / 1.70 м); пути геометрии
    не масштабируются."""
    arm_data = arm_obj.data
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    h_ref = 1.70 * max(scale, 0.05)          # рост для порогов снапа
    made = 0
    for side, par in (("L", "farm_L"), ("R", "farm_R")):
        eb_p = arm_data.edit_bones.get(par)
        if eb_p is None:
            continue
        arm_dir = eb_p.vector
        if arm_dir.length < 1e-9:
            arm_dir = Vector((1.0, 0.0, 0.0))
        arm_dir = arm_dir.normalized()
        # веер: горизонталь поперёк руки, развёрнутая К ЛИЦУ (+X).
        # T-pose (рука вдоль ±Y): получаем +X с обеих сторон —
        # большой палец вперёд у левой И правой кисти
        spread = Vector((arm_dir.y * 1.0 - arm_dir.z * 0.0,
                         arm_dir.z * 0.0 - arm_dir.x * 1.0,
                         0.0))
        if spread.length < 1e-6:            # рука почти вертикальна
            spread = Vector((1.0, 0.0, 0.0))
        spread = spread.normalized()
        if spread.x < 0.0:
            spread = -spread
        # прогиб вниз (мировая -Z без компоненты вдоль руки);
        # у вертикальной руки вырождается — просто без прогиба
        droop = Vector((0.0, 0.0, -1.0))
        droop = droop - arm_dir * droop.dot(arm_dir)
        droop = droop.normalized() if droop.length > 1e-6 else None
        # -- v2.11.0: пути/рамка кисти из геометрии
        paths = None
        fan = None
        if finger_paths:
            ent = finger_paths.get(side)
            if isinstance(ent, (list, tuple)):
                ent = {"paths": list(ent), "fan": None}
            if ent:
                cand = list(ent.get("paths") or [])
                fan = ent.get("fan") or None
                if len(cand) >= 3:
                    paths = _order_finger_paths(cand, eb_p, arm_dir)
                    fan = None                      # веер не нужен

            # -- v2.12.0: контроль путей относительно РЕАЛЬНОЙ кости
            # farm. Каждый путь обязан висеть ПЕРЕД запястьем вдоль
            # предплечья и не дальше ладони от оси; рамка веера —
            # наружу от запястья и размером с кисть. Нарушение —
            # набор/рамка отбрасываются (анатомический веер от
            # запястья): кости пальцев физически не могут съехать под
            # локоть (жалоба 2.11)
            f_len = max(float((Vector(eb_p.tail) -
                               Vector(eb_p.head)).length), 1e-6)
            wr_v = Vector(eb_p.tail)
            if paths is not None:
                ok = True
                for fp in paths:
                    for q in fp:
                        dv = Vector(q) - wr_v
                        t = dv.dot(arm_dir)
                        lat = (dv - arm_dir * t).length
                        if t < -0.10 * f_len or t > 0.90 * f_len or \
                                lat > 0.85 * f_len:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    paths = None
                    fan = None
            if fan is not None:
                fd = Vector(fan.get("d") or (0.0, 0.0, 0.0))
                L = float(fan.get("L", 0.0))
                w = float(fan.get("w", 0.0))
                h_lim = 1.70 * max(scale, 0.05)
                if fd.length < 1e-9 or \
                        float(fd.normalized().dot(arm_dir)) < -0.3 or \
                        not (0.04 * h_lim <= L <= 0.30 * h_lim) or \
                        w > 1.5 * max(L, 1e-9):
                    fan = None
        if paths is not None:
            for (fname, _op, _od, _ln), fp in zip(FINGER_PLAN, paths):
                pts = [Vector(fp[0]), Vector(fp[1]), Vector(fp[2])]
                if mesh_obj is not None:
                    pts = [_snap_to_mesh(mesh_obj, arm_obj, p, h_ref)
                           for p in pts]
                # v2.11.0: parent каждого пальца — кисть (eb_p), а не
                # предыдущий палец: раньше parent протекал между
                # пальцами и все цепочки сливались в одну («пальцев: 2»
                # в отчёте вместо 10, IK пальцев вёл соседний палец)
                parent = eb_p
                for seg, (a, b) in enumerate(((pts[0], pts[1]),
                                              (pts[1], pts[2]))):
                    if (b - a).length < 1e-6:
                        continue
                    eb = arm_data.edit_bones.new(
                        "fng_%s_%d_%s" % (fname, seg + 1, side))
                    eb.head, eb.tail = a, b
                    eb.roll = 0.0
                    eb.parent = parent
                    parent = eb
                    made += 1
            continue
        if fan is not None and (fan.get("d") and fan.get("u")):
            # -- веер ПО РАМКЕ КИСТИ (геометрия есть, пальцы слиты)
            wr = Vector(fan["wr"])
            d = Vector(fan["d"])
            if d.length > 1e-9:
                d = d.normalized()
                u = Vector(fan["u"])
                if u.length > 1e-9 and fan.get("L", 0.0) > 1e-6:
                    u = u.normalized()
                    L = float(fan.get("L", 0.0))
                    w = float(fan.get("w", 0.0))
                    cu = float(fan.get("cu", 0.0))
                    knuck = wr + d * (0.42 * L)
                    lens = (0.78, 0.93, 1.0, 0.93, 0.78)
                    for (fname, _op, _od, _ln), off, lf in zip(
                            FINGER_PLAN,
                            (-0.32, -0.16, 0.0, 0.16, 0.32), lens):
                        # развод: вокруг поперечного центра кисти
                        base = knuck + u * (cu + off * w * 0.55)
                        tip = base + d * (L - 0.42 * L) * lf
                        mid = base + (tip - base) * 0.5
                        pts = [base, mid, tip]
                        if mesh_obj is not None:
                            pts = [_snap_to_mesh(mesh_obj, arm_obj, p,
                                                 h_ref) for p in pts]
                        # v2.11.0: parent каждого пальца — кисть
                        parent = eb_p
                        for seg, (a, b) in enumerate(((pts[0], pts[1]),
                                                      (pts[1], pts[2]))):
                            if (b - a).length < 1e-6:
                                continue
                            eb = arm_data.edit_bones.new(
                                "fng_%s_%d_%s" % (fname, seg + 1, side))
                            eb.head, eb.tail = a, b
                            eb.roll = 0.0
                            eb.parent = parent
                            parent = eb
                            made += 1
                    continue
        # -- анатомический веер: стандартная ладонь (геометрии нет)
        base = eb_p.tail + arm_dir * (0.02 * scale)
        for fname, off_p, off_d, ln in FINGER_PLAN:
            head = base + spread * (off_p * scale)
            if droop is not None and off_d > 1e-9:
                head = head + droop * (off_d * scale)
            parent = eb_p
            for seg in (1, 2):
                eb = arm_data.edit_bones.new(
                    "fng_%s_%d_%s" % (fname, seg, side))
                eb.head = head
                eb.tail = head + arm_dir * (ln * scale *
                                            (0.55 if seg == 1
                                             else 0.45))
                eb.roll = 0.0
                eb.parent = parent
                head = eb.tail
                parent = eb
                made += 1
    bpy.ops.object.mode_set(mode="OBJECT")
    return made


def _order_finger_paths(cand, eb_p, arm_dir):
    """Порядок большой->мизинец: ось развода пальцев из самих путей
    (PCA векторов середина-запястье без компоненты вдоль руки),
    ориентация «к лицу» (+X); у ладони, стоящей ребром (развод
    вертикальный), большой палец сверху — ориентир вверх (+Z)."""
    wr = Vector(eb_p.tail)
    mids = []
    for fp in cand:
        w = 0.5 * (Vector(fp[0]) + Vector(fp[2])) - wr
        mids.append(w - arm_dir * w.dot(arm_dir))
    cov = [[0.0] * 3 for _ in range(3)]
    for w in mids:
        for i in range(3):
            for j in range(3):
                cov[i][j] += w[i] * w[j]
    e = Vector((1.0, 0.0, 0.0))
    for _ in range(24):
        ne = Vector(tuple(
            sum(cov[i][j] * e[j] for j in range(3))
            for i in range(3)))
        if ne.length < 1e-12:
            break
        e = ne.normalized()
    if e.dot(Vector((1.0, 0.0, 0.0))) < 0.0:
        e = -e
    if abs(e.dot(Vector((1.0, 0.0, 0.0)))) < 0.4 and \
            e.dot(Vector((0.0, 0.0, 1.0))) < 0.0:
        e = -e
    return sorted(
        cand, key=lambda fp: -(0.5 * (Vector(fp[0]) + Vector(fp[2]))
                               - wr).dot(e))


def _snap_to_mesh(mesh_obj, arm_obj, pt, h_ref):
    """Подтянуть точку (пространство арматуры) к поверхности меша.

    Работает ТОЛЬКО для точек рядом с мешом (дальше 0.06*h точка не
    трогается — снап не должен уносить кости на чужие части тела).
    Точка кладётся чуть ВНУТРЬ поверхности (0.002*h по нормали):
    кость видна внутри пальца, а не висит на коже."""
    try:
        mw = mesh_obj.matrix_world
        amw = arm_obj.matrix_world
        amw_inv = amw.inverted()
        mw_inv = mw.inverted()
        w = amw @ pt
        loc_p = mw_inv @ w
        res, loc, nrm, _idx = mesh_obj.closest_point_on_mesh(loc_p)
        if not res:
            return pt
        dist = (Vector(loc) - loc_p).length
        if dist > 0.06 * h_ref:
            return pt                       # слишком далеко от меша
        inset = 0.002 * h_ref
        n = Vector(nrm)
        if n.length > 1e-9:
            n = n.normalized()
            inner = mw @ (Vector(loc) - n * inset)
            return amw_inv @ inner
        return amw_inv @ (mw @ Vector(loc))
    except Exception:
        return pt


def skin_mesh_to_armature(mesh, arm_obj):
    """Автовесы: меш -> арматура (vertex groups по именам костей).
    Группы по костям нужны потом X-Muscle (apply_musculature)."""
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    bpy.ops.object.select_all(action="DESELECT")
    mesh.select_set(True)
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")


# ------------------------------------------------- доклей весов пальцев

def _seg_dist_np(pts, a, b):
    """Расстояния от точек pts (N,3) до отрезка [a, b] (world)."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        d = pts - a
        return np.sqrt(np.einsum("ij,ij->i", d, d)), np.zeros(len(pts))
    t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0)
    proj = a + np.outer(t, ab)
    d = pts - proj
    return np.sqrt(np.einsum("ij,ij->i", d, d)), t


def _finger_chains_of(arm_obj):
    """Пальцевые цепочки по дереву костей: [(host_name, [b1, b2]), ...].

    host — кость-родитель корня цепочки (farm_*/hand/крыло)."""
    chains = []
    roots = []
    bones = arm_obj.data.bones
    for b in bones:
        if b.name.startswith("fng_") and \
                (b.parent is None or
                 not b.parent.name.startswith("fng_")):
            roots.append(b)
    for rb in roots:
        chain = [rb]
        while True:
            kids = [k for k in rb.children if k.name.startswith("fng_")]
            # у корня может быть 5 детей-пальцев — идём по одному:
            # сама цепочка линейна (root -> seg2)
            kids = [k for k in chain[-1].children
                    if k.name.startswith("fng_")]
            if len(kids) != 1:
                break
            chain.append(kids[0])
        host = rb.parent.name if rb.parent is not None else None
        chains.append((host, [b.name for b in chain]))
    return chains


def fix_finger_weights(mesh_obj, arm_obj, h_ref=None):
    """Гарантированный скининг пальцев (v2.13.0) — ГЛАВНЫЙ фикс итерации.

    Автовеса Blender (bone heat) на мешах Image-to-3D регулярно
    отдают вершины пальцев кости предплечья/кисти: кости пальцев
    «не попали в меш» (главный баг 2.12 — пальцы не сгибаются,
    вращение дисков ничего не делает). Здесь привязка правится
    ПО ГЕОМЕТРИИ, детерминированно: вершина, которая ближе к
    пальцевой цепочке, чем к кости-хозяину (farm/hand/крыло),
    перепривязывается к этой цепочке; вес делится между сегментами
    пальца по проекции вдоль цепочки (с плавной зоной у сустава).

    Работает для любого плана (human fng_*, creature fng_*_std_*).
    Возвращает строку отчёта («пальцы: перенесено N вершин» или "")."""
    if np is None or mesh_obj is None or arm_obj is None:
        return ""
    try:
        chains = _finger_chains_of(arm_obj)
        chains = [(host, names) for host, names in chains
                  if host and arm_obj.data.bones.get(host) is not None
                  and all(arm_obj.data.bones.get(n) is not None
                          for n in names)]
        if not chains:
            return ""
        me = mesh_obj.data
        mw = mesh_obj.matrix_world
        amw = arm_obj.matrix_world
        n_v = len(me.vertices)
        if n_v == 0:
            return ""
        # мировые координаты вершин
        co = np.empty((n_v, 3), dtype=np.float64)
        me.vertices.foreach_get("co", np.ravel(co))
        co = co.reshape(n_v, 3)
        mw3 = [list(mw.row[i])[:3] for i in range(3)]
        co = co @ np.array(mw3, dtype=np.float64).T + \
            np.array(mw.translation, dtype=np.float64)
        if h_ref is None:
            h_ref = float(arm_obj.get("ncp_height", 0.0) or 0.0) or 1.70
        # сегменты цепочек и хозяина в мировом пространстве
        def seg_pts(bone_name):
            b = arm_obj.data.bones[bone_name]
            m = amw @ b.matrix_local
            a = m.to_translation()
            t = m @ Vector((0.0, float(b.length), 0.0))
            return (np.array([a.x, a.y, a.z], dtype=np.float64),
                    np.array([t.x, t.y, t.z], dtype=np.float64))

        host_segs = {}
        chain_segs = []
        for host, names in chains:
            if host not in host_segs:
                host_segs[host] = seg_pts(host)
            segs = [seg_pts(n) for n in names]
            L = [float(np.linalg.norm(s[1] - s[0])) for s in segs]
            chain_segs.append((host, names, segs, L))

        d_host_best = np.full(n_v, np.inf)
        for host, seg in host_segs.items():
            d, _t = _seg_dist_np(co, seg[0], seg[1])
            d_host_best = np.minimum(d_host_best, d)

        # ближайшая цепочка пальца для каждой вершины
        best_d = np.full(n_v, np.inf)
        best_c = np.full(n_v, -1, dtype=np.int64)
        best_t = np.zeros(n_v)
        for ci, (host, names, segs, L) in enumerate(chain_segs):
            # расстояние до полилинии цепочки + доля вдоль неё
            d_min = np.full(n_v, np.inf)
            s_acc = np.zeros(n_v)
            s_run = 0.0
            total = sum(L)
            t_glob = np.zeros(n_v)
            for si, (a, b) in enumerate(segs):
                d, t = _seg_dist_np(co, a, b)
                cur = d < d_min
                t_glob[cur] = (s_run + t[cur] * L[si]) / max(total, 1e-9)
                d_min = np.where(cur, d, d_min)
                s_run += L[si]
            upd = d_min < best_d
            best_d[upd] = d_min[upd]
            best_c[upd] = ci
            best_t[upd] = t_glob[upd]

        # зона кисти/пальцев: не дальше 0.12*h от хозяина или пальца
        r_zone = 0.12 * max(h_ref, 1e-6)
        take = (best_c >= 0) & (best_d < d_host_best) & \
            (np.minimum(best_d, d_host_best) < r_zone)
        idxs = np.nonzero(take)[0]
        if len(idxs) == 0:
            return ""
        # группы вершин (создать при отсутствии)
        vgs = mesh_obj.vertex_groups
        need = set()
        for host, names, _segs, _L in chain_segs:
            need.add(host)
            need.update(names)
        for nm in need:
            if vgs.get(nm) is None:
                vgs.new(nm)
        gid = {nm: vgs[nm].index for nm in need}
        n_done = 0
        per_chain = {}
        for vi in idxs:
            ci = int(best_c[vi])
            host, names, segs, L = chain_segs[ci]
            vi = int(vi)
            v = me.vertices[vi]
            u = float(best_t[vi])
            # стык сегментов: перетекание веса у сустава пальца
            u_j = L[0] / max(sum(L), 1e-9)
            w2 = min(max((u - (u_j * 0.6)) / max(u_j * 0.8, 1e-9),
                         0.0), 1.0)
            w1 = 1.0 - w2
            if len(names) == 1:
                w1, w2 = 1.0, 0.0
            try:
                # снять ВСЕ группы у вершины (VertexGroup.remove)
                for g in list(v.groups):
                    try:
                        vgs[g.group].remove([vi])
                    except Exception:
                        pass
                if w1 > 0.001:
                    vgs[gid[names[0]]].add([vi], w1, "REPLACE")
                if len(names) > 1 and w2 > 0.001:
                    vgs[gid[names[1]]].add([vi], w2, "REPLACE")
            except Exception:
                continue
            n_done += 1
            per_chain[ci] = per_chain.get(ci, 0) + 1
        top = sorted(per_chain.items(), key=lambda kv: -kv[1])[:4]
        det = ", ".join("%s:%d" % (chain_segs[ci][1][0][4:], n)
                        for ci, n in top)
        return "пальцы на меш: %d вершин (%s)" % (n_done, det)
    except Exception as e:
        return ""


def check_bone_placement(arm_obj, mesh_obj, h_ref=None):
    """Проверка «кости на меше» (v2.13.0): кости пальцев должны стоять
    на реальных пальцах меша, кости рук/ног — на уровне поверхности.

    Считается расстояние середины каждой остовной кости до ближайшей
    точки поверхности меша (closest_point_on_mesh, мировые координаты).
    Возвращает короткую строку («пальцы ≤0.6 см, конечности ≤1.2 см»)
    или "" — вызывается после сборки, результат идёт в статус."""
    if mesh_obj is None or arm_obj is None:
        return ""
    try:
        if h_ref is None:
            h_ref = float(arm_obj.get("ncp_height", 0.0) or 0.0) or 1.70
        mw = mesh_obj.matrix_world
        mw_inv = mw.inverted()
        amw = arm_obj.matrix_world
        worst_f = 0.0
        worst_l = 0.0
        for b in arm_obj.data.bones:
            nm = b.name
            is_fng = nm.startswith("fng_")
            if not (is_fng or nm.startswith(("uarm_", "farm_", "thigh_",
                                             "shin_", "wing"))):
                continue
            if not getattr(b, "use_deform", True):
                continue
            m = amw @ b.matrix_local
            mid = m @ Vector((0.0, float(b.length) * 0.5, 0.0))
            loc_p = mw_inv @ mid
            res, loc, _nrm, _idx = mesh_obj.closest_point_on_mesh(loc_p)
            if not res:
                continue
            dist = float((Vector(loc) - loc_p).length)
            if is_fng:
                worst_f = max(worst_f, dist)
            else:
                worst_l = max(worst_l, dist)
        out = []
        if worst_f > 0.0:
            out.append("пальцы %s%.1f см" %
                       ("" if worst_f <= 0.03 * h_ref else "ВНЕ меша ",
                        worst_f * 100.0))
        if worst_l > 0.0:
            out.append("конечности %.1f см" % (worst_l * 100.0))
        return ", ".join(out)
    except Exception:
        return ""


# ---------------------------------------------------------------- мир

def build_world_props(world_def, name_prefix="world"):
    """Пропсы мира neurosim (вендоринг импортёра)."""
    out = []
    for i, prop in enumerate(world_def.get("props", [])):
        pos = Vector(prop["pos"])
        rpy = prop.get("rpy", [0, 0, 0])
        sh = prop["shape"]
        if sh == "box":
            hx, hy, hz = prop["half"]
            bpy.ops.mesh.primitive_cube_add(location=pos,
                                            rotation=(rpy[0], rpy[1], rpy[2]),
                                            scale=(hx, hy, hz))
        elif sh == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=prop["radius"],
                                                 segments=24, ring_count=12,
                                                 location=pos)
        else:
            bpy.ops.mesh.primitive_cylinder_add(radius=prop["radius"],
                                                depth=prop["length"],
                                                location=pos,
                                                rotation=(rpy[0], rpy[1],
                                                          rpy[2]))
        obj = bpy.context.active_object
        obj.name = "%s_%s_%d" % (name_prefix, sh, i)
        mat = bpy.data.materials.new(obj.name + "_m")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            c = prop.get("color", [0.6, 0.6, 0.6, 1])
            bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1)
        obj.data.materials.append(mat)
        out.append(obj)
    return out


def add_ground(size=60.0):
    """Земля (вендоринг импортёра)."""
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "neuro_ground"
    mat = bpy.data.materials.new("neuro_ground_m")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.28, 0.42, 0.22, 1)
    obj.data.materials.append(mat)
    return obj
