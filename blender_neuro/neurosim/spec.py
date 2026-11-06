# -*- coding: utf-8 -*-
"""
neurosim.spec — Спеки существ (creature specs) / Creature specifications.

Единый JSON-формат описания существа используется:
  - строитель физики в PyBullet (build.py)
  - экспортёром BVH (bvh.py)
  - Blender-аддоном (blender_neuro_importer.py)

Все координаты — в мировых осях покоя (rest pose): Z вверх, X вперёд, Y влево.
Все "рамки" звеньев в покое выровнены по миру, поэтому:
  anchor  = точка сустава в мире (покой)
  axis    = ось сустава в мире (покой)
  com     = центр масс звена в мире (покой)
Это делает математику одинаковой у всех потребителей спеки.
"""
import json
import math
import os

# ---------------------------------------------------------------- кватернионы

def quat_from_axis_angle(axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    x, y, z = x / n, y / n, z / n
    s = math.sin(angle / 2.0)
    return [x * s, y * s, z * s, math.cos(angle / 2.0)]


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def quat_conj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def quat_rotate(q, v):
    """Повернуть вектор v кватернионом q."""
    qv = [v[0], v[1], v[2], 0.0]
    t = quat_mul(quat_mul(q, qv), quat_conj(q))
    return t[:3]


def axis_shape_quat(axis):
    """Капсула/цилиндр в PyBullet и Blender направлены вдоль Z.
    Для оси 'y'/'x' возвращаем кватернион доворота формы."""
    if axis == "z":
        return [0.0, 0.0, 0.0, 1.0]
    if axis == "y":
        return quat_from_axis_angle([1, 0, 0], -math.pi / 2)
    if axis == "x":
        return quat_from_axis_angle([0, 1, 0], math.pi / 2)
    raise ValueError(axis)


def cap(axis, radius, half_len, color):
    return {"type": "capsule", "axis": axis, "radius": radius,
            "half_len": half_len, "color": color}


def box(half, color):
    return {"type": "box", "half": half, "color": color}


def sph(radius, color):
    return {"type": "sphere", "radius": radius, "color": color}


# ---------------------------------------------------------------- гуманоид

def _human_limb(L, s, y):
    """s: 'L'|'R', y: +0.11 / -0.11 — смещение ноги вбок."""
    yy = 0.11 if s == "L" else -0.11
    ya = 0.20 if s == "L" else -0.20
    sk = [0.95, 0.75, 0.55, 1]
    skin = [0.9, 0.8, 0.7, 1]
    pants = [0.2, 0.25, 0.45, 1]

    # --- рука: плечо pitch -> плечо roll -> предплечье (локоть)
    L.append(dict(name=f"shoulderP_{s}", parent="chest",
                  joint=dict(anchor=[0, ya, 1.40], axis=[0, 0, 1],
                             lo=-1.7, hi=1.7, max_force=60.0, range=1.0),
                  com=[0, 0.26 if s == "L" else -0.26, 1.40], mass=0.4,
                  shape=cap("y", 0.035, 0.035, sk)))
    L.append(dict(name=f"uarm_{s}", parent=f"shoulderP_{s}",
                  joint=dict(anchor=[0, 0.30 if s == "L" else -0.30, 1.40],
                             axis=[1, 0, 0], lo=-1.3, hi=1.3,
                             max_force=60.0, range=1.0),
                  com=[0, 0.45 if s == "L" else -0.45, 1.40], mass=2.2,
                  shape=cap("y", 0.048, 0.13, skin)))
    L.append(dict(name=f"farm_{s}", parent=f"uarm_{s}",
                  joint=dict(anchor=[0, 0.60 if s == "L" else -0.60, 1.40],
                             axis=[0, 0, 1], lo=-2.3, hi=0.05,
                             max_force=40.0, range=1.0),
                  com=[0, 0.76 if s == "L" else -0.76, 1.40], mass=1.4,
                  shape=cap("y", 0.042, 0.14, skin),
                  end=[0, 0.94 if s == "L" else -0.94, 1.40]))

    # --- нога: бедро roll -> бедро pitch -> голень -> стопа
    L.append(dict(name=f"hipR_{s}", parent="pelvis",
                  joint=dict(anchor=[0, yy, 0.94], axis=[1, 0, 0],
                             lo=-0.4, hi=0.4, max_force=120.0, range=0.5),
                  com=[0, yy, 0.94], mass=0.5,
                  shape=cap("z", 0.04, 0.04, sk)))
    L.append(dict(name=f"thigh_{s}", parent=f"hipR_{s}",
                  joint=dict(anchor=[0, yy, 0.92], axis=[0, 1, 0],
                             lo=-2.0, hi=0.5, max_force=180.0, range=1.0),
                  com=[0, yy, 0.72], mass=7.5,
                  shape=cap("z", 0.075, 0.18, pants)))
    L.append(dict(name=f"shin_{s}", parent=f"thigh_{s}",
                  joint=dict(anchor=[0, yy, 0.54], axis=[0, 1, 0],
                             lo=-0.05, hi=2.4, max_force=180.0, range=1.0),
                  com=[0, yy, 0.34], mass=4.2,
                  shape=cap("z", 0.058, 0.18, pants)))
    L.append(dict(name=f"foot_{s}", parent=f"shin_{s}",
                  joint=dict(anchor=[0, yy, 0.16], axis=[0, 1, 0],
                             lo=-0.7, hi=0.6, max_force=90.0, range=0.7),
                  com=[0.06, yy, 0.04], mass=1.3,
                  shape=box([0.12, 0.05, 0.04], [0.15, 0.15, 0.18, 1]),
                  end=[0.19, yy, 0.02]))


def make_humanoid_spec(name="human_v1"):
    """Детализированный гуманоид ~1.7 м, 17 управляемых суставов.
    Rest pose: T-поза, ступни на земле (z=0)."""
    L = []

    # --- корень: таз
    L.append(dict(name="pelvis", parent=None, joint=None,
                  com=[0.0, 0.0, 1.00], mass=12.0,
                  shape=cap("x", 0.09, 0.06, [0.25, 0.35, 0.85, 1])))

    # --- туловище и голова
    L.append(dict(name="spine", parent="pelvis",
                  joint=dict(anchor=[0, 0, 1.06], axis=[0, 1, 0],
                             lo=-0.5, hi=0.5, max_force=150.0, range=0.5),
                  com=[0, 0, 1.16], mass=9.0,
                  shape=cap("z", 0.085, 0.09, [0.95, 0.4, 0.3, 1])))
    L.append(dict(name="chest", parent="spine",
                  joint=dict(anchor=[0, 0, 1.24], axis=[0, 1, 0],
                             lo=-0.4, hi=0.4, max_force=150.0, range=0.4),
                  com=[0, 0, 1.33], mass=11.0,
                  shape=cap("z", 0.105, 0.12, [0.95, 0.4, 0.3, 1])))
    L.append(dict(name="head", parent="chest",
                  joint=dict(anchor=[0, 0, 1.46], axis=[0, 1, 0],
                             lo=-0.5, hi=0.5, max_force=40.0, range=0.5),
                  com=[0, 0, 1.56], mass=4.5,
                  shape=cap("z", 0.09, 0.05, [0.9, 0.8, 0.7, 1]),
                  end=[0, 0, 1.70]))

    _human_limb(L, "L", +1)
    _human_limb(L, "R", -1)

    return {"name": name, "root": "pelvis", "links": L,
            "notes": "detailed humanoid, T-pose rest, feet on ground"}


# ---------------------------------------------------------------- четвероногие

def make_quadruped_spec(name="fox", scale=1.0, color=(0.90, 0.45, 0.10, 1),
                        body_r=0.15, body_hl=0.26, leg_r=0.045,
                        body_z=0.42, mass=14.0):
    """Шаблон четвероногого: лиса/волк/медведь/заяц/барсук/кот/ёжик.
    scale — общий масштаб размеров, масса растёт ~ scale^3."""
    s = scale
    r, hl, bz = body_r * s, body_hl * s, body_z * s
    leg_r_ = leg_r * s
    m = mass * (s ** 3)
    L = []

    L.append(dict(name="body", parent=None, joint=None,
                  com=[0.0, 0.0, bz], mass=m,
                  shape=cap("x", r, hl, list(color))))

    L.append(dict(name="head", parent="body",
                  joint=dict(anchor=[0.30 * s, 0, bz + 0.08 * s],
                             axis=[0, 1, 0], lo=-0.7, hi=0.7,
                             max_force=40 * s ** 2, range=0.6),
                  com=[0.44 * s, 0, bz + 0.12 * s], mass=0.16 * m,
                  shape=cap("x", 0.10 * s, 0.08 * s,
                            [min(1, color[0] * 1.05), color[1], color[2], 1]),
                  end=[0.58 * s, 0, bz + 0.10 * s]))

    leg_mass_u = 0.085 * m
    leg_mass_l = 0.042 * m
    for fx in (+1, -1):
        for sy in (+1, -1):
            side = f"{'F' if fx > 0 else 'H'}{'L' if sy > 0 else 'R'}"
            bx, by = 0.20 * fx * s, 0.11 * sy * s
            L.append(dict(name=f"hipR_{side}", parent="body",
                          joint=dict(anchor=[bx, by * 0.82, bz + 0.02 * s],
                                     axis=[1, 0, 0], lo=-0.35, hi=0.35,
                                     max_force=60 * s ** 2, range=0.35),
                          com=[bx, by, bz + 0.02 * s], mass=0.01 * m,
                          shape=cap("z", 0.035 * s, 0.035 * s,
                                    [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, 1])))
            L.append(dict(name=f"hipP_{side}", parent=f"hipR_{side}",
                          joint=dict(anchor=[bx, by, bz],
                                     axis=[0, 1, 0], lo=-1.4, hi=1.0,
                                     max_force=90 * s ** 2, range=1.0),
                          com=[bx, by, bz - 0.12 * s], mass=leg_mass_u,
                          shape=cap("z", leg_r_, 0.08 * s,
                                    [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, 1])))
            L.append(dict(name=f"knee_{side}", parent=f"hipP_{side}",
                          joint=dict(anchor=[bx, by, bz - 0.20 * s],
                                     axis=[0, 1, 0], lo=-0.05, hi=2.0,
                                     max_force=70 * s ** 2, range=1.0),
                          com=[bx, by, bz - 0.30 * s], mass=leg_mass_l,
                          shape=cap("z", leg_r_ * 0.8, 0.08 * s,
                                    [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, 1]),
                          end=[bx, by, bz - 0.395 * s]))

    return {"name": name, "root": "body", "links": L,
            "notes": f"quadruped template {name}, scale={scale}"}


ANIMALS = {
    # герои лесного мультик-проекта: имя: (масштаб, цвет)
    "hedgehog": (0.55, (0.45, 0.32, 0.22, 1)),
    "cat":      (0.75, (0.85, 0.70, 0.35, 1)),
    "hare":     (0.60, (0.92, 0.90, 0.85, 1)),
    "fox":      (1.00, (0.90, 0.45, 0.10, 1)),
    "badger":   (0.85, (0.55, 0.55, 0.58, 1)),
    "wolf":     (1.15, (0.55, 0.57, 0.60, 1)),
    "bear":     (1.55, (0.45, 0.30, 0.18, 1)),
}


# ---------------------------------------------------------------- ввод/вывод

def validate(spec):
    names = {l["name"] for l in spec["links"]}
    for l in spec["links"]:
        if l["parent"] is not None and l["parent"] not in names:
            raise ValueError(f"link {l['name']}: unknown parent {l['parent']}")
    return True


def actuated(spec):
    """Список звеньев с управляемым суставом (в порядке спеки)."""
    return [l for l in spec["links"] if l.get("joint")]


def save(spec, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=1)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    validate(spec)
    return spec
