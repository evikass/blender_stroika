# -*- coding: utf-8 -*-
"""
neurosim.export — Экспорт движения: BVH (мокап) и JSON (для Blender-аддона).

BVH: Blender -> File -> Import -> Motion Capture (.bvh), Scale = 1.0.
JSON: File -> Import -> Neuro Sim Motion (.json) через аддон
      blender_neuro_importer.py — он строит арматуру, меши и мир.
"""
import json
import math
import os
import time

from . import spec as S

# ---------------------------------------------------------------- матрицы


def mat_axis_angle(axis, angle):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def mat_from_quat(q):
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_to_euler_xyz(R):
    """Разложение R = Rz(c) @ Ry(b) @ Rx(a); возвращает (a, b, c) в градусах.
    Именно так Blender собирает Euler('XYZ') из каналов X,Y,Z BVH."""
    sy = -R[2][0]
    sy = max(-1.0, min(1.0, sy))
    b = math.asin(sy)
    if abs(sy) < 0.99999:
        a = math.atan2(R[2][1], R[2][2])
        c = math.atan2(R[1][0], R[0][0])
    else:
        a = math.atan2(-R[1][2], R[1][1])
        c = 0.0
    return [math.degrees(a), math.degrees(b), math.degrees(c)]


# ---------------------------------------------------------------- иерархия

def _dfs_order(spec):
    """Звенья с суставами в DFS-порядке (как в BVH HIERARCHY)."""
    kids = {l["name"]: [] for l in spec["links"]}
    root = None
    for l in spec["links"]:
        if l["parent"] is None:
            root = l
        else:
            kids[l["parent"]].append(l)
    order, stack = [], [root]
    while stack:
        node = stack.pop()
        children = [c for c in kids[node["name"]] if c.get("joint")]
        stack.extend(reversed(children))
        if node.get("joint"):
            order.append(node)
        # end-точки добавляются на месте
    return order, kids


def _anchor(l):
    return l["joint"]["anchor"]


# ---------------------------------------------------------------- BVH

def export_bvh(frames, spec, path, fps=48.0):
    order, kids = _dfs_order(spec)
    root = spec["links"][0]
    lines = ["HIERARCHY"]
    lines.append(f"ROOT {root['name']}")
    lines.append("{")
    lines.append("\tOFFSET 0.0 0.0 0.0")
    lines.append("\tCHANNELS 6 Xposition Yposition Zposition "
                 "Xrotation Yrotation Zrotation")

    def walk(link, depth, par_anchor):
        ind = "\t" * depth
        for child in [c for c in kids[link["name"]] if c.get("joint")]:
            a = _anchor(child)
            off = [a[k] - par_anchor[k] for k in range(3)]
            lines.append(f"{ind}JOINT {child['name']}")
            lines.append(f"{ind}{{")
            lines.append(f"{ind}\tOFFSET {off[0]:.6f} {off[1]:.6f} {off[2]:.6f}")
            lines.append(f"{ind}\tCHANNELS 3 Xrotation Yrotation Zrotation")
            if not any(c.get("joint") for c in kids[child["name"]]):
                end = child.get("end")
                if end is None:
                    end = child["com"]
                e = [end[k] - a[k] for k in range(3)]
                lines.append(f"{ind}\tEnd Site")
                lines.append(f"{ind}\t{{")
                lines.append(f"{ind}\t\tOFFSET {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}")
                lines.append(f"{ind}\t}}")
            walk(child, depth + 1, a)
            lines.append(f"{ind}}}")

    walk(root, 1, root["com"])
    lines.append("}")
    lines.append("MOTION")
    lines.append(f"Frames: {len(frames)}")
    lines.append(f"Frame Time: {1.0 / fps:.6f}")

    ang_names = [l["name"] for l in order]
    for fr in frames:
        vals = [fr["root_pos"][0], fr["root_pos"][1], fr["root_pos"][2]]
        R = mat_from_quat(fr["root_quat"])
        vals += [f"{v:.4f}" for v in mat_to_euler_xyz(R)]
        jd = dict(zip(ang_names, fr["joints"]))
        for l in order:
            Rj = mat_axis_angle(l["joint"]["axis"], jd[l["name"]])
            vals += [f"{v:.4f}" for v in mat_to_euler_xyz(Rj)]
        lines.append(" ".join(str(v) for v in vals))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- JSON

def export_motion_json(frames, spec, path, world_def=None, task="",
                       fps=48.0, extra=None):
    actuated = S.actuated(spec)
    payload = {
        "format": "neurosim-motion",
        "version": 1,
        "name": spec["name"],
        "task": task,
        "fps": fps,
        "joints": [l["name"] for l in actuated],
        "spec": spec,
        "world": world_def,
        "frames": frames,
        "extra": extra or {},
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return path
