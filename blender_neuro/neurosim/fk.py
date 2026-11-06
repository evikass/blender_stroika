# -*- coding: utf-8 -*-
"""
neurosim.fk — Прямая кинематика по спеке (чистый Python, без PyBullet).

Та же математика используется:
  - в экспортёре BVH
  - в Blender-аддоне (портирована на mathutils)
Сверка с PyBullet: tests/test_headless.py
"""
from .spec import quat_mul, quat_rotate, quat_conj, quat_from_axis_angle


def _compose(t1, t2):
    """(pos, quat) композиция: T = T1 * T2."""
    p1, q1 = t1
    p2, q2 = t2
    return ([p1[k] + quat_rotate(q1, p2)[k] for k in range(3)],
            quat_mul(q1, q2))


def fk_link_transforms(spec, root_pos, root_quat, joint_angles):
    """Мировые COM-позы всех звеньев (в порядке spec['links']).

    joint_angles: dict {имя_звена: угол} — углы управляемых суставов.
    Возвращает список [(pos[3], quat[4]), ...] длиной len(spec['links']).
    """
    idx_of = {l["name"]: i for i, l in enumerate(spec["links"])}
    T = [None] * len(spec["links"])
    T[0] = (list(root_pos), list(root_quat))  # COM-рамка корня

    for i, l in enumerate(spec["links"]):
        if l["parent"] is None:
            continue
        j = l["joint"]
        par = spec["links"][idx_of[l["parent"]]]
        # якорь сустава в локальной рамке родителя (рамки в покое выровнены с миром)
        l_a = [j["anchor"][k] - par["com"][k] for k in range(3)]
        # сдвиг COM ребёнка в его собственной рамке
        l_c = [l["com"][k] - j["anchor"][k] for k in range(3)]
        ang = float(joint_angles.get(l["name"], 0.0))
        t = ((l_a, [0.0, 0.0, 0.0, 1.0]),
             ([0.0, 0.0, 0.0], quat_from_axis_angle(j["axis"], ang)),
             (l_c, [0.0, 0.0, 0.0, 1.0]))
        T[i] = _compose(_compose(T[idx_of[l["parent"]]], t[0]), t[1])
        T[i] = _compose(T[i], t[2])
    return T


def fk_com_positions(spec, root_pos, root_quat, joint_angles):
    """Только позиции COM — для быстрой сверки с PyBullet."""
    return [t[0] for t in fk_link_transforms(spec, root_pos, root_quat,
                                             joint_angles)]
