# -*- coding: utf-8 -*-
"""
neurosim.world — Миры: земля, гравитация, статические препятствия.

Пресеты миров:
  empty   — ровный пол (для баланса)
  walk    — полоса препятствий: ящики и рампа (для ходьбы)
  forest  — лес с деревьями, камнями и домиком (для "дойти до цели",
            под мультик-проект: лес + домики + герои)
Мир можно описать своим JSON (см. presets/world_*.json).
"""
import math

from . import spec as S


def quat_rpy(roll, pitch, yaw):
    """Кватернион из эйлеровых углов (порядок применения Z*Y*X)."""
    qz = S.quat_from_axis_angle([0, 0, 1], yaw)
    qy = S.quat_from_axis_angle([0, 1, 0], pitch)
    qx = S.quat_from_axis_angle([1, 0, 0], roll)
    return S.quat_mul(S.quat_mul(qz, qy), qx)


def make_world_empty():
    return {"name": "empty", "props": []}


def make_world_walk():
    props = []
    # ящики-препятствия
    for i, (x, y, h) in enumerate([(2.0, 0.6, 0.30), (3.2, -0.5, 0.42),
                                   (4.5, 0.2, 0.34), (6.0, -0.3, 0.5),
                                   (7.5, 0.4, 0.38)]):
        props.append({"shape": "box", "half": [h / 2, h / 2, h / 2],
                      "pos": [x, y, h / 2], "rpy": [0, 0, 0.3 * (1 if i % 2 else -1)],
                      "color": [0.65, 0.45, 0.25, 1], "mass": 0})
    # рампа
    props.append({"shape": "box", "half": [0.9, 0.6, 0.04],
                  "pos": [9.0, 0.0, 0.17], "rpy": [0, math.radians(-12), 0],
                  "color": [0.55, 0.55, 0.6, 1], "mass": 0})
    # барьер-пороть
    props.append({"shape": "box", "half": [0.06, 1.2, 0.18],
                  "pos": [11.0, 0.0, 0.18], "rpy": [0, 0, 0],
                  "color": [0.8, 0.3, 0.2, 1], "mass": 0})
    return {"name": "walk", "props": props}


def make_world_forest():
    import random
    rnd = random.Random(7)
    props = []
    # деревья: ствол + крона
    tree_xya = [(2.5, 3.0), (3.8, -2.6), (5.5, 2.2), (6.8, -1.8), (8.5, 2.8),
                (9.0, -3.0), (11.0, 1.5), (-2.5, 2.8), (-4.0, -2.5), (-5.5, 2.0),
                (13.0, -1.0), (12.5, 3.2)]
    for x, y in tree_xya:
        a = rnd.uniform(0, 6.28)
        r = rnd.uniform(0.1, 0.22)
        h = rnd.uniform(2.2, 3.4)
        props.append({"shape": "cylinder", "radius": r, "length": h,
                      "pos": [x, y, h / 2], "rpy": [0, 0, a],
                      "color": [0.42, 0.28, 0.15, 1], "mass": 0})
        props.append({"shape": "sphere", "radius": rnd.uniform(0.7, 1.1),
                      "pos": [x, y, h + 0.5], "rpy": [0, 0, 0],
                      "color": [0.20, 0.45, 0.18, 1], "mass": 0})
    # камни
    for x, y, r in [(4.6, 1.4, 0.28), (7.2, 0.8, 0.35), (10.2, -1.2, 0.3),
                    (-3.2, 0.5, 0.26)]:
        props.append({"shape": "sphere", "radius": r,
                      "pos": [x, y, r * 0.6], "rpy": [0, 0, 0],
                      "color": [0.5, 0.5, 0.52, 1], "mass": 0})
    # домик у опушки: две стены + крыша
    hx, hy = -3.0, -4.2
    props.append({"shape": "box", "half": [1.6, 0.08, 1.1],
                  "pos": [hx, hy - 1.4, 1.1], "rpy": [0, 0, 0],
                  "color": [0.75, 0.6, 0.42, 1], "mass": 0})
    props.append({"shape": "box", "half": [0.08, 1.4, 1.1],
                  "pos": [hx - 1.6, hy, 1.1], "rpy": [0, 0, 0],
                  "color": [0.75, 0.6, 0.42, 1], "mass": 0})
    props.append({"shape": "box", "half": [0.08, 1.4, 1.1],
                  "pos": [hx + 1.6, hy, 1.1], "rpy": [0, 0, 0],
                  "color": [0.75, 0.6, 0.42, 1], "mass": 0})
    props.append({"shape": "box", "half": [1.9, 1.7, 0.08],
                  "pos": [hx, hy, 2.35], "rpy": [0, math.radians(6), 0],
                  "color": [0.5, 0.25, 0.18, 1], "mass": 0})
    return {"name": "forest", "props": props}


WORLDS = {"empty": make_world_empty, "walk": make_world_walk,
          "forest": make_world_forest}


def build_world(client, world_def, ground_color=(0.28, 0.42, 0.22, 1)):
    """Создаёт пол и статические препятствия. Возвращает (ground_id, prop_ids)."""
    client.resetSimulation()
    client.setGravity(0, 0, -9.81)
    client.setTimeStep(1.0 / 240.0)
    client.setPhysicsEngineParameter(numSolverIterations=100)

    gcs = client.createCollisionShape(client.GEOM_PLANE)
    gvs = client.createVisualShape(client.GEOM_PLANE, rgbaColor=ground_color)
    ground = client.createMultiBody(baseMass=0, baseCollisionShapeIndex=gcs,
                                    baseVisualShapeIndex=gvs)
    client.changeDynamics(ground, -1, lateralFriction=1.0, restitution=0.0)

    prop_ids = []
    for prop in world_def.get("props", []):
        q = quat_rpy(*prop.get("rpy", [0, 0, 0]))
        t = prop["shape"]
        if t == "box":
            cs = client.createCollisionShape(client.GEOM_BOX,
                                             halfExtents=prop["half"])
            vs = client.createVisualShape(client.GEOM_BOX,
                                          halfExtents=prop["half"],
                                          rgbaColor=prop["color"])
        elif t == "sphere":
            cs = client.createCollisionShape(client.GEOM_SPHERE,
                                             radius=prop["radius"])
            vs = client.createVisualShape(client.GEOM_SPHERE,
                                          radius=prop["radius"],
                                          rgbaColor=prop["color"])
        elif t == "cylinder":
            cs = client.createCollisionShape(client.GEOM_CYLINDER,
                                             radius=prop["radius"],
                                             height=prop["length"])
            vs = client.createVisualShape(client.GEOM_CYLINDER,
                                          radius=prop["radius"],
                                          length=prop["length"],
                                          rgbaColor=prop["color"])
        else:
            raise ValueError(t)
        bid = client.createMultiBody(baseMass=prop.get("mass", 0),
                                     baseCollisionShapeIndex=cs,
                                     baseVisualShapeIndex=vs,
                                     basePosition=prop["pos"],
                                     baseOrientation=q)
        client.changeDynamics(bid, -1, lateralFriction=0.9)
        prop_ids.append(bid)
    return ground, prop_ids


def add_goal_marker(client, pos, radius=0.3):
    vs = client.createVisualShape(client.GEOM_SPHERE, radius=radius,
                                  rgbaColor=[0.95, 0.85, 0.1, 0.6])
    return client.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                                  basePosition=pos)
