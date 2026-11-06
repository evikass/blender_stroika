# -*- coding: utf-8 -*-
"""
neurosim.build — Существо в PyBullet из JSON-спеки.

Спека -> URDF (временный кэш-файл) -> loadURDF.
URDF-семантика кадров однозначна:
  - рамка звена = его COM-рамка (inertial origin = 0)
  - origin сустава = anchor - com_родителя (в покое все рамки выровнены с миром)
  - ось сустава = ось из спеки
Порядок звеньев после loadURDF может отличаться от спеки, поэтому везде
используется маппинг "имя звена -> pybullet-индекс".
"""
import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET

import pybullet as p

from . import spec as S


# ------------------------------------------------------------------ инерция

def _inertia_diag(mass, shape):
    """Диагональ инерции по габаритам фигуры (приближённо, ось капсулы = Z фигуры)."""
    t = shape["type"]
    if t == "sphere":
        i = 0.4 * mass * shape["radius"] ** 2
        return [i, i, i]
    if t == "box":
        hx, hy, hz = shape["half"]
        x, y, z = 2 * hx, 2 * hy, 2 * hz
    else:  # capsule вдоль Z
        r, hl = shape["radius"], shape["half_len"]
        x = y = 2 * r
        z = 2 * (hl + r)
    ix = mass / 12 * (y * y + z * z)
    iy = mass / 12 * (x * x + z * z)
    iz = mass / 12 * (x * x + y * y)
    # поворот оси фигуры: capsule вдоль 'y'/'x' -> перестановка компонент
    ax = shape.get("axis", "z")
    if ax == "y":
        ix, iy, iz = ix, iz, iy
    elif ax == "x":
        ix, iy, iz = iy, ix, iz
    return [ix, iy, iz]


def _shape_rpy(axis):
    if axis == "z":
        return "0 0 0"
    if axis == "y":
        return f"{-math.pi / 2:.6f} 0 0"
    if axis == "x":
        return f"0 {math.pi / 2:.6f} 0"
    raise ValueError(axis)


# ------------------------------------------------------------------ URDF

def _geometry(parent, shape):
    g = ET.SubElement(parent, "geometry")
    t = shape["type"]
    if t == "capsule":
        ET.SubElement(g, "capsule", radius=f"{shape['radius']:.5f}",
                      length=f"{2 * shape['half_len']:.5f}")
    elif t == "box":
        hx, hy, hz = shape["half"]
        ET.SubElement(g, "box", size=f"{2*hx:.5f} {2*hy:.5f} {2*hz:.5f}")
    elif t == "sphere":
        ET.SubElement(g, "sphere", radius=f"{shape['radius']:.5f}")
    else:
        raise ValueError(t)


def spec_to_urdf(spec):
    root = ET.Element("robot", name=spec["name"])
    mats = {}
    for i, l in enumerate(spec["links"]):
        link = ET.SubElement(root, "link", name=l["name"])
        # рамка звена = сустав (для корня = COM). COM и фигуры смещены на d = com - anchor
        is_root = l["parent"] is None
        d = [0.0, 0.0, 0.0] if is_root else \
            [l["com"][k] - l["joint"]["anchor"][k] for k in range(3)]
        dxyz = " ".join(f"{v:.6f}" for v in d)
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "origin", xyz=dxyz, rpy="0 0 0")
        ET.SubElement(inert, "mass", value=f"{l['mass']:.5f}")
        ix, iy, iz = _inertia_diag(l["mass"], l["shape"])
        ET.SubElement(inert, "inertia",
                      ixx=f"{ix:.6e}", iyy=f"{iy:.6e}", izz=f"{iz:.6e}",
                      ixy="0", ixz="0", iyz="0")
        for tag in ("collision", "visual"):
            el = ET.SubElement(link, tag)
            rpy = _shape_rpy(l["shape"].get("axis", "z")) \
                if l["shape"]["type"] != "sphere" else "0 0 0"
            ET.SubElement(el, "origin", xyz=dxyz, rpy=rpy)
            _geometry(el, l["shape"])
        col = l["shape"]["color"]
        mat = f"mat{i}"
        v = root.findall(f"link[@name='{l['name']}']/visual")[0]
        m = ET.SubElement(v, "material", name=mat)
        ET.SubElement(m, "color", rgba=" ".join(f"{c:.3f}" for c in col))

    for l in spec["links"]:
        if l["parent"] is None:
            continue
        j = l["joint"]
        par = next(x for x in spec["links"] if x["name"] == l["parent"])
        # рамка родителя: у корня = COM, у остальных = их сустав (anchor)
        par_frame = par["com"] if par["parent"] is None else par["joint"]["anchor"]
        joint = ET.SubElement(root, "joint", name=f"{l['name']}_j",
                              type="revolute")
        origin = [j["anchor"][k] - par_frame[k] for k in range(3)]
        ET.SubElement(joint, "origin",
                      xyz=" ".join(f"{v:.6f}" for v in origin), rpy="0 0 0")
        ET.SubElement(joint, "parent", link=par["name"])
        ET.SubElement(joint, "child", link=l["name"])
        ET.SubElement(joint, "axis", xyz=" ".join(f"{v:.4f}" for v in j["axis"]))
        ET.SubElement(joint, "limit",
                      lower=f"{j['lo']:.4f}", upper=f"{j['hi']:.4f}",
                      effort=f"{j['max_force']:.1f}", velocity="12.0")
        ET.SubElement(joint, "dynamics", damping="0.3", friction="0.0")
    return ET.tostring(root, encoding="unicode")


def _urdf_path(spec):
    h = hashlib.md5(spec_to_urdf(spec).encode()).hexdigest()[:10]
    d = os.path.join(tempfile.gettempdir(), "neurosim_urdf")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{spec['name']}_{h}.urdf")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(spec_to_urdf(spec))
    return path


# ------------------------------------------------------------------ существо

class Creature:
    def __init__(self, spec, client, base_pos=(0.0, 0.0, 0.0), friction=1.0):
        self.spec = spec
        self.client = client
        root = spec["links"][0]
        self.root_rest_z = root["com"][2]

        urdf = _urdf_path(spec)
        self.body = client.loadURDF(
            urdf,
            basePosition=[base_pos[0] + root["com"][0],
                          base_pos[1] + root["com"][1],
                          base_pos[2] + root["com"][2]],
            useFixedBase=False)

        # маппинг имён -> pybullet-индексы
        self.name2idx = {}
        self.joint_name2id = {}
        for j in range(client.getNumJoints(self.body)):
            info = client.getJointInfo(self.body, j)
            link_name = info[12].decode()
            joint_name = info[1].decode()
            self.name2idx[link_name] = j
            self.joint_name2id[joint_name] = j
        self.name2idx[root["name"]] = -1

        self.actuated_links = S.actuated(spec)
        self.joint_ids = [self.joint_name2id[f"{l['name']}_j"]
                          for l in self.actuated_links]
        self.joint_names = [l["name"] for l in self.actuated_links]
        self.ranges = [l["joint"]["range"] for l in self.actuated_links]
        self.max_forces = [l["joint"]["max_force"] for l in self.actuated_links]
        self.n_joints = len(self.joint_ids)

        # трение: ступни/лапы цепче
        for name, idx in self.name2idx.items():
            if idx < 0:
                continue
            f = friction * 1.5 if ("foot" in name or "knee" in name or
                                   "hipP" in name) else friction
            client.changeDynamics(self.body, idx, lateralFriction=f,
                                  spinningFriction=0.001, restitution=0.0)

    # ------------------------------------------------------------- управление

    def apply_action(self, action, gain=0.45):
        """action [-1,1] -> целевые углы вокруг позы покоя."""
        targets = [float(a) * r for a, r in zip(action, self.ranges)]
        self.client.setJointMotorControlArray(
            self.body, self.joint_ids, self.client.POSITION_CONTROL,
            targetPositions=targets, forces=self.max_forces,
            positionGains=[gain] * self.n_joints)

    def hold_rest(self):
        self.apply_action([0.0] * self.n_joints, gain=0.45)

    def motors_off(self):
        self.client.setJointMotorControlArray(
            self.body, self.joint_ids, self.client.VELOCITY_CONTROL,
            forces=[0.0] * self.n_joints)

    # ----------------------------------------------------------------- сенсоры

    def root_state(self):
        pos, quat = self.client.getBasePositionAndOrientation(self.body)
        return list(pos), list(quat)

    def joint_states(self):
        st = self.client.getJointStates(self.body, self.joint_ids)
        return [s[0] for s in st], [s[1] for s in st]

    def link_states_com(self):
        """Мировые COM-позы всех звеньев в порядке спеки (для FK-проверки)."""
        out = [tuple(self.client.getBasePositionAndOrientation(self.body))]
        for l in self.spec["links"][1:]:
            ls = self.client.getLinkState(self.body, self.name2idx[l["name"]],
                                          computeForwardKinematics=1)
            out.append((list(ls[0]), list(ls[1])))
        return out

    def contacts_with(self, other_body, names=("foot_L", "foot_R")):
        cps = self.client.getContactPoints(bodyA=self.body, bodyB=other_body)
        links = {cp[4] for cp in cps if cp[3] >= 0}
        present = [n for n in names if n in self.name2idx]
        if len(present) >= 2:
            return [1.0 if self.name2idx[n] in links else 0.0
                    for n in present[:2]]
        # четвероногие: контакт передней / задней пары лап
        fr = [self.name2idx.get(f"knee_F{s}", -1) for s in ("L", "R")]
        bk = [self.name2idx.get(f"knee_H{s}", -1) for s in ("L", "R")]
        return [1.0 if any(i in links for i in fr) else 0.0,
                1.0 if any(i in links for i in bk) else 0.0]

    def root_up_z(self):
        _, quat = self.root_state()
        return S.quat_rotate(quat, [0, 0, 1])[2]

    def is_fallen(self):
        pos, _ = self.root_state()
        return (pos[2] < 0.5 * self.root_rest_z) or (self.root_up_z() < 0.35)

    def remove(self):
        self.client.removeBody(self.body)
