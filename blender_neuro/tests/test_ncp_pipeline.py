# -*- coding: utf-8 -*-
"""Смоук-тест neuro_character_pipeline v2.6+ без Blender (стабы bpy/mathutils).

Проверяет универсальный авто-риг (neuro_creature) на синтетических
существах: человек T/A-pose (human-контракт 18 костей), quadruped,
bird, dragon (winged_quad), biped_tail, пальцы (5 растопыренных /
варежка / generic std), инвариантность к yaw, _rotz_spec round-trip,
AST-ловушку import-тени numpy (урок v2.3.1), сборку арматуры и
пальцевых цепей на стабе bpy, регистрацию аддона (bl_info, классы,
пропсы Body plan) и vendor-состав.

Запуск:  python blender_neuro/tests/test_ncp_pipeline.py
(или через обёртку scripts/test_ncp.py)
"""
import ast
import importlib.util
import inspect
import json
import math
import os
import sys
import time
import tomllib
import types

HERE = os.path.dirname(os.path.abspath(__file__))
_CANDS = (
    os.path.join(HERE, "..", "character", "neuro_character_pipeline"),
    "/home/z/my-project/blender_stroika/blender_neuro/character/"
    "neuro_character_pipeline",
)
ADDON = next((c for c in _CANDS
              if os.path.isfile(os.path.join(c, "neuro_creature.py"))),
             _CANDS[-1])

fails = []


def check(name, cond, extra=""):
    print(("OK  " if cond else "FAIL") + " " + name +
          (" | " + str(extra) if extra and not cond else ""))
    if not cond:
        fails.append(name)


# ============================================================ стаб bpy
class FakeVector:
    __slots__ = ("x", "y", "z")

    def __init__(self, *args):
        if len(args) == 1:
            v = args[0]
            if isinstance(v, FakeVector):
                x, y, z = v.x, v.y, v.z
            else:
                x, y, z = v
        else:
            x, y, z = args
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __add__(self, o):
        return FakeVector(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return FakeVector(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k):
        return FakeVector(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    @property
    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def __repr__(self):
        return "V(%.3f,%.3f,%.3f)" % (self.x, self.y, self.z)


class FakeMatrix4:
    """4x4, только то, что нужно тесту: __matmul__ на Vector, индексация
    строк (mw[0][3]) и конструирование из кортежа строк."""

    def __init__(self, rows=None):
        self.rows = [list(r) for r in rows] if rows else \
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    def __getitem__(self, i):
        return self.rows[i]

    def __matmul__(self, v):
        r = self.rows
        return FakeVector(
            r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z + r[0][3],
            r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z + r[1][3],
            r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z + r[2][3])

    def inverted(self):
        return FakeMatrix4()


class FakeQuaternion:
    def __init__(self, *a):
        self.q = a

    def to_matrix(self):
        return FakeMatrix4()

    def to_4x4(self):
        return FakeMatrix4()


class FakeEditBone:
    def __init__(self, name):
        self.name = name
        self.head = None
        self.tail = None
        self.roll = 0.0
        self.parent = None

    @property
    def vector(self):
        return self.tail - self.head

    @property
    def length(self):
        return (self.tail - self.head).length


class FakeEditBones:
    def __init__(self, arm):
        self._arm = arm
        self._items = []

    def new(self, name):
        b = FakeEditBone(name)
        self._items.append(b)
        return b

    def get(self, name):
        for b in self._items:
            if b.name == name:
                return b
        return None

    def __getitem__(self, name):
        b = self.get(name)
        if b is None:
            raise KeyError(name)
        return b

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class FakeBone:
    def __init__(self, name):
        self.name = name


class FakeArmature:
    def __init__(self, name):
        self.name = name
        self.edit_bones = FakeEditBones(self)
        self.bones = []
        self.users_obj = None


class FakePoseBone:
    def __init__(self, name):
        self.name = name
        self.rotation_mode = "XYZ"
        self.location = (0.0, 0.0, 0.0)
        self.matrix = None


class FakeArmObj:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        data.users_obj = self
        self.pose = types.SimpleNamespace(bones=[])
        self.matrix_world = FakeMatrix4()
        self.location = (0.0, 0.0, 0.0)
        self.select_set = lambda flag: None

    def finalize(self):
        self.data.bones = [FakeBone(eb.name) for eb in
                           self.data.edit_bones]
        self.pose.bones = [FakePoseBone(b.name) for b in self.data.bones]


class FakeVerts:
    def __init__(self, pts):
        self.pts = pts

    def __len__(self):
        return len(self.pts)

    def foreach_get(self, attr, flat):
        flat[:] = self.pts.reshape(-1)


class FakeMeshData:
    def __init__(self, pts):
        self.vertices = FakeVerts(pts)


class FakeMeshObj:
    def __init__(self, name, pts):
        self.name = name
        self.data = FakeMeshData(pts)
        self.matrix_world = FakeMatrix4()
        self.location = (0.0, 0.0, 0.0)
        self.rotation_euler = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.scale = (1.0, 1.0, 1.0)
        self.select_set = lambda flag: None

    @property
    def bound_box(self):
        pts = self.data.vertices.pts
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        return [FakeVector((x, y, z))
                for x in (mn[0], mx[0])
                for y in (mn[1], mx[1])
                for z in (mn[2], mx[2])]


class FakeEmpty:
    def __init__(self, name):
        self.name = name
        self.location = (0.0, 0.0, 0.0)
        self.empty_display_type = "PLAIN_AXES"
        self.empty_display_size = 0.1


class FakeArmaturesColl:
    def __init__(self):
        self._list = []

    def new(self, name):
        a = FakeArmature(name)
        self._list.append(a)
        return a

    def __iter__(self):
        return iter(self._list)


class FakeObjectsColl:
    def __init__(self):
        self._dict = {}

    def new(self, name, data):
        if isinstance(data, FakeArmature):
            o = FakeArmObj(name, data)
        else:
            o = FakeEmpty(name)
        self._dict[name] = o
        return o

    def get(self, name):
        return self._dict.get(name)


class FakeActionsColl:
    def __init__(self):
        self._dict = {}

    def new(self, name):
        a = types.SimpleNamespace(name=name)
        self._dict[name] = a
        return a


class FakeData:
    def __init__(self):
        self.armatures = FakeArmaturesColl()
        self.objects = FakeObjectsColl()
        self.actions = FakeActionsColl()


class _ObjColl:
    def link(self, o):
        pass


class FakeCollection:
    def __init__(self):
        self.objects = _ObjColl()


class FakeScene:
    def __init__(self):
        self.collection = FakeCollection()
        self.render = types.SimpleNamespace(fps=24)
        self.frame_start = 1
        self.frame_end = 250

    def frame_set(self, f):
        pass


bpy = types.ModuleType("bpy")
bpy.props = types.ModuleType("bpy.props")
bpy.data = FakeData()
bpy.context = types.SimpleNamespace(
    scene=FakeScene(),
    view_layer=types.SimpleNamespace(
        update=lambda: None,
        objects=types.SimpleNamespace(
            active=None, selected=[])),
    preferences=types.SimpleNamespace(addons={}),
    mode="OBJECT")
bpy.path = types.SimpleNamespace(abspath=lambda p: p)
bpy.app = types.SimpleNamespace(
    timers=types.SimpleNamespace(
        is_registered=lambda f: False,
        register=lambda *a, **k: None,
        unregister=lambda *a, **k: None),
    version=(5, 4, 0),
    driver_namespace={})
bpy.ops = types.SimpleNamespace()


class _PropStub:
    def __init__(self, kind, *a, **k):
        self.kind = kind
        self.args = a
        self.kwargs = k


def _prop_stub(kind):
    def f(*a, **k):
        return _PropStub(kind, *a, **k)
    return f


for _n in ("BoolProperty", "EnumProperty", "FloatProperty", "IntProperty",
           "PointerProperty", "StringProperty", "CollectionProperty"):
    setattr(bpy.props, _n, _prop_stub(_n))

_OP_MODE = {"mode": "OBJECT"}


def _mode_set(mode="OBJECT", **k):
    _OP_MODE["mode"] = mode
    if mode == "OBJECT":
        for arm in bpy.data.armatures:
            if arm.users_obj is not None:
                arm.users_obj.finalize()


def _select_all(action="DESELECT", **k):
    pass


def _parent_set(type="ARMATURE_AUTO", **k):
    pass


def _transform_apply(**k):
    pass


bpy.ops.object = types.SimpleNamespace(
    mode_set=_mode_set, select_all=_select_all,
    parent_set=_parent_set, transform_apply=_transform_apply)
bpy.ops.mesh = types.SimpleNamespace(
    primitive_cube_add=lambda **k: None,
    primitive_uv_sphere_add=lambda **k: None,
    primitive_cylinder_add=lambda **k: None,
    primitive_plane_add=lambda **k: None)


class _TypesNS(types.ModuleType):
    pass


bpy.types = _TypesNS("bpy.types")
for _n in ("Operator", "Panel", "PropertyGroup", "AddonPreferences",
           "Menu", "UIList", "Header", "NodeTree", "Node", "NodeSocket",
           "SpaceView3D", "SpaceNodeEditor", "FileHandler",
           "OperatorFileListElement", "Scene", "Object", "Mesh"):
    setattr(bpy.types, _n, type(_n, (), {}))
bpy.utils = types.SimpleNamespace(
    register_class=lambda c: None,
    unregister_class=lambda c: None,
    previews=types.SimpleNamespace(new=lambda: None))

sys.modules["bpy"] = bpy
sys.modules["bpy.props"] = bpy.props
sys.modules["bpy.types"] = bpy.types

mu = types.ModuleType("mathutils")
mu.Vector = FakeVector
mu.Matrix = FakeMatrix4
mu.Quaternion = FakeQuaternion
sys.modules["mathutils"] = mu

bpye = types.ModuleType("bpy_extras")
bpye.io_utils = types.SimpleNamespace(ImportHelper=object)
sys.modules["bpy_extras"] = bpye
sys.modules["bpy_extras.io_utils"] = bpye.io_utils

# ---- импорт модулей аддона (топ-левел для анализа + пакет для __init__)
sys.path.insert(0, ADDON)
import neuro_rig  # noqa: E402
import neuro_autorig  # noqa: E402
import neuro_creature as nc  # noqa: E402
import ncp_comfy  # noqa: E402

import numpy as np  # noqa: E402

print("ADDON:", ADDON)

# ============================================= A. neuro_rig: спека и API
spec18 = neuro_rig.load_spec()
check("load_spec() -> human_v1, 18 links",
      spec18.get("name") == "human_v1" and len(spec18["links"]) == 18,
      (spec18.get("name"), len(spec18["links"])))
check("spec_height(human) == 1.70",
      abs(neuro_rig.spec_height(spec18) - 1.70) < 1e-6)
names18 = [l["name"] for l in neuro_rig._FALLBACK_SPEC["links"]]
check("bone names match human contract", names18 == [
    "pelvis", "spine", "chest", "head", "shoulderP_L", "uarm_L", "farm_L",
    "hipR_L", "thigh_L", "shin_L", "foot_L", "shoulderP_R", "uarm_R",
    "farm_R", "hipR_R", "thigh_R", "shin_R", "foot_R"], names18)

fn = neuro_rig.finger_bone_names()
check("finger_bone_names: 20 уникальных",
      len(fn) == 20 and len(set(fn)) == 20, len(fn))
check("fingers не пересекаются со спекой",
      not set(fn) & set(names18))
check("FINGER_PLAN: 5 пальцев", len(neuro_rig.FINGER_PLAN) == 5)
check("fit_mesh_to_skeleton принимает rot_z_deg",
      "rot_z_deg" in inspect.signature(
          neuro_rig.fit_mesh_to_skeleton).parameters)
o = neuro_rig.orientation_from_extents
check("orient: стоит, лицо +X", o(0.30, 1.80, 1.75) == (0.0, 0.0))
check("orient: стоит, лицо -Y", o(1.80, 0.30, 1.75) == (0.0, 90.0))
check("orient: лежит на спине", o(1.80, 1.75, 0.30) == (90.0, 90.0))

# ============================== B. арматура + пальцы на стабе bpy
arm = neuro_rig.build_armature(neuro_rig._FALLBACK_SPEC,
                               name="NeuroCharacter")
_mode_set("OBJECT")
check("build_armature: 18 костей", len(arm.data.edit_bones) == 18,
      len(arm.data.edit_bones))
eb_uarm = arm.data.edit_bones.get("uarm_L")
check("uarm_L parented to shoulderP_L",
      eb_uarm is not None and eb_uarm.parent is not None and
      eb_uarm.parent.name == "shoulderP_L")
made = neuro_rig.add_finger_chains(arm, scale=1.0)
check("add_finger_chains: 20 костей", made == 20, made)
fng_root = arm.data.edit_bones.get("fng_middle_1_L")
check("fng_middle_1_L parented to farm_L",
      fng_root is not None and fng_root.parent is not None and
      fng_root.parent.name == "farm_L")
check("пальцевые кости у конца farm (не в центре ладони)",
      fng_root is not None and
      (fng_root.head - eb_uarm.tail).length > 0.05,
      None if fng_root is None else
      (fng_root.head - eb_uarm.tail).length)

# ========================= C. mesh_world_points + AST-ловушка тени np
class _MW(FakeMatrix4):
    def __init__(self):
        super().__init__()


mesh_ok = FakeMeshObj("m", np.random.default_rng(1).uniform(
    -0.2, 0.2, (120, 3)))
pts_m = neuro_autorig.mesh_world_points(mesh_ok)
check("mesh_world_points: форма (120,3)", pts_m.shape == (120, 3))
mesh_tiny = FakeMeshObj("t", np.random.default_rng(1).uniform(
    -0.1, 0.1, (30, 3)))
try:
    neuro_autorig.mesh_world_points(mesh_tiny)
    check("mesh_world_points: <40 вершин -> RigFitError", False)
except neuro_autorig.RigFitError:
    check("mesh_world_points: <40 вершин -> RigFitError", True)


def _np_in_functions(path):
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    mods = ([a.name for a in sub.names]
                            if isinstance(sub, ast.Import)
                            else [sub.module or ""])
                    if any("numpy" in (m or "") for m in mods):
                        bad.append((node.name, sub.lineno))
    return bad


shadow_total = {}
for mod in ("__init__.py", "ncp_comfy.py", "ncp_img3d.py",
            "ncp_muscles.py", "neuro_rig.py", "neuro_autorig.py",
            "neuro_creature.py"):
    bad = _np_in_functions(os.path.join(ADDON, mod))
    shadow_total[mod] = bad
    check("AST: нет import numpy внутри функций — %s" % mod,
          not bad, bad)

# ===================================== D. синтетические существа + analyze


def cap(a, b, r, n, seed):
    return nc._capsule_pts(a, b, r, n, seed=seed)


def _hand(y_side, y0, z0, rods=True, n_rod=400):
    """Кисть T-pose: вертикальная ладонь + 5 пальцев вдоль +Y.
    Кисть ЦЕЛИКОМ ниже верха торса (как у реального T-pose) — иначе
    путь ядра прыгает на ладонь в зазоре торс-голова."""
    out = [cap((0.0, y_side * (y0 - 0.02), z0 - 0.09),
               (0.0, y_side * (y0 + 0.02), z0 + 0.09), 0.03, 500,
               300 + y_side)]
    if rods:
        for i, zoff in enumerate((-0.07, -0.035, 0.0, 0.035, 0.07)):
            out.append(cap((0.0, y_side * (y0 - 0.02), z0 + zoff),
                           (0.0, y_side * (y0 + 0.18), z0 + zoff),
                           0.0075, n_rod, 400 + i))
    return out


def build_human_t(rods=True):
    """Человек T-pose, h~1.7, лицо +X, бока ±Y, низ z=0.
    Ноги заходят внутрь торса (без зазора — как у реального меша)."""
    P = [cap((0.0, 0.10, 1.00), (0.02, 0.10, 0.06), 0.06, 3000, 1),
         cap((0.0, -0.10, 1.00), (0.02, -0.10, 0.06), 0.06, 3000, 2),
         cap((0.02, 0.10, 0.028), (0.16, 0.10, 0.02), 0.025, 500, 3),
         cap((0.02, -0.10, 0.028), (0.16, -0.10, 0.02), 0.025, 500, 4),
         cap((0.0, 0.0, 0.95), (0.0, 0.0, 1.48), 0.14, 7000, 5),
         cap((0.0, 0.0, 1.58), (0.0, 0.0, 1.70), 0.095, 1200, 6),
         cap((0.0, 0.18, 1.36), (0.0, 0.64, 1.36), 0.045, 1500, 7),
         cap((0.0, -0.18, 1.36), (0.0, -0.64, 1.36), 0.045, 1500, 8)]
    P += _hand(+1, 0.64, 1.36, rods=rods)
    P += _hand(-1, 0.64, 1.36, rods=rods)
    return np.vstack(P)


def build_human_a():
    """Человек A-pose: руки под ~25 град вниз, пальцы вдоль руки.
    Кисти выше поясной полосы (crotch..+0.10h) и ниже верха торса."""
    d = np.array([0.0, 0.91, -0.42])
    d /= np.linalg.norm(d)
    P = [cap((0.0, 0.10, 1.00), (0.02, 0.10, 0.06), 0.06, 3000, 11),
         cap((0.0, -0.10, 1.00), (0.02, -0.10, 0.06), 0.06, 3000, 12),
         cap((0.02, 0.10, 0.028), (0.16, 0.10, 0.02), 0.025, 500, 13),
         cap((0.02, -0.10, 0.028), (0.16, -0.10, 0.02), 0.025, 500, 14),
         cap((0.0, 0.0, 0.95), (0.0, 0.0, 1.48), 0.14, 7000, 15),
         cap((0.0, 0.0, 1.58), (0.0, 0.0, 1.70), 0.095, 1200, 16)]
    for s in (+1, -1):
        ds = np.array([0.0, s * 0.91, -0.42])
        ds /= np.linalg.norm(ds)
        sh = np.array([0.0, s * 0.18, 1.44])
        hd = sh + ds * 0.43
        P.append(cap(sh, hd, 0.045, 1500, 17 + s))
        palm = hd + ds * 0.04
        P.append(cap(palm - ds * 0.02, palm + ds * 0.02, 0.03, 300, 19))
        for i, off in enumerate((-0.06, -0.03, 0.0, 0.03, 0.06)):
            base = hd + ds * 0.03
            tip = hd + ds * 0.20
            perp = np.cross(ds, np.array([1.0, 0.0, 0.0]))
            perp /= max(np.linalg.norm(perp), 1e-9)
            P.append(cap(base + perp * off, tip + perp * off,
                         0.0075, 240, 40 + i))
    return np.vstack(P)


def build_quadruped(tail=True):
    """Собака: горизонтальный корпус, 4 ноги, шея, голова, хвост."""
    P = [cap((0.30, 0.0, 0.55), (-0.30, 0.0, 0.55), 0.11, 7000, 51),
         cap((0.34, 0.0, 0.62), (0.48, 0.0, 0.76), 0.055, 600, 52),
         cap((0.52, 0.0, 0.78), (0.55, 0.0, 0.78), 0.075, 700, 53)]
    for x, nm in ((0.26, 60), (-0.26, 70)):
        for s, sv in ((+1, 1), (-1, 2)):
            P.append(cap((x, s * 0.13, 0.58), (x + 0.02, s * 0.13, 0.04),
                         0.05, 3000, nm + sv))
            P.append(cap((x + 0.02, s * 0.13, 0.035),
                         (x + 0.13, s * 0.13, 0.025), 0.03, 120, nm + sv * 3))
    if tail:
        P.append(cap((-0.34, 0.0, 0.46), (-0.55, 0.0, 0.30), 0.04,
                     600, 81))
    return np.vstack(P)


def build_bird():
    """Птица: вертикальный корпус, 2 тонкие ноги, 2 плоских крыла."""
    P = [cap((0.03, 0.0, 0.30), (-0.03, 0.0, 0.60), 0.095, 3000, 91),
         cap((-0.02, 0.0, 0.60), (0.02, 0.0, 0.70), 0.045, 400, 92),
         cap((0.03, 0.0, 0.76), (0.04, 0.0, 0.76), 0.075, 700, 93)]
    for s in (+1, -1):
        P.append(cap((0.045, s * 0.045, 0.28), (0.05, s * 0.045, 0.02),
                     0.018, 1500, 94 + s))
        P.append(cap((0.05, s * 0.045, 0.02), (0.10, s * 0.045, 0.015),
                     0.015, 150, 97 + s))
        # крыло-перепонка: тонкое по Z
        ys = np.linspace(0.12, 0.44, 40) * s
        xs = np.linspace(-0.05, 0.18, 26)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([gx.ravel(),
                               gy.ravel(),
                               np.full(gx.size, 0.46) +
                               np.random.default_rng(99).uniform(
                                   -0.006, 0.006, gx.size)])
        P.append(pts)
    return np.vstack(P)


def build_dragon():
    """Дракон: 4 ноги + 2 плоских крыла + длинный хвост + шея."""
    P = [cap((0.30, 0.0, 0.55), (-0.30, 0.0, 0.55), 0.11, 7000, 101),
         cap((0.34, 0.0, 0.62), (0.48, 0.0, 0.74), 0.055, 600, 102),
         cap((0.52, 0.0, 0.78), (0.55, 0.0, 0.78), 0.075, 700, 103),
         cap((-0.34, 0.0, 0.48), (-0.78, 0.0, 0.34), 0.05, 700, 104)]
    for x, nm in ((0.30, 110), (-0.24, 120)):
        for s, sv in ((+1, 1), (-1, 2)):
            P.append(cap((x, s * 0.13, 0.58), (x + 0.02, s * 0.13, 0.04),
                         0.05, 3000, nm + sv))
            P.append(cap((x + 0.02, s * 0.13, 0.035),
                         (x + 0.15, s * 0.13, 0.025), 0.03, 130, nm + sv * 3))
    for s in (+1, -1):
        ys = np.linspace(0.14, 0.55, 46) * s
        xs = np.linspace(-0.02, 0.28, 30)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        pts = np.column_stack([gx.ravel(), gy.ravel(),
                               np.full(gx.size, 0.56) +
                               np.random.default_rng(199).uniform(
                                   -0.008, 0.008, gx.size)])
        P.append(pts)
    return np.vstack(P)


def build_biped_tail(rods):
    """Человек T-pose с хвостом (универсальный biped_tail).
    Хвост массивный и заметно позади корпуса — чтобы его блоб был
    крупнейшим в зоне поиска (иначе вертикальный обрубок торса
    побеждает и отбрасывается по направлению)."""
    P = [cap((-0.20, 0.0, 0.92), (-0.62, 0.0, 0.66), 0.055, 3000, 131)]
    return np.vstack([build_human_t(rods=rods)] + P)


def rotz(pts, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.asarray(pts) @ R.T


def spec_coms(spec):
    return np.array([l["com"] for l in spec["links"]], dtype=float)


t0 = time.time()

# ---- человек T-pose -> human-контракт 18 костей
pts = build_human_t()
res = nc.analyze(pts, base_spec=spec18)
check("human T: plan == human", res["plan"] == "human", res["plan"])
check("human T: спека 18 звеньев", len(res["spec"]["links"]) == 18,
      len(res["spec"]["links"]))
check("human T: имена костей = контракт",
      [l["name"] for l in res["spec"]["links"]] == names18)
check("human T: рост считан", abs(res["h"] - 1.75) < 0.06, res["h"])
check("human T: лицо определено (в report/notes)",
      any("лицо определено" in n for n in res["notes"]) or
      "лицо определено" in res["report"],
      (res["notes"], res["report"]))
jn = [j[0] for j in res["joints"]]
check("human T: joints имеют hip/knee/ankle",
      all(k in jn for k in ("hip", "knee", "ankle", "attach")))

# ---- арматура по fitted-спеке + пальцы человеческого контракта
arm2 = neuro_rig.build_armature(res["spec"], name="NeuroCharacter2")
_mode_set("OBJECT")
check("human T: 18 костей арматуры", len(arm2.data.edit_bones) == 18)
made2 = neuro_rig.add_finger_chains(arm2, scale=res["h"] / 1.70)
check("human T: 20 пальцевых костей добавлено", made2 == 20, made2)

# ---- человек A-pose -> human-контракт
res_a = nc.analyze(build_human_a(), base_spec=spec18)
check("human A: plan == human", res_a["plan"] == "human", res_a["plan"])
check("human A: 18 звеньев", len(res_a["spec"]["links"]) == 18)

# ---- человек без кистей (варежка-рука) -> human не роняется
res_m = nc.analyze(build_human_t(rods=False), base_spec=spec18)
check("human mitten: plan == human", res_m["plan"] == "human",
      res_m["plan"])

# ---- универсальный biped (принудительный план QUADRUPED на человеке)
res_b = nc.analyze(pts, base_spec=None, plan="QUADRUPED")
check("biped: plan == biped (hint QUADRUPED)", res_b["plan"] == "biped",
      res_b["plan"])
check("biped: заметка о заданном плане",
      any("quadruped" in n.lower() for n in res_b["notes"]),
      res_b["notes"])
bn = [l["name"] for l in res_b["spec"]["links"]]
check("biped: цепочки ног leg_L/leg_R",
      all(n in bn for n in ("leg_L_upper", "leg_L_lower", "leg_L_foot",
                            "leg_R_upper", "leg_R_lower", "leg_R_foot")))
check("biped: цепочки рук arm_L/arm_R",
      all(n in bn for n in ("arm_L_1", "arm_L_4", "arm_R_1", "arm_R_4")))
fng = [n for n in bn if n.startswith("fng_")]
check("biped: растопыренная кисть = 5 пальцев (геометрия)",
      len(fng) == 20 and "fng_arm_L_1_1" in bn and
      not any("_std_" in n for n in fng),
      (len(fng), sorted(n for n in bn if n.startswith("fng_"))[:4]))

# ---- варежка -> generic 5 пальцев std
res_bm = nc.analyze(build_human_t(rods=False), base_spec=None,
                    plan="QUADRUPED")
bnm = [l["name"] for l in res_bm["spec"]["links"]]
fngm = [n for n in bnm if n.startswith("fng_")]
check("biped mitten: generic std-пальцы (5x2x2)",
      len(fngm) == 20 and "fng_arm_L_std_1_1" in bnm,
      (len(fngm), sorted(fngm)[:4]))

# ---- biped_tail
res_bt = nc.analyze(build_biped_tail(rods=False))
check("biped_tail: plan == biped_tail", res_bt["plan"] == "biped_tail",
      res_bt["plan"])
bn2 = [l["name"] for l in res_bt["spec"]["links"]]
check("biped_tail: хвост tail_1..",
      any(n.startswith("tail_") for n in bn2), bn2[:6])
check("biped_tail: руки и ноги на месте",
      all(n in bn2 for n in ("arm_L_1", "leg_R_foot")))

# ---- quadruped
res_q = nc.analyze(build_quadruped())
check("quadruped: plan == quadruped", res_q["plan"] == "quadruped",
      res_q["plan"])
qn = [l["name"] for l in res_q["spec"]["links"]]
check("quadruped: 4 ноги front/hind L/R",
      all(n in qn for n in ("leg_front_L_upper", "leg_front_R_foot",
                            "leg_hind_L_lower", "leg_hind_R_upper")),
      qn)
check("quadruped: нет рук/крыльев",
      not any(n.startswith(("arm_", "wing_")) for n in qn))
check("quadruped: хвост (собака)",
      any(n.startswith("tail_") for n in qn))
legs_q = [n for n in qn if n.endswith(("upper", "lower", "foot"))]
check("quadruped: 12 костей ног", len(legs_q) == 12, len(legs_q))

# ---- bird
res_bird = nc.analyze(build_bird())
check("bird: plan == bird", res_bird["plan"] == "bird",
      res_bird["plan"])
bd = [l["name"] for l in res_bird["spec"]["links"]]
check("bird: 2 ноги + 2 крыла",
      all(n in bd for n in ("leg_L_upper", "leg_R_foot",
                            "wing_L_1", "wing_R_4")), bd[:8])
check("bird: нет рук и нет пальцев на крыльях",
      not any(n.startswith(("arm_", "fng_")) for n in bd))

# ---- dragon
res_d = nc.analyze(build_dragon())
check("dragon: plan == winged_quad", res_d["plan"] == "winged_quad",
      res_d["plan"])
dn = [l["name"] for l in res_d["spec"]["links"]]
check("dragon: 4 ноги + 2 крыла + хвост",
      all(n in dn for n in ("leg_front_L_upper", "leg_hind_R_foot",
                            "wing_L_1", "wing_R_4", "tail_1")),
      dn[:6])
check("dragon: длинная шея (neck_*)",
      any(n.startswith("neck_") for n in dn))
check("dragon: имена костей без дубликатов",
      len(set(dn)) == len(dn))

# ---- мусор и принудительный план
try:
    nc.analyze(np.zeros((30, 3)))
    check("мусор: <60 вершин -> CreatureError", False)
except nc.CreatureError:
    check("мусор: <60 вершин -> CreatureError", True)
try:
    nc.analyze(build_quadruped(), base_spec=None, plan="HUMAN")
    check("HUMAN на четвероногом -> CreatureError", False)
except nc.CreatureError:
    check("HUMAN на четвероногом -> CreatureError", True)

# ================================== E. инвариантность к yaw, face_rotation
res0 = nc.analyze(build_human_t(), base_spec=spec18)
pts37 = rotz(build_human_t(), 37.0)
res37 = nc.analyze(pts37, base_spec=spec18)
check("yaw 37: plan == human", res37["plan"] == "human", res37["plan"])
check("yaw 37: face_rotation ~ -37",
      abs(nc.face_rotation(pts37) - (-37.0)) < 6.0,
      nc.face_rotation(pts37))
c0 = spec_coms(res0["spec"])
c37 = spec_coms(res37["spec"])
err = np.abs(c37 - rotz(c0, 37.0)).max()
check("yaw 37: спека в мировых координатах (coms макс. ошибка)",
      err < 0.02 * res0["h"], err)
check("yaw 37: 18 звеньев и пальцевые имена те же",
      [l["name"] for l in res37["spec"]["links"]] == names18)

pts_q25 = rotz(build_quadruped(), -25.0)
res_q25 = nc.analyze(pts_q25)
check("quadruped yaw -25: plan == quadruped", res_q25["plan"] ==
      "quadruped", res_q25["plan"])
cq0 = spec_coms(res_q["spec"])
cq25 = spec_coms(res_q25["spec"])
errq = np.abs(cq25 - rotz(cq0, -25.0)).max()
check("quadruped yaw -25: спека в мировых координатах",
      errq < 0.10 * res_q["h"] and
      [l["name"] for l in res_q25["spec"]["links"]] ==
      [l["name"] for l in res_q["spec"]["links"]],
      (errq, res_q25["plan"]))

# ---- _rotz_spec round-trip
sp = res_b["spec"]
sp_rt = nc._rotz_spec(nc._rotz_spec(sp, 37.0), -37.0)
err_rt = np.abs(spec_coms(sp_rt) - spec_coms(sp)).max()
check("_rotz_spec round-trip (coms)", err_rt < 1e-3, err_rt)

# ==================================== F. аддон: версия, классы, пропсы
spec_imp = importlib.util.spec_from_file_location(
    "neuro_character_pipeline", os.path.join(ADDON, "__init__.py"),
    submodule_search_locations=[ADDON])
pkg = importlib.util.module_from_spec(spec_imp)
sys.modules["neuro_character_pipeline"] = pkg
spec_imp.loader.exec_module(pkg)

check("bl_info version 2.6.0", pkg.bl_info["version"] == (2, 6, 0),
      pkg.bl_info["version"])
cls_names = [c.__name__ for c in pkg.classes]
check("NCP_OT_build_skeleton зарегистрирован",
      "NCP_OT_build_skeleton" in cls_names)
panels = [n for n in cls_names if n.startswith("NCP_PT_")]
check("10 панелей", len(panels) == 10, panels)
check("AddonPreferences класс есть", "NCP_AddonPrefs" in cls_names)
ann = getattr(pkg.NCP_Props, "__annotations__", {})
for p in ("auto_rig", "body_plan", "add_fingers", "orient_auto",
          "mesh_rot_x", "mesh_rot_z", "debug_landmarks"):
    check("prop NCP_Props.%s" % p, p in ann)
bp = ann.get("body_plan")
bp_ids = [it[0] for it in bp.kwargs["items"]] if isinstance(bp, _PropStub) \
    else []
check("Body plan enum: AUTO/HUMAN/QUADRUPED/WINGED_QUAD/BIRD/DRAGON",
      set(bp_ids) == {"AUTO", "HUMAN", "QUADRUPED", "WINGED_QUAD",
                      "BIRD", "DRAGON"}, bp_ids)
with open(os.path.join(ADDON, "blender_manifest.toml"), "rb") as f:
    man = tomllib.load(f)
check("manifest version 2.6.0", man["version"] == "2.6.0",
      man["version"])
with open(os.path.join(ADDON, "INSTALL_RU.txt"), encoding="utf-8") as f:
    install_txt = f.read()
check("INSTALL_RU: секция 2.6.0", "ЧТО НОВОГО В 2.6.0" in install_txt)
check("INSTALL_RU: секция про пальцы/поворот (2.5.0)",
      "ЧТО НОВОГО В 2.5.0" in install_txt)

# ============================================= G. vendor All-in-One
v_img3d = os.path.join(ADDON, "vendor", "img3d")
check("vendor/img3d вшит", os.path.isdir(v_img3d))
check("vendor/xmuscle вшит",
      os.path.isdir(os.path.join(ADDON, "vendor", "xmuscle")))
check("vendor/xpiemenu вшит",
      os.path.isdir(os.path.join(ADDON, "vendor", "xpiemenu")))
check("vendor/img3d: engine присутствует",
      os.path.isdir(os.path.join(v_img3d, "engine")))

print()
print("время: %.1f с" % (time.time() - t0))
print("RESULT:", "ALL OK" if not fails else ("FAILED: %s" % fails))
sys.exit(1 if fails else 0)
