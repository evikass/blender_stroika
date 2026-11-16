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

    def __neg__(self):
        return FakeVector(-self.x, -self.y, -self.z)

    def __mul__(self, k):
        return FakeVector(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return FakeVector(self.y * o.z - self.z * o.y,
                          self.z * o.x - self.x * o.z,
                          self.x * o.y - self.y * o.x)

    @property
    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalized(self):
        L = self.length
        if L < 1e-9:
            return FakeVector(1.0, 0.0, 0.0)
        return FakeVector(self.x / L, self.y / L, self.z / L)

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


class FakeMatrix3:
    """3x3-рамка кости: .col — оси X/Y/Z (Y вдоль кости)."""

    def __init__(self, x, y, z):
        self.col = (x, y, z)


def _stub_cross(a, b):
    return FakeVector(a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x)


class FakeEditBone:
    def __init__(self, name):
        self.name = name
        self.head = None
        self.tail = None
        self.roll = 0.0
        self.parent = None
        self.use_deform = True
        self.use_connect = False
        self.show_wire = False

    @property
    def vector(self):
        return self.tail - self.head

    @property
    def length(self):
        return (self.tail - self.head).length

    @property
    def matrix(self):
        """Рамка кости как у Blender (roll=0): Y вдоль кости, X =
        Y x ref (для вертикальной кости X = +X мира)."""
        v = self.tail - self.head
        if v.length < 1e-9:
            y = FakeVector(0.0, 0.0, 1.0)
        else:
            y = v.normalized()
        ref = FakeVector(0.0, 1.0, 0.0) if abs(y.y) < 0.9 \
            else FakeVector(0.0, 0.0, 1.0)
        x = _stub_cross(y, ref)
        x = x.normalized() if x.length > 1e-9 else FakeVector(1.0, 0.0, 0.0)
        z = _stub_cross(x, y)
        return FakeMatrix3(x, y, z)


class FakeEditBones:
    def __init__(self, arm):
        self._arm = arm
        self._items = []

    def new(self, name):
        b = FakeEditBone(name)
        self._items.append(b)
        return b

    def remove(self, b):
        self._items.remove(b)

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
        self.hide = False


class FakeBoneList:
    def __init__(self, items=None):
        self._items = items or []

    def get(self, name):
        for b in self._items:
            if b.name == name:
                return b
        return None

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class FakeArmature:
    def __init__(self, name):
        self.name = name
        self.edit_bones = FakeEditBones(self)
        self.bones = FakeBoneList()
        self.users_obj = None


class FakeConstraint:
    def __init__(self, ctype):
        self.type = ctype
        self.name = ctype
        self.influence = 1.0
        self.subtarget = ""
        self.target = None
        self.chain_count = 0


class FakeConstraints:
    def __init__(self):
        self._items = []

    def new(self, ctype):
        c = FakeConstraint(ctype)
        self._items.append(c)
        return c

    def remove(self, c):
        self._items.remove(c)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class FakePoseBone:
    def __init__(self, name):
        self.name = name
        self.rotation_mode = "XYZ"
        self.location = (0.0, 0.0, 0.0)
        self.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
        self.scale = (1.0, 1.0, 1.0)
        self.matrix = None
        self.constraints = FakeConstraints()
        self.select = False
        self.custom_shape = None
        self.keys = []

    def keyframe_insert(self, dp, frame=1):
        self.keys.append((dp, int(frame)))


class FakePoseBones:
    def __init__(self, items=None):
        self._items = items or []
        self.active = None

    def get(self, name):
        for b in self._items:
            if b.name == name:
                return b
        return None

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class FakeArmObj:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.type = "ARMATURE"
        data.users_obj = self
        self.pose = types.SimpleNamespace(bones=FakePoseBones())
        self.matrix_world = FakeMatrix4()
        self.location = (0.0, 0.0, 0.0)
        self.select_set = lambda flag: None
        self.show_in_front = False
        self._props = {}

    # custom props (arm["ncp_height"] = ...; arm.get(...))
    def __getitem__(self, k):
        return self._props[k]

    def __setitem__(self, k, v):
        self._props[k] = v

    def __delitem__(self, k):
        del self._props[k]

    def __contains__(self, k):
        return k in self._props

    def get(self, k, default=None):
        return self._props.get(k, default)

    def finalize(self):
        self.data.bones = FakeBoneList(
            [FakeBone(eb.name) for eb in self.data.edit_bones])
        self.pose.bones = FakePoseBones(
            [FakePoseBone(b.name) for b in self.data.bones])


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


class FakeShapeMesh:
    def __init__(self, name):
        self.name = name
        self.faces = []

    def from_pydata(self, verts, edges, faces):
        self.faces = [tuple(f) for f in faces]


class FakeMeshesColl:
    def __init__(self):
        self._dict = {}

    def get(self, name):
        return self._dict.get(name)

    def new(self, name):
        m = FakeShapeMesh(name)
        self._dict[name] = m
        return m


class FakeData:
    def __init__(self):
        self.armatures = FakeArmaturesColl()
        self.objects = FakeObjectsColl()
        self.actions = FakeActionsColl()
        self.meshes = FakeMeshesColl()


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
        self.frame_current = 1
        self.tool_settings = types.SimpleNamespace(
            use_keyframe_insert_auto=False)

    def frame_set(self, f):
        self.frame_current = int(f)


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
    handlers=types.SimpleNamespace(depsgraph_update_post=[]),
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
import neuro_controls as nctl  # noqa: E402
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
eb_farm = arm.data.edit_bones.get("farm_L")
fng_root = arm.data.edit_bones.get("fng_middle_1_L")
check("fng_middle_1_L parented to farm_L",
      fng_root is not None and fng_root.parent is not None and
      fng_root.parent.name == "farm_L")
# v2.9.0: кости пальцев растут ОТ ЗАПЯСТЬЯ (хвост farm), в плоскости
# ладони (веер вдоль +X, лёгкий прогиб вниз), пальцы вдоль руки
check("v2.9.0: пальцы у запястья (base = farm.tail + 2% предплечья)",
      fng_root is not None and eb_farm is not None and
      (fng_root.head - eb_farm.tail).length < 0.05,
      None if fng_root is None else
      (fng_root.head - eb_farm.tail).length)
thumbL = arm.data.edit_bones.get("fng_thumb_1_L")
pinkyL = arm.data.edit_bones.get("fng_pinky_1_L")
check("v2.9.0: веер в плоскости ладони: thumb вперёд (+X) pinky",
      thumbL is not None and pinkyL is not None and
      thumbL.head.x - pinkyL.head.x > 0.04,
      None if thumbL is None or pinkyL is None else
      thumbL.head.x - pinkyL.head.x)
check("v2.9.0: веер НЕ по мировому Z (ладонь не повернута на 90°)",
      thumbL is not None and pinkyL is not None and
      abs(thumbL.head.z - pinkyL.head.z) < 0.03,
      None if thumbL is None or pinkyL is None else
      abs(thumbL.head.z - pinkyL.head.z))
mid_dir = fng_root.tail - fng_root.head
check("v2.9.0: пальцы направлены вдоль руки (+Y), не вверх",
      abs(mid_dir.y) > 2.0 * abs(mid_dir.z), mid_dir)
thumbR = arm.data.edit_bones.get("fng_thumb_1_R")
pinkyR = arm.data.edit_bones.get("fng_pinky_1_R")
check("v2.9.0: правая кисть — thumb тоже вперёд (+X)",
      thumbR is not None and pinkyR is not None and
      thumbR.head.x - pinkyR.head.x > 0.04,
      None if thumbR is None or pinkyR is None else
      thumbR.head.x - pinkyR.head.x)
check("v2.9.0: прогиб вниз — головки пальцев не выше запястья",
      thumbL is not None and fng_root is not None and
      thumbL.head.z <= eb_farm.tail.z + 1e-6 and
      fng_root.head.z <= eb_farm.tail.z + 1e-6,
      (thumbL.head.z if thumbL else None, eb_farm.tail.z))

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
            "neuro_creature.py", "neuro_controls.py"):
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

# =========================== H. слой рычагов (neuro_controls)
# arm из секции B: human-контракт 18 костей + 20 пальцев

rep = nctl.build_controls(arm, ik_on=True, hide_deform=False)
check("controls: отчёт не пуст", bool(rep), rep)
check("controls: отчёт упоминает 7 рычагов", "рычаги: 7" in rep, rep)
check("controls: отчёт — 4 полюса + 10 пальцев + 8 хелперов",
      "полюсов: 4" in rep and "пальцев: 10" in rep and
      "IK-хелперов: 8" in rep, rep)
hn = nctl.handle_names(arm)
for nm in ("CTRL_Root", "CTRL_Body", "CTRL_Head", "CTRL_Arm_L",
           "CTRL_Arm_R", "CTRL_Foot_L", "CTRL_Foot_R"):
    check("controls: %s создан" % nm, nm in hn, hn)
# 7 главных + 4 полюса + 10 пальцев
check("controls: у human 21 CTRL-кость (7+4 полюса+10 пальцев)",
      len(hn) == 21, hn)
for nm in ("CTRL_Pole_Knee_L", "CTRL_Pole_Knee_R",
           "CTRL_Pole_Elbow_L", "CTRL_Pole_Elbow_R"):
    check("controls: полюс %s" % nm, nm in hn, hn)
fing = sorted(n for n in hn if n.startswith("CTRL_Finger_"))
check("controls: 10 рычагов пальцев, имена человеческие",
      len(fing) == 10 and "CTRL_Finger_thumb_L" in fing and
      "CTRL_Finger_index_R" in fing, fing[:4])

# полюс колена: у головы голени, смотрит вперёд (+X)
eb_knee = arm.data.edit_bones.get("CTRL_Pole_Knee_L")
eb_shin = arm.data.edit_bones.get("shin_L")
check("controls: полюс колена у головы голени, смотрит вперёд (+X)",
      eb_knee is not None and eb_shin is not None and
      abs(eb_knee.head.x - eb_shin.head.x) < 1e-6 and
      abs(eb_knee.head.z - eb_shin.head.z) < 1e-6 and
      eb_knee.tail.x > eb_knee.head.x + 0.05,
      (eb_knee.head, eb_knee.tail) if eb_knee else None)

# рычаг ступни: ХВОСТ в голеностопе (голова foot_L)
eb_foot = arm.data.edit_bones.get("CTRL_Foot_L")
eb_foot_b = arm.data.edit_bones.get("foot_L")
check("controls: хвост CTRL_Foot_L в голеностопе",
      eb_foot is not None and eb_foot_b is not None and
      abs(eb_foot.tail.x - eb_foot_b.head.x) < 1e-6 and
      abs(eb_foot.tail.z - eb_foot_b.head.z) < 1e-6,
      (eb_foot.tail, eb_foot_b.head) if eb_foot else None)

eb_pelvis = arm.data.edit_bones.get("pelvis")
check("controls: pelvis parented to CTRL_Body",
      eb_pelvis is not None and eb_pelvis.parent is not None and
      eb_pelvis.parent.name == "CTRL_Body")
eb_croot = arm.data.edit_bones.get("CTRL_Root")
eb_cbody = arm.data.edit_bones.get("CTRL_Body")
check("controls: CTRL_Root без родителя, CTRL_Body <- Root",
      eb_croot is not None and eb_croot.parent is None and
      eb_cbody is not None and eb_cbody.parent is eb_croot)
check("controls: CTRL кости use_deform=False",
      eb_croot.use_deform is False and
      arm.data.edit_bones.get("CTRL_Foot_L").use_deform is False)
check("controls: deform кости не тронуты (use_deform=True)",
      arm.data.edit_bones.get("thigh_L").use_deform is True)


def _pb(arm_obj, name):
    return arm_obj.pose.bones.get(name)


c_farmh = [c for c in _pb(arm, "NCPK_Arm_L_2").constraints
           if c.type == "IK"]
check("controls: IK руки на хелпере NCPK_Arm_L_2 (chain 2, use_tail)",
      len(c_farmh) == 1 and c_farmh[0].subtarget == "CTRL_Arm_L" and
      c_farmh[0].chain_count == 2 and
      getattr(c_farmh[0], "use_tail", True) is True,
      [(c.name, c.subtarget, c.chain_count) for c in c_farmh])
check("controls: IK руки имеет полюс локтя",
      c_farmh[0].pole_target == arm and
      c_farmh[0].pole_subtarget == "CTRL_Pole_Elbow_L" and
      isinstance(c_farmh[0].pole_angle, float),
      (c_farmh[0].pole_subtarget, c_farmh[0].pole_angle))
# хелпер плечо-локоть вдоль +Y, roll=0, сгиб назад (-X): theta=±180°
check("controls: полюс локтя ~ 180° (сгиб назад)",
      abs(abs(c_farmh[0].pole_angle) - math.pi) < 1e-6,
      c_farmh[0].pole_angle)
# хелперы натянуты сустав-в-сустав (плечо->локоть, локоть->запястье)
eb_h1 = arm.data.edit_bones.get("NCPK_Arm_L_1")
eb_uarm = arm.data.edit_bones.get("uarm_L")
eb_farm = arm.data.edit_bones.get("farm_L")
check("controls: NCPK_Arm_L_1 = плечо->локоть (настоящая длина)",
      eb_h1 is not None and eb_uarm is not None and eb_farm is not None and
      abs(eb_h1.head.x - eb_uarm.head.x) < 1e-6 and
      abs(eb_h1.tail.x - eb_farm.head.x) < 1e-6 and
      eb_h1.length > eb_uarm.length,
      (eb_h1.head, eb_h1.tail) if eb_h1 else None)
check("controls: NCPK_Arm_L_2 = локоть->запястье",
      abs(eb_h1.tail.x - eb_farm.head.x) < 1e-6 and
      abs(arm.data.edit_bones.get("NCPK_Arm_L_2").tail.x -
          eb_farm.tail.x) < 1e-6)
# остовные кости следуют за хелперами через COPY_TRANSFORMS
ct_uarm = [c for c in _pb(arm, "uarm_L").constraints
           if c.type == "COPY_TRANSFORMS"]
check("controls: uarm_L COPY_TRANSFORMS -> NCPK_Arm_L_1",
      len(ct_uarm) == 1 and ct_uarm[0].subtarget == "NCPK_Arm_L_1")
ct_thigh = [c for c in _pb(arm, "thigh_L").constraints
            if c.type == "COPY_TRANSFORMS"]
check("controls: thigh_L COPY_TRANSFORMS -> NCPK_Foot_L_1",
      len(ct_thigh) == 1 and ct_thigh[0].subtarget == "NCPK_Foot_L_1")
c_shinh = [c for c in _pb(arm, "NCPK_Foot_L_2").constraints
           if c.type == "IK"]
check("controls: IK ноги на хелпере NCPK_Foot_L_2 -> CTRL_Foot_L",
      len(c_shinh) == 1 and c_shinh[0].subtarget == "CTRL_Foot_L" and
      c_shinh[0].chain_count == 2)
check("controls: IK ноги имеет полюс колена (прямая нога: 0°)",
      c_shinh[0].pole_subtarget == "CTRL_Pole_Knee_L" and
      abs(c_shinh[0].pole_angle) < 1e-6,
      (c_shinh[0].pole_subtarget, c_shinh[0].pole_angle))
check("controls: у shin_L нет IK (только COPY_TRANSFORMS)",
      not any(c.type == "IK" for c in _pb(arm, "shin_L").constraints))
c_rot = [c for c in _pb(arm, "foot_L").constraints
         if c.type == "COPY_ROTATION"]
check("controls: foot_L COPY_ROTATION -> CTRL_Foot_L",
      len(c_rot) == 1 and c_rot[0].subtarget == "CTRL_Foot_L")
c_head = [c for c in _pb(arm, "head").constraints if c.type == "IK"]
check("controls: head IK -> CTRL_Head (chain 1, без полюса)",
      len(c_head) == 1 and c_head[0].subtarget == "CTRL_Head" and
      c_head[0].chain_count == 1 and
      getattr(c_head[0], "pole_subtarget", "") == "")
# пальцы: IK на последнем сегменте цепочки (кончик тянется к рычагу)
c_fng = [c for c in _pb(arm, "fng_thumb_2_L").constraints
         if c.type == "IK"]
check("controls: fng_thumb_2_L IK -> CTRL_Finger_thumb_L (chain 2)",
      len(c_fng) == 1 and c_fng[0].subtarget == "CTRL_Finger_thumb_L" and
      c_fng[0].chain_count == 2 and
      getattr(c_fng[0], "use_tail", True) is True,
      [(c.name, c.subtarget) for c in c_fng])
n_fng_ik = sum(1 for pb in arm.pose.bones
               if pb.name.startswith("fng_") and
               any(c.type == "IK" and c.name.startswith("NCP_")
                   for c in pb.constraints))
check("controls: все 10 пальцев имеют IK на свой рычаг",
      n_fng_ik == 10, n_fng_ik)
check("controls: у pelvis нет констрейнтов (анимация чистая)",
      len(_pb(arm, "pelvis").constraints) == 0)
n_cons = sum(1 for pb in arm.pose.bones
             for c in pb.constraints if c.name.startswith("NCP_"))
check("controls: 25 констрейнтов NCP_* (4 IK на хелперах, 8 CT, "
      "1 IK головы, 10 IK пальцев, 2 ROT)", n_cons == 25, n_cons)
check("controls: influence = 1 при ik_on=True", nctl.ik_on(arm))

nctl.set_ik(arm, False)
check("controls: set_ik(False) гасит influence (IK, CT и полюсы)",
      all(float(c.influence) == 0.0
          for pb in arm.pose.bones for c in pb.constraints
          if c.name.startswith("NCP_")) and not nctl.ik_on(arm))
nctl.set_ik(arm, True)
check("controls: set_ik(True) включает",
      nctl.ik_on(arm) and
      all(float(c.influence) == 1.0
          for pb in arm.pose.bones for c in pb.constraints
          if c.name.startswith("NCP_")))
n_reset = nctl.reset_helpers(arm)
check("controls: reset_helpers сбросил 8 хелперов", n_reset == 8, n_reset)

nctl.set_deform_hidden(arm, True)
check("controls: deform скрыты, CTRL видимы",
      arm.data.bones.get("thigh_L").hide is True and
      arm.data.bones.get("CTRL_Foot_L").hide is False)
nctl.set_deform_hidden(arm, False)
check("controls: show обратно",
      arm.data.bones.get("thigh_L").hide is False)

# пересборка: старые рычаги удалены, дубликатов нет, IK по флагу
rep2 = nctl.build_controls(arm, ik_on=False, hide_deform=True)
hn2 = nctl.handle_names(arm)
check("controls: пересборка — те же 21 CTRL-кость",
      sorted(hn2) == sorted(hn), hn2)
check("controls: пересборка — IK выключен", not nctl.ik_on(arm))
check("controls: пересборка — deform скрыты",
      arm.data.bones.get("thigh_L").hide is True and
      arm.data.bones.get("farm_L").hide is True)
nctl.build_controls(arm, ik_on=False, fingers=False, poles=False)
check("controls: пересборка без полюсов/пальцев — только 7 главных",
      sorted(nctl.handle_names(arm)) ==
      sorted(n for n in hn
             if not n.startswith(("CTRL_Finger_", "CTRL_Pole_"))),
      nctl.handle_names(arm))
nctl.build_controls(arm, ik_on=False)
nctl.set_deform_hidden(arm, False)
sel = nctl.select_handles(bpy.context, arm)
check("controls: select_handles выделил все и только CTRL",
      sorted(sel) == sorted(nctl.handle_names(arm)) and
      all(pb.select == pb.name.startswith("CTRL_")
          for pb in arm.pose.bones))

# ============ A2. v2.9.0: кольца/диски, режим T-поза/анимация, запись
finger_pbs = [pb for pb in arm.pose.bones
              if pb.name.startswith("CTRL_Finger_")]
check("v2.9.0: диски пальцев — custom_shape = NCP_Shape_Disc",
      len(finger_pbs) == 10 and
      all(pb.custom_shape is not None and
          pb.custom_shape.name == "NCP_Shape_Disc"
          for pb in finger_pbs),
      [pb.custom_shape.name if pb.custom_shape else None
       for pb in finger_pbs])
ring_names = ("CTRL_Root", "CTRL_Foot_L", "CTRL_Arm_L",
              "CTRL_Pole_Knee_L", "CTRL_Head")
check("v2.9.0: КОЛЬЦА на суставах (Root/Foot/Arm/Pole/Head)",
      all(_pb(arm, n) is not None and _pb(arm, n).custom_shape is not None
          and _pb(arm, n).custom_shape.name == "NCP_Shape_Ring"
          for n in ring_names),
      [(n, _pb(arm, n).custom_shape.name
        if _pb(arm, n) is not None and _pb(arm, n).custom_shape else None)
       for n in ring_names])
ring_mesh = bpy.data.meshes.get("NCP_Shape_Ring")
check("v2.9.0: кольцо полое (24 квада-ободка)",
      ring_mesh is not None and len(ring_mesh.faces) == 24,
      None if ring_mesh is None else len(ring_mesh.faces))
disc_mesh = bpy.data.meshes.get("NCP_Shape_Disc")
check("v2.9.0: диск полный (1 n-gon из 25 вершин)",
      disc_mesh is not None and len(disc_mesh.faces) == 1,
      None if disc_mesh is None else len(disc_mesh.faces))

# --- режим: T-поза / анимация
hpb = _pb(arm, "CTRL_Foot_L")
hpb.location = (0.3, -0.2, 0.1)
hpb.rotation_quaternion = (0.7, 0.1, 0.1, 0.7)
rep_t = nctl.set_mode(arm, "TPOSE")
check("v2.9.0: set_mode(TPOSE) — IK выкл, поза рычагов сброшена",
      not nctl.ik_on(arm) and arm["ncp_mode"] == "TPOSE" and
      tuple(hpb.location) == (0.0, 0.0, 0.0) and
      tuple(hpb.rotation_quaternion) == (1.0, 0.0, 0.0, 0.0),
      (rep_t, arm.get("ncp_mode")))
n_hlp = nctl.reset_pose(arm)
check("v2.9.0: reset_pose сбрасывает CTRL+NCPK (21+8 = 29), не трогает "
      "остовные", n_hlp == 29, n_hlp)
rep_a = nctl.set_mode(arm, "ANIM")
check("v2.9.0: set_mode(ANIM) — IK включён, режим на арматуре",
      nctl.ik_on(arm) and arm["ncp_mode"] == "ANIM", rep_a)

# --- запись движений по слоям
check("v2.9.0: rec_layer_of разложил рычаги по слоям",
      nctl.rec_layer_of("CTRL_Finger_thumb_L") == "fingers" and
      nctl.rec_layer_of("CTRL_Arm_L") == "arms" and
      nctl.rec_layer_of("CTRL_Pole_Elbow_R") == "arms" and
      nctl.rec_layer_of("CTRL_Foot_L") == "legs" and
      nctl.rec_layer_of("CTRL_Pole_Knee_L") == "legs" and
      nctl.rec_layer_of("CTRL_Root") == "body" and
      nctl.rec_layer_of("CTRL_Head") == "body" and
      nctl.rec_layer_of("thigh_L") is None)
n_rec_all = len(nctl.rec_layer_bones(arm))
check("v2.9.0: все 21 рычага распределены по слоям записи",
      n_rec_all == len(nctl.handle_names(arm)) == 21, n_rec_all)
nctl.rec_set_state(arm, {"fingers": False})
en = nctl.rec_layers_state(arm)
check("v2.9.0: rec_set_state — слои хранятся на арматуре",
      en["fingers"] is False and en["body"] is True and
      len(nctl.rec_layer_bones(arm)) == 11, en)
nctl.rec_set_state(arm, {})
for pb in arm.pose.bones:
    if pb.name.startswith("CTRL_Finger_"):
        pb.location = (0.05, 0.0, 0.0)
n_k = nctl.record_frame(arm, {"fingers": True})
fps = [pb for pb in arm.pose.bones if pb.name.startswith("CTRL_Finger_")]
check("v2.9.0: record_frame — 10 пальцев закейфреймены (loc+quat)",
      n_k == 10 and all(len(pb.keys) == 2 for pb in fps), n_k)
ok_on = nctl.record_set_enabled(True)
check("v2.9.0: record_set_enabled(True) — хэндлер зарегистрирован",
      ok_on and nctl.record_is_enabled() and
      nctl._rec_tick in bpy.app.handlers.depsgraph_update_post)
tick_scene = types.SimpleNamespace(objects=[arm])
nctl._rec_tick(tick_scene, None)
check("v2.9.0: автозапись — tick ставит кейфрейм рычагам слоёв",
      len(_pb(arm, "CTRL_Foot_L").keys) == 2,
      len(_pb(arm, "CTRL_Foot_L").keys))
nctl._rec_tick(tick_scene, None)
check("v2.9.0: поза не менялась — дублей нет",
      len(_pb(arm, "CTRL_Foot_L").keys) == 2)
_pb(arm, "CTRL_Foot_L").location = (0.1, 0.0, 0.0)
nctl._rec_tick(tick_scene, None)
check("v2.9.0: поза изменилась — кейфрейм добавлен",
      len(_pb(arm, "CTRL_Foot_L").keys) == 4,
      len(_pb(arm, "CTRL_Foot_L").keys))
ok_off = nctl.record_set_enabled(False)
check("v2.9.0: record_set_enabled(False) — хэндлер снят",
      not ok_off and not nctl.record_is_enabled() and
      nctl._rec_tick not in bpy.app.handlers.depsgraph_update_post)

# ---- creature-планы: рычаги по факту цепочек
arm_b2 = neuro_rig.build_armature(res_b["spec"], name="BipedArm")
_mode_set("OBJECT")
rep_b = nctl.build_controls(arm_b2, ik_on=True)
hn_b = nctl.handle_names(arm_b2)
check("controls biped: Foot/Arm/Head по имени",
      all(n in hn_b for n in ("CTRL_Foot_L", "CTRL_Foot_R", "CTRL_Arm_L",
                              "CTRL_Arm_R", "CTRL_Head")), hn_b)
# 7 главных + 2 полюса коленей + 10 пальцев (4-костные руки — без полюсов)
check("controls biped: полюсы коленей есть, локтей нет",
      "CTRL_Pole_Knee_L" in hn_b and
      "CTRL_Pole_Elbow_L" not in hn_b, hn_b)
check("controls biped: 19 CTRL-костей (7+2 полюса+10 пальцев)",
      len(hn_b) == 19, hn_b)
c_a4 = [c for c in _pb(arm_b2, "arm_L_4").constraints if c.type == "IK"]
check("controls biped: кончик руки arm_L_4 IK (chain 3, без полюса)",
      len(c_a4) == 1 and c_a4[0].subtarget == "CTRL_Arm_L" and
      c_a4[0].chain_count == 3 and
      getattr(c_a4[0], "pole_subtarget", "") == "")
c_ll = [c for c in _pb(arm_b2, "leg_L_lower").constraints
        if c.type == "IK"]
check("controls biped: leg_L_lower IK -> CTRL_Foot_L с полюсом колена",
      len(c_ll) == 1 and c_ll[0].subtarget == "CTRL_Foot_L" and
      c_ll[0].pole_subtarget == "CTRL_Pole_Knee_L")
c_fng_b = [c for c in _pb(arm_b2, "fng_arm_L_1_2").constraints
           if c.type == "IK"]
check("controls biped: палец fng_arm_L_1_2 -> CTRL_Finger_arm_L_1",
      len(c_fng_b) == 1 and
      c_fng_b[0].subtarget == "CTRL_Finger_arm_L_1", hn_b)
check("controls biped: рычаг пальца parent к кисти",
      arm_b2.data.edit_bones.get("CTRL_Finger_arm_L_1").parent is not None
      and arm_b2.data.edit_bones.get(
          "CTRL_Finger_arm_L_1").parent.name == "arm_L_4")

arm_q = neuro_rig.build_armature(res_q["spec"], name="QuadArm")
_mode_set("OBJECT")
rep_q = nctl.build_controls(arm_q, ik_on=True)
hn_q = nctl.handle_names(arm_q)
check("controls quad: 4 ноги front/hind L/R",
      all(n in hn_q for n in ("CTRL_Foot_front_L", "CTRL_Foot_front_R",
                              "CTRL_Foot_hind_L", "CTRL_Foot_hind_R")),
      hn_q)
check("controls quad: Root/Body/Head/Tail",
      all(n in hn_q for n in ("CTRL_Root", "CTRL_Body", "CTRL_Head",
                              "CTRL_Tail")), hn_q)
# 8 главных + 4 полюса (все ноги 2-бонные) + 0 пальцев
check("controls quad: 12 CTRL-костей (8+4 полюса)", len(hn_q) == 12, hn_q)
check("controls quad: 4 полюса коленей front/hind",
      all(n in hn_q for n in ("CTRL_Pole_Knee_front_L",
                              "CTRL_Pole_Knee_front_R",
                              "CTRL_Pole_Knee_hind_L",
                              "CTRL_Pole_Knee_hind_R")), hn_q)
c_t3 = [c for c in _pb(arm_q, "tail_3").constraints if c.type == "IK"]
check("controls quad: хвост tail_3 IK -> CTRL_Tail (chain 3, без полюса)",
      len(c_t3) == 1 and c_t3[0].subtarget == "CTRL_Tail" and
      c_t3[0].chain_count == 3 and
      getattr(c_t3[0], "pole_subtarget", "") == "")
c_fl = [c for c in _pb(arm_q, "leg_front_L_lower").constraints
        if c.type == "IK"]
check("controls quad: leg_front_L_lower IK -> CTRL_Foot_front_L",
      len(c_fl) == 1 and c_fl[0].subtarget == "CTRL_Foot_front_L" and
      c_fl[0].pole_subtarget == "CTRL_Pole_Knee_front_L")

arm_d = neuro_rig.build_armature(res_d["spec"], name="DragonArm")
_mode_set("OBJECT")
nctl.build_controls(arm_d, ik_on=True)
hn_d = nctl.handle_names(arm_d)
check("controls dragon: крылья CTRL_Wing_L/R",
      all(n in hn_d for n in ("CTRL_Wing_L", "CTRL_Wing_R")), hn_d)
c_w = [c for c in _pb(arm_d, "wing_L_4").constraints if c.type == "IK"]
check("controls dragon: wing_L_4 IK -> CTRL_Wing_L (chain 3)",
      len(c_w) == 1 and c_w[0].subtarget == "CTRL_Wing_L" and
      c_w[0].chain_count == 3)
c_hd = [c for c in _pb(arm_d, "head").constraints if c.type == "IK"]
check("controls dragon: шея+голова в IK головы (chain 3)",
      len(c_hd) == 1 and c_hd[0].subtarget == "CTRL_Head" and
      c_hd[0].chain_count == 3)
# косметика не роняет сборку: custom shapes в стабе нет — ок
# 10 главных + 4 полюса + пальцы крыльев (по факту спеки)
n_fng_d = len([n for n in dn if n.startswith("fng_")])
check("controls dragon: всего %d CTRL (10+4 полюса+%d пальцев)" % (
        14 + n_fng_d // 2, n_fng_d // 2),
      len(hn_d) == 14 + n_fng_d // 2, (len(hn_d), n_fng_d))

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

check("bl_info version 2.9.0", pkg.bl_info["version"] == (2, 9, 0),
      pkg.bl_info["version"])
cls_names = [c.__name__ for c in pkg.classes]
check("NCP_OT_build_skeleton зарегистрирован",
      "NCP_OT_build_skeleton" in cls_names)
panels = [n for n in cls_names if n.startswith("NCP_PT_")]
check("12 панелей", len(panels) == 12, panels)
check("панель рычагов NCP_PT_handles есть",
      "NCP_PT_handles" in panels and "NCP_OT_select_handles" in cls_names
      and "NCP_OT_rebuild_handles" in cls_names)
check("v2.9.0: панель режима NCP_PT_modes + операторы set_mode/record_key",
      "NCP_PT_modes" in panels and "NCP_OT_set_mode" in cls_names and
      "NCP_OT_record_key" in cls_names)
check("AddonPreferences класс есть", "NCP_AddonPrefs" in cls_names)
ann = getattr(pkg.NCP_Props, "__annotations__", {})
for p in ("auto_rig", "body_plan", "add_fingers", "orient_auto",
          "mesh_rot_x", "mesh_rot_z", "debug_landmarks",
          "ctrl_ik", "ctrl_hide_deform", "ctrl_poles", "ctrl_fingers",
          "rig_mode", "rec_auto", "rec_body", "rec_arms", "rec_legs",
          "rec_fingers", "rec_wings"):
    check("prop NCP_Props.%s" % p, p in ann)
rm = ann.get("rig_mode")
rm_ids = [it[0] for it in rm.kwargs["items"]] if isinstance(rm, _PropStub) \
    else []
check("rig_mode enum: ANIM/TPOSE",
      set(rm_ids) == {"ANIM", "TPOSE"}, rm_ids)
bp = ann.get("body_plan")
bp_ids = [it[0] for it in bp.kwargs["items"]] if isinstance(bp, _PropStub) \
    else []
check("Body plan enum: AUTO/HUMAN/QUADRUPED/WINGED_QUAD/BIRD/DRAGON",
      set(bp_ids) == {"AUTO", "HUMAN", "QUADRUPED", "WINGED_QUAD",
                      "BIRD", "DRAGON"}, bp_ids)
with open(os.path.join(ADDON, "blender_manifest.toml"), "rb") as f:
    man = tomllib.load(f)
check("manifest version 2.9.0", man["version"] == "2.9.0",
      man["version"])
with open(os.path.join(ADDON, "INSTALL_RU.txt"), encoding="utf-8") as f:
    install_txt = f.read()
check("INSTALL_RU: секция 2.9.0 (кольца + режим + запись по слоям)",
      "ЧТО НОВОГО В 2.9.0" in install_txt and
      "кольц" in install_txt.lower() and
      "T-Поза" in install_txt and
      "сло" in install_txt.lower())
check("INSTALL_RU: секция 2.8.0 (полюсы + пальцы)",
      "ЧТО НОВОГО В 2.8.0" in install_txt and
      "полюс" in install_txt.lower() and
      "CTRL_Finger" in install_txt)
check("INSTALL_RU: секция 2.7.0 (рычаги)",
      "ЧТО НОВОГО В 2.7.0" in install_txt and
      "рычаг" in install_txt.lower())
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
