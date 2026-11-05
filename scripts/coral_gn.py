#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coral_gn.py — v13 «blender_stroika»
Процедурный коралл по референсу 0682af35916265e314b019420a043024.jpg:
  форма — ВЕРТИКАЛЬНЫЙ скульптурный рост: ствол с луковичными утолщениями
          и перетяжками, S-изгиб, спиральный перекрут рёбер-складок,
          3 ответвления, отходящие вверх-наружу; всё сглажено (blur);
  дырки — крупные глубокие полости (Boolean Difference, сферы) + ПЛОТНАЯ
          сетка мелких пор между ними (второй boolean-проход), как на
          керамическом референсе.

База строится Python'ом (bmesh, parallel-transport sweep) как набор
ЗАМКНУТЫХ СОЛИДОВ (внешняя + внутренняя поверхность, стенка CORAL_WALL =
глубина пор). Сферы-резаки внутри КАЖДОГО прохода не перекрываются
(spacing ≥ 2·r_max), иначе FLOAT-солвер даёт кольца-шрамы.

env: CORAL_SEED (7), CORAL_H (1.34), CORAL_WALL (0.030), CORAL_SOLVER (FLOAT),
     CORAL_LOW=1 (черновой рендер), CORAL_OUT=path, CORAL_BLEND=path,
     CORAL_DIAG=1, GN_<ИМЯ>=знач (переопределение слайдеров).
"""

import bpy, bmesh, math, os, time, random
from math import cos, sin, pi
from mathutils import Vector, Matrix

T0 = time.time()
LOW   = os.environ.get('CORAL_LOW') == '1'
DIAG  = os.environ.get('CORAL_DIAG') == '1'
OUT   = os.environ.get('CORAL_OUT')
BLEND = os.environ.get('CORAL_BLEND')

# ---------------------------------------------------------------- утилиты GN
class G:
    """Обёртка над построением дерева нод. Все «возвращающие» хелперы дают (node, socket)."""

    def __init__(self, ng):
        self.ng = ng
        self._row = 0

    def n(self, t, x=None, y=None, **props):
        nd = self.ng.nodes.new(t)
        if x is None:
            x = -1400
            y = 900 - self._row * 40
            self._row += 1
        nd.location = (x, y)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    def link(self, a, ao, b, bi=None):
        """Две формы: link(node, out, node, in) или link((node,out), node, in)."""
        if isinstance(a, tuple):
            src_node, src_sock = a[0], a[1]
            dst_node, dst_sock = ao, b
        else:
            src_node, src_sock, dst_node, dst_sock = a, ao, b, bi
        self.ng.links.new(src_node.outputs[src_sock], dst_node.inputs[dst_sock])

    def _src(self, v):
        if isinstance(v, tuple):
            return v
        if isinstance(v, (bpy.types.Node,)):
            return (v, 0)
        return None

    def m(self, op, a=None, b=None, clamp=False):
        """Скалярный Math. Скалярные аргументы пишем в сокет, ссылочные — линкуем."""
        nd = self.n('ShaderNodeMath', operation=op, use_clamp=clamp)
        for i, v in ((0, a), (1, b)):
            if v is None:
                continue
            s = self._src(v)
            if s is None:
                nd.inputs[i].default_value = float(v)
            else:
                self.link(s[0], s[1], nd, i)
        return (nd, 0)

    def vec(self, op, a=None, b=None, scale=None):
        nd = self.n('ShaderNodeVectorMath', operation=op)
        for i, v in ((0, a), (1, b)):
            if v is None:
                continue
            s = self._src(v)
            if s is None:
                raise ValueError('vector input got scalar')
            self.link(s[0], s[1], nd, i)
        if scale is not None:
            sidx = next((i for i, s in enumerate(nd.inputs) if s.name == 'Scale'), 2)
            s = self._src(scale)
            if s is None:
                nd.inputs[sidx].default_value = float(scale)
            else:
                self.link(s[0], s[1], nd, sidx)
        return (nd, 0)

    def val(self, v, name=None):
        nd = self.n('ShaderNodeValue')
        nd.outputs[0].default_value = float(v)
        if name:
            nd.label = name
        return (nd, 0)

    def v3(self, x, y, z, name=None):
        nd = self.n('FunctionNodeInputVector')
        nd.vector = (x, y, z)
        if name:
            nd.label = name
        return (nd, 'Vector')

    def sep(self, vec_t):
        nd = self.n('ShaderNodeSeparateXYZ')
        self.link(vec_t[0], vec_t[1], nd, 0)
        return nd

    def comb(self, x, y, z):
        nd = self.n('ShaderNodeCombineXYZ')
        for i, v in ((0, x), (1, y), (2, z)):
            s = self._src(v)
            if s is None:
                nd.inputs[i].default_value = float(v)
            else:
                self.link(s[0], s[1], nd, i)
        return (nd, 0)

    def smooth(self, x, e0, e1):
        """smoothstep(x, e0, e1); края — константы. Возвращает (node, 0)."""
        span = float(e1) - float(e0)
        t = self.m('DIVIDE', self.m('SUBTRACT', x, float(e0)), span, clamp=True)
        t2 = self.m('MULTIPLY', t, t)
        s = self.m('MULTIPLY', t2, self.m('SUBTRACT', 3.0, self.m('MULTIPLY', t, 2.0)))
        return s


def set_pos(g, cur, pos_t):
    if isinstance(pos_t, bpy.types.Node):
        pos_t = (pos_t, 0)
    sp = g.n('GeometryNodeSetPosition')
    g.link(cur[0], cur[1], sp, 'Geometry')
    g.link(pos_t[0], pos_t[1], sp, 'Position')
    return (sp, 0)


def pos_in(g):
    return (g.n('GeometryNodeInputPosition'), 'Position')


def nrm_in(g):
    return (g.n('GeometryNodeInputNormal'), 'Normal')


# ---------------------------------------------------------------- математика формы
def sstep(x, e0, e1):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


def catmull(ks, z):
    """Catmull-Rom по опорным точкам ks=[(z,r),...] (z отсортированы)."""
    n = len(ks)
    if z <= ks[0][0]:
        return ks[0][1]
    if z >= ks[-1][0]:
        return ks[-1][1]
    i = 0
    while i < n - 2 and z > ks[i + 1][0]:
        i += 1
    z0, z1 = ks[i][0], ks[i + 1][0]
    t = (z - z0) / (z1 - z0)
    p1 = ks[i][1];        p2 = ks[i + 1][1]
    p0 = ks[max(i - 1, 0)][1]
    p3 = ks[min(i + 2, n - 1)][1]
    a = 2*t**3 - 3*t**2 + 1
    b = t**3 - 2*t**2 + t
    c = -2*t**3 + 3*t**2
    d = t**3 - t**2
    return a*p1 + b*(p2-p0)*0.5 + c*p2 + d*(p3-p1)*0.5


# ---------------------------------------------------------------- база: ствол+ветви
def build_base_mesh(name='CoralBase'):
    """Вертикальный скульптурный коралл: ствол с луковичными утолщениями,
    S-изгиб, спиральный перекрут рёбер, 3 ответвления. Каждый элемент —
    замкнутый полый солид (стенка CORAL_WALL)."""
    wall = float(os.environ.get('CORAL_WALL', '0.030'))
    H    = float(os.environ.get('CORAL_H', '1.34'))
    seed = int(float(os.environ.get('CORAL_SEED', '7')))
    rng  = random.Random(seed)

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    # ---- ствол: ось (S-изгиб) ----
    def trunk_pt(s):
        z = s * H
        ramp = sstep(s, 0.10, 0.26)   # ножка вертикальна: низ не тонет под пол
        x = ramp * (0.15 * sin(2.5 * pi * s + 0.4) + 0.05 * sin(6.1 * s))
        y = ramp * (0.11 * sin(1.9 * pi * s + 1.7) + 0.04 * cos(5.2 * s + 0.6))
        return Vector((x, y, z))

    # ---- ствол: радиус (нога -> перетяжки/луковицы -> кончик) ----
    KS = [(0.00, 0.340), (0.05, 0.380), (0.14, 0.320), (0.22, 0.200),
          (0.29, 0.185), (0.36, 0.235), (0.44, 0.250), (0.53, 0.190),
          (0.62, 0.250), (0.73, 0.275), (0.84, 0.185), (0.93, 0.205),
          (1.06, 0.140), (1.18, 0.105), (1.30, 0.070), (1.36, 0.035)]
    KS = [(z * H / 1.34, r) for (z, r) in KS]

    def trunk_r(s):
        z = s * H
        r = catmull(KS, z)
        r *= 1.0 + 0.028 * sin(9.0 * s * pi + 1.3)          # средняя волна
        return r

    # ---- складки-лужи: k гармоник с ДРЕЙФОМ фазы по высоте (перекрут) ----
    HARM = [(3, 0.100, 2.4 * rng.uniform(0.85, 1.15), rng.uniform(0, 6.28)),
            (4, 0.045, -1.8 * rng.uniform(0.85, 1.15), rng.uniform(0, 6.28)),
            (5, 0.035, 1.25 * rng.uniform(0.85, 1.15), rng.uniform(0, 6.28))]

    def lump(ths, s, base_r, amp=1.0, end_fade=0.88):
        f = sstep(s, 0.08, 0.22) * (1.0 - sstep(s, end_fade, 0.995))
        v = 1.0
        for (k, a, w, ph) in HARM:
            v += a * amp * f * math.cos(k * ths + ph + w * s * H)
        return base_r * v

    def frames(pts):
        """Parallel transport: n,b вдоль точек spine."""
        n = len(pts)
        tans = []
        for i in range(n):
            a = pts[max(i - 1, 0)]; b = pts[min(i + 1, n - 1)]
            tans.append((b - a).normalized())
        nrm = Vector((1, 0, 0))
        if abs(nrm.dot(tans[0])) > 0.9:
            nrm = Vector((0, 1, 0))
        nrm = (nrm - tans[0] * nrm.dot(tans[0])).normalized()
        out = []
        for i in range(n):
            if i > 0:
                q = tans[i - 1].rotation_difference(tans[i])
                nrm = (q @ nrm).normalized()
            out.append((tans[i], nrm.copy(), tans[i].cross(nrm)))
        return out

    def tube(pts, fr, rad_fun, ns):
        """rad_fun(s, theta)->r; возвращает rows[i][j] точек."""
        rows = []
        for i, (t, nv, bv) in enumerate(fr):
            s = i / (len(fr) - 1)
            row = []
            for j in range(ns):
                th = 2.0 * pi * j / ns
                r = rad_fun(s, th)
                row.append(bm.verts.new(pts[i] + nv * (r * cos(th)) + bv * (r * sin(th))))
            rows.append(row)
        return rows

    def stitch(rows):
        for i in range(len(rows) - 1):
            for j in range(len(rows[0])):
                j2 = (j + 1) % len(rows[0])
                bm.faces.new((rows[i][j], rows[i][j2], rows[i + 1][j2], rows[i + 1][j]))

    def hollow_solid(pts, fr, rad_fun, ns):
        """Замкнутый полый солид: внешняя+внутренняя труба, ободки на торцах."""
        ro = tube(pts, fr, lambda s, th: max(rad_fun(s, th), wall * 3.0), ns)
        ri = tube(pts, fr, lambda s, th: max(rad_fun(s, th) - wall, wall * 0.4), ns)
        stitch(ro); stitch(ri)
        nsl = len(ro[0])
        for j in range(nsl):                      # ободок внизу
            j2 = (j + 1) % nsl
            bm.faces.new((ro[0][j], ro[0][j2], ri[0][j2], ri[0][j]))
        for j in range(nsl):                      # ободок наверху
            j2 = (j + 1) % nsl
            bm.faces.new((ro[-1][j2], ro[-1][j], ri[-1][j], ri[-1][j2]))
        r_e = max(rad_fun(0.0, 0.0) - wall, wall * 0.4) * 0.6
        c0 = bm.verts.new(pts[0] + fr[0][0] * r_e)    # конус-дно полости
        for j in range(nsl):
            j2 = (j + 1) % nsl
            bm.faces.new((ri[0][j2], ri[0][j], c0))
        r_t = max(rad_fun(1.0, 0.0) - wall, wall * 0.4) * 0.6
        c1 = bm.verts.new(pts[-1] - fr[-1][0] * r_t)  # конус у кончика
        for j in range(nsl):
            j2 = (j + 1) % nsl
            bm.faces.new((ri[-1][j], ri[-1][j2], c1))

    # ---- ствол ----
    NZ, NS = 130, 56
    pts = [trunk_pt(i / (NZ - 1)) for i in range(NZ)]
    fr = frames(pts)
    def trunk_rad(s, th):
        r = lump(th, s, trunk_r(s))
        return max(r, wall * 3.0)
    hollow_solid(pts, fr, trunk_rad, NS)
    print(f'[trunk] rows={NZ} ring={NS}', flush=True)

    # ---- ответвления: (t0 на стволе, азимут, длина, толщина) ----
    branches = []
    base_b = [(0.30, -0.35, 0.55, 0.62),
              (0.56, 2.80, 0.60, 0.66),
              (0.80, 1.35, 0.40, 0.70),
              (1.00, 0.40, 0.26, 0.55)]
    for (t0, phi, L, rs) in base_b:
        t0 += rng.uniform(-0.03, 0.03)
        phi += rng.uniform(-0.15, 0.15)
        branches.append((t0, phi, L, rs))

    if os.environ.get('CORAL_TRUNK_ONLY'):
        branches = []
    for bi, (t0, phi, L, rs) in enumerate(branches):
        p0 = trunk_pt(t0)
        r0 = trunk_r(t0) * rs
        radial = Vector((cos(phi), sin(phi), 0.0))
        side = Vector((-sin(phi), cos(phi), 0.0))
        splay = 0.75 - 0.35 * t0                       # чем выше — тем вертикальнее
        hb = 0.72 * L
        w1, w2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        NBZ, NBS = 66, 48
        bpts = []
        for i in range(NBZ):
            s = i / (NBZ - 1)
            p = (p0 + radial * (L * splay * s ** 0.9)
                     + Vector((0, 0, hb * s))
                     + side * (0.10 * L * sin(pi * s * 1.15 + w1))
                     + radial * (0.06 * L * sin(2.0 * pi * s * 0.9 + w2)))
            bpts.append(p)
        bfr = frames(bpts)
        ph3, ph4 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        tw3 = rng.uniform(1.5, 2.6)
        def b_rad(s, th, r0=r0, ph3=ph3, ph4=ph4, tw3=tw3):
            grow = 1.0 + 0.16 * sin(pi * s ** 0.85)
            taper = 1.0 - 0.74 * sstep(s, 0.50, 0.99)
            f = sstep(s, 0.05, 0.25) * (1.0 - sstep(s, 0.86, 0.995))
            v = 1.0 + 0.09 * f * cos(3 * th + ph3 + tw3 * s)
            v += 0.055 * f * cos(4 * th + ph4 - 1.4 * s)
            return max(r0 * grow * taper * v, wall * 3.0)
        hollow_solid(bpts, bfr, b_rad, NBS)
        print(f'[branch{bi}] t0={t0:.2f} phi={phi:.2f} L={L} r0={r0:.3f}', flush=True)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------- дерево GN
SOCKETS = [
    # (имя, тип, дефолт, min, max)
    ('Geometry',        'NodeSocketGeometry', None, None, None),
    ('Wobble',          'NodeSocketFloat', 0.006, 0.0, 0.08),
    ('Pores Enable',    'NodeSocketFloat', 1.0, 0.0, 1.0),
    ('Pore Spacing',    'NodeSocketFloat', 0.155, 0.05, 0.3),
    ('Pore Radius',     'NodeSocketFloat', 0.068, 0.005, 0.09),
    ('Pore Var',        'NodeSocketFloat', 0.12, 0.0, 1.0),
    ('Fine Spacing',    'NodeSocketFloat', 0.064, 0.02, 0.2),
    ('Fine Radius',     'NodeSocketFloat', 0.024, 0.005, 0.05),
    ('Extra Object',    'NodeSocketObject', None, None, None),
]

def build_gn(name='CoralGN'):
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    for (nm, st, dv, mn, mx) in SOCKETS:
        it = ng.interface.new_socket(nm, in_out='INPUT', socket_type=st)
        if dv is not None:
            it.default_value = dv
        if mn is not None:
            it.min_value = mn
            it.max_value = mx
    g = G(ng)
    gi = g.n('NodeGroupInput'); gi.location = (-1800, 0)
    go = g.n('NodeGroupOutput'); go.location = (1600, 0)
    stages = {}

    def S(nm):
        return (gi, nm)

    # ---- вход: базовая сетка + Extra Object ----
    cur = S('Geometry')
    oi_e = g.n('GeometryNodeObjectInfo', transform_space='ORIGINAL')
    g.link(S('Extra Object'), oi_e, 'Object')
    oi_e.inputs['As Instance'].default_value = False
    jn = g.n('GeometryNodeJoinGeometry')
    g.link(cur[0], cur[1], jn, 'Geometry')
    g.link(oi_e, 'Geometry', jn, 'Geometry')
    cur = (jn, 0)

    # ---- WOBBLE: лёгкий органический шум вдоль нормали ----
    p1 = pos_in(g); se1 = g.sep(p1)
    noi = g.n('ShaderNodeTexNoise')
    g.link(g.comb((se1, 'X'), (se1, 'Y'), (se1, 'Z')), noi, 'Vector')
    noi.inputs['Scale'].default_value = 3.0
    noi.inputs['Detail'].default_value = 2.0
    wob = g.m('MULTIPLY', (noi, 'Fac'), S('Wobble'))
    wob = g.m('SUBTRACT', wob, g.m('MULTIPLY', S('Wobble'), 0.5))
    cur = set_pos(g, cur, g.vec('ADD', p1, g.vec('SCALE', nrm_in(g), scale=wob)))
    stages['wobble'] = cur

    # ---- PORES: маска «наружу + по высоте»; крупные полости — пятнами (noise) ----
    p4 = pos_in(g); se4 = g.sep(p4)
    x4, y4, z4 = (se4, 'X'), (se4, 'Y'), (se4, 'Z')
    sen = g.sep(nrm_in(g))
    nxn, nyn = (sen, 'X'), (sen, 'Y')
    lr = g.m('POWER', g.m('ADD', g.m('MULTIPLY', x4, x4),
                          g.m('MULTIPLY', y4, y4)), 0.5)
    dotp = g.m('ADD', g.m('MULTIPLY', nxn, x4), g.m('MULTIPLY', nyn, y4))
    cosb = g.m('DIVIDE', dotp, g.m('MAXIMUM', lr, 0.02))
    out_ok = g.smooth(cosb, 0.02, 0.18)
    en = S('Pores Enable')
    zk_b = g.m('MULTIPLY', g.smooth(z4, 0.10, 0.20),
               g.m('SUBTRACT', 1.0, g.smooth(z4, 1.12, 1.30)))
    noi2 = g.n('ShaderNodeTexNoise')
    g.link(g.comb(x4, y4, z4), noi2, 'Vector')
    noi2.inputs['Scale'].default_value = 2.4
    noi2.inputs['Detail'].default_value = 2.0
    clump = g.m('ADD', 0.45, g.m('MULTIPLY', (noi2, 'Fac'), 0.60))
    mask_b = g.m('MULTIPLY', out_ok, zk_b)
    mask_b = g.m('MULTIPLY', mask_b, clump)
    mask_b = g.m('MULTIPLY', mask_b, en)
    mask_f = g.m('MULTIPLY', out_ok, g.smooth(z4, 0.045, 0.095))
    mask_f = g.m('MULTIPLY', mask_f, en)

    def pore_spheres(mask_, dens, dist_min, rad, var, seed, mesh, depth=2.0):
        dpf = g.n('GeometryNodeDistributePointsOnFaces',
                  distribute_method='POISSON')
        g.link(mesh[0], mesh[1], dpf, 'Mesh')
        dpf.inputs['Density Max'].default_value = dens  # в POISSON именно Density Max
        dpf.inputs['Seed'].default_value = seed
        g.link(dist_min[0], dist_min[1], dpf, 'Distance Min')
        g.link(mask_[0], mask_[1], dpf, 'Density Factor')
        rnd = g.n('FunctionNodeRandomValue', data_type='FLOAT')
        rnd.inputs['Seed'].default_value = 11 + seed
        g.link(g.m('SUBTRACT', 1.0, g.m('MULTIPLY', var, 0.5)), rnd, 'Min')
        g.link(g.m('ADD', 1.0, g.m('MULTIPLY', var, 0.5)), rnd, 'Max')
        ri = g.m('MULTIPLY', (rnd, 'Value'), rad)
        ico = g.n('GeometryNodeMeshIcoSphere')
        ico.inputs['Radius'].default_value = 1.0
        ico.inputs['Subdivisions'].default_value = 3
        inst = g.n('GeometryNodeInstanceOnPoints')
        g.link(dpf, 'Points', inst, 'Points')
        g.link(dpf, 'Rotation', inst, 'Rotation')   # +Z вдоль нормали
        g.link(ico, 'Mesh', inst, 'Instance')
        # эллипсоид: глубина вдоль нормали xdepth -> пробивает стенку и сплавы
        g.link(g.comb(ri, ri, g.m('MULTIPLY', ri, depth)), inst, 'Scale')
        rlz = g.n('GeometryNodeRealizeInstances')
        rlz.inputs['Realize All'].default_value = True
        g.link(inst, 'Instances', rlz, 'Geometry')
        return (rlz, 0)

    def boolean_diff(mesh_a, cut):
        boo = g.n('GeometryNodeMeshBoolean', operation='DIFFERENCE')
        try:
            boo.solver = os.environ.get('CORAL_SOLVER', 'FLOAT')
        except Exception as e:
            print(f'[pores] solver set failed: {e}', flush=True)
        for opt in ('Self Intersection', 'Hole Tolerant'):
            if opt in boo.inputs:          # у FLOAT-солвера этих сокетов нет
                boo.inputs[opt].default_value = True
        g.link(mesh_a[0], mesh_a[1], boo, 'Mesh 1')
        g.link(cut[0], cut[1], boo, 'Mesh 2')
        return (boo, 0)

    # проход 1: плотная мелкая губа по всему (сферы между собой не перекрываются)
    s2 = pore_spheres(mask_f, 520.0, S('Fine Spacing'), S('Fine Radius'),
                      S('Pore Var'), 5, cur, depth=1.5)
    cur = boolean_diff(cur, s2)
    # проход 2: крупные полости ПОД конец — чистая ровная кромка сквозь губку
    s1 = pore_spheres(mask_b, 240.0, S('Pore Spacing'), S('Pore Radius'),
                      S('Pore Var'), 3, cur, depth=1.9)
    cur = boolean_diff(cur, s1)
    # сварка: убивает вырожденные грани/рёбра boolean (источник «шипов»)
    mbd = g.n('GeometryNodeMergeByDistance')
    g.link(cur[0], cur[1], mbd, 'Geometry')
    mbd.inputs['Distance'].default_value = 0.0008
    cur = (mbd, 0)
    stages['pores'] = cur

    final = stages.get(os.environ.get('CORAL_STAGE', ''), cur)
    g.link(final[0], final[1], go, 'Geometry')

    # ---- верификация достижимости от Group Output ----
    reach = set()
    stack = [lnk.from_node for inp in go.inputs for lnk in inp.links]
    while stack:
        nd = stack.pop()
        if nd in reach:
            continue
        reach.add(nd)
        for i in nd.inputs:
            for lnk in i.links:
                stack.append(lnk.from_node)
    dead = [nd.name for nd in ng.nodes if nd not in reach]
    print(f'[verify] nodes={len(ng.nodes)} links={len(ng.links)} '
          f'reachable={len(reach)} dead={dead}', flush=True)
    mism = []
    for lnk in ng.links:
        ft, tt = lnk.from_socket.type, lnk.to_socket.type
        if ft != tt and not (ft in ('FLOAT', 'INT', 'BOOLEAN', 'VALUE') and
                             tt in ('FLOAT', 'INT', 'BOOLEAN', 'VALUE')):
            mism.append(f'{lnk.from_node.name}:{lnk.from_socket.name}'
                        f'({ft})->{lnk.to_node.name}:{lnk.to_socket.name}({tt})')
    if mism:
        print(f'[verify] TYPE-MISMATCH: {mism}', flush=True)
    return ng


# ---- пост-обработка: blur + shade smooth (ПОСЛЕ voxel-remesh) ----
def build_post(name='CoralPost'):
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
    it = ng.interface.new_socket('Smooth', in_out='INPUT', socket_type='NodeSocketInt')
    it.default_value = 3
    it.min_value = 0; it.max_value = 24
    g = G(ng)
    gi = g.n('NodeGroupInput'); gi.location = (-600, 0)
    go = g.n('NodeGroupOutput'); go.location = (600, 0)
    blr = g.n('GeometryNodeBlurAttribute')
    blr.data_type = 'FLOAT_VECTOR'
    pos = pos_in(g)
    g.link(pos, blr, 'Value')
    blr.inputs['Weight'].default_value = 1.0
    g.link((gi, 'Smooth'), blr, 'Iterations')
    cur = set_pos(g, (gi, 'Geometry'), blr)
    sss = g.n('GeometryNodeSetShadeSmooth')
    sss.domain = 'FACE'
    g.link(cur[0], cur[1], sss, 'Geometry')
    sss.inputs['Shade Smooth'].default_value = True
    g.link((sss, 0), go, 'Geometry')
    return ng


# ---------------------------------------------------------------- материал/свет
def build_scene():
    obj = bpy.data.objects.get('CoralBase')
    mat = bpy.data.materials.new('CoralWhite')
    mat.use_nodes = True
    b = mat.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (0.93, 0.92, 0.90, 1.0)
    b.inputs['Roughness'].default_value = 0.48
    if 'Specular IOR Level' in b.inputs:
        b.inputs['Specular IOR Level'].default_value = 0.35
    obj.data.materials.append(mat)

    # пол-подиум в тон фона (мягкая тень, как на референсе)
    pme = bpy.data.meshes.new('Floor')
    pmb = bmesh.new()
    v = [pmb.verts.new((-4, -4, 0)), pmb.verts.new((4, -4, 0)),
         pmb.verts.new((4, 4, 0)), pmb.verts.new((-4, 4, 0))]
    pmb.faces.new(v)
    pmb.to_mesh(pme); pmb.free()
    flo = bpy.data.objects.new('Floor', pme)
    bpy.context.collection.objects.link(flo)
    fmat = bpy.data.materials.new('FloorMat')
    fmat.use_nodes = True
    fb = fmat.node_tree.nodes['Principled BSDF']
    fb.inputs['Base Color'].default_value = (0.87, 0.88, 0.90, 1.0)
    fb.inputs['Roughness'].default_value = 0.65
    pme.materials.append(fmat)

    w = bpy.context.scene.world or bpy.data.worlds.new('World')
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.87, 0.88, 0.90, 1.0)
    bg.inputs['Strength'].default_value = 0.45

    AIM = Vector((-0.05, 0, 0.60))

    def light(name, loc, size, energy):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.size = size; ld.energy = energy
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        bpy.context.collection.objects.link(lo)
        d = AIM - lo.location
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return lo
    light('Key', (3.0, -2.6, 2.6), 3.5, 135)
    light('Fill', (-3.2, -1.6, 1.2), 4.0, 48)
    light('Rim', (0.0, 2.8, 2.2), 3.0, 75)

    cd = bpy.data.cameras.new('Cam')
    cd.lens = 55
    cam = bpy.data.objects.new('Cam', cd)
    bpy.context.collection.objects.link(cam)
    cam.location = (2.2, -2.75, 0.92)
    d = AIM - cam.location
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.camera = cam

    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 24 if LOW else 72
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 800 if LOW else 1000
    sc.render.resolution_y = 880 if LOW else 1100
    sc.render.image_settings.file_format = 'PNG'
    sc.view_settings.view_transform = 'Standard'


# ---------------------------------------------------------------- main
def main():
    print(f'[coral] v14 start LOW={LOW} DIAG={DIAG}', flush=True)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    obj = build_base_mesh()
    print(f'[mesh] verts={len(obj.data.vertices)} faces={len(obj.data.polygons)}', flush=True)

    ng = build_gn()
    # env-переопределения дефолтов ДО создания модификатора: 5.3 кэширует
    # значения сокетов в момент mod.node_group = ng, позже не меняются
    for item in ng.interface.items_tree:
        if item.in_out == 'INPUT' and item.item_type == 'SOCKET':
            env = 'GN_' + item.name.upper().replace(' ', '_')
            if env in os.environ:
                raw = os.environ[env]
                try:
                    if 'Int' in item.socket_type:
                        item.default_value = int(float(raw))
                    elif 'Float' in item.socket_type:
                        item.default_value = float(raw)
                except Exception as e:
                    print(f'[env] FAIL {env}={raw}: {e}', flush=True)
                else:
                    print(f'[env] {env}={raw}', flush=True)

    mod = obj.modifiers.new('CoralGN', 'NODES')
    mod.node_group = ng

    # voxel-remesh — OPT-IN (ломается на тонких стенках после float-boolean):
    # CORAL_REMESH=1 CORAL_VOXEL_SIZE=0.009
    if os.environ.get('CORAL_REMESH', '0') == '1':
        rm = obj.modifiers.new('Remesh', 'REMESH')
        rm.mode = 'VOXEL'
        rm.voxel_size = float(os.environ.get('CORAL_VOXEL_SIZE', '0.009'))
        try:
            rm.smooth_shading = True
        except Exception:
            pass
    ng2 = build_post()
    if 'GN_SMOOTH' in os.environ:   # blur живёт в пост-группе
        try:
            for it2 in ng2.interface.items_tree:
                if it2.name == 'Smooth':
                    it2.default_value = int(float(os.environ['GN_SMOOTH']))
            print(f"[env] GN_SMOOTH={os.environ['GN_SMOOTH']} (post)", flush=True)
        except Exception as e:
            print(f'[env] FAIL GN_SMOOTH: {e}', flush=True)
    mod2 = obj.modifiers.new('CoralPost', 'NODES')
    mod2.node_group = ng2
    for item in ng.interface.items_tree:          # Object-сокеты — только так
        if item.in_out == 'INPUT' and item.item_type == 'SOCKET' \
                and 'Object' in item.socket_type:
            env = 'GN_' + item.name.upper().replace(' ', '_')
            if env in os.environ:
                try:
                    mod[item.identifier] = bpy.data.objects.get(os.environ[env])
                except Exception as e:
                    print(f'[env] FAIL {env}: {e}', flush=True)

    vals = []
    for it in ng.interface.items_tree:
        if it.in_out == 'INPUT' and it.item_type == 'SOCKET' and it.name != 'Geometry':
            try:
                vals.append(f"{it.name}={it.default_value}")
            except Exception:
                vals.append(f"{it.name}=<obj>")
    print('[sock] ' + ', '.join(vals), flush=True)

    build_scene()

    if DIAG:
        def bbox(o):
            xs = [v[0] for v in o.bound_box]; ys = [v[1] for v in o.bound_box]
            zs = [v[2] for v in o.bound_box]
            return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))
        print('[diag] base bbox=' + str([[round(a, 3) for a in t] for t in bbox(obj)]), flush=True)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        print('[diag] mod  bbox=' + str([[round(a, 3) for a in t] for t in bbox(ev)]), flush=True)
        me = ev.to_mesh()
        print(f'[diag] evaluated verts={len(me.vertices)}', flush=True)
        ev.to_mesh_clear()

    if BLEND:
        bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)
        print(f'[blend] saved -> {BLEND}', flush=True)
    if OUT:
        bpy.context.scene.render.filepath = OUT
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f'[render] {OUT} in {time.time()-t:.1f}s', flush=True)
    print(f'[coral] done in {time.time()-T0:.1f}s', flush=True)

main()
