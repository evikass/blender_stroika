#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coral_gn.py — v11 «blender_stroika»
Процедурный коралл по референсам Qwen21_00002.png + Qwen21_00015/00020.png:
  низ — пухлое основание с НАСТОЯЩИМИ губчатыми дырками (Boolean Difference
        со сферами: сквозные поры с чистой круглой кромкой, как у губки),
  середина — гладкая скрученная талия,
  верх — два веера с радиальными складками и мелкими ровными дырочками.

Толщина стенки (= глубина пор) заложена в базовую сетку (сдвиг профиля),
env CORAL_WALL (по умолчанию 0.024). Сферы-резаки НЕ перекрываются
(spacing ≥ 2·r_max), иначе FLOAT-солвер boolean даёт кольца-шрамы.

Архитектура: Python-базовая сетка (тело вращения) + модификатор CoralGN (Geometry Nodes).
Конвенция хелперов: скалярные/векторные хелперы возвращают (node, socket);
аргументы принимают float | (node, socket) | node (голая нода = output 0).
После сборки — трассировка достижимости от Group Output.

env:
  CORAL_LOW=1      черновой рендер (меньше сэмплов/размер)
  CORAL_OUT=path   куда писать рендер PNG
  CORAL_BLEND=path куда сохранить .blend
  CORAL_DIAG=1     диагностика: bbox до/после модификатора, счётчики
  GN_<ИМЯ>=знач.   переопределение слайдеров (имя сокета: верхний регистр, пробелы -> _)
"""

import bpy, bmesh, math, os, time
from math import cos, sin, pi
from mathutils import Vector

T0 = time.time()
LOW   = os.environ.get('CORAL_LOW') == '1'
DIAG  = os.environ.get('CORAL_DIAG') == '1'
OUT   = os.environ.get('CORAL_OUT')
BLEND = os.environ.get('CORAL_BLEND')

H = 1.50          # высота тела
RINGS = 248       # колец тела вращения
SEGS = 256        # сегментов по кругу

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

    # ---- разбор аргумента: float | (node, sock) | node ----
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
            # в 5.3 у VectorMath 4 входа: [0] Vector, [1] Vector, [2] Vector, [3] Scale(float)
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
        return nd  # используем как (nd, 'X'|'Y'|'Z')

    def comb(self, x, y, z):
        nd = self.n('ShaderNodeCombineXYZ')
        for i, v in ((0, x), (1, y), (2, z)):
            s = self._src(v)
            if s is None:
                nd.inputs[i].default_value = float(v)
            else:
                self.link(s[0], s[1], nd, i)
        return (nd, 0)

    def rep(self, s):
        """Скаляр -> вектор (s,s,s) (вход Scale у VectorMath в 5.3 — векторный)."""
        return self.comb(s, s, s)

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


# ---------------------------------------------------------------- профиль
PROFILE = [
    (0.001, 0.000), (0.300, 0.020), (0.520, 0.100), (0.620, 0.300),
    (0.600, 0.480), (0.470, 0.660), (0.300, 0.820), (0.185, 0.970),
    (0.175, 1.060), (0.220, 1.160), (0.300, 1.260), (0.385, 1.360),
    (0.435, 1.440), (0.443, 1.498), (0.420, 1.545),
    (0.396, 1.538), (0.385, 1.490),
]

def catmull(p0, p1, p2, p3, t):
    t2, t3 = t * t, t * t * t
    return tuple(0.5 * ((2 * p1[i]) + (-p0[i] + p2[i]) * t +
                        (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * t2 +
                        (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * t3)
                 for i in range(2))

def sample_profile(rings):
    pts = PROFILE
    out = []
    n_span = len(pts) - 1
    per = max(2, rings // n_span)
    for s in range(n_span):
        p0 = pts[max(s - 1, 0)]; p1 = pts[s]; p2 = pts[s + 1]
        p3 = pts[min(s + 2, n_span)]
        for k in range(per):
            out.append(catmull(p0, p1, p2, p3, k / per))
    out.append(pts[-1])
    return out


def build_base_mesh(name='CoralBase'):
    """ЗАМКНУТЫЙ СОЛИД: внешняя поверхность + внутренняя (сдвиг профиля на
    CORAL_WALL по 2D-нормали) + швы снизу/сверху. Стенка = глубина губчатых пор.
    Boolean сфер против солида — стандартный случай (на открытой оболочке
    exact-boolean в 5.3 съедает всю память — проверено, OOM-kill)."""
    wall = float(os.environ.get('CORAL_WALL', '0.024'))
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    prof = sample_profile(RINGS)
    n = len(prof)

    def n2d(i):
        a = prof[max(i - 1, 0)]; b = prof[min(i + 1, n - 1)]
        tr, tz = b[0] - a[0], b[1] - a[1]
        l = math.hypot(tr, tz) or 1.0
        return (tz / l, -tr / l)

    prof_in = []
    for i, (r, z) in enumerate(prof):
        nx, nz = n2d(i)
        prof_in.append((max(0.012, r - wall * nx),
                        max(0.004, z - wall * nz)))
    grid = []
    for (r, z), (ri, zi) in zip(prof, prof_in):
        ro = []; rn = []
        for j in range(SEGS):
            a = 2 * pi * j / SEGS
            ca, sa = cos(a), sin(a)
            ro.append(bm.verts.new((r * ca, r * sa, z)))
            rn.append(bm.verts.new((ri * ca, ri * sa, zi)))
        grid.append((ro, rn))
    bm.verts.ensure_lookup_table()
    for i in range(n - 1):          # внешняя стенка
        for j in range(SEGS):
            j2 = (j + 1) % SEGS
            bm.faces.new((grid[i][0][j], grid[i][0][j2],
                          grid[i + 1][0][j2], grid[i + 1][0][j]))
    for i in range(n - 1):          # внутренняя стенка
        for j in range(SEGS):
            j2 = (j + 1) % SEGS
            bm.faces.new((grid[i][1][j], grid[i + 1][1][j],
                          grid[i + 1][1][j2], grid[i][1][j2]))
    for j in range(SEGS):           # швы низ/верх
        j2 = (j + 1) % SEGS
        bm.faces.new((grid[0][0][j], grid[0][1][j],
                      grid[0][1][j2], grid[0][0][j2]))
        bm.faces.new((grid[n - 1][0][j], grid[n - 1][0][j2],
                      grid[n - 1][1][j2], grid[n - 1][1][j]))
    # дно: внешняя крышка (полюс) + внутренняя крышка полости
    c = bm.verts.new((0.0, 0.0, 0.0))
    ci = bm.verts.new((0.0, 0.0, grid[0][1][0].co.z))
    for j in range(SEGS):
        j2 = (j + 1) % SEGS
        bm.faces.new((c, grid[0][0][j2], grid[0][0][j]))
        bm.faces.new((ci, grid[0][1][j], grid[0][1][j2]))
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
    ('Twist',           'NodeSocketFloat', 0.70, -2.0, 2.0),
    ('Wobble',          'NodeSocketFloat', 0.012, 0.0, 0.08),
    ('Fan Spread',      'NodeSocketFloat', 0.62, 0.0, 1.5),
    ('Fan Curl',        'NodeSocketFloat', 0.30, 0.0, 1.2),
    ('Gap Rise',        'NodeSocketFloat', 0.25, 0.0, 0.6),
    ('Lobe Sharp',      'NodeSocketFloat', 4.2, 0.5, 6.0),
    ('Pleat Count',     'NodeSocketFloat', 36.0, 0.0, 120.0),
    ('Pleat Depth',     'NodeSocketFloat', 0.040, 0.0, 0.1),
    ('Pores Enable',    'NodeSocketFloat', 1.0, 0.0, 1.0),
    ('Pore Spacing',    'NodeSocketFloat', 0.088, 0.04, 0.3),
    ('Pore Radius',     'NodeSocketFloat', 0.030, 0.005, 0.09),
    ('Pore Var',        'NodeSocketFloat', 0.75, 0.0, 1.0),
    ('Fine Spacing',    'NodeSocketFloat', 0.066, 0.02, 0.2),
    ('Fine Radius',     'NodeSocketFloat', 0.024, 0.005, 0.05),
    ('Smooth',          'NodeSocketInt',   4, 0, 24),
    ('Cutter Scale',    'NodeSocketFloat', 1.0, 0.1, 5.0),
    ('Cutter Object',   'NodeSocketObject', None, None, None),
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

    # ---- вход: базовая сетка + Extra Object (деформируется вместе с кораллом) ----
    cur = S('Geometry')
    oi_e = g.n('GeometryNodeObjectInfo', transform_space='ORIGINAL')
    g.link(S('Extra Object'), oi_e, 'Object')
    oi_e.inputs['As Instance'].default_value = False
    jn = g.n('GeometryNodeJoinGeometry')
    g.link(cur[0], cur[1], jn, 'Geometry')
    g.link(oi_e, 'Geometry', jn, 'Geometry')
    cur = (jn, 0)

    # ---- TWIST: поворот вокруг Z, угол растёт от талии (скрутка) ----
    p0 = pos_in(g); se0 = g.sep(p0)
    x, y, z = (se0, 'X'), (se0, 'Y'), (se0, 'Z')
    th = g.m('ARCTAN2', y, x)
    zn = g.m('DIVIDE', z, H)
    a = g.m('MULTIPLY', g.m('SUBTRACT', zn, 0.5), S('Twist'))
    th2 = g.m('ADD', th, a)
    # ln = sqrt(x*x + y*y) — чистая float-математика (LENGTH в 5.3 молча даёт 0.5)
    xx = g.m('MULTIPLY', x, x)
    yy = g.m('MULTIPLY', y, y)
    ln = g.m('POWER', g.m('ADD', xx, yy), 0.5)
    xt = g.m('MULTIPLY', g.m('COSINE', th2), ln)
    yt = g.m('MULTIPLY', g.m('SINE', th2), ln)
    cur = set_pos(g, cur, g.comb(xt, yt, z))
    stages['twist'] = cur

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

    # ---- PLEATS: радиальные складки на веерах ----
    p2 = pos_in(g); se2 = g.sep(p2)
    x2, y2, z2 = (se2, 'X'), (se2, 'Y'), (se2, 'Z')
    th2_ = g.m('ARCTAN2', y2, x2)
    rid = g.m('SINE', g.m('MULTIPLY', th2_, S('Pleat Count')))
    amp = g.m('MULTIPLY', rid, S('Pleat Depth'))
    up1 = g.smooth(z2, 1.02, 1.16)
    up2 = g.m('SUBTRACT', 1.0, g.smooth(z2, 1.45, 1.525))
    pmask = g.m('MULTIPLY', up1, up2)
    amp = g.m('MULTIPLY', amp, pmask)
    cur = set_pos(g, cur, g.vec('ADD', p2, g.vec('SCALE', nrm_in(g), scale=amp)))
    stages['pleats'] = cur

    # ---- LOBES: два веера в стороны (+X/-X) с загибом вниз ----
    p3 = pos_in(g); se3 = g.sep(p3)
    x3, y3, z3 = (se3, 'X'), (se3, 'Y'), (se3, 'Z')
    th3 = g.m('ARCTAN2', y3, x3)
    c2 = g.m('COSINE', g.m('MULTIPLY', th3, 2.0))
    lobe = g.m('POWER', g.m('ADD', g.m('MULTIPLY', c2, 0.5), 0.5), S('Lobe Sharp'))
    zr = g.smooth(z3, 0.98, 1.10)
    rimr = g.smooth(z3, 1.00, 1.44)
    push = g.m('MULTIPLY', lobe, S('Fan Spread'))
    push = g.m('MULTIPLY', push, zr)
    push = g.m('MULTIPLY', push, rimr)
    # рваная кромка: рифление модулирует разлёт крыла
    rip = g.m('SINE', g.m('MULTIPLY', th3, S('Pleat Count')))
    push = g.m('MULTIPLY', push, g.m('ADD', 1.0, g.m('MULTIPLY', rip, 0.28)))
    pushv = g.vec('SCALE', g.vec('NORMALIZE', g.comb(x3, y3, 0.0)), scale=push)
    psp = g.sep(pushv)
    curl = g.m('MULTIPLY', lobe, S('Fan Curl'))
    curl = g.m('MULTIPLY', curl, zr)
    curl = g.m('MULTIPLY', curl, rimr)
    # между веерами зона приподнимается -> глубокий V-вырез
    rise = g.m('MULTIPLY', g.m('SUBTRACT', 1.0, lobe), S('Gap Rise'))
    rise = g.m('MULTIPLY', rise, zr)
    znew = g.m('SUBTRACT', z3, curl)
    znew = g.m('ADD', znew, rise)
    cur = set_pos(g, cur, g.comb(g.m('ADD', x3, (psp, 'X')),
                                 g.m('ADD', y3, (psp, 'Y')), znew))
    stages['lobes'] = cur

    # ---- PORES: настоящие губчатые дырки — Boolean Difference со сферами ----
    p4 = pos_in(g); se4 = g.sep(p4)
    x4, y4, z4 = (se4, 'X'), (se4, 'Y'), (se4, 'Z')
    ln4 = g.m('POWER', g.m('ADD', g.m('MULTIPLY', x4, x4),
                           g.m('MULTIPLY', y4, y4)), 0.5)
    # зоны по финальным позициям: низ — крупные поры, веера — мелкие дырочки;
    # фильтр по нормали — резаем только НАРУЖНУЮ поверхность солида
    sen = g.sep(nrm_in(g))
    nxn, nyn, nzn = (sen, 'X'), (sen, 'Y'), (sen, 'Z')
    dotp = g.m('ADD', g.m('ADD', g.m('MULTIPLY', nxn, x4),
                          g.m('MULTIPLY', nyn, y4)),
               g.m('MULTIPLY', nzn, z4))
    plen = g.m('POWER', g.m('ADD', g.m('ADD', g.m('MULTIPLY', x4, x4),
                                       g.m('MULTIPLY', y4, y4)),
                             g.m('MULTIPLY', z4, z4)), 0.5)
    cosb = g.m('DIVIDE', dotp, plen)
    mask_b = g.m('MULTIPLY', g.smooth(z4, 0.02, 0.10),
                 g.m('SUBTRACT', 1.0, g.smooth(z4, 0.78, 0.98)))
    mask_b = g.m('MULTIPLY', mask_b, g.smooth(cosb, 0.0, 0.2))
    mask_t = g.m('MULTIPLY', g.smooth(z4, 1.08, 1.22),
                 g.m('SUBTRACT', 1.0, g.smooth(ln4, 1.20, 1.30)))
    mask_t = g.m('MULTIPLY', mask_t, g.smooth(nzn, 0.25, 0.45))
    en = S('Pores Enable')
    mask_b = g.m('MULTIPLY', mask_b, en)
    mask_t = g.m('MULTIPLY', mask_t, en)

    def pore_spheres(mask_, dens, dist_min, rad, var, seed):
        """DPF(Poisson) -> пер-точечный радиус -> сферы на поверхности -> realize.
        Центр сфер лежит НА поверхности (Capture/SetPosition в 5.3 лишены
        data_type — убраны ради надёжности; кромка и так круглая)."""
        dpf = g.n('GeometryNodeDistributePointsOnFaces',
                  distribute_method='POISSON')
        g.link(cur[0], cur[1], dpf, 'Mesh')
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
        ico.inputs['Subdivisions'].default_value = 2
        inst = g.n('GeometryNodeInstanceOnPoints')
        g.link(dpf, 'Points', inst, 'Points')
        g.link(ico, 'Mesh', inst, 'Instance')
        g.link(g.comb(ri, ri, ri), inst, 'Scale')
        rlz = g.n('GeometryNodeRealizeInstances')
        rlz.inputs['Realize All'].default_value = True
        g.link(inst, 'Instances', rlz, 'Geometry')
        return (rlz, 0)

    s1 = pore_spheres(mask_b, 260.0, S('Pore Spacing'), S('Pore Radius'),
                      S('Pore Var'), 3)
    s2 = pore_spheres(mask_t, 520.0, S('Fine Spacing'), S('Fine Radius'),
                      S('Pore Var'), 5)
    jn2 = g.n('GeometryNodeJoinGeometry')
    g.link(s1[0], s1[1], jn2, 'Geometry')
    g.link(s2[0], s2[1], jn2, 'Geometry')
    boo = g.n('GeometryNodeMeshBoolean', operation='DIFFERENCE')
    try:
        boo.solver = os.environ.get('CORAL_SOLVER', 'EXACT')
    except Exception as e:
        print(f'[pores] solver set failed: {e}', flush=True)
    for opt in ('Self Intersection', 'Hole Tolerant'):
        if opt in boo.inputs:          # у FLOAT-солвера этих сокетов нет
            boo.inputs[opt].default_value = True
    g.link(cur[0], cur[1], boo, 'Mesh 1')
    g.link(jn2, 0, boo, 'Mesh 2')
    cur = (boo, 0)
    stages['pores'] = cur

    # ---- SMOOTH pass: Blur Attribute по позиции ----
    blr = g.n('GeometryNodeBlurAttribute')
    blr.data_type = 'FLOAT_VECTOR'
    g.link(pos_in(g), blr, 'Value')
    blr.inputs['Weight'].default_value = 1.0
    g.link(S('Smooth'), blr, 'Iterations')
    cur = set_pos(g, cur, blr)
    stages['blur'] = cur

    # ---- Shade Smooth ----
    sss = g.n('GeometryNodeSetShadeSmooth')
    sss.domain = 'FACE'
    g.link(cur[0], cur[1], sss, 'Geometry')
    sss.inputs['Shade Smooth'].default_value = True
    cur = (sss, 0)
    stages['shade'] = cur

    # ---- BOOLEAN: слот Cutter Object ----
    oi_c = g.n('GeometryNodeObjectInfo', transform_space='ORIGINAL')
    g.link(S('Cutter Object'), oi_c, 'Object')
    oi_c.inputs['As Instance'].default_value = False
    cs = S('Cutter Scale')
    tr = g.n('GeometryNodeTransform')
    g.link(oi_c, 'Geometry', tr, 'Geometry')
    g.link(g.comb(cs, cs, cs), tr, 'Scale')
    boo = g.n('GeometryNodeMeshBoolean', operation='DIFFERENCE')
    g.link(cur[0], cur[1], boo, 0)
    g.link(tr, 'Geometry', boo, 1)
    cur = (boo, 0)
    stages['bool'] = cur

    final = stages.get(os.environ.get('CORAL_STAGE', ''), cur)
    g.link(final[0], final[1], go, 'Geometry')

    # ---- верификация достижимости от Group Output (обход по ВХОДАМ выхода) ----
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
    # проверка типов: vector->float и прочие неявные конверсии, которые 5.3 молча ломает
    mism = []
    for lnk in ng.links:
        ft, tt = lnk.from_socket.type, lnk.to_socket.type
        if ft != tt and not (ft in ('FLOAT', 'INT', 'BOOLEAN', 'VALUE') and
                             tt in ('FLOAT', 'INT', 'BOOLEAN', 'VALUE')):
            mism.append(f'{lnk.from_node.name}:{lnk.from_socket.name}'
                        f'({ft})->{lnk.to_node.name}:{lnk.to_socket.name}({tt})')
    if mism:
        print(f'[verify] TYPE-MISMATCH: {mism}', flush=True)
    if os.environ.get('CORAL_DEBUG_LINKS'):
        for nd in ng.nodes:
            if nd.type not in ('MATH', 'VECT_MATH', 'SEPRGB', 'COMBXYZ'):
                continue
            for i, inp in enumerate(nd.inputs):
                if inp.is_linked:
                    src = inp.links[0].from_node.name + ':' + inp.links[0].from_socket.name
                else:
                    try:
                        src = f'def={inp.default_value}'
                    except Exception:
                        src = 'def=?'
                tag = ' <<<UNLINKED' if (i == 1 and nd.type == 'MATH'
                                         and not inp.is_linked) else ''
                print(f'[lnk] {nd.name:>18} [{i}] {inp.name:<10} {src}{tag}', flush=True)
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

    w = bpy.context.scene.world or bpy.data.worlds.new('World')
    bpy.context.scene.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.87, 0.88, 0.90, 1.0)
    bg.inputs['Strength'].default_value = 0.45

    def light(name, loc, size, energy):
        ld = bpy.data.lights.new(name, 'AREA')
        ld.size = size; ld.energy = energy
        lo = bpy.data.objects.new(name, ld)
        lo.location = loc
        bpy.context.collection.objects.link(lo)
        d = Vector((0, 0, 0.7)) - lo.location
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return lo
    light('Key', (3.0, -2.6, 2.6), 3.5, 120)
    light('Fill', (-3.2, -1.6, 1.2), 4.0, 45)
    light('Rim', (0.0, 2.8, 2.2), 3.0, 70)

    cd = bpy.data.cameras.new('Cam')
    cd.lens = 50
    cam = bpy.data.objects.new('Cam', cd)
    bpy.context.collection.objects.link(cam)
    cam.location = (2.2, -3.6, 2.35)
    d = Vector((0, 0, 0.72)) - cam.location
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
    print(f'[coral] v11 start LOW={LOW} DIAG={DIAG}', flush=True)
    # чистим factory-startup сцену (куб/камера/свет)
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    obj = build_base_mesh()
    print(f'[mesh] verts={len(obj.data.vertices)}', flush=True)

    ng = build_gn()
    mod = obj.modifiers.new('CoralGN', 'NODES')
    mod.node_group = ng

    # env-переопределения слайдеров
    for item in ng.interface.items_tree:
        if item.in_out == 'INPUT' and item.item_type == 'SOCKET':
            env = 'GN_' + item.name.upper().replace(' ', '_')
            if env in os.environ:
                raw = os.environ[env]
                try:
                    if 'Int' in item.socket_type:
                        mod[item.identifier] = int(float(raw))
                    elif 'Float' in item.socket_type:
                        mod[item.identifier] = float(raw)
                    elif 'Object' in item.socket_type:
                        mod[item.identifier] = bpy.data.objects.get(raw)
                except Exception as e:
                    print(f'[env] FAIL {env}={raw}: {e}', flush=True)
                else:
                    print(f'[env] {env}={raw}', flush=True)

    vals = []
    for it in ng.interface.items_tree:
        if it.in_out == 'INPUT' and it.item_type == 'SOCKET' and it.name != 'Geometry':
            try:
                vals.append(f"{it.name}={mod[it.identifier]}")
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
