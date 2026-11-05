#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coral_gn.py — v12 «blender_stroika»
Процедурный коралл по референсам Qwen21_00015/00020.png:
  форма — закрученное кольцо-«мёбиус»: труба переменного сечения, свитая
          по окружности, сечение доворачивается на 1.5 полуоборота
          (CORAL_TWIST), рёбра-складки на внутренней стороне спиралью;
  дырки — крупные и плотные губчатые поры (Boolean Difference со сферами),
          внешняя сторона трубы; внутренняя (к зёву) — гладкая с рёбрами.

База строится Python-sweep'ом как ЗАМКНУТЫЙ СОЛИД (внешняя + внутренняя
поверхности, стенка CORAL_WALL=0.028 = глубина пор). Сферы-резаки НЕ
перекрываются (spacing ≥ 2·r_max), иначе FLOAT-солвер даёт кольца-шрамы.

env sweep-параметров: CORAL_TWIST (полуобороты, 1.5), CORAL_RING_R (0.43),
CORAL_PLEAT_COUNT (36), CORAL_PLEAT_DEPTH (0.022), CORAL_WALL (0.028).

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

# --- форма задаётся sweep'ом в build_base_mesh; HERE больше нет тела вращения ---

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


# ---------------------------------------------------------------- база-sweep
def build_base_mesh(name='CoralBase'):
    """Кольцо-«мёбиус»: труба переменного сечения вдоль окружности радиуса
    CORAL_RING_R, сечение доворачивается на CORAL_TWIST полуоборотов (лестница
    целых сдвигов — шов замыкается без дыр), рёбра — синус по сечению.
    ЗАМКНУТЫЙ СОЛИД: внешняя + внутренняя (минус CORAL_WALL) поверхности.
    (Вращение сечения — ориентированный солид; в 5.3 boolean сфер против
    открытой оболочки падает: EXACT — segfault, FLOAT — OOM.)"""
    wall  = float(os.environ.get('CORAL_WALL', '0.028'))
    twist = float(os.environ.get('CORAL_TWIST', '1.5'))
    pl_n  = max(0, int(float(os.environ.get('CORAL_PLEAT_COUNT', '36')))) // 2 * 2
    pl_d  = float(os.environ.get('CORAL_PLEAT_DEPTH', '0.024'))
    R     = float(os.environ.get('CORAL_RING_R', '0.43'))
    NR, NS = 380, 176
    TOT   = int(round(twist * NS / 2.0))       # полный сдвиг сечения в столбцах
    print(f'[sweep] R={R} twist={twist} TOT={TOT} wall={wall} '
          f'pleats={pl_n}x{pl_d} NR={NR} NS={NS}', flush=True)
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    def tube_r(u):
        # гармонические биения толщины трубы (замкнуты по u, зёв круглее)
        return (0.175 + 0.032 * math.cos(2.0 * u)
                + 0.018 * math.sin(3.0 * u + 1.0)
                - 0.012 * math.cos(5.0 * u + 2.1))

    def sstep(x, e0, e1):
        t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
        return t * t * (3.0 - 2.0 * t)

    def make_rows(shrink):
        rows = []
        for i in range(NR):
            u = 2.0 * pi * i / NR
            rt = tube_r(u) - shrink
            sh = round(TOT * i / NR)           # целочисленная лестница сдвигов
            erx, erz = math.sin(u), math.cos(u)
            px, pz = R * erx, R * erz
            bf = 1.12 - 0.08 * math.cos(2.0 * u + 0.7)   # вариативный эллипс
            row = []
            for j in range(NS):
                psi = 2.0 * pi * (j + sh) / NS
                a = rt * math.cos(psi)
                b = rt * math.sin(psi) * bf
                if pl_d > 0 and shrink == 0.0:
                    # рёбра только на внутренней (к зёву) стороне — как на референсе;
                    # зёв = psi≈π (направление −e_r), маска непрерывна на шве
                    a += pl_d * math.sin(psi * pl_n + 2.0 * u) * \
                        sstep(-math.cos(psi), 0.15, 0.55)
                row.append(bm.verts.new((px + a * erx, b, pz + a * erz)))
            rows.append(row)
        return rows

    outer = make_rows(0.0)
    inner = make_rows(wall)

    def stitch(rA, rB, off):
        """Квады между соседними рядами; сдвиг уже зашит в координаты вершин,
        поэтому off нужен только на шве (последний ряд -> первый)."""
        for j in range(NS):
            j2 = (j + 1) % NS
            k  = (j + off) % NS
            k2 = (j2 + off) % NS
            bm.faces.new((rA[j], rA[j2], rB[k2], rB[k]))

    for i in range(NR - 1):
        stitch(outer[i], outer[i + 1], 0)
    stitch(outer[NR - 1], outer[0], TOT % NS)      # шов внешней
    for i in range(NR - 1):
        stitch(inner[i], inner[i + 1], 0)
    stitch(inner[NR - 1], inner[0], TOT % NS)      # шов внутренней
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
    ('Wobble',          'NodeSocketFloat', 0.012, 0.0, 0.08),
    ('Pores Enable',    'NodeSocketFloat', 1.0, 0.0, 1.0),
    ('Pore Spacing',    'NodeSocketFloat', 0.112, 0.04, 0.3),
    ('Pore Radius',     'NodeSocketFloat', 0.040, 0.005, 0.09),
    ('Pore Var',        'NodeSocketFloat', 0.55, 0.0, 1.0),
    ('Fine Spacing',    'NodeSocketFloat', 0.074, 0.02, 0.2),
    ('Fine Radius',     'NodeSocketFloat', 0.028, 0.005, 0.05),
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

    # ---- PORES: губчатые дырки на ВНЕШНЕЙ стороне кольца ----
    # кольцо лежит в XZ (ось — Y): наружная сторона трубы = нормаль сонаправлена
    # радиалу (x,0,z); фильтр |xz| > R отсекает стенку внутренней полости
    p4 = pos_in(g); se4 = g.sep(p4)
    x4, y4, z4 = (se4, 'X'), (se4, 'Y'), (se4, 'Z')
    sen = g.sep(nrm_in(g))
    nxn, nzn = (sen, 'X'), (sen, 'Z')
    lxz = g.m('POWER', g.m('ADD', g.m('MULTIPLY', x4, x4),
                           g.m('MULTIPLY', z4, z4)), 0.5)
    dotp = g.m('ADD', g.m('MULTIPLY', nxn, x4), g.m('MULTIPLY', nzn, z4))
    cosb = g.m('DIVIDE', dotp, lxz)
    RING = float(os.environ.get('CORAL_RING_R', '0.43'))
    out_ok = g.smooth(lxz, RING + 0.005, RING + 0.045)
    en = S('Pores Enable')
    # крупные поры: внешняя половина трубы
    mask_b = g.m('MULTIPLY', g.smooth(cosb, 0.28, 0.38), out_ok)
    mask_b = g.m('MULTIPLY', mask_b, en)
    # мелкие: широкий пояс от зёва до крупной зоны (зоны НЕ перекрываются)
    mask_f = g.m('MULTIPLY', g.smooth(cosb, -0.25, -0.12),
                 g.m('SUBTRACT', 1.0, g.smooth(cosb, 0.14, 0.22)))
    mask_f = g.m('MULTIPLY', mask_f, out_ok)
    mask_f = g.m('MULTIPLY', mask_f, en)

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

    s1 = pore_spheres(mask_b, 240.0, S('Pore Spacing'), S('Pore Radius'),
                      S('Pore Var'), 3)
    s2 = pore_spheres(mask_f, 420.0, S('Fine Spacing'), S('Fine Radius'),
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
        d = Vector((0, 0, 0.0)) - lo.location
        lo.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        return lo
    light('Key', (3.0, -2.6, 2.6), 3.5, 120)
    light('Fill', (-3.2, -1.6, 1.2), 4.0, 45)
    light('Rim', (0.0, 2.8, 2.2), 3.0, 70)

    cd = bpy.data.cameras.new('Cam')
    cd.lens = 50
    cam = bpy.data.objects.new('Cam', cd)
    bpy.context.collection.objects.link(cam)
    cam.location = (2.5, -3.4, 1.0)
    d = Vector((0, 0, 0.0)) - cam.location
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
    print(f'[coral] v12 start LOW={LOW} DIAG={DIAG}', flush=True)
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
