#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coral_gn.py — v18 «blender_stroika» (глубже/темнее внутри + плотная мелкая сетка)
Процедурный коралл по референсу 0682af35916265e314b019420a043024.jpg:
  форма — ВЕРТИКАЛЬНЫЙ скульптурный рост: ствол с луковичными утолщениями
          и перетяжками, S-изгиб, спиральный перекрут рёбер-складок,
          4 ответвления, отходящие вверх-наружу; всё сглажено (blur);
  дырки — крупные глубокие БУРЫ (Boolean Difference, эллипсоиды x2.2)
          + плотная сетка мелких пор (второй boolean-проход), как на
          керамическом референсе.

v16: тело МОНОЛИТНОЕ (только внешняя поверхность, БЕЗ внутренней полости —
иначе сквозь дырку видно «две стенки и пустоту между ними»). Каждый эллипсоид-
резак вгрызается в материал на depth·r: взгляд в дырку видит ОДНУ гладкую
стенку, плавно уходящую вглубь. Сферы-резаки внутри прохода не перекрываются
(spacing ≥ 2·r_max), иначе FLOAT-солвер даёт кольца-шрамы.

v17: поры не пересекаются -> кромки ровные. Источники рвани и фиксы:
  1) мелкая губа (boolean-проход 2) сеялась по ВСЕЙ поверхности, включая
     кромки крупных буров -> кромки изъедены. Фикс: центры буров (points
     DPF) передаются во 2-й проход, мелкие поры СНОСЯТСЯ Proximity-отсечкой
     в радиусе r_бура·pr + r_мелкой + margin. Кромка бура режется последней
     по чистой зоне — ровная окружность.
  2) мелкие boolean-поры сами по себе «звёзды»-рвань (FLOAT-сливеры на
     r=0.024). Фикс: мелкие поры — НЕ boolean, а ДИМПЛЫ: мягкое вдавливание
     вершин внутрь по Proximity к центрам пор (fall = 1 − sstep(d, 0.55r, r),
     глубина 0.6·r). Ни одного резака — ни щелей, ни звёзд. Буры остаются
     boolean (глубокие лунки с одной гладкой стенкой).
  3) буры на тонких местах (вершина ствола R≈0.09, концы ветвей R≈0.03)
     шире самой детали -> резаки сходятся внутри. Фикс: в Python на каждую
     вершину пишем атрибут pr = clamp(R/0.16, 0.45, 1): радиус бура
     масштабируется локальной толщиной (глубина 2.2·r ≤ 2R всегда — пробоя
     насквозь нет), маска mb = sstep(R, 0.05, 0.07) запрещает поры на
     кончиках (они остаются гладкими).
  4) буры у СТЫКОВ ствол/ветвь рвут шов двух солидов. Фикс: маска mb
     гасится sstep-ом возле оси соседней детали (d < 1.1·R_соседа + 0.14).
     Атрибуты после boolean теряются — поэтому мелкие поры считаются на
     чистой поверхности ДО boolean (порядок: центры буров -> центры мелких
     -> димплы -> boolean буров).

v18: «глубже/темнее внутри (как на керамике) + плотнее сетка мелких пор»:
  1) ТЕМНОТА — в материал CoralWhite добавлен шейдерный Ambient Occlusion
     (only_local, дистанция 0.22): Base Color = mix(тёмный тёплый, белый,
     1 − AO^1.5). Стенки буров и димплов уходят в тень тем глубже, чем
     глубже полость — взгляд читает ГЛУБИНУ, как на керамическом референсе.
  2) ГЛУБИНА — буры 3.0→3.8·r, средние 1.6→2.0·r, димплы 1.05→1.35·r
     (falloff уже: sstep 0.28r..0.80r — ямки круглее и глубже); guard'ы
     pr/mb/mf прежние — пробоя тонких мест нет (≤0.94·R).
  3) ПЛОТНЕЕ мелкая сетка — Fine Spacing 0.072→0.054, Fine Radius
     0.034→0.028 (пор ×1.8 по площади), Density Max 380→700 (кап DPF выше
     нужного), меш плотнее: ствол 176×88, ветви 92×70 (~3 вершины поперёк
     ямки — димплы не алиасятся).

env: CORAL_SEED (7), CORAL_H (1.34), CORAL_SOLVER (FLOAT),
     CORAL_LOW=1 (черновой рендер), CORAL_OUT=path, CORAL_BLEND=path,
     CORAL_DIAG=1, CORAL_CLOSE=1 (камера вблизи для контроля дырок),
     GN_<ИМЯ>=знач (переопределение слайдеров).
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
        """smoothstep(x, e0, e1); e0/e1 — скаляры или поля (node, sock).
        Края — константы. Возвращает (node, 0)."""
        e0t, e1t = self._src(e0), self._src(e1)
        if e0t is None and e1t is None:
            span = float(e1) - float(e0)
            t = self.m('DIVIDE', self.m('SUBTRACT', x, float(e0)), span, clamp=True)
        else:
            e0v = e0t if e0t is not None else float(e0)
            e1v = e1t if e1t is not None else float(e1)
            span = self.m('SUBTRACT', e1v, e0v)
            t = self.m('DIVIDE', self.m('SUBTRACT', x, e0v), span, clamp=True)
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
    S-изгиб, спиральный перекрут рёбер, 4 ответвления. Каждый элемент —
    МОНОЛИТНЫЙ замкнутый солид (без внутренней полости)."""
    H    = float(os.environ.get('CORAL_H', '1.34'))
    seed = int(float(os.environ.get('CORAL_SEED', '7')))
    rng  = random.Random(seed)

    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    # слои атрибутов пор — создаём ДО вершин: добавление слоёв инвалидирует
    # уже выданные BMVert-обёртки (ReferenceError «has been removed»)
    if os.environ.get('CORAL_ATTRS', '1') != '0':
        lay_pr = bm.verts.layers.float.new('pr')   # множитель радиуса бура
        lay_pf = bm.verts.layers.float.new('pf')   # множитель радиуса мелкой поры
        lay_mb = bm.verts.layers.float.new('mb')   # маска «крупный бур разрешён»
        lay_mm = bm.verts.layers.float.new('mm')   # маска «средний бур разрешён»
        lay_mf = bm.verts.layers.float.new('mf')   # маска «мелкая пора разрешена»
    else:
        lay_pr = lay_pf = lay_mb = lay_mm = lay_mf = None

    VRT_R = []   # (вершина, локальный радиус детали, часть: 0 ствол, 1+ ветвь)

    def pore_collect(v, r, part):
        VRT_R.append((v, max(r, 0.0), part))

    # ---- ствол: ось (S-изгиб) ----
    def trunk_pt(s):
        z = s * H
        ramp = sstep(s, 0.10, 0.26)   # ножка вертикальна: низ не тонет под пол
        x = ramp * (0.13 * sin(2.2 * pi * s + 0.4) + 0.04 * sin(5.3 * s))
        y = ramp * (0.09 * sin(1.7 * pi * s + 1.7) + 0.03 * cos(4.6 * s + 0.6))
        return Vector((x, y, z))

    # ---- ствол: радиус (нога -> перетяжки/луковицы -> кончик) ----
    KS = [(0.00, 0.315), (0.06, 0.352), (0.16, 0.318), (0.28, 0.252),
          (0.38, 0.222), (0.50, 0.244), (0.62, 0.230), (0.74, 0.254),
          (0.86, 0.205), (0.98, 0.162), (1.10, 0.128), (1.22, 0.104),
          (1.31, 0.092), (1.36, 0.085)]
    KS = [(z * H / 1.34, r) for (z, r) in KS]

    def trunk_r(s):
        z = s * H
        r = catmull(KS, z)
        r *= 1.0 + 0.015 * sin(4.5 * s * pi + 1.3)          # мягкая волна
        return r

    # ---- складки-лужи: k гармоник с ДРЕЙФОМ фазы по высоте (перекрут) ----
    HARM = [(3, 0.060, 2.2 * rng.uniform(0.85, 1.15), rng.uniform(0, 6.28)),
            (4, 0.028, -1.6 * rng.uniform(0.85, 1.15), rng.uniform(0, 6.28))]

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

    def tube(pts, fr, rad_fun, ns, part=0):
        """rad_fun(s, theta)->r; возвращает rows[i][j] точек."""
        rows = []
        for i, (t, nv, bv) in enumerate(fr):
            s = i / (len(fr) - 1)
            row = []
            for j in range(ns):
                th = 2.0 * pi * j / ns
                r = rad_fun(s, th)
                v = bm.verts.new(pts[i] + nv * (r * cos(th)) + bv * (r * sin(th)))
                pore_collect(v, r, part)
                row.append(v)
            rows.append(row)
        return rows

    def stitch(rows):
        for i in range(len(rows) - 1):
            for j in range(len(rows[0])):
                j2 = (j + 1) % len(rows[0])
                bm.faces.new((rows[i][j], rows[i][j2], rows[i + 1][j2], rows[i + 1][j]))

    def solid_tube(pts, fr, rad_fun, ns, flat_base=False, part=0):
        """Замкнутый МОНОЛИТНЫЙ солид: только внешняя труба + конические
        торцы. Полости нет: дырка-бур показывает одну стенку, уходящую
        вглубь (плавный переход), а не «две стенки с пустотой"."""
        ro = tube(pts, fr, lambda s, th: max(rad_fun(s, th), 0.015), ns, part)
        stitch(ro)
        nsl = len(ro[0])
        if flat_base:                             # дно на полу (не видно)
            c0 = bm.verts.new(pts[0])
            pore_collect(c0, 0.0, part)
            for j in range(nsl):
                j2 = (j + 1) % nsl
                bm.faces.new((ro[0][j2], ro[0][j], c0))
        else:
            r_e = max(rad_fun(0.0, 0.0), 0.015) * 0.6
            c0 = bm.verts.new(pts[0] + fr[0][0] * r_e)   # конус-корень внутрь
            pore_collect(c0, 0.0, part)
            for j in range(nsl):
                j2 = (j + 1) % nsl
                bm.faces.new((ro[0][j2], ro[0][j], c0))
        r_t = max(rad_fun(1.0, 0.0), 0.015) * 0.6
        c1 = bm.verts.new(pts[-1] - fr[-1][0] * r_t)     # конус-кончик внутрь
        pore_collect(c1, 0.0, part)
        for j in range(nsl):
            j2 = (j + 1) % nsl
            bm.faces.new((ro[-1][j], ro[-1][j2], c1))

    # ---- ствол ----
    NZ, NS = 176, 88
    pts = [trunk_pt(i / (NZ - 1)) for i in range(NZ)]
    fr = frames(pts)
    def trunk_rad(s, th):
        r = lump(th, s, trunk_r(s))
        return max(r, 0.015)
    solid_tube(pts, fr, trunk_rad, NS, flat_base=True)
    print(f'[trunk] rows={NZ} ring={NS}', flush=True)

    # ---- ответвления: (t0 на стволе, азимут, длина, толщина) ----
    branches = []
    base_b = [(0.30, -0.35, 0.55, 0.62),
              (0.56, 2.80, 0.60, 0.66),
              (0.80, 1.35, 0.40, 0.70),
              (0.85, 0.40, 0.26, 0.80)]
    for (t0, phi, L, rs) in base_b:
        t0 += rng.uniform(-0.03, 0.03)
        phi += rng.uniform(-0.15, 0.15)
        branches.append((t0, phi, L, rs))

    if os.environ.get('CORAL_TRUNK_ONLY'):
        branches = []
    baxes = []   # оси ветвей (точка, локальный радиус) — гашение пор у стыков
    for bi, (t0, phi, L, rs) in enumerate(branches):
        p0 = trunk_pt(t0)
        r0 = trunk_r(t0) * rs
        radial = Vector((cos(phi), sin(phi), 0.0))
        side = Vector((-sin(phi), cos(phi), 0.0))
        splay = 0.75 - 0.35 * t0                       # чем выше — тем вертикальнее
        hb = 0.72 * L
        w1, w2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)
        NBZ, NBS = 92, 70
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
            grow = 1.0 + 0.10 * sin(pi * s ** 0.85)
            taper = 1.0 - 0.74 * sstep(s, 0.50, 0.99)
            f = sstep(s, 0.05, 0.25) * (1.0 - sstep(s, 0.86, 0.995))
            v = 1.0 + 0.050 * f * cos(3 * th + ph3 + tw3 * s)
            v += 0.028 * f * cos(4 * th + ph4 - 1.4 * s)
            return max(r0 * grow * taper * v, 0.012)
        solid_tube(bpts, bfr, b_rad, NBS, part=bi + 1)
        bax = []
        for i in range(0, NBZ, 3):
            s = i / (NBZ - 1)
            rb = r0 * (1.0 + 0.10 * sin(pi * s ** 0.85)) * (1.0 - 0.74 * sstep(s, 0.50, 0.99))
            bax.append((bpts[i], rb))
        baxes.append(bax)
        print(f'[branch{bi}] t0={t0:.2f} phi={phi:.2f} L={L} r0={r0:.3f}', flush=True)

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # ---- запись атрибутов пор (слои созданы до вершин — обёртки живы):
    # радиус/маска от локальной толщины R + гашение буров у СТЫКОВ деталей
    # (boolean по шву двух солидов даёт рвань) ----
    tax = [(pts[i], trunk_r(i / (NZ - 1))) for i in range(0, NZ, 3)]
    axes = [tax] + baxes
    for (v, r, part) in (VRT_R if os.environ.get('CORAL_ATTRS', '1') != '0' else []):
        # три гашения у швов деталей (разные классы пор ломают шов по-разному):
        # b — крупные буры (мягкое), m — средние (жёсткое, источник «шара»),
        # f — мелкие димплы (среднее, источник щели шва)
        fade_b = fade_m = fade_f = 1.0
        p = v.co
        for qi, ax in enumerate(axes):
            if qi == part:
                continue
            dmin = 1e18
            rq = 0.1
            for (ap, ar) in ax:
                dx = p[0] - ap.x
                dy = p[1] - ap.y
                dz = p[2] - ap.z
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < dmin:
                    dmin = d2
                    rq = ar
            d = math.sqrt(dmin)
            for (fade, lo_k, lo_a, hi_k, hi_a, hard) in (
                    (0, 1.1, 0.03, 1.1, 0.16, False),
                    (1, 1.05, 0.10, 1.05, 0.26, True),
                    (2, 1.05, 0.05, 1.05, 0.20, False)):
                lo = rq * lo_k + lo_a
                hi = rq * hi_k + hi_a
                if d < hi:
                    val = 0.0 if (hard and d < lo) else sstep(d, lo, hi)
                    if fade == 0:
                        fade_b *= val
                    elif fade == 1:
                        fade_m *= val
                    else:
                        fade_f *= val
        v[lay_pr] = min(1.0, max(0.35, r / 0.16))
        v[lay_pf] = min(1.0, max(0.60, r / 0.09))
        v[lay_mb] = sstep(r, 0.05, 0.07) * fade_b
        v[lay_mm] = sstep(r, 0.05, 0.07) * fade_m
        # fade и на мелкие поры: ямка у шва вдавливает вершины СКВОЗЬ стенку
        # соседней детали и раскрывает щель шва
        v[lay_mf] = sstep(r, 0.045, 0.06) * fade_f
    print(f'[pores-attrs] verts={len(bm.verts)} collected={len(VRT_R)}', flush=True)

    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------- дерево GN
SOCKETS = [
    # (имя, тип, дефолт, min, max)
    ('Geometry',        'NodeSocketGeometry', None, None, None),
    ('Wobble',          'NodeSocketFloat', 0.0035, 0.0, 0.08),
    ('Pores Enable',    'NodeSocketFloat', 1.0, 0.0, 1.0),
    ('Pore Spacing',    'NodeSocketFloat', 0.122, 0.05, 0.3),
    ('Pore Radius',     'NodeSocketFloat', 0.055, 0.005, 0.09),
    ('Pore Var',        'NodeSocketFloat', 0.05, 0.0, 1.0),
    ('Med Spacing',     'NodeSocketFloat', 0.085, 0.02, 0.3),
    ('Med Radius',      'NodeSocketFloat', 0.032, 0.005, 0.09),
    ('Fine Spacing',    'NodeSocketFloat', 0.038, 0.02, 0.2),
    ('Fine Radius',     'NodeSocketFloat', 0.019, 0.005, 0.05),
    ('Smooth',          'NodeSocketInt', 7, 0, 24),
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
    noi.inputs['Scale'].default_value = 1.8
    noi.inputs['Detail'].default_value = 1.0
    wob = g.m('MULTIPLY', (noi, 'Fac'), S('Wobble'))
    wob = g.m('SUBTRACT', wob, g.m('MULTIPLY', S('Wobble'), 0.5))
    cur = set_pos(g, cur, g.vec('ADD', p1, g.vec('SCALE', nrm_in(g), scale=wob)))
    stages['wobble'] = cur

    # ---- SUBDIVIDE базы — OPT-IN (CORAL_SUBDIV=1): на 5.3 boolean по
    # измельчённой базе даёт БОЛЬШЕ мусора, чем по coarse; по умолч. выкл.
    if os.environ.get('CORAL_SUBDIV', '0') != '0':
        sub = g.n('GeometryNodeSubdivideMesh')
        sub.inputs['Level'].default_value = int(float(os.environ['CORAL_SUBDIV']))
        g.link(cur[0], cur[1], sub, 'Mesh')
        cur = (sub, 0)
    stages['subdiv'] = cur

    # ---- PORES: маска «наружу + по высоте»; крупные полости — пятнами (noise) ----
    p4 = pos_in(g); se4 = g.sep(p4)
    x4, y4, z4 = (se4, 'X'), (se4, 'Y'), (se4, 'Z')
    sen = g.sep(nrm_in(g))
    nxn, nyn = (sen, 'X'), (sen, 'Y')
    lr = g.m('POWER', g.m('ADD', g.m('MULTIPLY', x4, x4),
                          g.m('MULTIPLY', y4, y4)), 0.5)
    dotp = g.m('ADD', g.m('MULTIPLY', nxn, x4), g.m('MULTIPLY', nyn, y4))
    cosb = g.m('DIVIDE', dotp, g.m('MAXIMUM', lr, 0.02))
    out_ok = g.smooth(cosb, -0.02, 0.12)   # шире: чуть за силуэт — наклонные овальные устья как на референсе
    en = S('Pores Enable')
    zk_b = g.m('MULTIPLY', g.smooth(z4, 0.07, 0.16),
               g.m('SUBTRACT', 1.0, g.smooth(z4, 1.20, 1.34)))
    noi2 = g.n('ShaderNodeTexNoise')
    g.link(g.comb(x4, y4, z4), noi2, 'Vector')
    noi2.inputs['Scale'].default_value = 2.4
    noi2.inputs['Detail'].default_value = 2.0
    clump = g.m('ADD', 0.85, g.m('MULTIPLY', (noi2, 'Fac'), 0.15))
    mask_b = g.m('MULTIPLY', out_ok, zk_b)
    mask_b = g.m('MULTIPLY', mask_b, clump)
    mask_b = g.m('MULTIPLY', mask_b, en)
    mask_f = g.m('MULTIPLY', out_ok, g.smooth(z4, 0.045, 0.095))
    # v18: мягкая пятнистость мелкой сетки — зоны гуще/реже, как на керамике
    noi3 = g.n('ShaderNodeTexNoise')
    g.link(g.comb(x4, y4, z4), noi3, 'Vector')
    noi3.inputs['Scale'].default_value = 3.2
    noi3.inputs['Detail'].default_value = 2.0
    clump_f = g.m('ADD', 0.62, g.m('MULTIPLY', (noi3, 'Fac'), 0.38))
    mask_f = g.m('MULTIPLY', mask_f, clump_f)
    mask_f = g.m('MULTIPLY', mask_f, en)

    def attr_f(name):
        """Named Attribute (FLOAT) с базовой сетки (pr/pf/mb/mf)."""
        na = g.n('GeometryNodeInputNamedAttribute', data_type='FLOAT')
        try:
            na.inputs['Name'].default_value = name
        except Exception:
            try:
                na.attribute_name = name
            except Exception:
                na.inputs[0].default_value = name
        return (na, 0)

    def pore_spheres(mask_, dens, dist_min, rad, var, seed, mesh, depth=2.0,
                     subd=2, rattr=None, mattr=None, exclude=None, margin=0.012,
                     cut=True):
        """DPF-поры -> эллипсоиды-резаки. rattr — атрибут-множитель радиуса
        (локальная толщина), mattr — атрибут-маска. exclude — points-облако
        центров ЧУЖИХ пор: точки ближе r_чужой·pr + r_своя_max + margin
        сносятся (кромки чужих пор остаются чистыми — без пересечений).
        cut=False — только центры (для димплов). Возвращает (резаки|None,
        centers points)."""
        dpf = g.n('GeometryNodeDistributePointsOnFaces',
                  distribute_method='POISSON')
        g.link(mesh[0], mesh[1], dpf, 'Mesh')
        dpf.inputs['Density Max'].default_value = dens  # в POISSON именно Density Max
        dpf.inputs['Seed'].default_value = seed
        mk = mask_
        if mattr and os.environ.get('CORAL_MB_ATTR', '1') != '0':
            mk = g.m('MULTIPLY', mask_, attr_f(mattr))
        if os.environ.get('CORAL_DPF_CONST') == '1':
            # диагностический режим: константные factor/dmin
            dpf.inputs['Density Factor'].default_value = 1.0
            dpf.inputs['Distance Min'].default_value = float(os.environ.get('CORAL_DPF_DMIN', '0.155'))
        else:
            g.link(dist_min[0], dist_min[1], dpf, 'Distance Min')
            g.link(mk[0], mk[1], dpf, 'Density Factor')
        rnd = g.n('FunctionNodeRandomValue', data_type='FLOAT')
        rnd.inputs['Seed'].default_value = 11 + seed
        g.link(g.m('SUBTRACT', 1.0, g.m('MULTIPLY', var, 0.5)), rnd, 'Min')
        g.link(g.m('ADD', 1.0, g.m('MULTIPLY', var, 0.5)), rnd, 'Max')
        ri = (rnd, 'Value')
        if rattr and os.environ.get('CORAL_PR_ATTR', '1') != '0':
            ri = g.m('MULTIPLY', ri, attr_f(rattr))
        ri = g.m('MULTIPLY', ri, rad)
        pts_src = (dpf, 'Points')
        own_max = g.m('MULTIPLY', rad, g.m('ADD', 1.0, g.m('MULTIPLY', var, 0.5)))
        for (pts_o, rad_o) in (exclude or []):
            # порог отсечки от ЧУЖИХ пор: r_чужой_max·pr(локально) + своя_max + margin
            thr = g.m('MULTIPLY', attr_f('pr'),
                      g.m('MULTIPLY', rad_o,
                          g.m('ADD', 1.0, g.m('MULTIPLY', S('Pore Var'), 0.5))))
            thr = g.m('ADD', thr, own_max)
            thr = g.m('ADD', thr, margin)
            prox = g.n('GeometryNodeProximity', target_element='POINTS')
            g.link(pts_o[0], pts_o[1], prox, 'Target')
            sel = g.m('LESS_THAN', (prox, 'Distance'), thr)
            dele = g.n('GeometryNodeDeleteGeometry', domain='POINT', mode='ALL')
            g.link(pts_src, dele, 'Geometry')
            g.link(sel[0], sel[1], dele, 'Selection')
            pts_src = (dele, 0)
        if not cut:
            return None, pts_src
        ico = g.n('GeometryNodeMeshIcoSphere')
        ico.inputs['Radius'].default_value = 1.0
        ico.inputs['Subdivisions'].default_value = subd
        inst = g.n('GeometryNodeInstanceOnPoints')
        g.link(pts_src, inst, 'Points')
        g.link(dpf, 'Rotation', inst, 'Rotation')   # +Z вдоль нормали
        g.link(ico, 'Mesh', inst, 'Instance')
        # эллипсоид: глубина вдоль нормали xdepth -> пробивает стенку и сплавы
        g.link(g.comb(ri, ri, g.m('MULTIPLY', ri, depth)), inst, 'Scale')
        rlz = g.n('GeometryNodeRealizeInstances')
        rlz.inputs['Realize All'].default_value = True
        g.link(inst, 'Instances', rlz, 'Geometry')
        return (rlz, 0), pts_src

    def boolean_diff(mesh_a, cut):
        boo = g.n('GeometryNodeMeshBoolean', operation='DIFFERENCE')
        try:
            boo.solver = os.environ.get('CORAL_SOLVER', 'FLOAT')
        except Exception as e:
            print(f'[pores] solver set failed: {e}', flush=True)
        # НЕ трогаем Self Intersection / Hole Tolerant: принудительный True
        # на EXACT «съедает» глубокие буры (проверено v16_j)
        g.link(mesh_a[0], mesh_a[1], boo, 'Mesh 1')
        g.link(cut[0], cut[1], boo, 'Mesh 2')
        return (boo, 0)

    # A: центры КРУПНЫХ БУРОВ на чистой поверхности (атрибуты pr/mb живы)
    s1, pts1 = pore_spheres(mask_b, 500.0, S('Pore Spacing'), S('Pore Radius'),
                            S('Pore Var'), 3, cur, depth=3.8, rattr='pr', mattr='mb')
    if os.environ.get('CORAL_STAGE') == 'pts1':
        # счётчик: ico r=0.004 на точки -> realize (42 верша/точка)
        icoD = g.n('GeometryNodeMeshIcoSphere')
        icoD.inputs['Radius'].default_value = 0.004
        icoD.inputs['Subdivisions'].default_value = 1
        insD = g.n('GeometryNodeInstanceOnPoints')
        g.link(pts1[0], pts1[1], insD, 'Points')
        g.link(icoD, 'Mesh', insD, 'Instance')
        rlzD = g.n('GeometryNodeRealizeInstances')
        rlzD.inputs['Realize All'].default_value = True
        g.link(insD, 'Instances', rlzD, 'Geometry')
        stages['pts1'] = (rlzD, 0)
    # A2: центры СРЕДНИХ буров — та же маска, гуще; отсечка от крупных
    if os.environ.get('CORAL_NO_MED', '0') != '1':
        s1b, pts1b = pore_spheres(mask_b, 800.0, S('Med Spacing'), S('Med Radius'),
                                  S('Pore Var'), 5, cur, depth=2.0, rattr='pr', mattr='mm',
                                  exclude=[(pts1, S('Pore Radius'))])
    else:
        s1b, pts1b = None, None
    if os.environ.get('CORAL_STAGE') == 'ptsmed' and pts1b is not None:
        icoM = g.n('GeometryNodeMeshIcoSphere')
        icoM.inputs['Radius'].default_value = 0.007
        icoM.inputs['Subdivisions'].default_value = 1
        insM = g.n('GeometryNodeInstanceOnPoints')
        g.link(pts1b[0], pts1b[1], insM, 'Points')
        g.link(icoM, 'Mesh', insM, 'Instance')
        rlzM = g.n('GeometryNodeRealizeInstances')
        rlzM.inputs['Realize All'].default_value = True
        g.link(insM, 'Instances', rlzM, 'Geometry')
        jnM = g.n('GeometryNodeJoinGeometry')
        g.link(cur[0], cur[1], jnM, 'Geometry')
        g.link(rlzM, 'Geometry', jnM, 'Geometry')
        stages['ptsmed'] = (jnM, 0)
    # B: центры МЕЛКИХ пор (димплы) — отсечка и от крупных, и от средних
    if os.environ.get('CORAL_NO_PASSB', '0') == '1':
        pts2 = None
    else:
        excl2 = [(pts1, S('Pore Radius'))]
        if pts1b is not None:
            excl2.append((pts1b, S('Med Radius')))
        _s2, pts2 = pore_spheres(mask_f, 2500.0, S('Fine Spacing'), S('Fine Radius'),
                                 S('Pore Var'), int(float(os.environ.get('CORAL_SEED_FINE', '9'))),
                                 cur, depth=1.35, cut=False, mattr='mf', exclude=excl2, margin=0.006)
        if os.environ.get('CORAL_STAGE') == 'pts2':
            icoD2 = g.n('GeometryNodeMeshIcoSphere')
            icoD2.inputs['Radius'].default_value = 0.004
            icoD2.inputs['Subdivisions'].default_value = 1
            insD2 = g.n('GeometryNodeInstanceOnPoints')
            g.link(pts2[0], pts2[1], insD2, 'Points')
            g.link(icoD2, 'Mesh', insD2, 'Instance')
            rlzD2 = g.n('GeometryNodeRealizeInstances')
            rlzD2.inputs['Realize All'].default_value = True
            g.link(insD2, 'Instances', rlzD2, 'Geometry')
            stages['pts2'] = (rlzD2, 0)
    # E: BLUR ФОРМЫ — ДО boolean: лапласов blur ПОСЛЕ буров заливает узкие
    # глубокие колодцы (7 итераций ~ 5-8 см > радиуса бура — проверено
    # draft v17_j/k). Сначала гладим форму, потом режем чёткие дыры.
    blr = g.n('GeometryNodeBlurAttribute', data_type='FLOAT_VECTOR')
    posB = pos_in(g)
    g.link(posB, blr, 'Value')
    blr.inputs['Weight'].default_value = 1.0
    g.link(S('Smooth'), blr, 'Iterations')
    cur = set_pos(g, cur, blr)
    # D: буры — крупные затем средние; кромка каждого режется по чистой зоне
    # (чужие центры рядом снесены отсечкой), стенка лунки одна гладкая
    cur = boolean_diff(cur, s1)
    if s1b is not None:
        cur = boolean_diff(cur, s1b)
    if os.environ.get('CORAL_STAGE') == 'bool1':
        stages['bool1'] = cur
    # сварка: убивает вырожденные грани/рёбра boolean (источник «шипов»).
    # CORAL_MBD=0 — отключить: перед voxel-remesh mbd вреден (несвязная
    # топология ломает openvdb -> «лоскуты»)
    if os.environ.get('CORAL_MBD', '1') != '0':
        mbd = g.n('GeometryNodeMergeByDistance')
        g.link(cur[0], cur[1], mbd, 'Geometry')
        mbd.inputs['Distance'].default_value = 0.0012
        cur = (mbd, 0)
    # P: СУБДИВ ПОСЛЕ boolean (сам boolean — по coarse, без мусора), ПОД ДИМПЛАМИ:
    # ямки получают x4 плотности вершин -> мельче, круглее, без изломов
    # (плотная губка как на референсе). CORAL_POST_SUBDIV=0 — выкл.
    if os.environ.get('CORAL_POST_SUBDIV', '1') != '0':
        sub2 = g.n('GeometryNodeSubdivideMesh')
        sub2.inputs['Level'].default_value = int(float(os.environ.get('CORAL_POST_SUBDIV', '1')))
        g.link(cur[0], cur[1], sub2, 'Mesh')
        cur = (sub2, 0)
    # F: ДИМПЛЫ (мелкие поры) — после boolean, БЕЗ blur за ними: ни одного
    # резака — ни FLOAT-щелей, ни «звёзд», рвани нет в принципе. Кромки буров
    # чисты: их зоны свободны от мелких центров (отсечка в шаге B), ямка туда
    # не дотягивается. ×Pores Enable — пустое облако не должно вжимать всё.
    if os.environ.get('CORAL_NO_DIMPLE', '0') != '1' and pts2 is not None:
        prox2 = g.n('GeometryNodeProximity', target_element='POINTS')
        g.link(pts2[0], pts2[1], prox2, 'Target')
        r0f = S('Fine Radius')
        fall = g.m('SUBTRACT', 1.0, g.smooth((prox2, 'Distance'),
                                             g.m('MULTIPLY', r0f, 0.26),
                                             g.m('MULTIPLY', r0f, 0.74)))
        dep = g.m('MULTIPLY', r0f, 1.25)
        offv = g.vec('SCALE', nrm_in(g),
                     scale=g.m('MULTIPLY', dep,
                               g.m('MULTIPLY', fall, g.m('MULTIPLY', S('Pores Enable'), -1.0))))
        sp2 = g.n('GeometryNodeSetPosition')
        g.link(cur[0], cur[1], sp2, 'Geometry')
        g.link(offv[0], offv[1], sp2, 'Offset')
        cur = (sp2, 0)
    # G: shade smooth — в самом конце, по финальной геометрии
    sss = g.n('GeometryNodeSetShadeSmooth', domain='FACE')
    g.link(cur[0], cur[1], sss, 'Geometry')
    sss.inputs['Shade Smooth'].default_value = True
    cur = (sss, 0)
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

    if os.environ.get('CORAL_DUMP'):
        import re as _re
        def _nm(s):
            return _re.sub(r'\.\d{3}$', '', s)
        for nd in ng.nodes:
            print(f'[dump] N {nd.bl_idname}', flush=True)
        for lnk in ng.links:
            print(f'[dump] L {_nm(lnk.from_node.bl_idname)}:{lnk.from_socket.name} -> {_nm(lnk.to_node.bl_idname)}:{lnk.to_socket.name}', flush=True)
    return ng


# ---------------------------------------------------------------- материал/свет
def build_scene():
    obj = bpy.data.objects.get('CoralBase')
    mat = bpy.data.materials.new('CoralWhite')
    mat.use_nodes = True
    nt = mat.node_tree
    b = nt.nodes['Principled BSDF']
    b.inputs['Roughness'].default_value = 0.48
    if 'Specular IOR Level' in b.inputs:
        b.inputs['Specular IOR Level'].default_value = 0.35
    # v18: тёмные полости как на керамике — шейдерный Ambient Occlusion:
    # чем глубже точка (полость бура/димпла, складка), тем темнее Base Color.
    ao = nt.nodes.new('ShaderNodeAmbientOcclusion')
    ao.only_local = True
    ao.samples = 8
    for sname in ('Distance', 'Radius'):
        try:
            ao.inputs[sname].default_value = 0.25
            break
        except Exception:
            continue
    ao2 = nt.nodes.new('ShaderNodeAmbientOcclusion')
    ao2.only_local = True
    ao2.samples = 8
    for sname in ('Distance', 'Radius'):
        try:
            ao2.inputs[sname].default_value = 0.05
            break
        except Exception:
            continue
    aoM = nt.nodes.new('ShaderNodeMath'); aoM.operation = 'MULTIPLY'
    pwn = nt.nodes.new('ShaderNodeMath'); pwn.operation = 'POWER'
    pwn.inputs[1].default_value = 1.6
    mx = nt.nodes.new('ShaderNodeMix'); mx.data_type = 'RGBA'
    mx.blend_type = 'MIX'
    # Factor = (AO_макро · AO_микро)^1.6: открыто -> B белый, любая полость
    # (бур или микроямка) -> A тёмный; микро-канал затеняет саму губку
    mx.inputs[6].default_value = (0.085, 0.072, 0.065, 1.0)   # A: тёмная полость
    mx.inputs[7].default_value = (0.93, 0.92, 0.90, 1.0)      # B: базовый белый
    nt.links.new(ao.outputs['AO'], aoM.inputs[0])
    nt.links.new(ao2.outputs['AO'], aoM.inputs[1])
    nt.links.new(aoM.outputs[0], pwn.inputs[0])
    nt.links.new(pwn.outputs[0], mx.inputs['Factor'])
    nt.links.new(mx.outputs[2], b.inputs['Base Color'])
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
    CLOSE = os.environ.get('CORAL_CLOSE') == '1'

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
    az = math.radians(float(os.environ.get('CORAL_AZ', '0')))
    cam_loc = (1.15, -1.45, 0.62) if CLOSE else (2.2, -2.75, 0.92)
    cam.location = Matrix.Rotation(az, 4, 'Z') @ Vector(cam_loc)
    if CLOSE:
        cd.lens = 62
        AIM = Vector((-0.02, 0, 0.52))
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
    print(f'[coral] v18 start LOW={LOW} DIAG={DIAG}', flush=True)
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

    # voxel-remesh — теперь БЕЗОПАСЕН: тело монолитное, тонких стенок нет.
    # Перестраивает топологию после float-boolean (убивает рваные кромки)
    # и сплавляет пересекающиеся ствол/ветви в один объём.
    if os.environ.get('CORAL_REMESH', '0') == '1':
        rm = obj.modifiers.new('Remesh', 'REMESH')
        rm.mode = 'VOXEL'
        rm.voxel_size = float(os.environ.get('CORAL_VOXEL_SIZE', '0.008'))
        try:
            rm.smooth_shading = True
        except Exception:
            pass
    # blur/shade-smooth теперь внутри CoralGN (шаги E/G): димплы наносятся
    # после blur и остаются чёткими; отдельный модификатор CoralPost удалён.
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
        for nm4 in ('pr', 'pf', 'mb', 'mf'):
            a4 = obj.data.attributes.get(nm4)
            if a4 is None:
                print(f'[diag] attr {nm4}: MISSING', flush=True)
            else:
                v4 = [d.value for d in a4.data]
                print(f'[diag] attr {nm4}: min={min(v4):.3f} max={max(v4):.3f} '
                      f'avg={sum(v4) / len(v4):.3f}', flush=True)

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
