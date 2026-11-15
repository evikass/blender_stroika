# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.neuro_autorig — авто-риг в стиле Hyper3D.

Скелет подгоняется под тело меша, а не меш под скелет:
по геометрии стоящего (Z-up) гуманоида находятся анатомические точки
(промежность, тазобедренные, колени, щиколотки, плечи, локти, запястья,
шея, макушка), и из них строится тот же neuro-скелет (18 костей, контракт
human.json), но с пропорциями конкретного персонажа.

Если меш не похож на гуманоида — кидается RigFitError, вызывающий код
падает обратно на старое поведение (меш масштабируется под скелет 1.70 м).
"""
import math

import bpy

try:
    import numpy as np
except ImportError:                                    # numpy есть в Blender
    np = None


class RigFitError(RuntimeError):
    """Меш не удалось разобрать как гуманоида."""


NBINS = 200          # срезов по высоте
ROOT = "pelvis"


# ------------------------------------------------------------- точки меша

def mesh_world_points(mesh_obj):
    """(N,3) мировые координаты вершин меша (numpy).

    ВАЖНО: локальный import numpy внутри функции делать нельзя — он
    делает np локальной переменной всей функции, и проверка np is None
    выше падает UnboundLocalError ещё до импорта. numpy импортируется
    один раз на уровне модуля (в Blender он есть всегда)."""
    if np is None:
        raise RigFitError("numpy недоступен")
    me = mesh_obj.data
    n = len(me.vertices)
    if n < 40:
        raise RigFitError("слишком мало вершин (%d)" % n)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    pts = co.reshape(-1, 3)
    mw = mesh_obj.matrix_world
    m = [list(mw[0]), list(mw[1]), list(mw[2])]
    pts = pts @ np.asarray(m, dtype=np.float64)[:, :3].T + \
        np.asarray([mw[0][3], mw[1][3], mw[2][3]], dtype=np.float64)
    if n > 600000:                                     # ускорение анализа
        pts = pts[::2]
    return pts


# --------------------------------------------------------- срезы по высоте

def _slices(pts, z0, h):
    """Список срезов: dict(i, z, y0, y1, x0, x1, cy, cx, n, ys, xs)."""
    zn = (pts[:, 2] - z0) / h
    idx = np.clip((zn * NBINS).astype(np.int64), 0, NBINS - 1)
    out = []
    for i in range(NBINS):
        sel = pts[idx == i]
        if len(sel) < 4:
            out.append(None)
            continue
        ys, xs = sel[:, 1], sel[:, 0]
        out.append({
            "i": i, "n": len(sel),
            "y0": float(ys.min()), "y1": float(ys.max()),
            "x0": float(xs.min()), "x1": float(xs.max()),
            "cy": float(ys.mean()), "cx": float(xs.mean()),
            "ys": ys, "xs": xs, "zs": sel[:, 2],
        })
    return out


def _slice_z(sl, z0, h):
    return z0 + (sl["i"] + 0.5) / NBINS * h


def _leg_split(sl, yc, gap):
    """Разделены ли ноги на этом срезе: в центральной полосе ищем
    самый пустой зазор (долину плотности).

    Старый тест «полоса |y| < gap пуста» ломался на толстых бёдрах:
    внутренние края труб дают пики плотности у центра (касательные
    к поверхности), и срез раздельных ног выглядел «заполненным в
    середине». Теперь: сглаженная гистограмма полосы |y| < 2.5*gap,
    минимум у центра — долина; она обязана быть < 22% плотности ног
    по бокам. Возвращает (y_left, y_right) центроиды ног или None.
    """
    if sl is None:
        return None
    y = sl["ys"] - yc
    n = int(len(y))
    if n < 12:
        return None
    lo, hi = -2.5 * gap, 2.5 * gap
    sel = y[np.abs(y) < hi]
    if len(sel) < 10:
        return None
    nb = 20
    hist = np.histogram(sel, bins=nb, range=(lo, hi))[0].astype(float)
    ker = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    sm = np.convolve(hist, ker / ker.sum(), mode="same")
    mid = nb // 2
    win = 4
    v_off = int(np.argmin(sm[mid - win:mid + win + 1]))
    v_i = mid - win + v_off
    v_y = lo + (v_i + 0.5) * (hi - lo) / nb
    valley = float(sm[v_i])
    l_lo, l_hi = max(0, v_i - win - 4), max(1, v_i - 2)
    r_lo, r_hi = v_i + 3, min(nb, v_i + win + 5)
    left = float(sm[l_lo:l_hi].max()) if l_hi > l_lo else 0.0
    right = float(sm[r_lo:r_hi].max()) if r_hi > r_lo else 0.0
    sides = min(left, right)
    if sides <= 0 or valley > 0.22 * sides:
        return None
    yl = y[y > v_y]
    yr = y[y < v_y]
    if len(yl) < 0.12 * n or len(yr) < 0.12 * n:
        return None
    return (float(yl.mean()), float(yr.mean()))


def _cluster_y(sl, side, yc, min_y=0.01):
    """Центроид y кластера ноги (+1 левая/-1 правая) на срезе."""
    y = sl["ys"] - yc
    sel = y[(y * side > min_y)]
    if len(sel) < 3:
        return None
    return float(sel.mean())


def _arm_verts(pts, z0, h, z_lo_n, y_sh, y_hand):
    """Вершины рук: боковые, выше z_lo_n, |y| за пределами торса."""
    y = pts[:, 1]
    z = (pts[:, 2] - z0) / h
    sel = (z > z_lo_n) & (np.abs(y) > y_sh * 1.05)
    return pts[sel]


def _median_z_of(pts, z0, h, y_lo, y_hi):
    """Медианная z (м) вершин с |y| в полосе [y_lo, y_hi]."""
    y = np.abs(pts[:, 1])
    sel = (y >= y_lo) & (y <= y_hi)
    if sel.sum() < 5:
        return None
    return float(np.median(pts[sel, 2]))


# ------------------------------------------------------------- анализ

def analyze_humanoid(pts):
    """Точки тела -> landmarks (м, мировые координаты). RigFitError если
    меш не гуманоид. Персонаж: стоит (Z-up), лицо вдоль +X, левая сторона +Y.
    """
    if np is None:
        raise RigFitError("numpy недоступен")
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 40:
        raise RigFitError("слишком мало вершин")
    z0 = float(pts[:, 2].min())
    z1 = float(pts[:, 2].max())
    h = z1 - z0
    if h < 0.05:
        raise RigFitError("высота меша %.3f м" % h)
    ymin, ymax = float(pts[:, 1].min()), float(pts[:, 1].max())
    xmin, xmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    yc, xc = 0.5 * (ymin + ymax), 0.5 * (xmin + xmax)
    span = ymax - ymin

    sl = _slices(pts, z0, h)
    gap = 0.020 * h          # зазор между ногами масштабируется от роста

    # ---- промежность: верхний срез с раздельными ногами
    crotch_i = None
    for i in range(10, 145):
        sp = _leg_split(sl[i], yc, gap)
        if sp is not None:
            crotch_i = i
    if crotch_i is None:
        raise RigFitError("не найдена промежность (ноги не разделяются)")
    crotch_z = z0 + (crotch_i + 1) / NBINS * h
    crotch_n = (crotch_z - z0) / h
    hip_z = crotch_z + 0.045 * h
    hip_n = (hip_z - z0) / h

    # ---- ширина торса: медианная по полосе таза НАД промежностью
    # (рук там нет, ноги уже слились с тазом)
    band = [s for s in sl[crotch_i:crotch_i + int(0.10 * NBINS)] if s]
    if len(band) < 5:
        raise RigFitError("пустые срезы в поясе таза")
    w_torso = float(np.median([s["y1"] - s["y0"] for s in band]))
    if w_torso < 0.08 * h:
        raise RigFitError("ширина таза %.3f м при росте %.3f м" %
                          (w_torso, h))

    # ---- y тазобедренных: центроиды бёдер чуть ниже промежности
    thigh_sl = sl[max(0, crotch_i - 4)]
    y_hipL = _cluster_y(thigh_sl, +1, yc, 0.005 * h) if thigh_sl else None
    y_hipR = _cluster_y(thigh_sl, -1, yc, 0.005 * h) if thigh_sl else None
    if y_hipL is None or y_hipR is None or \
            abs(y_hipL - y_hipR) < 0.05 * h:
        raise RigFitError("не найдены тазобедренные суставы")

    # ---- ноги: колено/щиколотка по центроидам кластеров
    ankle_z = z0 + 0.045 * h
    knee_z = crotch_z * 0.58 + ankle_z * 0.42
    if knee_z <= ankle_z or knee_z >= crotch_z:
        knee_z = 0.5 * (crotch_z + ankle_z)

    def leg_pt(zq, side):
        i = int(np.clip((zq - z0) / h * NBINS - 0.5, 0, NBINS - 1))
        for di in (0, -1, 1, -2, 2):
            s = sl[i + di] if 0 <= i + di < NBINS else None
            cy = _cluster_y(s, side, yc, 0.005 * h) if s else None
            if cy is not None:
                cx = float(s["xs"].mean())
                return cx, cy + yc
        return 0.0, (y_hipL if side > 0 else y_hipR) + yc

    kneeL = leg_pt(knee_z, +1)
    kneeR = leg_pt(knee_z, -1)
    ankL = leg_pt(ankle_z, +1)
    ankR = leg_pt(ankle_z, -1)

    # ---- ступни: носок/пятка в полосе щиколотки
    def foot_pts(ank, side):
        i0 = int(0.005 * NBINS)
        i1 = int(0.065 * NBINS)
        xs, ys = [], []
        for i in range(i0, i1):
            s = sl[i]
            if s is None:
                continue
            y = s["ys"] - yc
            sel = (y * side > 0.005 * h) & (np.abs(y - (ank[1] - yc)) <
                                            0.06 * h)
            if sel.sum() >= 3:
                xs.append(s["xs"][sel])
                ys.append(s["ys"][sel])
        if not xs:
            return ank[0] + 0.05 * h, ank[0] - 0.02 * h
        xa = np.concatenate(xs)
        toe = float(xa.max())
        heel = float(xa.min())
        return toe, heel

    footL = foot_pts(ankL, +1)
    footR = foot_pts(ankR, -1)

    # ---- руки: полоса, где меш шире торса. Верх полосы = линия плеч,
    # низ полосы = подмышка (T-pose) или кисти (A-pose); ширина груди
    # меряется ПОД низом полосы
    wide = [s for s in sl if s and (s["y1"] - s["y0"]) > 1.45 * w_torso]
    upper_wide = [s for s in wide if s["i"] > 0.45 * NBINS]
    arm_note = ""
    if upper_wide:
        lo_i = min(s["i"] for s in upper_wide)
        hi_i = max(s["i"] for s in upper_wide)
        shoulder_z = z0 + (hi_i + 1.0) / NBINS * h
        pit = [s for s in sl[max(0, lo_i - 10):lo_i] if s]
        y_sh = 0.5 * float(np.median([(s["y1"] - s["y0"]) for s in pit])) \
            if len(pit) >= 3 else 0.55 * w_torso
    else:
        # руки прижаты/не шире торса: плечи по антропометрии
        shoulder_n = crotch_n + 0.68 * (1.0 - crotch_n)
        shoulder_z = z0 + shoulder_n * h
        y_sh = 0.55 * w_torso
        arm_note = "руки не шире торса, плечи по пропорциям"
    y_sh = min(y_sh, 0.5 * span - 0.02 * h)

    # кисти: самые боковые вершины верхней половины тела
    up = pts[(pts[:, 2] - z0) / h > max(0.35, hip_n)]
    if len(up) < 20:
        raise RigFitError("нет вершин верхней половины тела")
    y_handL = float(up[:, 1].max())
    y_handR = float(up[:, 1].min())
    if y_handL - y_sh < 0.10 * h:
        raise RigFitError("размах рук мал (%.2f от роста)" %
                          ((y_handL - y_sh) / h))
    av = _arm_verts(pts, z0, h, max(0.30, crotch_n - 0.05), y_sh, y_handL)

    def arm_pts(side):
        yh = y_handL if side > 0 else y_handR
        yh = abs(yh)
        elbow_y = 0.5 * (y_sh + yh)
        wrist_y = yh - 0.11 * max(yh - y_sh, 1e-6)
        ez = _median_z_of(av, z0, h,
                          elbow_y - 0.06 * h, elbow_y + 0.06 * h)
        wz = _median_z_of(av, z0, h,
                          wrist_y - 0.05 * h, yh)
        if ez is None:
            ez = shoulder_z
        if wz is None:
            wz = ez
        return elbow_y, ez, wrist_y, wz

    elL_y, elL_z, wrL_y, wrL_z = arm_pts(+1)
    elR_y, elR_z, wrR_y, wrR_z = arm_pts(-1)

    # ---- толщина руки (для радиусов костей): вертикальный размах рук
    # в полосе середины плеча
    arm_th = 0.09 * h
    if len(av) > 20:
        mid_y = 0.5 * (y_sh + y_handL)
        band = av[np.abs(av[:, 1]) > mid_y - 0.05 * h]
        band = band[np.abs(band[:, 1]) < mid_y + 0.05 * h]
        if len(band) > 15:
            arm_th = float(np.percentile(band[:, 2], 92) -
                           np.percentile(band[:, 2], 8))
            arm_th = min(max(arm_th, 0.03 * h), 0.16 * h)

    # ---- шея/голова
    head_band = [s for s in sl[int(0.88 * NBINS):] if s]
    w_head = float(np.median([s["y1"] - s["y0"] for s in head_band])) \
        if len(head_band) >= 4 else 0.14 * h
    neck_i = None
    for i in range(int(0.95 * NBINS), int(0.62 * NBINS), -1):
        s = sl[i]
        if s and (s["y1"] - s["y0"]) > 1.6 * w_head:
            neck_i = i
            break
    if neck_i is None:
        neck_i = min(NBINS - 1, int(0.86 * NBINS))
    neck_z = z0 + (neck_i + 0.5) / NBINS * h
    neck_z = max(neck_z, shoulder_z + 0.02 * h)

    # ---- контрольная сборка landmarks
    lm = {
        "height": h, "span": span, "z0": z0, "xc": xc, "yc": yc,
        "crotch_z": crotch_z, "hip_z": hip_z,
        "y_hipL": y_hipL + yc, "y_hipR": y_hipR + yc,
        "knee_z": knee_z,
        "kneeL": kneeL, "kneeR": kneeR,
        "ankle_z": ankle_z, "ankL": ankL, "ankR": ankR,
        "footL": footL, "footR": footR,
        "shoulder_z": shoulder_z, "y_sh": y_sh,
        "elbowL_y": elL_y + yc, "elbowL_z": elL_z,
        "elbowR_y": -(elR_y) + yc, "elbowR_z": elR_z,
        "wristL_y": wrL_y + yc, "wristL_z": wrL_z,
        "wristR_y": -(wrR_y) + yc, "wristR_z": wrR_z,
        "handL_y": y_handL, "handR_y": y_handR,
        "neck_z": neck_z, "head_top_z": z1,
        "w_torso": w_torso, "arm_th": arm_th, "note": arm_note,
    }
    _sanity(lm)
    return lm


def _sanity(lm):
    h = lm["height"]
    cn = (lm["crotch_z"] - lm["z0"]) / h
    sn = (lm["shoulder_z"] - lm["z0"]) / h
    nn = (lm["neck_z"] - lm["z0"]) / h
    ok = (
        0.30 < cn < 0.62 and
        0.55 < sn < 0.95 and
        sn < nn < 0.97 and
        lm["z0"] + 0.005 * h < lm["ankle_z"] < lm["knee_z"] < cn * h and
        abs(lm["y_hipL"] - lm["y_hipR"]) > 0.05 * h and
        lm["handL_y"] - lm["y_sh"] > 0.08 * h
    )
    if not ok:
        raise RigFitError("точки тела не похожи на гуманоида "
                          "(промежность/плечи/ноги не сошлись)")


# --------------------------------------------- fitted spec (контракт 18)

def build_fitted_spec(base_spec, lm):
    """Спека human.json с теми же именами/иерархией/осями суставов,
    но позиции костей — из landmarks конкретного персонажа."""
    h = lm["height"]
    z0, xc = lm["z0"], lm["xc"]
    hip_z = lm["hip_z"]
    sh_z = lm["shoulder_z"]
    torso = sh_z - hip_z
    if torso <= 0.05 * h:
        raise RigFitError("торс короче 5% роста")

    def z(frac):
        return hip_z + frac * torso

    # радиусы из реальных толщин
    r_pelvis = max(0.045 * h, 0.46 * lm["w_torso"])
    r_chest = max(0.05 * h, 0.48 * lm["w_torso"])
    r_thigh = max(0.03 * h, 0.30 * abs(lm["y_hipL"] - lm["y_hipR"]))
    r_shin = 0.62 * r_thigh
    r_uarm = 0.42 * lm["arm_th"]
    r_farm = 0.36 * lm["arm_th"]
    r_head = max(0.04 * h, 0.31 * lm["w_torso"])
    foot_len = max(lm["footL"][0] - lm["footL"][1], 0.05 * h)

    S = {l["name"]: l for l in base_spec["links"]}
    if set(S) == set() or len(base_spec["links"]) != 18:
        raise RigFitError("ожидается спека human.json (18 звеньев)")

    out = {"name": base_spec.get("name", "human_v1") + "_fitted",
           "root": base_spec.get("root", ROOT), "links": []}

    def link(name, com, anchor=None, shape=None):
        l = dict(S[name])
        l["com"] = [round(float(v), 5) for v in com]
        if anchor is not None and l.get("joint"):
            l["joint"] = dict(l["joint"])
            l["joint"]["anchor"] = [round(float(v), 5) for v in anchor]
        if shape:
            l["shape"] = shape
        out["links"].append(l)

    # таз/корень
    link("pelvis", [xc, 0.5 * (lm["y_hipL"] + lm["y_hipR"]), hip_z],
         shape={"type": "capsule", "axis": "x", "radius": r_pelvis,
                "half_len": 0.55 * r_pelvis})
    # позвоночник/грудь/голова
    link("spine", [xc, 0, z(0.48)], [xc, 0, z(0.26)],
         {"type": "capsule", "axis": "z",
          "radius": 0.8 * r_chest, "half_len": 0.10 * torso})
    link("chest", [xc, 0, z(0.83)], [xc, 0, z(0.62)],
         {"type": "capsule", "axis": "z", "radius": r_chest,
          "half_len": 0.16 * torso})
    link("head", [xc, 0, 0.5 * (lm["neck_z"] + lm["head_top_z"])],
         [xc, 0, lm["neck_z"]],
         {"type": "capsule", "axis": "z", "radius": r_head,
          "half_len": 0.30 * (lm["head_top_z"] - lm["neck_z"])})
    for sgn, sfx in ((+1, "L"), (-1, "R")):
        yh = lm["y_hip%s" % sfx]
        ank = lm["ank%s" % sfx]
        knee = lm["knee%s" % sfx]
        toe, heel = lm["foot%s" % sfx]
        el_y = lm["elbow%s_y" % sfx]
        el_z = lm["elbow%s_z" % sfx]
        wr_y = lm["wrist%s_y" % sfx]
        wr_z = lm["wrist%s_z" % sfx]
        link("shoulderP_" + sfx,
             [xc, sgn * 0.55 * lm["y_sh"], sh_z],
             [xc, sgn * 0.28 * lm["y_sh"], sh_z],
             {"type": "capsule", "axis": "y",
              "radius": 0.30 * r_uarm, "half_len": 0.14 * lm["y_sh"]})
        link("uarm_" + sfx, [xc, el_y, el_z],
             [xc, sgn * lm["y_sh"], sh_z],
             {"type": "capsule", "axis": "y", "radius": r_uarm,
              "half_len": max(0.5 * abs(el_y - sgn * lm["y_sh"]),
                              0.05 * h)})
        link("farm_" + sfx, [xc, wr_y, wr_z], [xc, el_y, el_z],
             {"type": "capsule", "axis": "y", "radius": r_farm,
              "half_len": max(0.5 * abs(wr_y - el_y), 0.05 * h)})
        link("hipR_" + sfx, [xc, sgn * 0.8 * abs(yh), hip_z],
             [xc, yh, hip_z],
             {"type": "capsule", "axis": "z",
              "radius": 0.35 * r_thigh, "half_len": 0.35 * r_thigh})
        link("thigh_" + sfx, [knee[0], knee[1], lm["knee_z"]],
             [xc, yh, hip_z],
             {"type": "capsule", "axis": "z", "radius": r_thigh,
              "half_len": max(0.4 * (hip_z - lm["knee_z"]),
                              0.06 * h)})
        link("shin_" + sfx, [ank[0], ank[1],
                             0.5 * (lm["knee_z"] + lm["ankle_z"])],
             [knee[0], knee[1], lm["knee_z"]],
             {"type": "capsule", "axis": "z", "radius": r_shin,
              "half_len": max(0.4 * (lm["knee_z"] - lm["ankle_z"]),
                              0.05 * h)})
        foot_z = min(lm["ankle_z"] - 0.6 * r_shin, lm["z0"] + 0.035 * h)
        foot_z = max(foot_z, lm["z0"] + 0.02 * h)
        link("foot_" + sfx,
             [ank[0] + 0.45 * foot_len, ank[1], foot_z],
             [ank[0], ank[1], lm["ankle_z"] - 0.3 * r_shin],
             {"type": "box",
              "half": [max(0.5 * foot_len, 0.05 * h),
                       max(0.34 * r_shin, 0.02 * h),
                       max(0.025 * h, 0.35 * r_shin)]})
    names = [l["name"] for l in out["links"]]
    if sorted(names) != sorted(S.keys()):
        raise RigFitError("потеряны звенья спеки")
    return out


def fit_spec_to_mesh(base_spec, pts):
    """(fitted_spec, info): анализ + сборка спеки под тело."""
    lm = analyze_humanoid(pts)
    spec = build_fitted_spec(base_spec, lm)
    info = ("Auto-Rig: рост %.2f м, размах рук %.2f м, плечи %.2f м, "
            "промежность %.2f м%s" %
            (lm["height"], lm["span"], 2.0 * lm["y_sh"],
             lm["crotch_z"] - lm["z0"],
             ("; " + lm["note"]) if lm["note"] else ""))
    return spec, lm, info


# -------------------------------------------------- вспомогательное

def place_mesh(mesh, rot_x_deg=0.0, rot_z_deg=0.0, do_apply=True):
    """Повернуть меш (авто-ориентация) и поставить: низ в z=0, центр xy в
    (0,0). Размер НЕ меняем — скелет подгоняется под тело."""
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except RuntimeError:
        pass
    if abs(rot_x_deg) > 1e-6:
        mesh.rotation_euler.x += math.radians(rot_x_deg)
    if abs(rot_z_deg) > 1e-6:
        mesh.rotation_euler.z += math.radians(rot_z_deg)
    bpy.context.view_layer.update()
    try:
        from .neuro_rig import _world_bbox
    except ImportError:
        from neuro_rig import _world_bbox
    mn, mx = _world_bbox(mesh)
    cx, cy = 0.5 * (mn.x + mx.x), 0.5 * (mn.y + mx.y)
    mesh.location.x += -cx
    mesh.location.y += -cy
    mesh.location.z += -mn.z
    bpy.context.view_layer.update()
    if do_apply:
        bpy.ops.object.select_all(action="DESELECT")
        mesh.select_set(True)
        bpy.context.view_layer.objects.active = mesh
        bpy.ops.object.transform_apply(location=False, rotation=True,
                                       scale=True)


def make_debug_empties(lm):
    """Пустышки по ключевым точкам — визуальный контроль подгонки."""
    outs = []
    keys = [
        ("ncp_lm_hipL", lm["xc"], lm["y_hipL"], lm["hip_z"]),
        ("ncp_lm_hipR", lm["xc"], lm["y_hipR"], lm["hip_z"]),
        ("ncp_lm_kneeL", lm["kneeL"][0], lm["kneeL"][1], lm["knee_z"]),
        ("ncp_lm_kneeR", lm["kneeR"][0], lm["kneeR"][1], lm["knee_z"]),
        ("ncp_lm_ankleL", lm["ankL"][0], lm["ankL"][1], lm["ankle_z"]),
        ("ncp_lm_ankleR", lm["ankR"][0], lm["ankR"][1], lm["ankle_z"]),
        ("ncp_lm_shoulderL", lm["xc"], lm["y_sh"], lm["shoulder_z"]),
        ("ncp_lm_shoulderR", lm["xc"], -lm["y_sh"], lm["shoulder_z"]),
        ("ncp_lm_elbowL", lm["xc"], lm["elbowL_y"], lm["elbowL_z"]),
        ("ncp_lm_elbowR", lm["xc"], lm["elbowR_y"], lm["elbowR_z"]),
        ("ncp_lm_wristL", lm["xc"], lm["wristL_y"], lm["wristL_z"]),
        ("ncp_lm_wristR", lm["xc"], lm["wristR_y"], lm["wristR_z"]),
        ("ncp_lm_neck", lm["xc"], 0.0, lm["neck_z"]),
        ("ncp_lm_headtop", lm["xc"], 0.0, lm["head_top_z"]),
        ("ncp_lm_crotch", lm["xc"], 0.0, lm["crotch_z"]),
    ]
    for name, x, y, zq in keys:
        e = bpy.data.objects.get(name)
        if e is None:
            e = bpy.data.objects.new(name, None)
            bpy.context.scene.collection.objects.link(e)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = max(0.012 * lm["height"], 0.01)
        e.location = (x, y, zq)
        outs.append(e)
    return outs
