# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.neuro_creature — универсальный авто-риг существ.

Переписан с нуля (v2.4.0). Работает с ЛЮБЫМ планом тела: двуногий,
четвероногий, многоногий, с крыльями, с хвостом, с шеей. По геометрии
меша модуль сам определяет:

  - сколько НОГ (кластеры у земли; пары по зеркальной симметрии);
  - есть ли РУКИ и КРЫЛЬЯ (крыло = тонкое длинное латеральное плечо
    и/или крепление за спиной; у четвероногого любая латеральная пара
    считается крыльями);
  - есть ли ХВОСТ (масса позади задней кромки торса) и какой длины;
  - есть ли ШЕЯ (путь от верха торса до головы) и сколько звеньев;
  - сколько ПАЛЬЦЕВ на кисти/лапе/крыле (стойкие кластеры в сечениях
    дистальной части конечности).

Из найденного собирается спека в той же схеме, что и human.json
(links: name/parent/joint/com/shape) — её строит тот же
neuro_rig.build_armature, весы назначает тот же автоматический
скининг. Человек (2 ноги, без крыльев/хвоста) сначала пробуется как
human-контракт 18 костей (neuro_autorig: мышцы и анимация neurosim
работают); не сошлось — строится универсальный двуногий.

numpy импортируется ТОЛЬКО на уровне модуля (урок v2.3.1: локальный
import внутри функции затеняет имя и падает UnboundLocalError).
"""
import math

import bpy

try:
    import numpy as np
except ImportError:                                    # в Blender есть всегда
    np = None


class CreatureError(RuntimeError):
    """Меш не удалось разобрать в план тела."""


NBINS = 200


# ------------------------------------------------------------ утилиты

def _clusters(pts_xy, cell, min_pts):
    """2D-кластеры (flood fill по сетке, 8-связность).

    pts_xy: (N,2). Возвращает список dict(idx, c=(cx,cy), n, sp=(sx,sy))
    — по убыванию размера.
    """
    if pts_xy is None or len(pts_xy) == 0:
        return []
    ij = np.floor(pts_xy / cell).astype(np.int64)
    cells = {}
    for k in range(len(pts_xy)):
        cells.setdefault((int(ij[k, 0]), int(ij[k, 1])), []).append(k)
    nbrs = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1))
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for d in nbrs:
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(members) < min_pts:
            continue
        P = pts_xy[members]
        out.append({
            "idx": members, "n": len(members),
            "c": (float(P[:, 0].mean()), float(P[:, 1].mean())),
            "sp": (float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1]))),
        })
    out.sort(key=lambda d: -d["n"])
    return out


def _blobs3d(pts, cell, min_pts):
    """3D-кластеры (flood fill по сетке, 26-связность).

    Возвращает список dict(c, n, ext=(ex,ey,ez), dir=главная ось PCA,
    P=точки) — по убыванию размера.
    """
    if pts is None or len(pts) == 0:
        return []
    ij = np.floor(pts / cell).astype(np.int64)
    cells = {}
    for k in range(len(pts)):
        cells.setdefault((int(ij[k, 0]), int(ij[k, 1]), int(ij[k, 2])),
                         []).append(k)
    offs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
            for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        stack, members = [start], []
        seen.add(start)
        while stack:
            cur = stack.pop()
            members.extend(cells[cur])
            for d in offs:
                nxt = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(members) < min_pts:
            continue
        P = pts[members]
        ctr = P.mean(axis=0)
        dmain = _cov_axis(P)
        out.append({
            "idx": members, "n": len(members), "P": P,
            "c": (float(ctr[0]), float(ctr[1]), float(ctr[2])),
            "ext": (float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1])),
                    float(np.ptp(P[:, 2]))),
            "dir": dmain,
        })
    out.sort(key=lambda d: -d["n"])
    return out


def _sym_yc(pts):
    """Плоскость симметрии y = yc.

    Старт: центр bbox (у зеркально-симметричного тела min/max
    симметричны). Уточнение: профиль «масса на расстоянии k от
    плоскости» слева и справа сравнивается по ВСЕМУ размаху — узкие
    окна у гладких мест (кончик руки) совпадают на любом сдвиге и
    уводят метрику к краям."""
    y = pts[:, 1]
    lo, hi = float(y.min()), float(y.max())
    span = max(hi - lo, 1e-9)
    total = max(float(len(y)), 1.0)
    yc0 = 0.5 * (lo + hi)
    nb, bw = 60, span / 60.0
    best_d, best_c = None, yc0
    for c in np.linspace(yc0 - 0.12 * span, yc0 + 0.12 * span, 17):
        rel = y - c
        Lh = np.histogram(-rel[rel < 0], bins=nb, range=(0, span))[0]
        Rh = np.histogram(rel[rel >= 0], bins=nb, range=(0, span))[0]
        d = float(np.abs(Lh - Rh).sum()) / total
        if best_d is None or d < best_d:
            best_d, best_c = d, float(c)
    return best_c


def _cov_axis(P):
    """Главная ось облака точек (единичный вектор)."""
    ctr = P.mean(axis=0)
    try:
        w, v = np.linalg.eigh(np.cov((P - ctr).T))
        return v[:, int(np.argmax(w))]
    except np.linalg.LinAlgError:
        return np.array([0.0, 0.0, 1.0])


def _perp(d):
    """Единичный вектор, перпендикулярный d."""
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else \
        np.array([0.0, 1.0, 0.0])
    u = np.cross(d, a)
    u /= max(float(np.linalg.norm(u)), 1e-9)
    return u


def _dist_to_path(P, path):
    """Минимальное расстояние от точек P (N,3) до ломаной path."""
    best = np.full(len(P), 1e18)
    for a, b in zip(path[:-1], path[1:]):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab = b - a
        L2 = max(float(ab @ ab), 1e-12)
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(P - proj, axis=1))
    return best


def _joint_axis(d):
    """Ось сустава (косметика позирования): перпендикуляр к кости."""
    d = np.asarray(d, dtype=float)
    n = max(float(np.linalg.norm(d)), 1e-9)
    d = d / n
    if abs(d[2]) > 0.7:                    # вертикальная кость — pitch
        return [0, 1, 0]
    if abs(d[1]) > 0.7:                    # боковая кость — swing
        return [1, 0, 0]
    return [0, 1, 0]


# --------------------------------------------------- ориентация (2 стадии)

def orient_stand(mesh):
    """Стадия 1: поднять меш (rot X), если он лежит (Y-up движка).
    Только подъём; доворот до +X делает face_rotation по спайну/ногам."""
    try:
        from .neuro_rig import _world_bbox
    except ImportError:
        from neuro_rig import _world_bbox
    mn, mx = _world_bbox(mesh)
    ex, ey, ez = mx.x - mn.x, mx.y - mn.y, mx.z - mn.z
    return 90.0 if ez <= min(ex, ey) else 0.0


def _facing_offset(pts, z0, h):
    """Куда смотрит существо: сдвиг центроидов стоп от голеней.

    Носки/пальцы лап выдаются вперёд от оси ноги — средний (по массе)
    вектор «стопа минус голень» показывает лицо. Возвращает (fx, fy)
    или None, если стопы не читаются (не существо/нет стоп)."""
    foot = pts[(pts[:, 2] >= z0 + 0.015 * h) &
               (pts[:, 2] <= z0 + 0.055 * h)]
    leg = pts[(pts[:, 2] > z0 + 0.06 * h) &
              (pts[:, 2] <= z0 + 0.11 * h)]
    if len(foot) < 20 or len(leg) < 20:
        return None
    cf = _clusters(foot[:, :2], 0.018 * h,
                   max(6, int(0.004 * len(foot))))
    cl = _clusters(leg[:, :2], 0.018 * h,
                   max(6, int(0.004 * len(leg))))
    if not cf or not cl:
        return None
    sx = sy = w = 0.0
    for f in cf[:8]:
        best, bd = None, None
        for l in cl[:8]:
            dd = ((l["c"][0] - f["c"][0]) ** 2 +
                  (l["c"][1] - f["c"][1]) ** 2)
            if bd is None or dd < bd:
                best, bd = l, dd
        if best is None or bd > (0.12 * h) ** 2:
            continue
        sx += (f["c"][0] - best["c"][0]) * f["n"]
        sy += (f["c"][1] - best["c"][1]) * f["n"]
        w += f["n"]
    if w <= 0:
        return None
    fx, fy = sx / w, sy / w
    if math.hypot(fx, fy) < 0.012 * h:
        return None
    return fx, fy


def face_rotation(pts):
    """Стадия 2: rot Z, чтобы лицо смотрело в +X.

    Основной способ — по носкам/пальцам лап (они торчат вперёд):
    знак поворота становится правильным в обоих направлениях
    (v2.4.0 у двуногих крутил вслепую на +90 — персонаж мог встать
    лицом назад). Запасной путь — старая эвристика по числу ног и
    глубине/длине корпуса."""
    z0 = float(pts[:, 2].min())
    z1 = float(pts[:, 2].max())
    h = z1 - z0
    fac = _facing_offset(pts, z0, h)
    if fac is not None:
        return math.degrees(math.atan2(-fac[1], fac[0]))
    band = pts[(pts[:, 2] >= z0 + 0.04 * h) &
               (pts[:, 2] <= z0 + 0.08 * h)]
    n_legs = 0
    cls = _clusters(band[:, :2], 0.025 * h, max(8, int(0.008 * len(band))))
    if cls:
        n_legs = len([c for c in cls if c["n"] >= 0.25 * cls[0]["n"]])
    mid = pts[(pts[:, 2] >= z0 + 0.35 * h) & (pts[:, 2] <= z0 + 0.75 * h)]
    if len(mid) < 20:
        mid = pts
    ex = float(np.ptp(mid[:, 0]))
    ey = float(np.ptp(mid[:, 1]))
    top = pts[pts[:, 2] >= z1 - 0.25 * h]
    rz = 0.0
    if n_legs >= 4:                        # спайн горизонтален
        if ey > ex:                        # длина вдоль Y -> на X
            rz = -90.0 if float(top[:, 1].mean()) >= 0 else 90.0
        else:                              # длина вдоль X
            cx = float(mid[:, 0].mean())
            if float(top[:, 0].mean()) < cx - 0.04 * h:
                rz = 180.0                 # голова в -X -> развернуть
    else:                                  # двуногий: глубина -> X
        if ey < ex:                        # глубина вдоль Y
            rz = 90.0
    return rz


# ------------------------------------------------------------- ноги

def _leg_paths(pts, z0, h, yc):
    """Ноги: кластеры у земли -> медиальные пути вверх до слияния с телом.

    Возвращает список dict(side, r, ankle, hip, knee, toe, path, top_z).
    Считает ЛЮБОЕ число конечностей (2, 4, 6...).
    """
    zA, zB = z0 + 0.035 * h, z0 + 0.08 * h
    band = pts[(pts[:, 2] >= zA) & (pts[:, 2] <= zB)]
    if len(band) < 30:
        raise CreatureError("нижняя станция пуста — меш не стоит на земле")
    cls = _clusters(band[:, :2], 0.022 * h, max(8, int(0.006 * len(band))))
    if not cls:
        raise CreatureError("у земли не найдено ни одной конечности")
    top_n = cls[0]["n"]
    cls = [c for c in cls if c["n"] >= 0.22 * top_n]
    if len(cls) > 12:
        raise CreatureError("у земли %d кластеров — мусор или не существо"
                            % len(cls))
    legs = []
    for c in cls:
        r = max(0.5 * min(c["sp"]), 0.008 * h)
        ankle = np.array([c["c"][0], c["c"][1], zA])
        path = [tuple(ankle)]
        cur = np.array(c["c"])
        z = zA
        merge_n = max(25, int(1.3 * c["n"]))
        hip = None
        while z < z0 + 0.80 * h:
            z += 0.02 * h
            sel = pts[(np.abs(pts[:, 2] - z) < 0.012 * h) &
                      (((pts[:, 0] - cur[0]) ** 2 +
                        (pts[:, 1] - cur[1]) ** 2) < (2.4 * r) ** 2)]
            if len(sel) < 5:
                break
            sp = max(float(np.ptp(sel[:, 0])),
                     float(np.ptp(sel[:, 1])))
            if len(sel) >= merge_n or sp > 4.0 * r:
                hip = np.array([float(cur[0]), float(cur[1]), float(z)])
                break
            cur = np.array([float(sel[:, 0].mean()),
                            float(sel[:, 1].mean())])
            path.append((float(cur[0]), float(cur[1]), float(z)))
        if hip is None:
            hip = np.array(path[-1])
        path.append(tuple(hip))
        # колено: точка пути с максимальным отклонением от хорды
        P = np.array(path)
        a, b = P[0], P[-1]
        ab = b - a
        L2 = max(float(ab @ ab), 1e-12)
        t = ((P - a) @ ab) / L2
        dev = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        window = (t >= 0.30) & (t <= 0.70)
        if bool(window.any()) and float(dev[window].max()) > 0.02 * h:
            knee = P[int(np.argmax(np.where(window, dev, -1.0)))]
        else:
            knee = 0.5 * (a + b)
        # стопа: передняя кромка низа ноги
        foot = pts[(pts[:, 2] < z0 + 0.05 * h) &
                   (np.linalg.norm(pts[:, :2] - ankle[:2], axis=1) <
                    3.0 * r)]
        if len(foot) >= 10:
            toe = np.array([float(foot[:, 0].max()), float(ankle[1]),
                            z0 + 0.025 * h])
        else:
            toe = ankle + np.array([0.5 * r, 0.0, 0.0])
        legs.append({
            "r": float(r), "path": [tuple(p) for p in P],
            "ankle": ankle, "hip": hip, "knee": knee, "toe": toe,
            "top_z": float(hip[2]),
        })
    return legs


def _assign_sides(legs, yc):
    """Стороны L/R по плоскости y = yc.

    Если плоскость симметрии сползла и ВСЕ ноги легли по одну
    сторону — делим по самим ногам (медиана y): иначе у пары ног
    будут одинаковые суффиксы и имена костей задвоятся
    ( CreatureError «дубликаты имён: ['leg_L_upper']» в v2.4.0 )."""
    if not legs:
        return legs
    ys = [float(l["hip"][1]) for l in legs]
    if all(y >= yc for y in ys) or all(y < yc for y in ys):
        s = sorted(ys)
        yc = 0.5 * (s[0] + s[-1]) if len(s) == 2 else s[len(s) // 2]
    for l in legs:
        l["side"] = +1 if float(l["hip"][1]) >= yc else -1
    return legs


def _pair_by_mirror(items, yc, tol):
    """Сопоставить объекты зеркально относительно плоскости y = yc.

    Возвращает (pairs, singles): pairs — [(a, b)] с обеими массами,
    singles — непарные."""
    left = [o for o in items if float(o["c"][1]) >= yc]
    right = [o for o in items if float(o["c"][1]) < yc]
    pairs, singles, used = [], [], set()
    for a in left:
        ay = float(a["c"][1])
        mirror = 2.0 * yc - ay
        best, bd = None, None
        for i, b in enumerate(right):
            if i in used:
                continue
            d = abs(float(b["c"][1]) - mirror)
            if bd is None or d < bd:
                best, bd = i, d
        if best is not None and bd <= tol:
            used.add(best)
            pairs.append((a, right[best]))
        else:
            singles.append(a)
    singles.extend(right[i] for i in range(len(right)) if i not in used)
    return pairs, singles


# -------------------------------------------- торс, руки/крылья, хвост, шея

def _torso_profile(pts, legs, z0, h):
    """Ядро тела: центральный кластер по срезам выше ног.

    Возвращает dict(path=[(x,y,z)] снизу вверх, zs, top_z, belly_w)."""
    z_leg_top = max(l["top_z"] for l in legs) if legs else z0 + 0.3 * h
    z = z_leg_top - 0.06 * h
    z_hi = float(pts[:, 2].max())
    path, zs = [], []
    prev = None
    while z <= z_hi:
        sl = pts[np.abs(pts[:, 2] - z) < 0.012 * h]
        zc = z
        z += 0.015 * h
        if len(sl) < 8:
            continue
        if prev is None:
            c = (float(sl[:, 0].mean()), float(sl[:, 1].mean()))
        else:
            cand = _clusters(sl[:, :2], 0.03 * h, 5)
            if cand:
                cand.sort(key=lambda d: (d["c"][0] - prev[0]) ** 2 +
                          (d["c"][1] - prev[1]) ** 2)
                c = cand[0]["c"]
            else:
                c = (float(sl[:, 0].mean()), float(sl[:, 1].mean()))
        prev = c
        path.append((c[0], c[1], float(zc)))
        zs.append(float(zc))
    if len(path) < 4:
        raise CreatureError("ядро тела не найдено")
    # ширина торса: 70-й процентиль полуширины |y - yc| по срезам
    # (медиана ломается на длинной шее: её срезов больше, чем срезов торса)
    ws = []
    for (px, py, pz) in path:
        sl = pts[np.abs(pts[:, 2] - pz) < 0.012 * h]
        if len(sl) >= 8:
            ws.append(float(np.abs(sl[:, 1] - py).max()))
    belly_w = float(np.percentile(ws, 70)) if ws else 0.1 * h
    belly_w = max(belly_w, 0.02 * h)
    return {"path": path, "zs": zs, "top_z": zs[-1], "belly_w": belly_w}


def _blob_flatness(P, attach=None):
    """Плоскость сечения СРЕДНЕЙ части массы: min/max поперечные
    размахи перпендикулярно главной оси (0..1).

    Мерен средний сегмент (30-70% длины от крепления):
    у настоящего крыла там перепонка (<0.25), у руки — круглое
    предплечье (~0.4-0.9). Дисталь брать нельзя — растопыренная
    кисть плоская, как перепонка (v2.5.0 из-за этого терял руку);
    корень тоже нельзя — плечо крыла круглое (птица считалась
    рукой). НЕ зависит от направления конечности: T-pose рука
    тонка по Z, как крыло, но сечение круглое (v2.4.0 мерил
    тонкость по Z и превращал T-pose человека в biped_wings —
    терялись пальцы и human-контракт)."""
    P = np.asarray(P, dtype=float)
    d = _cov_axis(P)
    ctr = P.mean(axis=0)
    t = (P - ctr) @ d
    if attach is not None:
        a = np.asarray(attach, dtype=float)
        if float(d @ (ctr - a)) < 0:       # d — наружу от крепления
            d = -d
            t = -t
    t_min, t_max = float(t.min()), float(t.max())
    span = max(t_max - t_min, 1e-9)
    rel = (t - t_min) / span
    mid = P[(rel >= 0.30) & (rel <= 0.70)]
    if len(mid) < 20:
        mid = P
    d2 = _cov_axis(mid)
    u = _perp(d2)
    v = np.cross(d2, u)
    a2 = float(np.ptp(mid @ u))
    b2 = float(np.ptp(mid @ v))
    return min(a2, b2) / max(a2, b2, 1e-6)


def _classify_lateral(blobs, torso, h, yc, n_legs):
    """Латеральные массы -> (arms, wings). Каждая масса — отдельный
    dict (пары не склеиваются: левая и правая строятся независимо)."""
    arms, wings = [], []
    pairs, singles = _pair_by_mirror(blobs, yc, 0.10 * h)
    cand = []
    for a, b in pairs:
        cand += [a, b]
    cand += singles
    for bl in cand:
        P = bl["P"]
        ext = bl["ext"]
        main = max(ext)
        # крепление: точка блока ближе всего к плоскости симметрии
        attach = P[int(np.argmin(np.abs(P[:, 1] - yc)))]
        flat = _blob_flatness(P, attach)
        # крыло = плоская перепонка; рука = круглое сечение
        is_wing = (n_legs >= 4) or (flat < 0.25 and main > 0.30 * h)
        zi = min(range(len(torso["zs"])),
                 key=lambda i: abs(torso["zs"][i] - float(attach[2])))
        sx = torso["path"][zi][0]
        if float(attach[0]) < sx - 0.06 * h:   # крепится за спиной
            is_wing = True
        entry = {"c": bl["c"], "P": P, "dir": bl["dir"], "ext": ext,
                 "n": bl["n"], "attach": attach, "flat": flat,
                 "side": +1 if float(bl["c"][1]) >= yc else -1}
        (wings if is_wing else arms).append(entry)
    return arms, wings


def _one_per_side(entries):
    """Оставить самую крупную массу на каждую сторону (L/R)."""
    by = {}
    for e in entries:
        cur = by.get(e["side"])
        if cur is None or e["n"] > cur["n"]:
            by[e["side"]] = e
    return list(by.values())


def _limb_chain(P, attach, h):
    """Станции цепочки вдоль главной оси облака P от attach наружу.
    Возвращает (stations [(point, r)], dir).

    Ось разворачивается НАРУЖУ (от крепления к центру масс облака):
    v2.4.0 разворачивал наоборот (attach - mean), ось смотрела внутрь,
    t уходил в минус, L = percentile97(t) схлопывался и все 4 станции
    цепи слипались у плеча — руки/крылья без пальцев и с костями-
    точками."""
    d = _cov_axis(P)
    if float(d @ (P.mean(axis=0) - np.asarray(attach, dtype=float))) < 0:
        d = -d
    rel = P - np.asarray(attach, dtype=float)
    t = rel @ d
    perp = np.linalg.norm(rel - t[:, None] * d, axis=1)
    L = float(np.percentile(t, 97))
    stations = []
    for f in (0.12, 0.40, 0.70, 0.96):
        tm = f * L
        sel = np.abs(t - tm) < 0.08 * max(L, 0.02 * h)
        if int(sel.sum()) >= 5:
            c = np.asarray(attach) + d * tm
            r = 0.8 * float(np.median(perp[sel]))
        else:
            c = np.asarray(attach) + d * tm
            r = 0.02 * h
        stations.append((tuple(float(v) for v in c),
                         max(float(r), 0.008 * h)))
    return stations, d


def _split_1d(vals, min_gap):
    """Разрезать отсортированные значения на группы по разрывам
    больше min_gap (1D-кластеризация без сетки — не мостится
    случайными точками). Возвращает списки индексов."""
    order = np.argsort(vals)
    v = vals[order]
    cuts = np.where(np.diff(v) > min_gap)[0]
    return np.split(order, cuts + 1)


def _count_fingers(P, p_root, p_tip, h):
    """Число пальцев: стойкие кластеры в сечениях дистальной части.

    Возвращает (n, fingers): fingers — пути [(p_a, p_b, p_tip)]."""
    p_root = np.asarray(p_root, dtype=float)
    p_tip = np.asarray(p_tip, dtype=float)
    d = p_tip - p_root
    L = float(np.linalg.norm(d))
    if L < 1e-6 or len(P) < 30:
        return 0, []
    d = d / L
    u = _perp(d)
    v = np.cross(d, u)
    rel = P - p_root
    t = rel @ d
    tracks = {}
    for f in (0.35, 0.55, 0.75, 0.93):
        sel = np.abs(t - f * L) < 0.06 * max(L, 0.02 * h)
        if int(sel.sum()) < 10:
            continue
        Pl = np.column_stack([rel[sel] @ u, rel[sel] @ v])
        wu, wv = float(np.ptp(Pl[:, 0])), float(np.ptp(Pl[:, 1]))
        ax = 0 if wu >= wv else 1
        w = max(wu, wv, 1e-6)
        # пальцы разделяются по доминирующей поперечной оси кисти
        cls = []
        for g in _split_1d(Pl[:, ax], 0.10 * w):
            if len(g) < max(3, int(0.08 * len(Pl))):
                continue
            cu = float(Pl[g, 0].mean())
            cv = float(Pl[g, 1].mean())
            cls.append({"c": (cu, cv), "n": len(g)})
        # глобальное жадное паросочетание: ближайшие (кластер, трек)
        # соединяются первыми — пальцы не перепрыгивают между треками
        pairs = []
        for ci, c in enumerate(cls):
            for tid, tr in tracks.items():
                px, py = tr[-1][1]
                dd = (px - c["c"][0]) ** 2 + (py - c["c"][1]) ** 2
                if dd <= (0.35 * w) ** 2:
                    pairs.append((dd, ci, tid))
        pairs.sort()
        used_c, used_t = set(), set()
        for dd, ci, tid in pairs:
            if ci in used_c or tid in used_t:
                continue
            used_c.add(ci)
            used_t.add(tid)
            c = cls[ci]
            pt3 = p_root + d * (f * L) + u * c["c"][0] + v * c["c"][1]
            tracks[tid].append((f, (c["c"][0], c["c"][1]), pt3))
        for ci, c in enumerate(cls):
            if ci in used_c:
                continue
            pt3 = p_root + d * (f * L) + u * c["c"][0] + v * c["c"][1]
            tracks[len(tracks)] = [(f, (c["c"][0], c["c"][1]), pt3)]
    fingers = []
    for tr in tracks.values():
        if len(tr) >= 3:
            pts = sorted(tr, key=lambda e: e[0])
            fingers.append([pts[0][2], pts[len(pts) // 2][2],
                            pts[-1][2]])
    return len(fingers), fingers


def _blob_behind(pts, torso, legs, z0, h, z_limit, excl_pts):
    """Хвост: масса позади задней кромки ядра торса, крепящаяся ниже
    низа головы (иначе затылок человека считается хвостом)."""
    x_rear = min(p[0] for p in torso["path"])
    sel = pts[(pts[:, 0] < x_rear - 0.01 * h) &
              (pts[:, 2] > z0 + 0.20 * h) &
              (pts[:, 2] < z_limit)]
    sel = _exclude_by_voxels(sel, excl_pts, 0.03 * h)
    for l in legs:
        if len(sel):
            sel = sel[_dist_to_path(sel, l["path"]) > 2.2 * l["r"]]
    if len(sel) < 40:
        return None
    bl = _blobs3d(sel, 0.025 * h, max(20, int(0.003 * len(pts))))
    if not bl:
        return None
    b = bl[0]
    if float(b["c"][0]) > x_rear + 0.02 * h or \
            max(b["ext"]) < 0.12 * h or abs(float(b["dir"][0])) < 0.5:
        return None
    return b


def _exclude_by_voxels(P, excl, cell):
    """Убрать из P точки, чьи воксели заняты точками excl (крылья/
    руки не должны попадать в поиск головы)."""
    if excl is None or len(excl) == 0 or len(P) == 0:
        return P
    keys = set(map(tuple, np.floor(excl / cell).astype(np.int64)))
    mask = np.array([tuple(k) not in keys for k in
                     np.floor(P / cell).astype(np.int64)])
    return P[mask]


def _head_neck(pts, torso, h, excl_pts):
    """Голова (масса у верхней точки) и шея (путь ядра до головы)."""
    top = pts[pts[:, 2] >= torso["top_z"] - 0.24 * h]
    top = _exclude_by_voxels(top, excl_pts, 0.03 * h)
    if len(top) < 15:
        return None
    bl = _blobs3d(top, 0.03 * h, max(12, int(0.002 * len(pts))))
    if not bl:
        return None
    zmax = float(pts[:, 2].max())
    head = bl[0]
    for b in bl:
        if float(b["c"][2]) + 0.5 * b["ext"][2] > zmax - 0.06 * h:
            head = b
            break
    P = head["P"]
    head_r = max(0.5 * min(head["ext"]) * 1.15, 0.015 * h)
    head_c = np.asarray(head["c"], dtype=float)
    z_hb = float(P[:, 2].min())
    neck_path = [(px, py, pz) for (px, py, pz) in torso["path"]
                 if pz >= z_hb - 0.03 * h]
    neck_len = 0.0
    if neck_path:
        nz = [p[2] for p in neck_path]
        neck_len = float(max(nz) - min(nz))
    return {"head_c": head_c, "head_r": float(head_r), "head_P": P,
            "neck_path": neck_path, "neck_len": neck_len,
            "z_bottom": z_hb}


# ------------------------------------------------------- сборка спеки

def _capsule(a, b, r, axis_hint=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    L = float(np.linalg.norm(d))
    if axis_hint is None:
        ax = max(range(3), key=lambda i: abs(d[i]))
        axis = "xyz"[ax] if L > 2.0 * r else "z"
    else:
        axis = axis_hint
    return {"type": "capsule", "axis": axis, "radius": round(r, 5),
            "half_len": round(max(0.5 * L - r, 0.0), 5)}


def _link(name, parent, anchor, com, r, d, shape=None):
    return {
        "name": name, "parent": parent,
        "joint": None if parent is None else
        {"anchor": [round(float(v), 5) for v in anchor],
         "axis": _joint_axis(d)},
        "com": [round(float(v), 5) for v in com],
        "shape": shape or _capsule(
            com if parent is None else anchor, com, r),
    }


def _chain_spec(out, names, anchors, coms, radii, dirs, base_parent=None):
    """Последовательные звенья: anchors[i] — точка крепления звена i
    к предыдущему (для i=0 — точка на base_parent)."""
    prev = base_parent
    for i, nm in enumerate(names):
        anchor = anchors[i] if (i > 0 or base_parent is not None) \
            else coms[0]
        out.append(_link(nm, prev, anchor, coms[i], radii[i], dirs[i]))
        prev = nm


# --------------------------------------------------------------- анализ

def analyze(pts, base_spec=None, plan="AUTO", add_fingers=True):
    """Точки меша (Z-up, мировые) -> результат авто-рига.

    plan: AUTO | HUMAN | QUADRUPED | WINGED_QUAD | BIRD | DRAGON.
    Возвращает dict(plan, spec, h, report, joints, notes).
    plan результата: human | biped | bird | biped_wings | biped_tail |
    quadruped | winged_quad | creature.
    """
    if np is None:
        raise CreatureError("numpy недоступен")
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 60:
        raise CreatureError("слишком мало вершин (%d)" % len(pts))
    z0 = float(pts[:, 2].min())
    z1 = float(pts[:, 2].max())
    h = z1 - z0
    if h < 0.05:
        raise CreatureError("высота меша %.3f м" % h)
    yc = _sym_yc(pts)
    notes = []
    joints = []

    legs = _leg_paths(pts, z0, h, yc)
    legs = _assign_sides(legs, yc)
    n_legs = len(legs)
    for l in legs:
        joints += [("hip", *l["hip"]), ("knee", *l["knee"]),
                   ("ankle", *l["ankle"])]

    torso = _torso_profile(pts, legs, z0, h)

    # латеральные массы (руки/крылья) за пределами ширины торса
    z_hip_line = min(l["top_z"] for l in legs)
    lat = pts[(pts[:, 2] > z_hip_line - 0.02 * h) &
              (np.abs(pts[:, 1] - yc) > torso["belly_w"] + 0.018 * h)]
    for l in legs:
        if len(lat):
            lat = lat[_dist_to_path(lat, l["path"]) > 2.2 * l["r"]]
    blobs = []
    if len(lat) >= 30:
        blobs = _blobs3d(lat, 0.028 * h, max(20, int(0.0035 * len(pts))))
    blobs = [b for b in blobs if max(b["ext"]) > 0.10 * h]
    arms_raw, wings_raw = _classify_lateral(blobs, torso, h, yc, n_legs)
    arms = _one_per_side(arms_raw)
    wings = _one_per_side(wings_raw)
    for w in (arms + wings):
        joints.append(("attach", *w["attach"][:3]))

    excl = None
    if arms or wings:
        excl = np.vstack([w["P"] for w in (arms + wings)])
    head = _head_neck(pts, torso, h, excl)
    # хвост: крепление должно быть ниже низа головы (иначе затылок
    # человека попадает в «массу позади таза»)
    z_limit = (float(head["z_bottom"]) - 0.05 * h) if head \
        else z0 + 0.95 * h
    tail = _blob_behind(pts, torso, legs, z0, h, z_limit, excl)

    tail_yes = tail is not None
    neck_long = bool(head and head["neck_len"] > 0.16 * h)

    # ----------------------------------------- человек? -> контракт 18
    if plan in ("AUTO", "HUMAN") and n_legs == 2 and not wings \
            and not tail_yes:
        try:
            from . import neuro_autorig
        except ImportError:
            import neuro_autorig
        lm = None
        try:
            lm = neuro_autorig.analyze_humanoid(pts)
        except neuro_autorig.RigFitError:
            lm = None
        if lm is not None and base_spec is not None:
            try:
                fitted = neuro_autorig.build_fitted_spec(base_spec, lm)
            except neuro_autorig.RigFitError as e:
                # человек распознан, но контракт не собрался — не валим
                # весь Auto-Rig: строим универсального двуногого
                notes.append("human-контракт не собрался (%s)" % e)
            else:
                return {
                    "plan": "human", "spec": fitted,
                    "h": float(lm["height"]),
                    "report": ("План: человек — контракт 18 костей, мышцы "
                               "и анимация neurosim доступны"),
                    "joints": joints, "notes": notes,
                }

    if plan == "HUMAN":
        raise CreatureError("человек не распознан (план задан "
                            "принудительно)")

    # ----------------------------------------- универсальная спека
    out = []
    hip_z = float(np.mean([l["top_z"] for l in legs]))
    hip_pts = [l["hip"] for l in legs]
    pelvis_c = np.array([float(np.mean([p[0] for p in hip_pts])),
                         float(np.mean([p[1] for p in hip_pts])),
                         hip_z])
    spine_path = [p for p in torso["path"] if p[2] > hip_z - 0.05 * h]
    if not spine_path:
        spine_path = torso["path"]
    r_spine = max(0.6 * torso["belly_w"], 0.02 * h)
    s_names = ["pelvis", "spine", "chest"]
    s_anchor, s_com, s_r, s_d = [], [], [], []
    prev_pt = pelvis_c
    for i, f in enumerate((0.0, 0.45, 0.92)):
        idx = min(len(spine_path) - 1, int(f * (len(spine_path) - 1)))
        pt = pelvis_c if i == 0 else \
            np.asarray(spine_path[idx], dtype=float)
        s_com.append((float(pt[0]), float(pt[1]), float(pt[2])))
        s_anchor.append(tuple(prev_pt))
        s_r.append(r_spine * (1.1 if i == 0 else 1.0))
        s_d.append(pt - prev_pt if i > 0 else np.array([0.0, 0.0, 1.0]))
        prev_pt = pt
    _chain_spec(out, s_names, s_anchor, s_com, s_r, s_d)

    # голова/шея на грудь
    if head:
        head_c = head["head_c"]
        joints.append(("head", head_c[0], head_c[1], head_c[2]))
        hp = sorted(head["head_P"], key=lambda p: -p[2])[
            :max(3, int(0.2 * len(head["head_P"])))]
        head_top = np.mean(np.asarray(hp), dtype=float, axis=0)
        r_n = max(0.55 * torso["belly_w"], 0.012 * h)
        base = np.asarray(s_com[-1], dtype=float)
        if neck_long and len(head["neck_path"]) >= 2:
            npath = head["neck_path"]
            n2 = np.asarray(npath[-1], dtype=float)
            if head["neck_len"] > 0.22 * h and len(npath) >= 5:
                n1 = np.asarray(
                    npath[int(0.5 * (len(npath) - 1))], dtype=float)
                _chain_spec(out, ["neck_1", "neck_2"],
                            [base, n1], [n1, n2],
                            [r_n, r_n * 0.9], [n1 - base, n2 - n1],
                            base_parent="chest")
                last, last_pt = "neck_2", n2
            else:
                _chain_spec(out, ["neck_1"], [base], [n2], [r_n],
                            [n2 - base], base_parent="chest")
                last, last_pt = "neck_1", n2
            out.append(_link("head", last, last_pt, head_c,
                             head["head_r"], head_top - last_pt,
                             _capsule(last_pt, head_c,
                                      head["head_r"], "z")))
        else:
            out.append(_link("head", "chest", base, head_c,
                             head["head_r"], head_c - base,
                             _capsule(base, head_c, head["head_r"],
                                      "z")))

    # ноги: upper -> lower -> foot (любое число, пары по x-рангу)
    order = sorted(range(n_legs), key=lambda i: -float(legs[i]["hip"][0]))
    half = (n_legs + 1) // 2
    for rank, i in enumerate(order):
        l = legs[i]
        sfx = "L" if l["side"] > 0 else "R"
        if n_legs == 2:
            base = "leg"
        elif n_legs == 4:
            base = "leg_front" if rank < 2 else "leg_hind"
        else:
            base = "leg%d" % (rank // 2 + 1)
        hip_pt, knee, ankle, toe = l["hip"], l["knee"], l["ankle"], \
            l["toe"]
        r_u = max(l["r"] * 1.15, 0.01 * h)
        hip_anchor = np.array([pelvis_c[0], float(hip_pt[1]),
                               float(hip_pt[2])])
        un, ln, fn = ("%s_%s_upper" % (base, sfx),
                      "%s_%s_lower" % (base, sfx),
                      "%s_%s_foot" % (base, sfx))
        out.append(_link(un, "pelvis", hip_anchor, knee, r_u,
                         knee - hip_anchor,
                         _capsule(hip_anchor, knee, r_u)))
        out.append(_link(ln, un, knee, ankle, l["r"] * 0.85,
                         ankle - knee, _capsule(knee, ankle,
                                                l["r"] * 0.85)))
        foot_l = max(float(np.linalg.norm(toe - ankle)), 0.05 * h)
        out.append(_link(fn, ln, ankle, 0.5 * (ankle + toe), l["r"],
                         toe - ankle,
                         {"type": "box",
                          "half": [round(0.55 * foot_l, 5),
                                   round(max(l["r"], 0.015 * h), 5),
                                   round(max(0.3 * l["r"], 0.008 * h),
                                         5)]}))
        joints.append(("foot", float(toe[0]), float(toe[1]),
                       float(toe[2])))

    # руки/крылья: цепочка от груди наружу (+ пальцы)
    for kind, entries in (("arm", arms), ("wing", wings)):
        for w in entries:
            sfx = "L" if w["side"] > 0 else "R"
            st, d = _limb_chain(w["P"], w["attach"], h)
            ancs = [tuple(s_com[-1])] + [s[0] for s in st[:-1]]
            coms = [s[0] for s in st]
            rds = [s[1] for s in st]
            names = ["%s_%s_%d" % (kind, sfx, i + 1)
                     for i in range(len(st))]
            _chain_spec(out, names, ancs, coms, rds,
                        [d] * len(st), base_parent="chest")
            last_name = names[-1]
            tip = np.asarray(coms[-1], dtype=float)
            wr = np.asarray(coms[-2] if len(coms) > 1 else coms[0],
                            dtype=float)
            n_f, fpaths = _count_fingers(w["P"], wr, tip, h)
            if kind == "wing" and n_f < 3:
                n_f, fpaths = 0, []
            if n_f >= 3:
                r_f = max(0.3 * rds[-1], 0.005 * h)
                for fi, fp in enumerate(fpaths):
                    pprev = last_name
                    for seg in (0, 1):
                        nm = "fng_%s_%s_%d_%d" % (kind, sfx, fi + 1,
                                                  seg + 1)
                        a = np.asarray(fp[seg], dtype=float)
                        b = np.asarray(fp[seg + 1], dtype=float)
                        out.append(_link(nm, pprev, a, 0.5 * (a + b),
                                         r_f, b - a))
                        pprev = nm
            elif kind == "arm" and add_fingers and \
                    float(np.linalg.norm(
                        tip - np.asarray(w["attach"]))) > 0.06 * h:
                _generic_fingers(out, last_name, tip, d, rds[-1], h,
                                 sfx)
            joints.append((kind, float(tip[0]), float(tip[1]),
                           float(tip[2])))

    # хвост: цепочка назад от пелвиса
    if tail is not None:
        P = tail["P"]
        base = np.array([pelvis_c[0], pelvis_c[1],
                         float(P[:, 2].mean())])
        d = _cov_axis(P)
        if float(d[0]) > 0:
            d = -d                        # назад = -X
        rel = P - base
        t = rel @ d
        L = float(np.percentile(t, 97))
        r_t = max(0.5 * float(np.percentile(
            np.linalg.norm(rel - t[:, None] * d, axis=1), 80)),
            0.01 * h)
        n_t = 3 if L > 0.2 * h else 2
        prev = "pelvis"
        for i in range(n_t):
            c0 = base + d * (L * i / n_t)
            c1 = base + d * (L * (i + 1.0) / n_t)
            nm = "tail_%d" % (i + 1)
            out.append(_link(nm, prev, c0, c1,
                             r_t * (1.0 - 0.22 * i), d))
            prev = nm
        joints.append(("tail", *[float(v) for v in (base + d * L)]))

    if not out:
        raise CreatureError("не удалось собрать ни одного звена")
    names_all = [l["name"] for l in out]
    if len(set(names_all)) != len(names_all):
        raise CreatureError("дубликаты имён костей: %s" %
                            sorted({n for n in names_all
                                    if names_all.count(n) > 1})[:4])
    known = set(names_all)
    for l in out:
        if l["parent"] is not None and l["parent"] not in known:
            raise CreatureError("звено %s: неизвестный родитель %s"
                                % (l["name"], l["parent"]))

    plan_id = "creature"
    if wings and n_legs == 4:
        plan_id = "winged_quad"
    elif wings and n_legs == 2:
        plan_id = "bird" if not arms else "biped_wings"
    elif n_legs == 4:
        plan_id = "quadruped"
    elif n_legs == 2 and arms:
        plan_id = "biped_tail" if tail_yes else "biped"
    elif n_legs == 2:
        plan_id = "biped"
    if plan != "AUTO" and plan != plan_id.upper():
        notes.append("задан план %s, распознан %s" %
                     (plan.lower(), plan_id))

    rep = ("План: %s — ног %d, рук %d, крыльев %d, хвост %s, шея %s; "
           "костей %d%s" % (
               plan_id, n_legs, len(arms), len(wings),
               "да" if tail_yes else "нет",
               ("да, %d звена" % (3 if head and head["neck_len"] >
                                  0.22 * h else 2)) if neck_long
               else "короткая/нет",
               len(out),
               ("; " + "; ".join(notes)) if notes else ""))
    if plan_id != "human":
        rep += (". Мышцы X-Muscle и анимация neurosim — только для "
                "human-плана")
    return {"plan": plan_id,
            "spec": {"name": "creature_v1", "root": "pelvis",
                     "links": out},
            "h": h, "report": rep, "joints": joints, "notes": notes}


def _generic_fingers(out, parent_name, tip, d, r, h, sfx="L"):
    """Стандартные 5 пальцев веером у кончика кисти (когда геометрия
    не разделила пальцы)."""
    d = np.asarray(d, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    u = _perp(d)
    v = np.cross(d, u)
    L = 0.036 * h
    for fi in range(5):
        ang = -0.9 + 1.8 * fi / 4.0
        base = np.asarray(tip, dtype=float) - d * (0.5 * L) + \
            (u * math.cos(ang) + v * math.sin(ang)) * (1.6 * r)
        mid = base + d * (0.55 * L)
        end = base + d * L
        nm1 = "fng_%s_std_%d_1" % (sfx, fi + 1)
        nm2 = "fng_%s_std_%d_2" % (sfx, fi + 1)
        out.append(_link(nm1, parent_name, base, mid,
                         max(0.25 * r, 0.004 * h), mid - base))
        out.append(_link(nm2, nm1, mid, end,
                         max(0.2 * r, 0.003 * h), end - mid))


# -------------------------------------------------------- landmarks

def make_landmarks(joints, h):
    """Пустышки-маркеры найденных точек (визуальный контроль)."""
    outs = []
    seen = {}
    for item in joints:
        name, x, y, z = item[0], item[1], item[2], item[3]
        seen[name] = seen.get(name, 0) + 1
        nm = "ncp_lm_%s" % name if seen[name] == 1 else \
            "ncp_lm_%s_%d" % (name, seen[name])
        e = bpy.data.objects.get(nm)
        if e is None:
            e = bpy.data.objects.new(nm, None)
            bpy.context.scene.collection.objects.link(e)
        e.empty_display_type = "SPHERE"
        e.empty_display_size = max(0.012 * h, 0.01)
        e.location = (x, y, z)
        outs.append(e)
    return outs
