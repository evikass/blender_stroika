# -*- coding: utf-8 -*-
"""
neuro_character_pipeline.ncp_img3d — мост к аддону Image to 3D (image_to_3d).

Аддон сам держит сервер-подпроцесс и импортирует результат (GLB).
Мы: резолвим его модуль, проверяем готовность, выставляем его scene-пропсы,
запускаем bpy.ops.img3d.generate() и потом ловим появившиеся меши.
"""
import bpy
import sys


def resolve():
    """Модуль image_to_3d (legacy или bl_ext.*), None если не включён."""
    try:
        prefs = bpy.context.preferences.addons
    except Exception:
        return None
    for k in prefs.keys():
        if k == "image_to_3d" or k.endswith(".image_to_3d"):
            return sys.modules.get(k)
    return None


def installed():
    return resolve() is not None


def scene_ready():
    return hasattr(bpy.context.scene, "img3d")


def readiness(mod, engine):
    """(ready, software_ok, missing_groups)."""
    try:
        sw = bool(mod.software_ready(engine))
        miss = list(mod.models_missing(engine) or [])
        return (sw and not miss, sw, miss)
    except Exception as e:
        return (False, False, [str(e)])


def apply_settings(context, image_path):
    """Пропатрить scene.img3d нашими настройками перед генерацией."""
    st = context.scene.ncp
    s3 = context.scene.img3d
    s3.image_path = image_path
    s3.engine = st.engine
    s3.quality = st.quality
    if st.engine == "HUNYUAN":
        s3.faces_hy = int(st.faces)
    else:
        s3.faces = int(st.faces)
    s3.use_texture = "ON"
    s3.texture = st.texture
    s3.seed = int(st.seed)


def start_generation(context, image_path):
    """Запуск генерации. Возвращает снимок имён объектов сцены.

    Кидает RuntimeError с понятным текстом, если аддон не готов.
    """
    mod = resolve()
    if mod is None:
        raise RuntimeError("Аддон Image to 3D не включён")
    if not scene_ready():
        raise RuntimeError("Image to 3D не зарегистрировал scene.img3d")
    ready, sw, miss = readiness(mod, context.scene.ncp.engine)
    if not ready:
        if not sw:
            raise RuntimeError("Движок Image to 3D не установлен "
                               "(нажмите Install Engine)")
        raise RuntimeError("Не скачаны модели: %s (нажмите Download)"
                           % ", ".join(miss))
    apply_settings(context, image_path)
    snapshot = frozenset(o.name for o in bpy.data.objects)
    try:
        bpy.ops.img3d.generate()
    except RuntimeError as e:
        raise RuntimeError("img3d.generate отклонён: %s" % e) from None
    return snapshot


def job_running(mod):
    """Идёт ли генерация (CARD_JOB не None)."""
    try:
        return mod.CARD_JOB is not None
    except Exception:
        return False


def new_meshes(snapshot):
    """Новые меши img3d, появившиеся после старта (по custom prop)."""
    out = []
    for o in bpy.data.objects:
        if (o.type == "MESH" and o.name not in snapshot
                and o.get("img3d_image")):
            out.append(o)
    return out


def start_download(missing_groups):
    mod = resolve()
    if mod is None:
        raise RuntimeError("Аддон Image to 3D не включён")
    ok = mod.dl.start(mod.engine_dir(), list(missing_groups))
    if not ok:
        raise RuntimeError("Загрузка уже идёт или не запустилась")
