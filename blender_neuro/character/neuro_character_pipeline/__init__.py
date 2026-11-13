# -*- coding: utf-8 -*-
"""
Neuro Character Pipeline — мастер-аддон, связывающий:
  ComfyUI (картинка) -> Image to 3D (меш) -> neuro-скелет (human.json)
  -> X-Muscle System (мышцы) -> neuro-анимация (motion JSON).

N-панель 3D-вьюпорта, категория "NCP" (короткая вкладка); резервная
панель — Properties -> Scene. Постоянные настройки (папка движка,
URL ComfyUI, python.exe для ComfyUI) — в AddonPreferences, не слетают.
"""
bl_info = {
    "name": "Neuro Character Pipeline",
    "author": "blender_neuro",
    "version": (2, 3, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > N-panel > NCP | Properties > Scene",
    "description": "All-in-One: картинка -> 3D-меш -> neuro-скелет -> "
                   "мышцы -> анимация (движки вшиты)",
    "category": "Rigging",
}

import json
import importlib
import os
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, PointerProperty, StringProperty)
from bpy_extras.io_utils import ImportHelper

if "ncp_comfy" in locals():  # reload при правках в текст-редакторе
    import importlib
    importlib.reload(ncp_comfy)
    importlib.reload(ncp_img3d)
    importlib.reload(ncp_muscles)
    importlib.reload(neuro_rig)
    importlib.reload(neuro_autorig)
else:
    from . import ncp_comfy
    from . import ncp_img3d
    from . import ncp_muscles
    from . import neuro_rig
    from . import neuro_autorig


# ------------------------------------------------------------ состояние

JOB = {
    "active": False,
    "phase": "",            # COMFY_IMG / COMFY_REFINE / MESH / MUSCLES
    "next": "",             # RIG / NONE
    "chain_muscles": False,
    "prompt_id": "",
    "t0": 0.0,
    "snapshot": frozenset(),
    "tries": 0,
    "comfy": None,
}


_NCP_BASENAME = (__package__ or "neuro_character_pipeline").rsplit(".", 1)[-1]

COMFY_DEFAULT = "http://127.0.0.1:8188"
_COMFY_CACHE = {"ts": 0.0, "ok": None}


def _get_prefs():
    """AddonPreferences этого аддона (работает и для bl_ext.*)."""
    try:
        addons = bpy.context.preferences.addons
    except Exception:
        return None
    for key in addons.keys():
        if key.rsplit(".", 1)[-1].lower() == _NCP_BASENAME.lower():
            try:
                return addons[key].preferences
            except Exception:
                pass
    return None


def _pref_str(name):
    p = _get_prefs()
    if p is None:
        return ""
    try:
        return (getattr(p, name, "") or "").strip()
    except Exception:
        return ""


def _comfy_url(st=None):
    url = _pref_str("comfy_url")
    if not url and st is not None:
        url = (getattr(st, "comfy_url", "") or "").strip()
    return url or COMFY_DEFAULT


def _port_open(host, port, timeout=0.6):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _comfy_alive_cached():
    now = time.time()
    if now - _COMFY_CACHE["ts"] > 6.0:
        u = urlparse(_comfy_url())
        _COMFY_CACHE["ts"] = now
        _COMFY_CACHE["ok"] = _port_open(u.hostname or "127.0.0.1",
                                        u.port or 8188)
    return _COMFY_CACHE["ok"]


def _out_dir(context):
    d = _pref_str("out_dir")
    if not d:
        st = getattr(context.scene, "ncp", None)
        if st is not None:
            d = (getattr(st, "out_dir", "") or "").strip()
    d = d or os.path.join(os.path.expanduser("~"), "NeuroCharacter")
    d = bpy.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def _st(context):
    return context.scene.ncp


def _finish(context, msg):
    JOB.update(active=False, phase="", next="", comfy=None)
    st = _st(context)
    st.state = "DONE"
    st.status = msg
    for wm in (bpy.context.window_manager, ):
        if wm:
            for w in wm.windows:
                for area in w.screen.areas:
                    area.tag_redraw()


def _fail(context, msg):
    JOB.update(active=False, phase="", next="", comfy=None)
    st = _st(context)
    st.state = "ERROR"
    st.status = msg


# --------------------------------------------------- vendor (All-in-One)

# Раздельные копии движков: конфликтуют с вшитыми (двойная регистрация).
# Матчим по подстроке имени модуля (регистронезависимо): у старых копий
# ключи могут быть любыми - image_to_3d, "Image_to_3D", bl_ext.*.xmusclesystem...
_LEGACY_HINTS = ("image_to_3d", "img3d", "xmuscle", "x_muscle", "xpiemenu")
VENDOR_STATUS = {}   # имя -> "" (ок) или текст ошибки регистрации


def _legacy_conflicts():
    """Включённые раздельные копии Image to 3D / X-Muscle / Pie Menu."""
    try:
        keys = list(bpy.context.preferences.addons.keys())
    except Exception:
        return []
    ours = (__package__ or "neuro_character_pipeline").rsplit(".", 1)[-1]
    ours = ours.lower()
    out = []
    for k in keys:
        base = k.rsplit(".", 1)[-1].lower()
        if base == ours:
            continue
        if any(h in base for h in _LEGACY_HINTS):
            out.append(k)
    return out


def _disable_legacy():
    """All-in-One сам глушит старые раздельные копии при включении."""
    import addon_utils
    for k in _legacy_conflicts():
        try:
            addon_utils.disable(k, default_set=True)
            print("[Neuro Pipeline] старая копия отключена:", k)
        except Exception as exc:
            print("[Neuro Pipeline] не удалось отключить", k, ":", exc)


def _register_vendor():
    """Поднять вшитые движки (img3d, xmuscle, xpiemenu)."""
    VENDOR_STATUS.clear()
    for name in ("img3d", "xmuscle", "xpiemenu"):
        try:
            mod = importlib.import_module(".vendor." + name, __package__)
            mod.register()
            VENDOR_STATUS[name] = ""
        except Exception as e:
            VENDOR_STATUS[name] = "%s: %s" % (type(e).__name__, e)
            print("[Neuro Pipeline] vendor.%s не поднялся: %s" % (name, e))


def _unregister_vendor():
    for name in ("xpiemenu", "xmuscle", "img3d"):
        try:
            mod = importlib.import_module(".vendor." + name, __package__)
            mod.unregister()
        except Exception:
            pass


def _vendor_warnings():
    """Строки для карточки: конфликты/ошибки вшитых движков."""
    out = []
    legacy = _legacy_conflicts()
    if legacy:
        out.append("Отключите старые раздельные аддоны: "
                   + ", ".join(legacy))
    for name, err in VENDOR_STATUS.items():
        if err:
            out.append("Вшитый %s не поднялся: %s" % (name, err))
    return out


def _start_timer():
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=0.4)


def _find_skin_mesh(context):
    """Меш кожи: последний ncp-меш -> Character_Mesh -> img3d-меш -> активный."""
    cands = [o for o in bpy.data.objects
             if o.type == "MESH" and o.get("ncp_mesh")]
    if cands:
        return cands[-1]
    o = bpy.data.objects.get("Character_Mesh")
    if o:
        return o
    cands = [o for o in bpy.data.objects
             if o.type == "MESH" and o.get("img3d_image")]
    if cands:
        return cands[-1]
    ao = context.view_layer.objects.active
    if ao is not None and ao.type == "MESH":
        return ao
    return None


def _clear_old_rig():
    old = bpy.data.objects.get("NeuroCharacter")
    for o in list(bpy.data.objects):
        if o.name.startswith("ncp_lm_"):
            bpy.data.objects.remove(o)          # старые landmarks-маркеры
    if old is None:
        return
    for o in list(bpy.data.objects):
        if o.parent == old:
            mw = o.matrix_world.copy()
            o.parent = None
            o.matrix_world = mw
            for md in list(o.modifiers):
                if md.type == "ARMATURE":
                    o.modifiers.remove(md)
    bpy.data.objects.remove(old)


def _build_rig(context, mesh):
    st = _st(context)
    if bpy.data.objects.get("NeuroCharacter") is not None:
        if not st.replace_existing:
            raise RuntimeError("Скелет NeuroCharacter уже есть "
                               "(отключите Replace или удалите его)")
        if ncp_muscles.muscle_objects():
            raise RuntimeError("У меша уже есть мышцы X-Muscle — "
                               "пересоздавать скелет нельзя")
        _clear_old_rig()
    spec = neuro_rig.load_spec(bpy.path.abspath(st.spec_path)
                               if st.spec_path else "")
    rot_x, rot_z = st.mesh_rot_x, st.mesh_rot_z
    if st.orient_auto:
        rot_x, rot_z = neuro_rig.guess_orientation(mesh)
        st.mesh_rot_x, st.mesh_rot_z = rot_x, rot_z
    if st.auto_rig:
        try:
            return _build_rig_fitted(context, mesh, spec, rot_x, rot_z)
        except neuro_autorig.RigFitError as e:
            st.status = ("Auto-Rig не удался (%s) — включён старый режим: "
                         "меш масштабируется под скелет 1.70 м" % e)
    arm = neuro_rig.build_armature(spec, name="NeuroCharacter")
    if st.add_fingers:
        neuro_rig.add_finger_chains(arm)
    neuro_rig.fit_mesh_to_skeleton(
        mesh, spec, rot_x_deg=rot_x, rot_z_deg=rot_z,
        do_apply=st.apply_transforms)
    neuro_rig.skin_mesh_to_armature(mesh, arm)
    if st.orient_auto and (abs(rot_x) > 1e-6 or abs(rot_z) > 1e-6):
        _st(context).status = ("Меш повёрнут (авто): X%+g°, Z%+g° — "
                               "если смотрит назад, поставьте rot Z = -90"
                               % (rot_x, rot_z))
    return arm


def _build_rig_fitted(context, mesh, spec, rot_x, rot_z):
    """Auto-Rig в стиле Hyper3D: скелет подгоняется под тело.

    Меш поворачивается (авто-ориентация) и ставится низом в z=0,
    размер НЕ меняется. По вершинам ищутся точки тела, из них
    собирается та же 18-костная спека (контракт human.json), арматура
    строится по ней, меш скинится как есть. При неудаче кидается
    RigFitError — вызывающий код откатывается на старый режим."""
    st = _st(context)
    neuro_autorig.place_mesh(mesh, rot_x_deg=rot_x, rot_z_deg=rot_z,
                             do_apply=st.apply_transforms)
    pts = neuro_autorig.mesh_world_points(mesh)
    fitted, lm, info = neuro_autorig.fit_spec_to_mesh(spec, pts)
    arm = neuro_rig.build_armature(fitted, name="NeuroCharacter")
    arm["ncp_height"] = float(lm["height"])
    if st.add_fingers:
        neuro_rig.add_finger_chains(arm, scale=lm["height"] / 1.70)
    neuro_rig.skin_mesh_to_armature(mesh, arm)
    if st.debug_landmarks:
        neuro_autorig.make_debug_empties(lm)
    note = ("; смотрит назад — rot Z = -90"
            if st.orient_auto and (abs(rot_x) > 1e-6 or abs(rot_z) > 1e-6)
            else "")
    st.status = info + note
    return arm


# --------------------------------------------------- шаги конвейера

def _start_comfy_job(context, refine, chain_next, chain_muscles):
    st = _st(context)
    url = _comfy_url(st)
    cli = ncp_comfy.ComfyClient(url)
    if not cli.check():
        raise ncp_comfy.ComfyError(
            "ComfyUI не отвечает на %s (кнопка Start ComfyUI — в секции "
            "Files & Settings, или запустите ComfyUI сами)" % url)
    wf = ncp_comfy.ComfyClient.load_workflow(
        bpy.path.abspath(st.comfy_workflow) if st.comfy_workflow else "")
    seed = st.seed if st.seed > 0 else ncp_comfy.random_seed()
    if refine:
        src = bpy.path.abspath(st.sketch_path)
        if not st.sketch_path or not os.path.isfile(src):
            raise ncp_comfy.ComfyError("Сначала задайте эскиз (sketch)")
        srv_name = cli.upload_image(src)
        wf = ncp_comfy.ComfyClient.fill_workflow(
            wf, prompt=st.prompt, negative=st.negative, seed=seed,
            image_name=srv_name)
    else:
        wf = ncp_comfy.ComfyClient.fill_workflow(
            wf, prompt=st.prompt, negative=st.negative, seed=seed)
    pid = cli.submit(wf)
    JOB.update(active=True, comfy=cli, prompt_id=pid, t0=time.time(),
               phase="COMFY_REFINE" if refine else "COMFY_IMG",
               next=chain_next, chain_muscles=chain_muscles, tries=0)
    st.state = JOB["phase"]
    st.status = "ComfyUI: workflow выполняется..."
    _start_timer()


def _start_mesh_job(context, chain_next, chain_muscles):
    st = _st(context)
    img = bpy.path.abspath(st.image_path)
    if not st.image_path or not os.path.isfile(img):
        raise RuntimeError("Картинка не найдена — сгенерируйте или выберите")
    snapshot = ncp_img3d.start_generation(context, img)
    JOB.update(active=True, phase="MESH", next=chain_next,
               chain_muscles=chain_muscles, snapshot=snapshot, tries=0,
               comfy=None)
    st.state = "MESH"
    st.status = "Image to 3D: генерация меша (первая — минуты)..."
    _start_timer()


def _tick():
    """Таймер-машина состояний: ComfyUI -> Mesh -> Rig -> Muscles."""
    context = bpy.context
    scn = getattr(context, "scene", None)
    st = getattr(scn, "ncp", None) if scn else None
    if st is None or not JOB["active"]:
        return None
    ph = JOB["phase"]
    try:
        if ph in ("COMFY_IMG", "COMFY_REFINE"):
            cli = JOB.get("comfy")
            if cli is None:
                raise RuntimeError("потерян клиент ComfyUI")
            state, data = cli.poll(JOB["prompt_id"], st.comfy_timeout,
                                   JOB["t0"])
            if state == "running":
                return 1.0
            if state == "error":
                raise ncp_comfy.ComfyError(data)
            if not data:
                raise ncp_comfy.ComfyError("ComfyUI не вернул изображений")
            saved = cli.fetch(data[0], _out_dir(context))
            st.image_path = saved
            if ph == "COMFY_REFINE":
                st.sketch_path = saved
            if JOB["next"] == "RIG":
                st.status = "Картинка готова: %s. Меш..." % os.path.basename(
                    saved)
                _start_mesh_job(context, chain_next="RIG",
                                chain_muscles=JOB["chain_muscles"])
            else:
                _finish(context, "Картинка готова: %s"
                        % os.path.basename(saved))

        elif ph == "MESH":
            mod = ncp_img3d.resolve()
            if mod is None:
                raise RuntimeError("Image to 3D отключился")
            if ncp_img3d.job_running(mod):
                JOB["tries"] = 0
                return 1.0
            new = ncp_img3d.new_meshes(JOB["snapshot"])
            if not new:
                JOB["tries"] += 1
                stat = getattr(scn.img3d, "status", "") if hasattr(scn,
                                                                   "img3d") \
                    else ""
                if JOB["tries"] > 24:
                    raise RuntimeError("Меш не появился. %s" % (stat or ""))
                return 0.5
            mesh = new[0]
            mesh["ncp_mesh"] = True
            mesh.name = "Character_Mesh"
            if JOB["next"] == "RIG":
                st.status = "Меш готов. Ставлю neuro-скелет..."
                _build_rig(context, mesh)
                if JOB["chain_muscles"]:
                    JOB["phase"] = "MUSCLES"
                    st.state = "MUSCLES"
                    st.status = "Скелет готов. Расставляю мышцы..."
                    return 0.5
                _finish(context, "Меш + neuro-скелет + весы готовы")
            else:
                _finish(context, "Меш готов: %s" % mesh.name)

        elif ph == "MUSCLES":
            arm = bpy.data.objects.get("NeuroCharacter")
            if arm is None:
                raise RuntimeError("Нет скелета NeuroCharacter")
            mesh = _find_skin_mesh(context)
            created, delta, errors = ncp_muscles.create_all(context, arm)
            if _st(context).attach_skin:
                ncp_muscles.attach_to_skin(context, mesh)
                ncp_muscles.enable_dynamics()
            msg = "Мышцы: создано %d, пришито к коже" % created
            if errors:
                msg += ". Пропуски: " + "; ".join(errors[:3])
            _finish(context, msg)
        else:
            _fail(context, "Неизвестная фаза %r" % ph)
    except Exception as e:
        _fail(context, "%s: %s" % (type(e).__name__, e))
        return None
    return None


# --------------------------------------------------------------- пропсы

def _refresh_scene():
    for w in bpy.context.window_manager.windows:
        for area in w.screen.areas:
            area.tag_redraw()


class NCP_Props(bpy.types.PropertyGroup):
    # --- шаг 1: эскиз
    sketch_path: StringProperty(
        name="Sketch", subtype="FILE_PATH",
        description="Эскиз персонажа (PNG). Можно нарисовать в Krita "
                    "или другом редакторе и уточнить через ComfyUI")
    krita_path: StringProperty(
        name="Krita exe", default="",
        description="Путь к krita.exe. Пусто = открыть системным "
                    "приложением по умолчанию")
    # --- шаг 2: картинка
    comfy_url: StringProperty(
        name="ComfyUI URL", default="http://127.0.0.1:8188",
        description="Адрес локального ComfyUI API")
    comfy_workflow: StringProperty(
        name="Workflow (API)", subtype="FILE_PATH",
        description="Workflow в API-формате: в ComfyUI включите Dev mode "
                    "(Settings -> Dev mode) и сохраните через "
                    "Save (API Format)")
    prompt: StringProperty(
        name="Prompt",
        default="full body character, T-pose, front view, "
                "white background",
        description="Промпт: подставляется в первую текстовую ноду "
                    "workflow (CLIPTextEncode)")
    negative: StringProperty(
        name="Negative", default="",
        description="Негативный промпт: вторая текстовая нода workflow")
    comfy_timeout: IntProperty(
        name="Timeout, s", default=900, min=30, max=7200,
        description="Сколько ждать генерации ComfyUI")
    image_path: StringProperty(
        name="Image", subtype="FILE_PATH",
        description="Итоговая картинка персонажа: результат ComfyUI "
                    "или любой PNG/JPG с диска")
    out_dir: StringProperty(
        name="Output dir", subtype="FILE_PATH", default="",
        description="Куда складывать результаты (пусто = "
                    "~/NeuroCharacter)")
    # --- шаг 3: меш
    engine: EnumProperty(
        name="Engine",
        items=[("TRELLIS", "TRELLIS", "TRELLIS 2 (по умолчанию)"),
               ("HUNYUAN", "Hunyuan3D", "Hunyuan3D 2.1")],
        default="TRELLIS",
        description="Движок Image to 3D")
    quality: EnumProperty(
        name="Quality",
        items=[("FAST", "Fast (512)", ""), ("HIGH", "High (1024)", ""),
               ("ULTRA", "Ultra (1536)", "")],
        default="HIGH", description="Качество генерации Image to 3D")
    faces: IntProperty(
        name="Faces", default=200000, min=1000,
        description="Цель по полигонам (Hunyuan обычно 40000)")
    texture: EnumProperty(
        name="Texture",
        items=[("1024", "1024", ""), ("2048", "2048", ""),
               ("4096", "4096", "")],
        default="2048", description="Размер текстуры")
    seed: IntProperty(
        name="Seed", default=0, min=0,
        description="0 = случайный (для ComfyUI); для Image to 3D "
                    "передаётся как есть")
    # --- шаг 4: скелет
    spec_path: StringProperty(
        name="Creature spec", subtype="FILE_PATH", default="",
        description="JSON-пресет существа blender_neuro. Пусто = "
                    "presets/human.json рядом с аддоном")
    mesh_rot_x: FloatProperty(
        name="Mesh rot X", default=0.0, min=-180.0, max=180.0,
        description="Доворот меша вокруг X (градусы), если персонаж "
                    "лежит/стоит не так")
    orient_auto: BoolProperty(
        name="Auto-orient", default=True,
        description="Определить по bbox: лежит ли меш и куда смотрит; "
                    "найденные углы запишутся в Mesh rot X/Z")
    mesh_rot_z: FloatProperty(
        name="Mesh rot Z", default=0.0, min=-180.0, max=180.0,
        description="Доворот меша вокруг Z (куда смотрит персонаж). "
                    "0 = смотрит +X, как neuro-скелет; если после "
                    "авто-определения смотрит назад — поставьте -90")
    apply_transforms: BoolProperty(
        name="Apply transforms", default=True,
        description="Запечь масштаб/поворот меша перед скинингом "
                    "(рекомендуется)")
    auto_rig: BoolProperty(
        name="Auto-Rig (как в Hyper3D)", default=True,
        description="Скелет подгоняется под тело: по геометрии меша "
                    "находятся промежность, колени, плечи, локти, шея — "
                    "и кости ставятся туда. Меш остаётся в своих "
                    "пропорциях. Выключить = старый режим (меш "
                    "масштабируется под скелет 1.70 м)")
    debug_landmarks: BoolProperty(
        name="Show landmarks", default=False,
        description="Создать пустышки-маркеры в найденных точках тела "
                    "(промежность, плечи, локти, шея...) — для контроля "
                    "подгонки скелета")
    replace_existing: BoolProperty(
        name="Replace skeleton", default=True,
        description="Удалить старый NeuroCharacter и пересоздать")
    add_fingers: BoolProperty(
        name="Add fingers", default=False,
        description="Добавить 20 костей пальцев (10 на руку) поверх "
                    "18 neuro-костей — для позирования/экспорта. "
                    "neurosim анимирует только 18 основных костей, "
                    "пальцы едут вместе с рукой")
    # --- шаг 5: мышцы
    muscle_style: EnumProperty(
        name="Style",
        items=[("STYLIZED", "Stylized", "Стилизованная мышца"),
               ("BASIC", "Basic", "Базовая форма"),
               ("STRIP", "Strip", "Лента (для длинных мышц)")],
        default="STYLIZED", description="Форма мышц X-Muscle")
    attach_skin: BoolProperty(
        name="Attach to skin", default=True,
        description="После создания пришить мышцы к коже "
                    "(shrinkwrap + веса + драйверы)")
    # --- шаг 6: анимация
    frame_step: IntProperty(
        name="Frame step", default=1, min=1, max=10,
        description="Брать каждый N-й кадр из записи neurosim")
    build_world: BoolProperty(
        name="Import world", default=False,
        description="Построить пропсы мира из motion-файла")
    # --- статус
    state: StringProperty(default="IDLE")
    status: StringProperty(default="")


# Ошибки регистрации классов (обычно вторая старая копия аддона):
# показываются прямо в панели, чтобы "пропавший интерфейс" был видим.
REGISTER_ERRORS = []


def _engine_dir_update(self, context):
    """Папка движка изменилась - продублировать в json-конфиг вшитого img3d."""
    try:
        from .vendor import img3d as _i3d
        _i3d._save_config()
    except Exception:
        pass


class NCP_AddonPrefs(bpy.types.AddonPreferences):
    """Постоянные настройки: не сбрасываются при закрытии Blender
    (в отличие от настроек на Scene, которые живут только в .blend)."""

    bl_idname = __package__ if __package__ else "neuro_character_pipeline"

    engine_dir: StringProperty(
        name="Engine & Models Folder", subtype="DIR_PATH", default="",
        description="Куда вшитый Image to 3D ставит движок и качает AI-"
                    "модели (около 25 ГБ). Пусто = ~/3DRedbox/ImageTo3D. "
                    "Это НЕ папка ComfyUI - у движка свой Python-venv",
        update=_engine_dir_update)
    comfy_url: StringProperty(
        name="ComfyUI URL", default=COMFY_DEFAULT,
        description="Адрес ComfyUI API. Хранится постоянно, не слетает "
                    "между сеансами")
    out_dir: StringProperty(
        name="Output dir", subtype="FILE_PATH", default="",
        description="Куда складывать результаты (пусто = ~/NeuroCharacter)")
    comfyui_dir: StringProperty(
        name="ComfyUI folder", subtype="DIR_PATH", default="",
        description="Папка ComfyUI (где лежит main.py) — нужна только "
                    "для кнопки Start ComfyUI")
    comfyui_python: StringProperty(
        name="Python (ComfyUI)", subtype="FILE_PATH", default="",
        description="python.exe для запуска ComfyUI: в портативной сборке "
                    "это <ComfyUI>\\python_embeded\\python.exe")
    comfyui_port: IntProperty(
        name="Port", default=8188, min=1024, max=65535,
        description="Порт автозапуска (должен совпадать с URL)")
    gentle_download: BoolProperty(
        name="Gentle Download", default=False,
        description="Для слабых компьютеров: качать по одному файлу и "
                    "медленнее, чтобы компьютер не подвисал во время "
                    "загрузки AI-моделей")

    def draw(self, context):
        lay = self.layout
        col = lay.column(align=True)
        col.prop(self, "engine_dir")
        row = col.row(align=True)
        row.operator("img3d.pick_engine_dir", text="Choose Install Folder",
                     icon="FILE_FOLDER")
        row.operator("img3d.dl_folder", text="Open Folder",
                     icon="FOLDER_REDIRECT")
        col = lay.column(align=True)
        col.label(text="Здесь лежат движок и модели Image to 3D (~25 ГБ)",
                  icon="INFO")
        col.label(text="Папка сохраняется навсегда и переживает "
                       "переустановку аддона", icon="CHECKMARK")
        col.prop(self, "gentle_download")
        col = lay.column(align=True)
        col.label(text="Постоянные настройки конвейера", icon="LOCKED")
        col.prop(self, "comfy_url")
        col.prop(self, "out_dir")
        box = lay.box()
        box.label(text="Автозапуск ComfyUI", icon="PLAY")
        box.prop(self, "comfyui_dir")
        box.prop(self, "comfyui_python")
        box.prop(self, "comfyui_port")
        box.operator("ncp.start_comfy", icon="PLAY")
        lay.operator("ncp.env_report", icon="TEXT")


# ----------------------------------------------------------- операторы

class NCP_OT_new_sketch(bpy.types.Operator):
    bl_idname = "ncp.new_sketch"
    bl_label = "New Sketch"
    bl_description = ("Создать чистый лист 1024x1024 в выходной папке — "
                      "для рисования эскиза персонажа в Krita/редакторе")
    bl_options = {"REGISTER"}

    def execute(self, context):
        path = os.path.join(_out_dir(context), "sketch.png")
        img = bpy.data.images.new("ncp_sketch_tmp", 1024, 1024, alpha=False)
        try:
            img.pixels.foreach_set([1.0] * (1024 * 1024 * 4))
            img.filepath_raw = path
            img.file_format = "PNG"
            img.save()
        finally:
            bpy.data.images.remove(img)
        _st(context).sketch_path = path
        self.report({"INFO"}, "Эскиз создан: %s" % path)
        return {"FINISHED"}


class NCP_OT_open_sketch(bpy.types.Operator):
    bl_idname = "ncp.open_sketch"
    bl_label = "Open in Krita"
    bl_description = ("Открыть эскиз во внешнем редакторе (Krita), "
                      "сохраните файл там же — аддон возьмёт его дальше")
    bl_options = {"REGISTER"}

    def execute(self, context):
        st = _st(context)
        path = bpy.path.abspath(st.sketch_path) if st.sketch_path else ""
        if not path or not os.path.isfile(path):
            bpy.ops.ncp.new_sketch()
            path = bpy.path.abspath(st.sketch_path)
        kp = st.krita_path.strip()
        try:
            if kp:
                subprocess.Popen([kp, path])
            elif sys.platform == "win32":
                os.startfile(path)  # noqa
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.report({"ERROR"}, "Не удалось открыть редактор: %s" % e)
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_refine_sketch(bpy.types.Operator):
    bl_idname = "ncp.refine_sketch"
    bl_label = "Refine via ComfyUI"
    bl_description = ("Превратить эскиз в рендер персонажа через ваш "
                      "ComfyUI-workflow (img2img/ControlNet): эскиз "
                      "уйдёт в ноду LoadImage, промпт — в CLIPTextEncode")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return not JOB["active"]

    def execute(self, context):
        try:
            _start_comfy_job(context, refine=True, chain_next="NONE",
                             chain_muscles=False)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_generate_image(bpy.types.Operator):
    bl_idname = "ncp.generate_image"
    bl_label = "Generate Image"
    bl_description = ("Сгенерировать картинку персонажа в ComfyUI: "
                      "workflow + промпт уходят на сервер, результат "
                      "скачивается в Output dir")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return not JOB["active"]

    def execute(self, context):
        try:
            _start_comfy_job(context, refine=False, chain_next="NONE",
                             chain_muscles=False)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_pick_image(bpy.types.Operator, ImportHelper):
    bl_idname = "ncp.pick_image"
    bl_label = "Pick Image"
    bl_description = "Выбрать готовую картинку персонажа с диска"
    bl_options = {"REGISTER"}

    filename_ext = ".png;.jpg;.jpeg;.webp;.bmp"
    filter_glob: StringProperty(default="*.png;*.jpg;*.jpeg;*.webp;*.bmp",
                                options={"HIDDEN"})

    def execute(self, context):
        _st(context).image_path = self.filepath
        return {"FINISHED"}


class NCP_OT_generate_mesh(bpy.types.Operator):
    bl_idname = "ncp.generate_mesh"
    bl_label = "Generate Mesh"
    bl_description = ("Сгенерировать 3D-меш из картинки через Image to 3D "
                      "(движок должен быть установлен, модели скачаны)")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return not JOB["active"] and ncp_img3d.installed()

    def execute(self, context):
        try:
            _start_mesh_job(context, chain_next="NONE",
                            chain_muscles=False)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_install_engine(bpy.types.Operator):
    bl_idname = "ncp.install_engine"
    bl_label = "Install Engine"
    bl_description = ("Установить движок Image to 3D (venv + зависимости, "
                      "несколько ГБ, один раз)")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return ncp_img3d.installed()

    def execute(self, context):
        try:
            bpy.ops.img3d.install_engine()
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_dl_models(bpy.types.Operator):
    bl_idname = "ncp.dl_models"
    bl_label = "Download Models"
    bl_description = ("Скачать AI-модели выбранного движка "
                      "(16-25 ГБ, с докачкой)")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return ncp_img3d.installed()

    def execute(self, context):
        mod = ncp_img3d.resolve()
        _, _, miss = ncp_img3d.readiness(mod, _st(context).engine)
        if not miss:
            self.report({"INFO"}, "Все модели уже на месте")
            return {"FINISHED"}
        try:
            ncp_img3d.start_download(miss)
            self.report({"INFO"}, "Загрузка: %s" % ", ".join(miss))
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_build_skeleton(bpy.types.Operator):
    bl_idname = "ncp.build_skeleton"
    bl_label = "Build Skeleton"
    bl_description = ("Построить neuro-скелет (human.json) внутри меша, "
                      "подогнать меш под скелет и назначить автовесы")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not JOB["active"]

    def execute(self, context):
        mesh = _find_skin_mesh(context)
        if mesh is None:
            _fail(context, "Нет меша: сгенерируйте или выделите меш "
                           "персонажа")
            self.report({"ERROR"}, "Нет меша: сгенерируйте или выделите "
                                   "меш персонажа")
            return {"CANCELLED"}
        try:
            _build_rig(context, mesh)
        except Exception as e:
            _fail(context, "Скелет: %s: %s" % (type(e).__name__, e))
            self.report({"ERROR"}, "%s: %s" % (type(e).__name__, e))
            return {"CANCELLED"}
        self.report({"INFO"}, "Скелет NeuroCharacter готов, весы назначены")
        return {"FINISHED"}


class NCP_OT_add_muscles(bpy.types.Operator):
    bl_idname = "ncp.add_muscles"
    bl_label = "Add Muscles"
    bl_description = ("Автоматически создать мышцы X-Muscle на все пары "
                      "костей neuro-скелета (13 групп) и пришить к коже")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return not JOB["active"]

    def execute(self, context):
        st = _st(context)
        arm = bpy.data.objects.get("NeuroCharacter")
        if arm is None:
            _fail(context, "Сначала постройте скелет (шаг 4)")
            self.report({"ERROR"}, "Сначала постройте скелет (шаг 4)")
            return {"CANCELLED"}
        if not ncp_muscles.installed():
            msg = ("X-Muscle System не включён — включите его в "
                   "Preferences -> Add-ons (поиск: X-Muscle)")
            _fail(context, msg)
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        try:
            created, delta, errors = ncp_muscles.create_all(context, arm)
            if created == 0:
                raise RuntimeError("Ни одной мышцы не создано: %s"
                                   % ("; ".join(errors[:3])
                                      or "причина неизвестна"))
            if st.attach_skin:
                mesh = _find_skin_mesh(context)
                ncp_muscles.attach_to_skin(context, mesh)
                ncp_muscles.enable_dynamics()
                msg = "Мышцы: создано %d, пришито к коже" % created
            else:
                msg = "Мышцы создано: %d" % created
            if errors:
                msg += ". Пропуски: " + "; ".join(errors[:3])
            st.state = "IDLE"
            st.status = msg
            if errors:
                self.report({"WARNING"}, msg)
            else:
                self.report({"INFO"}, msg)
        except Exception as e:
            _fail(context, "Мышцы: %s" % e)
            self.report({"ERROR"}, "%s: %s" % (type(e).__name__, e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_attach_muscles(bpy.types.Operator):
    bl_idname = "ncp.attach_muscles"
    bl_label = "Attach to Skin"
    bl_description = ("Пришить существующие мышцы к коже заново "
                      "(shrinkwrap + веса + драйверы)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return ncp_muscles.installed() and bool(ncp_muscles.muscle_objects())

    def execute(self, context):
        try:
            n = ncp_muscles.attach_to_skin(context, _find_skin_mesh(context))
            self.report({"INFO"}, "Пришито мышц: %d" % n)
        except Exception as e:
            _fail(context, "Пришивание мышц: %s" % e)
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_bake_muscles(bpy.types.Operator):
    bl_idname = "ncp.bake_muscles"
    bl_label = "Bake Muscles"
    bl_description = ("Пересобрать и запечь softbody-кэши всех мышц "
                      "(запускайте ПОСЛЕ импорта анимации)")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return ncp_muscles.installed() and bool(ncp_muscles.muscle_objects())

    def execute(self, context):
        try:
            ncp_muscles.bake_caches(context)
            self.report({"INFO"}, "Кэши мышц запечены")
        except Exception as e:
            _fail(context, "Запекание мышц: %s" % e)
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_import_motion(bpy.types.Operator, ImportHelper):
    bl_idname = "ncp.import_motion"
    bl_label = "Import Neuro Motion"
    bl_description = ("Наложить движение из neurosim (motion JSON) на "
                      "скелет NeuroCharacter: мышца за мышцей всё "
                      "проиграется; отсутствующий скелет будет создан")
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        st = _st(context)
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if "spec" not in payload or "frames" not in payload:
                raise RuntimeError("Это не neuro motion JSON "
                                   "(нет spec/frames)")
            arm = bpy.data.objects.get("NeuroCharacter")
            if arm is None:
                ao = context.view_layer.objects.active
                if ao is not None and ao.type == "ARMATURE":
                    arm = ao
            if arm is None:
                arm = neuro_rig.build_armature(
                    payload["spec"], name="NeuroCharacter")
            n, fps = neuro_rig.apply_motion_to_armature(
                arm, payload, frame_step=st.frame_step)
            if st.build_world and payload.get("world"):
                neuro_rig.add_ground()
                neuro_rig.build_world_props(payload["world"])
        except Exception as e:
            _fail(context, "Импорт движения: %s: %s"
                  % (type(e).__name__, e))
            self.report({"ERROR"}, "%s: %s" % (type(e).__name__, e))
            return {"CANCELLED"}
        self.report({"INFO"}, "Анимация: %d кадров @ %.0f fps" % (n, fps))
        return {"FINISHED"}


class NCP_OT_run_all(bpy.types.Operator):
    bl_idname = "ncp.run_all"
    bl_label = "Full Cycle"
    bl_description = ("Полный конвейер одной кнопкой: ComfyUI -> картинка "
                      "-> меш -> neuro-скелет -> мышцы. Если картинка "
                      "уже задана — начнёт с меша")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return not JOB["active"] and ncp_img3d.installed()

    def execute(self, context):
        st = _st(context)
        try:
            if st.image_path and os.path.isfile(
                    bpy.path.abspath(st.image_path)):
                _start_mesh_job(context, chain_next="RIG",
                                chain_muscles=True)
            else:
                _start_comfy_job(context, refine=False, chain_next="RIG",
                                 chain_muscles=True)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class NCP_OT_cancel(bpy.types.Operator):
    bl_idname = "ncp.cancel"
    bl_label = "Cancel"
    bl_description = "Прервать текущий шаг конвейера (ComfyUI/меш/мышцы)"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context):
        return JOB["active"]

    def execute(self, context):
        JOB.update(active=False, phase="", next="", comfy=None)
        st = _st(context)
        st.state = "IDLE"
        st.status = "Отменено пользователем"
        self.report({"INFO"}, "Конвейер остановлен")
        return {"FINISHED"}


class NCP_OT_open_out(bpy.types.Operator):
    bl_idname = "ncp.open_out"
    bl_label = "Open Output Folder"
    bl_description = ("Открыть папку результатов (эскизы, картинки) "
                      "в проводнике")
    bl_options = {"REGISTER"}

    def execute(self, context):
        d = _out_dir(context)
        try:
            if sys.platform == "win32":
                os.startfile(d)  # noqa
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            self.report({"ERROR"}, "Не удалось открыть папку: %s" % e)
            return {"CANCELLED"}
        return {"FINISHED"}


def _build_env_report(context):
    L = []
    add = L.append
    ver = ".".join(str(x) for x in bl_info["version"])
    add("Neuro Character Pipeline — отчёт окружения v%s" % ver)
    add("дата: %s" % time.strftime("%Y-%m-%d %H:%M"))
    add("blender: %s | python: %s | platform: %s"
        % (bpy.app.version_string, sys.version.split()[0], sys.platform))
    add("")
    add("== включённые аддоны (neuro / image_to_3d / xmuscle) ==")
    try:
        keys = sorted(bpy.context.preferences.addons.keys())
    except Exception as e:
        keys = []
        add("  <ошибка: %s>" % e)
    for k in keys:
        lk = k.lower()
        if "neuro" in lk or "image_to_3d" in lk or "img3d" in lk \
                or "xmuscle" in lk or "xpiemenu" in lk:
            add("  " + k)
    if not any(k.rsplit(".", 1)[-1].lower() == _NCP_BASENAME.lower()
               for k in keys):
        add("  !!! %s НЕ В СПИСКЕ ВКЛЮЧЁННЫХ АДДОНОВ" % _NCP_BASENAME)
    add("")
    add("== физические копии ==")
    try:
        dirs = list(bpy.utils.script_paths(submodule="addons"))
    except Exception:
        dirs = []
    for d in dirs:
        good = os.path.join(d, _NCP_BASENAME, "__init__.py")
        bad = os.path.join(d, "character", _NCP_BASENAME, "__init__.py")
        if os.path.isfile(good):
            add("  OK      " + good)
        if os.path.isfile(bad):
            add("  ЛИШНЯЯ  " + bad + "   <- удалите папку character")
    if not dirs:
        add("  <не удалось получить пути аддонов>")
    add("")
    add("== зарегистрированные классы ==")
    for t in ("NCP_PT_status", "NCP_PT_pipeline", "NCP_PT_sketch",
              "NCP_PT_image", "NCP_PT_mesh", "NCP_PT_rig",
              "NCP_PT_muscles", "NCP_PT_anim", "NCP_PT_settings",
              "NCP_PT_scene", "NCP_Props"):
        add("  %-16s %s" % (t, "есть" if hasattr(bpy.types, t) else "НЕТ"))
    scn = getattr(bpy.context, "scene", None)
    st = getattr(scn, "ncp", None)
    add("  Scene.ncp: %s" % ("None (!)" if st is None else "ok"))
    add("")
    add("== ошибки регистрации ==")
    if REGISTER_ERRORS:
        for line in REGISTER_ERRORS:
            add("  " + line)
    else:
        add("  нет")
    add("")
    add("== вшитые движки (vendor) ==")
    if VENDOR_STATUS:
        for name, err in VENDOR_STATUS.items():
            add("  %-8s %s" % (name, err or "ок"))
    else:
        add("  <не регистрировались>")
    legacy = _legacy_conflicts()
    add("  старые раздельные копии: %s"
        % (", ".join(legacy) if legacy else "нет"))
    add("")
    add("== ComfyUI ==")
    add("  url: %s | отвечает: %s"
        % (_comfy_url(), "да" if _comfy_alive_cached() else "НЕТ"))
    p = _get_prefs()
    if p is not None:
        add("  comfyui_dir: %r" % (getattr(p, "comfyui_dir", "") or ""))
        add("  comfyui_python: %r" % (getattr(p, "comfyui_python", "") or ""))
        add("  engine_dir: %r" % (getattr(p, "engine_dir", "") or ""))
        add("  out_dir: %r" % (getattr(p, "out_dir", "") or ""))
        add("  gentle_download: %s" % bool(getattr(p, "gentle_download",
                                                   False)))
    else:
        add("  <AddonPreferences недоступны>")
    add("")
    add("== NCP статус в сцене ==")
    if st is not None:
        add("  state: %s | status: %s" % (st.state, st.status[:100]))
    return "\n".join(L)


class NCP_OT_start_comfy(bpy.types.Operator):
    bl_idname = "ncp.start_comfy"
    bl_label = "Start ComfyUI"
    bl_description = ("Запустить ComfyUI указанным Python (папку ComfyUI "
                      "и python.exe задайте в Files & Settings). Если "
                      "ComfyUI уже отвечает — ничего не делает")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        d = bpy.path.abspath(_pref_str("comfyui_dir"))
        py = bpy.path.abspath(_pref_str("comfyui_python"))
        port = 8188
        p = _get_prefs()
        if p is not None:
            port = int(getattr(p, "comfyui_port", 8188) or 8188)
        if not d or not os.path.isdir(d):
            self.report({"ERROR"}, "Укажите папку ComfyUI в настройках "
                                   "(N-панель -> Files & Settings)")
            return {"CANCELLED"}
        if not os.path.isfile(os.path.join(d, "main.py")):
            self.report({"ERROR"}, "В папке нет main.py: %s" % d)
            return {"CANCELLED"}
        if not py or not os.path.isfile(py):
            self.report({"ERROR"}, "Укажите python.exe (например "
                                   "<ComfyUI>\\python_embeded\\python.exe)")
            return {"CANCELLED"}
        if _port_open("127.0.0.1", port):
            self.report({"INFO"}, "ComfyUI уже отвечает на порту %d" % port)
            return {"FINISHED"}
        log = os.path.join(_out_dir(context), "comfyui_console.log")
        kw = {}
        if sys.platform == "win32":
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        try:
            with open(log, "a", encoding="utf-8") as lf:
                subprocess.Popen(
                    [py, "-s", "main.py", "--listen", "127.0.0.1",
                     "--port", str(port)],
                    cwd=d, stdout=lf, stderr=subprocess.STDOUT, **kw)
        except Exception as e:
            self.report({"ERROR"}, "Запуск не удался: %s" % e)
            return {"CANCELLED"}
        self.report({"INFO"}, "ComfyUI запускается на порту %d (лог: %s). "
                              "Подождите полминуты" % (port, log))
        return {"FINISHED"}


class NCP_OT_env_report(bpy.types.Operator):
    bl_idname = "ncp.env_report"
    bl_label = "Env Report"
    bl_description = ("Собрать отчёт окружения (что включено, какие копии "
                      "аддона, ошибки регистрации) в файл ~/ncp_report.txt "
                      "и показать его в Text Editor")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        text = _build_env_report(context)
        path = os.path.join(os.path.expanduser("~"), "ncp_report.txt")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            self.report({"ERROR"}, "Не удалось записать %s: %s" % (path, e))
            return {"CANCELLED"}
        print(text)  # дубль в системную консоль
        try:
            txt = bpy.data.texts.get("ncp_report.txt")
            if txt is None:
                txt = bpy.data.texts.new("ncp_report.txt")
            txt.clear()
            txt.write(text)
            for w in bpy.context.window_manager.windows:
                for a in w.screen.areas:
                    if a.type not in {"TEXT_EDITOR", "CONSOLE", "INFO"}:
                        continue
                    a.type = "TEXT_EDITOR"
                    a.spaces.active.text = txt
                    break
                else:
                    continue
                break
        except Exception:
            pass
        self.report({"INFO"}, "Отчёт: %s (и в Text Editor)" % path)
        return {"FINISHED"}


# ------------------------------------------------------------- панель
# Стиль повторяет язык дизайна Image to 3D: карточка статуса, трекер
# шагов с чекмарками, крупные главные кнопки, сворачиваемые секции.
# Кнопки EN, подсказки RU — как и раньше.

PHASE_RU = {
    "COMFY_IMG": "ComfyUI: генерация картинки",
    "COMFY_REFINE": "ComfyUI: уточнение эскиза",
    "MESH": "Image to 3D: генерация меша",
    "MUSCLES": "X-Muscle: расстановка мышц",
}

# (ключ, имя в трекере, иконка секции)
STEPS = (
    ("sketch", "Sketch", "BRUSHES_ALL"),
    ("image", "Image", "IMAGE_DATA"),
    ("mesh", "Mesh", "MESH_MONKEY"),
    ("rig", "Rig", "ARMATURE_DATA"),
    ("muscles", "Musc", "MOD_SOFT"),
    ("anim", "Anim", "PLAY"),
)

TRACKER_ICON = {"done": "CHECKMARK", "active": "PLAY",
                "pending": "BLANK1"}


def _i3d_ready(context):
    mod = ncp_img3d.resolve()
    if mod is None:
        return None, False
    ready, _, _ = ncp_img3d.readiness(mod, _st(context).engine)
    return mod, ready


def _step_flags(context, st):
    """Что уже готово (для трекера и иконок секций)."""
    mesh_ok = (any(o.type == "MESH" and (o.get("ncp_mesh")
                                         or o.get("img3d_image"))
                   for o in bpy.data.objects)
               or bpy.data.objects.get("Character_Mesh") is not None)
    arm = bpy.data.objects.get("NeuroCharacter")
    anim_ok = bool(arm is not None and arm.animation_data
                   and arm.animation_data.action)
    return {
        "sketch": bool(st.sketch_path),
        "image": bool(st.image_path and os.path.isfile(
            bpy.path.abspath(st.image_path))),
        "mesh": mesh_ok,
        "rig": arm is not None,
        "muscles": bool(ncp_muscles.muscle_objects()),
        "anim": anim_ok,
    }


def _tracker_states(flags, phase):
    """(состояние, ключ) для каждого шага: done / active / pending."""
    act = {"COMFY_IMG": "image", "COMFY_REFINE": "sketch", "MESH": "mesh",
           "MUSCLES": "muscles"}.get(phase, "")
    out = []
    for key, _name, _icon in STEPS:
        if key == act:
            out.append(("active", key))
        elif flags.get(key):
            out.append(("done", key))
        else:
            out.append(("pending", key))
    return out


def _hint(lay, text, icon="DOT"):
    row = lay.row()
    row.scale_y = 0.75
    row.label(text=text, icon=icon)


def _big(lay, op_id, text, icon=None):
    row = lay.row()
    row.scale_y = 1.45
    return row.operator(op_id, text=text, icon=icon)


def _status_card(lay, context, st):
    box = lay.box()
    row = box.row(align=True)
    row.scale_y = 1.2
    if st.state == "ERROR":
        row.alert = True
        row.label(text="Ошибка", icon="ERROR")
    elif st.state == "DONE":
        row.label(text="Готово", icon="CHECKMARK")
    elif st.state in PHASE_RU:
        sec = max(0, int(time.time() - JOB.get("t0", time.time())))
        row.label(text="%s  (%d:%02d)" % (PHASE_RU[st.state], sec // 60,
                                          sec % 60), icon="TIME")
    else:
        row.label(text="Конвейер персонажа", icon="USER")
    if JOB["active"]:
        row.operator("ncp.cancel", text="", icon="X")
    if st.status:
        srow = box.row()
        srow.scale_y = 0.8
        if st.state == "ERROR":
            srow.alert = True
        srow.label(text=st.status[:100])
    # трекер: 6 шагов, иконки состояния + подписи
    flags = _step_flags(context, st)
    states = _tracker_states(flags, JOB["phase"])
    icons_row = box.row(align=True)
    names_row = box.row(align=True)
    for (state, _key), (_k, name, _icon) in zip(states, STEPS):
        c1 = icons_row.row(align=True)
        c1.alignment = "CENTER"
        c1.ui_units_x = 1.9
        c1.label(text="", icon=TRACKER_ICON[state])
        c2 = names_row.row(align=True)
        c2.alignment = "CENTER"
        c2.ui_units_x = 1.9
        c2.scale_y = 0.7
        c2.label(text=name)


class _NCPGuard:
    """draw()/draw_header() не бросают исключений: Blender прячет панель,
    если draw упал, — вместо этого показываем ошибку и Env Report."""

    def _content(self, context):
        raise NotImplementedError

    def _head(self, context):
        pass

    def draw(self, context):
        try:
            self._content(context)
        except Exception as e:
            lay = self.layout
            box = lay.box()
            box.alert = True
            box.label(text="Ошибка интерфейса NCP", icon="ERROR")
            box.label(text=("%s: %s" % (type(e).__name__, e))[:90])
            box.operator("ncp.env_report", text="Env Report", icon="TEXT")

    def draw_header(self, context):
        try:
            self._head(context)
        except Exception:
            self.layout.label(icon="ERROR")


class _NCPSubPanel(_NCPGuard, bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NCP"
    bl_parent_id = "NCP_PT_pipeline"
    bl_options = {"DEFAULT_CLOSED"}


class NCP_PT_pipeline(_NCPGuard, bpy.types.Panel):
    bl_label = "Character Pipeline"
    bl_idname = "NCP_PT_pipeline"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NCP"
    bl_order = 1

    def _head(self, context):
        self.layout.label(icon="USER")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        for w in REGISTER_ERRORS:
            row = lay.row()
            row.alert = True
            row.label(text=("Вторая копия аддона? " + w)[:90], icon="ERROR")
        for w in _vendor_warnings():
            row = lay.row()
            row.alert = True
            row.label(text=w, icon="ERROR")
        _status_card(lay, context, st)
        _big(lay, "ncp.run_all", "FULL CYCLE", icon="PLAY")
        _hint(lay, "Всё одной кнопкой: картинка -> меш -> скелет -> мышцы")


class NCP_PT_sketch(_NCPSubPanel):
    bl_label = "1 · Sketch"
    bl_idname = "NCP_PT_sketch"

    def _head(self, context):
        st = context.scene.ncp
        self.layout.label(icon="CHECKMARK" if st.sketch_path
                          else "BRUSHES_ALL")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "sketch_path")
        row = lay.row(align=True)
        row.operator("ncp.new_sketch", text="New", icon="ADD")
        row.operator("ncp.open_sketch", text="Open in Krita",
                     icon="FILE_FOLDER")
        lay.prop(st, "krita_path")
        _big(lay, "ncp.refine_sketch", "Refine via ComfyUI", icon="SHADERFX")
        _hint(lay, "Эскиз превратится в рендер персонажа (img2img)")


class NCP_PT_image(_NCPSubPanel):
    bl_label = "2 · Image (ComfyUI)"
    bl_idname = "NCP_PT_image"

    def _head(self, context):
        st = context.scene.ncp
        ok = bool(st.image_path and os.path.isfile(
            bpy.path.abspath(st.image_path)))
        self.layout.label(icon="CHECKMARK" if ok else "IMAGE_DATA")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "prompt")
        lay.prop(st, "negative")
        lay.prop(st, "comfy_workflow")
        _hint(lay, "ComfyUI URL — в Files & Settings, хранится постоянно")
        row = lay.row(align=True)
        row.operator("ncp.pick_image", text="Pick Image", icon="EYEDROPPER")
        _big(lay, "ncp.generate_image", "Generate Image", icon="IMAGE_DATA")
        _hint(lay, "Промпт уйдёт в первую текстовую ноду workflow")


class NCP_PT_mesh(_NCPSubPanel):
    bl_label = "3 · Mesh (Image to 3D)"
    bl_idname = "NCP_PT_mesh"

    def _head(self, context):
        mesh_ok = any(o.type == "MESH" and (o.get("ncp_mesh")
                                            or o.get("img3d_image"))
                      for o in bpy.data.objects)
        self.layout.label(icon="CHECKMARK" if mesh_ok else "MESH_MONKEY")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        mod, ready = _i3d_ready(context)
        if mod is None:
            row = lay.row()
            row.alert = True
            row.label(text="Image to 3D не включён", icon="ERROR")
            _hint(lay, "Установите и включите аддон Image to 3D")
        else:
            lay.label(text="Движок и модели готовы" if ready
                      else "Нужно установить/докачать движок",
                      icon="CHECKMARK" if ready else "ERROR")
        lay.prop(st, "engine")
        row = lay.row(align=True)
        row.prop(st, "quality")
        row.prop(st, "texture")
        row = lay.row(align=True)
        row.prop(st, "faces")
        row.prop(st, "seed")
        _big(lay, "ncp.generate_mesh", "Generate Mesh", icon="MESH_MONKEY")
        if mod is not None and not ready:
            row = lay.row(align=True)
            row.operator("ncp.install_engine", text="Install Engine",
                         icon="ADD")
            row.operator("ncp.dl_models", text="Download Models",
                         icon="DOWNARROW_HLT")


class NCP_PT_rig(_NCPSubPanel):
    bl_label = "4 · Neuro Skeleton"
    bl_idname = "NCP_PT_rig"

    def _head(self, context):
        has_arm = bpy.data.objects.get("NeuroCharacter") is not None
        self.layout.label(icon="CHECKMARK" if has_arm else "ARMATURE_DATA")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "spec_path")
        _big(lay, "ncp.build_skeleton", "Build Skeleton",
             icon="ARMATURE_DATA")
        row = lay.row(align=True)
        row.prop(st, "mesh_rot_x")
        row.prop(st, "mesh_rot_z")
        row = lay.row(align=True)
        row.prop(st, "orient_auto")
        row.prop(st, "auto_rig", text="Auto-Rig")
        row = lay.row(align=True)
        row.prop(st, "add_fingers")
        row.prop(st, "debug_landmarks", text="Landmarks")
        row = lay.row(align=True)
        row.prop(st, "replace_existing", text="Replace")
        row.prop(st, "apply_transforms", text="Apply")
        _hint(lay, "Auto-Rig (вкл. по умолчанию): скелет подгоняется "
                   "под тело — промежность, плечи, колени ищутся по "
                   "геометрии меша, пропорции персонажа сохраняются. "
                   "Выключить = меш масштабируется под скелет 1.70 м")


class NCP_PT_muscles(_NCPSubPanel):
    bl_label = "5 · Muscles (X-Muscle)"
    bl_idname = "NCP_PT_muscles"

    def _head(self, context):
        self.layout.label(icon="CHECKMARK" if ncp_muscles.muscle_objects()
                          else "MOD_SOFT")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        if not ncp_muscles.installed():
            row = lay.row()
            row.alert = True
            row.label(text="X-Muscle System не включён — включите его в "
                           "Preferences -> Add-ons", icon="ERROR")
        else:
            lay.label(text="Мышц в сцене: %d"
                      % len(ncp_muscles.muscle_objects()), icon="MOD_SOFT")
        lay.prop(st, "muscle_style")
        lay.prop(st, "attach_skin")
        _big(lay, "ncp.add_muscles", "Add Muscles (13 групп)", icon="ADD")
        row = lay.row(align=True)
        row.operator("ncp.attach_muscles", text="Attach to Skin",
                     icon="LINKED")
        row.operator("ncp.bake_muscles", text="Bake", icon="RENDER_ANIM")


class NCP_PT_anim(_NCPSubPanel):
    bl_label = "6 · Animation (neurosim)"
    bl_idname = "NCP_PT_anim"

    def _head(self, context):
        arm = bpy.data.objects.get("NeuroCharacter")
        ok = bool(arm is not None and arm.animation_data
                  and arm.animation_data.action)
        self.layout.label(icon="CHECKMARK" if ok else "PLAY")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "frame_step")
        lay.prop(st, "build_world")
        _big(lay, "ncp.import_motion", "Import Neuro Motion", icon="PLAY")
        _hint(lay, "Motion JSON экспортируется из neurosim (кнопка Export)")


class NCP_PT_settings(_NCPSubPanel):
    bl_label = "Files & Settings"
    bl_idname = "NCP_PT_settings"

    def _head(self, context):
        self.layout.label(icon="PREFERENCES")

    def _content(self, context):
        st = context.scene.ncp
        lay = self.layout
        p = _get_prefs()
        if p is None:
            lay.prop(st, "out_dir")
            lay.prop(st, "comfy_timeout")
            lay.operator("ncp.open_out", icon="FILE_FOLDER")
            _hint(lay, "Настройки аддона не читаются — нажмите Env Report")
        else:
            lay.label(text="Постоянные настройки", icon="LOCKED")
            lay.prop(p, "comfy_url")
            lay.prop(p, "out_dir")
            lay.prop(st, "comfy_timeout")
            lay.operator("ncp.open_out", text="Open Output Folder",
                         icon="FILE_FOLDER")
            lay.separator()
            box = lay.box()
            box.label(text="Автозапуск ComfyUI", icon="PLAY")
            box.prop(p, "comfyui_dir")
            box.prop(p, "comfyui_python")
            box.prop(p, "comfyui_port")
            lay.operator("ncp.start_comfy", text="Start ComfyUI",
                         icon="PLAY")
            lay.operator("ncp.env_report", text="Env Report", icon="TEXT")
            _hint(lay, "comfyui_python — путь к python.exe ComfyUI "
                       "(например python_embeded\\python.exe)")
        # папка движка и моделей вшитого Image to 3D (хранится в Preferences,
        # не сбрасывается при закрытии Blender)
        if ncp_img3d.installed():
            lay.separator()
            box = lay.box()
            box.label(text="Engine & Models Folder", icon="MESH_MONKEY")
            d = ncp_img3d.engine_dir()
            if len(d) > 46:
                d = "..." + d[-43:]
            box.label(text=d or "~/3DRedbox/ImageTo3D (по умолчанию)",
                      icon="FILE_FOLDER")
            row = box.row(align=True)
            row.operator("img3d.pick_engine_dir", text="Choose Folder",
                         icon="FILE_FOLDER")
            row.operator("img3d.dl_folder", text="Open",
                         icon="FOLDER_REDIRECT")
            _hint(box, "Движок + модели (~25 ГБ), не папка ComfyUI; "
                       "хранится в Preferences аддона")
        _hint(lay, "Подробности и схематика — INSTALL_RU.txt рядом с аддоном")


class NCP_PT_status(_NCPGuard, bpy.types.Panel):
    """Панель-маяк: не трогает scene-пропсы, рисуется всегда.
    Если вы видите только её — сломан остальной интерфейс, и кнопка
    Env Report соберёт причину в файл ~/ncp_report.txt."""
    bl_label = "Neuro Character"
    bl_idname = "NCP_PT_status"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "NCP"
    bl_order = 0

    def _head(self, context):
        self.layout.label(icon="USER")

    def _content(self, context):
        lay = self.layout
        ver = ".".join(str(x) for x in bl_info["version"])
        lay.label(text="NCP v%s — интерфейс работает" % ver,
                  icon="CHECKMARK")
        if REGISTER_ERRORS:
            box = lay.box()
            box.alert = True
            box.label(text="Проблемы регистрации:", icon="ERROR")
            for line in REGISTER_ERRORS[:3]:
                box.label(text=line[:90])
            if len(REGISTER_ERRORS) > 3:
                box.label(text="... и ещё %d (см. Env Report)"
                          % (len(REGISTER_ERRORS) - 3))
        for w in _vendor_warnings()[:2]:
            row = lay.row()
            row.alert = True
            row.label(text=w[:90], icon="ERROR")
        ok = _comfy_alive_cached()
        lay.label(text="ComfyUI: %s" % ("отвечает" if ok else "не запущен"),
                  icon="CHECKMARK" if ok else "ERROR")
        row = lay.row(align=True)
        row.operator("ncp.start_comfy", text="Start ComfyUI", icon="PLAY")
        row.operator("ncp.env_report", text="", icon="TEXT")
        if not ncp_img3d.installed():
            _hint(lay, "Image to 3D не включён — меш строить нечем")
        _big(lay, "ncp.run_all", "FULL CYCLE", icon="PLAY")


class NCP_PT_scene(_NCPGuard, bpy.types.Panel):
    """Резервная панель в Properties -> Scene: видна всегда, даже если
    вкладка N-панели перекрыта другими аддонами."""
    bl_label = "Neuro Character Pipeline"
    bl_idname = "NCP_PT_scene"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "scene"

    def _head(self, context):
        self.layout.label(icon="USER")

    def _content(self, context):
        lay = self.layout
        st = getattr(context.scene, "ncp", None)
        if st is None:
            lay.label(text="Scene.ncp недоступен", icon="ERROR")
        elif st.status:
            srow = lay.row()
            if st.state == "ERROR":
                srow.alert = True
            srow.label(text=st.status[:80],
                       icon="ERROR" if st.state == "ERROR" else "INFO")
        col = lay.column(align=True)
        col.operator("ncp.run_all", text="FULL CYCLE", icon="PLAY")
        col.operator("ncp.build_skeleton", text="Build Skeleton",
                     icon="ARMATURE_DATA")
        col.operator("ncp.env_report", text="Env Report", icon="TEXT")
        _hint(lay, "Основной интерфейс — 3D-вьюпорт, клавиша N, вкладка NCP")


# ----------------------------------------------------------- регистрация

classes = (
    NCP_Props,
    NCP_AddonPrefs,
    NCP_OT_new_sketch,
    NCP_OT_open_sketch,
    NCP_OT_refine_sketch,
    NCP_OT_generate_image,
    NCP_OT_pick_image,
    NCP_OT_generate_mesh,
    NCP_OT_install_engine,
    NCP_OT_dl_models,
    NCP_OT_start_comfy,
    NCP_OT_env_report,
    NCP_OT_build_skeleton,
    NCP_OT_add_muscles,
    NCP_OT_attach_muscles,
    NCP_OT_bake_muscles,
    NCP_OT_import_motion,
    NCP_OT_run_all,
    NCP_OT_cancel,
    NCP_OT_open_out,
    NCP_PT_status,
    NCP_PT_pipeline,
    NCP_PT_sketch,
    NCP_PT_image,
    NCP_PT_mesh,
    NCP_PT_rig,
    NCP_PT_muscles,
    NCP_PT_anim,
    NCP_PT_settings,
    NCP_PT_scene,
)


def register():
    del REGISTER_ERRORS[:]
    for c in classes:
        try:
            bpy.utils.register_class(c)
        except Exception as e:
            # одна упавшая класс-регистрация больше не хоронит весь аддон:
            # обычно это вторая копия с теми же bl_idname - интерфейс
            # всё равно соберётся из того, что смогло
            REGISTER_ERRORS.append("%s: %s" % (c.__name__, e))
    try:
        bpy.types.Scene.ncp = PointerProperty(type=NCP_Props)
    except Exception as e:
        REGISTER_ERRORS.append("Scene.ncp: %s" % e)
    _disable_legacy()
    _register_vendor()
    if REGISTER_ERRORS:
        print("[Neuro Pipeline] часть классов не поднялась (обычно виновата "
              "вторая копия аддона - перезапустите Blender после удаления "
              "старых копий):")
        for x in REGISTER_ERRORS:
            print("   -", x)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    JOB.update(active=False, phase="", next="")
    _unregister_vendor()
    try:
        del bpy.types.Scene.ncp
    except Exception:
        pass
    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except Exception:
            pass


if __name__ == "__main__":
    register()
