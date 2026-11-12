# -*- coding: utf-8 -*-
"""
Neuro Character Pipeline — мастер-аддон, связывающий:
  ComfyUI (картинка) -> Image to 3D (меш) -> neuro-скелет (human.json)
  -> X-Muscle System (мышцы) -> neuro-анимация (motion JSON).

N-панель 3D-вьюпорта, категория "Neuro".
"""
bl_info = {
    "name": "Neuro Character Pipeline",
    "author": "blender_neuro",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D Viewport > N-panel > Neuro",
    "description": "Картинка -> 3D-меш -> neuro-скелет -> мышцы -> анимация",
    "category": "Rigging",
}

import json
import os
import subprocess
import sys
import time

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
else:
    from . import ncp_comfy
    from . import ncp_img3d
    from . import ncp_muscles
    from . import neuro_rig


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


def _out_dir(context):
    st = context.scene.ncp
    d = st.out_dir.strip() or os.path.join(os.path.expanduser("~"),
                                           "NeuroCharacter")
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
    arm = neuro_rig.build_armature(spec, name="NeuroCharacter")
    neuro_rig.fit_mesh_to_skeleton(
        mesh, spec, rot_x_deg=st.mesh_rot_x, do_apply=st.apply_transforms)
    neuro_rig.skin_mesh_to_armature(mesh, arm)
    return arm


# --------------------------------------------------- шаги конвейера

def _start_comfy_job(context, refine, chain_next, chain_muscles):
    st = _st(context)
    cli = ncp_comfy.ComfyClient(st.comfy_url)
    if not cli.check():
        raise ncp_comfy.ComfyError(
            "ComfyUI не отвечает на %s (запустите его или проверьте URL)"
            % st.comfy_url)
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
    apply_transforms: BoolProperty(
        name="Apply transforms", default=True,
        description="Запечь масштаб/поворот меша перед скинингом "
                    "(рекомендуется)")
    replace_existing: BoolProperty(
        name="Replace skeleton", default=True,
        description="Удалить старый NeuroCharacter и пересоздать")
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
            self.report({"ERROR"}, "Нет меша: сгенерируйте или выделите "
                                   "меш персонажа")
            return {"CANCELLED"}
        try:
            _build_rig(context, mesh)
        except Exception as e:
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
        return (not JOB["active"] and ncp_muscles.installed()
                and bpy.data.objects.get("NeuroCharacter") is not None)

    def execute(self, context):
        arm = bpy.data.objects.get("NeuroCharacter")
        try:
            created, delta, errors = ncp_muscles.create_all(context, arm)
            if _st(context).attach_skin:
                mesh = _find_skin_mesh(context)
                ncp_muscles.attach_to_skin(context, mesh)
                ncp_muscles.enable_dynamics()
                msg = "Мышцы: создано %d, пришито к коже" % created
            else:
                msg = "Мышцы создано: %d" % created
            if errors:
                self.report({"WARNING"}, msg + " | " + "; ".join(errors[:3]))
            else:
                self.report({"INFO"}, msg)
        except Exception as e:
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


class _NCPSubPanel(bpy.types.Panel):
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Neuro"
    bl_parent_id = "NCP_PT_pipeline"
    bl_options = {"DEFAULT_CLOSED"}


class NCP_PT_pipeline(bpy.types.Panel):
    bl_label = "Character Pipeline"
    bl_idname = "NCP_PT_pipeline"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Neuro"

    def draw_header(self, context):
        self.layout.label(icon="USER")

    def draw(self, context):
        st = context.scene.ncp
        lay = self.layout
        _status_card(lay, context, st)
        _big(lay, "ncp.run_all", "FULL CYCLE", icon="PLAY")
        _hint(lay, "Всё одной кнопкой: картинка -> меш -> скелет -> мышцы")


class NCP_PT_sketch(_NCPSubPanel):
    bl_label = "1 · Sketch"
    bl_idname = "NCP_PT_sketch"

    def draw_header(self, context):
        st = context.scene.ncp
        self.layout.label(icon="CHECKMARK" if st.sketch_path
                          else "BRUSHES_ALL")

    def draw(self, context):
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

    def draw_header(self, context):
        st = context.scene.ncp
        ok = bool(st.image_path and os.path.isfile(
            bpy.path.abspath(st.image_path)))
        self.layout.label(icon="CHECKMARK" if ok else "IMAGE_DATA")

    def draw(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "prompt")
        lay.prop(st, "negative")
        lay.prop(st, "comfy_url")
        lay.prop(st, "comfy_workflow")
        row = lay.row(align=True)
        row.operator("ncp.pick_image", text="Pick Image", icon="EYEDROPPER")
        _big(lay, "ncp.generate_image", "Generate Image", icon="IMAGE_DATA")
        _hint(lay, "Промпт уйдёт в первую текстовую ноду workflow")


class NCP_PT_mesh(_NCPSubPanel):
    bl_label = "3 · Mesh (Image to 3D)"
    bl_idname = "NCP_PT_mesh"

    def draw_header(self, context):
        mesh_ok = any(o.type == "MESH" and (o.get("ncp_mesh")
                                            or o.get("img3d_image"))
                      for o in bpy.data.objects)
        self.layout.label(icon="CHECKMARK" if mesh_ok else "MESH_MONKEY")

    def draw(self, context):
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

    def draw_header(self, context):
        has_arm = bpy.data.objects.get("NeuroCharacter") is not None
        self.layout.label(icon="CHECKMARK" if has_arm else "ARMATURE_DATA")

    def draw(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "spec_path")
        _big(lay, "ncp.build_skeleton", "Build Skeleton",
             icon="ARMATURE_DATA")
        row = lay.row(align=True)
        row.prop(st, "mesh_rot_x")
        row.prop(st, "replace_existing", text="Replace")
        lay.prop(st, "apply_transforms")
        _hint(lay, "human.json: 18 костей, меш подгонится, автовесы")


class NCP_PT_muscles(_NCPSubPanel):
    bl_label = "5 · Muscles (X-Muscle)"
    bl_idname = "NCP_PT_muscles"

    def draw_header(self, context):
        self.layout.label(icon="CHECKMARK" if ncp_muscles.muscle_objects()
                          else "MOD_SOFT")

    def draw(self, context):
        st = context.scene.ncp
        lay = self.layout
        if not ncp_muscles.installed():
            row = lay.row()
            row.alert = True
            row.label(text="X-Muscle System не включён (Blender 5.0)",
                      icon="ERROR")
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

    def draw_header(self, context):
        arm = bpy.data.objects.get("NeuroCharacter")
        ok = bool(arm is not None and arm.animation_data
                  and arm.animation_data.action)
        self.layout.label(icon="CHECKMARK" if ok else "PLAY")

    def draw(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "frame_step")
        lay.prop(st, "build_world")
        _big(lay, "ncp.import_motion", "Import Neuro Motion", icon="PLAY")
        _hint(lay, "Motion JSON экспортируется из neurosim (кнопка Export)")


class NCP_PT_settings(_NCPSubPanel):
    bl_label = "Files & Settings"
    bl_idname = "NCP_PT_settings"

    def draw_header(self, context):
        self.layout.label(icon="PREFERENCES")

    def draw(self, context):
        st = context.scene.ncp
        lay = self.layout
        lay.prop(st, "out_dir")
        lay.prop(st, "comfy_timeout")
        lay.operator("ncp.open_out", icon="FILE_FOLDER")
        _hint(lay, "Подробности и схематика — INSTALL_RU.txt рядом с аддоном")


# ----------------------------------------------------------- регистрация

classes = (
    NCP_Props,
    NCP_OT_new_sketch,
    NCP_OT_open_sketch,
    NCP_OT_refine_sketch,
    NCP_OT_generate_image,
    NCP_OT_pick_image,
    NCP_OT_generate_mesh,
    NCP_OT_install_engine,
    NCP_OT_dl_models,
    NCP_OT_build_skeleton,
    NCP_OT_add_muscles,
    NCP_OT_attach_muscles,
    NCP_OT_bake_muscles,
    NCP_OT_import_motion,
    NCP_OT_run_all,
    NCP_OT_cancel,
    NCP_OT_open_out,
    NCP_PT_pipeline,
    NCP_PT_sketch,
    NCP_PT_image,
    NCP_PT_mesh,
    NCP_PT_rig,
    NCP_PT_muscles,
    NCP_PT_anim,
    NCP_PT_settings,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.ncp = PointerProperty(type=NCP_Props)


def unregister():
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    JOB.update(active=False, phase="", next="")
    del bpy.types.Scene.ncp
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
