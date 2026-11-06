# -*- coding: utf-8 -*-
"""
blender_neuro_importer — аддон Blender для импорта результатов neurosim.

Установка: Blender -> Edit -> Preferences -> Add-ons -> Install ->
выбрать этот файл -> включить галочку.

Использование:
  File -> Import -> Neuro Sim Motion (.json)  — арматура + меши + мир + анимация
  File -> Import -> Neuro Sim World (.json)   — только мир (props)
  Альтернатива: File -> Import -> Motion Capture (.bvh) — чистый мокап,
  Scale = 1.0 (метры).

Файлы экспорта создаёт программа blender_neuro (папка outputs/).
"""
bl_info = {
    "name": "Neuro Sim Importer (blender_neuro)",
    "author": "blender_neuro",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "File > Import > Neuro Sim (*.json); N-панель > Neuro",
    "description": "Импорт самообучающихся существ и их движения из RL-симулятора",
    "category": "Import-Export",
}

import json
import math
import os

import bpy
from bpy.props import (BoolProperty, CollectionProperty, StringProperty)
from bpy_extras.io_utils import ImportHelper
from mathutils import Matrix, Quaternion, Vector


# ----------------------------------------------------------------- утилиты

def _axis_shape_euler(axis):
    """Поворот фигуры: капсулы/цилиндры направлены вдоль Z."""
    if axis == "z":
        return (0.0, 0.0, 0.0)
    if axis == "y":
        return (-math.pi / 2, 0.0, 0.0)
    if axis == "x":
        return (0.0, math.pi / 2, 0.0)
    return (0.0, 0.0, 0.0)


def _link_frames(spec, root_pos, root_quat, joint_angles):
    """Мировые 4x4 матрицы РАМОК звеньев (суставных), в порядке спеки.

    Рамка звена = его сустав (у корня = COM корня).
    root_quat — [x, y, z, w] (конвенция pybullet).
    """
    by_name = {l["name"]: l for l in spec["links"]}
    frames = {}
    for l in spec["links"]:
        if l["parent"] is None:
            R = Quaternion((root_quat[3], root_quat[0], root_quat[1],
                            root_quat[2])).to_matrix().to_4x4()
            frames[l["name"]] = Matrix.Translation(Vector(root_pos)) @ R
            continue
        j = l["joint"]
        par = by_name[l["parent"]]
        par_frame = (par["com"] if par["parent"] is None
                     else par["joint"]["anchor"])
        l_a = Vector(j["anchor"]) - Vector(par_frame)
        rot = Matrix.Rotation(float(joint_angles.get(l["name"], 0.0)), 4,
                              Vector(j["axis"]))
        frames[l["name"]] = (frames[par["name"]] @
                             Matrix.Translation(l_a) @ rot)
    return frames


def _com_frames(spec, frames):
    """COM-позы звеньев (для мешей): F_com = F_link @ Translation(d)."""
    out = {}
    for l in spec["links"]:
        if l["parent"] is None:
            out[l["name"]] = frames[l["name"]]
        else:
            d = Vector(l["com"]) - Vector(l["joint"]["anchor"])
            out[l["name"]] = frames[l["name"]] @ Matrix.Translation(d)
    return out


# ----------------------------------------------------------------- арматура

def build_armature(spec, name="NeuroCreature"):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj

    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones = {}
    for l in spec["links"]:
        eb = arm_data.edit_bones.new(l["name"])
        if l["parent"] is None:
            head = Vector(l["com"]) - Vector((0, 0, 0.12))
            tail = Vector(l["com"]) + Vector((0, 0, 0.03))
        else:
            head = Vector(l["joint"]["anchor"])
            tail = Vector(l["com"])
            if (tail - head).length < 0.02:
                tail = head + Vector((0, 0, 0.1))
        eb.head, eb.tail = head, tail
        eb.roll = 0.0
        edit_bones[l["name"]] = eb
    for l in spec["links"]:
        if l["parent"] is not None:
            edit_bones[l["name"]].parent = edit_bones[l["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")

    # покой: обнулить позу (rest pose уже покой)
    for pb in arm_obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    return arm_obj


def apply_animation(arm_obj, spec, frames, fps=48.0, frame_step=1):
    scene = bpy.context.scene
    scene.render.fps = int(round(fps))

    # покойные матрицы костей (для ориентации вдоль кости)
    rest = {pb.name: (arm_obj.matrix_world.inverted() @ pb.bone.matrix_local)
            for pb in arm_obj.pose.bones}

    act = arm_obj.animation_data_create()
    act.action = bpy.data.actions.new("neuro_motion")

    joint_names = [l["name"] for l in spec["links"] if l.get("joint")]
    n_frames = len(frames)
    for fi, fr in enumerate(frames):
        f = fi * frame_step + 1
        jd = dict(zip(joint_names, [float(a) for a in fr["joints"]]))
        frames_i = _link_frames(spec, fr["root_pos"], fr["root_quat"], jd)
        for l in spec["links"]:
            pb = arm_obj.pose.bones.get(l["name"])
            if pb is None:
                continue
            F = frames_i[l["name"]]
            R_rest = rest[l["name"]].to_3x3()
            if l["parent"] is None:
                head_local = Vector((0, 0, -0.12))  # голова кости таза ниже COM
            else:
                head_local = Vector((0, 0, 0))
            pos = (F @ Matrix.Translation(head_local)).to_translation()
            M = Matrix.Translation(pos) @ (F.to_3x3() @ R_rest).to_4x4()
            pb.matrix = M
            pb.keyframe_insert("location", frame=f)
            pb.keyframe_insert("rotation_quaternion", frame=f)
    scene.frame_start = 1
    scene.frame_end = max(2, n_frames * frame_step)
    scene.frame_set(1)


# -------------------------------------------------------------------- меши

def build_creature_meshes(spec, arm_obj, collection=None):
    """Капсулы/боксы, жёстко привязанные к костям (parent_type='BONE')."""
    col = collection or bpy.context.scene.collection
    meshes = []
    for l in spec["links"]:
        sh = l["shape"]
        com = Vector(l["com"])
        eul = _axis_shape_euler(sh.get("axis", "z"))
        parts = []
        if sh["type"] == "capsule":
            r, hl = sh["radius"], sh["half_len"]
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=2 * hl,
                                                location=com, rotation=eul,
                                                vertices=24)
            parts.append(bpy.context.active_object)
            ax = {"x": Vector((1, 0, 0)), "y": Vector((0, 1, 0)),
                  "z": Vector((0, 0, 1))}[sh.get("axis", "z")]
            for s in (+1, -1):
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=r, segments=24, ring_count=12,
                    location=com + ax * (hl * s))
                parts.append(bpy.context.active_object)
        elif sh["type"] == "box":
            hx, hy, hz = sh["half"]
            bpy.ops.mesh.primitive_cube_add(location=com, rotation=eul,
                                            scale=(hx, hy, hz))
            parts.append(bpy.context.active_object)
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=sh["radius"],
                                                 segments=24, ring_count=12,
                                                 location=com)
            parts.append(bpy.context.active_object)

        obj = parts[0]
        if len(parts) > 1:
            bpy.ops.object.select_all(action="DESELECT")
            for o in parts:
                o.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.join()
        obj.name = f"mesh_{l['name']}"
        if sh["type"] != "box" and hasattr(obj, "shade_smooth"):
            obj.shade_smooth()
        mat = bpy.data.materials.new(f"mat_{l['name']}")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            c = sh["color"]
            bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1)
            bsdf.inputs["Roughness"].default_value = 0.6
        obj.data.materials.append(mat)

        pb = arm_obj.pose.bones.get(l["name"])
        if pb is not None:
            # жёсткая привязка к кости: локаль = rest^-1 @ мировая поза покоя
            obj.parent = arm_obj
            obj.parent_type = "BONE"
            obj.parent_bone = l["name"]
            obj.matrix_parent_inverse.identity()
            rest_bone = pb.bone.matrix_local
            obj.matrix_basis = rest_bone.inverted() @ obj.matrix_basis
        else:
            obj.parent = arm_obj
        meshes.append(obj)
    return meshes


# --------------------------------------------------------------------- мир

def build_world_props(world_def, name_prefix="world"):
    col_out = []
    for i, prop in enumerate(world_def.get("props", [])):
        pos = Vector(prop["pos"])
        rpy = prop.get("rpy", [0, 0, 0])
        sh = prop["shape"]
        if sh == "box":
            hx, hy, hz = prop["half"]
            bpy.ops.mesh.primitive_cube_add(location=pos,
                                            rotation=(rpy[0], rpy[1], rpy[2]),
                                            scale=(hx, hy, hz))
        elif sh == "sphere":
            bpy.ops.mesh.primitive_uv_sphere_add(radius=prop["radius"],
                                                 segments=24, ring_count=12,
                                                 location=pos)
        else:  # cylinder
            bpy.ops.mesh.primitive_cylinder_add(radius=prop["radius"],
                                                depth=prop["length"],
                                                location=pos,
                                                rotation=(rpy[0], rpy[1], rpy[2]))
        obj = bpy.context.active_object
        obj.name = f"{name_prefix}_{sh}_{i}"
        mat = bpy.data.materials.new(obj.name + "_m")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            c = prop.get("color", [0.6, 0.6, 0.6, 1])
            bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], 1)
        obj.data.materials.append(mat)
        col_out.append(obj)
    return col_out


def add_ground(size=60.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "neuro_ground"
    mat = bpy.data.materials.new("neuro_ground_m")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.28, 0.42, 0.22, 1)
    obj.data.materials.append(mat)
    return obj


# ---------------------------------------------------------------- операторы

class NEURO_OT_import_motion(bpy.types.Operator, ImportHelper):
    bl_idname = "neuro.import_motion"
    bl_label = "Импорт Neuro Sim движения (.json)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    build_meshes: BoolProperty(name="Меши существа", default=True)
    build_world: BoolProperty(name="Мир из файла", default=True)
    frame_step: StringProperty(name="Шаг кадров (1=N)", default="1")

    def execute(self, ctx):
        with open(self.filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        spec = payload["spec"]
        frames = payload["frames"]
        fps = float(payload.get("fps", 48.0))

        try:
            step = max(1, int(self.frame_step))
        except ValueError:
            step = 1
        if step > 1:
            frames = frames[::step]

        arm = build_armature(spec, name=payload.get("name", "NeuroCreature"))
        apply_animation(arm, spec, frames, fps=fps / step, frame_step=1)
        if self.build_meshes:
            build_creature_meshes(spec, arm)
        if self.build_world and payload.get("world"):
            add_ground()
            build_world_props(payload["world"])
        self.report({"INFO"}, f"Готово: {len(frames)} кадров, fps={fps / step:.0f}")
        return {"FINISHED"}


class NEURO_OT_import_world(bpy.types.Operator, ImportHelper):
    bl_idname = "neuro.import_world"
    bl_label = "Импорт Neuro Sim мира (.json)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, ctx):
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        world = data.get("world", data)  # и motion-json, и чистый world-json
        add_ground()
        build_world_props(world)
        self.report({"INFO"}, "Мир построен")
        return {"FINISHED"}


class NEURO_PT_panel(bpy.types.Panel):
    bl_label = "Neuro Sim"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Neuro"

    def draw(self, ctx):
        col = self.layout.column()
        col.operator("neuro.import_motion", icon="ARMATURE_DATA")
        col.operator("neuro.import_world", icon="MESH_CUBE")


def _menu_import(self, ctx):
    self.layout.operator(NEURO_OT_import_motion.bl_idname,
                         text="Neuro Sim Motion (.json)")
    self.layout.operator(NEURO_OT_import_world.bl_idname,
                         text="Neuro Sim World (.json)")


classes = (NEURO_OT_import_motion, NEURO_OT_import_world, NEURO_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(_menu_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
