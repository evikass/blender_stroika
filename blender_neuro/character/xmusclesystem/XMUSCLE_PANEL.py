'''
Copyright (C) 2024 k44dev

Created by Albert Makac

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import bpy

class XMuscleCreate(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Create"
    bl_category = "x-muscle Sys"
    bl_idname = "OBJECT_PT_A_createMuscleSys"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    @classmethod
    def poll(self, context):

        return (context.mode == 'OBJECT' or context.mode == 'POSE')

    def draw(self, context):
        layout = self.layout
        scn = context.scene
        scale = True

        col = layout.column(align = True)
        row = col.row()
        col.label(text="CREATE:")
        col.operator("muscle.add_basic_muscle", text="Add Basic Muscle", icon = 'RIGHTARROW_THIN')
        col.operator("muscle.add_muscle", text="Add Stylized Muscle", icon = 'RIGHTARROW_THIN')
        col.operator("muscle.add_strip_muscle", text="Add Strip Muscle", icon = 'RIGHTARROW_THIN')

        if context.object and context.object.type == 'MESH':
            if not getattr(context.object, "Muscle_XID", None):
                col = layout.column(align = False)
                col.operator("muscle.convert_to_muscle", text="Convert Mesh to Muscle", icon = 'AUTOMERGE_ON')
                scale = False

        col = layout.column(align = True)
        if scale:
            row = col.row()
            row.prop(scn, "Muscle_Scale", text="Muscle Scale", slider =  True)
        row = col.row()
        row.label(icon = 'CURSOR')
        row.label(text="Targeting Method")
        row = col.row()
        row.prop(scn, "Create_Type", expand=True)
    
        row = col.row()
        row.label(text="Muscle Name:")
        row = col.row()
        row.prop(scn, "Muscle_Name", text = "", icon='FORCE_LENNARDJONES')
        row.operator("muscle.rename_muscle", text="", icon = 'OUTLINER_DATA_GP_LAYER')
        row = col.row()
        row.label(text="Affixes:")
        row = col.row()
        row = col.row()
        row = col.row()
        row.label(icon = 'TEXT')
        row.prop(scn, "use_Affixes", text="use Affixes")
        row = col.row()
        if scn.use_Affixes:
            row = col.row()
            row.label(text="Prefix:")
            row.label(text="Suffix:")
            row = col.row()
            row.prop(scn, "Prefix", text="")
            row.prop(scn, "Suffix",text="")
        else:
            row = col.row()
            row.label(text="Use affixes for special features")
            row.label(text="", icon='PLUGIN')


class XMuscleSystem(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Muscle"
    bl_category = "x-muscle Sys"
    bl_idname = "OBJECT_PT_B_MuscleGeneralSys"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "objectmode"

    @classmethod
    def poll(self, context):

        if context.active_object == None:
            return False
        return context.object and context.object.type == 'MESH' and context.active_object.select_get() == True and getattr(context.object, "Muscle_XID", None)

    def draw(self, context):

        if not getattr(context.object, "Muscle_XID", None):
            return

        layout = self.layout
        obj = context.object
        scn = context.scene

        softbody = obj.soft_body
        col = layout.column(align=True)

        col.label(text="Muscle Mesh", icon='MOD_ARMATURE')

        row = col.row(align = True)

        split = row.split(factor=0.85)
        row = split.row()
        row.prop(obj, "Muscle_Render")
        row.prop(obj, "Muscle_View3D")
        row = col.row(align = True)
        row.label(text="")
        row = col.row(align = True)
        row.label(text="Resolution:")
        row.prop(obj, "View3D_Resolution", text="View3D",slider = False)
        row.prop(obj, "Render_Resolution", text="Render", slider = False)
        col = layout.column(align=True)
        col.prop(obj, "Muscle_Type_INT", text="Flexor to Extensor")
        col.prop(obj, "Base_Length_INT", slider = True)
        col.prop(obj, "Muscle_Shape", slider = True)
        col.prop(obj, "Volume_INT", slider = True)

        col.prop(obj, "Muscle_Size", slider = True)
        col.prop(obj, "Muscle_Offset", slider = True)

        col = layout.column(align=False)
        col = layout.column(align=False)
        col = layout.column(align=True)

        col.label(text="Muscle Dynamics", icon='ARMATURE_DATA')
        row = col.row(align = True)
        split = row.split(factor=0.85)
        row = split.row()
        row.prop(obj, "Dynamics_Render")
        row.prop(obj, "Dynamics_View3D")
        row = col.row(align = True)
        row.label(text="Jiggle:")
        col.prop(obj, "Jiggle_Springiness", slider = True)
        col.prop(obj, "Jiggle_Stiffness", slider = True)
        col.prop(obj, "Jiggle_Mass", slider = True)
        col.prop(obj, "Jiggle_Damping", slider = True)
        col.prop_search(softbody, "vertex_group_mass", obj, "vertex_groups", text="Vertex Group")

        #row is a single row, column groups them together instead
        row = col.row(align=False)
        row = layout.row()
        row.operator("muscle.smart_update", text="Smart Update", icon = "FILE_REFRESH")
        row = layout.row()
        col = layout.column(align=True)

        row.label(text="")
        row = layout.row()
        row.operator("muscle.reset_props", text="Reset Properties", icon = "MODIFIER")
        row = layout.row()
        row.operator("muscle.delete_muscle", text="Delete Muscle", icon = "CANCEL")


class OBJECT_UL_items(bpy.types.UIList):

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)

        row.prop(item, "name", text="", emboss=False, translate=False)

    def invoke(self, context, event):
        pass


class XMuscleMicroCtl(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Skin MicroControllers"
    bl_category = "x-muscle Sys"
    bl_idname = "OBJECT_PT_C_muscleMicroControllers"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "objectmode" #TODO

    @classmethod
    def poll(self, context):

        if context.active_object == None:
            return False
        if context.object and context.object.type == 'MESH' and context.active_object.select_get() == True and getattr(context.object, "Muscle_XID", None):
            return True
        if context.object.parent != None:
            if context.object and context.object.type == 'EMPTY' and context.active_object.select_get() == True and getattr(context.object.parent, "Muscle_XID", None):
                return True

    def draw(self, context):

        if not getattr(context.object, "Muscle_XID", None):
            if context.object.parent == None:
                return
            elif not getattr(context.object.parent, "Muscle_XID", None):
                return
            else:
                obj = context.object.parent
        else:
            obj = context.object


        layout = self.layout
        scn = context.scene

        col = layout.column(align=True)

        row = layout.row()
        col = layout.column(align=True)
        col.label(text="Setup:", icon='UV_SYNC_SELECT')
        col = layout.column(align=True)
        col.prop(scn, "Micro_Distance", text="Distance", slider = True)
        col.prop(scn, "Micro_Influence", text="Influence", slider = True)
        col.prop(scn, "Micro_Radius", text="Radius", slider = True)
        if obj.Micro_Controller:
            col.operator("muscle.set", text="Set", icon = "OUTLINER_OB_LIGHT")
        col.prop(obj, "Micro_Controller", text="microController")
        if obj.Micro_Controller:
            row = layout.row()
            row = layout.row(align = True)
            split = row.split(factor=0.85)
            row = split.row()
            row.prop(obj, "Micro_Controller_Render")
            row.prop(obj, "Micro_Controller_View3D")
            row = layout.row(align = True)
            row = layout.row()
            row.prop(obj, "Skin_Tension", text="Skin Tension", slider = True)
            row = layout.row()
            

class XMusclePinning(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Muscle Pinning"
    bl_category = "x-muscle Sys"
    bl_idname = "OBJECT_PT_C_musclePinning"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    

    @classmethod
    def poll(self, context):

        if context.active_object == None:
            return False
        
        if context.mode == 'OBJECT': 
            if context.object and context.object.type == 'MESH' and context.active_object.select_get() == True and getattr(context.object, "Muscle_XID", None):
                return True
            else:
               if context.object and context.object.type == 'EMPTY' and context.object.empty_display_type == 'PLAIN_AXES' and context.active_object.select_get() == True:
                    for o in context.scene.objects:
                        if getattr(o, "Muscle_XID", None) and o.type == 'MESH' and getattr(o.custom, context.object.name, None):
                            return True
        #variable force update
        A = context.scene.Create_Type == 'AUTOAIM'
        if context.mode == 'POSE' and A:
            result = False

            count = 0
            for o in context.selected_objects:
                if getattr(o, "Muscle_XID", None) and o.select_get() == True and o.type == 'MESH':
                    count = count + 1
                    if count > 1:
                        result = False
                        break
                    result = True

            return (context.active_bone and len(context.selected_pose_bones) == 1 and result)

    def draw(self, context):

        if not getattr(context.object, "Muscle_XID", None) and not context.active_bone:
            if not context.object.type == 'EMPTY' and not context.object.empty_display_type == 'PLAIN_AXES' and not context.active_object.select_get() == True:
                return
        
        layout = self.layout
        obj = None
        for o in context.scene.objects:
            if getattr(o, "Muscle_XID", None) and o.type == 'MESH' and o.select_get() == True:
                obj = o
                break
            elif getattr(o, "Muscle_XID", None) and o.type == 'MESH' and getattr(o.custom, context.object.name, None):
                obj = o
                break
            else:
                obj = context.object
            
        row = layout.row()
        split = row.split(factor=0.85)
        row = split.row()
        row.prop(obj, "Pinning_Render")
        row.prop(obj, "Pinning_View3D")
        row = layout.row(align = True)
        row.label(text="")
        rows = 2
        row = layout.row()
        row.template_list("OBJECT_UL_items", "", obj, "custom", obj, "custom_index", rows=rows)

        col = row.column(align=True)
        col.operator("muscle.list_action", icon='ADD', text="").action = 'ADD'
        col.operator("muscle.list_action", icon='REMOVE', text="").action = 'REMOVE'

        row = layout.row()
        col = row.column(align=True)
        col.operator("muscle.select_toggle", icon="WORDWRAP_ON")
        col.operator("muscle.reset_selected", icon="UNPINNED")
        row = layout.row()
        row.prop(obj, "Pin_Size", text="Pins Display Size", slider = True)


class XMuscleDisplay(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Display"
    bl_category = "x-muscle Sys"
    bl_idname = "OBJECT_PT_C_muscleDispSys"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'

    @classmethod
    def poll(self, context):
       # return context.object and context.object.type == 'MESH'
        return (context.mode == 'OBJECT' or context.mode == 'POSE')

    def draw(self, context):

        current_theme = None
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        current_theme = space.shading
                        break


        layout = self.layout
        row = layout.row()
        row.label(text="Visibility:")
        row = layout.row()
        if context.object != None:

            if getattr(context.object, "Muscle_XID", None) and context.object.type == 'MESH' and context.active_object != None:
                if context.active_object.select_get() == True:
                    obj = context.object
                    scn = context.scene

                    mat = obj.active_material
                    row = layout.row(align = True)
                    if mat:
                        row.prop(current_theme, "object_outline_color", text = "", icon = 'COLOR')
                        row.prop(mat, "diffuse_color", text="", icon = 'COLOR')

                    row.operator("muscle.set_materials", text="Use Material Set", icon= 'MATERIAL')

                    col = layout.column(align = True)
                    row = col.row(align= True)
                    row.operator("muscle.visibility", text="Muscle", icon = 'RESTRICT_VIEW_OFF')
                    row.operator("muscle.dispnames", text="Names", icon = 'SORTALPHA')
                    row = col.row(align= True)
                    row.operator("muscle.xray", text="X-Ray", icon ="X")
                    row.operator("muscle.wireframe", text="Wireframe", icon = 'SHADING_WIRE')
                    row = col.row(align= True)
                    row.operator("muscle.pins_visibility", text="Pins", icon = 'RESTRICT_VIEW_OFF')
                    row.prop(scn, "Show_MicroControllers", icon = 'UV_SYNC_SELECT')
                    return

        row.operator("muscle.visibility", text="Muscles", icon = 'RESTRICT_VIEW_OFF')
        row = layout.row()
        row.operator("muscle.pins_visibility", text="Pins", icon = 'RESTRICT_VIEW_OFF')



class XMusculature(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Musculature"
    bl_category = "x-muscle Sys"
    bl_idname = "OBJECT_PT_D_musculatureSys"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_context = "objectmode"

    @classmethod
    def poll(self, context):
        if context.active_object == None:
            return False
        return context.object and context.object.type == 'MESH' and context.active_object.select_get() == True

    def draw(self, context):
        found = False
        layout = self.layout
        obj = context.object
        scn = context.scene

        for ob in  context.selected_objects:
            if ob.type == 'MESH':
                if  len(context.selected_objects) > 1 and getattr(ob, "Muscle_XID", None):
                    found = True
                    break
        col = layout.column(align=True)
        row = col.row(align = True)
        
        if context.object and context.object.type == 'MESH' and context.active_object.select_get() == True:

            #fixme error
            if context.active_object != None:
                skin = context.active_object
                m = len(skin.modifiers)
                #find skin
                fnd = False
                for n in range(0, m):
                    if skin.modifiers[n].type == 'SHRINKWRAP' and m > 0: 
                        if skin.modifiers[n].name.endswith("Shrinkwrap") and len(context.selected_objects) == 1:
                            for n in range(0, m):
                                if skin.modifiers[n].name.startswith("XMSL_Skin"):
                                    fnd = True
                                    break
                if fnd:
                    # row = col.row()
                    row.label(icon = 'MOD_SKIN', text="Skin Corrector:")
                    row = col.row(align = True)
                    split = row.split(factor=0.85)
                    row = split.row()
                    row.prop(skin, "Skin_Corrector_Render")
                    row.prop(skin, "Skin_Corrector_View3D")
                    row = col.row(align = True)
                    row = col.row()
                    row.label( text="Setup:")
                    row = col.row()
                    col.prop(skin, "Skin_Smooth_Tension", slider = True)
                    col.prop(skin, "Skin_Smooth_Iterations", slider = True)
                    row = col.row()
                
                
        if getattr(context.object, "Muscle_XID", None):
            row.label(text="X-Musculature:")
            row = col.row()
            box = row.box()
            #fixme error
            if context.active_object != None:
                if context.active_object.type == 'MESH':
                    layout.row().separator()
                    box.label(text=context.get_xmusculature)
            row = col.row()
            row.label(text = "")
            row = col.row()
            row = col.row()
            row.operator("muscle.selectsingle", text="Select Muscle", icon='POSE_HLT')
            row = col.row()
            row.operator("muscle.isolate", text="Isolate Selected", icon='ZOOM_SELECTED')
            row = col.row()
            row.operator("muscle.selectall", text="Select X-Musculature", icon='OBJECT_HIDDEN')
            row = col.row()
            row.operator("muscle.xmirror", text="X-Mirror Muscles", icon='MOD_MIRROR')
            row = col.row()
            row.label(icon = 'COLLAPSEMENU')
            row.prop(scn, "Keep_Over_Subdiv", text = "Keep over Subdiv")
            row = col.row()
            row.label(icon = 'MOD_SKIN')
            row.prop(scn, "Skin_Corrector", text = "Skin Corrector")
            row = col.row()

            row.label(icon = 'STICKY_UVS_LOC')
            row.prop(scn, "Allow_Duplications", text = "Allow Duplications")
            row = col.row()
            row.label(icon = 'ACTION')
            row.prop(scn, "Animate_Properties", text = "Animate Properties")
            row = col.row()

        if context.active_object != None:
            if context.active_object.type != 'MESH':
                found = False
            if getattr(context.active_object, "Muscle_XID", None):
                found = False

        if found:
            row.label(text="Active Object: " + obj.name)
            row = col.row()
            row.label(text="")
            row = col.row()
            row.label(icon = 'RECOVER_LAST')
            row.prop(scn, "Perform_Clean", text = "Perform Clean")
            row = col.row()

            row.label(icon = 'COLLAPSEMENU')
            row.prop(scn, "Keep_Over_Subdiv", text = "Keep over Subdiv")
            row = col.row()
            row.label(icon = 'MOD_SKIN')
            row.prop(scn, "Skin_Corrector", text = "Skin Corrector")
            row = col.row()

            row.label(icon = 'STICKY_UVS_LOC')
            row.prop(scn, "Allow_Duplications", text = "Allow Duplications")
            row = col.row()
            row.operator("muscle.apply_musculature", text="Apply Muscles to Body", icon = 'OUTLINER_OB_ARMATURE')
            row = col.row()
            # row.label(text="", icon='GROUP_VERTEX')
            #row.label("vertex group for current muscle on skin ")

def register():
    bpy.utils.register_class(XMuscleCreate)
    bpy.utils.register_class(XMuscleSystem)
    bpy.utils.register_class(OBJECT_UL_items)
    bpy.utils.register_class(XMusclePinning)
    bpy.utils.register_class(XMuscleMicroCtl)
    bpy.utils.register_class(XMuscleDisplay)
    bpy.utils.register_class(XMusculature)


def unregister():
    bpy.utils.unregister_class(XMuscleCreate)
    bpy.utils.unregister_class(XMuscleSystem)
    bpy.utils.unregister_class(XMusclePinning)
    bpy.utils.unregister_class(XMuscleMicroCtl)
    bpy.utils.unregister_class(OBJECT_UL_items)
    bpy.utils.unregister_class(XMuscleDisplay)
    bpy.utils.unregister_class(XMusculature)


