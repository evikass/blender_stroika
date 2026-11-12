import bpy
from bpy.types import Menu


def add_subdiv_modifier(subdiv_viewport_levels, subdiv_render_levels):
    obj = bpy.context.active_object

    if obj is None:
        print("warning: no active object")
        return

    if obj.type != "MESH":
        print("warning: the active object need to be a mesh")
        return

    bpy.ops.object.modifier_add(type="SUBSURF")
    bpy.context.object.modifiers["Subdivision"].levels = subdiv_viewport_levels
    bpy.context.object.modifiers["Subdivision"].render_levels = subdiv_render_levels


class MESH_OT_add_subdiv_mod(bpy.types.Operator):
    """Add a subdivision surf modifier to the active mesh"""

    bl_idname = "mesh.add_subdiv_mod"
    bl_label = "Add Subdivision Surf Modifier to the Active Mesh Object"
    bl_options = {"REGISTER", "UNDO"}

    subdiv_viewport_lvl: bpy.props.IntProperty(
        name="Subdiv Viewport",
        default=1,
        min=1,
        max=3,
        description="The Subdivision Levels applied in the Viewport",
    )

    subdiv_render_lvl: bpy.props.IntProperty(
        name="Subdiv Render",
        default=3,
        min=3,
        max=7,
        description="The Subdivision Levels applied during the Viewport",
    )

    def execute(self, context):
        add_subdiv_modifier(self.subdiv_viewport_lvl, self.subdiv_render_lvl)

        return {"FINISHED"}


class VIEW3D_MT_PIE_xmusclepie(Menu):
    # label is displayed at the center of the pie menu.
    bl_label = "X-Muscle Pie"

    def draw(self, context):
        layout = self.layout
        pie = layout.menu_pie()
        column = pie.split().column(align = True)
        scn = context.scene
    
        if context.mode == 'OBJECT':
            #CREATE
            column.label(text="Create:")
            column.prop(scn, "Muscle_Scale", text="Muscle Scale", slider = True)
            column.operator("muscle.add_basic_muscle", text="Add Basic Muscle", icon = 'RIGHTARROW_THIN')
            column.operator("muscle.add_muscle", text="Add Stylized Muscle", icon = 'RIGHTARROW_THIN')
            column.operator("muscle.add_strip_muscle", text="Add Strip Muscle", icon = 'RIGHTARROW_THIN')
            
            #TODO move to context.mode == OBJECT
            if context.object and context.object.type == 'MESH':
                if not getattr(context.object, "Muscle_XID", None):               
                    if len(context.selected_objects) > 1:
                        column = pie.split().column()
                        column.operator("muscle.convert_to_muscle", text="Convert Mesh to Muscle", icon = 'AUTOMERGE_ON')
                        column = pie.split().column(align = True)
                    else:
                        column.operator("muscle.convert_to_muscle", text="Convert Mesh to Muscle", icon = 'AUTOMERGE_ON')

            #Targeting Method
            column.label(text="Targeting:")
            column.prop(scn, "Create_Type", expand=True, text="Targeting Method")
            column = pie.split().column()

            # column.operator("muscle.convert_to_muscle", text="Convert XX", icon = 'RNA_ADD')
             #Muscle Name rename
            row = column.row(align=False)
            row.label(text="Name:")
            row = column.row(align = False)
            row.prop(scn, "Muscle_Name", text = "")
            row.operator("muscle.rename_muscle", text="", icon = 'OUTLINER_DATA_GP_LAYER')
            row = column.row(align = False)
            row.label(text="Prefix:")
            row.prop(scn, "Prefix", text="")
            row = column.row(align = False)
            row.label(text="Suffix:")
            row.prop(scn, "Suffix",text="")

            #affixes
            column = pie.split().column()
            #Display
            current_theme = None
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            current_theme = space.shading
                            break

            if context.object != None:

                if getattr(context.object, "Muscle_XID", None) and context.object.type == 'MESH' and context.active_object != None:
                    if context.active_object.select_get() == True:
                        obj = context.object

                        mat = obj.active_material
                        row = column.row(align = True)
                        row.label(text="Visibility:")
                        row = column.row(align = False)
                        if mat:
                            row.prop(current_theme, "object_outline_color", text = "", icon = 'COLOR')
                            row.prop(mat, "diffuse_color", text="", icon = 'COLOR')

                        row.operator("muscle.set_materials", text="Use Material Set", icon= 'MATERIAL')

                        row = column.row(align= False)
                        row.operator("muscle.visibility", text="Muscle", icon = 'RESTRICT_VIEW_OFF')
                        row.operator("muscle.dispnames", text="Names", icon = 'SORTALPHA')
                        row = column.row(align= False)
                        row.operator("muscle.xray", text="X-Ray", icon ="X")
                        row.operator("muscle.wireframe", text="Wireframe", icon = 'SHADING_WIRE')
                        row = column.row(align= False)
                        row.operator("muscle.pins_visibility", text="Pins", icon = 'RESTRICT_VIEW_OFF')
                        row.prop(scn, "Show_MicroControllers", icon = 'UV_SYNC_SELECT')

                if not getattr(context.object, "Muscle_XID", None) or context.active_object.select_get() == False:
                        row = column.row(align = False)
                        row.label(text="Visibility:")
                        row = column.row(align = False)
                        row.operator("muscle.visibility", text="Muscles", icon = 'RESTRICT_VIEW_OFF')
                        row.operator("muscle.pins_visibility", text="Pins", icon = 'RESTRICT_VIEW_OFF')



                if getattr(context.object, "Muscle_XID", None) and context.object.type == 'MESH' and context.active_object != None:
                    if context.active_object.select_get() == True:
                        obj = context.object

                        #starting from here
                        column = pie.split().column()
                        row = column.row(align = False)
                        #delete muscle
                        row.operator("muscle.delete_muscle", text="Delete Muscle", icon = "CANCEL")
                        # row = column.row(align = True)
                        #reset properties
                        row.operator("muscle.reset_props", text="Reset Properties", icon = "MODIFIER")
                        row = column.row(align = True)
                        row = column.row(align = True)
                        #select xmusculature
                        row.operator("muscle.selectall", text="Select X-Musculature", icon='OBJECT_HIDDEN')
                        #x-mirror
                        row.operator("muscle.xmirror", text="X-Mirror Muscles", icon='MOD_MIRROR')
                        row = column.row(align = True)
                        #smart update
                        row.prop(obj, "Dynamics_View3D", text="View3D Dynamics", icon= "ARMATURE_DATA")
                        row.operator("muscle.smart_update", text="Smart Update", icon = "FILE_REFRESH")
                        row = column.row(align = True)
                        row.prop(obj, "Base_Length_INT", slider = True)
                        row.prop(obj, "Volume_INT", slider = True)
                        row.prop(obj, "Muscle_Size", slider = True)
                        
                        row = column.row(align = True)
                        #flexor to ext
                        row.prop(obj, "Muscle_Type_INT", text="Flexor to Extensor")
                        #microCtrl
                        # row = column.row(align = True)
                        row.prop(obj, "Micro_Controller", text="microController")
                        # row = column.row(align = True)
                        #add pin
                        row.operator("muscle.list_action", icon='ADD', text="Add Pin").action = 'ADD'
                        
                        #edit muscle
                        #enter object into edit mode

                        #select
                        row = column.row(align = True)
                        row = column.row(align = True)
                        row.operator("muscle.selectsingle", text="Select Muscle", icon='POSE_HLT')
                        #isolate
                        row.operator("muscle.isolate", text="Isolate Selected", icon='ZOOM_SELECTED')
                    else:
                        column = pie.split().column()
                        row = column.row(align = False)
                        row.operator("muscle.selectall", text="Select X-Musculature", icon='OBJECT_HIDDEN')
                        row.operator("muscle.smart_update", text="Smart Update", icon = "FILE_REFRESH")
                    

        if context.mode == 'POSE':
            #CREATE
            column.label(text="Create:")
            column.prop(scn, "Muscle_Scale", text="Muscle Scale", slider = True)
            column.operator("muscle.add_basic_muscle", text="Add Basic Muscle", icon = 'RIGHTARROW_THIN')
            column.operator("muscle.add_muscle", text="Add Stylized Muscle", icon = 'RIGHTARROW_THIN')
            column.operator("muscle.add_strip_muscle", text="Add Strip Muscle", icon = 'RIGHTARROW_THIN')
            
            #TODO move to context.mode == OBJECT
            if context.object and context.object.type == 'MESH':
                if not getattr(context.object, "Muscle_XID", None):
                    column.operator("muscle.convert_to_muscle", text="Convert Mesh to Muscle", icon = 'AUTOMERGE_ON')

            #Targeting Method
            column.label(text="Targeting:")
            column.prop(scn, "Create_Type", expand=True, text="Targeting Method")
            column = pie.split().column()

            #add pin
            found = False
            for o in context.selected_objects:
                if getattr(o, "Muscle_XID", None):
                    found = True

            if scn.Create_Type == 'AUTOAIM' and found and len(context.selected_objects) == 2:
                column.operator("muscle.list_action", icon='ADD', text="Add Muscle Pin to bone").action = 'ADD'
                column = pie.split().column()

            # column.operator("muscle.convert_to_muscle", text="Convert XX", icon = 'RNA_ADD')
             #Muscle Name rename
            row = column.row(align=False)
            row.label(text="Name:")
            row = column.row(align = False)
            row.prop(scn, "Muscle_Name", text = "")
            row.operator("muscle.rename_muscle", text="", icon = 'OUTLINER_DATA_GP_LAYER')
            row = column.row(align = False)
            row.label(text="Prefix:")
            row.prop(scn, "Prefix", text="")
            row = column.row(align = False)
            row.label(text="Suffix:")
            row.prop(scn, "Suffix",text="")

            #affixes
            column = pie.split().column()
            #Display

            if not getattr(context.object, "Muscle_XID", None) or context.active_object.select_get() == False:
                    row = column.row(align = False)
                    row.label(text="Visibility:")
                    row = column.row(align = False)
                    row.operator("muscle.visibility", text="Muscles", icon = 'RESTRICT_VIEW_OFF')
                    row.operator("muscle.pins_visibility", text="Pins", icon = 'RESTRICT_VIEW_OFF')


global_addon_keymaps = []


def register():
    bpy.utils.register_class(MESH_OT_add_subdiv_mod)
    bpy.utils.register_class(VIEW3D_MT_PIE_xmusclepie)

    window_manager = bpy.context.window_manager
    if window_manager.keyconfigs.addon:
        keymap = window_manager.keyconfigs.addon.keymaps.new(name="3D View", space_type="VIEW_3D")
        keymap_item = keymap.keymap_items.new("wm.call_menu_pie", "X", "PRESS", ctrl=True, alt=True)
        keymap_item.properties.name = "VIEW3D_MT_PIE_xmusclepie"

        # save the key map to deregister later
        global_addon_keymaps.append((keymap, keymap_item))


def unregister():
    bpy.utils.unregister_class(VIEW3D_MT_PIE_xmusclepie)
    bpy.utils.unregister_class(MESH_OT_add_subdiv_mod)

    window_manager = bpy.context.window_manager
    if window_manager and window_manager.keyconfigs and window_manager.keyconfigs.addon:
        for keymap, keymap_item in global_addon_keymaps:
            keymap.keymap_items.remove(keymap_item)

    global_addon_keymaps.clear()
