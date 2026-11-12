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
import io
import bmesh
import math
from . import XMUSCLE_DATA
from mathutils import Vector, Matrix, Euler
from bpy.props import FloatProperty, BoolProperty, IntProperty, StringProperty, EnumProperty
from bpy.app.handlers import persistent

oldPref = ""
oldSuff = ""
oldName = ""
oldPivot = []
on_xMirror = False

#first free
def free(l):
    if l == []: return 0
    if l[0] > 1: return 1
    if l[-1] - l[0] + 1 == len(l): return l[-1] + 1
    for i in range(len(l)):
        try:
            if l[i+1] - l[i] > 1: break
        except IndexError:
            break
    return l[i] + 1

def setAffixes(newName, prefix, suffix, useAffixes):

    global oldPref
    global oldSuff
    global oldName


    for key, value in names.items():

        if oldPref != "" and value.startswith(oldPref):
            value = value[len(oldPref):]
        if oldSuff != "" and value.endswith(oldSuff):
            value = value[:-len(oldSuff)]

        if useAffixes:
            if value.startswith(prefix) and value.endswith(suffix):
                continue
            # microController renaming is done elsewhere...
            if key.startswith("micro_"):
                continue
            if value.count(oldName) > 0:
                name = prefix + newName  + value.replace(oldName,"",1)
            else:
                name = prefix + newName + value

        else:
            if prefix != "" and value.startswith(prefix):
                value = value[len(prefix):]
            if suffix != "" and value.endswith(suffix):
                value = value[:-len(suffix)]

            if value.count(oldName) > 0:
                name = newName + value.replace(oldName,"",1)
            else:
                name = newName + value

        names[key] = name

    oldPref = prefix
    oldSuff = suffix
    oldName = newName


def isRight(suffix):
        sign = [".", "_"]
        Left = [ "L", "LEFT"]
        Right = ["R", "RIGHT"]
        isRight = False

        for s in  sign:
            for l in Left:
                if suffix.lower() == (s + l.lower()):
                    isRight = False
            for r in Right:
                if suffix.lower() == (s + r.lower()):
                    isRight = True

        return isRight

#nobody's using regex
def counterSuffix(suffix):

        sign = [".", "_"]
        sfx = [ "L", "R", "l", "r", "left", "right", "Left", "Right", "LEFT", "RIGHT"]
        counterSuffix = ""

        if suffix == "":
            counterSuffix = ""
            return counterSuffix

        if isRight(suffix):
            for s in  sign:
                idx = 0
                for x in sfx:
                    if suffix == (s + x):
                        counterSuffix = (s + sfx[idx - 1])
                    idx = idx + 1
        else:
              for s in  sign:
                idx = 0
                for x in sfx:
                    if suffix == (s + x):
                        counterSuffix = (s + sfx[idx + 1])
                    idx = idx + 1

        return counterSuffix


def separateSuffixAndName(name, autonum = 4):
    sign = [".", "_"]
    sfx = [ "L", "R", "l", "r", "left", "right", "Left", "Right", "LEFT", "RIGHT"]

    suffix = ""
    if autonum != 0:
        name = name[:-autonum]

    for s in  sign:
        idx = 0
        for x in sfx:
            if name.endswith(s+x):
                suffix = s+x
                break

    if suffix != "":
        name = name[:-len(suffix)]

    return suffix, name


def getMirrorName(name, autonum = 4):

    suffix, name = separateSuffixAndName(name, autonum)

    if suffix != "":
        suffix = counterSuffix(suffix)

    mirrorName = name + suffix
    return mirrorName

def rename(muscleObj, mname, arname, bname, ctrlname):

    muscleObj.name = mname
    Arm = muscleObj.parent
    Arm.name = arname
    bone = Arm.pose.bones[muscleObj.parent_bone]
    bone.name = bname

    for b in Arm.pose.bones:
                for c in b.constraints:
                    if c.target.type == 'EMPTY':
                        ctrl = c.target

    ctrl.name = ctrlname
    return{'FINISHED'}


def decorateName(obj, suffix, pin = None, PinOnly = False):
    decor = ""
    count = 0
    noNumberExists = False
    numberLen = 4
    l = []

    if not PinOnly:
        for o in bpy.context.scene.objects:
            if getattr(o, "Muscle_XID", None) and obj != o:
                if o.name.startswith(names["muscleName"]):
                    if o.name.startswith(names["muscleName"]+".") and len(o.name) == len(names["muscleName"] + suffix)+numberLen and o.name.endswith(suffix):
                        if suffix == "":
                            new = o.name[len(names["muscleName"]+"."):]
                        else:
                            new = o.name[len(names["muscleName"]+"."):-len(suffix)]
                        if new != "":
                            l.append(int(new))
                    if len(o.name) == len(names["muscleName"] + suffix) and o.name.endswith(suffix):
                        l.append(0)
                        noNumberExists = True

    else:
        s,n = separateSuffixAndName(obj.name, autonum = 0)
        for o in bpy.context.scene.objects:
            if n + "_pin" in o.name and pin != o:

                if o.name.startswith(n + "_pin"):

                    if o.name.startswith(n + "_pin"+".") and len(o.name) == len(n+"_pin" + suffix)+numberLen and o.name.endswith(suffix):

                        if suffix == "":
                            new = o.name[len(n + "_pin"+"."):]
                        else:
                            new = o.name[len(n + "_pin"+"."):-len(suffix)]
                        if new != "":
                            l.append(int(new))
                    if len(o.name) == len(n + "_pin" + suffix) and o.name.endswith(suffix):
                        l.append(0)
                        noNumberExists = True


    l.sort()
    count = free(l)
    if noNumberExists:
        decor = ".%03d" % count + suffix
    else:
        decor = suffix
    if not PinOnly:
        rename(obj, names["muscleName"] + decor, names["musculatureName"]+ decor, names["mbName"]+ suffix, names["mctrlName"]+ decor)
    else:
        pin.name = n + "_pin" + decor



names = dict(muscleName = "",
    micro_SysName = "System_Micro",
    micro_ctrlName = "micro_ctrl",
    micro_rtbName = "_root_bone",
    micro_empbName = "_empty_bone",
    micro_tgtbName = "_target_bone",
    micro_rt_copyLocConstName = " Copy Location Arm",
    micro_rt_copyRotConstName = " Copy Rotation Arm",
    micro_rt_copyScaConstName = " Copy Scale Arm ",
    micro_emp_copyLocConstName = " Copy Location micrctrl",
    micro_emp_copyRotConstName = " Copy Rotation micrctrl",
    micro_emp_copyScaConstName = " Copy Scale micrctrl",
    micro_tgt_copyLocConstName = " Copy Location emp",
    micro_tgt_copyRotConstName = " Copy Rotation emp",
    micro_tgt_copyScaConstName = " Copy Scale emp",
    micro_HookName = " Hook",

    musculatureName = " System",
    mctrlName = "_ctrl",
    vertexGroupName = "_jiggle",
    softBodyName = " Softbody",
    muscleSubsurfName = " Subsurf",

    muscleThickName = "_thick",
    muscleThinName =   "_thin",
    boneCustomShapeName = " boneCustomShape",
    muscleShapeKeyName = " ShapeKey",
    mbName = "_bone",
    extbName = "_extensor_bone",
    flxbName = "_flexor_bone",
    extctrlbName = "_extensor_ctrl_bone",
    cntrbName = "_counter_bone",
    ext_ctrl_copyLocConstName = " Copy Location ext",
    ext_copyLocConstName = " Copy Location cntr",
    ext_copyRotConstName = " Copy Rotation ext",
    msl_copyRotConstName = " Copy Rotation flx",
    msl_copyLocConstName = " Copy Location flx",
    msl_stretchToCtrlConstName = " Stretch To ctrl",
    msl_stretchToExtConstName = " Stretch To ext" ,
    flx_stretchToConstName = " Stretch To ctrl",)

defaultGroupName = "XMusculature"
defaultMaterialName = "XMat_"
material_diffuse = (0.8, 0.0211635, 0.00853679, 1.0)

def Muscle_Size(self, context):
    obj = self

    if not getattr(obj, "Muscle_XID", None):
        return
    if obj != None:
        if obj.type == 'MESH':
            scale = obj.Muscle_Size
            obj.scale = Vector((obj.scale.x, scale * 1.7, scale * 1.7))

def Base_Length_INT(self, context):

    obj = self

    if not getattr(obj, "Muscle_XID", None):
        return
    if obj != None:
        if obj.type == 'MESH':
            if obj.parent != None:
                if obj.parent.type == 'ARMATURE':
                    obj.parent.Base_Length = obj.Base_Length_INT
                    #force-refresh workaround
                    if obj.parent.select_get() == True and obj != context.active_object:
                        return

                    oldSelect = obj.parent.select_get()
                    obj.parent.select_set(True)
                    override = get_override( 'VIEW_3D', 'WINDOW' )
                    tmp_override(override, 'translate')
                    obj.parent.select_set(oldSelect)


def update_pin_name(self, context):
    global on_xMirror
    objects = bpy.data.objects
   

    if self.name == self.oldname or on_xMirror:
        return
    
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
      
        #rename vg
        for v in obj.vertex_groups.keys(): #TODO validate
            if v == self.oldname:
                obj.vertex_groups[self.oldname].name = self.name

        for m in obj.modifiers: 
            if m.type == 'HOOK':
                if m.name == self.oldname:
                    m.name = self.name
                if m.object.name == self.oldname:
                    m.object.name = self.name
                    m.vertex_group = self.name

    self.oldname= self.name


def Muscle_Pin_Size(self, context):
    objects = bpy.data.objects
    
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
       
        for c in obj.custom:
            e = bpy.data.objects[c.name] #TODO validate
            if e.type == 'EMPTY':
                e.empty_display_size = obj.Pin_Size
             

def Skin_Corrector(self, context):
    obj = context.object
    scn = self
    skin = None

    for o in scn.objects:
        for m in o.modifiers:
            if m.type == 'SHRINKWRAP':
                if m.target != None:
                    if m.target.name == obj.name:
                        skin = o
                        break


    #add corrective_smooth if enabled
    if scn.Skin_Corrector and skin:

        found = False
        for m in skin.modifiers:
            if m.name == "XMSL_SkinCorrector" and m.type == 'CORRECTIVE_SMOOTH':
                found = True
                break
        if not found:
            m = skin.modifiers.new(type='CORRECTIVE_SMOOTH', name= "XMSL_SkinCorrector")
            
        m.show_expanded = False
        
        m = len(skin.modifiers)
        for n in range(0, m):

            if skin.modifiers[n].type == 'SHRINKWRAP' and m > 0:
                bpy.ops.object.modifier_move_to_index(modifier="XMSL_SkinCorrector", index=n)
            if skin.modifiers[n].type == 'SUBSURF' and m > 0:
                bpy.ops.object.modifier_move_to_index(modifier="XMSL_SkinCorrector", index=n-1)

        addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'show_viewport' , "Skin_Corrector_View3D")
        addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'show_render' , "Skin_Corrector_Render")

        addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'factor' , "Skin_Smooth_Tension")
        addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'iterations' , "Skin_Smooth_Iterations")
        skin.Skin_Corrector_View3D = 1
        skin.Skin_Corrector_Render = 1
        skin.Skin_Smooth_Tension = 0.5
        skin.Skin_Smooth_Iterations = 5


def Volume_INT(self, context):
    obj = self
    if not getattr(obj, "Muscle_XID", None):
        return
    if obj != None:
        if obj.type == 'MESH':
            if obj.parent != None:
                if obj.parent.type == 'ARMATURE':
                    obj.parent.Volume = obj.Volume_INT
                    #force-refresh workaround
                    if obj.parent.select_get() == True and obj != context.active_object:
                        return

                    oldSelect = obj.parent.select_get()
                    obj.parent.select_set(True)
                    override = get_override( 'VIEW_3D', 'WINDOW' )
                    tmp_override(override, 'translate')
                    obj.parent.select_set(oldSelect)


def Muscle_Type_INT(self, context):
    obj = self
    if not getattr(obj, "Muscle_XID", None):
        return
    if obj != None:
        if obj.type == 'MESH':
            if obj.parent != None:
                if obj.parent.type == 'ARMATURE':
                    obj.parent.Muscle_Type = obj.Muscle_Type_INT
                    #force-refresh workaround
                    if obj.parent.select_get() == True and obj != context.active_object:
                        return
                    oldSelect = obj.parent.select_get()
                    obj.parent.select_set(True)
                    override = get_override( 'VIEW_3D', 'WINDOW' )
                    tmp_override(override, 'translate')
                    obj.parent.select_set(oldSelect)


def temp_unhide_bone(rig, bname):
    version, ver_major, ver_minor = bpy.app.version
    hide_bone_id = None

    if version >= 4:
        for x in rig.data.collections:
            if x.name in rig.data.bones[bname].collections:
                if rig.data.collections[x.name].is_visible == False:
                    rig.data.collections[x.name].is_visible = True
                    hide_bone_id = x.name
                    return hide_bone_id
    else:
        idx = 0
        for x in rig.data.layers:
            if rig.data.bones[bname].layers[idx] == True:
                if rig.data.layers[idx] == False:
                    rig.data.layers[idx] = True
                    hide_bone_id = idx
                    return hide_bone_id
            idx = idx + 1
               
    return hide_bone_id


def Micro_Controller(self, context):

    if self == None or self.type == 'ARMATURE' or self.type == 'EMPTY':
        return

    scn = context.scene
    obj = self

    if not getattr(obj, "Muscle_XID", None):
        return
    else:
        if obj != None:
            if obj.type == 'MESH':

                obj.Micro_Controller_View3D = obj.Micro_Controller
                obj.Micro_Controller_Render = obj.Micro_Controller


                if obj.Micro_Controller == True:

                    #make sure micro_controllers are always applied before pinning
                    oldPinning = None
                    oldPinning = obj.Pinning_View3D

                    obj.Pinning_View3D = False

                    microSys = None
                    mic_ctrl = None

                    #search for existing microcontrollers
                    sname = obj.parent.name.replace("System", names["micro_SysName"])
                    mname = obj.parent.name.replace("System", names["micro_ctrlName"])

                    oldActive = bpy.context.active_object
                    oldSelection = bpy.context.selected_objects
                    rig = None
                    oldPosePosition = None
                    oldMode = None
                    oldLocation = None
                    oldRotation = None
                    oldScale = None
                    oldHide = None
                    oldSubView = None
                    skin = None

                    #in case of active muscle corruption search for other armature connections in selected muscles
                    muscleSys = obj.parent
                    if muscleSys != None:
                        rigboneName = muscleSys.parent_bone

                        if rigboneName != None:
                            for b in muscleSys.pose.bones:
                                for c in b.constraints:
                                    if c.target.type == 'EMPTY':
                                        ctrl = c.target

                            if ctrl.parent != None and rigboneName != "":
                                if ctrl.parent.type == 'ARMATURE':
                                    rig = ctrl.parent
                                    oldPosePosition = rig.data.pose_position

                                    oldMode = rig.mode

                                    #fixing the controllers  orientation issues
                                    oldLocation = Vector(rig.location)
                                    oldRotation = Euler(rig.rotation_euler)
                                    oldScale = Vector(rig.scale)
                                    oldHide = rig.hide_viewport
                                    rig.hide_viewport = False
                                    bpy.ops.object.select_all(action='DESELECT')
                                    bpy.context.view_layer.objects.active  =rig
                                    rig.select_set(True)
                                    bpy.ops.object.mode_set(mode='OBJECT')
                                    bpy.ops.object.rotation_clear(clear_delta=False)
                                    bpy.ops.object.location_clear(clear_delta=False)
                                    bpy.ops.object.scale_clear(clear_delta=False)

                                    rig.data.pose_position = 'REST'
                                    rig.select_set(False)

                    #ends

                    #preventing duplication
                    if sname in scn.objects.keys() or mname in scn.objects.keys(): #TODO validate
                        deleteMicroCtrl(obj, sname, mname)

                    size = obj.dimensions[0] * 0.1
                    if oldScale != None:
                        #todo sophisticated pattern here
                        scale = obj.Muscle_Size * size * oldScale.x

                    else:
                        scale = obj.Muscle_Size * size


                    muscleHeight =  Vector((0.0,0.0,3.0*scale))
                    override = get_override( 'VIEW_3D', 'WINDOW' )
                    oldCursor = Vector(scn.cursor.location)
                    tmp_override(override, 'snap_cursor_to_selected')
                    mic_ctrl = createController(names["micro_ctrlName"], None, scale, 'SPHERE', False)
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.parent_no_inverse_set()
                    mic_ctrl.location = mic_ctrl.location + muscleHeight
                    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

                    #parenting mic_ctrl to muscle mesh triangle
                    if not dp(mic_ctrl, obj):
                        bpy.ops.object.delete()
                        scn.cursor.location = oldCursor
                        #reverting changes before unsuccessful exiting
                        if oldPosePosition != None:
                            rig.data.pose_position = oldPosePosition
                        if oldMode != None:
                            bpy.context.view_layer.objects.active = rig
                            rig.select_set(True)
                            bpy.ops.object.mode_set(mode='OBJECT')

                        if oldLocation != None:
                            rig.location = oldLocation
                        if oldRotation != None:
                            rig.rotation_euler = oldRotation
                        if oldScale != None:
                            rig.scale = oldScale

                        bpy.ops.object.select_all(action='DESELECT')

                        return

                    scn.cursor.location = oldCursor

                    NAME_COL = 0
                    boneTable = [(names["micro_rtbName"], None, (0.1,0,0)),
                                 (names["micro_empbName"], None, (0.1,0,0)),
                                 (names["micro_tgtbName"], names["micro_rtbName"], (0.1,0,0)),]



                    location = Vector((0,0,0))
                    #create and setup microSys armature
                    microSys = createArmature(names["micro_SysName"], boneTable, location)

                    root_pBone = microSys.pose.bones[boneTable[NAME_COL][NAME_COL]]
                    emptyConst_pBone = microSys.pose.bones[boneTable[NAME_COL+1][NAME_COL]]
                    target_pBone = microSys.pose.bones[boneTable[NAME_COL+2][NAME_COL]]

                    bpy.ops.object.mode_set(mode='EDIT')
                    microSys.data.edit_bones[root_pBone.name].head = (0,0,0)
                    microSys.data.edit_bones[root_pBone.name].tail = (1,0,0)
                    microSys.data.edit_bones[target_pBone.name].head = (0,0,0)
                    microSys.data.edit_bones[target_pBone.name].tail = (0,1,0)
                    microSys.data.edit_bones[emptyConst_pBone.name].head = (0,0,0)
                    microSys.data.edit_bones[emptyConst_pBone.name].tail = (1,0,0)
                    bpy.ops.object.mode_set(mode='OBJECT')

                    #total 180degree translation

                    bpy.ops.object.select_all(action='DESELECT')
                    microSys.select_set(True)
                    bpy.ops.transform.rotate(value=3.14159, orient_axis='Z', constraint_axis=(False, False, True), orient_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1)

                    #exit from rig pose mode after parenting
                    bpy.ops.object.mode_set(mode='OBJECT')
                    bpy.ops.object.select_all(action='DESELECT')
                    #ctrl.parent_bone = ctrlParent  # needs reparenting and switching parent name suffix
                    #parent micSys to the rig if available (separate function) (in restmode)
                    microSys.select_set(True)
                    if ctrl.parent != None and rigboneName != "":

                        # just for convenience  - if bone layer was hidden make it visible to succeed
                    

                        hide_bone_id = temp_unhide_bone(rig, rigboneName)
                        #parent System_Micro to Armature's same bone as parented muscleSys
                        b = rig.pose.bones[rigboneName].bone
                        rig.pose.bones[rigboneName].select = True
                        # b.select = True #TODO validate
                        rig.data.bones.active = b
                        bpy.context.view_layer.objects.active  = rig
                        bpy.ops.object.parent_set(type='BONE')

                        if hide_bone_id != None:
                            version, ver_major, ver_minor = bpy.app.version
                            if version >= 4:
                                rig.data.collections[hide_bone_id].is_visible = False
                            else:
                                rig.data.layers[hide_bone_id] = False

                    #set root constraints target to Armature
                    rt_cpLoConst = addConstraints(root_pBone, type = 'COPY_LOCATION', name = names["micro_rt_copyLocConstName"], target = rig)
                    rt_cpRotConst = addConstraints(root_pBone, type = 'COPY_ROTATION', name = names["micro_rt_copyRotConstName"], target = rig)
                    rt_scaConst = addConstraints(root_pBone, type = 'COPY_SCALE', name = names["micro_rt_copyScaConstName"], target = rig)

                    emp_cpLoConst = addConstraints(emptyConst_pBone, type = 'COPY_LOCATION', name = names["micro_emp_copyLocConstName"], target = mic_ctrl)
                    emp_cpRotConst = addConstraints(emptyConst_pBone, type = 'COPY_ROTATION', name = names["micro_emp_copyRotConstName"], target = mic_ctrl)
                    emp_scaConst = addConstraints(emptyConst_pBone, type = 'COPY_SCALE', name = names["micro_emp_copyScaConstName"], target = mic_ctrl)

                    tgt_cpLoConst = addConstraints(target_pBone, type = 'COPY_LOCATION', name = names["micro_tgt_copyLocConstName"], target = microSys, subtarget = emptyConst_pBone.name, target_space ='LOCAL', owner_space ='LOCAL')
                    tgt_cpRotConst = addConstraints(target_pBone, type = 'COPY_ROTATION', name = names["micro_tgt_copyRotConstName"], target = microSys, subtarget = emptyConst_pBone.name, target_space ='LOCAL', owner_space ='LOCAL')
                    tgt_scaConst = addConstraints(target_pBone, type = 'COPY_SCALE', name = names["micro_tgt_copyScaConstName"], target = microSys, subtarget = emptyConst_pBone.name, target_space ='LOCAL', owner_space ='LOCAL')

                    #move to different layer?

                    ####todo delayed parenting rootBone (when skinning) if muscle is not bound to armature ?
                    #setup the micSys rootbone constraint if muscle is parented


                    ##### modify existing code

                    #####add microSys to xmirror ?

                    microSys.select_set(True)
                    mic_ctrl.select_set(True)

                    #add  microController's objects to the muscle group
                    ob = context.selected_objects

                    xmusclea = bpy.data.collections.get(obj.name, None)
                    # bpy.ops.collection.objects_remove_all()

                    for o in ob:
                        for c in o.users_collection:
                            c.objects.unlink(o)

                    for o in ob:
                        if not xmusclea in o.users_collection:
                            xmusclea.objects.link(o)

                    # for group in bpy.data.groups:
                    #   if muscleSys.name in group.objects:
                    #       for o in ob:
                    #         group.objects.link(o)


                    #hide microSys armature
                    microSys.hide_viewport = True
                    root_pBone.hide = True
                    emptyConst_pBone.hide = True
                    target_pBone.hide = True


                    #add microSys to naming system
                    microSys.name = muscleSys.name.replace("System", microSys.name)
                    mic_ctrl.name = muscleSys.name.replace("System", mic_ctrl.name)

                    bpy.context.view_layer.objects.active = mic_ctrl

                    for o in scn.objects:
                        for m in o.modifiers:
                            if m.type == 'SHRINKWRAP':
                                if m.target != None:
                                    if m.target.name == obj.name:
                                        skin = o
                                        break

                    if skin != None:
                        #move shrinkwrap modifier above Subsurface Modifier Keep_Over_Subdiv feature
                        if scn.Keep_Over_Subdiv:
                            m = len(skin.modifiers)
                            #find subdiv
                            for n in range(0, m):
                                if skin.modifiers[n].type == 'SUBSURF' and m > 0:
                                    oldSubView = skin.modifiers[n].show_viewport
                                    skin.modifiers[n].show_viewport = False
                                    mod = skin.modifiers[n]
                                    break
                    removeHookSetAttributesFrommicroCtrl(mic_ctrl)
                    addHookSetAttributesTomicroCtrl(obj, mic_ctrl)
                    addMicroToSkin(obj, mic_ctrl, microSys, scale, mirror = False)


                    if oldPosePosition != None:
                        rig.data.pose_position = oldPosePosition
                    if oldMode != None:
                        bpy.context.view_layer.objects.active = rig
                        rig.select_set(True)
                        bpy.ops.object.mode_set(mode='OBJECT')

                    if oldLocation != None:
                        rig.location = oldLocation
                    if oldRotation != None:
                        rig.rotation_euler = oldRotation
                    if oldScale != None:
                        rig.scale = oldScale

                    if oldHide != None:
                        rig.hide_viewport = oldHide

                    bpy.ops.object.select_all(action='DESELECT')

                    for o in oldSelection:
                        o.select_set(True)

                    bpy.context.view_layer.objects.active = oldActive

                    if oldSubView:
                        mod.show_viewport = oldSubView


                    obj.Pinning_View3D = oldPinning

                elif obj.Micro_Controller == False:

                    sname = obj.parent.name.replace("System", names["micro_SysName"])
                    mname = obj.parent.name.replace("System", names["micro_ctrlName"])

                    deleteMicroCtrl(obj, sname, mname)



def deleteMicroCtrl(obj, sysName, mName):
    scn = bpy.context.scene
    microSys = None
    mic_ctrl = None

    for o in scn.objects:
        if o.name == sysName:
            microSys = o

        if o.name == mName:
            mic_ctrl = o

        if microSys != None and mic_ctrl != None:
            break

    if microSys != None and mic_ctrl != None:
        #delete microSys
        if microSys.type == 'ARMATURE' and mic_ctrl.type == 'EMPTY':

            removeMicroFromSkin(obj, microSys)

            microSys.hide_viewport = False
            mic_ctrl.hide_viewport = False
            microSys.select_set(False)
            mic_ctrl.select_set(False)

            #delete microcontroller
            # for group in bpy.data.groups:
            #     if microSys.name in group.objects:
            #         group.objects.unlink(microSys)

            # for group in bpy.data.groups:
            #     if mic_ctrl.name in group.objects:
            #         group.objects.unlink(mic_ctrl)
            xmusclea = bpy.data.collections.get(obj.name, None)

            if xmusclea in microSys.users_collection:
                xmusclea.objects.unlink(microSys)
            if xmusclea in mic_ctrl.users_collection:
                xmusclea.objects.unlink(mic_ctrl)

            bpy.data.objects.remove(microSys, do_unlink= True)
            bpy.data.objects.remove(mic_ctrl, do_unlink= True)
    else:
        return


def addMicroToSkin(muscleObj, mic_ctrl, micSys, scale, mirror = False):

    scn = bpy.context.scene

    skin = None

    for obj in scn.objects:
        for m in obj.modifiers:
            if m.type == 'SHRINKWRAP':
                if m.target != None:
                    if m.target.name == muscleObj.name:
                        skin = obj

                        break
    if skin == None:
        return

    if micSys == None:
        return


    #add apply button (optional)

    #we need to override the context of our operator
    bpy.context.view_layer.objects.active = mic_ctrl

    #needed for x-mirror empty hookpoint swap
    oldxloc = Vector(mic_ctrl.location)
    oldxrot = Euler(mic_ctrl.rotation_euler)
    oldxscale = Vector(mic_ctrl.scale)
    #get tension from muscleObj
    oldxtens = muscleObj.Skin_Tension

    override = get_override( 'VIEW_3D', 'WINDOW' )
    #get and store the original origin for mirring
    oldCursor = Vector(scn.cursor.location)
    tmp_override(override, 'snap_cursor_to_active')

    #doublecheck the mirroring
    xname = None
    xname = getMirrorName(mic_ctrl.name, autonum = 0)
    for x in scn.objects:
        if x.name == xname:
            m = True
            break
        else:
            m = False

    if mirror == True:
        mirror = m


    if mirror:
        executeHookSetAttributes(muscleObj, mic_ctrl)

    else:

        #create a UVSphere in the  micro controller's coordinates with size scaled on influence
        # bpy.ops.mesh.primitive_uv_sphere_add(size=scale, view_align=False, enter_editmode=False, location = scn.cursor.location)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=scale, enter_editmode=False, location = scn.cursor.location)
        brush = bpy.context.object
        brush.name = muscleObj.name + "_dynamic"
        #match uvsphere to empty mic_ctrl coordinates and scale
        updateHookSetAttributes(muscleObj, mic_ctrl)
        bpy.ops.transform.resize(value=mic_ctrl.scale, constraint_axis=(False, False, False), orient_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1)
        #make the UVSphere as dynamic paint brush and setup (hidden)
        bpy.context.object.modifiers.new(type='DYNAMIC_PAINT', name="Dynamic Paint")
        brush.modifiers["Dynamic Paint"].ui_type = 'BRUSH'
        bpy.ops.dpaint.type_toggle(type='BRUSH')
        brush.modifiers["Dynamic Paint"].brush_settings.paint_source = 'VOLUME_DISTANCE'

        brush.modifiers["Dynamic Paint"].brush_settings.paint_distance = scn.Micro_Distance


    bpy.ops.object.select_all(action='DESELECT')
    #make the skin as dynamic paint canvas and setup (vertex groups)
    skin.select_set(True)
    skinOldHide = skin.hide_viewport
    skin.hide_viewport = False
    bpy.context.view_layer.objects.active = skin

    #there is no need for mesh cloning when mirroring vertex groups
    if not scn.Allow_Duplications:

        for v in skin.vertex_groups.keys(): #TODO validate
            if v == micSys.name:
                skin.vertex_groups.active_index = skin.vertex_groups[micSys.name].index
                bpy.ops.object.vertex_group_remove()

                break
    if mirror:
        skin.vertex_groups.active_index =  skin.vertex_groups[getMirrorName(micSys.name, autonum = 0)].index
        bpy.ops.object.vertex_group_copy()

        bpy.ops.object.vertex_group_mirror(flip_group_names= False, use_topology = False)
        skin.vertex_groups.active.name = micSys.name

    else:

        cloned = None
        #duplicate object if there are shapekeys found or multires assigned

        multires = False
        multiname = ""
        for m in skin.modifiers:
            if m.type == 'MULTIRES':
                multires = True
                multiname = m.name
                break
        if skin.data.shape_keys != None or multires == True:
            bpy.ops.object.duplicate()
            cloned = bpy.context.active_object
            cloned.select_set(True)

        if skin.data.shape_keys != None:
            for k in cloned.data.shape_keys.key_blocks:
                cloned.shape_key_remove(k)

        #get rid of multires that causes error exception
        if multires == True:
            bpy.ops.object.modifier_remove(modifier = multiname)

        #only if shapekeys exist all further operations are taken on mesh copy
        #otherwise original mesh is processed

        #autoweight setup
        dynaPaint = bpy.context.object.modifiers.new(type='DYNAMIC_PAINT', name="Dynamic Paint")
        bpy.ops.dpaint.type_toggle(type='CANVAS')

        if cloned != None:
            dynaPaint = cloned.modifiers["Dynamic Paint"]
        else:
            dynaPaint = skin.modifiers["Dynamic Paint"]

        dynaPaint.canvas_settings.canvas_surfaces.active.surface_type = 'WEIGHT'
        #add VertexGroup name it micSys.name
        dynaPaint.canvas_settings.canvas_surfaces.active.output_name_a = micSys.name
        dynaPaint.canvas_settings.canvas_surfaces.active.use_dissolve = True
        dynaPaint.canvas_settings.canvas_surfaces.active.use_dissolve_log = False
        dynaPaint.canvas_settings.canvas_surfaces.active.dissolve_speed = 1

        dynaPaint.canvas_settings.canvas_surfaces.active.brush_influence_scale = scn.Micro_Influence
        dynaPaint.canvas_settings.canvas_surfaces.active.brush_radius_scale = scn.Micro_Radius
        bpy.ops.dpaint.output_toggle(output='A')

        #force dynamic paint draw
        oldFrame = scn.frame_current
        scn.frame_current = scn.frame_start

        bpy.ops.object.mode_set(mode='WEIGHT_PAINT', toggle = False)

        version, ver_major, ver_minor = bpy.app.version
        if version == 2 and ver_major >= 90 or version >= 3:
            bpy.ops.object.modifier_apply(modifier = dynaPaint.name)
        else:
            bpy.ops.object.modifier_apply(apply_as='DATA', modifier = dynaPaint.name)

        bpy.ops.object.mode_set(mode='OBJECT', toggle = False)
        scn.frame_current = oldFrame

        if cloned != None:
            #transfer vertex groups and weights from cloned model to original
            weights = {}
            #temporarly store  weights
            weights = getWeightSet(cloned, cloned.vertex_groups[micSys.name], weights)

            #clean up the scene
            bpy.ops.object.delete()
            bpy.ops.object.select_all(action='DESELECT')
            #bringing back focus to the original body mesh again
            skin.select_set(True)
            bpy.context.view_layer.objects.active = skin

            #include all new vertices to new vertex group
            bpy.ops.object.vertex_group_add()
            #assign weights to the skin
            skin.vertex_groups.active.name = micSys.name
            skin.vertex_groups[micSys.name].add(list(weights.keys()), 1.0, 'REPLACE') #TODO validate

            setWeights(skin, skin.vertex_groups[micSys.name], weights)

    #add and configure hook modifier to the skin mesh if exist (separate function)
    hook = bpy.context.object.modifiers.new(type='HOOK', name=micSys.name)

    #todo move modifier to the top (above armature)
    hook.show_expanded = False

    #todo sorting modifiers stock algorithm here for better symmetry

    while skin.modifiers[0].name != hook.name:
        bpy.ops.object.modifier_move_up(modifier=hook.name)


    #set target
    hook.object = micSys
    hook.subtarget = micSys.pose.bones[names["micro_tgtbName"]].name

    #add  microController's vertex group to the skin mesh if available  (separate function) //add new vertex group to hook modifier
    hook.vertex_group = micSys.name

    #add influence slider property in the panel to define weight strength (optional)
    hook.strength = mic_ctrl.HookSet_Attributes[0].tension


    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.hook_reset(modifier=hook.name)
    #todo add property microController

    bpy.ops.object.mode_set(mode='OBJECT')

    #add enable in render viewport drivers
    addDriver(skin.modifiers[hook.name], muscleObj, 'strength' , "Skin_Tension")
    addDriver(skin.modifiers[hook.name], muscleObj, 'show_viewport' , "Micro_Controller_View3D")
    addDriver(skin.modifiers[hook.name], muscleObj, 'show_render' , "Micro_Controller_Render")

    bpy.ops.object.select_all(action='DESELECT')

    if not mirror:
        brush.select_set(True)
        bpy.context.view_layer.objects.active = brush
        bpy.ops.object.delete(use_global=False)

    else:
        #bring back microController's  original orientation and location
        mic_ctrl.location = oldxloc
        mic_ctrl.rotation_euler = oldxrot
        mic_ctrl.scale = oldxscale
        muscleObj.Skin_Tension = oldxtens

    bpy.ops.object.select_all(action='DESELECT')

    if skin != None:
        skin.hide_viewport = skinOldHide

    scn.cursor.location = oldCursor


def removeMicroFromSkin(muscleObj, micSys):
    skinOldHide = False
    oldSubView = False
    if micSys.type != 'ARMATURE':
        return

    scn = bpy.context.scene
    skin = None
    for obj in scn.objects:
        for m in obj.modifiers:
            if m.type == 'SHRINKWRAP':
                if m.target != None:
                    if m.target.name == muscleObj.name:
                        skin = obj
                        break

    if skin == None:
        return

    if skin != None:

        skinOldHide = skin.hide_viewport
        #move shrinkwrap modifier above Subsurface Modifier Keep_Over_Subdiv feature
        if scn.Keep_Over_Subdiv:
            m = len(skin.modifiers)
            #find subdiv
            for n in range(0, m):
                if skin.modifiers[n].type == 'SUBSURF' and m > 0:
                    oldSubView = skin.modifiers[n].show_viewport
                    skin.modifiers[n].show_viewport = False   #for performance
                    mod = skin.modifiers[n]
                    break

    bpy.ops.object.mode_set(mode='OBJECT')
    #remove enable in render and viewport drivers before modifier is removed
    for m in skin.modifiers.keys(): #TODO validate
        if m == micSys.name:
            for d in skin.animation_data.drivers:
                if d.data_path.startswith("modifiers["+'"'+micSys.name+'"'):
                    skin.driver_remove(d.data_path, -1)

    #remove hook modifiers from the skin mesh
    bpy.ops.object.select_all(action='DESELECT')
    skin.select_set(True)
    skin.hide_viewport = False
    bpy.context.view_layer.objects.active = skin
    bpy.ops.object.modifier_remove(modifier=micSys.name)

    #remove microController's vertex group from the skin mesh
    for v in skin.vertex_groups.keys():     #TODO validate
        if v == micSys.name:
            skin.vertex_groups.active_index = skin.vertex_groups[v].index
            bpy.ops.object.vertex_group_remove()

    if oldSubView:
        mod.show_viewport = oldSubView


    #in case if it's weightpaint mode make sure skin is in Object Mode
    bpy.ops.object.mode_set(mode='OBJECT', toggle = False)
    bpy.ops.object.select_all(action='DESELECT')

    if skin != None:
        skin.hide_viewport = skinOldHide


def setMicroController(obj, context):

        scn = context.scene
        microSys = None
        mic_ctrl = None
        sname = obj.parent.name.replace("System", names["micro_SysName"])
        mname = obj.parent.name.replace("System", names["micro_ctrlName"])

        size = obj.dimensions[0] * 0.1
        scale = obj.Muscle_Size * size

        for o in scn.objects.keys(): #TODO validate
            if sname == o:
                microSys = scn.objects[o]
            if mname == o:
                mic_ctrl = scn.objects[o]

        removeMicroFromSkin(obj, microSys)
        rig = None
        oldPosePosition = None
        oldLocation = None
        oldRotation = None
        oldScale = None
        oldHide = None
        oldMode = None
        #in case of active muscle corruption search for other armature connections in selected muscles
        muscleSys = obj.parent
        if muscleSys != None:
            rigboneName = muscleSys.parent_bone

            if rigboneName != None:
                for b in muscleSys.pose.bones:
                    for c in b.constraints:
                        if c.target.type == 'EMPTY':
                            ctrl = c.target

                if ctrl.parent != None and rigboneName != "":
                    if ctrl.parent.type == 'ARMATURE':
                        rig = ctrl.parent
                        oldPosePosition = rig.data.pose_position
                        oldMode = rig.mode
                        oldHide = rig.hide_viewport
                        rig.hide_viewport = False

                        #fixing the controllers  orientation issues
                        oldLocation = Vector(rig.location)
                        oldRotation = Euler(rig.rotation_euler)
                        oldScale = Vector(rig.scale)
                        #in case if it's weightpaint mode make sure skin is in Object Mode
                        bpy.ops.object.mode_set(mode='OBJECT', toggle = False)

                        bpy.ops.object.select_all(action='DESELECT')
                        bpy.context.view_layer.objects.active = rig
                        rig.select_set(True)
                        bpy.ops.object.mode_set(mode='OBJECT')
                        bpy.ops.object.rotation_clear(clear_delta=False)
                        bpy.ops.object.location_clear(clear_delta=False)
                        bpy.ops.object.scale_clear(clear_delta=False)

                        rig.data.pose_position = 'REST'
                        rig.select_set(False)

        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        mic_ctrl.select_set(True)
        bpy.context.view_layer.objects.active = mic_ctrl
        updateHookSetAttributes(obj, mic_ctrl)
        addMicroToSkin(obj, mic_ctrl, microSys, scale, mirror = False)


        if oldPosePosition != None:
            rig.data.pose_position = oldPosePosition
        if oldMode != None:
            bpy.context.view_layer.objects.active = rig
            rig.select_set(True)
            bpy.ops.object.mode_set(mode='OBJECT')

        if oldLocation != None:
            rig.location = oldLocation
        if oldRotation != None:
            rig.rotation_euler = oldRotation
        if oldScale != None:
            rig.scale = oldScale

        if oldHide != None:
            rig.hide_viewport = oldHide

        bpy.ops.object.select_all(action='DESELECT')


def Show_MicroControllers(self, context):
    objects = bpy.context.scene.objects
    wasHidden = False
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if obj != None:
            if obj.type == 'MESH':
                if obj.children != None:
                    for c in obj.children:
                        if c.type == 'EMPTY':
                            mname = obj.parent.name.replace("System", names["micro_ctrlName"])

                            if mname not in bpy.context.scene.objects.keys(): #TODO validate
                                return
                            if c.hide_viewport == True:
                                wasHidden = True
                            if wasHidden:
                                c.hide_viewport = False
                            else:
                                c.hide_viewport = not c.hide_viewport


def Create_Type(self, context):
    if not self.Create_Type == 'AUTOAIM':
        return
    if context.mode == 'OBJECT':
        pass
        # bpy.ops.object.select_all(action='DESELECT')
    if context.mode == 'POSE':
        pass
        # bpy.ops.pose.select_all(action='DESELECT')


def Muscle_Rename(obj, context):
    if obj == None:
        return

    muscleSys = obj.parent
    if muscleSys == None:
        return

    microSys = None
    oldNameObj = None
    oldNameMicroSys = None
    mic_ctrl = None
    ctrl = None

    for b in muscleSys.pose.bones:
                for c in b.constraints:
                    if c.target.type == 'EMPTY':
                        ctrl = c.target

    if ctrl == None:
        return
    scn = context.scene
    newName = scn.Muscle_Name
    useAff = scn.use_Affixes
    oldSelection = bpy.context.selected_objects
    prefix = scn.Prefix
    suffix = scn.Suffix

    group_name = prefix + defaultGroupName
    if not useAff:
        group_name = defaultGroupName
    old_group_name = ""

    group = get_xmusculature(obj)
    if group != "" and group.endswith(defaultGroupName):
             old_group_name = group

    if newName == "":
        return

    if not useAff:
        suffix = ""

    if  obj != None and getattr(obj, "Muscle_XID", None):
        if obj.select_get() == True and obj.type == 'MESH':
            oldNameObj = obj.name

            sname = obj.parent.name.replace("System", names["micro_SysName"])
            for o in scn.objects.keys():
                if sname == o:
                    if scn.objects[o].type == 'ARMATURE':
                        microSys = scn.objects[o]
                        oldNameMicroSys = microSys.name
            if obj.children != None:
                for c in obj.children:
                    if c.type == 'EMPTY':

                        mname = obj.parent.name.replace("System", names["micro_ctrlName"])
                        if mname not in scn.objects.keys():
                            return
                        mic_ctrl = c
                        break

            setAffixes(newName, prefix, suffix, useAff)

        #decorate name with suffix and number
        decorateName(obj, suffix)
        pins = []
        #decorate names for muscle pins and pin vertexgroups and mods
        for m in obj.modifiers:
            if m.type == 'HOOK':
                if m.object != None:
                    oldVisible = m.object.hide_viewport
                    m.object.hide_viewport = False
                    decorateName(obj, suffix, m.object, PinOnly = True)
                    m.name = m.object.name
                    pins.append(m.object)
                    for v in obj.vertex_groups:
                                if v.name == m.vertex_group:
                                    #update itemlist first
                                    for c in obj.custom:
                                        if c.name == v.name:
                                            c.name = m.object.name
                                    v.name = m.object.name
                    m.vertex_group = m.object.name
            
        if microSys != None:
            microSys.name = obj.parent.name.replace("System", names["micro_SysName"])
        if mic_ctrl != None:
            mic_ctrl.name = obj.parent.name.replace("System", names["micro_ctrlName"])

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    muscleSys.select_set(True)
    ctrl.select_set(True)
    for p in pins:
        p.select_set(True)

    if microSys != None:
        microSys.hide_viewport = False
        microSys.select_set(True)
    if mic_ctrl != None:
        mic_ctrl.hide_viewport = False
        mic_ctrl.select_set(True)

    objects = bpy.context.selected_objects
    #moving from old object group to new or adding to new group

    xmusculature = bpy.data.collections.get(group_name)
    xmusclea = bpy.data.collections.get(obj.name, None)

    if not xmusculature:
        xmusculature = bpy.data.collections.new(group_name)
        bpy.context.scene.collection.children.link(xmusculature)
    if not xmusclea:
        xmusclea = bpy.data.collections.new(obj.name)
        xmusculature.children.link(xmusclea)


    # bpy.ops.collection.objects_remove_all()
    for ob in objects:
        for c in ob.users_collection:
            c.objects.unlink(ob)

    for ob in objects:
        if not ob.name in xmusclea.objects:
            xmusclea.objects.link(ob)

    # remove old group if empty, not used
     
    group = bpy.data.collections.get(old_group_name, None)
    oldxmusclea = bpy.data.collections.get(oldNameObj, None)
    if oldxmusclea:
        if oldxmusclea != xmusclea:
            group.children.unlink(oldxmusclea)
            bpy.data.collections.remove(oldxmusclea)
           
    if old_group_name != group_name:
        if old_group_name in bpy.data.collections:
            if len(group.children.items()) == 0:
                bpy.context.scene.collection.children.unlink(group)
                bpy.data.collections.remove(group)


    #         for ob in objects:
    #             if ob.name in group.objects:
    #                 group.objects.unlink(ob)

    setMaterials(obj, group_name, material_diffuse)

    if microSys != None:
        microSys.hide_viewport = True

    #todo optionally rename skin vertexGroups for muscle, microController, drivers and reset the vgroupName in modifiers

    #move it before names have been changed or use oldNames

    skin = None

    for o in scn.objects:
        if  obj != None and getattr(obj, "Muscle_XID", None):
                if obj.select_get() == True and obj.type == 'MESH':
                    for m in o.modifiers:
                        if 'SOFT_BODY' in m.type:
                            xname = get_xmusculature(obj)
                            m.name = obj.name + " Softbody"
                            m.settings.collision_collection = bpy.data.collections.get(xname, None)
                            break
                    for m in o.modifiers:
                        if 'SUBSURF' in m.type:
                            m.name = obj.name + " Subsurf"
                            break
                    

        for m in o.modifiers:
            if m.type == 'SHRINKWRAP':
                if m.target != None:
                    if m.target.name == obj.name:
                        skin = o
                        break

    newv = None
    if skin != None and oldNameObj != None:
        for m in skin.modifiers:
            if m.type == 'SHRINKWRAP' and m.name == oldNameObj + " Shrinkwrap":

                if m.target != None:
                    if m.target.name == obj.name:
                        m.name = obj.name + " Shrinkwrap"
                        for v in skin.vertex_groups:
                            if v.name == m.vertex_group:
                                oldsuf, oldn = separateSuffixAndName(oldNameObj, autonum = 0)
                                newsuf, newn = separateSuffixAndName(obj.name, autonum = 0)
                                newv = v.name.replace(oldn, newn)
                                newv = newv.replace(oldsuf ,newsuf)
                                v.name = newv

                        m.vertex_group = newv

            if m.type == 'HOOK' and m.name == oldNameMicroSys:
                if m.object != None:
                    if m.object.name == microSys.name:
                        m.name = microSys.name
                        for v in skin.vertex_groups:
                            if v.name == m.vertex_group:
                                v.name = microSys.name
                        m.vertex_group = microSys.name
    # reverting back the original selection
 
    for o in oldSelection:
        o.select_set(True)


enum_items = [("MANUAL" , "Manual" , "Creates Muscles at the center of the scene"),
              ("ATCURSOR", "At Cursor", "Creates Muscles at cursor position"),
              ("AUTOAIM", "Auto Aim", "Creates and links Muscles at bone pair selection in Pose Mode")]


# Assign a boneID collection to muscle, needed for autoaim

class ObjectSettingItem(bpy.types.PropertyGroup):
    name : bpy.props.StringProperty(name="Test objSetting Prop", default="Unknown")


def addBoneOrderPropsToMuscle(obj, bones):

    for b in bones:
        my_item = obj.Bone_List.add()
        my_item.name = b.name


#x-mirror support for microController, attributes hold orientation, location and scale for microController at time when is hooked
class hookSetAttributes(bpy.types.PropertyGroup):
    location : bpy.props.FloatVectorProperty(name="Test hook location", default=(0.0, 0.0, 0.0), subtype = "XYZ")
    rotation : bpy.props.FloatVectorProperty(name="Test hook rotation", default=(0.0, 0.0, 0.0), subtype = "EULER")
    scale : bpy.props.FloatVectorProperty(name="Test hook scale", default=(0.0, 0.0, 0.0), subtype = "XYZ")
    tension : bpy.props.FloatProperty(name = "Skin Tension", default = 0.5, min=0.001, max=1, description="Sets the skin tension to the microController")



def updateHookSetAttributes(obj, microCtrl):


    # very educational and very helpful
    # obj.matrix_world.to_translation()
    # obj.matrix_world.to_scale()
    # loc, rot, scale = mw.decompose()
    #C.object.matrix_world.to_euler()
    #mw * vert.co  //gives global vert coord
    #mw * vert.co  //gives global vert coord

    #update microController with global coords
    if len(microCtrl.HookSet_Attributes) > 0:
        microCtrl.HookSet_Attributes[0].location = microCtrl.matrix_world.to_translation()
        microCtrl.HookSet_Attributes[0].rotation = microCtrl.matrix_world.to_euler()
        microCtrl.HookSet_Attributes[0].scale = microCtrl.matrix_world.to_scale()

        #get tension from muscleObj
        microCtrl.HookSet_Attributes[0].tension = obj.Skin_Tension


def addHookSetAttributesTomicroCtrl(obj, microCtrl):

    my_item = microCtrl.HookSet_Attributes.add()
    my_item.name = microCtrl.name
    my_item.location = microCtrl.location

    my_item.rotation = microCtrl.rotation_euler

    my_item.scale = microCtrl.scale

    #get tension from muscleObj
    my_item.tension = obj.Skin_Tension


def executeHookSetAttributes(obj, microCtrl):

    microCtrl.location = microCtrl.HookSet_Attributes[0].location
    microCtrl.rotation_euler = microCtrl.HookSet_Attributes[0].rotation
    microCtrl.scale = microCtrl.HookSet_Attributes[0].scale
    #get tension from muscleObj
    obj.Skin_Tension = microCtrl.HookSet_Attributes[0].tension


    # zeroes all hook attributes for mctrl
def removeHookSetAttributesFrommicroCtrl(microCtrl):
    microCtrl.HookSet_Attributes.clear()

    #pinning and bendy muscles features will use this hookmirroring function instead of memorizing hookattributes
def xhook(modifier, oldmeshScale):

    #for modifier in obj.modifiers:
        #if modifier.type == 'HOOK':

            #returns the copy of matrix_inverse
            mat = modifier.matrix_inverse

            mat.invert()


            #get the loc rot scale from the matrix
            loc, rotQuat, scale = mat.decompose()

            # create a location matrix
            mat_loc = Matrix.Translation((-loc.x, loc.y, loc.z))

            # create an identitiy matrix
            mat_sca = Matrix.Scale(scale.x, 4, (1.0, 0.0, 0.0))
            mat_sca = mat_sca @ Matrix.Scale(scale.y, 4, (0.0, 1.0, 0.0))
            mat_sca = mat_sca @ Matrix.Scale(scale.z, 4, (0.0, 0.0, 1.0))

            # create a rotation matrix
            rot = rotQuat.to_euler()
            rot.x = rot.x
            rot.y = -rot.y
            rot.z = -rot.z
            mat_rot = Matrix.Rotation(rot.x, 3, 'X')
            mat_rot.rotate(Matrix.Rotation(rot.y, 3, 'Y'))
            mat_rot.rotate(Matrix.Rotation(rot.z, 3, 'Z'))


            # combine transformations
            mat = mat_loc @ mat_rot.to_4x4() @ mat_sca


            #put it all back to hook
            mat.invert()
            modifier.matrix_inverse = mat
            #modifier.matrix_inverse.normalize()


def hookscalex(obj, oldmeshScale):

    for modifier in obj.modifiers:
        if modifier.type == 'HOOK':


            #returns the copy of matrix_inverse
            mat = modifier.matrix_inverse

            mat.invert()


            mat_scale = Matrix.Scale(oldmeshScale.x,4,(1,0,0))
            mat_scale = mat_scale @ Matrix.Scale(oldmeshScale.y,4,(0,1,0))
            mat_scale = mat_scale @ Matrix.Scale(oldmeshScale.z,4,(0,0,1))

            mat = mat_scale @ mat

            mat.invert()
            modifier.matrix_inverse = mat


def atCursor(location):
    atCursor = bpy.context.scene.Create_Type
    useAff = bpy.context.scene.use_Affixes
    suffix = bpy.context.scene.Suffix

    pos = Vector((0,0,0))

    if atCursor == "ATCURSOR":
        pos = bpy.context.scene.cursor.location + location
        if not useAff or suffix =="":
            pos = bpy.context.scene.cursor.location

    return pos

def setPivotPoint(type):
    global oldPivot

    if type not in ('BOUNDING_BOX_CENTER', 'CURSOR', 'INDIVIDUAL_ORIGINS',
    'MEDIAN_POINT', 'ACTIVE_ELEMENT'):
        return False
    pivot = bpy.context.scene.tool_settings.transform_pivot_point
    oldPivot = pivot
    bpy.context.scene.tool_settings.transform_pivot_point = type



def get_override(area_type, region_type):
    for area in bpy.context.screen.areas:
        if area.type == area_type:
            for region in area.regions:
                if region.type == region_type:
                    override = bpy.context.copy()
                    override['area'] = area
                    override['region'] = region
                    return override
    #error message if the area or region wasn't found
    raise RuntimeError("Wasn't able to find", region_type," in area ", area_type,
                        "\n Make sure it's open while executing script.")

def tmp_override(override, override_type):

    version, ver_major, ver_minor = bpy.app.version
    if version >= 4:
        if override_type == 'translate':
            with bpy.context.temp_override(area=override['area'], region=override['region']):
                bpy.ops.transform.translate(value=(0.0, 0.0, 0.0))
        elif override_type == 'snap_cursor_to_active':
            with bpy.context.temp_override(area=override['area'], region=override['region']):
                bpy.ops.view3d.snap_cursor_to_active()
        elif override_type == 'mirror':
            with bpy.context.temp_override(area=override['area'], region=override['region']):
                bpy.ops.transform.mirror(constraint_axis=(True, False, False))
        elif override_type == 'snap_cursor_to_selected':
            with bpy.context.temp_override(area=override['area'], region=override['region']):
                bpy.ops.view3d.snap_cursor_to_selected()
        elif override_type == 'snap_cursor_to_center':
            with bpy.context.temp_override(area=override['area'], region=override['region']):
                bpy.ops.view3d.snap_cursor_to_center()
        elif override_type == 'ptcache_add':
            with bpy.context.temp_override(blend_data=override['blend_data'], scene=override['scene'], active_object=override['active_object'], point_cache=override['point_cache']):
                bpy.ops.ptcache.add()
        elif override_type == 'ptcache_remove':
            with bpy.context.temp_override(blend_data=override['blend_data'], scene=override['scene'], active_object=override['active_object'], point_cache=override['point_cache']):
                bpy.ops.ptcache.remove()
        
    else:
        if override_type == 'translate':
            bpy.ops.transform.translate(override, value=(0.0, 0.0, 0.0))
        elif override_type == 'snap_cursor_to_active':
            bpy.ops.view3d.snap_cursor_to_active(override)
        elif override_type == 'mirror':
            bpy.ops.transform.mirror(override, constraint_axis=(True, False, False))
        elif override_type == 'snap_cursor_to_selected':
            bpy.ops.view3d.snap_cursor_to_selected(override)
        elif override_type == 'snap_cursor_to_center':
            bpy.ops.view3d.snap_cursor_to_center(override)
        elif override_type == 'ptcache_add':
            bpy.ops.ptcache.add(override)
        elif override_type == 'ptcache_remove':
            bpy.ops.ptcache.remove(override)
        

def readMesh(meshData, meshName, scale):

    buffer = io.StringIO(meshData)

    def line_to_face(line):
        # Each triplet is an xyz float
        line_split = line.split()
        try:
            line_split_float = map(float, line_split)
        except:
            return None

        if len(line_split) in {9, 12}:
            return zip(*[iter(line_split_float)] * 3)  # group in 3's
        else:
            return None

    faces = []
    for line in buffer.readlines():
        face = line_to_face(line)
        if face:
            faces.append(face)

    # Generate verts and faces lists, without duplicates
    verts = []
    coords = {}
    index_tot = 0
    faces_indices = []

    for f in faces:
        fi = []
        for i, v in enumerate(f):
            index = coords.get(v)

            if index is None:
                index = coords[v] = index_tot
                index_tot += 1
                verts.append(v)

            fi.append(index)

        faces_indices.append(fi)

    mesh = bpy.data.meshes.new(meshName)
    mesh.from_pydata(verts, [], faces_indices)
    for vert in mesh.vertices:
            vert.co = vert.co * scale
    return mesh


def addMeshObj(mesh, objName, origin, group_name):
    scn = bpy.context.scene

    for o in scn.objects:
        o.select_set(False)

    mesh.update()
    mesh.validate()

    nobj = bpy.data.objects.new(objName, mesh)

    xmusculature = bpy.data.collections.get(group_name, None)
    if not xmusculature:
        xmusculature = bpy.data.collections.new(group_name)
        bpy.context.scene.collection.children.link(xmusculature)
    # bpy.ops.collection.objects_remove_all()

    for c in nobj.users_collection:
        c.objects.unlink(nobj)

    if not nobj.name in xmusculature.objects:
        xmusculature.objects.link(nobj)

    nobj.select_set(True)
    nobj.location = origin

    if bpy.context.view_layer.objects.active is None:
        bpy.context.view_layer.objects.active = nobj

    return nobj


def addDriver(source, target, prop, dataPath, index = -1, negative = False, type = 'AVERAGE', func = ''):
    ''' Add driver to source prop (at index), driven by target dataPath '''

    if index != -1:

        fcurve = source.driver_add( prop, index )
        d = fcurve.driver
    else:
        fcurve = source.driver_add( prop )
        d = fcurve.driver

    v = d.variables.new()
    v.name                 = prop
    v.targets[0].id        = target
    v.targets[0].data_path = dataPath

    d.type = type
    d.expression = func + "(" + v.name + ")" if func else v.name
    d.expression = d.expression if not negative else "-1 * " + d.expression

    return fcurve

def addShapeKey(sourceMesh, targetObj, shapeKeyName, Type = ''):

    if Type == 'EMPTY':
        sk = targetObj.shape_key_add(from_mix = False)
        sk.name = shapeKeyName
    else:
        shapeKey = targetObj.shape_key_add(from_mix = False)
        shapeKey.name = shapeKeyName

        for vert in sourceMesh.vertices:
            shapeKey.data[vert.index].co = vert.co

def setMaterials(obj, matName, color):
    ob = obj

    # Get material
    mat = bpy.data.materials.get(matName)
    if mat is None:
        # create material
        mat = bpy.data.materials.new(name = matName)
        mat.diffuse_color = color

    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)


def addModifiers(obj, vertexGroupName = "VertexGroup", softBodyName = "Softbody", SubsurfName = "Subsurf"):
    subSlevel = 0
    bpy.ops.object.shade_smooth()
    mod = bpy.context.object.modifiers.new(type='SOFT_BODY', name=softBodyName)

    mod.settings.vertex_group_mass = vertexGroupName
    xname = get_xmusculature(obj)
    mod.settings.collision_collection = bpy.data.collections.get(xname, None)
    mod.show_expanded = False
    s = bpy.context.scene.frame_start
    e = bpy.context.scene.frame_end

    ptc = mod.point_cache
    ptc.frame_start = s - 3
    ptc.frame_end = e + 3
    ptc.frame_step = 1
    ptc.name = softBodyName + "_cache"

    mod = bpy.context.object.modifiers.new(type='SUBSURF', name=SubsurfName)
    mod.levels = subSlevel
    mod.show_on_cage = True
    mod.show_expanded = False


def addVertexGroup(idx, obj, vgroupName):
    obj.vertex_groups.new(name=vgroupName)

    #for all vertex groups
    for grp in obj.vertex_groups:
        grp.add(idx, 1.0, 'REPLACE')

def addMuscleCustomProperties(obj, muscleSys):

    scn = bpy.context.scene

    scn.Micro_Distance = 1.0
    scn.Micro_Influence =  0.5
    scn.Micro_Radius =  0.5

    muscleSys.Base_Length = 1.0
    muscleSys.Volume = 1.0
    muscleSys.Muscle_Type = 0

    #duplicated for workaround to provide dependency cycle and endless recursion
    obj.Base_Length_INT = 0
    obj.Volume_INT = 1.0
    obj.Muscle_Type_INT = 0


    obj.Muscle_Shape = 0
    obj.View3D_Resolution = 1
    obj.Render_Resolution = 1
    obj.Jiggle_Springiness = 0.75
    obj.Jiggle_Damping = 37.5
    obj.Jiggle_Mass = 0.75
    obj.Jiggle_Stiffness = 0.75
    obj.Muscle_View3D = 1
    obj.Muscle_Render = 1
    obj.Pinning_View3D = 1
    obj.Pinning_Render = 1
    obj.Dynamics_View3D = 0
    obj.Dynamics_Render = 0
    obj.Muscle_Size = 0.6
    obj.Muscle_Offset = 0
    obj.Micro_Controller = 0
    obj.Micro_Controller_View3D = 0
    obj.Micro_Controller_Render = 0
    obj.Skin_Tension = 0.5
    obj.Muscle_XID = 1
    obj.Bone_List.clear()

    return{'FINISHED'}


def createController(name, origin, draw_size, draw_type = 'CUBE', lock = True):

    if origin != None:
        bpy.ops.object.add(type ='EMPTY', location = origin)
    else:
        bpy.ops.object.add(type ='EMPTY')
    empty = bpy.context.object
    empty.name = name
    empty.show_in_front = True
    empty.empty_display_type = draw_type
    empty.empty_display_size = 0.35 * draw_size

    if lock:
        bpy.context.object.lock_rotation = (True, True, True)

    return empty

def addConstraints(obj, type = '', name = '', target = '', subtarget = '', head_tail = 0, rest_length = 0, target_space = 'WORLD', owner_space = 'WORLD'):

    isSpace = False
    if type == 'COPY_LOCATION':
        isSpace = True
    elif type == 'COPY_ROTATION':
        isSpace = True
    elif type == 'COPY_SCALE':
        isSpace = True
    elif type == 'STRETCH_TO':
        isSpace = False
    else: return

    const = obj.constraints.new(type=type)
    const.name = name
    const.target = target
    const.subtarget = subtarget

    version, ver_major, ver_minor = bpy.app.version
    if version >= 3:
        if type == 'STRETCH_TO':
            const.keep_axis = 'PLANE_X'
    if isSpace:
        const.target_space = target_space
        const.owner_space = owner_space
    if head_tail != 0:
        const.head_tail = head_tail
    if rest_length != 0:
        const.rest_length = rest_length

    const.show_expanded = False

    return const

def createArmature(name, boneTable, origin):
    # Create armature and object
    bpy.ops.object.add(
        type='ARMATURE',
        enter_editmode = True,
        location=origin)
    ob = bpy.context.object
    ob.show_in_front = True
    ob.name = name
    amt = ob.data
    amt.name = name + 'Amt'
    amt.show_axes = False

    # Create bones
    bpy.ops.object.mode_set(mode='EDIT')
    for (bname, pname, vector) in boneTable:
        bone = amt.edit_bones.new(bname)
        if pname:
            parent = amt.edit_bones[pname]
            bone.parent = parent
            bone.head = parent.tail
            bone.use_connect = False
            (trans, rot, scale) = parent.matrix.decompose()
        else:
            bone.head = (0,0,0)
            rot = Matrix.Translation((0,0,0))   # identity matrix
        bone.tail = rot @ Vector(vector) + bone.head
        bone.use_deform = False


    bpy.ops.object.mode_set(mode='OBJECT')

    return ob

#todo instead of appending to temp list, append it directly to object's property list
def iterate_faces(me, idx):
    bm = bmesh.new()
    bm.from_mesh(me)
    indices = list()
    for f in bm.faces:
        if f.index == idx:
            for v in reversed(f.verts):
                if len(indices) == 3:
                    return indices
                else:
                    indices.append(v.index)
                    

def micCtrl_idx_parent():
    context = bpy.context
    scene = context.scene

    #mesh
    srcObj = context.active_object
    #muscle
    targetObj = None

    for o in context.selected_objects:
        if o.type == 'MESH' and getattr(o, "Muscle_XID", None):
            targetObj = o
            break
    me = targetObj.data
    bm = bmesh.new()

    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()

    #TODO FIXME get ctrl_micro parent verts
    vert = bm.verts[9]
  
    v1 = targetObj.matrix_world @ vert.co # global face median
    smwi = srcObj.matrix_world.inverted()
    local_pos = smwi @ v1  # face cent in srcObj local space

    (hit, loc, norm, face_index) = srcObj.closest_point_on_mesh(local_pos)
    if hit:
        v2 = srcObj.matrix_world @ loc
        # bpy.ops.object.empty_add(location=v2)
        indices = iterate_faces(srcObj.data, face_index)

    bm.clear()
    return indices


def update_dependencies(ob):
    def updateExp(d):
        d.driver.expression += " "
        d.driver.expression = d.driver.expression[:-1]
    try:
        drivers = ob.animation_data.drivers
        for d in drivers:
            updateExp(d)
    except AttributeError:
        return
        
def convertAim(Type):

    scn = bpy.context.scene
    object = bpy.context.object
    prefix = scn.Prefix #TODO validate
    suffix = scn.Suffix
    useAff = scn.use_Affixes
    newName = scn.Muscle_Name
    oldLock = scn.tool_settings.lock_object_mode
    oldLock = scn.tool_settings.lock_object_mode = False

    scale = scn.Muscle_Scale

    override = get_override( 'VIEW_3D', 'WINDOW' )
    #get and store the original cursor's origin for autoaim

    locx = []

    #mesh object
    sourceObj = None
    #muscle object
    targetObj = None
    target_origin = None
    trgt_muscleSys = None
    trgt_ctrl = None

    if bpy.context.active_object.type == 'MESH' and not getattr(bpy.context.active_object, "Muscle_XID", None): #TODO validate
        sourceObj = bpy.context.active_object

    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):
            targetObj = obj
            if obj.parent != None:
                trgt_muscleSys = obj.parent
            if trgt_muscleSys != None:

                for b in trgt_muscleSys.pose.bones:
                    for c in b.constraints:
                        if c.target.type == 'EMPTY':
                            trgt_ctrl = c.target
                        break

    rig = trgt_ctrl.parent
    if rig is None:
        return
    
    indices = micCtrl_idx_parent()
    activeb = rig.pose.bones[trgt_ctrl.parent_bone].bone  
    selectb = rig.pose.bones[trgt_muscleSys.parent_bone].bone 
  
    oldCursor = Vector(scn.cursor.location)
    if object != None:
        if object.type == 'MESH':

            if bpy.context.mode != 'OBJECT':
                return

            trgt_muscleSys.select_set(True)
            bpy.context.view_layer.objects.active = trgt_muscleSys
            tmp_override(override, 'snap_cursor_to_active')
            trgt_muscleSys.select_set(False)
            locx.append(Vector(scn.cursor.location))

            trgt_ctrl.select_set(True)
            bpy.context.view_layer.objects.active = trgt_ctrl
            tmp_override(override, 'snap_cursor_to_active')
            trgt_ctrl.select_set(False)
            locx.append(Vector(scn.cursor.location))
            if len(locx) == 2:
                lengthx = (locx[1]-locx[0]).length
                scale = lengthx / 5.16 # muscle ratio
            else:
                scale = scn.Muscle_Scale
   
    scn.cursor.location = oldCursor
    version, ver_major, ver_minor = bpy.app.version
    if version >= 3:
        BONE_CUSTOM_SHAPE_SCALE = Vector((5 * scale, 5 * scale, 5 * scale))
    else:
        BONE_CUSTOM_SHAPE_SCALE = 5 * scale

    group_name = prefix + defaultGroupName

    if newName == "":
        return

    setAffixes(newName, prefix, suffix, useAff)
    
    bpy.ops.object.select_all(action='DESELECT')
    sourceObj.select_set(True)
    oldSrcOrigin = Vector(sourceObj.location)
    bpy.context.view_layer.objects.active = sourceObj
    scn.cursor.location = (0,0,0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

    me = bpy.context.object.data.copy()
    vertexGroupName = names["vertexGroupName"]
    idx = []

    boneCustomMeshData = XMUSCLE_DATA.boneCustomMeshData

    muscleLength =  5.16 * scale # 5.16 defaults
    muscleHeight =  3 * scale # defaults
    location = Vector((muscleLength / 2,0,0))

    if useAff:
        if isRight(suffix):
            location = -location

    else:
        group_name = defaultGroupName

    origin = atCursor(location)

    NAME_COL = 0
    boneTable = [(names["mbName"], None, (0.1,0,0)),
                 (names["extbName"], None, (0.1,0,0)),
                 (names["flxbName"], None, (0.1,0,0)),
                 (names["extctrlbName"], None, (0,0,0.1)),
        (names["cntrbName"], names["flxbName"], (0.1,0,0)),]


    bpy.context.view_layer.objects.active = None
    muscleSys = createArmature(names["musculatureName"], boneTable, -location)

    muscle_pBone = muscleSys.pose.bones[boneTable[NAME_COL][NAME_COL]]
    extensor_pBone = muscleSys.pose.bones[boneTable[NAME_COL+1][NAME_COL]]

    flexor_pBone = muscleSys.pose.bones[boneTable[NAME_COL+2][NAME_COL]]
    extensor_ctrl_pBone = muscleSys.pose.bones[boneTable[NAME_COL+3][NAME_COL]]
    counter_pBone = muscleSys.pose.bones[boneTable[NAME_COL+4][NAME_COL]]

    bpy.ops.object.mode_set(mode='EDIT')
    muscleSys.data.edit_bones[counter_pBone.name].head = (0,0,0)
    muscleSys.data.edit_bones[counter_pBone.name].tail = (-muscleLength, 0, 0)
    muscleSys.data.edit_bones[extensor_pBone.name].tail = (muscleLength * 2,0,0)
    muscleSys.data.edit_bones[extensor_ctrl_pBone.name].head = (muscleLength * 2,0,0)
    muscleSys.data.edit_bones[extensor_ctrl_pBone.name].tail = (muscleLength * 2,0,1)
    bpy.ops.object.mode_set(mode='OBJECT')

    ctrl = createController(names["mctrlName"], location, scale)

        #workaround for constraint reorder to work
        #override={'constraint':ctrl2.constraints["Copy Location"]}
        #bpy.ops.constraint.move_up(override, constraint='Copy Location', owner='OBJECT')

    customShapeMesh = readMesh(boneCustomMeshData, names["boneCustomShapeName"], 1)
    boneShapeObj = addMeshObj(customShapeMesh, names["boneCustomShapeName"], (0,0,0), group_name)

    #todo cleanup the code
    oc = boneShapeObj.copy()
    oc.data = boneShapeObj.data.copy()
 
    flexor_pBone.custom_shape = oc
    version, ver_major, ver_minor = bpy.app.version
    if version >= 3:
        flexor_pBone.custom_shape_scale_xyz = BONE_CUSTOM_SHAPE_SCALE
    else:
        flexor_pBone.custom_shape_scale = BONE_CUSTOM_SHAPE_SCALE
    flexor_pBone.custom_shape_transform = flexor_pBone
    flexor_pBone.bone.show_wire = True

    #remove temporary custom bone Shape mesh
    boneShapeObj.select_set(True)
    bpy.ops.object.delete()

    ext_ctrl_cpLoConst = addConstraints(extensor_ctrl_pBone, type = 'COPY_LOCATION', name = names["ext_ctrl_copyLocConstName"], target = muscleSys, subtarget = extensor_pBone.name, head_tail = 1)
    msl_cpRotConst = addConstraints(muscle_pBone, type = 'COPY_ROTATION', name = names["msl_copyRotConstName"], target = muscleSys, subtarget = flexor_pBone.name)
    msl_cpLoConst = addConstraints(muscle_pBone, type = 'COPY_LOCATION', name = names["msl_copyLocConstName"], target = muscleSys, subtarget = flexor_pBone.name)
    msl_strToConst = addConstraints(muscle_pBone, type = 'STRETCH_TO', name = names["msl_stretchToCtrlConstName"], target = ctrl, rest_length = muscleLength)
    msl_strToConst2 = addConstraints(muscle_pBone, type = 'STRETCH_TO', name = names["msl_stretchToExtConstName"], target = muscleSys, subtarget = extensor_ctrl_pBone.name, rest_length = muscleLength)
    ext_cpLoConst = addConstraints(extensor_pBone, type = 'COPY_LOCATION', name = names["ext_copyLocConstName"], target = muscleSys, subtarget = counter_pBone.name, head_tail = 1)
    ext_cpRotConst = addConstraints(extensor_pBone, type = 'COPY_ROTATION', name = names["ext_copyRotConstName"], target = muscleSys, subtarget = flexor_pBone.name)
    flx_strToConst = addConstraints(flexor_pBone, type = 'STRETCH_TO', name = names["flx_stretchToConstName"], target = ctrl, rest_length = muscleLength)

    counter_pBone.hide = True
    muscle_pBone.hide = True
    extensor_pBone.hide = True
    extensor_ctrl_pBone.hide = True

    bpy.context.view_layer.objects.active = None

    objName = names["muscleName"]
    mesh = me
    mesh.name = objName
    
    obj = addMeshObj(mesh, objName, (0,0,0), group_name)
    addVertexGroup(idx, obj, vertexGroupName)

    #parenting 'to bone' mesh with armature
    obj.select_set(True)
    b = muscle_pBone.bone
    
    muscle_pBone.select = True
    #b.select = True #TODO verify if still needed
    muscleSys.data.bones.active = b
    bpy.context.view_layer.objects.active = muscleSys

    bpy.ops.object.parent_set(type='BONE')
    
    muscleSys.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    addModifiers(obj, vertexGroupName, names["softBodyName"], names["muscleSubsurfName"])
    #adding shapekeys
    objName = names["muscleShapeKeyName"]
    addShapeKey(mesh, obj, "Basis", "EMPTY")

    addShapeKey(mesh, obj, names["muscleThinName"], "EMPTY")
    addShapeKey(mesh, obj, names["muscleThickName"], "EMPTY")

    setMaterials(obj, group_name, material_diffuse)

    #add drivers
    #flexor
    fcurve = addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToCtrlConstName"]], muscleSys, 'rest_length' , "Base_Length")
    # set coefficient polynomial x to 5.16 the aspect ratio of muscle length flexor and extensor
    fcurve.modifiers.new(type='GENERATOR')
    fcurve.modifiers[0].coefficients[1] = muscleLength

    addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToCtrlConstName"]], muscleSys, 'bulge' , "Volume")

    #extensor
    fcurve = addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToExtConstName"]], muscleSys, 'rest_length' , "Base_Length")
    fcurve.modifiers.new(type='GENERATOR')
    fcurve.modifiers[0].coefficients[1] = muscleLength

    addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToExtConstName"]], muscleSys, 'bulge' , "Volume")
    addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToExtConstName"]], muscleSys, 'influence', "Muscle_Type")

    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'goal_min' , "Jiggle_Springiness")
    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'goal_friction' , "Jiggle_Damping")
    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'mass' , "Jiggle_Mass")
    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'goal_spring' , "Jiggle_Stiffness")
    addDriver(obj.modifiers[names["muscleSubsurfName"]], obj, 'render_levels' , "Render_Resolution")
    addDriver(obj.modifiers[names["muscleSubsurfName"]], obj, 'levels' , "View3D_Resolution")

    addDriver(obj.data.shape_keys.key_blocks[names["muscleThickName"]], obj, "value", "Muscle_Shape")
    addDriver(obj.data.shape_keys.key_blocks[names["muscleThinName"]], obj, "value", "Muscle_Shape")

    #invert driver's muscle_thin function to range of -1 to 0
    shapeKeys = obj.data.shape_keys
    drivers = shapeKeys.animation_data.drivers[1]
    drivers.modifiers.new(type='GENERATOR')
    drivers.modifiers[0].coefficients[1] = -1

    addDriver(obj.modifiers[names["softBodyName"]], obj, 'show_viewport' , "Dynamics_View3D")
    addDriver(obj.modifiers[names["softBodyName"]], obj, 'show_render' , "Dynamics_Render")

    #select controllers leaving the muscle active
    obj.select_set(True)
    muscleSys.select_set(True)
    ctrl.select_set(True)
    # create xmusculature group and link with the muscles
    objects = bpy.context.selected_objects
    xmusculature = bpy.data.collections.get(group_name, None)
    xmusclea = bpy.data.collections.get(newName, None)

    if not xmusculature:
        xmusculature = bpy.data.collections.new(group_name)
        bpy.context.scene.collection.children.link(xmusculature)
    if not xmusclea:
        xmusclea = bpy.data.collections.new(newName)
        xmusculature.children.link(xmusclea)

    # if group_name in bpy.data.groups:
    #     group = bpy.data.groups[group_name]
    # else:
    #     group = bpy.data.groups.new(group_name)
    # bpy.ops.collection.objects_remove_all()
    for ob in objects:
        for c in ob.users_collection:
            c.objects.unlink(ob)

    for ob in objects:
        if not ob.name in xmusclea.objects:
            xmusclea.objects.link(ob)

    if not useAff:
        suffix = ""

    scn.tmp_Bone_List.clear()
    for b in targetObj.Bone_List:
        my_item = scn.tmp_Bone_List.add()
        my_item.name = b.name
    #getting original muscle origin before it's being removed
    targetObj.select_set(True)
    bpy.context.view_layer.objects.active = targetObj
    tmp_override(override, 'snap_cursor_to_active')
    trgt_muscleSys.select_set(False)
    target_origin = Vector(scn.cursor.location)
    scn.cursor.location = (0,0,0)

    #remove old, target muscle before old and new naming collides
    collection = bpy.data.collections.get(targetObj.users_collection[0].name)
    for ob in collection.objects:
        ob.name = "x"
        # bpy.data.objects.remove(ob, do_unlink=True)
        #strange workaround but it works id user decrement error:
    bpy.data.collections.remove(collection)

    #decorate name with suffix and autonumber
    decorateName(obj, suffix)
    xmusclea.name = obj.name
    obj.modifiers[names["softBodyName"]].name = obj.name + " Softbody"
    obj.modifiers[names["muscleSubsurfName"]].name = obj.name + " Subsurf"
    #FIXME move below decorateName() but remove old, target muscle before decorateName() ! 
    #some of properties update functions depend on naming so order of execution matters and function is triggered after decorations
    addMuscleCustomProperties(obj, muscleSys)
    addBoneOrderPropsToMuscle(obj, scn.tmp_Bone_List)
    # clear parenting 'to bone' mesh with armature
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.parent_clear(type='CLEAR')

    #move mesh layer to active layer only
    # for i in range(len(obj.layers)):
    #     obj.layers[i] = (i == scn.active_layer)

    bpy.ops.object.select_all(action='DESELECT')

    muscleSys.location = locx[0]
    ctrl.location = locx[1]

    ctrl.select_set(True)
    b = activeb
    
    # b.select = True
    bpy.data.objects[rig.name].pose.bones[b.name].select = True
    rig.data.bones.active = b
    bpy.context.view_layer.objects.active = rig

    bpy.ops.object.parent_set(type='BONE')

    ctrl.select_set(False)
    #b.select = False #TODO verify if still needed
    bpy.data.objects[rig.name].pose.bones[b.name].select = False


    muscleSys.select_set(True)
    b = selectb
    # b.select = True
    bpy.data.objects[rig.name].pose.bones[b.name].select = True
    rig.data.bones.active = b
    bpy.context.view_layer.objects.active = rig

    bpy.ops.object.parent_set(type='BONE')

    #reparenting 'to bone' mesh with armature
    ##
    muscleSys.select_set(False)
    bpy.data.objects[rig.name].pose.bones[b.name].select = False
    #b.select = False #TODO verify if still needed
    obj.select_set(True)
    scn.cursor.location = target_origin
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR') 
    scn.cursor.location = oldCursor
    b = muscle_pBone.bone
    muscle_pBone.select = True
    #b.select = True
    muscleSys.data.bones.active = b
    bpy.context.view_layer.objects.active = muscleSys
    bpy.ops.object.parent_set(type='BONE')
    ##

    bpy.ops.object.select_all(action='DESELECT')
    sourceObj.select_set(True)
    bpy.context.view_layer.objects.active = sourceObj
    scn.cursor.location = oldSrcOrigin
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    sourceObj.select_set(False)
    print("rig", rig.name)
    print("bone", b.name)
    print("muscleSys", muscleSys.name)
    bpy.data.objects[muscleSys.name].pose.bones[b.name].select = False #FIXME autoaim convert error bone name not found
    #b.select = False
    obj.select_set(True)
    muscleSys.select_set(True)
    ctrl.select_set(True)
    bpy.context.view_layer.objects.active = obj
   
    for ob in bpy.data.objects:
        update_dependencies(ob)
        update_dependencies(shapeKeys)
    idxConvertedParent(indices, obj)
    scn.tool_settings.lock_object_mode = oldLock
    return {'FINISHED'}



def createMuscle(Type):

    scn = bpy.context.scene
    object = bpy.context.object
    prefix = scn.Prefix
    suffix = scn.Suffix
    useAff = scn.use_Affixes
    newName = scn.Muscle_Name
    oldLock = scn.tool_settings.lock_object_mode
    oldLock = scn.tool_settings.lock_object_mode = False

    scale = scn.Muscle_Scale

    override = get_override( 'VIEW_3D', 'WINDOW' )
    #get and store the original cursor's origin for autoaim

    locx = []
    selectedBones = []
    oldCursor = Vector(scn.cursor.location)
    if scn.Create_Type == 'AUTOAIM':
        if object != None:
            
            if scn.Create_Type == 'AUTOAIM' and Type == 'CONVERT' and bpy.context.active_object.type == 'MESH' and len(bpy.context.selected_objects) >= 2:
                convertAim(Type)
                return{'FINISHED'}
            
            if object.type == 'ARMATURE':

                if bpy.context.mode != 'POSE':
                    return

                rig = object
                # selectedBones = bpy.context.selected_order_pose_bones
                selectedBones = get_ordered_selection_objects()

                if selectedBones == None or len(selectedBones) == 0:
                    return
                first = selectedBones[0] # [-1] for the last element

                activeb = rig.data.bones.active
                if rig.data.bones[first.name] == None:
                    return
                if rig.data.bones[first.name] == activeb:
                    return


                selectb = None              
                tmp_override(override, 'snap_cursor_to_active')
                locx.append(Vector(scn.cursor.location))
                #activeb.select = False #TODO verify if there's a better way
                bpy.data.objects[rig.name].pose.bones[activeb.name].select = False
                rig.data.bones.active = rig.data.bones[first.name]
                tmp_override(override, 'snap_cursor_to_active')
                locx.append(Vector(scn.cursor.location))
                selectb = rig.data.bones[first.name]
                if len(locx) == 2:
                    lengthx = (locx[1]-locx[0]).length
                    scale = lengthx / 5.16 # muscle ratio
                else:
                    scale = scn.Muscle_Scale
    else:
        scale = scn.Muscle_Scale

    scn.cursor.location = oldCursor

    version, ver_major, ver_minor = bpy.app.version
    if version >= 3:
        BONE_CUSTOM_SHAPE_SCALE = Vector((5 * scale, 5 * scale, 5 * scale))
    else:
        BONE_CUSTOM_SHAPE_SCALE = 5 * scale

    group_name = prefix + defaultGroupName

    if newName == "":
        return

    setAffixes(newName, prefix, suffix, useAff)

    if Type == "BASIC":
        muscleBasisMeshData = XMUSCLE_DATA.muscleBasisMeshData
        muscleThinMeshData = XMUSCLE_DATA.muscleBasisMeshData
        muscleThickMeshData = XMUSCLE_DATA.muscleBasisMeshData
        #mesh type indicator by vertexgroup name
        vertexGroupName = names["vertexGroupName"] + "_b"
        idx = XMUSCLE_DATA.idx

    elif Type == "STRIP":
        muscleBasisMeshData = XMUSCLE_DATA.muscleStripMeshData
        muscleThinMeshData = XMUSCLE_DATA.muscleStripMeshData
        muscleThickMeshData = XMUSCLE_DATA.muscleStripMeshData
        #mesh type indicator by vertexgroup name
        vertexGroupName = names["vertexGroupName"] + "_st"
        idx = XMUSCLE_DATA.idxStrip
    elif Type == "CONVERT":
        me = bpy.context.object.data.copy()
        vertexGroupName = names["vertexGroupName"]
        scale = 1
        idx = []
    else:
        muscleBasisMeshData = XMUSCLE_DATA.muscleStyleBasisMeshData
        muscleThinMeshData = XMUSCLE_DATA.muscleStyleThinMeshData
        muscleThickMeshData = XMUSCLE_DATA.muscleStyleThickMeshData
        vertexGroupName = names["vertexGroupName"] + "_s"
        idx = XMUSCLE_DATA.idxStyle

    boneCustomMeshData = XMUSCLE_DATA.boneCustomMeshData

    muscleLength =  5.16 * scale # 5.16 defaults
    muscleHeight =  3 * scale # defaults
    location = Vector((muscleLength / 2,0,0))

    if useAff:
        if isRight(suffix):
            location = -location

    else:
        group_name = defaultGroupName

    origin = atCursor(location)

    NAME_COL = 0
    boneTable = [(names["mbName"], None, (0.1,0,0)),
                 (names["extbName"], None, (0.1,0,0)),
                 (names["flxbName"], None, (0.1,0,0)),
                 (names["extctrlbName"], None, (0,0,0.1)),
        (names["cntrbName"], names["flxbName"], (0.1,0,0)),]


    bpy.context.view_layer.objects.active = None
    muscleSys = createArmature(names["musculatureName"], boneTable, -location)

    muscle_pBone = muscleSys.pose.bones[boneTable[NAME_COL][NAME_COL]]
    extensor_pBone = muscleSys.pose.bones[boneTable[NAME_COL+1][NAME_COL]]

    flexor_pBone = muscleSys.pose.bones[boneTable[NAME_COL+2][NAME_COL]]
    extensor_ctrl_pBone = muscleSys.pose.bones[boneTable[NAME_COL+3][NAME_COL]]
    counter_pBone = muscleSys.pose.bones[boneTable[NAME_COL+4][NAME_COL]]

    bpy.ops.object.mode_set(mode='EDIT')
    muscleSys.data.edit_bones[counter_pBone.name].head = (0,0,0)
    muscleSys.data.edit_bones[counter_pBone.name].tail = (-muscleLength, 0, 0)
    muscleSys.data.edit_bones[extensor_pBone.name].tail = (muscleLength * 2,0,0)
    muscleSys.data.edit_bones[extensor_ctrl_pBone.name].head = (muscleLength * 2,0,0)
    muscleSys.data.edit_bones[extensor_ctrl_pBone.name].tail = (muscleLength * 2,0,1)
    bpy.ops.object.mode_set(mode='OBJECT')

    ctrl = createController(names["mctrlName"], location, scale)

        #workaround for constraint reorder to work
        #override={'constraint':ctrl2.constraints["Copy Location"]}
        #bpy.ops.constraint.move_up(override, constraint='Copy Location', owner='OBJECT')

    customShapeMesh = readMesh(boneCustomMeshData, names["boneCustomShapeName"], 1)
    boneShapeObj = addMeshObj(customShapeMesh, names["boneCustomShapeName"], (0,0,0), group_name)
    oc = boneShapeObj.copy()
    oc.data = boneShapeObj.data.copy()
    #todo cleanup the code
    flexor_pBone.custom_shape = oc
    version, ver_major, ver_minor = bpy.app.version
    if version >= 3:
        flexor_pBone.custom_shape_scale_xyz = BONE_CUSTOM_SHAPE_SCALE
    else:
        flexor_pBone.custom_shape_scale = BONE_CUSTOM_SHAPE_SCALE
    flexor_pBone.custom_shape_transform = flexor_pBone
    flexor_pBone.bone.show_wire = True

    #remove temporary custom bone Shape mesh
    boneShapeObj.select_set(True)
    bpy.ops.object.delete()

    ext_ctrl_cpLoConst = addConstraints(extensor_ctrl_pBone, type = 'COPY_LOCATION', name = names["ext_ctrl_copyLocConstName"], target = muscleSys, subtarget = extensor_pBone.name, head_tail = 1)
    msl_cpRotConst = addConstraints(muscle_pBone, type = 'COPY_ROTATION', name = names["msl_copyRotConstName"], target = muscleSys, subtarget = flexor_pBone.name)
    msl_cpLoConst = addConstraints(muscle_pBone, type = 'COPY_LOCATION', name = names["msl_copyLocConstName"], target = muscleSys, subtarget = flexor_pBone.name)
    msl_strToConst = addConstraints(muscle_pBone, type = 'STRETCH_TO', name = names["msl_stretchToCtrlConstName"], target = ctrl, rest_length = muscleLength)
    msl_strToConst2 = addConstraints(muscle_pBone, type = 'STRETCH_TO', name = names["msl_stretchToExtConstName"], target = muscleSys, subtarget = extensor_ctrl_pBone.name, rest_length = muscleLength)
    ext_cpLoConst = addConstraints(extensor_pBone, type = 'COPY_LOCATION', name = names["ext_copyLocConstName"], target = muscleSys, subtarget = counter_pBone.name, head_tail = 1)
    ext_cpRotConst = addConstraints(extensor_pBone, type = 'COPY_ROTATION', name = names["ext_copyRotConstName"], target = muscleSys, subtarget = flexor_pBone.name)
    flx_strToConst = addConstraints(flexor_pBone, type = 'STRETCH_TO', name = names["flx_stretchToConstName"], target = ctrl, rest_length = muscleLength)

    counter_pBone.hide = True
    muscle_pBone.hide = True
    extensor_pBone.hide = True
    extensor_ctrl_pBone.hide = True

    bpy.context.view_layer.objects.active = None

    objName = names["muscleName"]

    if Type == "CONVERT":
        mesh = me
        mesh.name = objName
    else:
        mesh = readMesh(muscleBasisMeshData, objName, scale)

    obj = addMeshObj(mesh, objName, (0,0,0), group_name)
    addVertexGroup(idx, obj, vertexGroupName)

    #parenting 'to bone' mesh with armature
    obj.select_set(True)
    b = muscle_pBone.bone
    muscle_pBone.select = True
    # b.select = True

    muscleSys.data.bones.active = b
    bpy.context.view_layer.objects.active = muscleSys

    bpy.ops.object.parent_set(type='BONE')

    muscleSys.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    addModifiers(obj, vertexGroupName, names["softBodyName"], names["muscleSubsurfName"])

    #adding shapekeys
    objName = names["muscleShapeKeyName"]
    addShapeKey(mesh, obj, "Basis", "EMPTY")

    if Type == "CONVERT":
        addShapeKey(mesh, obj, names["muscleThinName"], "EMPTY")
        addShapeKey(mesh, obj, names["muscleThickName"], "EMPTY")
    else:
        #create additional temoporary muscle meshes for shapekeys
        mesh = readMesh(muscleThinMeshData, objName, scale)
        addShapeKey(mesh, obj, names["muscleThinName"])
        mesh = readMesh(muscleThickMeshData, objName, scale)
        addShapeKey(mesh, obj, names["muscleThickName"])

    setMaterials(obj, group_name, material_diffuse)

    # add drivers
    #flexor
    fcurve = addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToCtrlConstName"]], muscleSys, 'rest_length' , "Base_Length")
    # set coefficient polynomial x to 5.16 the aspect ratio of muscle length flexor and extensor
    fcurve.modifiers.new(type='GENERATOR')
    fcurve.modifiers[0].coefficients[1] = muscleLength
    addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToCtrlConstName"]], muscleSys, 'bulge' , "Volume")

    #extensor
    fcurve = addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToExtConstName"]], muscleSys, 'rest_length' , "Base_Length")
    fcurve.modifiers.new(type='GENERATOR')
    fcurve.modifiers[0].coefficients[1] = muscleLength

    addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToExtConstName"]], muscleSys, 'bulge' , "Volume")
    addDriver(muscleSys.pose.bones[names["mbName"]].constraints[names["msl_stretchToExtConstName"]], muscleSys, 'influence', "Muscle_Type")

    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'goal_min' , "Jiggle_Springiness")
    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'goal_friction' , "Jiggle_Damping")
    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'mass' , "Jiggle_Mass")
    addDriver(obj.modifiers[names["softBodyName"]].settings, obj, 'goal_spring' , "Jiggle_Stiffness")
    addDriver(obj.modifiers[names["muscleSubsurfName"]], obj, 'render_levels' , "Render_Resolution")
    addDriver(obj.modifiers[names["muscleSubsurfName"]], obj, 'levels' , "View3D_Resolution")

    addDriver(obj.data.shape_keys.key_blocks[names["muscleThickName"]], obj, "value", "Muscle_Shape")
    addDriver(obj.data.shape_keys.key_blocks[names["muscleThinName"]], obj, "value", "Muscle_Shape")

    #invert driver's muscle_thin function to range of -1 to 0
    shapeKeys = obj.data.shape_keys
    drivers = shapeKeys.animation_data.drivers[1]
    drivers.modifiers.new(type='GENERATOR')
    drivers.modifiers[0].coefficients[1] = -1

    addDriver(obj.modifiers[names["softBodyName"]], obj, 'show_viewport' , "Dynamics_View3D")
    addDriver(obj.modifiers[names["softBodyName"]], obj, 'show_render' , "Dynamics_Render")

    #select controllers leaving the muscle active
    obj.select_set(True)
    muscleSys.select_set(True)
    ctrl.select_set(True)
    # create xmusculature group and link with the muscles

    objects = bpy.context.selected_objects
    xmusculature = bpy.data.collections.get(group_name, None)
    xmusclea = bpy.data.collections.get(newName, None)

    if not xmusculature:
        xmusculature = bpy.data.collections.new(group_name)
        bpy.context.scene.collection.children.link(xmusculature)
    if not xmusclea:
        xmusclea = bpy.data.collections.new(newName)
        xmusculature.children.link(xmusclea)

    # if group_name in bpy.data.groups:
    #     group = bpy.data.groups[group_name]
    # else:
    #     group = bpy.data.groups.new(group_name)
    # bpy.ops.collection.objects_remove_all()
    for ob in objects:
        for c in ob.users_collection:
            c.objects.unlink(ob)

    for ob in objects:
        if not ob.name in xmusclea.objects:
            xmusclea.objects.link(ob)

    if not useAff:
        suffix = ""
    #decorate name with suffix and autonumber

    decorateName(obj, suffix)
    xmusclea.name = obj.name
    obj.modifiers[names["softBodyName"]].name = obj.name + " Softbody"
    obj.modifiers[names["muscleSubsurfName"]].name = obj.name + " Subsurf"

    #some of properties update functions depend on naming so order of execution matters and function is triggered after decorations
    addMuscleCustomProperties(obj, muscleSys)
    addBoneOrderPropsToMuscle(obj, selectedBones)

    #move mesh layer to active layer only
    # for i in range(len(obj.layers)):
    #     obj.layers[i] = (i == scn.active_layer)

    if scn.Create_Type == 'AUTOAIM' and len(locx) == 2:
        bpy.ops.object.select_all(action='DESELECT')
        ctrl.location = locx[0]
        muscleSys.location = locx[1]

        ctrl.select_set(True)
        b = activeb

        #b.select = True #TODO verify if there's a better way
        bpy.data.objects[rig.name].pose.bones[b.name].select = True
        rig.data.bones.active = b
        bpy.context.view_layer.objects.active = rig

        bpy.ops.object.parent_set(type='BONE')

        ctrl.select_set(False)
        #b.select = False #TODO verify if there's a better way
        bpy.data.objects[rig.name].pose.bones[b.name].select = False

        muscleSys.select_set(True)
        b = selectb

        # b.select = True
        bpy.data.objects[rig.name].pose.bones[b.name].select = True
        rig.data.bones.active = b
        bpy.context.view_layer.objects.active = rig

        bpy.ops.object.parent_set(type='BONE')

        #b.select = False #TODO verify if there's a better way
        bpy.data.objects[rig.name].pose.bones[b.name].select = False
        obj.select_set(True)
        muscleSys.select_set(True)
        ctrl.select_set(True)
        bpy.context.view_layer.objects.active = obj

    else:
        muscleSys.location += origin
        ctrl.location += origin

    scn.tool_settings.lock_object_mode = oldLock
    return{'FINISHED'}


def deleteMuscle(targetObj):

    #remove old, target muscle
    bpy.context.view_layer.objects.active = targetObj
    collection = bpy.data.collections.get(targetObj.users_collection[0].name)
    for ob in collection.objects:
        # bpy.data.objects.remove(ob, do_unlink=True)
        ob.name = "x"
    bpy.data.collections.remove(collection)

    return{'FINISHED'}


def XMirror():
    global on_xMirror
    on_xMirror = True
    success = False
    #part for mesh object
    scn = bpy.context.scene
    oldCursor = Vector(scn.cursor.location)
    setPivotPoint(type = 'CURSOR')
    piv = []
    piv = oldPivot
    activeobj = bpy.context.active_object
    version, ver_major, ver_minor = bpy.app.version

    if activeobj == None:
        return

    #get the Affixes from the names
    prefix = ""
    suffix = ""
    group = get_xmusculature(activeobj)

    if group != "" and group.endswith(defaultGroupName):
            prefix = group[:-len(defaultGroupName)]

    o = bpy.context.selected_objects

    #search for any corrupted muscle
    for n in o:
        if n.type == 'MESH' and getattr(n, "Muscle_XID", None):

            n.select_set(True)
            if n.parent == None:
                setPivotPoint(type= piv)
                return success
            for b in n.parent.pose.bones:
                for c in b.constraints:
                    if c.target != None:
                        if c.target.type == 'EMPTY':
                            ctrl = c.target
                    else:
                        setPivotPoint(type= piv)
                        return success
            ctrl.select_set(True)
            n.parent.hide_viewport = False
            n.parent.select_set(True)

    #get suffix from active bone for comparison
    currSufx = bpy.context.active_object.parent_bone
    idx = currSufx.find("bone")
    currSufx =  currSufx[idx+len("bone"):]


    #unselect muscles (mesh, empties, armatures) without any suffix or different than active muscle
    for obj in bpy.context.selected_objects:
        suffix = ""
        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):

            suffix = obj.parent_bone

            idx = suffix.find("bone")
            suffix = suffix[idx+len("bone"):]

            if  suffix == "" or suffix != currSufx:
                obj.select_set(False)
                for b in obj.parent.pose.bones:
                    for c in b.constraints:
                        if c.target.type == 'EMPTY':
                            ctrl = c.target
                ctrl.select_set(False)
                obj.parent.select_set(False)

            #do not allow to create duplicates, if mirrored muscle already exist
            if not scn.Allow_Duplications:
                for n in scn.objects:
                    if n.name == getMirrorName(obj.name, autonum = 0):
                        obj.select_set(False)
                        for b in obj.parent.pose.bones:
                            for c in b.constraints:
                                if c.target.type == 'EMPTY':
                                    ctrl = c.target
                        ctrl.select_set(False)
                        obj.parent.select_set(False)


    oldPosePosition = None
    oldMode = None
    oldLocation = None
    oldRotation = None
    oldScale = None
    oldHide  = None

    rig = None

    #in case of active muscle corruption search for other armature connections in selected muscles
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):
            muscleSys = obj.parent
            if muscleSys != None:
                rigboneName = muscleSys.parent_bone

                if rigboneName != None:
                    for b in muscleSys.pose.bones:
                            for c in b.constraints:
                                if c.target.type == 'EMPTY':
                                    ctrl = c.target

                    if ctrl.parent != None and rigboneName != "":
                            if ctrl.parent.type == 'ARMATURE':
                                rig = ctrl.parent
                                oldPosePosition = rig.data.pose_position
                                oldMode = rig.mode
                                rig.data.pose_position = 'REST'
                                break
    #ends



    #fixing the x-mirror orientation issues
    if rig != None:
        #preserving the selection
        oldSelection = bpy.context.selected_objects

        oldLocation = Vector(rig.location)
        oldRotation = Euler(rig.rotation_euler)
        oldScale = Vector(rig.scale)
        oldHide = rig.hide_viewport
        rig.hide_viewport = False
        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.rotation_clear(clear_delta=False)
        bpy.ops.object.location_clear(clear_delta=False)
        bpy.ops.object.scale_clear(clear_delta=False)
        rig.select_set(False)
        # reverting back the original selection
        for obj in oldSelection:
                obj.select_set(True)


    #unselect muscles (mesh, empties, armatures) without any suffix or different than active muscle
    for obj in bpy.context.selected_objects:
        suffix = ""
        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):
            #unhide muscle pins
            for m in obj.modifiers:
                if m.type == 'HOOK':
                    m.object.hide_viewport = False
            #unhide microController, muscleSys, microSys and unparent mctrl for x-mirroring (prevents unwanted relocation during
            #transformation apply to muscle meshes)

            if obj.Micro_Controller and obj.select_get() == True:
                microSys = None
                mic_ctrl = None
                sysName = obj.parent.name.replace("System", names["micro_SysName"])
                mName = obj.parent.name.replace("System", names["micro_ctrlName"])

                for ok in scn.objects.keys(): #TODO validate 

                    if sysName == ok:
                        microSys = scn.objects[ok]
                        #unhide and select microSys
                        if microSys.type == 'ARMATURE':

                            microSys.select_set(False)

                    if mName == ok:
                        mic_ctrl = scn.objects[ok]
                        #unhide and select and unparent mctrl
                        if mic_ctrl.type == 'EMPTY':
                            oldmicroHidden = mic_ctrl.hide_viewport
                            mic_ctrl.hide_viewport = False
                            sel = bpy.context.selected_objects
                            bpy.ops.object.select_all(action='DESELECT')
                            mic_ctrl.select_set(True)
                            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
                            for n in sel:
                                n.select_set(True)


    for obj in bpy.context.selected_objects:
        if obj.type == 'EMPTY':
            obj.select_set(False)


    selectedMuscBon = []
    selectedMuscBon = bpy.context.selected_objects
    bpy.ops.object.select_all(action='DESELECT')
    if len(bpy.context.selected_objects) > 0:
        for obj in selectedMuscBon:
            if obj.type == 'ARMATURE':
                obj.select_set(False)
        #apply to all mesh only
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in selectedMuscBon:
            if obj.type == 'MESH':
                obj.select_set(False)
            if obj.type == 'ARMATURE':
                obj.select_set(True)
        #all armatures only

        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in selectedMuscBon:
            if obj.type == 'ARMATURE':
                obj.select_set(False)
            if obj.type == 'MESH':
                obj.select_set(True)
        #all mesh only again

        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    bpy.ops.object.select_all(action='DESELECT')


    #for the sake of simplicity we start all over again....
       #selectMusculature(bpy.context)
    for n in o:
        if n.type == 'MESH' and getattr(n, "Muscle_XID", None):

            n.select_set(True)
            for b in n.parent.pose.bones:
                for c in b.constraints:
                    if c.target.type == 'EMPTY':
                        ctrl = c.target
            ctrl.select_set(True)
            n.parent.select_set(True)


    #unselect muscles without any suffix or different than active muscle
    for obj in bpy.context.selected_objects:
        suffix = ""
        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):

            suffix = obj.parent_bone
            idx = suffix.find("bone")
            suffix = suffix[idx+len("bone"):]

            if  suffix == "" or suffix != currSufx:
                obj.select_set(False)
                for b in obj.parent.pose.bones:
                    for c in b.constraints:
                        if c.target.type == 'EMPTY':
                            ctrl = c.target
                ctrl.select_set(False)
                obj.parent.select_set(False)

            #do not allow to create duplicates, if mirrored muscle already exist
            if not scn.Allow_Duplications:
                for n in scn.objects:
                    if n.name == getMirrorName(obj.name, autonum = 0):
                        obj.select_set(False)
                        for b in obj.parent.pose.bones:
                            for c in b.constraints:
                                if c.target.type == 'EMPTY':
                                    ctrl = c.target
                        ctrl.select_set(False)
                        obj.parent.select_set(False)

            if obj.Micro_Controller and obj.select_get() == True:
                microSys = None
                mic_ctrl = None
                sysName = obj.parent.name.replace("System", names["micro_SysName"])
                mName = obj.parent.name.replace("System", names["micro_ctrlName"])

                for ok in scn.objects.keys(): #TODO validate

                    if sysName == ok:
                        microSys = scn.objects[ok]
                        #show and select microSys for duplication
                        if microSys.type == 'ARMATURE':
                            microSys.select_set(False)

                    if mName == ok:
                        mic_ctrl = scn.objects[ok]
                        #show, select and parent microCtrl for duplication
                        if mic_ctrl.type == 'EMPTY':
                            oldmicroHidden = mic_ctrl.hide_viewport
                            mic_ctrl.hide_viewport = False
                            sel = bpy.context.selected_objects
                            bpy.ops.object.select_all(action='DESELECT')
                            dp(mic_ctrl, obj)
                            mic_ctrl.select_set(True)
                            for n in sel:
                                n.select_set(True)
            #bendy muscle support and multiple hooks for muscle pinning
            if obj.select_get() == True:
                for m in obj.modifiers:
                    if m.type == 'HOOK':
                        if m.object != None:
                            o = m.object
                            if getMirrorName(o.name, autonum = 0) not in scn.objects.keys():  #TODO validate  
                                o.select_set(True)

    bpy.ops.object.duplicate()

    for obj in bpy.context.selected_objects:
        if obj.type == 'EMPTY':
            obj.select_set(False)


    for obj in bpy.context.selected_objects:
        oldPinning = None
        oldPinning = obj.Pinning_View3D
        #obj.Pinning_View3D = False

        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):

            #get parent before unparent
            mesh = obj
            bpy.context.view_layer.objects.active = mesh
            muscleSys = obj.parent # fixme no reference to muscleSys

            if muscleSys != None:
                rigboneName = muscleSys.parent_bone # fixme

            muscleBoneName = obj.parent_bone

            #we need to override the context of our operator
            override = get_override( 'VIEW_3D', 'WINDOW' )
            #get and store the original origin for mirring
            tmp_override(override, 'snap_cursor_to_active')  

            newCursor = Vector(scn.cursor.location)
            newCursor.x = -newCursor.x

            #unhide microController, mSys and unparent mctrl for x-mirroring (prevents unwanted relocation during transformation apply to muscle meshes)


            if obj.Micro_Controller:
                mic_ctrl = None
                mName = obj.parent.name.replace("System", names["micro_ctrlName"])

                for ok in scn.objects.keys(): #TODO validate


                    if mName == ok:
                        mic_ctrl = scn.objects[ok]
                        #unparent any new microcontroller and prepare for mirroring
                        if mic_ctrl.type == 'EMPTY':
                            sel = bpy.context.selected_objects
                            bpy.ops.object.select_all(action='DESELECT')
                            mic_ctrl.select_set(True)


                            mic_ctrl.name = getMirrorName(mic_ctrl.name)
                            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
                            for n in sel:
                                n.select_set(True)


            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)

            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

            oldmeshLocation = Vector(obj.location)
            oldmeshRotation = Euler(obj.rotation_euler)
            oldmeshScale = Vector(obj.scale)


            bpy.ops.object.rotation_clear(clear_delta=False)
            bpy.ops.object.location_clear(clear_delta=False)
            bpy.ops.object.scale_clear(clear_delta=False)

            #mirror muscleMesh
            scn.cursor.location = (0,0,0)
            override = get_override( 'VIEW_3D', 'WINDOW' )
            tmp_override(override, 'mirror')

            obj.location = oldmeshLocation
            obj.rotation_euler = oldmeshRotation

            obj.location[0] = -obj.location[0]
            obj.rotation_euler[1] = -obj.rotation_euler[1]
            obj.rotation_euler[2] = -obj.rotation_euler[2]
            obj.scale = oldmeshScale
            obj.scale[0] = -obj.scale[0]

            scn.cursor.location = newCursor
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR')

            #normalize mirrored muscleMesh scale
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            hookscalex(obj, oldmeshScale)
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')

            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode='OBJECT')

            #part for muscle armature bones
            obj.select_set(False)
            muscleSys.select_set(True)
            bpy.context.view_layer.objects.active = muscleSys

            #fixme error
            #fixme orientation accuracy issue
            bpy.ops.object.select_all(action='DESELECT')
            muscleSys.select_set(True)

            if muscleSys.parent != None:
                        if muscleSys.parent.type == 'ARMATURE':
                            mSysParentReady = True

            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

            muscleSys.location[0] = -muscleSys.location[0]
            muscleSys.rotation_euler[1] = -muscleSys.rotation_euler[1]
            muscleSys.rotation_euler[2] = -muscleSys.rotation_euler[2]
            #edit mode bones
            bpy.ops.object.mode_set(mode='EDIT')

            #select all for xmirror in edit mode
            if muscleSys.type == 'ARMATURE':
                armature = muscleSys.data

                for bone in armature.edit_bones:
                    bone.select = True
                    bone.select_tail = True
                    bone.select_head = True

                #xmirror bones median and move -x
                #we need to override the context of our operator
                scn.cursor.location = (0,0,0)
                override = get_override( 'VIEW_3D', 'WINDOW' )

                for bone in armature.edit_bones:
                    bone.select = False
                    bone.select_tail = False
                    bone.select_head = False

                #selecting the head of the bone for setting the origin point
                armature.edit_bones[muscleBoneName].select_head = True

                override = get_override( 'VIEW_3D', 'WINDOW' )
                tmp_override(override, 'snap_cursor_to_selected')

                bpy.ops.object.mode_set(mode='OBJECT')
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR')


                #setting the empty position results with final muscle position
                for b in muscleSys.pose.bones:
                    for c in b.constraints:
                        if c.target.type == 'EMPTY':
                            ctrl = c.target

                ctrl.select_set(True)
                rigboneNameCtrl = ctrl.parent_bone
                rig = ctrl.parent
                bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
                ctrl.location.x = -ctrl.location.x

                #make auto parenting on mirrored side to model's armature if available

                #note - These two elements must be parented before muscle shape is.
                #pseudocode
                if rig != None and rigboneName != "":
                    if rig.type == 'ARMATURE':


                      if mSysParentReady:

                        #exit from rig pose mode before parenting
                        bpy.ops.object.mode_set(mode='OBJECT')
                        #parenting 'to bone' mesh with armature
                        bpy.ops.object.select_all(action='DESELECT')
                        muscleSys.select_set(True)

                        xname = getMirrorName(rigboneName, autonum = 0)

                        #doublecheck if xmirrored name is really valid and search for suffix another time
                        if xname == rigboneName:
                            suffix, name = separateSuffixAndName(rigboneName)

                            if suffix != "":
                                csuffix = counterSuffix(suffix)
                                xname = rigboneName.replace(suffix, csuffix, 1)

                        # just for convenience  - if bone layer was hidden make it visible to succeed                                                                      
                        hide_bone_id = temp_unhide_bone(rig, xname)

                        rig.pose.bones[xname].select = True
                        b = rig.pose.bones[xname].bone
                        # b.select = True
                        rig.data.bones.active = b


                        if rig.data.bones.active == None:
                            ctrl.parent = None
                            ctrl.parent_bone = ""
                        else:
                            bpy.context.view_layer.objects.active = rig

                            bpy.ops.object.parent_set(type='BONE')


                            #exit from rig pose mode after parenting
                            bpy.ops.object.mode_set(mode='OBJECT')

                            bpy.ops.object.select_all(action='DESELECT')

                            #ctrl.parent_bone = ctrlParent  # needs reparenting and switching parent name suffix
                            ctrl.select_set(True)

                            xname = getMirrorName(rigboneNameCtrl, autonum = 0)

                            #doublecheck if xmirrored name is really valid and search for suffix another time
                            if xname == rigboneNameCtrl:
                                suffix, name = separateSuffixAndName(rigboneNameCtrl)

                                if suffix != "":
                                    csuffix = counterSuffix(suffix)
                                    xname = rigboneNameCtrl.replace(suffix, csuffix, 1)

                            rig.pose.bones[xname].select = True
                            b = rig.pose.bones[xname].bone

                            # b.select = True
                            rig.data.bones.active = b
                            bpy.context.view_layer.objects.active = rig

                            bpy.ops.object.parent_set(type='BONE')

                            if hide_bone_id != None:
                                if version >= 4:
                                    rig.data.collections[hide_bone_id].is_visible = False
                                else:
                                    rig.data.layers[hide_bone_id] = False

                            #remember to mirror names for the list of bones muscles (AUTOAIM) feature
                            if obj.Bone_List != None:
                                for o in obj.Bone_List:
                                    #todo check if they are still valid
                                    o.name = getMirrorName(o.name, autonum = 0)
                else:
                    ctrl.parent_bone = ""
                    ctrl.parent = None

                #make muscle shape parenting back to bone mechanism
                bpy.ops.object.select_all(action='DESELECT')
                mesh.select_set(True)
                b = muscleSys.pose.bones[muscleBoneName].bone
                muscleSys.pose.bones[muscleBoneName].select = True
                #b.select = True
                muscleSys.data.bones.active = b
                bpy.context.view_layer.objects.active = muscleSys

                bpy.ops.object.parent_set(type='BONE')
                #FIXME move to better place (include hooks and microctrls)
                xmusculature_nme = get_xmusculature(obj)
                xmusculature = bpy.data.collections.get(xmusculature_nme, None)
                oldxmusclea = obj.users_collection[0]
                rename(obj, getMirrorName(obj.name), getMirrorName(muscleSys.name), getMirrorName(muscleBoneName, autonum = 0), getMirrorName(ctrl.name))
                xmusclea = bpy.data.collections.get(obj.name, None)

                if not xmusclea:
                    xmusclea = bpy.data.collections.new(obj.name)
                    xmusculature.children.link(xmusclea)

                xmusclea.objects.link(obj)
                xmusclea.objects.link(ctrl)
                xmusclea.objects.link(muscleSys)
                if obj.Micro_Controller:
                    xmusclea.objects.link(mic_ctrl)
                    oldxmusclea.objects.unlink(mic_ctrl)

                oldxmusclea.objects.unlink(obj)
                oldxmusclea.objects.unlink(ctrl)
                oldxmusclea.objects.unlink(muscleSys)

                #support for xmirroring bendy muscles and multiple hooks for muscle pinning
                for m in obj.modifiers:
    
                    if 'SOFT_BODY' in m.type:
                        xname = get_xmusculature(obj)
                        m.name = obj.name + " Softbody"
                        
                    if 'SUBSURF' in m.type:
                        m.name = obj.name + " Subsurf"
                        
                    if m.type == 'HOOK':
                        if m.object != None:

                            bpy.ops.object.select_all(action='DESELECT')

                            xhook(m, oldmeshScale)
                            suff, name = separateSuffixAndName(m.object.name, autonum = 4)

                            #fixme not always ctrl will be mirrored
                            if suff != "" and ('_ctrl' not in m.object.name and 'System' not in m.object.name):

                                obj_suff, obj_name = separateSuffixAndName(obj.name, autonum = 0)

                                if counterSuffix(suff) == obj_suff:

                                    rigboneName = None
                                    if m.object.parent != None:
                                        rig = m.object.parent
                                        rigboneName = rig.pose.bones[getMirrorName(m.object.parent_bone, autonum = 0)].bone.name

                                    #if hook is mirrored already don't proceed any transformations
                                    oldObj = scn.objects[getMirrorName(m.object.name, autonum = 0)]

                                    m.object.select_set(True)
                                    bpy.context.view_layer.objects.active = m.object
                                    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
                                    m.object.location.x = -m.object.location.x
                                    m.object.rotation_euler.y = -m.object.rotation_euler.y
                                    m.object.rotation_euler.z = -m.object.rotation_euler.z

                                    #decorate names for muscle pins and pin vertexgroups and mods
                                    m.object.name = getMirrorName(m.object.name, autonum = 4)
                                    xmusclea.objects.link(m.object)
                                    oldxmusclea.objects.unlink(m.object)

                                    m.name = m.object.name
                                    for v in obj.vertex_groups:
                                                if v.name == m.vertex_group:
                                                    #update itemlist first
                                                    for c in obj.custom:
                                                        if c.name == v.name:
                                                            c.name = m.object.name
                                                    v.name = m.object.name
                                    m.vertex_group = m.object.name


                                    if rigboneName != None:
                                        m.object.select_set(True)

                                        # just for convenience  - if bone layer was hidden make it visible to succeed
                             
                                        hide_bone_id = temp_unhide_bone(rig, rigboneName)
                                        b = rig.pose.bones[rigboneName].bone
                                        rig.pose.bones[rigboneName].select = True
                                        # b.select = True
                                        rig.data.bones.active = b
                                        bpy.context.view_layer.objects.active = rig
                                        bpy.ops.object.parent_set(type='BONE')


                                        if hide_bone_id != None:                                            
                                            if version >= 4:
                                                rig.data.collections[hide_bone_id].is_visible = False
                                            else:
                                                rig.data.layers[hide_bone_id] = False
                                        rig.select_set(False)
                                        #break


                #parent back the microController
                if obj.Micro_Controller:

                    mic_ctrl = None
                    microSys = None


                    #search for existing microcontrollers
                    sName = obj.parent.name.replace("System", names["micro_SysName"])
                    mName = obj.parent.name.replace("System", names["micro_ctrlName"])
                    #parent mirrored microSys to the armature's opposite bone

                    for ok in scn.objects.keys(): #TODO validate

                        if mName == ok:
                            mic_ctrl = scn.objects[ok]
                            #unparent any new microcontroller and prepare for mirroring
                            if mic_ctrl.type == 'EMPTY':

                                skinFound = False
                                #getting the skin
                                for o in scn.objects:
                                    for m in o.modifiers:
                                        if m.type == 'SHRINKWRAP':
                                            if m.target != None:

                                                if m.target.name == getMirrorName(obj.name, autonum = 0):
                                                    skinFound = True
                                                    break
                                if not skinFound:
                                    updateHookSetAttributes(obj, mic_ctrl)

                                sel = bpy.context.selected_objects
                                bpy.ops.object.select_all(action='DESELECT')
                                setPivotPoint(type = 'CURSOR')


                                #x-mirroring the microController
                                scn.cursor.location = (0,0,0)

                                bpy.ops.object.select_all(action='DESELECT')


                                mic_ctrl.select_set(True)
                                bpy.context.view_layer.objects.active = mic_ctrl

                                #todo create shadow microController from the "Set" state and also bind it to the empty helper
                                bpy.ops.object.duplicate()
                                shadowHook = bpy.context.active_object
                                shadowHook.name = "shadowHook" + shadowHook.name

                                #select shadowHook for parenting to the
                                oldTension = obj.Skin_Tension
                                executeHookSetAttributes(obj, shadowHook)


                                mic_ctrl.location[0] = -mic_ctrl.location[0]

                                mic_ctrl.rotation_euler[1] = -mic_ctrl.rotation_euler[1]
                                mic_ctrl.rotation_euler[2] = -mic_ctrl.rotation_euler[2]
                                #transfer attributes to mic_ctrl
                                mic_ctrl.HookSet_Attributes[0].location = shadowHook.location
                                mic_ctrl.HookSet_Attributes[0].location[0] = -shadowHook.location[0]
                                mic_ctrl.HookSet_Attributes[0].rotation[0] = shadowHook.rotation_euler[0]
                                mic_ctrl.HookSet_Attributes[0].rotation[1] = -shadowHook.rotation_euler[1]
                                mic_ctrl.HookSet_Attributes[0].rotation[2] = -shadowHook.rotation_euler[2]
                                mic_ctrl.HookSet_Attributes[0].scale = shadowHook.scale
                                #get tension from muscleObj
                                mic_ctrl.HookSet_Attributes[0].tension = obj.Skin_Tension
                                obj.Skin_Tension = oldTension

                                bpy.ops.object.select_all(action='DESELECT')
                                #delete empty helper and shadow empty
                                shadowHook.select_set(True)
                                mic_ctrl.select_set(False)

                                bpy.ops.object.delete()
                                bpy.ops.object.select_all(action='DESELECT')
                                mic_ctrl.select_set(True)
                                #parent back to the mesh

                                dp(mic_ctrl, obj)

                                for n in sel:
                                    n.select_set(True)

                                #todo refactor to function recreate micSystem
                                NAME_COL = 0
                                boneTable = [(names["micro_rtbName"], None, (0.1,0,0)),
                                             (names["micro_empbName"], None, (0.1,0,0)),
                                             (names["micro_tgtbName"], names["micro_rtbName"], (0.1,0,0)),]


                                #in case of active muscle corruption search for other armature connections in selected muscles
                                muscleSys = obj.parent
                                      #in case of active muscle corruption search for other armature connections in selected muscles

                                if muscleSys != None:
                                    rigboneName = muscleSys.parent_bone

                                    if rigboneName != None:
                                        for b in muscleSys.pose.bones:
                                                for c in b.constraints:
                                                    if c.target.type == 'EMPTY':
                                                        ctrl = c.target

                                        if ctrl.parent != None and rigboneName != "":
                                                if ctrl.parent.type == 'ARMATURE':
                                                    rig = ctrl.parent

                                                    #fixing the controllers  orientation issues

                                                    bpy.ops.object.select_all(action='DESELECT')
                                                    bpy.context.view_layer.objects.active = rig
                                                    rig.select_set(True)
                                                    bpy.ops.object.mode_set(mode='OBJECT')
                                                    bpy.ops.object.rotation_clear(clear_delta=False)
                                                    bpy.ops.object.location_clear(clear_delta=False)
                                                    bpy.ops.object.scale_clear(clear_delta=False)

                                                    rig.data.pose_position = 'REST'
                                                    rig.select_set(False)


                                location = Vector((0,0,0))
                                #create and setup microSys armature
                                microSys = createArmature(names["micro_SysName"], boneTable, location)

                                root_pBone = microSys.pose.bones[boneTable[NAME_COL][NAME_COL]]
                                emptyConst_pBone = microSys.pose.bones[boneTable[NAME_COL+1][NAME_COL]]
                                target_pBone = microSys.pose.bones[boneTable[NAME_COL+2][NAME_COL]]

                                bpy.ops.object.mode_set(mode='EDIT')
                                microSys.data.edit_bones[root_pBone.name].head = (0,0,0)
                                microSys.data.edit_bones[root_pBone.name].tail = (-1,0,0)
                                microSys.data.edit_bones[target_pBone.name].head = (0,0,0)
                                microSys.data.edit_bones[target_pBone.name].tail = (0,1,0)
                                microSys.data.edit_bones[emptyConst_pBone.name].head = (0,0,0)
                                microSys.data.edit_bones[emptyConst_pBone.name].tail = (-1,0,0)
                                bpy.ops.object.mode_set(mode='OBJECT')

                                #total 180degree translation

                                bpy.ops.object.select_all(action='DESELECT')
                                microSys.select_set(True)
                                bpy.ops.transform.rotate(value=3.14159, orient_axis='Z', constraint_axis=(False, False, True), orient_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1)

                                #exit from rig pose mode after parenting
                                bpy.ops.object.mode_set(mode='OBJECT')
                                bpy.ops.object.select_all(action='DESELECT')
                                #ctrl.parent_bone = ctrlParent  # needs reparenting and switching parent name suffix
                                #parent micSys to the rig if available (separate function) (in restmode)
                                microSys.select_set(True)
                                if ctrl.parent != None and rigboneName != "":

                                    # just for convenience  - if bone layer was hidden make it visible to succeed
                             
                                    hide_bone_id = temp_unhide_bone(rig, rigboneName)
                                    #parent System_Micro to Armature's same bone as parented muscleSys
                                    b = rig.pose.bones[rigboneName].bone
                                    rig.pose.bones[rigboneName].select = True
                                    #b.select = True 
                                    rig.data.bones.active = b
                                    bpy.context.view_layer.objects.active = rig
                                    bpy.ops.object.parent_set(type='BONE')

                                    if hide_bone_id:                                        
                                        if version >= 4:
                                            rig.data.collections[hide_bone_id].is_visible = False
                                        else:
                                            rig.data.layers[hide_bone_id] = False
                                    rig.select_set(False)

                                #set root constraints target to Armature
                                #setup the micSys rootbone constraint if muscle is parented
                                rt_cpLoConst = addConstraints(root_pBone, type = 'COPY_LOCATION', name = names["micro_rt_copyLocConstName"], target = rig)
                                rt_cpRotConst = addConstraints(root_pBone, type = 'COPY_ROTATION', name = names["micro_rt_copyRotConstName"], target = rig)
                                rt_scaConst = addConstraints(root_pBone, type = 'COPY_SCALE', name = names["micro_rt_copyScaConstName"], target = rig)

                                emp_cpLoConst = addConstraints(emptyConst_pBone, type = 'COPY_LOCATION', name = names["micro_emp_copyLocConstName"], target = mic_ctrl)
                                emp_cpRotConst = addConstraints(emptyConst_pBone, type = 'COPY_ROTATION', name = names["micro_emp_copyRotConstName"], target = mic_ctrl)
                                emp_scaConst = addConstraints(emptyConst_pBone, type = 'COPY_SCALE', name = names["micro_emp_copyScaConstName"], target = mic_ctrl)

                                tgt_cpLoConst = addConstraints(target_pBone, type = 'COPY_LOCATION', name = names["micro_tgt_copyLocConstName"], target = microSys, subtarget = emptyConst_pBone.name, target_space ='LOCAL', owner_space ='LOCAL')
                                tgt_cpRotConst = addConstraints(target_pBone, type = 'COPY_ROTATION', name = names["micro_tgt_copyRotConstName"], target = microSys, subtarget = emptyConst_pBone.name, target_space ='LOCAL', owner_space ='LOCAL')
                                tgt_scaConst = addConstraints(target_pBone, type = 'COPY_SCALE', name = names["micro_tgt_copyScaConstName"], target = microSys, subtarget = emptyConst_pBone.name, target_space ='LOCAL', owner_space ='LOCAL')

                                #move to different layer?

                                microSys.select_set(True)

                                #add  microController's objects to the muscle group
                                ob = bpy.context.selected_objects
                                #FIXME

                                xmusclea = bpy.data.collections.get(obj.name, None)
                                # bpy.ops.collection.objects_remove_all()
                                for o in ob:
                                    for c in o.users_collection:
                                        c.objects.unlink(o)
                                for o in ob:
                                    if not xmusclea in o.users_collection:
                                        xmusclea.objects.link(o)

                                # for group in bpy.data.groups:
                                #   if muscleSys.name in group.objects:
                                #       for o in ob:
                                #         group.objects.link(o)

                                #hide microSys armature
                                microSys.hide_viewport = True
                                root_pBone.hide = True
                                emptyConst_pBone.hide = True
                                target_pBone.hide = True

                                #add microSys to naming system
                                microSys.name = muscleSys.name.replace("System", microSys.name)



    if oldPosePosition != None:
        rig.data.pose_position = oldPosePosition
    if oldMode != None:
        bpy.context.view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode=oldMode)


        if oldMode == 'POSE':
            bpy.ops.pose.select_all(action='DESELECT')
        else:
            bpy.ops.object.select_all(action='DESELECT')

    if oldLocation != None:
        rig.location = oldLocation
    if oldRotation != None:
        rig.rotation_euler = oldRotation
    if oldScale != None:
        rig.scale = oldScale

    if oldHide != None:
        rig.hide_viewport = oldHide

    #recover cursor's position
    setPivotPoint(type= piv)
    scn.cursor.location = oldCursor
    success = True
    on_xMirror = False
    return success


def smartUpdate(context):
    scn = bpy.context.scene
    s = scn.frame_start
    e = scn.frame_end
    bpy.context.view_layer.objects.active = None

    bpy.ops.ptcache.free_bake_all()
    oldFrame = scn.frame_current
    scn.frame_current = 0

    for ob in scn.objects:
        if getattr(ob, "Muscle_XID", None) and ob.type == 'MESH':
            for mod in ob.modifiers:
                if 'SOFT_BODY' in mod.type:
                    ob.select_set(False)
                    ptc = mod.point_cache
                    for n in ptc.point_caches:
                        n.name = n.name + "_old"

                    override = bpy.context.copy()
                    override['blend_data'] =  bpy.data
                    override['scene'] = scn
                    override['active_object'] = ob.name
                    override['point_cache'] = mod.point_cache
                    tmp_override(override, 'ptcache_add')
                    ptc = mod.point_cache
                    ptc.frame_step = 1

                    ptc.frame_start = s
                    ptc.frame_end = e
                    ptc.name =  "XMuscle_cache"
                    override = bpy.context.copy()
                    override['blend_data'] =  bpy.data
                    override['scene'] = scn
                    override['active_object'] = ob.name
                    override['point_cache'] = mod.point_cache

                    end = ptc.point_caches.active_index
                    for i in range(0, end):
                        end = ptc.point_caches.active_index = i
                        override = bpy.context.copy()
                        override['blend_data'] =  bpy.data
                        override['scene'] = scn
                        override['active_object'] = ob.name
                        override['point_cache'] = mod.point_cache
                        tmp_override(override, 'ptcache_remove')


    bpy.ops.ptcache.bake_all(bake=True)
    scn.frame_current = oldFrame
    bpy.ops.ptcache.free_bake_all()



def displayNames(context):

    objects = bpy.data.objects
    isShown = False

    for obj in objects:

        if not getattr(obj, "Muscle_XID", None):
            continue
        if obj.show_name == True:
            isShown = True
            break


    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if isShown:
            if obj.show_name == True:
                obj.show_name = False

        else:
            if obj.select_get() == True:
                obj.show_name = True


def display_pin_Visibility(context):
    objects = bpy.data.objects
    xray = False
    isHidden = False
    done = False

    for obj in objects:

        if not getattr(obj, "Muscle_XID", None):
            continue

        #unhide muscle pins
        for m in obj.modifiers:
            if m.type == 'HOOK':
                if m.object.hide_viewport == True:
                    isHidden = True
                    break
            
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if isHidden:
            for m in obj.modifiers:
                if m.type == 'HOOK':
                    if m.object.hide_viewport == True:
                        m.object.hide_viewport = False
                        done = True

        else:
            if obj.select_get() == True:
                for m in obj.modifiers: 
                    if m.type == 'HOOK':
                        m.object.hide_viewport = True
                        done = True
    if done:
        return
    
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if obj.select_get() == False and obj.hide_viewport == False and obj.show_in_front == False:
            xray= True
            break
        else:
            xray = False
            break
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        for m in obj.modifiers:
            if m.type == 'HOOK':
                if m.object.hide_viewport == True:
                    m.object.hide_viewport = False
                    #do something
                if obj.select_get() == False and obj.hide_viewport == False:
                    for m in obj.modifiers:
                        if m.type == 'HOOK':
                            if m.object.hide_viewport == False:
                                m.object.show_in_front = xray


def displayVisibility(context):

    objects = bpy.data.objects
    xray = False
    isHidden = False
    done = False

    for obj in objects:

        if not "Muscle_XID" in obj.keys():
            continue
        if obj.hide_viewport == True:
            isHidden = True
            break
    
    #very important fix
    isolateSelected(bpy.context, unhide_all=True)
    
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if isHidden:
            if obj.hide_viewport == True:
                obj.hide_viewport = False
                done = True

        else:
            if obj.select_get() == True:
                obj.hide_viewport = True
                done = True
    if done:
        return
    
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if obj.select_get() == False and obj.hide_viewport == False and obj.show_in_front == False:
            xray= True
            break
        else:
            xray = False
            break
    for obj in objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if obj.select_get() == False and obj.hide_viewport == False:
            obj.show_in_front = xray
        

def displayWireframe(context):
    objects = bpy.data.objects

    for obj in objects:
            if not getattr(obj, "Muscle_XID", None):
                continue
            else:
                if obj.select_get() == True:
                    if obj.type == 'MESH':

                        if  obj.display_type == 'WIRE':
                            obj.display_type = 'TEXTURED'
                            obj.show_wire =  not obj.show_wire
                            obj.show_all_edges =  not obj.show_all_edges
                        elif obj.show_wire:
                            obj.display_type = 'WIRE'
                            obj.show_wire = False
                            obj.show_all_edges =  False
                        if  obj.display_type == 'TEXTURED':
                            obj.show_wire =  not obj.show_wire
                            obj.show_all_edges =  not obj.show_all_edges


def displayXray(context):

    objects = bpy.data.objects

    for obj in objects:
            if not getattr(obj, "Muscle_XID", None):
                continue
            else:
                if obj.select_get() == True:
                    if obj.type == 'MESH':
                        obj.show_in_front = not obj.show_in_front


def get_parent_collection_names(collection, parent_names):
  for parent_collection in bpy.data.collections:
    if collection.name in parent_collection.children.keys(): #TODO validate
      parent_names.append(parent_collection.name)
      get_parent_collection_names(parent_collection, parent_names)
      return

#TODO rename - get xmu name
def get_xmusculature(obj):
    parent_collection = obj.users_collection[0]
    parent_names      = []
    parent_names.append(parent_collection.name)
    get_parent_collection_names(parent_collection, parent_names)
    prefix = ""

    for name in parent_names:
        if name != "" and name.endswith(defaultGroupName):
            prefix = name[:-len(defaultGroupName)]
            break
    group_name = prefix + defaultGroupName
    return group_name


def selectMusculature(context):
    activeobj = context.active_object
    if activeobj == None:
        return

    group_name = get_xmusculature(activeobj)

    xmusculature = bpy.data.collections.get(group_name, None)
    for b in xmusculature.children:
    # bpy.ops.object.select_same_group(group=group_name)
        bpy.ops.object.select_same_collection(collection=b.name)


def selectSingleMuscle(context):
    activeobj = context.active_object
    scn = context.scene
    if activeobj == None:
        return

    activeobj.select_set(True)
    for b in activeobj.parent.pose.bones:
        for c in b.constraints:
            if c.target.type == 'EMPTY':
                ctrl = c.target
    ctrl.select_set(True)
    activeobj.parent.select_set(True)

    mname = activeobj.parent.name.replace("System", names["micro_ctrlName"])
    for o in scn.objects.keys(): #TODO validate
        if mname == o:
            mic_ctrl = scn.objects[o]
            if mic_ctrl.type == 'EMPTY':
                mic_ctrl.select_set(True)


#todo microController implementation



class OBJECT_OT_addMuscle(bpy.types.Operator):
    """Add stylized Muscle"""
    bl_idname = "muscle.add_muscle"
    bl_label = "Add stylized Muscle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        createMuscle("STYLIZED")
        return{'FINISHED'}


class OBJECT_OT_addStripMuscle(bpy.types.Operator):
    """Add strip Muscle"""
    bl_idname = "muscle.add_strip_muscle"
    bl_label = "Add Strip Muscle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        createMuscle("STRIP")
        return{'FINISHED'}


class OBJECT_OT_addBasicMuscle(bpy.types.Operator):
    """Add basic Muscle"""
    bl_idname = "muscle.add_basic_muscle"
    bl_label = "Add basic Muscle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        createMuscle("BASIC")
        return{'FINISHED'}


class OBJECT_OT_convertToMuscle(bpy.types.Operator):
    """Convert mesh to muscle"""
    bl_idname = "muscle.convert_to_muscle"
    bl_label = "Convert mesh to muscle. Works also with auto-aim"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        createMuscle("CONVERT")
        return{'FINISHED'}

class OBJECT_OT_setMaterials(bpy.types.Operator):
    """Set new material to  active muscle"""
    bl_idname = "muscle.set_materials"
    bl_label = "Set new material to  active muscle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.selected_objects
        if context.active_object == None and context.active_object.type == 'MESH':
            return {'FINISHED'}
        for o in obj:
            if o.type == 'MESH':
                setMaterials(o, defaultMaterialName+context.active_object.name, material_diffuse)
        return{'FINISHED'}


class OBJECT_OT_resetProps(bpy.types.Operator):
    """Reset muscle properties"""
    bl_idname = "muscle.reset_props"
    bl_label = "Reset muscle properties"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        if context.active_object == None:
            return{'FINISHED'}
        addMuscleCustomProperties(context.object, context.object.parent)
        return{'FINISHED'}


class OBJECT_OT_smartUpdate(bpy.types.Operator):
    """Forces dynamics update for all muscles"""
    bl_idname = "muscle.smart_update"
    bl_label = "Forces update dynamics for all muscles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        smartUpdate(context)
        return{'FINISHED'}

class OBJECT_OT_wireframe(bpy.types.Operator):
    """Toggle change muscle display to wireframe"""
    bl_idname = "muscle.wireframe"
    bl_label = "Toggle change muscle display to wireframe"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        displayWireframe(context)
        return{'FINISHED'}


class OBJECT_OT_xray(bpy.types.Operator):
    """Toggle X-Ray display for Muscles"""
    bl_idname = "muscle.xray"
    bl_label = "Toggle X-Ray display for Muscles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        displayXray(context)
        return{'FINISHED'}

class OBJECT_OT_dispNames(bpy.types.Operator):
    """Display muscle names"""
    bl_idname = "muscle.dispnames"
    bl_label = "Display muscle names"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        displayNames(context)
        return{'FINISHED'}


class OBJECT_OT_visibility(bpy.types.Operator):
    """Toggle Hide or Show muscles"""
    bl_idname = "muscle.visibility"
    bl_label = "Toggle Hide or Show muscles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        displayVisibility(context)
        return{'FINISHED'}

class OBJECT_OT_pins_visibility(bpy.types.Operator):
    """Toggle Hide or Show muscle pins"""
    bl_idname = "muscle.pins_visibility"
    bl_label = "Toggle Hide or Show muscle pins"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        display_pin_Visibility(context)
        return{'FINISHED'}
    
class OBJECT_OT_selectMusculature(bpy.types.Operator):
    """Selects active muscle's XMusculature"""
    bl_idname = "muscle.selectall"
    bl_label = "Selects active muscle's XMusculature"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selectMusculature(context)
        return{'FINISHED'}


class OBJECT_OT_selectMuscle(bpy.types.Operator):
    """Selects all elements of active muscle"""
    bl_idname = "muscle.selectsingle"
    bl_label = "Selects all elements of active muscle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selectSingleMuscle(context)
        return{'FINISHED'}


class OBJECT_OT_isolateMuscle(bpy.types.Operator):
    """Toggle isolation selected muscles"""
    bl_idname = "muscle.isolate"
    bl_label = "Toggles isolation of selected muscles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        isolateSelected(context)
        return{'FINISHED'}


def isolateSelected(context, unhide_all = False):
    for obj in context.scene.objects:
        if not getattr(obj, "Muscle_XID", None):
            continue
        if obj != None:
            if obj.type == 'MESH':
                if obj.select_get() == False and obj.hide_viewport == False and not unhide_all:

                    obj.Muscle_View3D = False
                    obj.Micro_Controller_View3D = False
                    obj.hide_viewport = True
                    for m in obj.modifiers:
                        if m.type == 'HOOK':
                            if m.object.type == 'EMPTY':
                                m.object.hide_viewport = True

                    for b in obj.parent.pose.bones:
                        for c in b.constraints:
                            if c.target != None:
                                if c.target.type == 'EMPTY':
                                    ctrl = c.target

                                    ctrl.hide_viewport = True
                                    obj.parent.hide_viewport = True
                    if obj.Micro_Controller and obj.select_get() == False:
                        mic_ctrl = None
                        mName = obj.parent.name.replace("System", names["micro_ctrlName"])
                        for ok in context.scene.objects.keys(): #TODO validate
                            if mName == ok:
                                mic_ctrl = context.scene.objects[ok]
                                if mic_ctrl.type == 'EMPTY':
                                    mic_ctrl.hide_viewport = True

                elif obj.select_get() == False and obj.hide_viewport == True:

                    obj.Muscle_View3D = True
                    obj.Micro_Controller_View3D = True
                    obj.hide_viewport = False

                    for b in obj.parent.pose.bones:
                        for c in b.constraints:
                            if c.target != None:
                                if c.target.type == 'EMPTY':
                                    ctrl = c.target

                                    ctrl.hide_viewport = False
                                    obj.parent.hide_viewport = False
                    if obj.Micro_Controller and obj.select_get() == False:
                        mic_ctrl = None
                        mName = obj.parent.name.replace("System", names["micro_ctrlName"])
                        for ok in context.scene.objects.keys():
                            if mName == ok:
                                mic_ctrl = context.scene.objects[ok]
                                if mic_ctrl.type == 'EMPTY':
                                    mic_ctrl.hide_viewport = False

                oldFrame = context.scene.frame_current
                context.scene.frame_current = oldFrame+1
                context.scene.frame_current = oldFrame


class OBJECT_OT_setMicroController(bpy.types.Operator):
    """Applies active muscle's MicroController's properties"""
    bl_idname = "muscle.set"
    bl_label = "Set MicroController"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        if type(self.obj) is list:
            if len(self.obj) > 1:
                for o in self.obj:
                    if not getattr(o, "Muscle_XID", None):
                        continue
                    setMicroController(o, context)
        else:
            setMicroController(self.obj, context)

        return{'FINISHED'}



    def invoke(self, context, event):

        if event.alt: # for Multi-selection
            self.obj = context.selected_objects
        else:
            self.obj = context.active_object
        return self.execute(context)


class OBJECT_OT_xMirror(bpy.types.Operator):
    """XMirror selected muscles"""
    bl_idname = "muscle.xmirror"
    bl_label = "XMirror selected muscles"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        success = XMirror()

        if not success:
            self.report({'ERROR'}, "Corrupted muscle detected")
        return{'FINISHED'}


class OBJECT_OT_Muscle_Rename(bpy.types.Operator):
    """Renames selected muscle and its components"""
    bl_idname = "muscle.rename_muscle"
    bl_label = "Rename selected muscle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):

        if type(self.obj) is list:
            if len(self.obj) > 1:
                for o in self.obj:
                    if not getattr(o, "Muscle_XID", None):
                        continue
                    Muscle_Rename(o, context)
        else:
            Muscle_Rename(self.obj, context)

        return{'FINISHED'}

    def invoke(self, context, event):

        if event.alt: # Multi-rename
            self.obj = context.selected_objects
        else:
            self.obj = context.active_object
        return self.execute(context)


class OBJECT_OT_applyMuscleToMesh(bpy.types.Operator):
    """Applies selected muscles to active body"""
    bl_idname = "muscle.apply_musculature"
    bl_label = "Apply selected muscles to active body (mesh)"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
    #loop allover the selected muscles and apply the shrinkwrap to active mesh
    #create new vertex group

        clean = bpy.context.scene.Perform_Clean
        scn = context.scene
        objects = bpy.context.selected_objects
        skin = bpy.context.active_object
        oldSubView = False


        if skin == None:
            return {'FINISHED'}
        if skin.type != 'MESH':
            return {'FINISHED'}

        if clean:
            if len(objects) > 1 and not getattr(skin, "Muscle_XID", None): #TODO validate
                for ob in objects:
                    if getattr(ob, "Muscle_XID", None) and ob.type == 'MESH':

                        group = get_xmusculature(ob)

                        if group != "" and group.endswith(defaultGroupName):
                                prefix = group[:-len(defaultGroupName)]

                        for v in skin.vertex_groups.keys(): #TODO validate
                            if v.startswith(prefix):
                                skin.vertex_groups.active_index = skin.vertex_groups[v].index
                                bpy.ops.object.vertex_group_remove()

                        #preserving order of removing the drivers before removing the corresponding modifiers

                        for m in skin.modifiers.keys():
                            if m.startswith(prefix):
                                for d in skin.animation_data.drivers:
                                    if d.data_path.startswith("modifiers["+'"'+prefix):
                                        skin.driver_remove(d.data_path, -1)
                                bpy.ops.object.modifier_remove(modifier=m)


        if len(objects) > 1 and not getattr(skin, "Muscle_XID", None):
            for ob in objects:
                if getattr(ob, "Muscle_XID", None) and ob.type == 'MESH':
                    #check are muscles already attached to body

                    dupl = False
                    #if duplications allowed the loop will go implicitly
                    if not scn.Allow_Duplications:
                        m = len(skin.modifiers)
                        #find the current muscle as a target
                        for n in range(0, m):
                            if skin.modifiers[n].type == 'SHRINKWRAP' and m > 0:
                                    if skin.modifiers[n].target == ob:
                                        dupl = True
                                        break
                        if dupl:
                            continue

                    #check if mirrored muscles are already attached to body

                    NoSuffixName = ""
                    suffix = ""
                    mirrorWeights = False
                    vertexGroup = None
                    orphaned = False

                    suffix, NoSuffixName = separateSuffixAndName(ob.name, autonum = 0)
                    xname = getMirrorName(ob.name, autonum = 0)
                    for ox in scn.objects:
                        if getattr(ox, "Muscle_XID", None) and ox.type == 'MESH':
                            if suffix != "":
                                if ox.name == xname:

                                    m = len(skin.modifiers)
                                    #find the current muscle as a target
                                    for n in range(0, m):
                                        if skin.modifiers[n].type == 'SHRINKWRAP' and m > 0:
                                            if skin.modifiers[n].target == ox:

                                                #xmirrored muscle will xmirror the vertex group
                                                #check is current vertex group exist if yes remove it or overwrite it.

                                                vgname = getMirrorName(skin.modifiers[n].vertex_group, autonum = 0)
                                                mirrorWeights = True

                                                if not scn.Allow_Duplications:
                                                    for v in skin.vertex_groups.keys(): #TODO validate
                                                        if v == vgname:

                                                            skin.vertex_groups.active_index = skin.vertex_groups[vgname].index
                                                            bpy.ops.object.vertex_group_remove()

                                                            break

                                                skin.vertex_groups.active_index =  skin.vertex_groups[skin.modifiers[n].vertex_group].index
                                                bpy.ops.object.vertex_group_copy()

                                                bpy.ops.object.vertex_group_mirror()

                                                #change name to something relevant
                                                skin.vertex_groups.active.name = vgname

                                                break
                                    break
                                #dupl False and removing orphaned vertex group if none of muscles are attached to the body
                                elif ox.name == ob.name and dupl == False:

                                    orphaned = True
                                    #skin.vertex_groups.active_index = skin.vertex_groups[skin.modifiers[n].vertex_group].index
                                    #bpy.ops.object.vertex_group_remove()
                                    break


                    shrnk = bpy.context.object.modifiers.new(type='SHRINKWRAP', name= "Shrinkwrap")
                    shrnk.target = ob


                    if mirrorWeights == False:
                        vertexGroup = ob.parent.parent_bone


                    vgfound = False
                    #create vertex group and copy data from parented bone's vertex group and apply new by default
                    for v in skin.vertex_groups.keys():
                        if v == vertexGroup:

                            s = ""
                            if vertexGroup.endswith(suffix) and ob.name.endswith(suffix) and suffix != "":

                                suffix = ""
                            else:
                                #prevent repetition(doubling) of suffix and make it follow the Muscle suffix correctly
                                s,n = separateSuffixAndName(v, autonum = 0)
                                if len(s) > 1:
                                    v = v[:-len(s)]
                            skin.vertex_groups.active_index = skin.vertex_groups[v + s].index
                            bpy.ops.object.vertex_group_copy()
                            vactive = skin.vertex_groups.active

                            if scn.Allow_Duplications:

                                for v in skin.vertex_groups.keys():
                                    if v == NoSuffixName+" "+v+suffix:
                                        vactive.name = NoSuffixName+" "+v+"_dupl"+suffix
                                        break

                            n = NoSuffixName + " " + v + suffix

                            if orphaned:
                                for vg in skin.vertex_groups.keys():
                                    if vg == n:
                                        skin.vertex_groups.active_index = skin.vertex_groups[n].index
                                        bpy.ops.object.vertex_group_remove()

                                        break

                            skin.vertex_groups.active_index = vactive.index
                            vactive.name = NoSuffixName + " " + v + suffix
                            shrnk.vertex_group = n


                            #todo add blending vertex groups from mutliselected bones
                            #todo check if they are still valid (bones deleted or ranamed)
                            if len(ob.Bone_List) > 0: #TODO validate
                                weights = {}
                                for l in ob.Bone_List:

                                    if l != ob.Bone_List[-1]:
                                        weights = getWeightSet(skin, skin.vertex_groups[l.name], weights)
                                #include all new vertices to new vertex group
                                skin.vertex_groups[n].add(list(weights.keys()), 1.0, 'REPLACE')
                                setWeights(skin, skin.vertex_groups[n], weights)
                            vgfound = True
                            break
                    if mirrorWeights:
                        shrnk.vertex_group = vgname

                    elif not vgfound:

                        bpy.ops.object.vertex_group_add()
                        skin.vertex_groups[-1].name = ob.name
                        shrnk.vertex_group =  ob.name
                    shrnk.wrap_method = 'PROJECT'
                    shrnk.wrap_mode = 'OUTSIDE'
                    shrnk.cull_face = 'FRONT'
                    shrnk.show_expanded = False
                    shrnk.name = ob.name + " Shrinkwrap"

                    addDriver(skin.modifiers[shrnk.name], ob, 'show_viewport' , "Muscle_View3D")
                    addDriver(skin.modifiers[shrnk.name], ob, 'show_render' , "Muscle_Render")
                    addDriver(skin.modifiers[shrnk.name], ob, 'offset' , "Muscle_Offset")


                    #move shrinkwrap modifier above Subsurface Modifier
                    if scn.Keep_Over_Subdiv:
                        m = len(skin.modifiers)
                        #find subdiv
                        for n in range(0, m):
                            if skin.modifiers[n].type == 'SUBSURF' and m > 0:
                                oldSubView = skin.modifiers[n].show_viewport
                                skin.modifiers[n].show_viewport = False
                                mod = skin.modifiers[n]
                                break
                        #move up until over subdiv
                        while skin.modifiers[n].name != shrnk.name:
                            bpy.ops.object.modifier_move_up(modifier=shrnk.name)

                    
                    # binds microController if exists
                    if ob.Micro_Controller:

                        microSys = None
                        mic_ctrl = None
                        sname = ob.parent.name.replace("System", names["micro_SysName"])
                        mname = ob.parent.name.replace("System", names["micro_ctrlName"])
                        xname = None

                        size = ob.dimensions[0] * 0.1
                        scale = ob.Muscle_Size * size

                        for o in scn.objects.keys(): #TODO validate
                            if sname == o:
                                microSys = scn.objects[o]
                                xname = getMirrorName(microSys.name, autonum = 0)

                            if mname == o:
                                mic_ctrl = scn.objects[o]

                        rig = None
                        oldPosePosition = None
                        oldLocation = None
                        oldRotation = None
                        oldScale = None
                        oldHide = None
                        oldMode = None
                        #in case of active muscle corruption search for other armature connections in selected muscles
                        muscleSys = ob.parent
                        if muscleSys != None:
                            rigboneName = muscleSys.parent_bone

                            if rigboneName != None:
                                for b in muscleSys.pose.bones:
                                    for c in b.constraints:
                                        if c.target.type == 'EMPTY':
                                            ctrl = c.target

                                if ctrl.parent != None and rigboneName != "":
                                    if ctrl.parent.type == 'ARMATURE':
                                        rig = ctrl.parent
                                        oldPosePosition = rig.data.pose_position
                                        oldMode = rig.mode

                                        #fixing the controllers  orientation issues
                                        oldLocation = Vector(rig.location)
                                        oldRotation = Euler(rig.rotation_euler)
                                        oldScale = Vector(rig.scale)
                                        oldHide = rig.hide_viewport
                                        rig.hide_viewport = False
                                        bpy.ops.object.select_all(action='DESELECT')
                                        bpy.context.view_layer.objects.active = rig
                                        rig.select_set(True)
                                        bpy.ops.object.mode_set(mode='OBJECT')
                                        bpy.ops.object.rotation_clear(clear_delta=False)
                                        bpy.ops.object.location_clear(clear_delta=False)
                                        bpy.ops.object.scale_clear(clear_delta=False)
                                        rig.data.pose_position = 'REST'
                                        rig.select_set(False)

                        bpy.ops.object.mode_set(mode='OBJECT')
                        bpy.ops.object.select_all(action='DESELECT')
                        mic_ctrl.select_set(True)
                        bpy.context.view_layer.objects.active = mic_ctrl

                        #looking for already existing micro vertex group to mirror it
                        mirror = False
                        for v in skin.vertex_groups.keys(): #TODO validate
                            if v == xname:
                                skin.vertex_groups.active_index = skin.vertex_groups[v].index

                                if not scn.Perform_Clean:
                                    mirror = True
                                break
                        #todo set x-mirrored hook attributes
                        addMicroToSkin(ob, mic_ctrl, microSys, scale, mirror)

                        if oldPosePosition != None:
                            rig.data.pose_position = oldPosePosition
                        if oldMode != None:
                            bpy.context.view_layer.objects.active = rig
                            rig.select_set(True)
                            bpy.ops.object.mode_set(mode='OBJECT')

                        if oldLocation != None:
                            rig.location = oldLocation
                        if oldRotation != None:
                            rig.rotation_euler = oldRotation
                        if oldScale != None:
                            rig.scale = oldScale
                        if oldHide != None:
                            rig.hide_viewport = oldHide

                        bpy.ops.object.select_all(action='DESELECT')
                        bpy.context.view_layer.objects.active = skin

                    if oldSubView:
                        mod.show_viewport = oldSubView

        #add corrective_smooth if enabled
        if scn.Skin_Corrector:

            found = False
            for m in skin.modifiers:
                if m.name == "XMSL_SkinCorrector" and m.type == 'CORRECTIVE_SMOOTH':
                    found = True
                    break
            if not found:
                m = skin.modifiers.new(type='CORRECTIVE_SMOOTH', name= "XMSL_SkinCorrector")
                
            m.show_expanded = False
            
            m = len(skin.modifiers)
            for n in range(0, m):

                if skin.modifiers[n].type == 'SHRINKWRAP' and m > 0:
                    bpy.ops.object.modifier_move_to_index(modifier="XMSL_SkinCorrector", index=n)
                if skin.modifiers[n].type == 'SUBSURF' and m > 0:
                    bpy.ops.object.modifier_move_to_index(modifier="XMSL_SkinCorrector", index=n-1)

            addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'show_viewport' , "Skin_Corrector_View3D")
            addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'show_render' , "Skin_Corrector_Render")

            addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'factor' , "Skin_Smooth_Tension")
            addDriver(skin.modifiers["XMSL_SkinCorrector"], skin, 'iterations' , "Skin_Smooth_Iterations")
            skin.Skin_Corrector_View3D = 1
            skin.Skin_Corrector_Render = 1
            skin.Skin_Smooth_Tension = 0.5
            skin.Skin_Smooth_Iterations = 5

        else:
            return{'FINISHED'}
        return{'FINISHED'}



# Create custom property group for pins
class CustomProp(bpy.types.PropertyGroup):
    '''id = IntProperty() '''
    id : IntProperty()
    name : StringProperty(name ="NAME", default = "pin", update= update_pin_name)
    oldname : StringProperty()

# ui list item actions
class OBJECT_OT_MusclePinning_Uilist_actions(bpy.types.Operator):
    """Add  or remove muscle pins for selected muscle"""
    bl_idname = "muscle.list_action"
    bl_label = "List Action"
    bl_options = {'REGISTER', 'UNDO'}

    action : bpy.props.EnumProperty(
        items=(
            ('UP', "Up", ""),
            ('DOWN', "Down", ""),
            ('REMOVE', "Remove", ""),
            ('ADD', "Add", ""),
        )
    )

    def invoke(self, context, event):
        obj = None
        for o in context.selected_objects:
            if getattr(o, "Muscle_XID", None) and o.select_get() == True:
                obj = o
                break
        if obj == None:
            for o in context.scene.objects:
                if getattr(o, "Muscle_XID", None) and o.type == 'MESH' and getattr(o.custom, context.object.name, None):
                    obj = o
                    break
        
        idx = obj.custom_index

        try:
            item = obj.custom[idx]
        except IndexError:
            pass

        else:
            if self.action == 'DOWN' and idx < len(obj.custom) - 1:
                item_next = obj.custom[idx+1].name
                obj.custom_index += 1


            elif self.action == 'UP' and idx >= 1:
                item_prev = obj.custom[idx-1].name
                obj.custom_index -= 1


            elif self.action == 'REMOVE':
                pin = obj.custom[obj.custom_index]


                for o in bpy.data.objects.keys(): #TODO validate
                    if o == pin.name:
                        removeMusclePin(obj, bpy.data.objects[pin.name])
                obj.custom_index -= 1

                if obj.custom_index < 0:
                    obj.custom_index = 0

                obj.custom.remove(idx)

        if self.action == 'ADD':
            pin = addMusclePin(obj)
            item = obj.custom.add()
            item.id = len(obj.custom)
            item.name = pin.name # assign name of selected object
            item.oldname = pin.name
            obj.custom_index = (len(obj.custom)-1)

        return {"FINISHED"}


    #todo renaming
def addMusclePin(muscleObj):
    scn = bpy.context.scene
    loc, rot, scl = muscleObj.matrix_world.decompose()
 
    s,n = separateSuffixAndName(muscleObj.name, autonum = 0)
    autoaim = scn.Create_Type == 'AUTOAIM' and (bpy.context.active_bone and len(bpy.context.selected_pose_bones) == 1)
    
    if autoaim:
        loc = bpy.context.active_pose_bone.head
        rig = bpy.context.view_layer.objects.active 
        b = rig.data.bones.active
        oldMode = bpy.context.object.mode
        bpy.ops.object.mode_set(mode='OBJECT')

        pin = createController(n + "_pin" + s, loc, muscleObj.Pin_Size, 'PLAIN_AXES', False)
        bpy.ops.object.select_all(action='DESELECT')
        
        pin.select_set(True)
        b.select = True
        rig.data.bones.active = b
        bpy.context.view_layer.objects.active = rig
        
        bpy.ops.object.parent_set(type='BONE')
        bpy.ops.object.select_all(action='DESELECT')
        b.select = False
        rig.data.bones.active = None
        bpy.ops.object.mode_set(mode=oldMode)
        bpy.context.view_layer.objects.active = None
       
    else:
        pin = createController(n + "_pin" + s, loc, muscleObj.Pin_Size, 'PLAIN_AXES', False)
        
    decorateName(muscleObj , s, pin, PinOnly = True)
    
    muscleObj.select_set(True)
    bpy.context.view_layer.objects.active = muscleObj

    #add and configure hook modifier to the muscle mesh if exist (separate function)
    hook = muscleObj.modifiers.new(type='HOOK', name=pin.name)
    hook.show_expanded = False

    #todo sorting modifiers stock algorithm here for better symmetry

    while muscleObj.modifiers[0].name != hook.name:
        bpy.ops.object.modifier_move_up(modifier=hook.name)

    #include all new vertices to new vertex group
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    OldVgweight = bpy.context.scene.tool_settings.vertex_group_weight
    bpy.context.scene.tool_settings.vertex_group_weight = 0
    bpy.ops.object.vertex_group_assign_new()

    #assign weights to the muscle
    muscleObj.vertex_groups.active.name = pin.name
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.context.scene.tool_settings.vertex_group_weight = OldVgweight
    #set target
    hook.object = pin

    #add  microController's vertex group to the muscle mesh if available  (separate function) //add new vertex group to hook modifier
    hook.vertex_group = pin.name

    bpy.ops.object.hook_reset(modifier=hook.name)
    #todo add property microController

    bpy.ops.object.mode_set(mode='OBJECT')

    #add enable in render viewport drivers
    addDriver(muscleObj.modifiers[hook.name], muscleObj, 'show_viewport' , "Pinning_View3D")
    addDriver(muscleObj.modifiers[hook.name], muscleObj, 'show_render' , "Pinning_Render")

    bpy.ops.object.select_all(action='DESELECT')
    pin.select_set(True)
    # bpy.ops.collection.objects_remove_all()
    for c in pin.users_collection:
        c.objects.unlink(pin)

    pin.select_set(False)
    muscleObj.select_set(True)
    xmusclea = bpy.data.collections.get(muscleObj.name, None)

    xmusclea.objects.link(pin)
    bpy.context.view_layer.objects.active = muscleObj
    
    if autoaim:
        pin.hide_viewport = True

    muscleObj.select_set(False)
    bpy.context.view_layer.objects.active = pin
    pin.select_set(True)
    return pin


def removeMusclePin(muscleObj, pin):

    if pin.type != 'EMPTY':
        return

    scn = bpy.context.scene

    bpy.ops.object.mode_set(mode='OBJECT', toggle = False)
    #remove enable in render and viewport drivers before modifier is removed
    for m in muscleObj.modifiers.keys(): #TODO validate
        if m == pin.name:
            for d in muscleObj.animation_data.drivers:
                if d.data_path.startswith("modifiers["+'"'+pin.name+'"'):
                    muscleObj.driver_remove(d.data_path, -1)

    #remove hook modifiers from the muscle mesh
    bpy.ops.object.select_all(action='DESELECT')
    muscleObj.select_set(True)
    muscleObj.hide_viewport = False
    bpy.context.view_layer.objects.active = muscleObj
    bpy.ops.object.modifier_remove(modifier=pin.name)

    #remove microController's vertex group from the muscle mesh
    for v in muscleObj.vertex_groups.keys():
        if v == pin.name:
            muscleObj.vertex_groups.active_index = muscleObj.vertex_groups[v].index
            bpy.ops.object.vertex_group_remove()

    bpy.ops.object.select_all(action='DESELECT')
    pin.hide_viewport = False
    pin.select_set(True)
    bpy.context.view_layer.objects.active = pin
    bpy.ops.object.delete()

    #in case if it's weightpaint mode make sure muscle is in Object Mode
    bpy.ops.object.select_all(action='DESELECT')

    muscleObj.select_set(True)
    bpy.context.view_layer.objects.active = muscleObj

# print button
class OBJECT_OT_SelectToggle(bpy.types.Operator):
    bl_idname = "muscle.select_toggle"
    bl_label = "Select Toggle"
    bl_description = "Pin Select Toggle"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
       
        for o in context.scene.objects:
            if getattr(o, "Muscle_XID", None)  and o.type == 'MESH' and getattr(o.custom, context.object.name, None):
                obj = o
                break
            else:
                obj = context.object

        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):

            for i in obj.custom:
                if i.id == obj.custom_index + 1:
                    for o in bpy.context.scene.objects:
                        if o.name == i.name:
                            pin = bpy.data.objects[i.name]
                            #add unhide pin
                            pin.hide_viewport = False
                            pin.select_set(not pin.select_get())
        return{'FINISHED'}


# select button
class OBJECT_OT_ResetSelected(bpy.types.Operator):
    bl_idname = "muscle.reset_selected"
    bl_label = "Reset Selected"
    bl_description = "Resets pinpoints for all selected pins"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for o in context.scene.objects:
            if getattr(o, "Muscle_XID", None) and o.type == 'MESH' and getattr(o.custom, context.object.name, None):
                obj = o
                break
            else:
                obj = context.object
        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None):
            for m in obj.modifiers:
                if m.type == 'HOOK' and m.object.select_get() == True:
                    oldActive = context.active_object
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.object.hook_reset(modifier=m.name)
                    bpy.ops.mesh.select_all(action='DESELECT')
                    bpy.ops.object.mode_set(mode='OBJECT')
                    bpy.context.view_layer.objects.active = oldActive

        return{'FINISHED'}


class OBJECT_OT_deleteMuscle(bpy.types.Operator):
    bl_idname = "muscle.delete_muscle"
    bl_label = "Delete active Muscle"
    bl_description = "Deletes active Mesh Muscle and its components"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.object

        if obj.type == 'MESH' and getattr(obj, "Muscle_XID", None) and obj.select_get() == True:
            deleteMuscle(obj)
        return{'FINISHED'}


#calculates indices in converted mesh to muscle for microController parenting to vertex
def idxConvertedParent(indices, mesh):
    try:
        for x in indices:
            if len(mesh.idx_Converted_Parent) < 3:
                item = mesh.idx_Converted_Parent.add()
                item.id = x
   
            else:
                break
    except:
        mesh.idx_Converted_Parent.clear()


#set up coordinates and parent empty to muscle
def dp(source, targetMesh):
    idx = None

    #define the muscle type
    for v in targetMesh.vertex_groups:
        if v.name.endswith("_b"):
            idx = XMUSCLE_DATA.idxParent
            break
        if v.name.endswith("_st"):
            idx = XMUSCLE_DATA.idxStripParent
            break
        elif v.name.endswith("_s"):
            idx = XMUSCLE_DATA.idxStyleParent
            break
        elif v.name.endswith("jiggle"):
            idx = list()
            for item in targetMesh.idx_Converted_Parent:
                idx.append(item.id)
            break
    if idx == None:
        return False

    selectedVerts = SelectVertices(targetMesh, idx)
    if selectedVerts == 0:
        return False

    bpy.ops.object.select_all(action='DESELECT')
    source.select_set(True)
    targetMesh.select_set(True)
    bpy.context.view_layer.objects.active = targetMesh
    bpy.ops.object.parent_set(type='VERTEX_TRI')

    return True


def SelectVertices(meshObj, idx):
    bpy.context.view_layer.objects.active = meshObj
    mode = meshObj.mode
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode = mode)

    for i in idx:
        meshObj.data.vertices[i].select = True

    selectedVerts = [v.index for v in meshObj.data.vertices if v.select]

    return selectedVerts


def getWeightSet(ob, vgroup, weights):
    for vidx, vert in enumerate(ob.data.vertices):
        for g in vert.groups:
            if g.group == vgroup.index:
                if not vidx in weights.keys(): #TODO validate
                    weights[vidx] = g.weight
                else:
                    weights[vidx] =  g.weight + ((1 - g.weight) * weights[vidx])
                break
    return weights


def setWeights(ob, vgroup, weights):
    for vidx, vert in enumerate(ob.data.vertices):
        for g in vert.groups: #vertexgroupElement group/weight
            if g.group == vgroup.index:
                #default weight value is zero when cannot be find by index
                g.weight = weights.get(vidx, 0)
                #g.weight = weights[vidx]
                break

_selected_order_pose_bones = []
def getSelection(self):
    if not hasattr(self, 'object'):
        return
    if self.object == None:
        return
    if not self.object.type == 'ARMATURE':
        return
    bones = [b for b in self.object.pose.bones if b.select == True]

    dels = set(_selected_order_pose_bones) - set(bones)
    if len(dels):
        for b in dels:
            _selected_order_pose_bones.remove(b)

    for b in bones:
        if b not in _selected_order_pose_bones:
            _selected_order_pose_bones.append(b)
    return _selected_order_pose_bones


def getXMusculature(self):
    if not hasattr(self, 'object'):
        return
    if self.object == None:
        return
    if not self.object.type == 'MESH':
        return
    parent_collection = self.object.users_collection[0]
    parent_names = []
    parent_names.append(parent_collection.name)
    get_parent_collection_names(parent_collection, parent_names)
    prefix = ""

    for name in parent_names:
        if name != "" and name.endswith(defaultGroupName):
            prefix = name[:-len(defaultGroupName)]
            break
    group_name = prefix + defaultGroupName
    return group_name

#this part goes to the xsystem code
# bpy.types.Context.selected_order_pose_bones = property(getSelection)
bpy.types.Context.get_xmusculature = property(getXMusculature)

#startup defaults
@persistent
def startup_init(dummy):
    # bpy.context.scene.Create_Type = "MANUAL"
    bpy.context.scene.Animate_Properties = bpy.context.scene.Animate_Properties

@persistent
def xframe_change_post(scene):
    if bpy.context.scene.Animate_Properties:
        objects = scene.objects
        for obj in objects:
            if not getattr(obj, "Muscle_XID", None):
                continue
            if obj != None:
                if obj.type == 'MESH':
                    obj.View3D_Resolution = obj.View3D_Resolution
                    obj.Render_Resolution = obj.Render_Resolution
                    obj.Muscle_Type_INT = obj.Muscle_Type_INT
                    obj.Base_Length_INT = obj.Base_Length_INT
                    obj.Muscle_Shape = obj.Muscle_Shape
                    obj.Volume_INT = obj.Volume_INT
                    obj.Muscle_Size = obj.Muscle_Size
                    obj.Pin_Size = obj.Pin_Size
                    obj.Muscle_Offset = obj.Muscle_Offset
                    obj.Jiggle_Springiness = obj.Jiggle_Springiness
                    obj.Jiggle_Stiffness = obj.Jiggle_Stiffness
                    obj.Jiggle_Mass = obj.Jiggle_Mass
                    obj.Jiggle_Damping = obj.Jiggle_Damping
                    obj.Muscle_View3D = obj.Muscle_View3D
                    obj.Muscle_Render = obj.Muscle_Render
                    obj.Pinning_View3D = obj.Pinning_View3D
                    obj.Pinning_Render = obj.Pinning_Render
                    obj.Dynamics_View3D = obj.Dynamics_View3D
                    obj.Dynamics_Render = obj.Dynamics_Render
                    obj.Micro_Controller_View3D = obj.Micro_Controller_View3D
                    obj.Micro_Controller_Render = obj.Micro_Controller_Render
                    obj.Skin_Tension = obj.Skin_Tension
              

def xAnimate_Properties(self, context):
    if context.scene.Animate_Properties:
        bpy.app.handlers.frame_change_post.append(xframe_change_post)
    else:
        for h in bpy.app.handlers.frame_change_pre:
            if h.__name__ == 'xframe_change_post':
                bpy.app.handlers.frame_change_post.remove(h)


def get_ordered_selection_objects():
    if bpy.context.mode != "POSE":
        return
    tagged_objects = []
    for o in bpy.context.pose_object.pose.bones:
        order_index = o.get("selection_order", -1)
        if order_index >= 0:
            tagged_objects.append((order_index, o))
    tagged_objects = sorted(tagged_objects, key=lambda item: item[0])
    return [o for i, o in tagged_objects]

def clear_order_flag(obj):
    try:
        del obj["selection_order"]
    except KeyError:
        pass

def update_selection_order():
    if not bpy.context.selected_pose_bones:
        for o in bpy.context.pose_object.pose.bones:
            clear_order_flag(o)
        return
    selection_order = get_ordered_selection_objects()
    idx = 0
    for o in selection_order:
        if not o.select:
            selection_order.remove(o)
            clear_order_flag(o)
        else:
            o["selection_order"] = idx
            idx += 1
    for o in bpy.context.selected_pose_bones:
        if o not in selection_order:
            o["selection_order"] = len(selection_order)
            selection_order.append(o)

@persistent
def selection_change_handler(scene):
    if bpy.context.mode != "POSE":
        return
    is_selection_update = any(
        not u.is_updated_geometry
        and not u.is_updated_transform
        and not u.is_updated_shading
        for u in bpy.context.view_layer.depsgraph.updates
    )
    if is_selection_update:
        update_selection_order()


def register():
    #registering propertyGroup
    bpy.utils.register_class(ObjectSettingItem)
    bpy.types.Object.Bone_List = bpy.props.CollectionProperty(type=ObjectSettingItem)
    bpy.types.Scene.tmp_Bone_List = bpy.props.CollectionProperty(type=ObjectSettingItem)

    bpy.utils.register_class(hookSetAttributes)
    bpy.types.Object.HookSet_Attributes = bpy.props.CollectionProperty(type=hookSetAttributes)
    #todo move some props to the scene
    bpy.types.Object.Base_Length_INT =  FloatProperty(name = "Base Length", update = Base_Length_INT, default = 1.0, min=1, max=2, description="Sets the base lenght of the muscle (interacts with Volume)")
    bpy.types.Object.Muscle_Shape = FloatProperty(name = "Shape",  default = 0.0, min=-1, max=1, description="Changes the Shape of the muscle (for editing change the shape keys)")
    bpy.types.Object.Volume_INT = FloatProperty(name = "Volume", update = Volume_INT, default =1.0, min=0.1, max=100, description="Sets the volume of the muscle(interacts with base lenght)")
    #to keep up the standard props duplication is required to prevent from endless recursion
    bpy.types.Object.Base_Length =  FloatProperty(name = "Base Length", default = 1.0, min=1, max=2, description="Sets the base lenght of the muscle (interacts with Volume)")
    bpy.types.Object.Pin_Size =  FloatProperty(name = "Pins Size", default = 0.5, min=0.01, max=3, update = Muscle_Pin_Size, description="Sets the Visual size of Muscle Pins")
    bpy.types.Object.Muscle_Type = BoolProperty(name = "Muscle Type", default =0, description="Set the Muscle type to be a flexor or extensor")
    bpy.types.Object.Muscle_XID = BoolProperty(name = "Muscle XID", default =0, description="Internal system marker for the muscle object")
    bpy.types.Object.Muscle_Type_INT = BoolProperty(name = "Muscle Type", update = Muscle_Type_INT, default =0, description="Set the Muscle type to be a flexor or extensor")
    bpy.types.Object.Volume = FloatProperty(name = "Volume", default =1.0, min=0.1, max=100, description="Sets the volume of the muscle(interacts with base lenght)")
    bpy.types.Object.Muscle_Size = FloatProperty(name = "Size", update = Muscle_Size, default = 0.6, min=0.05, max=2.0, description="Controls the size of the muscle")
    bpy.types.Object.Muscle_Offset = FloatProperty(name = "Offset", default = 0.0, min=-0.5, max=0.5, step=1, description="Controls the skin offset of the muscle")
    bpy.types.Object.View3D_Resolution = IntProperty(name = "View3D Resolution",  default =1, min=0, max=3, description="Resolution of the muscle to perform in the 3D View")
    bpy.types.Object.Render_Resolution = IntProperty(name = "Render Resolution", default =1, min=0, max=3, description="Resolution of the Muscle to perform when Rendering")
    bpy.types.Object.Jiggle_Springiness = FloatProperty(name = "Springiness", default = 0.75, min=0.001, max=1, description="The higher the number the less jiggle affects the muscle")
    bpy.types.Object.Jiggle_Damping = FloatProperty(name = "Damping", default = 37.5, min=0.001, max=50, description="Controls the damping of the muscle")
    bpy.types.Object.Jiggle_Mass = FloatProperty(name = "Mass", default = 0.75, min=0.001, max=1, description="Set the mass factor of the muscle")
    bpy.types.Object.Jiggle_Stiffness = FloatProperty(name = "Stiffness", default = 0.75, min=0.001, max=1, description="Controls the stiffness of the muscle")
    bpy.types.Object.Dynamics_View3D = BoolProperty(name = "View3D", default =0, description="Enables the Jiggle effect in 3d View")
    bpy.types.Object.Dynamics_Render = BoolProperty(name = "Render",  default =0, description="Enables the Jiggle effect in Render View")
    bpy.types.Object.Muscle_View3D = BoolProperty(name = "View3D", default =1, description="Enables Muscle deformation on model mesh in Viewport")
    bpy.types.Object.Muscle_Render = BoolProperty(name = "Render", default =1, description="Enables Muscle deformation on model mesh in Render View")
    bpy.types.Object.Micro_Controller_View3D = BoolProperty(name = "View3D", default =0, description="Enables microController's skin tension in Viewport")
    bpy.types.Object.Micro_Controller_Render = BoolProperty(name = "Render", default =0, description="Enables microController's skin tension in Render View")
    bpy.types.Object.Micro_Controller = BoolProperty(name = "MicroController", update = Micro_Controller, default =0, description="Enables skin microController for the Muscle")
    bpy.types.Object.Skin_Tension = FloatProperty(name = "Skin Tension", default = 0.5, min=0.001, max=1, description="Sets the skin tension to the microController")
    bpy.types.Scene.Micro_Distance = FloatProperty(name = "Micro_Distance", default = 1.0, min=0.01, max=25, description="set MicroController distance")
    bpy.types.Scene.Micro_Influence = FloatProperty(name = "Micro_Influence", default = 0.5, min=0.01, max=1, description="set MicroController influence")
    bpy.types.Scene.Micro_Radius = FloatProperty(name = "Micro_Radius", default = 0.5, min=0.01, max=1, description="set MicroController radius")
    bpy.types.Scene.Muscle_Scale = FloatProperty(name = "Muscle_Scale", default = 1.0, min=0.02, max=2, description="Base scale factor for the muscle creation")
    bpy.types.Scene.Show_MicroControllers = BoolProperty(name = "microControllers", update = Show_MicroControllers, default = 1, description = "Show or hide skin microControllers")
    bpy.types.Scene.Create_Type = EnumProperty(name = "Create Type", default = "MANUAL", items = enum_items, update = Create_Type, description = "Choose Muscle create method")
    bpy.types.Scene.use_Affixes = BoolProperty(name = "Use affixes", default = 1, description = "Adds prefix and suffix to the Muscle System naming (recommended)")
    bpy.types.Scene.Keep_Over_Subdiv = BoolProperty(name = "Keep over Subdiv", default = 1, description = "Keeps Musculature above the Subdiv Modifier for performance")
    bpy.types.Scene.Skin_Corrector = BoolProperty(name = "Skin Smooth Corrector", default = 0, update = Skin_Corrector, description = "Adds Corrective Smooth controller to skin object")
    bpy.types.Object.Skin_Corrector_View3D = BoolProperty(name = "View3D", default =1, description="Enables Skin Corrector in Viewport")
    bpy.types.Object.Skin_Corrector_Render = BoolProperty(name = "Render", default =1, description="Enables Skin Corrector in Render View")
    bpy.types.Object.Skin_Smooth_Tension = FloatProperty(name = "Tension", default = 0.5, min=0.001, max=1, description="Controls the Corrective Smoothness of the skin")
    bpy.types.Object.Skin_Smooth_Iterations = IntProperty(name = "Amount", default = 5, min=0, max=100, description="Set the Smoothness iteration of the skin")   
    bpy.types.Scene.Allow_Duplications = BoolProperty(name = "Allow duplications", default = 0, description = "Allows muscle to duplicate on X-Mirror or apply it multiple times to the skin")
    bpy.types.Scene.Animate_Properties = BoolProperty(name = "Animate Properties", update= xAnimate_Properties, default = 0, description = "Updates keyframed muscle properties on each frame on animation playback (affects performance).")
    bpy.types.Scene.Perform_Clean = BoolProperty(name = "Perform clean", default = 0, description = "Performs cleaning before appliance if current XMusculature was previously applied")
    bpy.types.Scene.Muscle_Name = StringProperty(name = "Muscle name", default ="Muscle", description="Set the Muscle default name. New names will get auto-numeration")
    bpy.types.Scene.Prefix = StringProperty(name = "Prefix", default ="XMSL_", description="Set the name Prefix for XMuscle System elements")
    bpy.types.Scene.Suffix = StringProperty(name = "Suffix", default =".L", description="Set the name Suffix for XMuscle System elements")

    bpy.types.Object.Pinning_View3D = BoolProperty(name = "View3D", default =1, description="Enables Muscle Pinning in Viewport")
    bpy.types.Object.Pinning_Render = BoolProperty(name = "Render", default =1, description="Enables Muscle Pinning in Render View")

    bpy.utils.register_class(CustomProp) 
    bpy.types.Object.custom = bpy.props.CollectionProperty(type=CustomProp)
    bpy.types.Object.custom_index = IntProperty()
    bpy.types.Object.idx_Converted_Parent = bpy.props.CollectionProperty(type=CustomProp)

    bpy.utils.register_class(OBJECT_OT_addMuscle)
    bpy.utils.register_class(OBJECT_OT_addBasicMuscle)
    bpy.utils.register_class(OBJECT_OT_addStripMuscle)
    bpy.utils.register_class(OBJECT_OT_convertToMuscle)
    bpy.utils.register_class(OBJECT_OT_deleteMuscle)
    bpy.utils.register_class(OBJECT_OT_setMaterials)
    bpy.utils.register_class(OBJECT_OT_resetProps)
    bpy.utils.register_class(OBJECT_OT_smartUpdate)
    bpy.utils.register_class(OBJECT_OT_wireframe)
    bpy.utils.register_class(OBJECT_OT_xray)
    bpy.utils.register_class(OBJECT_OT_dispNames)
    bpy.utils.register_class(OBJECT_OT_visibility)
    bpy.utils.register_class(OBJECT_OT_pins_visibility)
    bpy.utils.register_class(OBJECT_OT_selectMuscle)
    bpy.utils.register_class(OBJECT_OT_isolateMuscle)
    bpy.utils.register_class(OBJECT_OT_selectMusculature)
    bpy.utils.register_class(OBJECT_OT_setMicroController)
    bpy.utils.register_class(OBJECT_OT_xMirror)
    bpy.utils.register_class(OBJECT_OT_Muscle_Rename)
    bpy.utils.register_class(OBJECT_OT_applyMuscleToMesh)
    bpy.utils.register_class(OBJECT_OT_MusclePinning_Uilist_actions)
    bpy.utils.register_class(OBJECT_OT_SelectToggle)
    bpy.utils.register_class(OBJECT_OT_ResetSelected)

    #startup defaults
    bpy.app.handlers.load_post.append(startup_init)    
    bpy.app.handlers.depsgraph_update_post.append(selection_change_handler)



def unregister():
    del bpy.types.Object.custom
    del bpy.types.Object.custom_index

    bpy.utils.unregister_class(ObjectSettingItem)
    bpy.utils.unregister_class(hookSetAttributes)
    bpy.utils.unregister_class(OBJECT_OT_ResetSelected)
    bpy.utils.unregister_class(OBJECT_OT_SelectToggle)
    bpy.utils.unregister_class(OBJECT_OT_MusclePinning_Uilist_actions)
    bpy.utils.unregister_class(OBJECT_OT_applyMuscleToMesh)
    bpy.utils.unregister_class(OBJECT_OT_Muscle_Rename)
    bpy.utils.unregister_class(OBJECT_OT_xMirror)
    bpy.utils.unregister_class(OBJECT_OT_setMicroController)
    bpy.utils.unregister_class(OBJECT_OT_selectMusculature)
    bpy.utils.unregister_class(OBJECT_OT_isolateMuscle)
    bpy.utils.unregister_class(OBJECT_OT_selectMuscle)
    bpy.utils.unregister_class(OBJECT_OT_wireframe)
    bpy.utils.unregister_class(OBJECT_OT_visibility)
    bpy.utils.unregister_class(OBJECT_OT_pins_visibility)
    bpy.utils.unregister_class(OBJECT_OT_xray)
    bpy.utils.unregister_class(OBJECT_OT_dispNames)
    bpy.utils.unregister_class(OBJECT_OT_smartUpdate)
    bpy.utils.unregister_class(OBJECT_OT_resetProps)
    bpy.utils.unregister_class(OBJECT_OT_setMaterials)
    bpy.utils.unregister_class(OBJECT_OT_deleteMuscle)
    bpy.utils.unregister_class(OBJECT_OT_convertToMuscle)
    bpy.utils.unregister_class(OBJECT_OT_addStripMuscle)
    bpy.utils.unregister_class(OBJECT_OT_addBasicMuscle)
    bpy.utils.unregister_class(OBJECT_OT_addMuscle)
    bpy.utils.unregister_class(CustomProp)

