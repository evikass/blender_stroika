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

bl_info = {
    "name":        "XMuscle System",
    "description": "Muscle and Tissue System for Blender",
    "author":      "Albert Makac, k44dev",
    "version":     (4, 1, 9),
    "blender":     (2, 90, 0),
    "location":    "View 3D",
    "category":    "Object"
    }


import bpy
reload_flag = "bpy" in locals()

module_names = ("XMUSCLE_PANEL", "XMUSCLE_SYSTEM",)

from importlib import import_module, reload as reload_module

modules = {}
for name in module_names:

    mod = import_module("%s.%s" % (__package__, name))
    if reload_flag:
        reload_module(mod)
    modules[name] = mod

def register():
    for name, module in modules.items():
        module.register()

def unregister():
    for name, module in modules.items():
        module.unregister()

if __name__ == "__main__":
    for f in bpy.app.handlers.depsgraph_update_post:
        if f.__name__ == "selection_change_handler":
            bpy.app.handlers.depsgraph_update_post.remove(f)
  
    register()
