bl_info = {
    "name":        "VibeCut - Articulated Print Prep",
    "author":      "DIY",
    "version":     (1, 0, 0, 38),
    "blender":     (5, 1, 0),
    "location":    "View3D > N-Panel > VibeCut",
    "description": "Prepare organic meshes for articulated 3D printing with ball and socket joints",
    "category":    "Mesh",
}

import bpy
from . import (geometry, placement_tool, panels, joints, gizmos,
               detection, geodesic, supports, mesh_tools, auto_detect,
               tools, click_collapse, hinge)

def _reload_modules():
    import importlib
    importlib.reload(geometry)
    importlib.reload(joints)
    importlib.reload(supports)
    importlib.reload(mesh_tools)
    importlib.reload(placement_tool)
    importlib.reload(panels)
    importlib.reload(gizmos)
    importlib.reload(detection)
    importlib.reload(geodesic)
    importlib.reload(auto_detect)
    importlib.reload(tools)
    importlib.reload(click_collapse)
    importlib.reload(hinge)

if "bpy" in dir():
    _reload_modules()

_HANDLER_NAME = "_rc_depsgraph_invalidate"


def _scrub_depsgraph_handler():
    hs = bpy.app.handlers.depsgraph_update_post
    for h in [h for h in hs if getattr(h, "__name__", None) == _HANDLER_NAME]:
        try:
            hs.remove(h)
        except ValueError:
            pass


def register():
    _scrub_depsgraph_handler()
    bpy.app.handlers.depsgraph_update_post.append(geometry._rc_depsgraph_invalidate)
    geometry.register_props()
    joints.register_props()
    supports.register_props()
    mesh_tools.register_props()
    detection.register_props()
    tools.register_props()
    tools.register()
    hinge.register_props()
    hinge.register()
    placement_tool.register()
    gizmos.register()
    panels.register()
    supports.register()
    mesh_tools.register()
    auto_detect.register()
    click_collapse.register()

def unregister():
    _scrub_depsgraph_handler()
    click_collapse.unregister()
    auto_detect.unregister()
    mesh_tools.unregister()
    supports.unregister()
    panels.unregister()
    gizmos.unregister()
    placement_tool.unregister()
    hinge.unregister()
    hinge.unregister_props()
    tools.unregister()
    tools.unregister_props()
    detection.unregister_props()
    mesh_tools.unregister_props()
    supports.unregister_props()
    joints.unregister_props()
    geometry.unregister_props()

if __name__ == "__main__":
    register()
