import bpy
import gpu
import blf
import bpy_extras.view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Quaternion

from .geometry import (
    _state, HOVER, LOCKED,
    _pca_axis, _rebuild, _do_tilt, _nudge_plane,
    _ring_verts, _lock_phase_compute, _flush_cross,
)
from .geometry import _execute_cut

_h3d = _h2d = None


def _draw_3d():
    if not _state["active"]:
        return
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    sh.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')

    if _state["ring"]:
        col = (0.2, 1.0, 0.55, 1.0) if _state["phase"] == LOCKED else (0.0, 0.85, 1.0, 0.85)
        gpu.state.line_width_set(2.5)
        sh.uniform_float("color", col)
        batch_for_shader(sh, 'LINE_LOOP', {"pos": _state["ring"]}).draw(sh)

    if _state["cross"]:
        gpu.state.line_width_set(3.5)
        sh.uniform_float("color", (1.0, 0.55, 0.05, 1.0))
        batch_for_shader(sh, 'LINE_STRIP', {"pos": _state["cross"]}).draw(sh)

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


def _draw_2d():
    if not _state["active"]:
        return
    ctx = bpy.context
    if not ctx or not ctx.region:
        return
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    sh.bind()
    gpu.state.blend_set('ALPHA')

    locked = _state["phase"] == LOCKED
    label  = "VibeCut — LOCKED  (Enter=cut  Esc=back)" if locked \
             else "VibeCut — HOVER  (click to lock)"
    col    = (0.3, 1.0, 0.5, 1.0) if locked else (0.0, 0.85, 1.0, 1.0)
    blf.size(0, 12)
    blf.color(0, *col)
    blf.position(0, 14, ctx.region.height - 22, 0)
    blf.draw(0, label)
    gpu.state.blend_set('NONE')


class RC_OT_snap(bpy.types.Operator):
    bl_idname = "view3d.rc_snap"
    bl_label  = "RC Snap Normal"
    bl_description = "Snap the cut plane to an axis, the view, or the detected limb axis"
    bl_options = {'INTERNAL'}

    axis: bpy.props.StringProperty()

    def execute(self, context):
        if _state["plane_co"] is None:
            return {'CANCELLED'}
        rv3d = context.region_data
        if self.axis == 'X':
            _state["plane_no"] = Vector((1, 0, 0))
        elif self.axis == 'Y':
            _state["plane_no"] = Vector((0, 1, 0))
        elif self.axis == 'Z':
            _state["plane_no"] = Vector((0, 0, 1))
        elif self.axis == 'VIEW':
            _state["plane_no"] = (rv3d.view_rotation @ Vector((0, 0, -1))).normalized()
        elif self.axis == 'AUTO':
            obj  = _state["obj"]
            co   = _state["plane_co"]
            axis = _pca_axis(obj, co, _state["radius"]) if obj else None
            _state["plane_no"] = axis if axis else \
                (rv3d.view_rotation @ Vector((0, 0, -1))).normalized()
        _rebuild()
        if context.area:
            context.area.tag_redraw()
        return {'FINISHED'}


class RC_OT_hover(bpy.types.Operator):
    bl_idname  = "view3d.rc_hover"
    bl_label   = "VibeCut"
    bl_description = ("Move the mouse to aim the cut ring, then CLICK to lock it. "
                      "The orange outline shows exactly what will be cut. "
                      "Wheel = tilt, Shift/Alt+wheel = slide, Ctrl+wheel = resize. "
                      "Enter to cut, Esc to unlock")
    bl_options = {'REGISTER', 'UNDO'}

    def _cast(self, ctx, mx, my):
        region = ctx.region
        rv3d   = ctx.region_data
        orig   = bpy_extras.view3d_utils.region_2d_to_origin_3d(region, rv3d, (mx, my))
        dire   = bpy_extras.view3d_utils.region_2d_to_vector_3d(region, rv3d, (mx, my))
        return ctx.scene.ray_cast(ctx.view_layer.depsgraph, orig, dire)

    def _update_hover(self, ctx, event):
        ok, loc, _n, _fi, obj, _m = self._cast(
            ctx, event.mouse_region_x, event.mouse_region_y)
        if not (ok and obj and obj.type == 'MESH'):
            _state["active"] = False
            return

        rv3d = ctx.region_data
        view_normal = (rv3d.view_rotation @ Vector((0, 0, -1))).normalized()
        _state.update(active=True, obj=obj,
                      plane_co=loc.copy(),
                      plane_no=view_normal)
        _state["limb_axis"] = None
        _state["cross"] = []
        _state["ring"]  = _ring_verts(_state["plane_co"], _state["plane_no"],
                                       _state["radius"])

    def _finish(self, ctx):
        global _h3d, _h2d
        _state.update(active=False, phase=HOVER)
        for h in (_h3d, _h2d):
            if h:
                try:
                    bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
                except Exception:
                    pass
        _h3d = _h2d = None
        if ctx.area:
            ctx.area.tag_redraw()

    @staticmethod
    def _in_viewport(ctx, event):
        for region in ctx.area.regions:
            if region.type == 'WINDOW':
                mx, my = event.mouse_region_x, event.mouse_region_y
                return 0 <= mx <= region.width and 0 <= my <= region.height
        return True

    def _handle_hover(self, ctx, event):
        if event.type == 'MOUSEMOVE':
            self._update_hover(ctx, event)
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            d = 1 if event.type == 'WHEELUPMOUSE' else -1
            _state["radius"] = max(0.0001, min(50000.0, _state["radius"] * (1.12 ** d)))
            _rebuild()
        elif event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if _state["active"] and self._in_viewport(ctx, event):
                _lock_phase_compute()
                _state["phase"] = LOCKED
                ctx.area.tag_redraw()
                return {'RUNNING_MODAL'}
            else:
                return {'PASS_THROUGH'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            self._finish(ctx)
            return {'CANCELLED'}
        ctx.area.tag_redraw()
        return {'PASS_THROUGH'}

    def _handle_locked(self, ctx, event):
        _flush_cross()

        if event.type == 'MIDDLEMOUSE':
            return {'PASS_THROUGH'}
        if (event.type.startswith('NUMPAD') and
                event.type not in {'NUMPAD_ENTER', 'NUMPAD_PERIOD'}):
            return {'PASS_THROUGH'}

        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            d = 1 if event.type == 'WHEELUPMOUSE' else -1
            rv3d = ctx.region_data
            if event.ctrl:
                _state["radius"] = max(0.0001, min(50000.0, _state["radius"] * (1.12 ** d)))
                _rebuild(defer=True)
            elif event.alt:
                _nudge_plane(rv3d, 'NORMAL', d, defer=True)
            elif event.shift:
                _nudge_plane(rv3d, 'LATERAL', d, defer=True)
            else:
                _do_tilt(rv3d, d, defer=True)
            ctx.area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.value == 'PRESS' and _state["plane_co"] is not None:
            rv3d = ctx.region_data
            if event.type in {'X', 'Y', 'Z', 'V', 'N'}:
                axis_map = {
                    'X': 'X', 'Y': 'Y', 'Z': 'Z', 'V': 'VIEW', 'N': 'AUTO'}
                bpy.ops.view3d.rc_snap('INVOKE_DEFAULT', axis=axis_map[event.type])
                return {'RUNNING_MODAL'}

        if event.type in {'RET', 'NUMPAD_ENTER', 'SPACE'} and event.value == 'PRESS':
            _execute_cut(ctx)
            self._finish(ctx)
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
            _state["phase"] = HOVER
            _state["cross"]       = []
            _state["cross_dirty"] = False
            ctx.area.tag_redraw()
            return {'RUNNING_MODAL'}

        ctx.area.tag_redraw()
        return {'PASS_THROUGH'}

    def modal(self, ctx, event):
        if ctx.area is None:
            self._finish(ctx)
            return {'CANCELLED'}

        from .geometry import _is_valid
        if _state.get("obj") is not None and not _is_valid(_state["obj"]):
            _state["active"] = False
            _state["obj"]    = None
            self._finish(ctx)
            return {'FINISHED'}

        if _state.pop("_panel_cut", False):
            self._finish(ctx)
            return {'FINISHED'}

        scene = ctx.scene
        _state["opt_fill"] = scene.rc_opt_fill
        _state["opt_neg"]  = scene.rc_opt_neg
        _state["opt_pos"]  = scene.rc_opt_pos
        _state["opt_sep"]  = scene.rc_opt_sep
        if _state["phase"] == HOVER:
            return self._handle_hover(ctx, event)
        return self._handle_locked(ctx, event)


    def invoke(self, ctx, _event):
        global _h3d, _h2d
        if ctx.area.type != 'VIEW_3D':
            self.report({'WARNING'}, "Run from the 3D Viewport")
            return {'CANCELLED'}
        if _h3d is not None or _h2d is not None:
            self.report({'INFO'}, "Ring Cutter is already active")
            return {'CANCELLED'}
        obj = ctx.active_object
        if obj and obj.type == 'MESH':
            _state["radius"] = max(obj.dimensions) * 0.25
        _state.update(phase=HOVER, active=False)
        _h3d = bpy.types.SpaceView3D.draw_handler_add(_draw_3d, (), 'WINDOW', 'POST_VIEW')
        _h2d = bpy.types.SpaceView3D.draw_handler_add(_draw_2d, (), 'WINDOW', 'POST_PIXEL')
        ctx.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


classes = (
    RC_OT_snap,
    RC_OT_hover,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
