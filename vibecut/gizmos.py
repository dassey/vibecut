import bpy
import gpu
import blf
import bpy_extras.view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector, Quaternion

from .geometry import (
    _state, HOVER, LOCKED,
    _rebuild, _flush_cross,
)

def _screen_pos(context, world_pos):
    return bpy_extras.view3d_utils.location_3d_to_region_2d(
        context.region, context.region_data, world_pos)

def _world_at_mouse(context, mx, my, depth_ref):
    return bpy_extras.view3d_utils.region_2d_to_location_3d(
        context.region, context.region_data, (mx, my), depth_ref)

def _draw_line(sh, p0, p1, col, lw=2.0):
    gpu.state.line_width_set(lw)
    sh.uniform_float("color", col)
    batch_for_shader(sh, 'LINES', {"pos": [p0.to_tuple(), p1.to_tuple()]}).draw(sh)

def _draw_dot(sh, pos, col, size=8.0):
    gpu.state.point_size_set(size)
    sh.uniform_float("color", col)
    batch_for_shader(sh, 'POINTS', {"pos": [pos.to_tuple()]}).draw(sh)


# ══════════════════════════════════════════════════
#  Gizmo 1: Normal Slide (slide plane along limb)
# ══════════════════════════════════════════════════

class RC_GizmoNormal(bpy.types.Gizmo):
    bl_idname = "VIEW3D_GT_rc_normal"

    def _tip(self):
        s = _state
        if s["plane_co"] is None or s["plane_no"] is None:
            return None, None
        co = s["plane_co"]
        no = s["plane_no"].normalized()
        return co, co + no * s["radius"] * 0.65

    def draw(self, context):
        co, tip = self._tip()
        if co is None:
            return
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        sh.bind()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        col = (*self.color_highlight, self.alpha) if self.is_highlight else (*self.color, self.alpha)
        _draw_line(sh, co, tip, col, 3.0)
        _draw_dot(sh, tip, col, 12.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def test_select(self, context, location):
        co, tip = self._tip()
        if tip is None:
            return -1
        sp = _screen_pos(context, tip)
        if sp is None:
            return -1
        return 0 if (Vector(location) - sp).length < 18 else -1

    def invoke(self, context, event):
        self._start_co = _state["plane_co"].copy()
        self._start_mx = event.mouse_region_x
        self._start_my = event.mouse_region_y
        return {'RUNNING_MODAL'}

    def modal(self, context, event, tweak):
        no      = _state["plane_no"].normalized()
        start3d = _world_at_mouse(context, self._start_mx, self._start_my, self._start_co)
        curr3d  = _world_at_mouse(context, event.mouse_region_x, event.mouse_region_y, self._start_co)
        if start3d and curr3d:
            delta = (curr3d - start3d).dot(no)
            _state["plane_co"] = self._start_co + no * delta
            _rebuild(defer=True)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        if cancel and hasattr(self, '_start_co'):
            _state["plane_co"] = self._start_co
            _rebuild()
            _flush_cross()
        context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  Gizmo 2: Lateral Slide
# ══════════════════════════════════════════════════

class RC_GizmoLateral(bpy.types.Gizmo):
    bl_idname = "VIEW3D_GT_rc_lateral"

    def _tip(self, context):
        s = _state
        if s["plane_co"] is None or s["plane_no"] is None:
            return None, None
        no  = s["plane_no"].normalized()
        rv3d = context.region_data
        cam_right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized()
        lat = (cam_right - cam_right.dot(no) * no).normalized()
        if lat.length < 0.0001:
            lat = (rv3d.view_rotation @ Vector((0, 1, 0))).normalized()
            lat = (lat - lat.dot(no) * no).normalized()
        co  = s["plane_co"]
        tip = co + lat * s["radius"] * 0.65
        return co, tip, lat

    def draw(self, context):
        res = self._tip(context)
        if res[0] is None:
            return
        co, tip, _ = res
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        sh.bind()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        col = (*self.color_highlight, self.alpha) if self.is_highlight else (*self.color, self.alpha)
        _draw_line(sh, co, tip, col, 3.0)
        _draw_dot(sh, tip, col, 12.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def test_select(self, context, location):
        res = self._tip(context)
        if res[0] is None:
            return -1
        sp = _screen_pos(context, res[1])
        if sp is None:
            return -1
        return 0 if (Vector(location) - sp).length < 18 else -1

    def invoke(self, context, event):
        self._start_co = _state["plane_co"].copy()
        self._start_mx = event.mouse_region_x
        self._start_my = event.mouse_region_y
        res = self._tip(context)
        self._lat = res[2] if res[0] is not None else Vector((1, 0, 0))
        return {'RUNNING_MODAL'}

    def modal(self, context, event, tweak):
        start3d = _world_at_mouse(context, self._start_mx, self._start_my, self._start_co)
        curr3d  = _world_at_mouse(context, event.mouse_region_x, event.mouse_region_y, self._start_co)
        if start3d and curr3d:
            delta = (curr3d - start3d).dot(self._lat)
            _state["plane_co"] = self._start_co + self._lat * delta
            _rebuild(defer=True)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        if cancel and hasattr(self, '_start_co'):
            _state["plane_co"] = self._start_co
            _rebuild()
            _flush_cross()
        context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  Gizmo 3: Tilt (rotate plane around view-forward)
# ══════════════════════════════════════════════════

class RC_GizmoTilt(bpy.types.Gizmo):
    bl_idname = "VIEW3D_GT_rc_tilt"

    def _arc_tip(self, context):
        s = _state
        if s["plane_co"] is None or s["plane_no"] is None:
            return None
        rv3d = context.region_data
        view_up = (rv3d.view_rotation @ Vector((0, 1, 0))).normalized()
        no = s["plane_no"].normalized()
        lat = (view_up - view_up.dot(no) * no).normalized()
        if lat.length < 0.0001:
            lat = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized()
        return s["plane_co"] + lat * s["radius"] * 1.05

    def draw(self, context):
        tip = self._arc_tip(context)
        if tip is None:
            return
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        sh.bind()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        col = (*self.color_highlight, self.alpha) if self.is_highlight else (*self.color, self.alpha)
        _draw_dot(sh, tip, col, 14.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def test_select(self, context, location):
        tip = self._arc_tip(context)
        if tip is None:
            return -1
        sp = _screen_pos(context, tip)
        if sp is None:
            return -1
        return 0 if (Vector(location) - sp).length < 20 else -1

    def invoke(self, context, event):
        self._start_no = _state["plane_no"].copy()
        self._start_mx = event.mouse_region_x
        self._start_my = event.mouse_region_y
        self._last_angle = 0.0
        return {'RUNNING_MODAL'}

    def modal(self, context, event, tweak):
        s  = _state
        rv3d = context.region_data
        cen_sp = _screen_pos(context, s["plane_co"])
        if cen_sp is None:
            return {'RUNNING_MODAL'}
        v_start = Vector((self._start_mx, self._start_my)) - cen_sp
        v_curr  = Vector((event.mouse_region_x, event.mouse_region_y)) - cen_sp
        if v_start.length > 1 and v_curr.length > 1:
            angle = v_start.angle_signed(v_curr)
            d_angle = angle - self._last_angle
            self._last_angle = angle
            axis = (rv3d.view_rotation @ Vector((0, 0, -1))).normalized()
            q = Quaternion(axis, d_angle)
            _state["plane_no"] = (q @ _state["plane_no"]).normalized()
            _rebuild(defer=True)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        if cancel and hasattr(self, '_start_no'):
            _state["plane_no"] = self._start_no
            _rebuild()
            _flush_cross()
        context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  Gizmo 4: Resize
# ══════════════════════════════════════════════════

class RC_GizmoResize(bpy.types.Gizmo):
    bl_idname = "VIEW3D_GT_rc_resize"

    def _dot_pos(self, context):
        s = _state
        if s["plane_co"] is None or s["plane_no"] is None:
            return None
        rv3d = context.region_data
        cam_right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized()
        no  = s["plane_no"].normalized()
        lat = (cam_right - cam_right.dot(no) * no).normalized()
        if lat.length < 0.0001:
            lat = (rv3d.view_rotation @ Vector((0, -1, 0))).normalized()
        return s["plane_co"] + lat * s["radius"]

    def draw(self, context):
        pos = self._dot_pos(context)
        if pos is None:
            return
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        sh.bind()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        col = (*self.color_highlight, self.alpha) if self.is_highlight else (*self.color, self.alpha)
        _draw_dot(sh, pos, col, 16.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def test_select(self, context, location):
        pos = self._dot_pos(context)
        if pos is None:
            return -1
        sp = _screen_pos(context, pos)
        if sp is None:
            return -1
        return 0 if (Vector(location) - sp).length < 20 else -1

    def invoke(self, context, event):
        self._start_r  = _state["radius"]
        self._start_mx = event.mouse_region_x
        self._start_my = event.mouse_region_y
        cen_sp = _screen_pos(context, _state["plane_co"])
        self._start_dist = (Vector((self._start_mx, self._start_my)) - cen_sp).length \
            if cen_sp else 1.0
        return {'RUNNING_MODAL'}

    def modal(self, context, event, tweak):
        cen_sp = _screen_pos(context, _state["plane_co"])
        if cen_sp and self._start_dist > 0.5:
            d = (Vector((event.mouse_region_x, event.mouse_region_y)) - cen_sp).length
            _state["radius"] = max(0.0001, min(50000.0, self._start_r * (d / self._start_dist)))
            _rebuild(defer=True)
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        if cancel and hasattr(self, '_start_r'):
            _state["radius"] = self._start_r
            _rebuild()
            _flush_cross()
        context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  GizmoGroup — manages all 4 handles
# ══════════════════════════════════════════════════

class RC_GGT_cut_plane(bpy.types.GizmoGroup):
    bl_idname     = "RC_GGT_cut_plane"
    bl_label      = "Ring Cutter Handles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options    = {'3D', 'PERSISTENT', 'SHOW_MODAL_ALL'}

    @classmethod
    def poll(cls, context):
        return _state.get("phase") == LOCKED and _state.get("active", False)

    def setup(self, context):
        g = self.gizmos
        self.gz_normal  = g.new(RC_GizmoNormal.bl_idname)
        self.gz_lateral = g.new(RC_GizmoLateral.bl_idname)
        self.gz_tilt    = g.new(RC_GizmoTilt.bl_idname)
        self.gz_resize  = g.new(RC_GizmoResize.bl_idname)

        self.gz_normal.color         = 0.3, 1.0, 0.5
        self.gz_normal.color_highlight = 0.6, 1.0, 0.7
        self.gz_normal.alpha         = 0.9
        self.gz_normal.alpha_highlight = 1.0



        self.gz_lateral.color          = 1.0, 0.5, 0.2
        self.gz_lateral.color_highlight = 1.0, 0.7, 0.4
        self.gz_lateral.alpha          = 0.9
        self.gz_lateral.alpha_highlight = 1.0



        self.gz_tilt.color          = 0.2, 0.8, 1.0
        self.gz_tilt.color_highlight = 0.5, 0.95, 1.0
        self.gz_tilt.alpha          = 0.9
        self.gz_tilt.alpha_highlight = 1.0



        self.gz_resize.color          = 1.0, 0.9, 0.2
        self.gz_resize.color_highlight = 1.0, 1.0, 0.5
        self.gz_resize.alpha          = 0.9
        self.gz_resize.alpha_highlight = 1.0



    def draw_prepare(self, context):
        if context.area:
            context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  Post-cut gizmo: drag limb along cut normal
# ══════════════════════════════════════════════════

class RC_GizmoPostCutPull(bpy.types.Gizmo):
    bl_idname = "VIEW3D_GT_rc_post_pull"

    def _get_obj(self):
        name = getattr(self, '_target_prop', '')
        obj_name = bpy.context.scene.get(name, '')
        if obj_name:
            return bpy.data.objects.get(obj_name)
        return None

    def _get_normal(self):
        no_raw = bpy.context.scene.get("rc_plane_no")
        if no_raw:
            return Vector(no_raw).normalized()
        return Vector((0, 0, 1))

    def _tip(self):
        obj = self._get_obj()
        if obj is None:
            return None, None
        no = self._get_normal() * getattr(self, '_direction', 1)
        co = obj.matrix_world.translation.copy()
        tip = co + no * 0.04
        return co, tip

    def draw(self, context):
        co, tip = self._tip()
        if co is None:
            return
        sh = gpu.shader.from_builtin('UNIFORM_COLOR')
        sh.bind()
        gpu.state.blend_set('ALPHA')
        gpu.state.depth_test_set('NONE')
        col = (*self.color_highlight, self.alpha) if self.is_highlight else (*self.color, self.alpha)
        _draw_line(sh, co, tip, col, 4.0)
        _draw_dot(sh, tip, col, 14.0)
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('NONE')

    def test_select(self, context, location):
        _, tip = self._tip()
        if tip is None:
            return -1
        sp = _screen_pos(context, tip)
        if sp is None:
            return -1
        return 0 if (Vector(location) - sp).length < 24 else -1

    def invoke(self, context, event):
        obj = self._get_obj()
        if obj:
            self._start_loc = obj.location.copy()
        self._start_mx = event.mouse_region_x
        self._start_my = event.mouse_region_y
        return {'RUNNING_MODAL'}

    def modal(self, context, event, tweak):
        obj = self._get_obj()
        if obj is None:
            return {'RUNNING_MODAL'}
        no = self._get_normal() * getattr(self, '_direction', 1)
        start3d = _world_at_mouse(context, self._start_mx, self._start_my, self._start_loc)
        curr3d  = _world_at_mouse(context, event.mouse_region_x, event.mouse_region_y, self._start_loc)
        if start3d and curr3d:
            delta = (curr3d - start3d).dot(no)
            obj.location = self._start_loc + no * delta
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        if cancel and hasattr(self, '_start_loc'):
            obj = self._get_obj()
            if obj:
                obj.location = self._start_loc
        context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  Post-cut GizmoGroup
# ══════════════════════════════════════════════════

class RC_GGT_post_cut(bpy.types.GizmoGroup):
    bl_idname      = "RC_GGT_post_cut"
    bl_label       = "Post-Cut Handles"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options     = {'3D', 'PERSISTENT', 'SHOW_MODAL_ALL'}

    @classmethod
    def poll(cls, context):
        scene = context.scene
        body_name = scene.get("rc_body", "")
        limb_name = scene.get("rc_limb", "")
        if not body_name or not limb_name:
            return False
        body = bpy.data.objects.get(body_name)
        limb = bpy.data.objects.get(limb_name)
        if not body or not limb:
            return False
        if _state.get("active", False):
            return False
        return True

    def setup(self, context):
        g = self.gizmos
        self.gz_body = g.new(RC_GizmoPostCutPull.bl_idname)
        self.gz_body._target_prop = "rc_body"
        self.gz_body._direction = 1
        self.gz_body.color            = 0.3, 0.6, 1.0
        self.gz_body.color_highlight  = 0.5, 0.8, 1.0
        self.gz_body.alpha            = 0.9
        self.gz_body.alpha_highlight  = 1.0



        self.gz_limb = g.new(RC_GizmoPostCutPull.bl_idname)
        self.gz_limb._target_prop = "rc_limb"
        self.gz_limb._direction = -1
        self.gz_limb.color            = 1.0, 0.4, 0.3
        self.gz_limb.color_highlight  = 1.0, 0.6, 0.5
        self.gz_limb.alpha            = 0.9
        self.gz_limb.alpha_highlight  = 1.0



    def draw_prepare(self, context):
        if context.area:
            context.area.tag_redraw()


# ══════════════════════════════════════════════════
#  Registration
# ══════════════════════════════════════════════════

classes = (
    RC_GizmoNormal,
    RC_GizmoLateral,
    RC_GizmoTilt,
    RC_GizmoResize,
    RC_GGT_cut_plane,
    RC_GizmoPostCutPull,
    RC_GGT_post_cut,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
