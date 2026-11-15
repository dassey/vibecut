import bpy
import bmesh
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def nearest_neighbor(v, mat3=None):
    best, best_d = None, float("inf")
    for e in v.link_edges:
        o = e.other_vert(v)
        if o.hide:
            continue
        d = o.co - v.co
        if mat3 is not None:
            d = mat3 @ d
        if d.length_squared < best_d:
            best, best_d = o, d.length_squared
    return best


def collapse_vertex(bm, v, mat3=None):
    target = nearest_neighbor(v, mat3)
    if target is None:
        return None
    bmesh.ops.weld_verts(bm, targetmap={v: target})
    bm.select_flush_mode()
    return target


def project_verts(coords, mat, width, height):
    h = np.c_[coords, np.ones(len(coords))] @ np.array(mat).T
    w = h[:, 3]
    front = w > 1e-9
    w = np.where(front, w, 1.0)
    px = np.empty((len(coords), 2))
    px[:, 0] = (1.0 + h[:, 0] / w) * 0.5 * width
    px[:, 1] = (1.0 + h[:, 1] / w) * 0.5 * height
    return px, front


def pick_vertex(px, front, usable, mouse, radius, visible=None,
                max_tests=32):
    d2 = ((px - np.asarray(mouse, float)) ** 2).sum(axis=1)
    cand = np.flatnonzero(front & usable & (d2 <= radius * radius))
    if not len(cand):
        return -1
    for i in cand[np.argsort(d2[cand])[:max_tests]]:
        if visible is None or visible(int(i)):
            return int(i)
    return -1


def make_visibility_test(bvh, coords, eye=None, view_dir=None, far=1.0):
    def visible(i):
        p = Vector(coords[i])
        o = eye if eye is not None else p - view_dir * far
        d = p - o
        dist = d.length
        if dist < 1e-12:
            return True
        d /= dist
        hit = bvh.ray_cast(o, d, dist)
        return hit[0] is None or hit[3] >= dist * 0.999
    return visible


_hover = {"pair": None}
_draw_handle = None
_NAV_EVENTS = {
    'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE', 'MIDDLEMOUSE', 'WHEELUPMOUSE',
    'WHEELDOWNMOUSE', 'TRACKPADPAN', 'TRACKPADZOOM', 'MOUSEROTATE',
    'MOUSESMARTZOOM', 'TIMER', 'TIMER_REPORT', 'TIMERREGION',
}


def _draw():
    pair = _hover["pair"]
    if not pair:
        return
    import gpu
    from gpu_extras.batch import batch_for_shader
    gone, stays = pair
    sh = gpu.shader.from_builtin('UNIFORM_COLOR')
    sh.bind()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    if stays is not None:
        gpu.state.line_width_set(3.0)
        sh.uniform_float("color", (1.0, 0.85, 0.1, 0.9))
        batch_for_shader(sh, 'LINES', {"pos": [gone, stays]}).draw(sh)
        gpu.state.point_size_set(11.0)
        sh.uniform_float("color", (0.15, 1.0, 0.3, 1.0))
        batch_for_shader(sh, 'POINTS', {"pos": [stays]}).draw(sh)
    gpu.state.point_size_set(13.0)
    sh.uniform_float("color", (1.0, 0.4, 0.05, 1.0) if stays is not None
                     else (0.6, 0.6, 0.6, 1.0))
    batch_for_shader(sh, 'POINTS', {"pos": [gone]}).draw(sh)
    gpu.state.point_size_set(1.0)
    gpu.state.line_width_set(1.0)
    gpu.state.blend_set('NONE')


def _remove_draw_handler():
    global _draw_handle
    if _draw_handle is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        except Exception:
            pass
    _draw_handle = None
    _hover["pair"] = None


class VIBECUT_OT_click_collapse(bpy.types.Operator):
    """Click vertices one after another: each one merges into the closest
    vertex it shares an edge with (no selecting, no M). Orange = goes away,
    green = stays. Right-click or Esc to stop, Ctrl+Z undoes one click"""
    bl_idname  = "vibecut.click_collapse"
    bl_label   = "Click Collapse"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH'
                and ctx.area is not None and ctx.area.type == 'VIEW_3D')

    def invoke(self, ctx, event):
        global _draw_handle
        wm = ctx.window_manager
        if wm.vibecut_cc_running and _draw_handle is not None:
            wm.vibecut_cc_stop = True
            return {'CANCELLED'}
        _remove_draw_handler()
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
            _draw, (), 'WINDOW', 'POST_VIEW')
        self._obj_name = ctx.active_object.name
        self._cache = None
        self._swallow = False
        self._count = 0
        wm.vibecut_cc_running = True
        wm.vibecut_cc_stop = False
        ctx.area.header_text_set(
            "Click Collapse:  click a vertex to merge it into its nearest "
            "neighbour (orange goes, green stays)   |   Right-click / Esc: "
            "done   |   Ctrl+Z: undo one")
        ctx.window.cursor_modal_set('CROSSHAIR')
        wm.modal_handler_add(self)
        self._hover_at(ctx, event)
        return {'RUNNING_MODAL'}

    def modal(self, ctx, event):
        try:
            return self._modal(ctx, event)
        except Exception as e:
            self._finish(ctx)
            self.report({'ERROR'}, "Click Collapse stopped: %s" % e)
            return {'CANCELLED'}

    def _modal(self, ctx, event):
        ob = ctx.active_object
        if (ctx.window_manager.vibecut_cc_stop or ctx.area is None
                or ctx.mode != 'EDIT_MESH' or ob is None
                or ob.name != self._obj_name):
            return self._finish(ctx)
        if event.type == 'ESC' and event.value == 'PRESS':
            return self._finish(ctx)

        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            self._hover_at(ctx, event)
            return {'PASS_THROUGH'}

        region, rv3d, mouse = self._viewport_under(ctx, event)

        if event.type == 'RIGHTMOUSE' and region is not None:
            if event.value == 'PRESS':
                return self._finish(ctx)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value in {'PRESS', 'DOUBLE_CLICK'} and region is not None:
                if self._collapse_at(ctx, region, rv3d, mouse):
                    self._swallow = True
                    self._hover_at(ctx, event)
                    return {'RUNNING_MODAL'}
            elif self._swallow:
                if event.value == 'RELEASE':
                    self._swallow = False
                return {'RUNNING_MODAL'}

        if event.type not in _NAV_EVENTS:
            self._cache = None
            _hover["pair"] = None
            ctx.area.tag_redraw()
        return {'PASS_THROUGH'}

    def cancel(self, ctx):
        self._finish(ctx)

    def _finish(self, ctx):
        _remove_draw_handler()
        wm = ctx.window_manager
        wm.vibecut_cc_running = False
        wm.vibecut_cc_stop = False
        if ctx.window is not None:
            ctx.window.cursor_modal_restore()
        if ctx.area is not None:
            ctx.area.header_text_set(None)
            ctx.area.tag_redraw()
        if self._count:
            self.report({'INFO'}, "Collapsed %d vertices." % self._count)
            return {'FINISHED'}
        return {'CANCELLED'}

    @staticmethod
    def _viewport_under(ctx, event):
        x, y = event.mouse_x, event.mouse_y
        windows = []
        for r in ctx.area.regions:
            inside = r.x <= x < r.x + r.width and r.y <= y < r.y + r.height
            if not inside:
                continue
            if r.type != 'WINDOW':
                return None, None, None
            windows.append(r)
        if not windows or windows[0].data is None:
            return None, None, None
        r = windows[0]
        return r, r.data, (x - r.x, y - r.y)

    def _geometry(self, ctx):
        ob = ctx.active_object
        bm = bmesh.from_edit_mesh(ob.data)
        c = self._cache
        if c is not None and c["n"] == len(bm.verts):
            return bm, c
        ob.update_from_editmode()
        me = ob.data
        n = len(me.vertices)
        co = np.empty(n * 3, np.float32)
        me.vertices.foreach_get("co", co)
        hide = np.empty(n, bool)
        me.vertices.foreach_get("hide", hide)
        bm.verts.ensure_lookup_table()
        co = co.reshape(n, 3).astype(np.float64)
        size = float(np.ptp(co, axis=0).max()) if n else 1.0
        self._cache = c = {"n": n, "co": co, "usable": ~hide,
                           "bvh": BVHTree.FromBMesh(bm), "size": size}
        return bm, c

    def _pick(self, ctx, region, rv3d, mouse):
        ob = ctx.active_object
        bm, c = self._geometry(ctx)
        if not c["n"]:
            return bm, -1
        mw = ob.matrix_world
        px, front = project_verts(c["co"], rv3d.perspective_matrix @ mw,
                                  region.width, region.height)
        visible = None
        shading = ctx.space_data.shading
        xray = (shading.show_xray_wireframe if shading.type == 'WIREFRAME'
                else shading.show_xray)
        if not xray:
            inv = mw.inverted_safe()
            if rv3d.is_perspective:
                eye = inv @ rv3d.view_matrix.inverted().translation
                visible = make_visibility_test(c["bvh"], c["co"], eye=eye)
            else:
                vd = inv.to_3x3() @ (rv3d.view_rotation @ Vector((0, 0, -1)))
                visible = make_visibility_test(
                    c["bvh"], c["co"], view_dir=vd.normalized(),
                    far=c["size"] * 4.0 + 1.0)
        radius = 40.0 * ctx.preferences.system.pixel_size
        return bm, pick_vertex(px, front, c["usable"], mouse, radius, visible)

    def _hover_at(self, ctx, event):
        region, rv3d, mouse = self._viewport_under(ctx, event)
        pair = None
        if region is not None:
            bm, i = self._pick(ctx, region, rv3d, mouse)
            if i >= 0:
                bm.verts.ensure_lookup_table()
                v = bm.verts[i]
                mw = ctx.active_object.matrix_world
                nb = nearest_neighbor(v, mw.to_3x3())
                pair = ((mw @ v.co).to_tuple(),
                        None if nb is None else (mw @ nb.co).to_tuple())
        if pair != _hover["pair"]:
            _hover["pair"] = pair
            ctx.area.tag_redraw()

    def _collapse_at(self, ctx, region, rv3d, mouse):
        bm, i = self._pick(ctx, region, rv3d, mouse)
        if i < 0:
            return False
        bm.verts.ensure_lookup_table()
        ob = ctx.active_object
        if collapse_vertex(bm, bm.verts[i], ob.matrix_world.to_3x3()) is None:
            self.report({'WARNING'}, "That vertex has no neighbour to merge into.")
            return True
        bmesh.update_edit_mesh(ob.data, loop_triangles=True, destructive=True)
        bpy.ops.ed.undo_push(message="Click Collapse")
        self._cache = None
        self._count += 1
        return True


classes = (
    VIBECUT_OT_click_collapse,
)


def register():
    W = bpy.types.WindowManager
    W.vibecut_cc_running = bpy.props.BoolProperty(options={'SKIP_SAVE'})
    W.vibecut_cc_stop = bpy.props.BoolProperty(options={'SKIP_SAVE'})
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    _remove_draw_handler()
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.WindowManager.vibecut_cc_stop
    del bpy.types.WindowManager.vibecut_cc_running
