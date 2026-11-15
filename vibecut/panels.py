import json
import bpy
from .geometry import (_state, _rebuild, _pca_axis, _measure_cut_boundary,
                       _execute_cut, _fill_open_boundary)
from . import joints


class VIBECUT_PT_cutter(bpy.types.Panel):
    bl_label       = "Ring Cutter v1.0.0.38"
    bl_idname      = "VIBECUT_PT_cutter"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "VibeCut"
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L     = self.layout
        scene = ctx.scene
        phase = _state.get("phase", 0)
        active = _state.get("active", False)

        L.row().prop(scene, "rc_cut_mode", expand=True)
        if scene.rc_cut_mode == 'DISK':
            L.row().prop(scene, "rc_disk_thickness")
        L.separator()

        row = L.row()
        row.scale_y = 1.5
        row.operator("view3d.rc_hover", text="Activate Ring Cutter",
                     icon='MESH_CIRCLE',
                     depress=(active))

        L.separator()

        box = L.box()
        box.enabled = (phase == 1)
        box.label(text="Snap Normal To", icon='SNAP_NORMAL')
        row = box.row(align=True)
        for ax, col_name in (('X', 'X'), ('Y', 'Y'), ('Z', 'Z'), ('VIEW', 'View'), ('AUTO', 'Auto')):
            row.operator("view3d.rc_snap", text=col_name).axis = ax

        L.separator()

        box2 = L.box()
        box2.label(text="Handles (viewport)", icon='GIZMO')
        col = box2.column(align=True)
        col.scale_y = 0.85
        for txt in (
            "● Green arrow  — slide up/down limb",
            "● Orange arrow — slide left/right",
            "● Cyan dot     — tilt plane (drag)",
            "● Yellow dot   — resize ring (drag)",
        ):
            col.label(text=txt)
        col.separator()
        col.label(text="Scroll       → tilt plane")
        col.label(text="Alt+Scroll   → slide up/dn")
        col.label(text="Shift+Scroll → slide L/R")
        col.label(text="Ctrl+Scroll  → resize")
        col.label(text="X/Y/Z/V/N   → snap")
        col.label(text="Enter/Space  → CUT")
        col.label(text="Esc/RMB     → back to hover")

        L.separator()

        box3 = L.box()
        box3.label(text="Cut Options", icon='CHECKBOX_HLT')
        box3.prop(scene, "rc_opt_fill")
        box3.prop(scene, "rc_opt_sep")
        row2 = box3.row(align=True)
        row2.prop(scene, "rc_opt_neg", toggle=True)
        row2.prop(scene, "rc_opt_pos", toggle=True)

        L.separator()

        row3 = L.row(align=True)
        row3.enabled = bool(active)
        row3.scale_y = 1.4
        cut = row3.operator("view3d.rc_do_cut",  text="CUT",  icon='SHADERFX')
        bk  = row3.operator("view3d.rc_do_back", text="BACK", icon='BACK')


class VIBECUT_OT_do_cut(bpy.types.Operator):
    bl_idname = "view3d.rc_do_cut"
    bl_label  = "Execute Cut"
    bl_description = ("Perform the cut at the locked ring. Only geometry inside "
                      "the orange outline is affected")
    bl_options = {'INTERNAL'}

    def execute(self, ctx):
        if not _state.get("active"):
            return {'CANCELLED'}

        view_ctx = None
        for window in ctx.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for region in area.regions:
                        if region.type == 'WINDOW':
                            view_ctx = ctx.copy()
                            view_ctx['window'] = window
                            view_ctx['area']   = area
                            view_ctx['region'] = region
                            for space in area.spaces:
                                if space.type == 'VIEW_3D':
                                    view_ctx['region_data'] = space.region_3d
                                    view_ctx['space_data']  = space
                                    break
                            break
                if view_ctx:
                    break
            if view_ctx:
                break

        exec_ctx = view_ctx if view_ctx else ctx
        _execute_cut(exec_ctx)
        _state.update(active=False, phase=0, _panel_cut=True)
        if ctx.area:
            ctx.area.tag_redraw()
        if view_ctx and view_ctx.get('area'):
            view_ctx['area'].tag_redraw()
        return {'FINISHED'}


class VIBECUT_OT_do_back(bpy.types.Operator):
    bl_idname = "view3d.rc_do_back"
    bl_label  = "Back to Hover"
    bl_description = "Unlock the ring and go back to positioning it with the mouse"
    bl_options = {'INTERNAL'}

    def execute(self, ctx):
        from .geometry import HOVER
        _state["phase"] = HOVER
        if ctx.area:
            ctx.area.tag_redraw()
        return {'FINISHED'}


class VIBECUT_PT_joint(bpy.types.Panel):
    bl_label       = "Joint"
    bl_idname      = "VIBECUT_PT_joint"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "VibeCut"
    bl_parent_id   = "VIBECUT_PT_more"
    bl_order       = 4
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L     = self.layout
        scene = ctx.scene

        has_cut = (scene.get("rc_plane_co") is not None
                   and scene.get("rc_body", "") != ""
                   and scene.get("rc_limb", "") != "")
        has_preview = (scene.get("rc_joint_ball", "") != ""
                       and scene.get("rc_joint_socket", "") != "")

        if has_cut:
            info = L.box()
            c = info.column(align=True)
            c.label(text=f"Body : {scene.get('rc_body', '—')}",  icon='MESH_UVSPHERE')
            c.label(text=f"Limb : {scene.get('rc_limb', '—')}",  icon='MESH_CYLINDER')
            unit_scale = ctx.scene.unit_settings.scale_length
            r_mm = scene.get('rc_radius', 0) * unit_scale * 1000
            c.label(text=f"Ring r  : {r_mm:.2f} mm")
        else:
            L.label(text="No cut yet — use Ring Cutter", icon='INFO')
            L.label(text="  with 'Separate Halves' on.")

        L.separator()

        box = L.box()
        box.label(text="Joint Type", icon='CONSTRAINT_BONE')
        box.prop(scene, "rc_joint_type", expand=True)

        L.separator()

        joint_type = getattr(scene, "rc_joint_type", "LIBRARY")

        if joint_type == 'LIBRARY':
            pbox = L.box()
            pbox.label(text="Joint Library", icon='MESH_UVSPHERE')
            pbox.prop(scene, "rc_joint_file")
            pbox.prop(scene, "rc_ball_pct", slider=True)
        elif joint_type == 'PRESS_FIT':
            pbox = L.box()
            pbox.label(text="Press Fit Parameters", icon='SNAP_FACE')
            pbox.prop(scene, "rc_pin_pct",       slider=True)
            pbox.prop(scene, "rc_pin_depth")
            pbox.prop(scene, "rc_pin_tolerance")
        else:
            pbox = L.box()
            pbox.label(text="Screw Top Parameters", icon='RECOVER_LAST')
            pbox.prop(scene, "rc_screw_pct",       slider=True)
            pbox.prop(scene, "rc_screw_pitch")
            pbox.prop(scene, "rc_screw_depth")
            pbox.prop(scene, "rc_screw_turns")
            pbox.prop(scene, "rc_screw_tolerance")
            pbox.prop(scene, "rc_screw_embed_pct", slider=True)

        L.separator()

        if not has_preview:
            row = L.row()
            row.scale_y = 1.6
            row.enabled = has_cut
            row.operator("mesh.rc_place_joint",
                         text="Place Joint",
                         icon='IMPORT')
        else:
            L.label(text="Move/rotate the joint, then:", icon='PIVOT_CURSOR')

            col = L.column(align=True)
            col.scale_y = 1.5
            col.operator("mesh.rc_apply_joint",
                         text="Apply Boolean",
                         icon='CHECKMARK')
            col.operator("mesh.rc_cancel_joint",
                         text="Cancel",
                         icon='X')


class VIBECUT_OT_place_joint(bpy.types.Operator):
    bl_description = ("Preview a joint at the last cut. Nothing is committed until "
                      "you press Apply Joint")
    bl_idname  = "mesh.rc_place_joint"
    bl_label   = "Place Joint"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        scene     = ctx.scene
        body_name = scene.get("rc_body", "")
        limb_name = scene.get("rc_limb", "")
        if not body_name or not limb_name:
            self.report({'WARNING'}, "No cut data. Run Ring Cutter first.")
            return {'CANCELLED'}

        plane_co = scene.get("rc_plane_co")
        plane_no = scene.get("rc_plane_no")
        if not plane_co or not plane_no:
            self.report({'WARNING'}, "Plane data missing — re-run the cut.")
            return {'CANCELLED'}

        from mathutils import Vector
        joint_type = getattr(scene, "rc_joint_type", "LIBRARY")

        if joint_type == 'PRESS_FIT':
            pin_pct       = getattr(scene, "rc_pin_pct", 0.45)
            target_radius = scene.get("rc_radius", 0.01) * pin_pct
            pin, hole = joints.place_press_fit(
                ctx,
                plane_co_w    = Vector(plane_co),
                plane_no_w    = Vector(plane_no),
                target_radius = target_radius,
            )
            if pin is not None:
                self.report({'INFO'},
                            "Press-fit placed. Move/rotate freely, then click Apply Boolean.")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Press-fit placement failed — check console.")
                return {'CANCELLED'}

        elif joint_type == 'SCREW_TOP':
            screw_pct     = getattr(scene, "rc_screw_pct", 0.55)
            target_radius = scene.get("rc_radius", 0.01) * screw_pct
            male, female = joints.place_screw_top(
                ctx,
                plane_co_w    = Vector(plane_co),
                plane_no_w    = Vector(plane_no),
                target_radius = target_radius,
            )
            if male is not None:
                self.report({'INFO'},
                            "Screw-top placed. Move/rotate freely, then click Apply Boolean.")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Screw-top placement failed — check console.")
                return {'CANCELLED'}

        else:
            blend_file = getattr(scene, "rc_joint_file", "none")
            if not blend_file or blend_file == "none":
                self.report({'WARNING'}, "No joint selected.")
                return {'CANCELLED'}

            ball_pct      = getattr(scene, "rc_ball_pct", 0.85)
            target_radius = scene.get("rc_radius", 0.01) * ball_pct

            ball, socket = joints.place_joint(
                ctx,
                set_id        = blend_file,
                plane_co_w    = Vector(plane_co),
                plane_no_w    = Vector(plane_no),
                target_radius = target_radius,
            )
            if ball is not None:
                self.report({'INFO'},
                            "Joint placed. Move/rotate freely, then click Apply Boolean.")
                return {'FINISHED'}
            else:
                self.report({'ERROR'}, "Joint placement failed — check console.")
                return {'CANCELLED'}


class VIBECUT_OT_apply_joint(bpy.types.Operator):
    bl_description = "Boolean the previewed joint into the two halves for good"
    bl_idname  = "mesh.rc_apply_joint"
    bl_label   = "Apply Joint Boolean"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        joint_type = getattr(ctx.scene, "rc_joint_type", "LIBRARY")
        if joint_type == 'PRESS_FIT':
            ok = joints.apply_press_fit_boolean(ctx)
        elif joint_type == 'SCREW_TOP':
            ok = joints.apply_screw_top_boolean(ctx)
        else:
            ok = joints.apply_joint_boolean(ctx)
        if ok:
            self.report({'INFO'}, "Joint boolean applied.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Apply failed — check console.")
            return {'CANCELLED'}


class VIBECUT_OT_cancel_joint(bpy.types.Operator):
    bl_description = "Discard the joint preview and leave the halves untouched"
    bl_idname  = "mesh.rc_cancel_joint"
    bl_label   = "Cancel Joint Placement"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        joints.cancel_joint_placement(ctx)
        self.report({'INFO'}, "Joint placement cancelled.")
        return {'FINISHED'}


class VIBECUT_OT_insert_joint(bpy.types.Operator):
    bl_idname  = "mesh.rc_insert_joint"
    bl_label   = "Insert Joint from Library"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        scene     = ctx.scene
        body_name = scene.get("rc_body", "")
        limb_name = scene.get("rc_limb", "")
        if not body_name or not limb_name:
            self.report({'WARNING'}, "No cut data. Run Ring Cutter first.")
            return {'CANCELLED'}

        body_obj = bpy.data.objects.get(body_name)
        limb_obj = bpy.data.objects.get(limb_name)
        if not body_obj or not limb_obj:
            self.report({'WARNING'}, "Body/Limb objects not found in scene.")
            return {'CANCELLED'}

        plane_co = scene.get("rc_plane_co")
        plane_no = scene.get("rc_plane_no")
        if not plane_co or not plane_no:
            self.report({'WARNING'}, "Plane data missing — re-run the cut.")
            return {'CANCELLED'}

        blend_file = getattr(scene, "rc_joint_file", "none")
        if not blend_file or blend_file == "none":
            self.report({'WARNING'}, "No joint selected.")
            return {'CANCELLED'}

        from mathutils import Vector
        ball_pct = getattr(scene, "rc_ball_pct", 0.85)
        target_radius = scene.get("rc_radius", 0.01) * ball_pct

        ok = joints.insert_joint(
            ctx,
            set_id        = blend_file,
            body_obj      = body_obj,
            limb_obj      = limb_obj,
            plane_co_w    = Vector(plane_co),
            plane_no_w    = Vector(plane_no),
            target_radius = target_radius,
        )
        if ok:
            self.report({'INFO'}, f"Joint inserted (r={target_radius:.4f} m)")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Joint insertion failed — check console.")
            return {'CANCELLED'}


class VIBECUT_OT_align_to_active(bpy.types.Operator):
    """Align selected objects to the active object along chosen axes"""
    bl_idname  = "vibecut.align_to_active"
    bl_label   = "Align to Active"
    bl_options = {'REGISTER', 'UNDO'}

    align_x: bpy.props.BoolProperty(name="X", default=False)
    align_y: bpy.props.BoolProperty(name="Y", default=False)
    align_z: bpy.props.BoolProperty(name="Z", default=True)
    mode: bpy.props.EnumProperty(
        name="Mode",
        items=[
            ('ORIGIN',        "Origin",        "Match object origins directly"),
            ('BOUNDS_MIN',    "Bounds Min",    "Match minimum bounding box side (e.g. flat on build plate)"),
            ('BOUNDS_CENTER', "Bounds Center", "Match bounding box centers"),
            ('BOUNDS_MAX',    "Bounds Max",    "Match maximum bounding box side"),
        ],
        default='BOUNDS_MIN',
    )

    def _world_bbox(self, obj):
        from mathutils import Vector
        return [obj.matrix_world @ Vector(c) for c in obj.bound_box]

    def _get_ref(self, obj, axis_idx):
        if self.mode == 'ORIGIN':
            return obj.location[axis_idx]
        vals = [c[axis_idx] for c in self._world_bbox(obj)]
        if self.mode == 'BOUNDS_MIN':
            return min(vals)
        if self.mode == 'BOUNDS_MAX':
            return max(vals)
        return (min(vals) + max(vals)) * 0.5

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_props_dialog(self)

    def draw(self, ctx):
        L = self.layout
        row = L.row(align=True)
        row.prop(self, "align_x", toggle=True)
        row.prop(self, "align_y", toggle=True)
        row.prop(self, "align_z", toggle=True)
        L.prop(self, "mode")

    def execute(self, ctx):
        active   = ctx.active_object
        selected = [o for o in ctx.selected_objects if o is not active]

        if active is None:
            self.report({'ERROR'}, "No active object.")
            return {'CANCELLED'}
        if not selected:
            self.report({'ERROR'}, "Select at least 2 objects (active = anchor, others move).")
            return {'CANCELLED'}

        active_axes = [i for i, flag in enumerate([self.align_x, self.align_y, self.align_z]) if flag]
        if not active_axes:
            self.report({'WARNING'}, "No axes ticked — nothing to do.")
            return {'CANCELLED'}

        for axis_idx in active_axes:
            ref = self._get_ref(active, axis_idx)
            for obj in selected:
                obj.location[axis_idx] += ref - self._get_ref(obj, axis_idx)

        axes_str = "".join("XYZ"[i] for i in active_axes)
        self.report(
            {'INFO'},
            f"Aligned {len(selected)} object(s) to '{active.name}' "
            f"[{axes_str}] — {self.mode.replace('_', ' ').title()}"
        )
        return {'FINISHED'}


class VIBECUT_OT_drop_to_floor(bpy.types.Operator):
    """Move each selected object straight down so its bottom (world-space
    bounding box min Z) rests on the build plate (Z=0)"""
    bl_idname  = "vibecut.drop_to_floor"
    bl_label   = "Drop to Floor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return any(o.type == 'MESH' for o in ctx.selected_objects)

    def execute(self, ctx):
        from mathutils import Vector
        moved = 0
        for obj in ctx.selected_objects:
            if obj.type != 'MESH':
                continue
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            min_z = min(c.z for c in corners)
            if abs(min_z) > 1e-6:
                obj.location.z -= min_z
                moved += 1
        self.report({'INFO'}, f"Dropped {moved} object(s) to Z=0.")
        return {'FINISHED'}


class VIBECUT_PT_align(bpy.types.Panel):
    bl_label       = "Align Tools"
    bl_idname      = "VIBECUT_PT_align"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "VibeCut"
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L = self.layout
        col = L.column(align=True)
        col.scale_y = 1.5
        col.operator("vibecut.align_to_active",
                     text="Align to Active",
                     icon='OBJECT_ORIGIN')
        col.operator("vibecut.drop_to_floor",
                     text="Drop to Floor",
                     icon='TRIA_DOWN_BAR')


classes = (
    VIBECUT_PT_cutter,
    VIBECUT_PT_joint,
    VIBECUT_OT_do_cut,
    VIBECUT_OT_do_back,
    VIBECUT_OT_place_joint,
    VIBECUT_OT_apply_joint,
    VIBECUT_OT_cancel_joint,
    VIBECUT_OT_insert_joint,
    VIBECUT_OT_align_to_active,
    VIBECUT_OT_drop_to_floor,
    VIBECUT_PT_align,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
