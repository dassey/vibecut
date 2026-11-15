import bpy
from mathutils import Vector

from . import detection


CANDIDATE_COLL_NAME = "VibeCut Candidates"
VG_PREFIX = "VC_limb_"


def _get_or_create_collection(scene, name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
        scene.collection.children.link(coll)
    return coll


def _wipe_collection(name):
    coll = bpy.data.collections.get(name)
    if coll is None:
        return 0
    n = len(coll.objects)
    for o in list(coll.objects):
        bpy.data.objects.remove(o)
    bpy.data.collections.remove(coll)
    return n


class VIBECUT_OT_detect_joints(bpy.types.Operator):
    """Run the voxel + SDF neck-detection pipeline on the active mesh.

    Each detected candidate is placed as a CIRCLE Empty in the
    'VibeCut Candidates' collection: location at the neck centroid, size at
    the neck radius, Y-axis rotated to match the neck normal. Empty name
    encodes the candidate score (higher = better)"""
    bl_idname  = "vibecut.detect_joints"
    bl_label   = "Find Joint Candidates"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.active_object is not None
                and ctx.active_object.type == 'MESH'
                and ctx.mode == 'OBJECT')

    def execute(self, ctx):
        scene = ctx.scene
        obj = ctx.active_object

        resolution = scene.vibecut_resolution if hasattr(scene, 'vibecut_resolution') else 128
        sensitivity = scene.vibecut_sensitivity if hasattr(scene, 'vibecut_sensitivity') else 0.3

        try:
            candidates = detection.detect_necks_blender(
                obj, resolution=resolution, sensitivity=sensitivity,
            )
        except Exception as e:
            self.report({'ERROR'}, f"Detection failed: {e}")
            return {'CANCELLED'}

        if not candidates:
            self.report({'WARNING'},
                        "No neck candidates found. Try lowering Sensitivity.")
            return {'CANCELLED'}

        _wipe_collection(CANDIDATE_COLL_NAME)
        coll = _get_or_create_collection(scene, CANDIDATE_COLL_NAME)

        for vg in [g for g in obj.vertex_groups if g.name.startswith(VG_PREFIX)]:
            obj.vertex_groups.remove(vg)

        n_with_limb = 0
        for i, cand in enumerate(candidates):
            score = cand['score']
            radius = cand['radius_world']
            pos = cand['world_pos']
            normal = cand['world_normal']
            limb = cand.get('limb_vert_indices') or set()

            name = f"VC_Cand_{i:02d}_score{int(score * 100):03d}"
            empty = bpy.data.objects.new(name, None)
            empty.empty_display_type = 'CIRCLE'
            empty.empty_display_size = max(0.001, radius)
            empty.location = pos
            up = Vector((0.0, 0.0, 1.0))
            quat = up.rotation_difference(normal.normalized())
            empty.rotation_mode = 'QUATERNION'
            empty.rotation_quaternion = quat
            empty["vibecut_score"] = float(score)
            empty["vibecut_radius"] = float(radius)
            empty["vibecut_normal"] = (normal.x, normal.y, normal.z)
            empty["vibecut_source"] = obj.name
            empty["vibecut_vcount"] = len(obj.data.vertices)

            if limb:
                vg = obj.vertex_groups.new(name=VG_PREFIX + name)
                vg.add(list(limb), 1.0, 'REPLACE')
                empty["vibecut_vgroup"] = vg.name
                n_with_limb += 1

            coll.objects.link(empty)

        extra = ""
        if n_with_limb == 0:
            extra = (" (no limb membership — object has modifiers; apply them "
                     "to enable one-click Cut Limb)")
        elif n_with_limb < len(candidates):
            extra = f" ({n_with_limb} cuttable)"
        self.report(
            {'INFO'},
            f"Found {len(candidates)} candidate(s). "
            f"Best score: {candidates[0]['score']:.2f}{extra}"
        )
        return {'FINISHED'}


def _active_candidate(ctx):
    coll = bpy.data.collections.get(CANDIDATE_COLL_NAME)
    if coll is None:
        return None
    act = ctx.active_object
    if act is not None and act.name in coll.objects:
        return act
    sel = [o for o in ctx.selected_objects if o.name in coll.objects]
    return sel[0] if len(sel) == 1 else None


class VIBECUT_OT_cut_at_candidate(bpy.types.Operator):
    """Cut the selected candidate's limb off the body in one step.

    Separates along the contour the detector found — which follows the actual
    neck, not a flat plane — caps both openings, and optionally adds a
    compression-fit pin and socket so the piece presses back on after printing"""
    bl_idname  = "vibecut.cut_at_candidate"
    bl_label   = "Cut Limb at Candidate"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'OBJECT' and _active_candidate(ctx) is not None

    def execute(self, ctx):
        from .geometry import _execute_contour_cut

        empty = _active_candidate(ctx)
        if empty is None:
            self.report({'ERROR'}, "Select one candidate circle first.")
            return {'CANCELLED'}

        src_name = empty.get("vibecut_source", "")
        obj = bpy.data.objects.get(src_name)
        if obj is None or obj.type != 'MESH':
            self.report({'ERROR'},
                        f"Source mesh '{src_name}' is gone. Re-run detection.")
            return {'CANCELLED'}

        vcount = empty.get("vibecut_vcount", -1)
        if vcount != len(obj.data.vertices):
            self.report({'ERROR'},
                        "'%s' has been edited since detection (%d verts now, "
                        "%d then). Re-run Find Joint Candidates."
                        % (obj.name, len(obj.data.vertices), vcount))
            return {'CANCELLED'}

        vg_name = empty.get("vibecut_vgroup", "")
        vg = obj.vertex_groups.get(vg_name) if vg_name else None
        if vg is None:
            self.report({'ERROR'},
                        "No limb membership stored for this candidate. "
                        "Re-run detection (and apply modifiers first).")
            return {'CANCELLED'}

        gi = vg.index
        limb = {v.index for v in obj.data.vertices
                if any(g.group == gi for g in v.groups)}

        n_total = len(obj.data.vertices)
        if not (0 < len(limb) < n_total):
            self.report({'ERROR'},
                        "Limb selection covers %d of %d verts — nothing "
                        "meaningful to separate. Try a higher Sensitivity."
                        % (len(limb), n_total))
            return {'CANCELLED'}

        pos    = empty.location.copy()
        radius = float(empty.get("vibecut_radius", 1.0))
        nrm    = empty.get("vibecut_normal", (0.0, 0.0, 1.0))
        normal = Vector((nrm[0], nrm[1], nrm[2])).normalized()

        body_obj, limb_obj = _execute_contour_cut(
            ctx, obj, limb, pos, normal, radius)

        if limb_obj is None:
            self.report({'ERROR'},
                        "Separation produced only one piece — the limb set may "
                        "not be topologically closed. Nothing was split off.")
            return {'CANCELLED'}

        msg = f"Cut '{limb_obj.name}' off '{body_obj.name}' ({len(limb)} verts)."

        if ctx.scene.vibecut_auto_pressfit:
            try:
                from . import joints
                joints.place_press_fit(ctx, pos, normal, radius)
                msg += " Press-fit joint placed — review, then Apply Joint."
            except Exception as e:
                self.report({'WARNING'}, f"Cut OK, press-fit failed: {e}")
                print(f"[VibeCut] press-fit after contour cut failed: {e}")

        try:
            if vg.name in [g.name for g in obj.vertex_groups]:
                obj.vertex_groups.remove(vg)
        except Exception:
            pass
        try:
            bpy.data.objects.remove(empty)
        except Exception:
            pass

        self.report({'INFO'}, msg)
        return {'FINISHED'}


class VIBECUT_OT_clear_candidates(bpy.types.Operator):
    """Delete all candidate empties and the VibeCut Candidates collection"""
    bl_idname  = "vibecut.clear_candidates"
    bl_label   = "Clear Candidates"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        n = _wipe_collection(CANDIDATE_COLL_NAME)
        if n == 0:
            self.report({'INFO'}, "Nothing to clear.")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Cleared {n} candidate(s).")
        return {'FINISHED'}


class VIBECUT_PT_auto_detect(bpy.types.Panel):
    bl_label       = "Auto Joint Detection"
    bl_idname      = "VIBECUT_PT_auto_detect"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "VibeCut"
    bl_parent_id   = "VIBECUT_PT_more"
    bl_order       = 3
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L = self.layout
        scene = ctx.scene

        if hasattr(scene, 'vibecut_resolution'):
            L.prop(scene, "vibecut_resolution")
        if hasattr(scene, 'vibecut_sensitivity'):
            L.prop(scene, "vibecut_sensitivity")

        row = L.row()
        row.scale_y = 1.4
        row.operator("vibecut.detect_joints",
                     text="Find Joint Candidates",
                     icon='ZOOM_SELECTED')

        coll = bpy.data.collections.get(CANDIDATE_COLL_NAME)
        if coll and len(coll.objects) > 0:
            L.label(text=f"{len(coll.objects)} candidate(s) shown",
                    icon='OUTLINER_OB_EMPTY')

            L.separator()
            cand = _active_candidate(ctx)
            box  = L.box()
            if cand is None:
                box.label(text="Click one circle to cut it", icon='RESTRICT_SELECT_ON')
            else:
                box.label(text=cand.name, icon='CHECKMARK')
                score = cand.get("vibecut_score", 0.0)
                box.label(text="score %.2f   radius %.2f"
                               % (score, cand.get("vibecut_radius", 0.0)))
                if not cand.get("vibecut_vgroup", ""):
                    box.label(text="no limb data — apply modifiers, re-detect",
                              icon='ERROR')
            box.prop(scene, "vibecut_auto_pressfit")
            row = box.row()
            row.scale_y = 1.5
            row.enabled = cand is not None and bool(cand.get("vibecut_vgroup", ""))
            row.operator("vibecut.cut_at_candidate",
                         text="Cut Limb Off", icon='SCULPTMODE_HLT')

            L.operator("vibecut.clear_candidates",
                       text="Clear Candidates",
                       icon='TRASH')
        else:
            L.label(text="(no candidates yet)", icon='INFO')

        L.separator()
        box = L.box()
        col = box.column(align=True)
        col.scale_y = 0.85
        col.label(text="Workflow:", icon='INFO')
        col.label(text="  1. Click Find Joint Candidates")
        col.label(text="  2. Click a circle in the viewport")
        col.label(text="  3. Click Cut Limb Off")
        col.label(text="  (or use the Ring Cutter by hand)")


classes = (
    VIBECUT_OT_detect_joints,
    VIBECUT_OT_cut_at_candidate,
    VIBECUT_OT_clear_candidates,
    VIBECUT_PT_auto_detect,
)


def register():
    bpy.types.Scene.vibecut_auto_pressfit = bpy.props.BoolProperty(
        name="Add press-fit joint",
        description=("After cutting, place a compression-fit pin and socket at "
                     "the seam so the piece presses back on once printed. "
                     "Review the preview, then Apply Joint"),
        default=True)
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.vibecut_auto_pressfit
