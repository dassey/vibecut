"""
joint_library.py -- Blend-file joint library: discover, import, scale, orient, boolean.

Two-phase workflow
------------------
1. place_joint()         -- import + orient joint objects, leave in scene for user
                            to freely move/rotate before committing
2. apply_joint_boolean() -- bake current position and run UNION booleans

insert_joint() is a one-shot convenience that combines both steps (no
repositioning step) -- used by scripts and tests.

File naming conventions (two schemes):

  Scheme A -- Paired files (RECOMMENDED):
    Joints/<setname>_ball.blend     -- single object (the ball piece)
    Joints/<setname>_socket.blend   -- single object (the socket cup)

  Scheme B -- Combined file (legacy):
    Joints/<name>.blend  -- contains BOTH joint_ball AND joint_socket objects

Object conventions:
  - ORIGIN at attachment face centre (the flat face that welds to the cut)
  - +Z axis points away from attachment face toward the mating part
  - Optional custom property joint_radius (float) on the ball object
    = sphere radius at scale 1.0; auto-detected from vertex extents if absent
"""

import bpy
import os
from mathutils import Vector

JOINTS_DIR = os.path.join(os.path.dirname(__file__), "Joints")

# Stable tag names used for placed preview objects
_BALL_TAG   = "rc_joint_ball_preview"
_SOCKET_TAG = "rc_joint_socket_preview"
_PIN_TAG    = "rc_press_pin_preview"
_HOLE_TAG   = "rc_press_hole_preview"
_SCREW_M_TAG = "rc_screw_male_preview"
_SCREW_F_TAG = "rc_screw_female_preview"


# ---------------------------------------------------------------------------
# Blend file discovery
# ---------------------------------------------------------------------------

def _find_joint_sets():
    """
    Scan Joints/ and return (set_id, label, description) tuples for EnumProperty.
    Scheme A pairs take priority; lone files are listed as Scheme B.
    """
    if not os.path.isdir(JOINTS_DIR):
        return [("none", "Joints/ folder not found", "")]

    files = {f for f in os.listdir(JOINTS_DIR) if f.endswith(".blend")}
    items = []
    seen  = set()

    for f in sorted(files):
        if not f.endswith("_ball.blend"):
            continue
        base   = f[: -len("_ball.blend")]
        socket = base + "_socket.blend"
        if socket not in files:
            continue
        items.append(("A:" + base, base.replace("_", " ").title() + " (paired)", ""))
        seen.add(f)
        seen.add(socket)

    for f in sorted(files):
        if f in seen:
            continue
        label = f.replace(".blend", "").replace("_", " ").title()
        items.append(("B:" + f, label, ""))

    return items or [("none", "No joints found in Joints/ folder", "")]


def joint_enum_items(self, context):
    return _find_joint_sets()


# ---------------------------------------------------------------------------
# Object import helper
# ---------------------------------------------------------------------------

def _import_object(blend_path, preferred_name, collection):
    """
    Append an object from blend_path into collection.
    Tries preferred_name first; falls back to the first object in the file.
    Returns the bpy.types.Object or None.
    """
    if not os.path.isfile(blend_path):
        print(f"[VibeCut] Blend not found: {blend_path}")
        return None

    with bpy.data.libraries.load(blend_path, link=False) as (src, dst):
        available = list(src.objects)
        if not available:
            print(f"[VibeCut] No objects in {os.path.basename(blend_path)}")
            return None
        target = preferred_name if preferred_name in available else available[0]
        if target != preferred_name:
            print(f"[VibeCut] '{preferred_name}' not in {os.path.basename(blend_path)}"
                  f" -- using '{target}' instead")
        dst.objects = [target]

    for obj in dst.objects:
        if obj is not None:
            try:
                collection.objects.link(obj)
            except RuntimeError:
                pass
            return obj
    return None


# ---------------------------------------------------------------------------
# Reference radius detection
# ---------------------------------------------------------------------------

def _get_ref_radius(obj):
    """
    Maximum XY radius of the object measured in WORLD space.
    Call BEFORE transform_apply so the blend-file location offset is still live.
    Forces a depsgraph update so matrix_world reflects the actual location.
    Falls back to joint_radius IDProperty when present.
    """
    if "joint_radius" in obj.keys():
        return float(obj["joint_radius"])

    import bmesh
    # Force scene graph evaluation so matrix_world includes the location offset
    bpy.context.view_layer.update()
    mw  = obj.matrix_world
    bm  = bmesh.new()
    bm.from_mesh(obj.data)
    if not bm.verts:
        bm.free()
        return obj.dimensions.x / 2.0
    max_r = max(((mw @ v.co).x ** 2 + (mw @ v.co).y ** 2) ** 0.5 for v in bm.verts)
    bm.free()
    print(f"[VibeCut] _get_ref_radius({obj.name}): matrix_world.t={list(mw.translation)[:3]}  max_r={max_r:.4f}")
    return max_r if max_r > 1e-9 else obj.dimensions.x / 2.0


def _get_stem_offset(obj):
    """
    Distance from the attachment face (world-space minimum Z) to the object
    world origin, measured PRE-normalize.
    Forces depsgraph update before reading matrix_world.
    A positive return value = origin is that many units ABOVE the attachment face.
    """
    import bmesh
    bpy.context.view_layer.update()
    mw = obj.matrix_world
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if not bm.verts:
        bm.free()
        return 0.0
    world_zs = [(mw @ v.co).z for v in bm.verts]
    bm.free()
    offset = -min(world_zs)   # how far origin is above attachment face
    print(f"[VibeCut] _get_stem_offset({obj.name}): min_world_z={min(world_zs):.4f}  offset={offset:.4f}")
    return offset


def _center_mesh_xy(obj):
    """
    Translate the mesh vertex coordinates so the XY bounding-box centre lands
    exactly at the local origin.  Call AFTER transform_apply to fix any
    authoring offsets that were baked in (e.g. ball Y-offset of 0.628).
    Also recentres Z so the minimum Z vertex (attachment face) is at Z=0.
    """
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if not bm.verts:
        bm.free()
        return
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = min(zs)                        # attachment face always at local Z = 0
    for v in bm.verts:
        v.co.x -= cx
        v.co.y -= cy
        v.co.z -= cz
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    print(f"[VibeCut] _center_mesh_xy({obj.name}): shifted XY=({cx:.4f},{cy:.4f})  Z={cz:.4f}")


# ---------------------------------------------------------------------------
# Shared prepare helper
# ---------------------------------------------------------------------------

def _prepare_and_orient(ctx, set_id, plane_co_w, plane_no_w, target_radius,
                        ball_name="joint_ball", socket_name="joint_socket"):
    """
    Import, measure, normalize, scale, and orient both joint objects at the
    cut plane.  Returns (ball_obj, socket_obj) or (None, None).

    Measurements (ref_r, stem_offset) are taken PRE-normalize while the
    object location is still in matrix_world.  Normalize is applied after
    measuring so vertex coordinates are baked for the boolean solver.

    The attachment face of each piece is placed exactly at plane_co_w.
    """
    from .geometry import _orient_matrix
    from mathutils import Matrix

    col = ctx.scene.collection

    if set_id.startswith("A:"):
        base        = set_id[2:]
        ball_path   = os.path.join(JOINTS_DIR, base + "_ball.blend")
        socket_path = os.path.join(JOINTS_DIR, base + "_socket.blend")
    elif set_id.startswith("B:"):
        ball_path = socket_path = os.path.join(JOINTS_DIR, set_id[2:])
    else:
        print(f"[VibeCut] _prepare_and_orient: unknown set_id {set_id!r}")
        return None, None

    ball   = _import_object(ball_path,   ball_name,   col)
    socket = _import_object(socket_path, socket_name, col)
    if ball is None or socket is None:
        for o in (ball, socket):
            if o:
                bpy.data.objects.remove(o, do_unlink=True)
        return None, None

    # Measure ref_r PRE-normalize using world-space vertices.
    # After transform_apply the location offset is baked into vertex coords,
    # skewing the XY extent calculation.  Measure first, normalize second.
    # Scale is derived from the SOCKET so its outer rim fits the cut opening.
    # The same scale factor is then applied to the ball, keeping both matched.
    ref_r         = _get_ref_radius(socket)
    ball_ref_r    = _get_ref_radius(ball)
    ball_stem_off = _get_stem_offset(ball)    # origin → attachment face distance
    sock_stem_off = _get_stem_offset(socket)

    print(f"[VibeCut] PRE-normalize: socket_ref_r={ref_r:.4f}  ball_ref_r={ball_ref_r:.4f}  "
          f"ball_stem_off={ball_stem_off:.4f}  sock_stem_off={sock_stem_off:.4f}")

    if ref_r < 1e-9:
        print("[VibeCut] _prepare_and_orient: zero socket reference radius.")
        for o in (ball, socket):
            bpy.data.objects.remove(o, do_unlink=True)
        return None, None

    # Normalize: bake blend-file location/rotation/scale into vertex coordinates.
    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (ball, socket):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    # Recenter: fix any XY baking offset so both pieces share the same axis.
    # Also pins attachment face to local Z=0 so stem_offset math stays exact.
    for jobj in (ball, socket):
        _center_mesh_xy(jobj)
    scale = target_radius / ref_r
    co    = Vector(plane_co_w)
    no    = Vector(plane_no_w).normalized()

    # Measure scaled heights of each joint piece
    import bmesh as _bm
    _b = _bm.new(); _b.from_mesh(ball.data)
    ball_h   = (max(v.co.z for v in _b.verts) - min(v.co.z for v in _b.verts)) * scale
    _b.free()
    _s = _bm.new(); _s.from_mesh(socket.data)
    socket_h = (max(v.co.z for v in _s.verts) - min(v.co.z for v in _s.verts)) * scale
    _s.free()

    # --- Adaptive embed depth ---
    # Target 5 mm, but cap at 30 % of the shorter joint piece so the joint
    # is never buried completely inside the mesh (would happen for small rings).
    unit_scale   = bpy.context.scene.unit_settings.scale_length
    embed_target = 0.005 / unit_scale          # 5 mm in scene units
    embed_depth  = min(embed_target, min(ball_h, socket_h) * 0.30)

    # Separation gap = the visible (non-embedded) protrusion of each piece,
    # so the gap is exactly wide enough to fit both joints end-to-end.
    sep_dist = (ball_h - embed_depth) + (socket_h - embed_depth)
    sep_dist = max(sep_dist, 0.0005 / unit_scale)   # floor at ~0.5 mm

    limb_obj = bpy.data.objects.get(bpy.context.scene.get("rc_limb", ""))
    if limb_obj:
        limb_obj.location += -no * sep_dist
        bpy.context.scene["rc_limb_offset"] = float(sep_dist)
        print(f"[VibeCut] limb moved {sep_dist:.4f} units along -no")

    print(f"[VibeCut] embed_depth={embed_depth:.4f}  sep_dist={sep_dist:.4f}")

    # Socket → body side: attachment face embed_depth inside body, cup faces gap
    # Ball   → limb side: attachment face embed_depth inside limb, ball faces gap
    socket_co = co + no * embed_depth
    ball_co   = co - no * (sep_dist + embed_depth)

    print(f"[VibeCut] scale={scale:.6f}  ball_h={ball_h:.4f}  socket_h={socket_h:.4f}  sep={sep_dist:.4f}")
    print(f"[VibeCut] ball_co={[f'{v:.4f}' for v in ball_co]}  socket_co={[f'{v:.4f}' for v in socket_co]}")

    scale_mat = Matrix.Scale(scale, 4)

    # Compose orient + scale into matrix_world in one shot.
    ball.matrix_world   = _orient_matrix(ball_co,    no)  @ scale_mat
    socket.matrix_world = _orient_matrix(socket_co, -no)  @ scale_mat

    print(f"[VibeCut] ball pos= {[f'{v:.4f}' for v in ball.matrix_world.translation]}")
    print(f"[VibeCut] socket pos= {[f'{v:.4f}' for v in socket.matrix_world.translation]}")
    return ball, socket


# ---------------------------------------------------------------------------
# Press-fit helpers
# ---------------------------------------------------------------------------

def _make_cylinder_obj(name, radius, height, collection, segments=48):
    """
    Build a capped cylinder mesh object with its flat base at local Z=0
    and its tip at local Z=height.  The boolean solver needs a clean,
    closed mesh — cap_ends=True gives us that.
    """
    import bmesh as _bm
    bm = _bm.new()
    _bm.ops.create_cone(
        bm, cap_ends=True, cap_tris=False,
        segments=segments, radius1=radius, radius2=radius, depth=height)
    # create_cone centres the cylinder at origin; shift +Z so base sits at Z=0
    dz = height / 2.0
    for v in bm.verts:
        v.co.z += dz
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def place_press_fit(ctx, plane_co_w, plane_no_w, target_radius):
    """
    Procedural compression-fit joint.

    Pin  (UNION      → limb):  cylinder that protrudes from limb toward body.
    Hole (DIFFERENCE → body):  slightly wider/deeper cylinder cut into body.

    target_radius is already scaled by rc_pin_pct in the caller.
    Returns (pin_obj, hole_obj) or (None, None).
    """
    from .geometry import _orient_matrix
    from mathutils import Vector

    scene      = ctx.scene
    unit_scale = scene.unit_settings.scale_length

    pin_r       = target_radius
    pin_depth   = scene.rc_pin_depth    / unit_scale   # total pin protrusion
    tolerance   = scene.rc_pin_tolerance / unit_scale
    hole_r      = pin_r + tolerance
    # Hole is 1 mm deeper than pin so it never bottoms out during assembly
    hole_depth  = pin_depth + (0.001 / unit_scale)

    col = scene.collection

    # Remove any previous press-fit or library previews
    for tag in (_PIN_TAG, _HOLE_TAG, _BALL_TAG, _SOCKET_TAG):
        old = bpy.data.objects.get(tag)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]

    co = Vector(plane_co_w)
    no = Vector(plane_no_w).normalized()

    # Embed the pin base 25 % of its depth into the limb for structural grip.
    # This means the pin protrudes (pin_depth * 0.75) beyond the cut face.
    embed      = pin_depth * 0.25
    pin_height = pin_depth + embed           # total cylinder height
    pin_co     = co - no * embed             # base is embed units behind cut plane

    # Hole opening is flush with the body cut face; cutter goes into body
    hole_co = co

    pin_obj  = _make_cylinder_obj(_PIN_TAG,  pin_r,  pin_height, col)
    hole_obj = _make_cylinder_obj(_HOLE_TAG, hole_r, hole_depth,  col)
    if pin_obj is None or hole_obj is None:
        for o in (pin_obj, hole_obj):
            if o:
                bpy.data.objects.remove(o, do_unlink=True)
        return None, None

    # _orient_matrix: local +Z → world `no` direction.
    # Pin base at pin_co, protrudes along +no (toward body).
    # Hole base at hole_co, cutter goes along +no (into body).
    pin_obj.matrix_world  = _orient_matrix(pin_co,  no)
    hole_obj.matrix_world = _orient_matrix(hole_co, no)

    pin_obj.name  = _PIN_TAG
    hole_obj.name = _HOLE_TAG

    # Re-use the same scene keys as the library workflow so Apply/Cancel
    # operators need no changes.
    scene["rc_joint_ball"]   = pin_obj.name
    scene["rc_joint_socket"] = hole_obj.name

    bpy.ops.object.select_all(action='DESELECT')
    pin_obj.select_set(True)
    hole_obj.select_set(True)
    ctx.view_layer.objects.active = pin_obj

    print(f"[VibeCut] press_fit: pin_r={pin_r:.4f}  hole_r={hole_r:.4f}  "
          f"pin_h={pin_height:.4f}  hole_h={hole_depth:.4f}  embed={embed:.4f}")
    return pin_obj, hole_obj


def apply_press_fit_boolean(ctx):
    """
    Bake transforms then run:
      pin  UNION      → limb   (pin becomes part of limb)
      hole DIFFERENCE → body   (hole cut into body)
    """
    from .geometry import _apply_boolean

    scene    = ctx.scene
    body_obj = bpy.data.objects.get(scene.get("rc_body",         ""))
    limb_obj = bpy.data.objects.get(scene.get("rc_limb",         ""))
    pin_obj  = bpy.data.objects.get(scene.get("rc_joint_ball",   ""))
    hole_obj = bpy.data.objects.get(scene.get("rc_joint_socket", ""))

    if not all([body_obj, limb_obj, pin_obj, hole_obj]):
        print("[VibeCut] apply_press_fit_boolean: missing objects")
        return False

    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (pin_obj, hole_obj):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    print(f"[VibeCut] press_fit boolean: pin UNION → {limb_obj.name}")
    _apply_boolean(limb_obj, pin_obj,  'UNION',      ctx)
    print(f"[VibeCut] press_fit boolean: hole DIFFERENCE → {body_obj.name}")
    _apply_boolean(body_obj, hole_obj, 'DIFFERENCE', ctx)

    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]
    return True


# ---------------------------------------------------------------------------
# Phase 1: Place
# ---------------------------------------------------------------------------

def place_joint(ctx, set_id, plane_co_w, plane_no_w, target_radius,
                ball_name="joint_ball", socket_name="joint_socket"):
    """
    Import, orient, and leave the joint objects in the scene as normal mesh
    objects that the user can freely reposition (G/R/S) before applying.

    Old preview objects are removed on each call.
    Object names are stored in scene custom props for apply_joint_boolean().
    Returns (ball_obj, socket_obj) or (None, None).
    """
    # Clean up any previous unapplied preview
    for tag in (_BALL_TAG, _SOCKET_TAG):
        old = bpy.data.objects.get(tag)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in ctx.scene:
            del ctx.scene[key]

    ball, socket = _prepare_and_orient(
        ctx, set_id, plane_co_w, plane_no_w, target_radius,
        ball_name, socket_name)
    if ball is None:
        return None, None

    # Tag with stable names so Apply can always find them
    ball.name   = _BALL_TAG
    socket.name = _SOCKET_TAG

    # Select both so user can grab + move immediately
    bpy.ops.object.select_all(action='DESELECT')
    ball.select_set(True)
    socket.select_set(True)
    ctx.view_layer.objects.active = ball

    # Store names for the Apply operator
    ctx.scene["rc_joint_ball"]   = ball.name
    ctx.scene["rc_joint_socket"] = socket.name

    return ball, socket


# ---------------------------------------------------------------------------
# Phase 2: Apply
# ---------------------------------------------------------------------------

def apply_joint_boolean(ctx):
    """
    Bake the current position/rotation of the preview joint objects, then run
    UNION booleans onto body and limb.  Reads all names from scene custom props.

    Boolean directions:
    ball  UNION --> limb_obj  (ball protrusion becomes part of limb)
    socket UNION --> body_obj (socket cup becomes part of body)
    Cutter objects are removed by _apply_boolean after the modifier is applied.
    Returns True on success.
    """
    from .geometry import _apply_boolean

    scene      = ctx.scene
    body_name  = scene.get("rc_body",         "")
    limb_name  = scene.get("rc_limb",         "")
    ball_name  = scene.get("rc_joint_ball",   "")
    sock_name  = scene.get("rc_joint_socket", "")

    body_obj   = bpy.data.objects.get(body_name)
    limb_obj   = bpy.data.objects.get(limb_name)
    ball_obj   = bpy.data.objects.get(ball_name)
    socket_obj = bpy.data.objects.get(sock_name)

    print(f"[VibeCut] apply_joint_boolean:")
    print(f"  body={body_name!r}  found={body_obj is not None}")
    print(f"  limb={limb_name!r}  found={limb_obj is not None}")
    print(f"  ball={ball_name!r}  found={ball_obj is not None}")
    print(f"  socket={sock_name!r}  found={socket_obj is not None}")

    if not body_obj or not limb_obj:
        print("[VibeCut] apply_joint_boolean: body or limb not found")
        return False
    if not ball_obj or not socket_obj:
        print("[VibeCut] apply_joint_boolean: preview objects not found")
        return False

    # Bake any user repositioning before the Exact boolean solver runs
    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (ball_obj, socket_obj):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    # ball → limb (ball protrudes toward body, attachment face embedded in limb)
    # socket → body (socket cup opens toward limb, attachment face embedded in body)
    print(f"[VibeCut] boolean: ball UNION into {limb_name!r}")
    _apply_boolean(limb_obj, ball_obj,   'UNION', ctx)
    print(f"[VibeCut] boolean: socket UNION into {body_name!r}")
    _apply_boolean(body_obj, socket_obj, 'UNION', ctx)

    print(f"[VibeCut] apply_joint_boolean: done. "
          f"limb verts={len(limb_obj.data.vertices)}  "
          f"body verts={len(body_obj.data.vertices)}")

    # Clear placement props (cutter objects removed by _apply_boolean)
    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]

    return True


# ---------------------------------------------------------------------------
# Cancel placement
# ---------------------------------------------------------------------------

def cancel_joint_placement(ctx):
    """Remove preview joint objects without applying booleans."""
    scene = ctx.scene
    for key in ("rc_joint_ball", "rc_joint_socket"):
        name = scene.get(key, "")
        obj  = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
        if key in scene:
            del scene[key]


# ---------------------------------------------------------------------------
# One-shot insert (no repositioning step -- scripting / tests)
# ---------------------------------------------------------------------------

def insert_joint(ctx, set_id, body_obj, limb_obj, plane_co_w, plane_no_w,
                 target_radius, ball_name="joint_ball", socket_name="joint_socket"):
    """Place + immediately apply boolean (skips user repositioning)."""
    from .geometry import _apply_boolean, _is_valid
    if not _is_valid(body_obj) or not _is_valid(limb_obj):
        print("[VibeCut] insert_joint: body or limb invalid")
        return False

    ball, socket = _prepare_and_orient(
        ctx, set_id, plane_co_w, plane_no_w, target_radius,
        ball_name, socket_name)
    if ball is None:
        return False

    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (ball, socket):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    _apply_boolean(limb_obj, ball,   'UNION', ctx)
    _apply_boolean(body_obj, socket, 'UNION', ctx)
    return True


# ---------------------------------------------------------------------------
# Screw-top joint (threaded male shaft + female cavity)
# ---------------------------------------------------------------------------
#
# Geometry: male = solid inner cylinder + helical thread ridge with triangular
# cross-section. Female = same shape inflated by `tolerance` radially, used as
# a DIFFERENCE cutter against the body so the body gets matching internal threads.
#
# Workflow mirrors press-fit:
#   place_screw_top()        -> builds male + female previews at the cut plane
#   apply_screw_top_boolean() -> male UNION -> limb (lid), female DIFFERENCE -> body
#
# Build order: lid screws DOWN onto body. Male shaft points DOWN from lid bottom.
# Female cavity opens UP into the body. So in cut-plane local coords:
#   - Limb (top piece): male shaft is built going AWAY from cut plane along -no
#   - Body (bottom piece): female cavity is carved INTO body along +no (from the
#     limb side, into the body)
# But after Ring Cutter, limb_obj sits ABOVE the cut and is moved +no during
# joint placement. So both pieces sit on either side of the cut plane normal.

def _build_thread_mesh(bm, base_r, thread_depth, pitch, n_turns, segments_per_turn=32):
    """Add to bm: an inner solid cylinder + outer helical thread ridge.

    Both volumes are MANIFOLD and slightly overlap radially (root recessed 0.5%
    into the cylinder) so a downstream UNION boolean fuses them into one solid.
    Cylinder base sits at local Z=0; height = n_turns * pitch.
    """
    import bmesh as _bm
    import math

    total_steps = int(n_turns * segments_per_turn)
    height = n_turns * pitch
    dtheta = 2.0 * math.pi / segments_per_turn
    dz_step = pitch / segments_per_turn

    # 1) Inner solid cylinder, base at Z=0, top at Z=height
    res = _bm.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                              segments=segments_per_turn,
                              radius1=base_r, radius2=base_r, depth=height)
    new_verts = res.get("verts", []) if isinstance(res, dict) else []
    # create_cone centers at origin -> lift so base is at z=0
    for v in new_verts:
        v.co.z += height / 2.0

    # 2) Helical thread ridge: triangular cross-section sweeping around
    r_root = base_r * 0.995    # recess slightly so volumes overlap
    r_crest = base_r + thread_depth

    rings = []
    for i in range(total_steps + 1):
        theta = i * dtheta
        z_base = i * dz_step
        cos_t = math.cos(theta); sin_t = math.sin(theta)
        # Triangle spans EXACTLY one pitch so consecutive turns share their
        # boundary verts after remove_doubles
        v_lo = bm.verts.new((r_root  * cos_t, r_root  * sin_t, z_base))
        v_cr = bm.verts.new((r_crest * cos_t, r_crest * sin_t, z_base + 0.5 * pitch))
        v_hi = bm.verts.new((r_root  * cos_t, r_root  * sin_t, z_base + pitch))
        rings.append((v_lo, v_cr, v_hi))

    # Quad strips around triangular cross-section between consecutive rings
    for i in range(total_steps):
        lo_a, cr_a, hi_a = rings[i]
        lo_b, cr_b, hi_b = rings[i + 1]
        for a, b, c, d in ((lo_a, cr_a, cr_b, lo_b),
                           (cr_a, hi_a, hi_b, cr_b),
                           (hi_a, lo_a, lo_b, hi_b)):
            try:
                bm.faces.new([a, b, c, d])
            except ValueError:
                pass

    # End-cap the ridge with triangular faces (open ends would be non-manifold)
    try: bm.faces.new(list(rings[0]))
    except ValueError: pass
    try: bm.faces.new(list(reversed(rings[-1])))
    except ValueError: pass


def _make_thread_obj(name, base_r, thread_depth, pitch, n_turns, collection,
                    segments_per_turn=32, do_union=True):
    """Build a screw-top thread object (manifold solid). Optionally collapses
    the inner cylinder + thread ridge into a single welded volume via a
    self-UNION boolean. Returns the bpy.types.Object.
    """
    import bmesh as _bm

    bm = _bm.new()
    _build_thread_mesh(bm, base_r, thread_depth, pitch, n_turns, segments_per_turn)
    bm.normal_update()
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh); bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    # Cleanup pass
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    if do_union:
        # Self-union: use a self-boolean to merge cylinder + ridge into one volume.
        # The MANIFOLD solver in 4.5+ requires two separate inputs, so we duplicate
        # and use the dup as the cutter (operation=UNION) — produces one welded mesh.
        dup_mesh = obj.data.copy()
        dup_obj = bpy.data.objects.new(name + "_dup", dup_mesh)
        collection.objects.link(dup_obj)

        mod = obj.modifiers.new("SelfUnion", type='BOOLEAN')
        mod.operation = 'UNION'
        mod.solver = 'MANIFOLD'
        mod.object = dup_obj
        try:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod.name)
        except Exception as e:
            print(f"[VibeCut] screw self-union (MANIFOLD) failed: {e} -- trying EXACT")
            try:
                if mod.name in [m.name for m in obj.modifiers]:
                    obj.modifiers.remove(mod)
                mod2 = obj.modifiers.new("SelfUnion", type='BOOLEAN')
                mod2.operation = 'UNION'
                mod2.solver = 'EXACT'
                mod2.object = dup_obj
                bpy.ops.object.modifier_apply(modifier=mod2.name)
            except Exception as e2:
                print(f"[VibeCut] screw self-union failed both solvers: {e2}")
        bpy.data.objects.remove(dup_obj, do_unlink=True)
        if dup_mesh.users == 0:
            bpy.data.meshes.remove(dup_mesh)

    return obj


def place_screw_top(ctx, plane_co_w, plane_no_w, target_radius):
    """Procedural threaded screw-top joint.

    Male  (UNION      -> limb / lid):   threaded shaft protruding from limb toward body
    Female (DIFFERENCE -> body):        slightly wider threaded shaft used as cutter

    target_radius: external male thread base radius (caller usually scales by rc_screw_pct)
    Returns (male_obj, female_obj) or (None, None).
    """
    from .geometry import _orient_matrix
    from mathutils import Vector

    scene = ctx.scene
    unit_scale = scene.unit_settings.scale_length
    # Properties are stored in scene units (NOT mm) for consistency with other tools
    pitch       = scene.rc_screw_pitch    / unit_scale
    depth       = scene.rc_screw_depth    / unit_scale
    n_turns     = max(1, int(scene.rc_screw_turns))
    tolerance   = scene.rc_screw_tolerance / unit_scale
    embed_pct   = max(0.0, min(0.5, scene.rc_screw_embed_pct))

    male_r   = target_radius
    female_r = male_r + tolerance
    shaft_h  = n_turns * pitch

    col = scene.collection

    # Clear previous previews (any flavor)
    for tag in (_PIN_TAG, _HOLE_TAG, _BALL_TAG, _SOCKET_TAG, _SCREW_M_TAG, _SCREW_F_TAG):
        old = bpy.data.objects.get(tag)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]

    co = Vector(plane_co_w)
    no = Vector(plane_no_w).normalized()

    # Convention:
    #   `body` lives on the +no side of the cut plane -> becomes the LID (gets male)
    #   `limb` lives on the -no side of the cut plane -> becomes the CONTAINER (gets female)
    # Male shaft straddles the cut plane:
    #   embed_pct fraction sits INSIDE body (above cut) -> fuses to lid on UNION
    #   (1 - embed_pct) fraction protrudes BELOW cut   -> the visible stem
    # _orient_matrix aligns local +Z with `no`; built mesh has base at local Z=0
    # and tip at local Z=shaft_h.
    shaft_below = shaft_h * (1.0 - embed_pct)
    male_co     = co - no * shaft_below
    # Female sits at the same axial position but bigger and slightly recessed so the
    # cavity opening extends past the cut plane -- guarantees clean boolean even with
    # small numerical offsets.
    female_recess = pitch * 0.5
    female_co     = co - no * (shaft_below + female_recess)

    male_obj = _make_thread_obj(_SCREW_M_TAG, male_r,   depth, pitch, n_turns, col,
                                segments_per_turn=32, do_union=True)
    female_obj = _make_thread_obj(_SCREW_F_TAG, female_r, depth, pitch, n_turns, col,
                                  segments_per_turn=32, do_union=True)
    if male_obj is None or female_obj is None:
        for o in (male_obj, female_obj):
            if o:
                bpy.data.objects.remove(o, do_unlink=True)
        return None, None

    # Local +Z axis -> world `no` direction (toward body across the cut)
    male_obj.matrix_world   = _orient_matrix(male_co,   no)
    female_obj.matrix_world = _orient_matrix(female_co, no)

    male_obj.name   = _SCREW_M_TAG
    female_obj.name = _SCREW_F_TAG

    # Reuse the same scene keys as press-fit so Apply/Cancel just work
    scene["rc_joint_ball"]   = male_obj.name
    scene["rc_joint_socket"] = female_obj.name

    bpy.ops.object.select_all(action='DESELECT')
    male_obj.select_set(True)
    female_obj.select_set(True)
    ctx.view_layer.objects.active = male_obj

    print(f"[VibeCut] screw_top: male_r={male_r:.4f}  female_r={female_r:.4f}  "
          f"pitch={pitch:.4f}  turns={n_turns}  shaft_h={shaft_h:.4f}  "
          f"shaft_below={shaft_below:.4f}  embed_pct={embed_pct:.2f}")
    return male_obj, female_obj


def apply_screw_top_boolean(ctx):
    """Bake placement then:
        male   UNION      -> limb (lid gets external threaded stem)
        female DIFFERENCE -> body (body gets internal threaded socket)
    """
    from .geometry import _apply_boolean

    scene      = ctx.scene
    body_obj   = bpy.data.objects.get(scene.get("rc_body",         ""))
    limb_obj   = bpy.data.objects.get(scene.get("rc_limb",         ""))
    male_obj   = bpy.data.objects.get(scene.get("rc_joint_ball",   ""))
    female_obj = bpy.data.objects.get(scene.get("rc_joint_socket", ""))

    if not all([body_obj, limb_obj, male_obj, female_obj]):
        print("[VibeCut] apply_screw_top_boolean: missing objects")
        return False

    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (male_obj, female_obj):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    # body = lid (gets external male thread), limb = container (gets female cavity).
    # This is intentionally OPPOSITE to press-fit which assumes the moving piece is
    # the limb -- for a screw-top container the static container body holds the cavity.
    print(f"[VibeCut] screw_top boolean: male UNION -> {body_obj.name} (lid)")
    _apply_boolean(body_obj, male_obj,   'UNION',      ctx)
    print(f"[VibeCut] screw_top boolean: female DIFFERENCE -> {limb_obj.name} (container)")
    _apply_boolean(limb_obj, female_obj, 'DIFFERENCE', ctx)

    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]
    return True


# ---------------------------------------------------------------------------
# Property registration
# ---------------------------------------------------------------------------

def register_props():
    bpy.types.Scene.rc_joint_file = bpy.props.EnumProperty(
        name        = "Joint",
        description = "Joint pair from the Joints/ library",
        items       = joint_enum_items,
    )
    # Screw-top thread parameters (stored in scene units; scaled by unit_scale at use)
    bpy.types.Scene.rc_screw_pct = bpy.props.FloatProperty(
        name="Thread Diameter",
        description="External thread base radius as fraction of ring radius",
        default=0.55, min=0.15, max=0.85, step=1, precision=2, subtype='FACTOR')
    bpy.types.Scene.rc_screw_pitch = bpy.props.FloatProperty(
        name="Pitch",
        description="Vertical distance per thread turn (metres)",
        default=0.003, min=0.0005, max=0.02, step=0.01, precision=4)
    bpy.types.Scene.rc_screw_depth = bpy.props.FloatProperty(
        name="Thread Depth",
        description="Radial height of the thread ridge (metres)",
        default=0.0008, min=0.0002, max=0.005, step=0.01, precision=4)
    bpy.types.Scene.rc_screw_turns = bpy.props.IntProperty(
        name="Turns",
        description="Number of thread revolutions",
        default=3, min=1, max=12)
    bpy.types.Scene.rc_screw_tolerance = bpy.props.FloatProperty(
        name="Fit Tolerance",
        description="Extra radius added to female for printable clearance (metres)",
        default=0.0003, min=0.0, max=0.002, step=0.01, precision=4)
    bpy.types.Scene.rc_screw_embed_pct = bpy.props.FloatProperty(
        name="Lid Embed",
        description="Fraction of shaft height that sits INSIDE the lid (vs protruding)",
        default=0.30, min=0.0, max=0.5, step=1, precision=2, subtype='FACTOR')


def unregister_props():
    for attr in ("rc_joint_file", "rc_screw_pct", "rc_screw_pitch",
                 "rc_screw_depth", "rc_screw_turns", "rc_screw_tolerance",
                 "rc_screw_embed_pct"):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
