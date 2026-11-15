import bpy
import os
from mathutils import Vector

JOINTS_DIR = os.path.join(os.path.dirname(__file__), "Joints")

_BALL_TAG   = "rc_joint_ball_preview"
_SOCKET_TAG = "rc_joint_socket_preview"
_PIN_TAG    = "rc_press_pin_preview"
_HOLE_TAG   = "rc_press_hole_preview"
_SCREW_M_TAG = "rc_screw_male_preview"
_SCREW_F_TAG = "rc_screw_female_preview"


def _find_joint_sets():
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


def _import_object(blend_path, preferred_name, collection):
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


def _get_ref_radius(obj):
    if "joint_radius" in obj.keys():
        return float(obj["joint_radius"])

    import bmesh
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
    offset = -min(world_zs)
    print(f"[VibeCut] _get_stem_offset({obj.name}): min_world_z={min(world_zs):.4f}  offset={offset:.4f}")
    return offset


def _center_mesh_xy(obj):
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
    cz = min(zs)
    for v in bm.verts:
        v.co.x -= cx
        v.co.y -= cy
        v.co.z -= cz
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.free()
    print(f"[VibeCut] _center_mesh_xy({obj.name}): shifted XY=({cx:.4f},{cy:.4f})  Z={cz:.4f}")


def _prepare_and_orient(ctx, set_id, plane_co_w, plane_no_w, target_radius,
                        ball_name="joint_ball", socket_name="joint_socket"):
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

    ref_r         = _get_ref_radius(socket)
    ball_ref_r    = _get_ref_radius(ball)
    ball_stem_off = _get_stem_offset(ball)
    sock_stem_off = _get_stem_offset(socket)

    print(f"[VibeCut] PRE-normalize: socket_ref_r={ref_r:.4f}  ball_ref_r={ball_ref_r:.4f}  "
          f"ball_stem_off={ball_stem_off:.4f}  sock_stem_off={sock_stem_off:.4f}")

    if ref_r < 1e-9:
        print("[VibeCut] _prepare_and_orient: zero socket reference radius.")
        for o in (ball, socket):
            bpy.data.objects.remove(o, do_unlink=True)
        return None, None

    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (ball, socket):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    for jobj in (ball, socket):
        _center_mesh_xy(jobj)
    scale = target_radius / ref_r
    co    = Vector(plane_co_w)
    no    = Vector(plane_no_w).normalized()

    import bmesh as _bm
    _b = _bm.new(); _b.from_mesh(ball.data)
    ball_h   = (max(v.co.z for v in _b.verts) - min(v.co.z for v in _b.verts)) * scale
    _b.free()
    _s = _bm.new(); _s.from_mesh(socket.data)
    socket_h = (max(v.co.z for v in _s.verts) - min(v.co.z for v in _s.verts)) * scale
    _s.free()

    unit_scale   = bpy.context.scene.unit_settings.scale_length
    embed_target = 0.005 / unit_scale
    embed_depth  = min(embed_target, min(ball_h, socket_h) * 0.30)

    sep_dist = (ball_h - embed_depth) + (socket_h - embed_depth)
    sep_dist = max(sep_dist, 0.0005 / unit_scale)

    limb_obj = bpy.data.objects.get(bpy.context.scene.get("rc_limb", ""))
    if limb_obj:
        limb_obj.location += -no * sep_dist
        bpy.context.scene["rc_limb_offset"] = float(sep_dist)
        print(f"[VibeCut] limb moved {sep_dist:.4f} units along -no")

    print(f"[VibeCut] embed_depth={embed_depth:.4f}  sep_dist={sep_dist:.4f}")

    socket_co = co + no * embed_depth
    ball_co   = co - no * (sep_dist + embed_depth)

    print(f"[VibeCut] scale={scale:.6f}  ball_h={ball_h:.4f}  socket_h={socket_h:.4f}  sep={sep_dist:.4f}")
    print(f"[VibeCut] ball_co={[f'{v:.4f}' for v in ball_co]}  socket_co={[f'{v:.4f}' for v in socket_co]}")

    scale_mat = Matrix.Scale(scale, 4)

    ball.matrix_world   = _orient_matrix(ball_co,    no)  @ scale_mat
    socket.matrix_world = _orient_matrix(socket_co, -no)  @ scale_mat

    print(f"[VibeCut] ball pos= {[f'{v:.4f}' for v in ball.matrix_world.translation]}")
    print(f"[VibeCut] socket pos= {[f'{v:.4f}' for v in socket.matrix_world.translation]}")
    return ball, socket


def _make_cylinder_obj(name, radius, height, collection, segments=48):
    import bmesh as _bm
    bm = _bm.new()
    _bm.ops.create_cone(
        bm, cap_ends=True, cap_tris=False,
        segments=segments, radius1=radius, radius2=radius, depth=height)
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
    from .geometry import _orient_matrix
    from mathutils import Vector

    scene      = ctx.scene
    unit_scale = scene.unit_settings.scale_length

    pin_r       = target_radius
    pin_depth   = scene.rc_pin_depth    / unit_scale
    tolerance   = scene.rc_pin_tolerance / unit_scale
    hole_r      = pin_r + tolerance
    hole_depth  = pin_depth + (0.001 / unit_scale)

    col = scene.collection

    for tag in (_PIN_TAG, _HOLE_TAG, _BALL_TAG, _SOCKET_TAG):
        old = bpy.data.objects.get(tag)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]

    co = Vector(plane_co_w)
    no = Vector(plane_no_w).normalized()

    embed      = pin_depth * 0.25
    pin_height = pin_depth + embed
    pin_co     = co - no * embed

    hole_co = co

    pin_obj  = _make_cylinder_obj(_PIN_TAG,  pin_r,  pin_height, col)
    hole_obj = _make_cylinder_obj(_HOLE_TAG, hole_r, hole_depth,  col)
    if pin_obj is None or hole_obj is None:
        for o in (pin_obj, hole_obj):
            if o:
                bpy.data.objects.remove(o, do_unlink=True)
        return None, None

    pin_obj.matrix_world  = _orient_matrix(pin_co,  no)
    hole_obj.matrix_world = _orient_matrix(hole_co, no)

    pin_obj.name  = _PIN_TAG
    hole_obj.name = _HOLE_TAG

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


def place_joint(ctx, set_id, plane_co_w, plane_no_w, target_radius,
                ball_name="joint_ball", socket_name="joint_socket"):
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

    ball.name   = _BALL_TAG
    socket.name = _SOCKET_TAG

    bpy.ops.object.select_all(action='DESELECT')
    ball.select_set(True)
    socket.select_set(True)
    ctx.view_layer.objects.active = ball

    ctx.scene["rc_joint_ball"]   = ball.name
    ctx.scene["rc_joint_socket"] = socket.name

    return ball, socket


def apply_joint_boolean(ctx):
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

    bpy.ops.object.select_all(action='DESELECT')
    for jobj in (ball_obj, socket_obj):
        jobj.select_set(True)
        ctx.view_layer.objects.active = jobj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        jobj.select_set(False)

    print(f"[VibeCut] boolean: ball UNION into {limb_name!r}")
    _apply_boolean(limb_obj, ball_obj,   'UNION', ctx)
    print(f"[VibeCut] boolean: socket UNION into {body_name!r}")
    _apply_boolean(body_obj, socket_obj, 'UNION', ctx)

    print(f"[VibeCut] apply_joint_boolean: done. "
          f"limb verts={len(limb_obj.data.vertices)}  "
          f"body verts={len(body_obj.data.vertices)}")

    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]

    return True


def cancel_joint_placement(ctx):
    scene = ctx.scene
    for key in ("rc_joint_ball", "rc_joint_socket"):
        name = scene.get(key, "")
        obj  = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
        if key in scene:
            del scene[key]


def insert_joint(ctx, set_id, body_obj, limb_obj, plane_co_w, plane_no_w,
                 target_radius, ball_name="joint_ball", socket_name="joint_socket"):
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


def _build_thread_mesh(bm, base_r, thread_depth, pitch, n_turns, segments_per_turn=32):
    import bmesh as _bm
    import math

    total_steps = int(n_turns * segments_per_turn)
    height = n_turns * pitch
    dtheta = 2.0 * math.pi / segments_per_turn
    dz_step = pitch / segments_per_turn

    res = _bm.ops.create_cone(bm, cap_ends=True, cap_tris=False,
                              segments=segments_per_turn,
                              radius1=base_r, radius2=base_r, depth=height)
    new_verts = res.get("verts", []) if isinstance(res, dict) else []
    for v in new_verts:
        v.co.z += height / 2.0

    r_root = base_r * 0.995
    r_crest = base_r + thread_depth

    rings = []
    for i in range(total_steps + 1):
        theta = i * dtheta
        z_base = i * dz_step
        cos_t = math.cos(theta); sin_t = math.sin(theta)
        v_lo = bm.verts.new((r_root  * cos_t, r_root  * sin_t, z_base))
        v_cr = bm.verts.new((r_crest * cos_t, r_crest * sin_t, z_base + 0.5 * pitch))
        v_hi = bm.verts.new((r_root  * cos_t, r_root  * sin_t, z_base + pitch))
        rings.append((v_lo, v_cr, v_hi))

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

    try: bm.faces.new(list(rings[0]))
    except ValueError: pass
    try: bm.faces.new(list(reversed(rings[-1])))
    except ValueError: pass


def _make_thread_obj(name, base_r, thread_depth, pitch, n_turns, collection,
                    segments_per_turn=32, do_union=True):
    import bmesh as _bm

    bm = _bm.new()
    _build_thread_mesh(bm, base_r, thread_depth, pitch, n_turns, segments_per_turn)
    bm.normal_update()
    mesh = bpy.data.meshes.new(name + "_mesh")
    bm.to_mesh(mesh); bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    if do_union:
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
    from .geometry import _orient_matrix
    from mathutils import Vector

    scene = ctx.scene
    unit_scale = scene.unit_settings.scale_length
    pitch       = scene.rc_screw_pitch    / unit_scale
    depth       = scene.rc_screw_depth    / unit_scale
    n_turns     = max(1, int(scene.rc_screw_turns))
    tolerance   = scene.rc_screw_tolerance / unit_scale
    embed_pct   = max(0.0, min(0.5, scene.rc_screw_embed_pct))

    male_r   = target_radius
    female_r = male_r + tolerance
    shaft_h  = n_turns * pitch

    col = scene.collection

    for tag in (_PIN_TAG, _HOLE_TAG, _BALL_TAG, _SOCKET_TAG, _SCREW_M_TAG, _SCREW_F_TAG):
        old = bpy.data.objects.get(tag)
        if old:
            bpy.data.objects.remove(old, do_unlink=True)
    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]

    co = Vector(plane_co_w)
    no = Vector(plane_no_w).normalized()

    shaft_below = shaft_h * (1.0 - embed_pct)
    male_co     = co - no * shaft_below
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

    male_obj.matrix_world   = _orient_matrix(male_co,   no)
    female_obj.matrix_world = _orient_matrix(female_co, no)

    male_obj.name   = _SCREW_M_TAG
    female_obj.name = _SCREW_F_TAG

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

    print(f"[VibeCut] screw_top boolean: male UNION -> {body_obj.name} (lid)")
    _apply_boolean(body_obj, male_obj,   'UNION',      ctx)
    print(f"[VibeCut] screw_top boolean: female DIFFERENCE -> {limb_obj.name} (container)")
    _apply_boolean(limb_obj, female_obj, 'DIFFERENCE', ctx)

    for key in ("rc_joint_ball", "rc_joint_socket"):
        if key in scene:
            del scene[key]
    return True


def register_props():
    bpy.types.Scene.rc_joint_file = bpy.props.EnumProperty(
        name        = "Joint",
        description = "Joint pair from the Joints/ library",
        items       = joint_enum_items,
    )
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
