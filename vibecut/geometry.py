import bpy
import bmesh
import math
import time
import numpy as np
import bpy_extras.view3d_utils

from mathutils import Vector, Quaternion, Matrix

HOVER  = 0
LOCKED = 1


def _is_valid(obj):
    if obj is None:
        return False
    try:
        _ = obj.name
        return obj.name in bpy.data.objects
    except ReferenceError:
        return False

_state = dict(
    active    = False,
    phase     = HOVER,
    ring      = [],
    cross     = [],
    plane_co  = None,
    plane_no  = None,
    radius    = 1.0,
    obj       = None,
    limb_axis = None,
    opt_fill  = True,
    opt_neg   = False,
    opt_pos   = False,
    opt_sep   = True,
    cross_dirty = False,
)


_mesh_cache     = {}
_MESH_CACHE_MAX = 2


def _invalidate_mesh_cache(obj=None):
    if obj is None:
        _mesh_cache.clear()
        return
    _mesh_cache.pop(getattr(obj, "name", obj), None)


def _rc_depsgraph_invalidate(scene, depsgraph):
    if not _mesh_cache:
        return
    for upd in depsgraph.updates:
        if getattr(upd, "is_updated_geometry", False):
            _mesh_cache.pop(getattr(upd.id, "name", None), None)
            orig = getattr(upd.id, "original", None)
            if orig is not None:
                _mesh_cache.pop(getattr(orig, "name", None), None)


def _get_mesh_cache(obj):
    me = obj.data
    if me.is_editmode:
        me.update()

    nv  = len(me.vertices)
    ne  = len(me.edges)
    key = (nv, ne)

    hit = _mesh_cache.get(obj.name)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2], hit[3]

    vflat = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get('co', vflat)
    verts = vflat.reshape(nv, 3)

    eflat = np.empty(ne * 2, dtype=np.int32)
    me.edges.foreach_get('vertices', eflat)
    pairs = eflat.reshape(ne, 2)
    e0 = np.ascontiguousarray(pairs[:, 0])
    e1 = np.ascontiguousarray(pairs[:, 1])

    if len(_mesh_cache) >= _MESH_CACHE_MAX:
        _mesh_cache.pop(next(iter(_mesh_cache)))
    _mesh_cache[obj.name] = (key, verts, e0, e1)
    return verts, e0, e1


def _pca_axis(obj, hit_w, radius):
    n = len(obj.data.vertices)
    if n < 6:
        return None
    pts, _e0, _e1 = _get_mesh_cache(obj)
    mat  = np.array(obj.matrix_world.to_3x3(), dtype=np.float32)
    loc  = np.array(obj.matrix_world.translation, dtype=np.float32)
    wpts = pts @ mat.T + loc
    hit  = np.array([hit_w.x, hit_w.y, hit_w.z], dtype=np.float32)
    near = wpts[np.linalg.norm(wpts - hit, axis=1) < radius * 1.5]
    if len(near) < 6:
        return None
    cen = near.mean(axis=0)
    cov = (near - cen).T @ (near - cen)
    v   = np.array([1.0, 0.5, 0.3], dtype=np.float64)
    v  /= np.linalg.norm(v)
    for _ in range(60):
        v2 = cov @ v
        nm = np.linalg.norm(v2)
        if nm < 1e-12:
            return None
        v = v2 / nm
    return Vector(v.tolist()).normalized()


def _ring_verts(center, normal, radius, segs=80):
    n   = normal.normalized()
    ref = Vector((0, 0, 1)) if abs(n.dot(Vector((0, 0, 1)))) < 0.9 else Vector((1, 0, 0))
    r   = n.cross(ref).normalized()
    u   = n.cross(r).normalized()
    return [
        (center + radius * (math.cos(a) * r + math.sin(a) * u)).to_tuple()
        for a in (i / segs * math.tau for i in range(segs))
    ]


def _cross_section(obj, co_w, no_w):
    if not _is_valid(obj):
        return [], co_w

    mi       = obj.matrix_world.inverted()
    mw       = obj.matrix_world
    co_l     = mi @ co_w
    no_l     = (mi.to_3x3() @ no_w).normalized()
    radius_l = (mi.to_3x3() @ Vector((_state["radius"] * 1.05, 0, 0))).length

    verts, e0, e1 = _get_mesh_cache(obj)
    if e0.size == 0:
        return [], co_w

    c   = np.array((co_l.x, co_l.y, co_l.z), dtype=np.float32)
    nrm = np.array((no_l.x, no_l.y, no_l.z), dtype=np.float32)

    d  = verts @ nrm
    d -= float(np.dot(c, nrm))

    sgn = d < 0.0
    idx = np.flatnonzero(sgn[e0] ^ sgn[e1])
    if idx.size == 0:
        return [], co_w

    j0 = e0[idx]
    j1 = e1[idx]
    v0 = verts[j0]
    v1 = verts[j1]
    r0 = v0 - c
    r1 = v1 - c
    rr = radius_l * radius_l
    keep = (np.einsum('ij,ij->i', r0, r0) <= rr) | \
           (np.einsum('ij,ij->i', r1, r1) <= rr)
    if not keep.any():
        return [], co_w

    v0 = v0[keep]
    v1 = v1[keep]
    d0 = d[j0[keep]]
    d1 = d[j1[keep]]

    t = (d0 / (d0 - d1)).astype(np.float32)[:, None]
    P = v0 + t * (v1 - v0)

    M  = np.array(mw.to_3x3(), dtype=np.float32)
    T  = np.array(mw.translation, dtype=np.float32)
    Pw = P @ M.T + T

    if Pw.shape[0] <= 2:
        return [tuple(p) for p in Pw.tolist()], co_w

    cen = Pw.mean(axis=0)

    nw  = no_w.normalized()
    ref = Vector((0, 0, 1)) if abs(nw.dot(Vector((0, 0, 1)))) < 0.9 else Vector((1, 0, 0))
    rx  = nw.cross(ref).normalized()
    ry  = nw.cross(rx).normalized()

    rel = Pw - cen
    ang = np.arctan2(rel @ np.array((ry.x, ry.y, ry.z), dtype=np.float32),
                     rel @ np.array((rx.x, rx.y, rx.z), dtype=np.float32))
    order = np.argsort(ang, kind='stable')

    pts = [tuple(p) for p in Pw[order].tolist()]
    pts.append(pts[0])
    return pts, Vector(cen.tolist())


_CROSS_THROTTLE = 0.06
_cross_last     = 0.0
_cross_cost     = 0.0


def _compute_cross():
    global _cross_last, _cross_cost
    s = _state
    t0 = time.perf_counter()
    pts, _ = _cross_section(s["obj"], s["plane_co"], s["plane_no"])
    _cross_cost      = time.perf_counter() - t0
    s["cross"]       = pts
    s["cross_dirty"] = False
    _cross_last      = time.perf_counter()


def _flush_cross():
    s = _state
    if not s.get("cross_dirty"):
        return
    if s["phase"] != LOCKED or not _is_valid(s["obj"]) \
            or s["plane_co"] is None or s["plane_no"] is None:
        s["cross_dirty"] = False
        return
    _compute_cross()


def _rebuild(defer=False):
    s = _state
    if s["plane_co"] is None or s["plane_no"] is None:
        return
    if not _is_valid(s["obj"]):
        s["active"] = False
        s["obj"]    = None
        return
    s["ring"] = _ring_verts(s["plane_co"], s["plane_no"], s["radius"])
    if s["phase"] == LOCKED:
        interval = max(_CROSS_THROTTLE, 3.0 * _cross_cost)
        if defer and (time.perf_counter() - _cross_last) < interval:
            s["cross_dirty"] = True
            return
        _compute_cross()
    else:
        s["cross"]       = []
        s["cross_dirty"] = False


def _lock_phase_compute():
    s = _state
    if not _is_valid(s["obj"]) or s["plane_co"] is None or s["plane_no"] is None:
        return
    _compute_cross()


def _fill_loops(bm, cut_edges, use_triangles=False):
    from collections import defaultdict
    vert_to_edges = defaultdict(list)
    for e in cut_edges:
        vert_to_edges[e.verts[0].index].append(e)
        vert_to_edges[e.verts[1].index].append(e)
    visited = set()
    loops = []
    for start in cut_edges:
        if id(start) in visited:
            continue
        loop = []
        stack = [start]
        while stack:
            e = stack.pop()
            if id(e) in visited:
                continue
            visited.add(id(e))
            loop.append(e)
            for v in e.verts:
                for nb in vert_to_edges[v.index]:
                    if id(nb) not in visited:
                        stack.append(nb)
        if loop:
            loops.append(loop)
    for loop_edges in loops:
        if use_triangles:
            try:
                bmesh.ops.triangle_fill(bm, edges=loop_edges, use_beauty=True)
                continue
            except Exception:
                pass
        try:
            bmesh.ops.edgeloop_fill(bm, edges=loop_edges)
        except Exception:
            try:
                bmesh.ops.contextual_create(bm, geom=loop_edges)
            except Exception:
                pass


def _fill_open_boundary(ctx, obj, plane_co_w=None, plane_no_w=None, radius=None,
                        plane_tol=None, use_triangles=False):
    if not _is_valid(obj):
        return
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bm = bmesh.from_edit_mesh(obj.data)
        bm.edges.ensure_lookup_table()

        all_boundary = [e for e in bm.edges if len(e.link_faces) == 1]
        if radius is not None and plane_co_w is not None:
            mw   = obj.matrix_world
            mi   = mw.inverted()
            co_l = mi @ plane_co_w
            r_l  = (mi.to_3x3() @ Vector((radius * 1.30, 0, 0))).length
            no_l = ((mi.to_3x3() @ plane_no_w).normalized()
                    if plane_no_w is not None else None)
            if plane_tol is not None:
                tol = (mi.to_3x3() @ Vector((plane_tol, 0, 0))).length
            else:
                tol = max(r_l * 0.05, 1e-4)
            boundary = []
            for e in all_boundary:
                v0 = e.verts[0].co
                v1 = e.verts[1].co
                if no_l is not None:
                    d0 = (v0 - co_l).dot(no_l)
                    d1 = (v1 - co_l).dot(no_l)
                    if abs(d0) > tol or abs(d1) > tol:
                        continue
                if (v0 - co_l).length > r_l and (v1 - co_l).length > r_l:
                    continue
                boundary.append(e)
            print(f"[VibeCut] _fill_open_boundary({obj.name}): "
                  f"{len(boundary)}/{len(all_boundary)} boundary edges after filter")
        else:
            boundary = all_boundary
            print(f"[VibeCut] _fill_open_boundary({obj.name}): "
                  f"{len(boundary)} boundary edges (no filter)")

        if boundary:
            _fill_loops(bm, boundary, use_triangles=use_triangles)
            bmesh.update_edit_mesh(obj.data)
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')


def _do_tilt(rv3d, direction, defer=False):
    if _state["plane_no"] is None:
        return
    axis = (rv3d.view_rotation @ Vector((0, 0, -1))).normalized()
    q    = Quaternion(axis, 0.05 * direction)
    _state["plane_no"] = (q @ _state["plane_no"]).normalized()
    _rebuild(defer)


def _nudge_plane(rv3d, mode, direction, defer=False):
    if _state["plane_co"] is None or _state["plane_no"] is None:
        return
    step = _state["radius"] * 0.04
    no   = _state["plane_no"].normalized()
    if mode == 'NORMAL':
        vec = no * step * direction
    else:
        cam_right = (rv3d.view_rotation @ Vector((1, 0, 0))).normalized()
        cam_right = (cam_right - cam_right.dot(no) * no).normalized()
        vec       = cam_right * step * direction
    _state["plane_co"] = _state["plane_co"] + vec
    _rebuild(defer)


def _snap_to_axis(axis_vec, rv3d=None):
    _state["plane_no"] = axis_vec.normalized()
    _rebuild()


def _orient_matrix(centroid, normal):
    no  = normal.normalized()
    ref = Vector((0, 0, 1)) if abs(no.dot(Vector((0, 0, 1)))) < 0.9 else Vector((1, 0, 0))
    rx  = no.cross(ref).normalized()
    ry  = no.cross(rx).normalized()
    rot = Matrix(((rx.x, ry.x, no.x),
                  (rx.y, ry.y, no.y),
                  (rx.z, ry.z, no.z))).to_4x4()
    rot.translation = centroid
    return rot


def _bm_to_obj(name, bm, collection):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _apply_boolean(target_obj, cutter_obj, operation='DIFFERENCE', ctx=None):
    cutter_obj.hide_set(False)
    cutter_obj.hide_render = False
    active_ctx = ctx if ctx is not None else bpy.context
    active_ctx.view_layer.objects.active = target_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    mod           = target_obj.modifiers.new('_RCJointBool', 'BOOLEAN')
    mod.operation = operation
    mod.object    = cutter_obj
    mod.solver    = 'EXACT'
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as e:
        print(f"[VibeCut] Boolean {operation} failed: {e}")
        target_obj.modifiers.remove(mod)
    bpy.data.objects.remove(cutter_obj, do_unlink=True)


def _safe_depth(scene, depsgraph, origin, direction, max_dist=500.0):
    ok, loc, _, _, _, _ = scene.ray_cast(
        depsgraph, origin + direction * 1e-3, direction)
    return (loc - origin).length if ok else max_dist


def _separate_halves(ctx, obj, plane_co_w, plane_no_w):
    orig_name = obj.name
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    mi    = obj.matrix_world.inverted()
    co_l  = mi @ plane_co_w
    no_l  = (mi.to_3x3() @ plane_no_w).normalized()

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    bpy.ops.mesh.select_all(action='DESELECT')

    best_v = None
    best_dist = 0.0
    for v in bm.verts:
        dist = (v.co - co_l).dot(no_l)
        if dist > best_dist:
            best_dist = dist
            best_v = v

    if best_v is not None and best_dist > 1e-4:
        best_v.select = True

    bmesh.update_edit_mesh(obj.data)

    if best_v is not None and best_dist > 1e-4:
        bpy.ops.mesh.select_linked()
        print(f"[VibeCut] _separate_halves: topological split successful (dist={best_dist:.4f})")
    else:
        print("[VibeCut] _separate_halves: no positive vertices found, separation will bypass")

    before = {o.name for o in bpy.data.objects}
    try:
        if best_v is not None and best_dist > 1e-4:
            bpy.ops.mesh.separate(type='SELECTED')
    except Exception as e:
        print(f"[VibeCut] _separate_halves separate() error: {e}")

    bpy.ops.object.mode_set(mode='OBJECT')

    after     = {o.name for o in bpy.data.objects}
    new_names = after - before
    if not new_names:
        obj.name = orig_name + "_body"
        return obj, None

    orig_obj = bpy.data.objects.get(orig_name)
    new_obj  = bpy.data.objects.get(list(new_names)[0])

    plane_no = plane_no_w.normalized()
    def bbox_cen(o):
        return Vector([
            sum((o.matrix_world @ Vector(c))[i] for c in o.bound_box) / 8
            for i in range(3)])

    pieces = [p for p in [orig_obj, new_obj] if p is not None]
    pieces.sort(key=lambda p: (bbox_cen(p) - plane_co_w).dot(plane_no), reverse=True)

    body_obj = pieces[0] if len(pieces) > 0 else None
    limb_obj = pieces[1] if len(pieces) > 1 else None

    if body_obj: body_obj.name = orig_name + "_body"
    if limb_obj: limb_obj.name = orig_name + "_limb"
    return body_obj, limb_obj


def _execute_cut(ctx):
    s   = _state
    obj = s["obj"]
    if obj is None:
        return
    _invalidate_mesh_cache()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    mi = obj.matrix_world.inverted()
    co = mi @ s["plane_co"]
    no = (mi.to_3x3() @ s["plane_no"]).normalized()

    mw   = obj.matrix_world
    co_w = s["plane_co"]
    no_w = s["plane_no"].normalized()
    scene = bpy.context.scene

    tagged_verts = set()
    ring_radius = s["radius"] * 1.20

    if scene.rc_cut_mode == 'DISK':
        depth = scene.rc_disk_thickness / 2.0
        for v in bm.verts:
            offset = (mw @ v.co) - co_w
            axial = offset.dot(no_w)
            if abs(axial) <= depth:
                radial = (offset - no_w * axial).length
                if radial <= ring_radius:
                    tagged_verts.add(v)
    else:
        max_dist = ring_radius
        for v in bm.verts:
            if ((mw @ v.co) - co_w).length <= max_dist:
                tagged_verts.add(v)

    ring_faces = [f for f in bm.faces if any(v in tagged_verts for v in f.verts)]
    if not ring_faces:
        print("[VibeCut] Warning: no faces within bounds — using full mesh")
        ring_faces = list(bm.faces)

    ring_verts = set()
    ring_edges = set()
    for f in ring_faces:
        for v in f.verts: ring_verts.add(v)
        for e in f.edges: ring_edges.add(e)
    local_geom = list(ring_verts) + list(ring_edges) + ring_faces

    result = bmesh.ops.bisect_plane(
        bm, geom=local_geom, plane_co=co, plane_no=no,
        clear_inner=s["opt_neg"], clear_outer=s["opt_pos"], dist=1e-5)

    cut_edges = [e for e in result["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]

    if not s["opt_sep"] and cut_edges and s["opt_fill"]:
        _fill_loops(bm, cut_edges)

    if s["opt_sep"] and cut_edges:
        bmesh.ops.split_edges(bm, edges=cut_edges)

    bmesh.update_edit_mesh(obj.data)
    bm.free()

    bpy.ops.object.mode_set(mode='OBJECT')

    plane_co_w = s["plane_co"]
    plane_no_w = s["plane_no"]

    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    d_body = _safe_depth(scene, depsgraph, plane_co_w,  plane_no_w)
    d_limb = _safe_depth(scene, depsgraph, plane_co_w, -plane_no_w)

    body_name = ""
    limb_name = ""
    if s["opt_sep"]:
        print("[VibeCut] _execute_cut: separating halves...")
        body_obj, limb_obj = _separate_halves(ctx, obj, plane_co_w, plane_no_w)
        if body_obj:
            body_name = body_obj.name
        if limb_obj:
            limb_name = limb_obj.name
        if s["opt_fill"]:
            print("[VibeCut] _execute_cut: filling cap on body...")
            if body_obj and _is_valid(body_obj):
                _fill_open_boundary(ctx, body_obj, plane_co_w, plane_no_w, s["radius"])
            print("[VibeCut] _execute_cut: filling cap on limb...")
            if limb_obj and _is_valid(limb_obj):
                _fill_open_boundary(ctx, limb_obj, plane_co_w, plane_no_w, s["radius"])
    print("[VibeCut] _execute_cut: done")

    scene["rc_plane_co"] = list(plane_co_w)
    scene["rc_plane_no"] = list(plane_no_w)
    scene["rc_radius"]   = s["radius"]
    scene["rc_d_body"]   = float(d_body)
    scene["rc_d_limb"]   = float(d_limb)
    scene["rc_body"]     = body_name
    scene["rc_limb"]     = limb_name


def _measure_cut_boundary(obj, plane_no_w):
    mw   = obj.matrix_world
    mi   = mw.inverted()
    no_l = (mi.to_3x3() @ plane_no_w).normalized()
    mesh = obj.data
    best_face = None
    best_dot  = -1.0
    for f in mesh.polygons:
        d = abs(f.normal.dot(no_l))
        if d > best_dot:
            best_dot  = d
            best_face = f
    if best_face is None or best_dot < 0.5:
        return None, None, None
    verts       = [mesh.vertices[i].co for i in best_face.vertices]
    world_verts = [(mw @ v) for v in verts]
    centroid    = Vector([sum(v[i] for v in world_verts) / len(world_verts)
                          for i in range(3)])
    avg_r    = sum((v - centroid).length for v in world_verts) / len(world_verts)
    normal_w = (mw.to_3x3() @ best_face.normal).normalized()
    return centroid, avg_r, normal_w


def _execute_contour_cut(ctx, mesh_obj, limb_vert_indices, plane_co_w, plane_no_w, radius):
    orig_name = mesh_obj.name
    scene = bpy.context.scene
    _invalidate_mesh_cache(mesh_obj)

    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    before = {o.name for o in bpy.data.objects}

    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.index_update()

        bpy.ops.mesh.select_all(action='DESELECT')

        for v in bm.verts:
            if v.index in limb_vert_indices:
                v.select = True

        for e in bm.edges:
            if all(v.index in limb_vert_indices for v in e.verts):
                e.select = True

        for f in bm.faces:
            if all(v.index in limb_vert_indices for v in f.verts):
                f.select = True

        bmesh.update_edit_mesh(mesh_obj.data)

        try:
            bpy.ops.mesh.separate(type='SELECTED')
        except Exception as e:
            print(f"[VibeCut] _execute_contour_cut separate() error: {e}")
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    after = {o.name for o in bpy.data.objects}
    new_names = after - before

    if not new_names:
        print("[VibeCut] _execute_contour_cut: nothing separated")
        scene["rc_body"] = ""
        scene["rc_limb"] = ""
        return mesh_obj, None

    orig_obj = bpy.data.objects.get(orig_name)
    new_obj = bpy.data.objects.get(list(new_names)[0])

    plane_no = plane_no_w.normalized()

    def bbox_cen(o):
        return Vector([
            sum((o.matrix_world @ Vector(c))[i] for c in o.bound_box) / 8
            for i in range(3)])

    pieces = [p for p in [orig_obj, new_obj] if p is not None]
    pieces.sort(key=lambda p: (bbox_cen(p) - plane_co_w).dot(plane_no), reverse=True)

    body_obj = pieces[0] if len(pieces) > 0 else None
    limb_obj = pieces[1] if len(pieces) > 1 else None

    if body_obj:
        body_obj.name = orig_name + "_body"
    if limb_obj:
        limb_obj.name = orig_name + "_limb"

    scene["rc_plane_co"] = list(plane_co_w)
    scene["rc_plane_no"] = list(plane_no_w)
    scene["rc_radius"]   = radius
    scene["rc_body"]     = body_obj.name if body_obj else ""
    scene["rc_limb"]     = limb_obj.name if limb_obj else ""

    seam_r = radius * 1.5
    if limb_obj and _is_valid(limb_obj):
        mwl = limb_obj.matrix_world
        far = 0.0
        for v in limb_obj.data.vertices:
            dd = ((mwl @ v.co) - plane_co_w).length
            if dd > far:
                far = dd
        seam_r = max(seam_r, far * 1.05)

    for piece in (body_obj, limb_obj):
        if piece and _is_valid(piece):
            _invalidate_mesh_cache(piece)
            _fill_open_boundary(ctx, piece, plane_co_w, None, seam_r,
                                use_triangles=True)

    return body_obj, limb_obj


def register_props():
    bpy.types.Scene.rc_cut_mode = bpy.props.EnumProperty(
        name="Cut Mode",
        items=[('NORMAL', "Normal", "Thin planar cut"),
               ('DISK', "Disk", "Thick disk boolean cut")],
        default='NORMAL'
    )
    bpy.types.Scene.rc_disk_thickness = bpy.props.FloatProperty(
        name="Disk Thickness", description="Thickness of the disk cut",
        default=0.05, min=0.001, max=50.0, step=1, precision=4)
    bpy.types.Scene.rc_ball_pct = bpy.props.FloatProperty(
        name="Ball Size", description="Ball radius as fraction of cut face radius",
        default=0.55, min=0.25, max=0.85, step=1, precision=2, subtype='FACTOR')
    bpy.types.Scene.rc_tolerance = bpy.props.FloatProperty(
        name="Tolerance", description="Socket radius = ball_r + tolerance (metres)",
        default=0.00015, min=0.00005, max=0.001, step=0.1, precision=5)
    bpy.types.Scene.rc_snap_pct = bpy.props.FloatProperty(
        name="Snap Strength", description="Socket offset fraction of ball radius",
        default=0.55, min=0.20, max=0.75, step=1, precision=2, subtype='FACTOR')
    bpy.types.Scene.rc_opt_fill = bpy.props.BoolProperty(
        name="Fill Cap", default=True)
    bpy.types.Scene.rc_opt_neg = bpy.props.BoolProperty(
        name="Remove (-) Side", default=False)
    bpy.types.Scene.rc_opt_pos = bpy.props.BoolProperty(
        name="Remove (+) Side", default=False)
    bpy.types.Scene.rc_opt_sep = bpy.props.BoolProperty(
        name="Separate Halves", default=True)
    bpy.types.Scene.rc_joint_type = bpy.props.EnumProperty(
        name="Joint Type",
        items=[('LIBRARY',    "Ball & Socket", "Import joint from .blend library"),
               ('PRESS_FIT',  "Press Fit",     "Procedural compression-fit pin and hole"),
               ('SCREW_TOP',  "Screw Top",     "Threaded screw-on lid (male shaft + female cavity)")],
        default='LIBRARY')
    bpy.types.Scene.rc_pin_pct = bpy.props.FloatProperty(
        name="Pin Size", description="Pin radius as fraction of ring radius",
        default=0.45, min=0.15, max=0.75, step=1, precision=2, subtype='FACTOR')
    bpy.types.Scene.rc_pin_depth = bpy.props.FloatProperty(
        name="Pin Depth", description="How far the pin/hole extends into each piece (metres)",
        default=0.006, min=0.001, max=0.05, step=0.1, precision=4)
    bpy.types.Scene.rc_pin_tolerance = bpy.props.FloatProperty(
        name="Fit Tolerance", description="Extra radius added to hole for friction fit (metres)",
        default=0.0002, min=0.0, max=0.001, step=0.01, precision=4)


def unregister_props():
    for attr in ('rc_ball_pct', 'rc_tolerance', 'rc_snap_pct',
                 'rc_opt_fill', 'rc_opt_neg', 'rc_opt_pos', 'rc_opt_sep',
                 'rc_cut_mode', 'rc_disk_thickness',
                 'rc_joint_type', 'rc_pin_pct', 'rc_pin_depth', 'rc_pin_tolerance'):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
