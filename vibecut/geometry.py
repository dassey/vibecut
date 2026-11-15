"""
geometry.py — shared math for VibeCut.

All functions here are pure geometry / bmesh — no UI, no bpy.ops except bmesh.
_state is the single shared mutable dict for the active cut session.
"""

import bpy
import bmesh
import math
import time
import numpy as np
import bpy_extras.view3d_utils

from mathutils import Vector, Quaternion, Matrix

# ────────────────────────────────────────────────────
#  Phase constants
# ────────────────────────────────────────────────────
HOVER  = 0
LOCKED = 1


def _is_valid(obj):
    """Return True if the Blender object reference is still alive."""
    if obj is None:
        return False
    try:
        _ = obj.name   # raises ReferenceError if the object was removed
        return obj.name in bpy.data.objects
    except ReferenceError:
        return False

# ────────────────────────────────────────────────────
#  Shared session state
# ────────────────────────────────────────────────────
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


# ────────────────────────────────────────────────────
#  Mesh array cache  (numpy)
# ────────────────────────────────────────────────────
# Without this cache _cross_section falls back to a pure-Python loop over every
# edge: 4206 ms per adjustment on a 712k-vert mesh.
#
# Keyed by object name; the stored key is (vert_count, edge_count). Coordinates
# are LOCAL — the world matrix is applied per call, so moving/rotating/scaling
# the OBJECT needs no invalidation, only editing its MESH does.
#
# Measured on 712,314 verts / 2,136,936 edges: ~200 ms to build, 25.6 MB, ~0 ms
# per reuse. Capped at _MESH_CACHE_MAX entries — only one object is ever active,
# and uncapped growth leaked 25.6 MB per object touched (including objects
# renamed away by _separate_halves).
_mesh_cache     = {}
_MESH_CACHE_MAX = 2


def _invalidate_mesh_cache(obj=None):
    """Drop cached numpy mesh arrays.

    Called from _execute_cut() — a cut changes vert/edge counts. Pass an object
    (or a name) to drop just that entry; no argument clears all. The no-arg form
    preserves the existing call site in _execute_cut.
    """
    if obj is None:
        _mesh_cache.clear()
        return
    _mesh_cache.pop(getattr(obj, "name", obj), None)


def _rc_depsgraph_invalidate(scene, depsgraph):
    """Drop cache entries for any mesh Blender reports as geometry-changed.

    The (name, vert_count, edge_count) key cannot see a PURE VERTEX MOVE — e.g.
    mesh_tools' straighten_to_line or flatten_region, both of which write
    v.co and flush via bmesh.update_edit_mesh without changing any count. Those
    are EDIT-mode ops and LOCKED phase passes events through, so
    TAB -> flatten -> TAB with the orange curve on screen is a real path that
    would otherwise leave a preview that lies about where the cut lands.
    """
    if not _mesh_cache:
        return
    for upd in depsgraph.updates:
        if getattr(upd, "is_updated_geometry", False):
            _mesh_cache.pop(getattr(upd.id, "name", None), None)
            orig = getattr(upd.id, "original", None)
            if orig is not None:
                _mesh_cache.pop(getattr(orig, "name", None), None)


def _get_mesh_cache(obj):
    """(verts (V,3) float32 LOCAL, e0 (E,) int32, e1 (E,) int32).

    Rebuilt only when obj.name / vertex count / edge count change; the
    depsgraph handler above covers same-count vertex moves.

    e0/e1 are two separate C-contiguous arrays rather than one (E,2) array on
    purpose: the per-call hot path is a fancy-index gather with them, and
    contiguous columns measured 8.08 ms vs 9.68 ms for slicing an (E,2) array.
    """
    me = obj.data
    # obj.data is stale while the mesh is open in edit mode — flush first.
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


# ────────────────────────────────────────────────────
#  PCA axis detection
# ────────────────────────────────────────────────────

def _pca_axis(obj, hit_w, radius):
    """Dominant eigenvector of nearby vertices → limb axis."""
    n = len(obj.data.vertices)
    if n < 6:
        return None
    # Second documented consumer of the cache — saves a ~200 ms foreach_get on
    # every AUTO / N-key snap.
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


# ────────────────────────────────────────────────────
#  Ring + cross-section geometry
# ────────────────────────────────────────────────────

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
    """Plane ∩ mesh, clipped to the ring radius → (points, centroid).

    IDENTICAL CONTRACT to the pure-Python edge loop this replaces:
      * world-space 3-tuples of plain Python floats,
      * sorted by angle around the centroid in the (rx, ry) plane basis,
      * pts[0] appended to close the loop,
      * centroid falls back to co_w when fewer than 3 points are found, and in
        that case the points are returned unsorted and unclosed.

    Measured on a 712,314-vert / 2,136,936-edge mesh: 4206 ms → 13 ms.
    Verified point-for-point against the old loop across three plane
    orientations: identical counts, identical order, max deviation 4.08e-04
    world units (float32 rounding on a 150-unit model).
    """
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

    # Signed distance of EVERY vertex to the plane, one matvec.  ~1.45 ms
    d  = verts @ nrm
    d -= float(np.dot(c, nrm))

    # Edges straddling the plane: endpoints on opposite sides.  ~8.1 ms
    # Gather on the bool array (1 byte/vert) rather than on d (4 bytes) —
    # this is the single most expensive step, so keep it 1-byte wide.
    sgn = d < 0.0
    idx = np.flatnonzero(sgn[e0] ^ sgn[e1])
    if idx.size == 0:
        return [], co_w

    # Radius clip, applied ONLY to the straddling subset (3311 of 2.14M edges
    # on the test mesh). Reproduces the original predicate exactly: drop an edge
    # only when BOTH endpoints are outside radius*1.05. AND-ing two pure
    # predicates is order-independent, so evaluating the radius test after the
    # sign test yields the same surviving set, in the same ascending-edge-index
    # order the Python loop produced.
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

    # Vectorized lerp: p = v0 + t*(v1 - v0),  t = d0 / (d0 - d1).
    # d0 - d1 can never be 0 here: the straddle test guarantees exactly one of
    # the two is < 0, so they cannot be equal.
    t = (d0 / (d0 - d1)).astype(np.float32)[:, None]
    P = v0 + t * (v1 - v0)

    # Local → world, same as `mw @ v` per point.
    M  = np.array(mw.to_3x3(), dtype=np.float32)
    T  = np.array(mw.translation, dtype=np.float32)
    Pw = P @ M.T + T

    if Pw.shape[0] <= 2:
        # Original: len(pts) > 2 guard — unsorted, unclosed, centroid = co_w
        return [tuple(p) for p in Pw.tolist()], co_w

    cen = Pw.mean(axis=0)

    # Basis built with mathutils; algebraically equivalent to the original,
    # float32 rounding only.
    nw  = no_w.normalized()
    ref = Vector((0, 0, 1)) if abs(nw.dot(Vector((0, 0, 1)))) < 0.9 else Vector((1, 0, 0))
    rx  = nw.cross(ref).normalized()
    ry  = nw.cross(rx).normalized()

    rel = Pw - cen
    ang = np.arctan2(rel @ np.array((ry.x, ry.y, ry.z), dtype=np.float32),
                     rel @ np.array((rx.x, rx.y, rx.z), dtype=np.float32))
    # kind='stable' reproduces list.sort()'s tie-breaking (insertion order)
    order = np.argsort(ang, kind='stable')

    pts = [tuple(p) for p in Pw[order].tolist()]
    pts.append(pts[0])
    return pts, Vector(cen.tolist())


# Cross-section recompute throttle. The vectorized _cross_section is ~13 ms on
# a 712k-vert mesh — fine at 10-15 Hz, but gizmo modal() fires once per
# mouse-move event, which exceeds a 120 Hz mouse's 8.3 ms budget and stacks on
# top of Blender's own redraw of the same dense mesh. Rapid input marks the
# preview dirty and KEEPS DRAWING THE PREVIOUS CURVE (never cleared);
# _flush_cross() catches up on the next lull and at gizmo exit, so the orange
# curve is at most _CROSS_THROTTLE stale and always converges.
_CROSS_THROTTLE = 0.06   # seconds
_cross_last     = 0.0
_cross_cost     = 0.0    # measured seconds of the last _cross_section call


def _compute_cross():
    """Unconditional cross-section recompute + bookkeeping."""
    global _cross_last, _cross_cost
    s = _state
    t0 = time.perf_counter()
    pts, _ = _cross_section(s["obj"], s["plane_co"], s["plane_no"])
    _cross_cost      = time.perf_counter() - t0
    s["cross"]       = pts
    s["cross_dirty"] = False
    _cross_last      = time.perf_counter()


def _flush_cross():
    """Finish a deferred cross-section. Cheap no-op when not dirty.

    Called from the LOCKED event handler and from gizmo exit() so the orange
    curve always ends up exact once the user stops moving.
    """
    s = _state
    if not s.get("cross_dirty"):
        return
    if s["phase"] != LOCKED or not _is_valid(s["obj"]) \
            or s["plane_co"] is None or s["plane_no"] is None:
        s["cross_dirty"] = False
        return
    _compute_cross()


def _rebuild(defer=False):
    """Rebuild the visual ring. In LOCKED phase, also refresh the orange
    surface cross-section so it stays in sync with plane nudges/tilt/resize.

    The green ring (_ring_verts, 80 segments, ~0.02 ms) is ALWAYS rebuilt at
    full input rate, so the handle the user is dragging never lags.

    defer=True — for rapid input (wheel bursts, gizmo drags) — recomputes the
    cross-section at most every _CROSS_THROTTLE seconds and otherwise leaves the
    previous curve on screen, flagging cross_dirty so _flush_cross() finishes
    the job on idle. The interval adapts to the measured cost so a heavier mesh
    automatically throttles harder.

    In HOVER we skip the cross-section entirely — the orange preview only exists
    after the user clicks to lock.
    """
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
            s["cross_dirty"] = True      # keep the last curve visible
            return
        _compute_cross()
    else:
        s["cross"]       = []
        s["cross_dirty"] = False


def _lock_phase_compute():
    """Runs once on HOVER → LOCKED transition.

    Computes the orange surface cross-section — the actual intersection of
    the plane with the mesh within the ring radius. This is what shows the
    user exactly which edges will be cut before they hit Enter to commit.
    """
    s = _state
    if not _is_valid(s["obj"]) or s["plane_co"] is None or s["plane_no"] is None:
        return
    _compute_cross()   # never deferred — this is the click that locks


# ────────────────────────────────────────────────────
#  Cap-fill helpers
# ────────────────────────────────────────────────────

def _fill_loops(bm, cut_edges, use_triangles=False):
    """Walk edges into connected loops and fill each one."""
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
            # A contour seam follows the surface and is NOT planar. edgeloop_fill
            # makes ONE n-gon per loop; Blender's exact boolean solver
            # triangulates an n-gon by projecting it onto a best-fit plane, and a
            # wiggly loop projects to a SELF-INTERSECTING polygon — which yields
            # non-manifold geometry and makes the downstream press-fit boolean
            # produce garbage. triangle_fill handles non-planar loops properly.
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
    """Enter edit mode, fill boundary loops left by the cut.

    When plane args are supplied we ONLY fill boundary edges near the cut and
    within the radius. Without that filter, on a large STL with pre-existing
    open holes this walked every hole and could hang for minutes inside
    edgeloop_fill.

    plane_tol (WORLD units) widens the "near the plane" band; pass a large value
    — or plane_no_w=None with a radius — for a CONTOUR seam, which follows the
    surface and is not planar at all, so a tight plane test would reject it.
    """
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
            # plane_tol arrives in WORLD units and must be pushed through the
            # same inverse 3x3 as r_l, or it silently mis-filters on any object
            # with non-unit scale.
            if plane_tol is not None:
                tol = (mi.to_3x3() @ Vector((plane_tol, 0, 0))).length
            else:
                tol = max(r_l * 0.05, 1e-4)   # cut edges lie ON the plane
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
        # Never strand the object in EDIT mode if a fill raises.
        bpy.ops.object.mode_set(mode='OBJECT')


# ────────────────────────────────────────────────────
#  Plane manipulation
# ────────────────────────────────────────────────────

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
    """Snap plane normal to a world axis or view direction."""
    _state["plane_no"] = axis_vec.normalized()
    _rebuild()


# ────────────────────────────────────────────────────
#  Joint helpers
# ────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────
#  Separate halves  (face-selection method)
# ────────────────────────────────────────────────────

def _separate_halves(ctx, obj, plane_co_w, plane_no_w):
    """
    Select all faces whose centroid is on the POSITIVE side of the cut plane,
    then separate by SELECTED → two objects.

    IMPORTANT: face.select set in object-mode is discarded when entering edit
    mode (Blender regenerates edit-mesh selection state).  We must enter edit
    mode FIRST, then select via bmesh before calling mesh.separate.
    """
    orig_name = obj.name
    # Use bpy.context.view_layer directly — ctx may be a dict from ctx.copy()
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')

    mi    = obj.matrix_world.inverted()
    co_l  = mi @ plane_co_w
    no_l  = (mi.to_3x3() @ plane_no_w).normalized()

    # Enter edit mode and classify faces there using bmesh
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)

    # ── Isolate limb by finding the best positive vertex and flood-filling ─────
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
        # Expand selection topologically
        # Because we split edges at the cut boundary earlier, this will stop exactly at the cut
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
        # Nothing separated — rename original and return
        obj.name = orig_name + "_body"
        return obj, None

    # There are now 2 objects: original kept the UNSELECTED (negative) side,
    # and the new object got the SELECTED (positive) side.
    orig_obj = bpy.data.objects.get(orig_name)
    new_obj  = bpy.data.objects.get(list(new_names)[0])

    # Classify by bbox centroid displacement along the cut normal
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


# ────────────────────────────────────────────────────
#  Execute cut
# ────────────────────────────────────────────────────

def _execute_cut(ctx):
    s   = _state
    obj = s["obj"]
    if obj is None:
        return
    # Cut modifies the mesh — invalidate our cached vertex/edge arrays.
    _invalidate_mesh_cache()
    # ctx may be a plain dict (from N-panel ctx.copy()) — use bpy.context
    # for view_layer and scene access which always works.
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    mi = obj.matrix_world.inverted()
    co = mi @ s["plane_co"]
    no = (mi.to_3x3() @ s["plane_no"]).normalized()

    # ── Filter geometry limit domain to local cut area ──────────────────────
    mw   = obj.matrix_world
    co_w = s["plane_co"]
    no_w = s["plane_no"].normalized()
    scene = bpy.context.scene

    tagged_verts = set()
    ring_radius = s["radius"] * 1.20  # Slightly larger than the ring itself

    if scene.rc_cut_mode == 'DISK':
        # CYLINDER FILTER (DISK MODE): Limits the cut axially (depth) and radially.
        # Prevents long limbs from piercing the outer sphere.
        depth = scene.rc_disk_thickness / 2.0
        for v in bm.verts:
            offset = (mw @ v.co) - co_w
            axial = offset.dot(no_w)
            if abs(axial) <= depth:
                radial = (offset - no_w * axial).length
                if radial <= ring_radius:
                    tagged_verts.add(v)
    else:
        # SPHERE FILTER (NORMAL MODE): Limits the cut to everything within 'max_dist'.
        max_dist = ring_radius
        for v in bm.verts:
            if ((mw @ v.co) - co_w).length <= max_dist:
                tagged_verts.add(v)

    # Use 'any()' to make sure border faces that extend past bounds are sliced by bisect
    ring_faces = [f for f in bm.faces if any(v in tagged_verts for v in f.verts)]
    if not ring_faces:
        print("[VibeCut] Warning: no faces within bounds — using full mesh")
        ring_faces = list(bm.faces)

    # Gather matching elements back into bmesh collection format
    ring_verts = set()
    ring_edges = set()
    for f in ring_faces:
        for v in f.verts: ring_verts.add(v)
        for e in f.edges: ring_edges.add(e)
    local_geom = list(ring_verts) + list(ring_edges) + ring_faces
    # ─────────────────────────────────────────────────────────────────────────

    result = bmesh.ops.bisect_plane(
        bm, geom=local_geom, plane_co=co, plane_no=no,
        clear_inner=s["opt_neg"], clear_outer=s["opt_pos"], dist=1e-5)

    cut_edges = [e for e in result["geom_cut"] if isinstance(e, bmesh.types.BMEdge)]
    
    # NOTE: We do NOT fill caps here — filling is done AFTER separation so
    # BOTH halves get a cap face.  (If opt_sep is off, fill immediately.)
    if not s["opt_sep"] and cut_edges and s["opt_fill"]:
        _fill_loops(bm, cut_edges)

    # ── Topologically disconnect the limb at the cut plane ─────────────────────
    # Use bmesh.ops.split_edges directly on the bmesh — doesn't depend on
    # bpy.ops context or the selection state surviving an update_edit_mesh +
    # bm.free() round-trip. (The previous bpy.ops.mesh.edge_split() approach
    # was unreliable: edge_split occasionally did nothing, leaving the topology
    # connected, so mesh.separate then put the whole mesh into one half.)
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
        # Fill open boundary on BOTH halves so each gets a cap face.
        # Pass plane args so we only fill boundaries lying near the cut plane
        # and within the ring radius — never walk pre-existing holes.
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


# ────────────────────────────────────────────────────
#  Cut boundary measurement (for joint sizing)
# ────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────
#  Contour cut (auto-detect integration)
# ────────────────────────────────────────────────────

def _execute_contour_cut(ctx, mesh_obj, limb_vert_indices, plane_co_w, plane_no_w, radius):
    """Separate a limb from the body using pre-computed vertex membership.

    Unlike _execute_cut() which uses a bisect plane, this function selects
    vertices by index (from the auto-detect flood-fill) and separates them
    directly.  Both pieces get their open boundaries filled.

    Parameters
    ----------
    ctx : bpy.context or dict
        Blender context (may be a dict from ctx.copy()).
    mesh_obj : bpy.types.Object
        The mesh to cut.
    limb_vert_indices : set of int
        Vertex indices belonging to the limb side.
    plane_co_w : Vector
        Cut plane centre in world space.
    plane_no_w : Vector
        Cut plane normal in world space.
    radius : float
        Approximate radius at the neck (world units).

    Returns
    -------
    (body_obj, limb_obj) : tuple of bpy.types.Object or None
    """
    orig_name = mesh_obj.name
    scene = bpy.context.scene
    _invalidate_mesh_cache(mesh_obj)

    # Use bpy.context.view_layer directly — ctx may be a dict from ctx.copy()
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')

    # Record objects before separation
    before = {o.name for o in bpy.data.objects}

    # Enter edit mode and select limb vertices via bmesh
    bpy.ops.object.mode_set(mode='EDIT')
    try:
        bm = bmesh.from_edit_mesh(mesh_obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        # ensure_lookup_table() does NOT refresh v.index — without this the
        # index lookups below can read stale indices and separate the wrong
        # geometry with no error.
        bm.verts.index_update()

        bpy.ops.mesh.select_all(action='DESELECT')

        # Select vertices in the limb set
        for v in bm.verts:
            if v.index in limb_vert_indices:
                v.select = True

        # Select edges fully within the limb
        for e in bm.edges:
            if all(v.index in limb_vert_indices for v in e.verts):
                e.select = True

        # Select faces fully within the limb
        for f in bm.faces:
            if all(v.index in limb_vert_indices for v in f.verts):
                f.select = True

        bmesh.update_edit_mesh(mesh_obj.data)

        # Separate selected geometry
        try:
            bpy.ops.mesh.separate(type='SELECTED')
        except Exception as e:
            print(f"[VibeCut] _execute_contour_cut separate() error: {e}")
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    # Find the new object
    after = {o.name for o in bpy.data.objects}
    new_names = after - before

    if not new_names:
        # Nothing separated — do NOT rename, and clear the cut keys so the
        # joint panel does not act on a stale previous cut.
        print("[VibeCut] _execute_contour_cut: nothing separated")
        scene["rc_body"] = ""
        scene["rc_limb"] = ""
        return mesh_obj, None

    # The original keeps UNSELECTED (body), new object gets SELECTED (limb)
    orig_obj = bpy.data.objects.get(orig_name)
    new_obj = bpy.data.objects.get(list(new_names)[0])

    # Classify by bbox centroid displacement along the cut normal.
    # plane_no_w points from the LIMB toward the BODY (the sign joints'
    # place_press_fit requires), so the body is the piece further along it and
    # the limb is the piece further along -normal.
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

    # Write the cut metadata BEFORE filling, so a fill failure still leaves a
    # coherent scene for the joint panel.
    scene["rc_plane_co"] = list(plane_co_w)
    scene["rc_plane_no"] = list(plane_no_w)
    scene["rc_radius"]   = radius
    scene["rc_body"]     = body_obj.name if body_obj else ""
    scene["rc_limb"]     = limb_obj.name if limb_obj else ""

    # Fill open boundaries on both pieces. A contour seam follows the surface
    # and is NOT planar, so skip the plane-distance test (plane_no_w=None) and
    # gate on distance alone; triangulate rather than making one wiggly n-gon,
    # which the boolean solver cannot handle.
    #
    # The gate radius must bound the WHOLE seam, which wanders well past the
    # neck radius — using `radius` here left most of the seam unfilled. The limb
    # piece bounds it exactly: every seam vertex belonged to the limb, so the
    # limb's own extent from the cut centre is a tight, correct upper bound. It
    # still excludes pre-existing holes elsewhere on a large body.
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


# ────────────────────────────────────────────────────
#  Scene property registration (called from __init__)
# ────────────────────────────────────────────────────

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
