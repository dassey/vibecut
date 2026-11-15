"""
mesh_tools.py — Mesh cleanup utilities for VibeCut.

Tools for high-poly meshes (AI-generated, dense scans, decimation artefacts):

  Sharp Ridge:
    - Select Sharp Line: from active vert, walk all connected sharp edges
    - Dissolve Interior: remove redundant verts on the chain (keep endpoints
      and branch points, remove only degree-2 interior verts)
    - Merge at Center: collapse the entire selection to its centroid

  Flat Region:
    - Select Flat Region: flood-fill from active face through coplanar neighbors
    - Flatten Region: project the region's verts onto its average plane

Designed to be safe on huge meshes — all walks are local to the selection.
"""

import math
import bpy
import bmesh
from mathutils import Vector


# ══════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════

def _is_sharp_edge(edge, threshold_rad):
    """Sharp = dihedral exceeds threshold, OR boundary, OR non-manifold."""
    nf = len(edge.link_faces)
    if nf == 0:
        return False                  # wire edge — ignore
    if nf == 1:
        return True                   # boundary — always feature
    if nf > 2:
        return True                   # non-manifold — definitely sharp
    f1, f2 = edge.link_faces
    return f1.normal.angle(f2.normal) > threshold_rad


def _walk_sharp_chain(start_vert, threshold_rad,
                      stop_at_branches=False, max_radius=0.0):
    """BFS through sharp edges. Returns set of BMVerts on the connected ridge.

    `stop_at_branches`: if True, vertices with 3+ qualifying edges are included
    but don't propagate further — confines the walk to a segment between
    junctions.
    `max_radius`: if > 0, limit the walk to verts within this distance (m) of
    start_vert.
    """
    visited = set()
    queue = [start_vert]
    start_co = start_vert.co
    while queue:
        v = queue.pop()
        if v in visited:
            continue
        if max_radius > 0 and (v.co - start_co).length > max_radius:
            continue
        visited.add(v)
        qualifying = [e for e in v.link_edges if _is_sharp_edge(e, threshold_rad)]
        if stop_at_branches and v is not start_vert and len(qualifying) >= 3:
            continue
        for e in qualifying:
            other = e.other_vert(v)
            if other not in visited:
                queue.append(other)
    return visited


def _walk_short_edges(start_vert, max_length,
                      stop_at_branches=False, max_radius=0.0):
    """BFS through edges shorter than `max_length`. Returns set of BMVerts on
    the densely-packed cluster.

    `stop_at_branches`: if True, vertices with 3+ qualifying edges are included
    but don't propagate further. Useful when feature lines branch into other
    feature lines (one click = one segment between junctions).
    `max_radius`: if > 0, limit the walk to verts within this distance of
    start_vert. Useful when the mesh has uniform tessellation and even
    edge-length filtering propagates too far.
    """
    visited = set()
    queue = [start_vert]
    start_co = start_vert.co
    while queue:
        v = queue.pop()
        if v in visited:
            continue
        if max_radius > 0 and (v.co - start_co).length > max_radius:
            continue
        visited.add(v)
        qualifying = [e for e in v.link_edges if e.calc_length() <= max_length]
        if stop_at_branches and v is not start_vert and len(qualifying) >= 3:
            continue
        for e in qualifying:
            other = e.other_vert(v)
            if other not in visited:
                queue.append(other)
    return visited


def _pca_dominant_axis(verts, centroid):
    """Return the principal axis (dominant eigenvector) of the verts' covariance.
    Uses numpy (available in Blender). Falls back to first axis if degenerate."""
    import numpy as np
    coords = np.array([[v.co.x - centroid.x,
                        v.co.y - centroid.y,
                        v.co.z - centroid.z] for v in verts])
    if coords.shape[0] < 2:
        return Vector((1.0, 0.0, 0.0))
    cov = np.cov(coords.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        dom = eigvecs[:, -1]  # largest eigenvalue = dominant axis
        v = Vector((float(dom[0]), float(dom[1]), float(dom[2])))
        if v.length < 1e-9:
            return Vector((1.0, 0.0, 0.0))
        return v.normalized()
    except np.linalg.LinAlgError:
        return Vector((1.0, 0.0, 0.0))


def _straighten_component(verts, target_count):
    """Snap each vert to one of `target_count` evenly-spaced positions along
    the cluster's principal axis. Returns the actual count of distinct positions
    populated."""
    n = max(2, target_count)

    centroid = Vector((0.0, 0.0, 0.0))
    for v in verts:
        centroid += v.co
    centroid /= len(verts)

    axis = _pca_dominant_axis(verts, centroid)

    ts = [(v.co - centroid).dot(axis) for v in verts]
    t_min, t_max = min(ts), max(ts)
    span = t_max - t_min
    if span < 1e-4:
        return 0  # cluster too tiny to straighten

    if n > len(verts):
        n = len(verts)

    used = set()
    for v, t in zip(verts, ts):
        normalized = (t - t_min) / span
        bin_idx = round(normalized * (n - 1))
        bin_idx = max(0, min(n - 1, bin_idx))
        used.add(bin_idx)
        new_t = t_min + span * bin_idx / (n - 1)
        v.co = centroid + axis * new_t

    return len(used)


def _straighten_component_to_axis(verts, mw, mw_inv, move=(True, True, True)):
    """Line the verts up along the WORLD axis (X, Y or Z) the cluster already
    points along. Each vert keeps its coordinate on that axis (no sliding, no
    bin-snap); the other coordinates go to the cluster centroid's, but only
    for axes ticked in `move` - an unticked axis is never touched (e.g. move
    X/Y only keeps every vert's Z). Returns the line's axis index, or -1 if
    the cluster is too tiny."""
    import numpy as np
    world = [mw @ v.co for v in verts]
    pts = np.array([tuple(co) for co in world])
    c = pts.mean(axis=0)
    centroid = Vector((float(c[0]), float(c[1]), float(c[2])))

    # Dominant direction (PCA), then the world axis closest to it.
    try:
        _, eigvecs = np.linalg.eigh(np.cov((pts - c).T))
        dom = eigvecs[:, -1]
    except np.linalg.LinAlgError:
        dom = pts.max(axis=0) - pts.min(axis=0)
    idx = int(np.argmax(np.abs(dom)))
    span = max(co[idx] for co in world) - min(co[idx] for co in world)
    if span < 1e-4:
        return -1

    flatten = [a for a in range(3) if a != idx and move[a]]
    for v, co in zip(verts, world):
        new = co.copy()
        for a in flatten:
            new[a] = centroid[a]
        v.co = mw_inv @ new
    return idx


def _ordered_chain(verts):
    """If `verts` form one open chain through selected edges (no branches, no
    loop), return them in walk order from one end to the other. Else None."""
    sel = set(verts)
    nbr = {v: [e.other_vert(v) for e in v.link_edges if e.other_vert(v) in sel]
           for v in verts}
    if any(len(n) > 2 for n in nbr.values()):
        return None
    ends = [v for v in verts if len(nbr[v]) == 1]
    if len(ends) != 2:
        return None
    order = [ends[0]]
    prev, cur = None, ends[0]
    while True:
        nxt = [o for o in nbr[cur] if o is not prev]
        if not nxt:
            break
        prev, cur = cur, nxt[0]
        order.append(cur)
    return order if len(order) == len(verts) else None


def _lowess_3d(t, pts, half_window=0.15, min_pts=6, iters=2):
    """Robust LOWESS: at each t_i fit a weighted quadratic (tricube weights,
    half-width `half_window` of the 0..1 range, widened to reach `min_pts`
    neighbours) and take its value at t_i. Bisquare re-weighting then drops
    stray points so they cannot pull the curve. Returns smoothed (n, 3)."""
    import numpy as np
    n = len(t)
    k = min(min_pts, n - 1)
    r = np.array([max(half_window, np.partition(np.abs(t - ti), k)[k])
                  for ti in t]) * 1.0001
    robust = np.ones(n)
    out = pts.copy()
    for it in range(iters + 1):
        for i in range(n):
            u = np.abs(t - t[i]) / r[i]
            m = u < 1.0
            w = (1.0 - u[m] ** 3) ** 3 * robust[m]
            if np.count_nonzero(w > 1e-12) < 3:
                w = (1.0 - u[m] ** 3) ** 3  # all neighbours down-weighted
            dt = t[m] - t[i]
            A = np.stack([np.ones_like(dt), dt, dt * dt], axis=1)
            sw = np.sqrt(w)[:, None]
            coef = np.linalg.lstsq(A * sw, pts[m] * sw, rcond=None)[0]
            out[i] = coef[0]
        if it == iters:
            break
        res = np.linalg.norm(pts - out, axis=1)
        s = np.median(res)
        if s < 1e-9:
            break
        robust = np.clip(1.0 - (res / (6.0 * s)) ** 2, 0.0, 1.0) ** 2
    return out


def _curve_component(verts):
    """Smooth the line along the bend it already has (robust LOWESS) and move
    every vert onto that curve. Stray verts get no say in the curve and are
    pulled back into line with the rest.

    A clean chain keeps its two end verts fixed and is parametrised by length
    along the chain (handles U-bends). A cluster without a clean chain order is
    parametrised by its PCA axis and nothing is pinned. Returns True if moved.
    """
    import numpy as np
    chain = _ordered_chain(verts)
    if chain is not None:
        verts = chain
    pts = np.array([tuple(v.co) for v in verts])
    if chain is not None:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        t = np.concatenate(([0.0], np.cumsum(seg)))
    else:
        c = pts.mean(axis=0)
        try:
            _, eigvecs = np.linalg.eigh(np.cov((pts - c).T))
        except np.linalg.LinAlgError:
            return False
        t = (pts - c) @ eigvecs[:, -1]
    t_min, span = t.min(), t.max() - t.min()
    if span < 1e-4:
        return False
    t = (t - t_min) / span

    new = _lowess_3d(t, pts)
    if chain is not None:
        # A spike inflates chain length around it and distorts t, which
        # smears its neighbours. Re-measure t along the smoothed line, refit.
        seg = np.linalg.norm(np.diff(new, axis=0), axis=1)
        t2 = np.concatenate(([0.0], np.cumsum(seg)))
        if t2[-1] > 1e-6:
            t = t2 / t2[-1]
            new = _lowess_3d(t, pts)
        # Pin the chain ends (they join the rest of the mesh). Each end's
        # correction fades out over the first/last quarter of the line so the
        # middle stays on the fitted curve.
        fade_a = np.clip(1.0 - t / 0.25, 0.0, 1.0) ** 2
        fade_b = np.clip(1.0 - (1.0 - t) / 0.25, 0.0, 1.0) ** 2
        new += (np.outer(fade_a, pts[0] - new[0])
                + np.outer(fade_b, pts[-1] - new[-1]))
    for v, co in zip(verts, new):
        v.co = Vector((float(co[0]), float(co[1]), float(co[2])))
    return True


def _connected_components(verts):
    """Group `verts` by mesh edge connectivity. Returns list of lists."""
    sel_set = set(verts)
    visited = set()
    components = []
    for start in verts:
        if start in visited:
            continue
        comp = []
        queue = [start]
        while queue:
            v = queue.pop()
            if v in visited:
                continue
            visited.add(v)
            comp.append(v)
            for e in v.link_edges:
                other = e.other_vert(v)
                if other in sel_set and other not in visited:
                    queue.append(other)
        components.append(comp)
    return components


def _walk_flat_region(seed_face, threshold_rad):
    """BFS through faces whose normal stays within threshold of seed normal.

    Uses seed normal (not running average) so we don't drift across
    slightly-curved surfaces.
    """
    if seed_face is None:
        return set()
    seed_normal = seed_face.normal.copy()
    # A zero-length normal makes mathutils' angle() raise ValueError. Degenerate
    # (zero-area) faces are routine in STL imports, and quad_convert's own
    # remove_doubles can create them, so guard both sides of the comparison.
    if seed_normal.length_squared < 1e-12:
        return set()
    region = {seed_face}
    queue = [seed_face]
    while queue:
        f = queue.pop()
        for edge in f.edges:
            for nf in edge.link_faces:
                if nf in region:
                    continue
                if nf.normal.length_squared < 1e-12:
                    continue
                if seed_normal.angle(nf.normal) > threshold_rad:
                    continue
                region.add(nf)
                queue.append(nf)
    return region


def _interior_verts_of_region(faces):
    """Verts whose ALL adjacent faces are in `faces` — safe to flatten without
    creating kinks at the region boundary."""
    face_set = set(faces)
    candidate = set()
    for f in faces:
        for v in f.verts:
            candidate.add(v)
    return {v for v in candidate if all(f in face_set for f in v.link_faces)}


def _face_components(faces):
    """Group `faces` into edge-connected islands. Returns list of lists.

    Face-level twin of _connected_components. Flattening must never fit ONE
    plane through two disjoint patches (e.g. both ears selected at once) —
    that pulls each patch toward the other and wrecks both.
    """
    face_set = set(faces)
    visited = set()
    components = []
    for start in faces:
        if start in visited:
            continue
        comp = []
        queue = [start]
        while queue:
            f = queue.pop()
            if f in visited:
                continue
            visited.add(f)
            comp.append(f)
            for e in f.edges:
                for nf in e.link_faces:
                    if nf in face_set and nf not in visited:
                        queue.append(nf)
        components.append(comp)
    return components


def _area_weighted_plane(faces):
    """(centroid, unit normal, flatness_ratio) fitted through face CENTERS,
    each weighted by face area.

    Better than fitting through raw vertices for a face region: a triangulated
    STL packs many tiny triangles into high-curvature areas, so an unweighted
    vertex fit is dragged toward whichever part of the patch happens to be
    finely tessellated. Weighting by area makes the fit reflect actual surface,
    not tessellation density.

    flatness_ratio is smallest/middle singular value — near 0 for a genuine
    planar patch, approaching 1 when the points are so collinear that the plane
    normal is arbitrary.
    """
    import numpy as np
    cs = []
    ws = []
    for f in faces:
        a = f.calc_area()
        if a <= 0.0:
            continue
        c = f.calc_center_median()
        cs.append((c.x, c.y, c.z))
        ws.append(a)
    if len(cs) < 3:
        return None
    pts = np.array(cs, dtype=np.float64)
    w   = np.array(ws, dtype=np.float64)
    w  /= w.sum()
    centroid = (pts * w[:, None]).sum(axis=0)
    centered = (pts - centroid) * np.sqrt(w)[:, None]
    try:
        _u, s, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    n = vh[-1]
    normal = Vector((float(n[0]), float(n[1]), float(n[2])))
    if normal.length < 1e-9:
        return None
    ratio = float(s[-1] / s[-2]) if len(s) >= 2 and s[-2] > 1e-12 else 1.0
    return (Vector((float(centroid[0]), float(centroid[1]), float(centroid[2]))),
            normal.normalized(), ratio)


def _best_fit_plane(coords):
    """Least-squares best-fit plane through an iterable of 3D points.
    Returns (centroid, unit normal). This is the same fit LoopTools'
    Flatten uses (smallest singular vector of the centered point cloud) -
    strictly better than averaging face normals, which biases toward the
    majority orientation and drifts on bumpy regions."""
    import numpy as np
    pts = np.array([[c[0], c[1], c[2]] for c in coords], dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1]
    except np.linalg.LinAlgError:
        n = (0.0, 0.0, 1.0)
    normal = Vector((float(n[0]), float(n[1]), float(n[2])))
    if normal.length < 1e-9:
        normal = Vector((0.0, 0.0, 1.0))
    return (Vector((float(centroid[0]), float(centroid[1]),
                    float(centroid[2]))), normal.normalized())


def _flatten_with_influence(verts, plane_co, plane_no, influence):
    """Move verts toward the plane by `influence` (0..1). Returns max
    remaining deviation."""
    worst = 0.0
    for v in verts:
        d = (v.co - plane_co).dot(plane_no)
        v.co -= plane_no * (d * influence)
        worst = max(worst, abs(d * (1.0 - influence)))
    return worst


# ══════════════════════════════════════════════════
#  Operators — Sharp Ridge
# ══════════════════════════════════════════════════

class VIBECUT_OT_select_sharp_line(bpy.types.Operator):
    """Select all vertices connected to the active vertex along a feature line.

    Two methods:
    - DISTANCE (default) — walks edges shorter than Cluster Distance. Best for
      densely tessellated / AI-generated meshes where feature lines are made of
      packed small triangles.
    - ANGLE — walks edges with dihedral angle exceeding Sharp Angle. Best for
      cleanly modeled meshes with proper hard edges."""
    bl_idname  = "vibecut.select_sharp_line"
    bl_label   = "Select Sharp Line"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        if not ctx.tool_settings.mesh_select_mode[0]:
            bpy.ops.mesh.select_mode(type='VERT')

        active = bm.select_history.active
        if not isinstance(active, bmesh.types.BMVert):
            sel_verts = [v for v in bm.verts if v.select]
            if not sel_verts:
                self.report({'WARNING'}, "Select a vertex on a feature line first.")
                return {'CANCELLED'}
            active = sel_verts[0]

        method = ctx.scene.vibecut_mt_select_method
        stop_at_branches = ctx.scene.vibecut_mt_stop_at_branches
        max_radius = ctx.scene.vibecut_mt_max_radius
        if method == 'DISTANCE':
            chain = _walk_short_edges(
                active,
                ctx.scene.vibecut_mt_cluster_dist,
                stop_at_branches=stop_at_branches,
                max_radius=max_radius,
            )
            param_hint = "Cluster Distance"
        else:  # ANGLE
            threshold_rad = math.radians(ctx.scene.vibecut_mt_sharp_angle)
            chain = _walk_sharp_chain(
                active, threshold_rad,
                stop_at_branches=stop_at_branches,
                max_radius=max_radius,
            )
            param_hint = "Sharp Angle"

        if len(chain) <= 1:
            self.report({'WARNING'},
                        f"Active vertex has no qualifying neighbors — try adjusting {param_hint}.")
            return {'CANCELLED'}

        for v in chain:
            v.select = True
        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'},
                    f"Selected {len(chain)} verts via {method.lower()} method.")
        return {'FINISHED'}


class VIBECUT_OT_fill_selection_holes(bpy.types.Operator):
    """Add to the selection any unselected vertex whose fraction of selected
    neighbors meets the Fill Threshold. Iterates until no more verts qualify.

    Useful for plugging gaps left by Select Line walks — when the cluster has
    a few edges slightly above the distance threshold, those verts get skipped.
    Fill Holes adds them back based on neighborhood, not edge length"""
    bl_idname  = "vibecut.fill_selection_holes"
    bl_label   = "Fill Holes"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)

        threshold = ctx.scene.vibecut_mt_fill_threshold

        added_total = 0
        for _ in range(10):  # safety cap on iterations
            to_add = []
            for v in bm.verts:
                if v.select:
                    continue
                neighbors = [e.other_vert(v) for e in v.link_edges]
                if not neighbors:
                    continue
                sel_count = sum(1 for n in neighbors if n.select)
                if sel_count / len(neighbors) >= threshold:
                    to_add.append(v)
            if not to_add:
                break
            for v in to_add:
                v.select = True
            added_total += len(to_add)

        bm.select_flush(True)
        bmesh.update_edit_mesh(obj.data)
        if added_total == 0:
            self.report({'INFO'},
                        "No holes to fill at this threshold "
                        "(try lowering Fill Threshold).")
        else:
            self.report({'INFO'}, f"Added {added_total} verts via hole fill.")
        return {'FINISHED'}


class VIBECUT_OT_dissolve_line_interior(bpy.types.Operator):
    """Dissolve interior vertices of the selection (verts with exactly 2
    selected neighbors). Endpoints (degree-1) and branch points (degree-3+)
    are preserved"""
    bl_idname  = "vibecut.dissolve_line_interior"
    bl_label   = "Dissolve Interior"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)

        selected = [v for v in bm.verts if v.select]
        if len(selected) < 3:
            self.report({'WARNING'},
                        "Need at least 3 selected verts (path with interior).")
            return {'CANCELLED'}

        sel_set = set(selected)
        endpoints = []
        interior = []
        branch_points = []
        for v in selected:
            count = sum(1 for e in v.link_edges if e.other_vert(v) in sel_set)
            if count == 1:
                endpoints.append(v)
            elif count == 2:
                interior.append(v)
            else:  # 0 (isolated) or 3+ (branch)
                if count >= 3:
                    branch_points.append(v)

        if not endpoints and not branch_points:
            self.report({'WARNING'},
                        "Closed loop with no endpoints or branches — refusing "
                        "to dissolve (would erase the whole loop).")
            return {'CANCELLED'}
        if not interior:
            self.report({'INFO'}, "No interior verts to dissolve.")
            return {'CANCELLED'}

        bmesh.ops.dissolve_verts(bm, verts=interior)
        bmesh.update_edit_mesh(obj.data)
        self.report(
            {'INFO'},
            f"Dissolved {len(interior)} interior verts "
            f"(kept {len(endpoints)} endpoints, {len(branch_points)} branch points)."
        )
        return {'FINISHED'}


class VIBECUT_OT_merge_line_center(bpy.types.Operator):
    """Merge ALL selected vertices to a single shared centroid (one point in
    space). Use this only when you have one connected line selected — for
    multiple disjoint lines, use 'Merge Each Line' instead"""
    bl_idname  = "vibecut.merge_line_center"
    bl_label   = "Merge All at Center"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'EDIT_MESH'

    def execute(self, ctx):
        try:
            bpy.ops.mesh.merge(type='CENTER')
        except Exception as e:
            self.report({'ERROR'}, f"Merge failed: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class VIBECUT_OT_straighten_to_line(bpy.types.Operator):
    """For each connected component of selected vertices: find its dominant
    axis (PCA), then snap all verts to evenly-spaced positions on a perfectly
    straight line between the cluster's extremes. Surrounding faces re-snap
    to the new line, producing the connected-planes / faceted look.

    Run Merge by Distance (built-in) or this operator's automatic cleanup
    afterward to collapse the duplicate-position verts into the final chain"""
    bl_idname  = "vibecut.straighten_to_line"
    bl_label   = "Straighten to Line"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)

        selected = [v for v in bm.verts if v.select]
        if len(selected) < 3:
            self.report({'WARNING'}, "Select at least 3 vertices.")
            return {'CANCELLED'}

        target = ctx.scene.vibecut_mt_straighten_count
        components = _connected_components(selected)

        processed = 0
        for comp in components:
            if len(comp) < 3:
                continue
            n_used = _straighten_component(comp, target)
            if n_used > 0:
                processed += 1

        # Collapse co-located verts (multiple verts that snapped to same bin).
        # Tiny merge distance — we explicitly placed them at the same coords.
        moved_verts = [v for v in selected if v.is_valid]
        bmesh.ops.remove_doubles(bm, verts=moved_verts, dist=0.0001)

        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'},
                    f"Straightened {processed} component(s) to ~{target} verts each.")
        return {'FINISHED'}


class VIBECUT_OT_straighten_to_axis(bpy.types.Operator):
    """For each connected component of selected vertices: find the world axis
    (X, Y or Z) it already points along, then line every vert up exactly on
    that axis. Verts keep their position ALONG the axis (no re-spacing, no
    tilt). A dialog picks which axes may move; unticked axes stay untouched"""
    bl_idname  = "vibecut.straighten_to_axis"
    bl_label   = "Straighten to Axis"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def invoke(self, ctx, event):
        return ctx.window_manager.invoke_props_dialog(self)

    def draw(self, ctx):
        L = self.layout
        L.label(text="Axes allowed to move:")
        L.row(align=True).prop(ctx.scene, "vibecut_mt_axis_move", toggle=True)

    def execute(self, ctx):
        move = tuple(ctx.scene.vibecut_mt_axis_move)
        if not any(move):
            self.report({'WARNING'}, "Tick at least one axis to move.")
            return {'CANCELLED'}
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)

        selected = [v for v in bm.verts if v.select]
        if len(selected) < 2:
            self.report({'WARNING'}, "Select at least 2 vertices.")
            return {'CANCELLED'}

        mw = obj.matrix_world
        mw_inv = mw.inverted_safe()
        used_axes = []
        for comp in _connected_components(selected):
            if len(comp) < 2:
                continue
            idx = _straighten_component_to_axis(comp, mw, mw_inv, move)
            if idx >= 0:
                used_axes.append("XYZ"[idx])

        # Verts sharing the same axis coordinate now sit on top of each other.
        moved_verts = [v for v in selected if v.is_valid]
        bmesh.ops.remove_doubles(bm, verts=moved_verts, dist=0.0001)

        bmesh.update_edit_mesh(obj.data)
        if not used_axes:
            self.report({'WARNING'}, "Selection too small to straighten.")
            return {'CANCELLED'}
        if not any(move[a] for ax in used_axes for a in range(3)
                   if a != "XYZ".index(ax)):
            self.report({'WARNING'},
                        f"Line runs along {used_axes[0]}, the only axis "
                        f"ticked - nothing to straighten. Tick another axis.")
            return {'CANCELLED'}
        self.report({'INFO'},
                    f"Straightened {len(used_axes)} component(s) along "
                    f"{', '.join(used_axes)}.")
        return {'FINISHED'}


class VIBECUT_OT_smooth_curve_line(bpy.types.Operator):
    """For each connected component of selected vertices: fit a smooth curve
    that follows the bend the line already has, then pull every vert onto it.
    Stray verts that stick out of line do not distort the curve and are
    brought back into line. A clean chain keeps its two end verts in place"""
    bl_idname  = "vibecut.smooth_curve_line"
    bl_label   = "Curve Line"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)

        selected = [v for v in bm.verts if v.select]
        if len(selected) < 4:
            self.report({'WARNING'}, "Select at least 4 vertices.")
            return {'CANCELLED'}

        processed = 0
        for comp in _connected_components(selected):
            if len(comp) >= 4 and _curve_component(comp):
                processed += 1

        # Side-by-side cluster verts can land on the same curve point.
        moved_verts = [v for v in selected if v.is_valid]
        bmesh.ops.remove_doubles(bm, verts=moved_verts, dist=0.0001)

        bmesh.update_edit_mesh(obj.data)
        if not processed:
            self.report({'WARNING'}, "Nothing to curve (need 4+ connected verts).")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Curved {processed} component(s).")
        return {'FINISHED'}


class VIBECUT_OT_merge_each_line(bpy.types.Operator):
    """Merge each connected component of the selection to its OWN centroid.
    Multiple disjoint lines collapse independently — each becomes one vertex"""
    bl_idname  = "vibecut.merge_each_line"
    bl_label   = "Merge Each Line"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)

        selected = [v for v in bm.verts if v.select]
        if len(selected) < 2:
            self.report({'WARNING'}, "Select at least 2 vertices.")
            return {'CANCELLED'}

        components = _connected_components(selected)
        merged = 0
        for comp in components:
            if len(comp) < 2:
                continue
            centroid = Vector((0.0, 0.0, 0.0))
            for v in comp:
                centroid += v.co
            centroid /= len(comp)
            bmesh.ops.pointmerge(bm, verts=comp, merge_co=centroid)
            merged += 1

        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'},
                    f"Merged {merged} component(s) — each to its own centroid.")
        return {'FINISHED'}


# ══════════════════════════════════════════════════
#  Operators — Flat Region
# ══════════════════════════════════════════════════

class VIBECUT_OT_select_flat_region(bpy.types.Operator):
    """Flood-fill from the active face to all connected faces whose normal is
    within Flat Angle of the active face's normal"""
    bl_idname  = "vibecut.select_flat_region"
    bl_label   = "Select Flat Region"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        # Face normals drive the whole walk; make sure they are current.
        bm.normal_update()

        # Read select_history BEFORE any mode switch — switching select mode
        # runs BM_select_history_validate, which drops a BMVert from history
        # and would force the arbitrary-seed fallback below.
        active = bm.select_history.active

        # Need face select mode for flood-fill from a face
        if not ctx.tool_settings.mesh_select_mode[2]:
            bpy.ops.mesh.select_mode(type='FACE')

        # Promote a clicked VERT/EDGE to an adjacent face rather than giving up,
        # so the tool works when entered from Vertex or Edge select mode.
        if isinstance(active, bmesh.types.BMVert):
            active = active.link_faces[0] if active.link_faces else None
        elif isinstance(active, bmesh.types.BMEdge):
            active = active.link_faces[0] if active.link_faces else None

        if not isinstance(active, bmesh.types.BMFace):
            sel_faces = [f for f in bm.faces if f.select]
            if not sel_faces:
                self.report({'WARNING'},
                            "Click a face on the flat region first.")
                return {'CANCELLED'}
            active = sel_faces[0]

        threshold_rad = math.radians(ctx.scene.vibecut_mt_flat_angle)
        region = _walk_flat_region(active, threshold_rad)

        if len(region) <= 1:
            self.report({'WARNING'},
                        "No coplanar neighbors — try increasing Flat Angle.")
            return {'CANCELLED'}

        # REPLACE the selection, don't add to it. Accumulating meant a stale
        # selection (or a second run) unioned two regions, and Flatten Region
        # then fitted ONE plane through the union — the main "flatten doesn't
        # work well" failure. select_all is the C-speed deselect; it operates on
        # the same live BMesh, so `region` stays valid.
        bpy.ops.mesh.select_all(action='DESELECT')
        for f in region:
            f.select = True
        bm.select_flush_mode()
        bm.select_history.clear()
        bm.select_history.add(active)
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'}, f"Selected {len(region)} flat faces.")
        return {'FINISHED'}


class VIBECUT_OT_flatten_region(bpy.types.Operator):
    """Flatten the selection onto its least-squares best-fit plane (the
    same fit LoopTools uses - no add-on needed). Works on selected FACES
    (region workflow) or, with no faces selected, on selected VERTS.
    Influence blends between untouched (0) and perfectly flat (1)"""
    bl_idname  = "vibecut.flatten_region"
    bl_label   = "Flatten Region"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        bm = bmesh.from_edit_mesh(obj.data)
        bm.normal_update()

        selected_verts = {v for v in bm.verts if v.select}
        selected_faces = [f for f in bm.faces if f.select]

        if len(selected_verts) < 3:
            self.report({'WARNING'},
                        "Select at least 3 vertices (or a face region).")
            return {'CANCELLED'}

        # ── Fit the plane ────────────────────────────────────────────────
        # Prefer an AREA-WEIGHTED fit through face centers, restricted to ONE
        # connected island. Fitting through every selected vert across disjoint
        # patches pulls each patch toward the other; and an unweighted vertex
        # fit is dragged toward whichever part of the patch happens to be
        # finely tessellated (STL triangles cluster in curved areas).
        fit          = None
        region_faces = None
        note         = ""
        if len(selected_faces) >= 3:
            comps = _face_components(selected_faces)
            act   = bm.select_history.active
            comp  = None
            if isinstance(act, bmesh.types.BMFace):
                for c in comps:
                    if act in set(c):
                        comp = c
                        break
            if comp is None:
                comp = max(comps, key=len)
            if len(comps) > 1:
                note = (" %d disjoint regions selected — flattened only the one "
                        "under the active face." % len(comps))
            region_faces = comp
            fit = _area_weighted_plane(region_faces)

        if fit is not None:
            plane_co, plane_no, ratio = fit
        else:
            # <3 faces, or no usable area (a lone n-gon, two triangles of a
            # quad, a pure vertex selection). Fall back to the plain vertex fit;
            # never cancel on a case the vertex fit can handle.
            plane_co, plane_no = _best_fit_plane([v.co for v in selected_verts])
            ratio = 0.0

        # ── Choose which verts move ──────────────────────────────────────
        if region_faces is not None and not ctx.scene.vibecut_mt_flatten_boundary:
            # Opt-out mode: preserve the region rim, move interior only.
            verts_to_flatten = _interior_verts_of_region(region_faces)
            if not verts_to_flatten:
                self.report(
                    {'WARNING'},
                    "No interior verts to move (the whole selection is "
                    "boundary). Enable Flatten Boundary to flatten it all.")
                return {'CANCELLED'}
        elif region_faces is not None:
            # Every selected vert of the chosen island, PLUS any selected vert
            # with no fully-selected face (the vert-selection fringe). Dropping
            # that fringe would leave the selection rim frozen in place.
            island = set(region_faces)
            verts_to_flatten = {
                v for v in selected_verts
                if any(f in island for f in v.link_faces)
                or not any(f.select for f in v.link_faces)
            }
        else:
            verts_to_flatten = selected_verts

        if ratio > 0.5:
            self.report({'WARNING'},
                        "Selection is nearly one-dimensional — the best-fit "
                        "plane is unstable. Select a wider patch.")

        influence = ctx.scene.vibecut_mt_flatten_influence
        _flatten_with_influence(verts_to_flatten, plane_co, plane_no,
                                influence)

        bm.normal_update()
        bmesh.update_edit_mesh(obj.data)
        self.report({'INFO'},
                    "Flattened %d verts (%s fit, %d%% influence).%s"
                    % (len(verts_to_flatten),
                       "area-weighted" if fit is not None else "best-fit",
                       round(influence * 100), note))
        return {'FINISHED'}


# ══════════════════════════════════════════════════
#  Operators — Quad Convert (STL triangles -> clean faces)
# ══════════════════════════════════════════════════

def _quad_convert_mesh(me, sharp_deg, planar_deg):
    """Triangle soup -> quad/panel faces, never merging across sharp edges.

    1. Weld micro-duplicate verts (STL insurance).
    2. Mark every edge sharper than sharp_deg as SHARP.
    3. join_triangles: pair coplanar-ish triangles into quads (cmp_sharp
       stops pairs from straddling a feature edge).
    4. dissolve_limit (delimit=SHARP): merge near-coplanar faces into big
       flat panels - a flat side becomes ONE face, sharp creases survive
       exactly.
    Returns (faces_before, faces_after, sharp_count, quads, ngons, tris).
    """
    bm = bmesh.new()
    bm.from_mesh(me)
    faces_before = len(bm.faces)

    span = max((max(v.co[i] for v in bm.verts)
                - min(v.co[i] for v in bm.verts))
               for i in range(3)) if bm.verts else 1.0
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:],
                             dist=max(1e-6, span * 1e-6))

    sharp_rad = math.radians(sharp_deg)
    sharp_count = 0
    for e in bm.edges:
        if len(e.link_faces) == 2:
            try:
                if e.calc_face_angle() > sharp_rad:
                    e.smooth = False
                    sharp_count += 1
            except ValueError:
                continue

    tris = [f for f in bm.faces if len(f.verts) == 3]
    try:
        bmesh.ops.join_triangles(
            bm, faces=tris,
            angle_face_threshold=math.radians(planar_deg * 2.0),
            angle_shape_threshold=math.radians(40.0),
            cmp_seam=False, cmp_sharp=True,
            cmp_uvs=False, cmp_vcols=False, cmp_materials=False)
    except TypeError:
        # Older/newer kwarg sets - run with the minimal form.
        bmesh.ops.join_triangles(
            bm, faces=tris,
            angle_face_threshold=math.radians(planar_deg * 2.0),
            angle_shape_threshold=math.radians(40.0))

    bmesh.ops.dissolve_limit(
        bm, angle_limit=math.radians(planar_deg),
        use_dissolve_boundaries=False,
        verts=bm.verts[:], edges=bm.edges[:],
        delimit={'SHARP'})

    # Pair up any leftover triangles on curved areas.
    tris = [f for f in bm.faces if len(f.verts) == 3]
    if tris:
        try:
            bmesh.ops.join_triangles(
                bm, faces=tris,
                angle_face_threshold=math.radians(planar_deg * 3.0),
                angle_shape_threshold=math.radians(45.0),
                cmp_seam=False, cmp_sharp=True,
                cmp_uvs=False, cmp_vcols=False, cmp_materials=False)
        except TypeError:
            pass

    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
    tri_n = sum(1 for f in bm.faces if len(f.verts) == 3)
    faces_after = len(bm.faces)
    bm.to_mesh(me)
    bm.free()
    return faces_before, faces_after, sharp_count, quads, ngons, tri_n


class VIBECUT_OT_quad_convert(bpy.types.Operator):
    """Convert a triangulated mesh (STL import) into clean quad / flat-panel
    faces. Sharp edges are detected first and NEVER merged across, so the
    model's creases stay exactly where they are. By default works on a
    duplicate and hides the original"""
    bl_idname  = "vibecut.quad_convert"
    bl_label   = "Convert Triangles to Quads"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'OBJECT'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        scene = ctx.scene
        targets = [o for o in ctx.selected_objects if o.type == 'MESH']
        if not targets and ctx.active_object:
            targets = [ctx.active_object]

        msgs = []
        for src in targets:
            if scene.vibecut_mt_quad_method == 'QUADRIFLOW':
                ob = src
                if scene.vibecut_mt_quad_duplicate:
                    ob = src.copy()
                    ob.data = src.data.copy()
                    ob.name = src.name + "_quads"
                    for c in src.users_collection:
                        c.objects.link(ob)
                    src.hide_set(True)
                ctx.view_layer.objects.active = ob
                before = len(ob.data.polygons)
                try:
                    bpy.ops.object.quadriflow_remesh(
                        use_mesh_symmetry=False,
                        use_preserve_sharp=True,
                        use_preserve_boundary=False,
                        mode='FACES',
                        target_faces=scene.vibecut_mt_quad_target,
                        seed=0)
                except Exception as e:
                    self.report({'ERROR'},
                                "QuadriFlow failed on %s: %s "
                                "(try the Flat Panels method)"
                                % (src.name, e))
                    return {'CANCELLED'}
                msgs.append("%s: %d -> %d faces (QuadriFlow)"
                            % (src.name, before, len(ob.data.polygons)))
            else:
                ob = src
                if scene.vibecut_mt_quad_duplicate:
                    ob = src.copy()
                    ob.data = src.data.copy()
                    ob.name = src.name + "_quads"
                    for c in src.users_collection:
                        c.objects.link(ob)
                    src.hide_set(True)
                ctx.view_layer.objects.active = ob
                fb, fa, sharp_n, q, ng, tr = _quad_convert_mesh(
                    ob.data,
                    scene.vibecut_mt_quad_sharp,
                    scene.vibecut_mt_quad_planar)
                msgs.append(
                    "%s: %d -> %d faces (%d%% fewer; %d quads, %d panels, "
                    "%d tris; %d sharp edges kept)"
                    % (src.name, fb, fa,
                       round(100 * (1 - fa / max(fb, 1))), q, ng, tr,
                       sharp_n))

        self.report({'INFO'}, " | ".join(msgs))
        return {'FINISHED'}


# ══════════════════════════════════════════════════
#  Operators — Auto Bevel
# ══════════════════════════════════════════════════

class VIBECUT_OT_auto_bevel(bpy.types.Operator):
    """Add (or update) an angle-limited Bevel modifier with print-friendly
    defaults on every selected mesh object. Sharp edges get rounded
    automatically - no edge selection needed, fully non-destructive"""
    bl_idname  = "vibecut.auto_bevel"
    bl_label   = "Auto Bevel (Modifier)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'OBJECT'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        scene = ctx.scene
        targets = [o for o in ctx.selected_objects if o.type == 'MESH']
        if not targets and ctx.active_object:
            targets = [ctx.active_object]

        done = 0
        for obj in targets:
            mod = obj.modifiers.get("VibeCut Bevel")
            if mod is None or mod.type != 'BEVEL':
                mod = obj.modifiers.new("VibeCut Bevel", 'BEVEL')
            mod.width = scene.vibecut_mt_bevel_width
            mod.segments = scene.vibecut_mt_bevel_segments
            mod.limit_method = 'ANGLE'
            mod.angle_limit = math.radians(scene.vibecut_mt_bevel_angle)
            mod.miter_outer = 'MITER_ARC'
            mod.use_clamp_overlap = True
            mod.loop_slide = True
            mod.profile = 0.5
            done += 1

        self.report({'INFO'},
                    "Bevel modifier set on %d object(s) - width %.2f mm, "
                    "%d segment(s), edges sharper than %.0f deg."
                    % (done, scene.vibecut_mt_bevel_width,
                       scene.vibecut_mt_bevel_segments,
                       scene.vibecut_mt_bevel_angle))
        return {'FINISHED'}


class VIBECUT_OT_bevel_sharp_edges(bpy.types.Operator):
    """Bevel edges directly into the mesh (baked). Uses the selected edges;
    with nothing selected, automatically picks every edge sharper than the
    Bevel Angle"""
    bl_idname  = "vibecut.bevel_sharp_edges"
    bl_label   = "Bevel Edges (Bake)"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH'
                and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')

    def execute(self, ctx):
        obj = ctx.active_object
        scene = ctx.scene
        bm = bmesh.from_edit_mesh(obj.data)

        edges = [e for e in bm.edges if e.select]
        auto_picked = False
        if not edges:
            thr = math.radians(scene.vibecut_mt_bevel_angle)
            edges = []
            for e in bm.edges:
                if len(e.link_faces) != 2:
                    continue
                try:
                    if e.calc_face_angle() > thr:
                        edges.append(e)
                except ValueError:
                    continue
            auto_picked = True

        if not edges:
            self.report({'WARNING'},
                        "No edges to bevel - select some, or lower the "
                        "Bevel Angle so sharp edges qualify.")
            return {'CANCELLED'}

        try:
            bmesh.ops.bevel(
                bm,
                geom=edges,
                offset=scene.vibecut_mt_bevel_width,
                offset_type='OFFSET',
                segments=scene.vibecut_mt_bevel_segments,
                profile=0.5,
                affect='EDGES',
                clamp_overlap=True,
                loop_slide=True,
            )
        except Exception as e:
            self.report({'ERROR'}, "Bevel failed: %s" % e)
            return {'CANCELLED'}

        bmesh.update_edit_mesh(obj.data)
        src = "auto-detected sharp" if auto_picked else "selected"
        self.report({'INFO'}, "Beveled %d %s edge(s)." % (len(edges), src))
        return {'FINISHED'}


# ══════════════════════════════════════════════════
#  Color Piece — paint a whole shell with a chill swatch
# ══════════════════════════════════════════════════
# Colour = a material per swatch ("VC Color <name>") assigned to the faces of
# the shell, so it shows in Solid view (color type MATERIAL) and travels with
# the mesh into 3MF / Bambu Studio as a per-face colour. The face's previous
# material index is kept in the int face layer "vc_color_orig" so Cancel puts
# back exactly what was there.

SWATCHES = (                       # (key, label, sRGB hex)
    ("AVOCADO",    "Avocado",    "568203"),
    ("SAGE",       "Sage",       "9CAF88"),
    ("MOSS",       "Moss",       "8A9A5B"),
    ("EUCALYPTUS", "Eucalyptus", "5F8575"),
    ("SEAFOAM",    "Seafoam",    "8FBFA8"),
    ("TEAL",       "Dusty Teal", "4A7C7E"),
    ("SLATE",      "Slate Blue", "6A7FA0"),
    ("LAVENDER",   "Lavender",   "A79BC8"),
    ("CLAY",       "Clay Pink",  "D49A89"),
    ("TERRACOTTA", "Terracotta", "C46A4A"),
    ("MUSTARD",    "Mustard",    "D4A537"),
    ("SAND",       "Sand",       "D8C3A5"),
)
_SWATCH = {k: (label, hx) for k, label, hx in SWATCHES}
_MAT_PREFIX = "VC Color "
_ORIG_LAYER = "vc_color_orig"
_swatch_icons = None               # bpy.utils.previews collection (register)


def _srgb_bytes(hx):
    return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))


def _srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _swatch_material(key):
    """Get or create the material for a swatch (viewport + node colour)."""
    label, hx = _SWATCH[key]
    name = _MAT_PREFIX + label
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    rgb = [_srgb_to_linear(b / 255.0) for b in _srgb_bytes(hx)]
    mat.diffuse_color = (*rgb, 1.0)
    mat["vibecut_color"] = key
    nt = getattr(mat, "node_tree", None)
    if nt is not None:
        for n in nt.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                n.inputs["Base Color"].default_value = (*rgb, 1.0)
    return mat


def _shell_faces(bm):
    """Faces of every connected piece that touches the current selection."""
    seeds = [v for v in bm.verts if v.select]
    seen = set(seeds)
    stack = list(seeds)
    while stack:
        v = stack.pop()
        for e in v.link_edges:
            o = e.other_vert(v)
            if o not in seen:
                seen.add(o)
                stack.append(o)
    return [f for f in bm.faces if f.verts[0] in seen]


def _shell_color(me, faces):
    """Swatch key the shell is painted with (most faces), or None."""
    counts = {}
    for f in faces:
        counts[f.material_index] = counts.get(f.material_index, 0) + 1
    if not counts:
        return None
    idx = max(counts, key=counts.get)
    mat = me.materials[idx] if idx < len(me.materials) else None
    return mat.get("vibecut_color") if mat is not None else None


def paint_shell(me, bm, key):
    """Paint the selected shell with swatch `key`, or restore it when key is
    None. Returns the number of faces changed (0 = nothing selected).

    The shell is gathered HERE, after the face layer exists: adding a bmesh
    layer reallocates the faces and kills any BMFace gathered before it."""
    layer = bm.faces.layers.int.get(_ORIG_LAYER)
    if key is not None and layer is None:
        layer = bm.faces.layers.int.new(_ORIG_LAYER)
    faces = _shell_faces(bm)
    if key is None:
        if layer is None:
            return 0
        n = 0
        for f in faces:
            mat = (me.materials[f.material_index]
                   if f.material_index < len(me.materials) else None)
            if mat is not None and mat.get("vibecut_color"):
                f.material_index = max(f[layer], 0)
                n += 1
        return n

    if len(me.materials) == 0:
        me.materials.append(None)          # slot 0 = "as it was"
    mat = _swatch_material(key)
    idx = next((i for i, m in enumerate(me.materials) if m == mat), None)
    if idx is None:
        me.materials.append(mat)
        idx = len(me.materials) - 1
    for f in faces:
        cur = (me.materials[f.material_index]
               if f.material_index < len(me.materials) else None)
        if cur is None or not cur.get("vibecut_color"):
            f[layer] = f.material_index    # remember the pre-paint material
        f.material_index = idx
    return len(faces)


class VIBECUT_OT_color_shell(bpy.types.Operator):
    bl_idname = "vibecut.color_shell"
    bl_label = "Color Piece"
    bl_options = {'REGISTER', 'UNDO'}

    color: bpy.props.StringProperty(default="NONE", options={'SKIP_SAVE'})

    @classmethod
    def description(cls, ctx, props):
        if props.color in _SWATCH:
            return "Color the selected piece " + _SWATCH[props.color][0]
        return "Remove the color from the selected piece"

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'EDIT_MESH'

    def execute(self, ctx):
        ob = ctx.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        key = self.color if self.color in _SWATCH else None
        n = paint_shell(ob.data, bm, key)
        if not n:
            self.report({'WARNING'}, "Select part of a piece first.")
            return {'CANCELLED'}
        bmesh.update_edit_mesh(ob.data)
        sp = ctx.space_data
        if (key and sp is not None and sp.type == 'VIEW_3D'
                and sp.shading.type == 'SOLID'):
            sp.shading.color_type = 'MATERIAL'
        if key:
            self.report({'INFO'}, "Piece colored %s (%d faces)."
                        % (_SWATCH[key][0], n))
        else:
            self.report({'INFO'}, "Color removed (%d faces)." % n)
        return {'FINISHED'}


class VIBECUT_OT_color_shell_menu(bpy.types.Operator):
    bl_idname = "vibecut.color_shell_menu"
    bl_label = "Color Piece"
    bl_description = ("Pick a color for the piece (connected shell) under "
                      "the selection")

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'EDIT_MESH'

    def invoke(self, ctx, event):
        ob = ctx.edit_object
        bm = bmesh.from_edit_mesh(ob.data)
        faces = _shell_faces(bm)
        if not faces:
            self.report({'WARNING'}, "Select part of a piece first.")
            return {'CANCELLED'}
        self._current = _shell_color(ob.data, faces)
        return ctx.window_manager.invoke_popup(self, width=200)

    def draw(self, ctx):
        L = self.layout
        grid = L.grid_flow(row_major=True, columns=4, even_columns=True,
                           even_rows=True, align=True)
        for key, label, _hx in SWATCHES:
            cell = grid.column(align=True)
            cell.scale_y = 1.6
            icon = _swatch_icons[key].icon_id if _swatch_icons else 0
            op = cell.operator("vibecut.color_shell", text="",
                               icon_value=icon, depress=(key == self._current))
            op.color = key
        if self._current in _SWATCH:
            L.separator()
            L.operator("vibecut.color_shell",
                       text="Cancel Color (%s)" % _SWATCH[self._current][0],
                       icon='X').color = "NONE"

    def execute(self, ctx):
        return {'FINISHED'}


def _build_swatch_icons():
    import bpy.utils.previews
    pc = bpy.utils.previews.new()
    size = 32
    for key, _label, hx in SWATCHES:
        r, g, b = _srgb_bytes(hx)
        px = r | (g << 8) | (b << 16) | (255 << 24)
        if px >= 1 << 31:                  # icon_pixels is signed int32
            px -= 1 << 32
        pv = pc.new(key)
        pv.icon_size = (size, size)
        pv.icon_pixels = [px] * (size * size)
    return pc


# ══════════════════════════════════════════════════
#  Panel
# ══════════════════════════════════════════════════

class VIBECUT_PT_mesh_tools(bpy.types.Panel):
    bl_label       = "Mesh Cleanup"
    bl_idname      = "VIBECUT_PT_mesh_tools"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = "VibeCut"
    bl_options     = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L = self.layout
        scene = ctx.scene

        # ── Auto Bevel (works in Object AND Edit mode) ──────────────────
        box = L.box()
        box.label(text="Auto Bevel", icon='MOD_BEVEL')
        col = box.column(align=True)
        col.prop(scene, "vibecut_mt_bevel_width")
        col.prop(scene, "vibecut_mt_bevel_segments")
        col.prop(scene, "vibecut_mt_bevel_angle")
        row = box.row()
        row.scale_y = 1.2
        if ctx.mode == 'EDIT_MESH':
            row.operator("vibecut.bevel_sharp_edges", icon='MOD_BEVEL')
        else:
            row.operator("vibecut.auto_bevel", icon='MOD_BEVEL')

        # ── Quad Convert (Object mode) ──────────────────────────────────
        if ctx.mode == 'OBJECT':
            box = L.box()
            box.label(text="Quad Convert (STL cleanup)", icon='MESH_GRID')
            col = box.column(align=True)
            col.prop(scene, "vibecut_mt_quad_method", text="")
            col.prop(scene, "vibecut_mt_quad_sharp")
            if scene.vibecut_mt_quad_method == 'PANELS':
                col.prop(scene, "vibecut_mt_quad_planar")
            else:
                col.prop(scene, "vibecut_mt_quad_target")
            col.prop(scene, "vibecut_mt_quad_duplicate")
            row = box.row()
            row.scale_y = 1.3
            row.operator("vibecut.quad_convert", icon='MOD_TRIANGULATE')

        edit_mode = (ctx.mode == 'EDIT_MESH')
        if not edit_mode:
            L.separator()
            L.label(text="Tab into Edit Mode to use the tools below.",
                    icon='INFO')

        # ── Mesh stats ──────────────────────────────────────────────────
        obj = ctx.active_object
        if edit_mode and obj is not None and obj.type == 'MESH':
            stats = L.row()
            stats.scale_y = 0.85
            stats.label(
                text=f"V: {len(obj.data.vertices):,}   "
                     f"F: {len(obj.data.polygons):,}",
                icon='MESH_DATA',
            )

        # Everything below is Edit-mode only, but stays VISIBLE (greyed out) in
        # Object mode — kept visible (greyed out) rather than hidden so the
        # tools stay discoverable.
        L = L.column()
        L.enabled = edit_mode

        # ── Sharp Ridge ─────────────────────────────────────────────────
        box = L.box()
        box.label(text="Sharp Ridge / Feature Line", icon='EDGESEL')
        cap = box.column(align=True)
        cap.scale_y = 0.8
        cap.label(text="Tidy a jagged ridge: select the line of")
        cap.label(text="sharp verts, then collapse it straight.")
        box.prop(scene, "vibecut_mt_select_method")
        if scene.vibecut_mt_select_method == 'DISTANCE':
            box.prop(scene, "vibecut_mt_cluster_dist")
        else:
            box.prop(scene, "vibecut_mt_sharp_angle")
        row = box.row(align=True)
        row.prop(scene, "vibecut_mt_stop_at_branches", toggle=True)
        box.prop(scene, "vibecut_mt_max_radius")

        col = box.column(align=True)
        col.scale_y = 1.2
        col.operator("vibecut.select_sharp_line",
                     text="Select Line",
                     icon='RESTRICT_SELECT_OFF')

        # Fill Holes — plug skipped verts in the walked cluster
        sub = box.column(align=True)
        sub.prop(scene, "vibecut_mt_fill_threshold")
        sub.operator("vibecut.fill_selection_holes",
                     text="Fill Holes",
                     icon='SELECT_EXTEND')

        col = box.column(align=True)
        col.label(text="Then collapse:")
        col.prop(scene, "vibecut_mt_straighten_count")
        col.operator("vibecut.straighten_to_line",
                     text="Straighten to Line",
                     icon='IPO_LINEAR')
        col.operator("vibecut.straighten_to_axis",
                     text="Straighten to Axis",
                     icon='EMPTY_AXIS')
        col.operator("vibecut.smooth_curve_line",
                     text="Curve Line",
                     icon='IPO_EASE_IN_OUT')
        col.operator("vibecut.merge_each_line",
                     text="Merge Each Line",
                     icon='AUTOMERGE_ON')
        row = col.row(align=True)
        row.operator("vibecut.dissolve_line_interior",
                     text="Dissolve Interior",
                     icon='X')
        row.operator("vibecut.merge_line_center",
                     text="Merge All to One",
                     icon='PARTICLE_POINT')

        L.separator()

        # ── Flat Region ─────────────────────────────────────────────────
        box = L.box()
        box.label(text="Flat Region", icon='SNAP_FACE')
        cap = box.column(align=True)
        cap.scale_y = 0.8
        cap.label(text="Make a nearly-flat patch truly flat:")
        cap.label(text="click one face, Select, then Flatten.")
        box.prop(scene, "vibecut_mt_flat_angle")
        box.prop(scene, "vibecut_mt_flatten_boundary")
        box.prop(scene, "vibecut_mt_flatten_influence")

        col = box.column(align=True)
        col.scale_y = 1.2
        col.operator("vibecut.select_flat_region",
                     text="Select Flat Region",
                     icon='RESTRICT_SELECT_OFF')
        col.operator("vibecut.flatten_region",
                     text="Flatten Region",
                     icon='MOD_SMOOTH')

        L.separator()

        # ── Color Piece ─────────────────────────────────────────────────
        box = L.box()
        box.label(text="Color Piece", icon='COLOR')
        cap = box.column(align=True)
        cap.scale_y = 0.8
        cap.label(text="Select any part of a piece, pick a")
        cap.label(text="color. Pick again on it to Cancel.")
        row = box.row()
        row.scale_y = 1.3
        row.operator("vibecut.color_shell_menu", text="Color Piece",
                     icon='COLOR')


# ══════════════════════════════════════════════════
#  Registration
# ══════════════════════════════════════════════════

classes = (
    VIBECUT_OT_select_sharp_line,
    VIBECUT_OT_fill_selection_holes,
    VIBECUT_OT_dissolve_line_interior,
    VIBECUT_OT_merge_line_center,
    VIBECUT_OT_merge_each_line,
    VIBECUT_OT_straighten_to_line,
    VIBECUT_OT_straighten_to_axis,
    VIBECUT_OT_smooth_curve_line,
    VIBECUT_OT_select_flat_region,
    VIBECUT_OT_flatten_region,
    VIBECUT_OT_auto_bevel,
    VIBECUT_OT_bevel_sharp_edges,
    VIBECUT_OT_quad_convert,
    VIBECUT_OT_color_shell,
    VIBECUT_OT_color_shell_menu,
    VIBECUT_PT_mesh_tools,
)


def register_props():
    S = bpy.types.Scene
    S.vibecut_mt_select_method = bpy.props.EnumProperty(
        name="Method",
        description="How to walk along a feature line from the active vertex",
        items=[
            ('DISTANCE', "By Distance",
             "Walk through edges shorter than Cluster Distance. Best for "
             "densely tessellated / AI-generated meshes."),
            ('ANGLE',    "By Sharp Angle",
             "Walk through edges with dihedral angle exceeding Sharp Angle. "
             "Best for cleanly modeled meshes with proper hard edges."),
        ],
        default='DISTANCE',
    )
    S.vibecut_mt_cluster_dist = bpy.props.FloatProperty(
        name="Cluster Distance",
        description="Maximum edge length to traverse when walking the cluster. "
                    "Set this slightly above the typical edge length within feature "
                    "lines, but below the typical edge length of flat-region triangles",
        default=2.0, min=0.01, max=50.0,
    )
    S.vibecut_mt_fill_threshold = bpy.props.FloatProperty(
        name="Fill Threshold",
        description="Fraction of neighbors that must be selected for an unselected "
                    "vert to be filled in. 0.5 = majority (default), "
                    "1.0 = strict (only fully surrounded), lower = more aggressive",
        default=0.5, min=0.01, max=1.0, subtype='FACTOR',
    )
    S.vibecut_mt_stop_at_branches = bpy.props.BoolProperty(
        name="Stop at Branches",
        description="When walking, treat vertices with 3+ qualifying edges as "
                    "junctions: include them but don't propagate past them. "
                    "Confines selection to one segment between junctions",
        default=True,
    )
    S.vibecut_mt_max_radius = bpy.props.FloatProperty(
        name="Max Radius",
        description="Limit the walk to verts within this distance of the active "
                    "vertex. 0 = unlimited. Useful for uniformly tessellated meshes "
                    "where the walk would otherwise traverse the whole feature network",
        default=0.0, min=0.0, max=1000.0,
    )
    S.vibecut_mt_straighten_count = bpy.props.IntProperty(
        name="Target Vertex Count",
        description="Number of evenly-spaced verts on the resulting straight line. "
                    "Lower = more aggressive simplification (cleaner faceted look)",
        default=20, min=2, max=200,
    )
    S.vibecut_mt_axis_move = bpy.props.BoolVectorProperty(
        name="Move",
        description="Straighten to Axis: which world axes the verts may move "
                    "on. Unticked axes are left exactly as they are",
        size=3, subtype='XYZ', default=(True, True, True),
    )
    S.vibecut_mt_sharp_angle = bpy.props.FloatProperty(
        name="Sharp Angle",
        description="Edges with dihedral angle exceeding this are considered sharp (degrees). "
                    "Lower = pickier (only very sharp edges); higher = lazier (also picks gentle creases)",
        default=30.0, min=1.0, max=89.0,
    )
    S.vibecut_mt_flat_angle = bpy.props.FloatProperty(
        name="Flat Angle",
        description="Faces within this angle of the seed face are part of the flat region (degrees). "
                    "Lower = stricter coplanarity; higher = tolerates curvature",
        default=5.0, min=0.1, max=45.0,
    )
    S.vibecut_mt_flatten_boundary = bpy.props.BoolProperty(
        name="Flatten Boundary",
        description="On (default) = flatten every selected vert, exactly like "
                    "LoopTools. Turn OFF to freeze the selection rim and only "
                    "move interior verts (preserves transitions to the "
                    "surrounding geometry, but the rim stays bumpy)",
        default=True,
    )
    S.vibecut_mt_flatten_influence = bpy.props.FloatProperty(
        name="Influence",
        description="How far verts move toward the best-fit plane. "
                    "1.0 = perfectly flat, 0.5 = halfway (softens bumps "
                    "while keeping some character)",
        default=1.0, min=0.0, max=1.0, subtype='FACTOR',
    )
    S.vibecut_mt_bevel_width = bpy.props.FloatProperty(
        name="Bevel Width",
        description="Bevel width in mm. 0.4 (one nozzle width) rounds edges "
                    "enough to print cleanly without changing the look much",
        default=0.4, min=0.02, max=10.0,
    )
    S.vibecut_mt_bevel_segments = bpy.props.IntProperty(
        name="Segments",
        description="Bevel segments. 1 = chamfer, 2-3 = round (recommended "
                    "for prints), more = smoother but heavier",
        default=2, min=1, max=8,
    )
    S.vibecut_mt_bevel_angle = bpy.props.FloatProperty(
        name="Bevel Angle",
        description="Only edges sharper than this get beveled (degrees). "
                    "30 catches real corners and leaves gentle facets alone",
        default=30.0, min=5.0, max=85.0,
    )
    S.vibecut_mt_quad_method = bpy.props.EnumProperty(
        name="Method",
        description="How to rebuild the triangulated mesh",
        items=[
            ('PANELS', "Flat Panels (exact edges)",
             "Pair triangles into quads and merge coplanar areas into "
             "single flat faces. Sharp edges are detected first and "
             "never merged across - creases stay EXACTLY where they "
             "are. Best for faceted/stylized models"),
            ('QUADRIFLOW', "Uniform Quads (organic)",
             "Rebuild as an even quad grid at a target face count "
             "(Blender's QuadriFlow, sharp preservation on). Best for "
             "smooth organic models; creases are approximate"),
        ],
        default='PANELS',
    )
    S.vibecut_mt_quad_sharp = bpy.props.FloatProperty(
        name="Sharp Angle",
        description="Edges with a crease sharper than this (degrees) are "
                    "protected feature edges",
        default=25.0, min=5.0, max=80.0,
    )
    S.vibecut_mt_quad_planar = bpy.props.FloatProperty(
        name="Flatness Tolerance",
        description="Faces within this angle (degrees) count as coplanar "
                    "and merge into one panel. Lower = more faithful, "
                    "higher = fewer faces",
        default=4.0, min=0.5, max=15.0,
    )
    S.vibecut_mt_quad_target = bpy.props.IntProperty(
        name="Target Faces",
        description="Approximate face count for the QuadriFlow rebuild",
        default=10000, min=500, max=100000,
    )
    S.vibecut_mt_quad_duplicate = bpy.props.BoolProperty(
        name="Keep Original (work on a copy)",
        description="Convert a duplicate named <name>_quads and hide the "
                    "original, so nothing is lost",
        default=True,
    )


def unregister_props():
    S = bpy.types.Scene
    for prop in ('vibecut_mt_select_method', 'vibecut_mt_cluster_dist',
                 'vibecut_mt_fill_threshold', 'vibecut_mt_stop_at_branches',
                 'vibecut_mt_max_radius', 'vibecut_mt_straighten_count',
                 'vibecut_mt_axis_move',
                 'vibecut_mt_sharp_angle', 'vibecut_mt_flat_angle',
                 'vibecut_mt_flatten_boundary', 'vibecut_mt_flatten_influence',
                 'vibecut_mt_bevel_width', 'vibecut_mt_bevel_segments',
                 'vibecut_mt_bevel_angle', 'vibecut_mt_quad_method',
                 'vibecut_mt_quad_sharp', 'vibecut_mt_quad_planar',
                 'vibecut_mt_quad_target', 'vibecut_mt_quad_duplicate'):
        if hasattr(S, prop):
            delattr(S, prop)


def register():
    global _swatch_icons
    _swatch_icons = _build_swatch_icons()
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    global _swatch_icons
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    if _swatch_icons is not None:
        import bpy.utils.previews
        bpy.utils.previews.remove(_swatch_icons)
        _swatch_icons = None
