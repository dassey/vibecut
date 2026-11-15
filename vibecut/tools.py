import heapq
import math

import bpy
import bmesh
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils import kdtree
from mathutils.geometry import intersect_point_line
import numpy as np


def _mesh_objects(ctx):
    obs = [o for o in ctx.selected_objects if o.type == 'MESH']
    if not obs and ctx.active_object and ctx.active_object.type == 'MESH':
        obs = [ctx.active_object]
    return obs


def _open(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    return bm


def _close(bm, me):
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    me.update()


def _diag(bm):
    if not bm.verts:
        return 0.0
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    return (Vector((max(xs) - min(xs), max(ys) - min(ys),
                    max(zs) - min(zs)))).length


def _boundary_loops(edges):
    edges = set(edges)
    loops = []
    while edges:
        e0 = edges.pop()
        loop, stack = [e0], [e0]
        while stack:
            e = stack.pop()
            for v in e.verts:
                for e2 in v.link_edges:
                    if e2 in edges:
                        edges.discard(e2)
                        loop.append(e2)
                        stack.append(e2)
        loops.append(loop)
    return loops


def fit_sphere(points):
    import numpy as np
    Q = np.array([tuple(p) for p in points], dtype=float)
    if len(Q) < 4:
        c = Q.mean(axis=0)
        return Vector(c), float(np.linalg.norm(Q - c, axis=1).mean())
    A = np.c_[2 * Q, np.ones(len(Q))]
    b = (Q ** 2).sum(1)
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    c = s[:3]
    return Vector(c), float(math.sqrt(max(s[3] + c @ c, 0.0)))


def _apply_modifier_stack(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def apply_rot_scale(ob):
    if ob.data.users > 1:
        return "skipped apply (mesh shared by %d objects)" % ob.data.users
    loc, rot, scl = ob.matrix_basis.decompose()
    if rot.angle < 1e-6 and all(abs(s - 1.0) < 1e-6 for s in scl):
        return "transform already clean"
    RS = rot.to_matrix().to_4x4() @ Matrix.Diagonal(scl.to_4d())
    ob.data.transform(RS)
    if RS.determinant() < 0:
        ob.data.flip_normals()
    ob.matrix_basis = Matrix.Translation(loc)
    ob.data.update()
    bpy.context.view_layer.update()
    return "applied rotation/scale"


def merge_close(bm, dist):
    n0 = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=dist)
    return n0 - len(bm.verts)


def delete_loose(bm, min_island_frac, min_size_frac=0.02):
    loose_e = [e for e in bm.edges if not e.link_faces]
    bmesh.ops.delete(bm, geom=loose_e, context='EDGES')
    loose_v = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose_v, context='VERTS')
    n_loose = len(loose_e) + len(loose_v)

    bm.faces.ensure_lookup_table()
    seen = set()
    islands = []
    for f in bm.faces:
        if f in seen:
            continue
        isl, stack = [], [f]
        seen.add(f)
        while stack:
            g = stack.pop()
            isl.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h not in seen:
                        seen.add(h)
                        stack.append(h)
        islands.append(isl)
    if len(islands) <= 1:
        return n_loose, 0
    total = len(bm.faces)
    islands.sort(key=len, reverse=True)
    model_d = _diag(bm)

    def isl_diag(isl):
        vs = {v.co.copy().freeze() for f in isl for v in f.verts}
        lo = Vector((min(v.x for v in vs), min(v.y for v in vs),
                     min(v.z for v in vs)))
        hi = Vector((max(v.x for v in vs), max(v.y for v in vs),
                     max(v.z for v in vs)))
        return (hi - lo).length

    dead = [isl for isl in islands[1:]
            if len(isl) < min_island_frac * total
            or isl_diag(isl) < min_size_frac * model_d]
    if dead:
        bmesh.ops.delete(bm, geom=[f for isl in dead for f in isl],
                         context='FACES')
    return n_loose, len(dead)


def dissolve_degenerate(bm, dist):
    f0 = len(bm.faces)
    bmesh.ops.dissolve_degenerate(bm, dist=dist, edges=bm.edges[:])
    return f0 - len(bm.faces)


def fix_normals(bm):
    before = bm.calc_volume(signed=True)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return before < 0


def fill_holes(bm, max_sides):
    loops = _boundary_loops([e for e in bm.edges if e.is_boundary])
    filled = left = 0
    for loop in loops:
        if len(loop) > max_sides:
            left += 1
            continue
        res = bmesh.ops.holes_fill(bm, edges=loop, sides=len(loop) + 1)
        if res["faces"]:
            bmesh.ops.triangulate(bm, faces=res["faces"],
                                  quad_method='BEAUTY', ngon_method='BEAUTY')
            filled += 1
        else:
            left += 1
    return filled, left


def limited_dissolve(bm, angle_deg):
    f0 = len(bm.faces)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle_deg),
                             use_dissolve_boundaries=False,
                             verts=bm.verts[:], edges=bm.edges[:])
    return f0 - len(bm.faces)


def smooth_keep_volume(bm, iterations, strength):
    verts = [v for v in bm.verts if not v.is_boundary]
    for _ in range(iterations):
        bmesh.ops.smooth_laplacian_vert(
            bm, verts=verts, lambda_factor=strength, lambda_border=0.0,
            use_x=True, use_y=True, use_z=True, preserve_volume=True)


def drop_to_floor(ob):
    me = ob.data
    if not me.vertices:
        return "empty mesh"
    bpy.context.view_layer.update()
    zmin = min((ob.matrix_world @ v.co).z for v in me.vertices)
    ob.location.z -= zmin
    bpy.context.view_layer.update()
    return "moved %.2f mm to sit on the bed" % -zmin


def mesh_report(ob):
    bm = _open(ob.data)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    holes = len(_boundary_loops([e for e in bm.edges if e.is_boundary]))
    tris = sum(1 for f in bm.faces if len(f.verts) == 3)
    quads = sum(1 for f in bm.faces if len(f.verts) == 4)
    ngons = len(bm.faces) - tris - quads
    degen = sum(1 for f in bm.faces if f.calc_area() < 1e-10)
    vol = bm.calc_volume(signed=True)
    seen, shells = set(), 0
    for v in bm.verts:
        if v in seen:
            continue
        shells += 1
        stack = [v]
        seen.add(v)
        while stack:
            x = stack.pop()
            for e in x.link_edges:
                w = e.other_vert(x)
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
    nv, nf = len(bm.verts), len(bm.faces)
    bm.free()
    d = ob.dimensions
    return {
        "verts": nv, "faces": nf, "tris": tris, "quads": quads,
        "ngons": ngons, "non_manifold_edges": nm, "holes": holes,
        "shells": shells, "degenerate_faces": degen,
        "inside_out": vol < 0, "watertight": nm == 0 and holes == 0,
        "dims_mm": (round(d.x, 2), round(d.y, 2), round(d.z, 2)),
    }


def prep_new_model(ob, opts):
    log = []
    if opts.get("apply", True):
        log.append(apply_rot_scale(ob))
    bm = _open(ob.data)
    diag = _diag(bm)
    if diag and max(ob.dimensions) < 5.0:
        log.append("WARNING: only %.2f units across - probably modelled in "
                   "meters. Use Scale to Size before printing."
                   % max(ob.dimensions))
    dist = opts.get("merge_dist", 0.0) or max(0.002, diag * 1e-5)
    n = merge_close(bm, dist)
    log.append("merged %d duplicate vertices (%.3f mm)" % (n, dist))
    lo, isl = delete_loose(bm, opts.get("island_frac", 0.002))
    log.append("removed %d loose bits, %d floating pieces" % (lo, isl))
    n = dissolve_degenerate(bm, dist)
    log.append("removed %d zero-size faces" % n)
    if opts.get("fill", True):
        f, left = fill_holes(bm, opts.get("max_hole", 200))
        log.append("filled %d holes%s" % (
            f, ", %d too big left open" % left if left else ""))
    if fix_normals(bm):
        log.append("fixed inside-out faces")
    else:
        log.append("normals already outward")
    if opts.get("dissolve", True):
        n = limited_dissolve(bm, opts.get("dissolve_angle", 1.0))
        log.append("limited dissolve removed %d flat faces" % n)
    _close(bm, ob.data)
    if opts.get("floor", True):
        log.append(drop_to_floor(ob))
    rep = mesh_report(ob)
    log.append("RESULT: %s, %d verts, %d faces, %d hole(s), %d bad edge(s)"
               % ("WATERTIGHT" if rep["watertight"] else "NOT watertight",
                  rep["verts"], rep["faces"], rep["holes"],
                  rep["non_manifold_edges"]))
    return log, rep


REMESH_MAX_FACES = 2000000


def voxel_remesh(ob, voxel, max_faces=REMESH_MAX_FACES):
    f0 = len(ob.data.polygons)
    sx, sy, sz = ob.matrix_world.to_scale()
    area = sum(p.area for p in ob.data.polygons) *         max(abs(sx), abs(sy), abs(sz)) ** 2
    if area / (voxel * voxel) > max_faces:
        voxel = math.sqrt(area / max_faces)
    m = ob.modifiers.new("vc_remesh", 'REMESH')
    m.mode = 'VOXEL'
    m.voxel_size = voxel
    m.adaptivity = 0.0
    m.use_smooth_shade = False
    _apply_modifier_stack(ob)
    return f0, len(ob.data.polygons), voxel


def decimate_to(ob, target_faces):
    f0 = len(ob.data.polygons)
    if target_faces >= f0:
        return f0, f0
    m = ob.modifiers.new("vc_decimate", 'DECIMATE')
    m.decimate_type = 'COLLAPSE'
    m.ratio = target_faces / max(f0, 1)
    m.use_collapse_triangulate = False
    _apply_modifier_stack(ob)
    return f0, len(ob.data.polygons)


def scale_to_size(ob, size_mm, axis):
    apply_rot_scale(ob)
    d = ob.dimensions
    cur = {"X": d.x, "Y": d.y, "Z": d.z}.get(axis, max(d))
    if cur <= 0:
        return 0.0
    f = size_mm / cur
    ob.scale = ob.scale * f
    bpy.context.view_layer.update()
    apply_rot_scale(ob)
    return f


def flat_bottom(ob, height):
    bpy.context.view_layer.update()
    mw = ob.matrix_world.copy()
    zmin = min((mw @ v.co).z for v in ob.data.vertices)
    inv = mw.inverted()
    co = inv @ Vector((0.0, 0.0, zmin + height))
    no = (mw.to_3x3().transposed() @ Vector((0.0, 0.0, 1.0))).normalized()
    bm = _open(ob.data)
    res = bmesh.ops.bisect_plane(
        bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=co,
        plane_no=no, clear_inner=True)
    cut = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)
           and e.is_boundary]
    n = 0
    for loop in _boundary_loops(cut):
        r = bmesh.ops.holes_fill(bm, edges=loop, sides=len(loop) + 1)
        if r["faces"]:
            bmesh.ops.triangulate(bm, faces=r["faces"],
                                  quad_method='BEAUTY', ngon_method='BEAUTY')
            n += 1
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    _close(bm, ob.data)
    drop_to_floor(ob)
    return n


def symmetrize(bm, direction, dist):
    bmesh.ops.symmetrize(bm, input=bm.verts[:] + bm.edges[:] + bm.faces[:],
                         direction=direction, dist=dist)


def thin_wall_faces(ob, min_mm):
    me = ob.data
    mw = ob.matrix_world
    nmat = mw.to_3x3().inverted().transposed()
    verts = [mw @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    bvh = BVHTree.FromPolygons(verts, polys)
    thin = []
    for p in me.polygons:
        n = (nmat @ p.normal).normalized()
        c = mw @ p.center
        hit = bvh.ray_cast(c - n * 1e-3, -n, min_mm)
        if hit[0] is not None and hit[2] != p.index:
            thin.append(p.index)
    return thin


def make_round(bm, verts):
    if len(verts) < 4:
        return None, 0.0, 0.0
    c, r = fit_sphere([v.co for v in verts])
    mv = 0.0
    for v in verts:
        d = v.co - c
        if d.length > 1e-9:
            new = c + d.normalized() * r
            mv = max(mv, (new - v.co).length)
            v.co = new
    return c, r, mv


def fill_hole_round(bm, loop_edges, rings=3):
    loop_verts = {v for e in loop_edges for v in e.verts}
    ring = set(loop_verts)
    frontier = set(loop_verts)
    for _ in range(rings):
        nxt = set()
        for v in frontier:
            for e in v.link_edges:
                w = e.other_vert(v)
                if w not in ring:
                    nxt.add(w)
        ring |= nxt
        frontier = nxt
    c, r = fit_sphere([v.co for v in ring])
    avg_len = sum(e.calc_length() for e in loop_edges) / len(loop_edges)

    outside = set(bm.faces)
    loop_set = set(loop_edges)
    res = bmesh.ops.holes_fill(bm, edges=loop_edges,
                               sides=len(loop_edges) + 1)
    if not res["faces"]:
        return c, r, 0

    def patch_faces():
        return [f for f in bm.faces if f not in outside]

    def project(vs):
        for v in vs:
            d = v.co - c
            if d.length > 1e-9:
                v.co = c + d.normalized() * r

    bmesh.ops.triangulate(bm, faces=patch_faces(), quad_method='BEAUTY',
                          ngon_method='BEAUTY')
    for _ in range(6):
        patch = patch_faces()
        inner_e = {e for f in patch for e in f.edges} - loop_set
        long_e = [e for e in inner_e if e.calc_length() > avg_len * 1.5]
        if not long_e:
            break
        bmesh.ops.subdivide_edges(bm, edges=long_e, cuts=1,
                                  use_grid_fill=True)
        patch = patch_faces()
        bmesh.ops.triangulate(bm, faces=patch, quad_method='BEAUTY',
                              ngon_method='BEAUTY')
        project({v for f in patch_faces() for v in f.verts} - loop_verts)
    patch = patch_faces()
    inner_v = [v for v in {v for f in patch for v in f.verts}
               if v not in loop_verts]
    for _ in range(3):
        bmesh.ops.smooth_vert(bm, verts=inner_v, factor=0.5,
                              use_axis_x=True, use_axis_y=True,
                              use_axis_z=True)
        project(inner_v)
    return c, r, len(patch)


def _xy_order(verts, P):
    idx = {v: i for i, v in enumerate(verts)}
    nbr = [[] for _ in verts]
    for v in verts:
        i = idx[v]
        for e in v.link_edges:
            w = e.other_vert(v)
            j = idx.get(w)
            if j is not None:
                d = math.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1])
                nbr[i].append((j, d))

    def sweep(src):
        dist = np.full(len(verts), np.inf)
        dist[src] = 0.0
        pq = [(0.0, src)]
        while pq:
            d, i = heapq.heappop(pq)
            if d > dist[i]:
                continue
            for j, w in nbr[i]:
                nd = d + w
                if nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(pq, (nd, j))
        return dist

    d0 = sweep(0)
    reach = np.isfinite(d0)
    a = int(np.argmax(np.where(reach, d0, -1)))
    return sweep(a), reach


def _moving_avg(y, k):
    n = len(y)
    if k <= 2 or n < 4:
        return y
    h = max(1, min(k // 2, (n - 1) // 2))
    x = np.arange(n, dtype=float)
    out = np.empty_like(y, dtype=float)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        xx = x[lo:hi] - i
        deg = 2 if hi - lo >= 4 else 1
        A = np.vander(xx, deg + 1)
        coef = np.linalg.lstsq(A, y[lo:hi], rcond=None)[0]
        out[i] = coef[-1]
    return out


def _catmull(C, sub=6):
    if len(C) < 3:
        return C
    P = np.r_[C[:1] * 2 - C[1:2], C, C[-1:] * 2 - C[-2:-1]]
    t = np.linspace(0, 1, sub, endpoint=False)[:, None]
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t
                          + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(C[-1:])
    return np.concatenate(out)


def _project_polyline(Q, C):
    A, B = C[:-1], C[1:]
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    seg_s = np.r_[0.0, np.cumsum(np.sqrt(L2))][:-1]
    best_d = np.full(len(Q), np.inf)
    best_p = np.zeros_like(Q)
    best_s = np.zeros(len(Q))
    best_k = np.zeros(len(Q), dtype=int)
    for lo in range(0, len(Q), 20000):
        q = Q[lo:lo + 20000]
        t = np.clip(((q[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None],
                    0.0, 1.0)
        pts = A[None] + t[..., None] * AB[None]
        d = ((q[:, None, :] - pts) ** 2).sum(2)
        k = d.argmin(1)
        r = np.arange(len(q))
        best_d[lo:lo + 20000] = d[r, k]
        best_p[lo:lo + 20000] = pts[r, k]
        best_s[lo:lo + 20000] = seg_s[k] + t[r, k] * np.sqrt(L2[k])
        best_k[lo:lo + 20000] = k
    tan = AB[best_k] / np.sqrt(L2[best_k])[:, None]
    return best_p, best_s, tan


def straighten_wall(verts, mw, vertical=True, edges='STRAIGHT', smooth=9):
    if len(verts) < 6:
        return None
    imw = mw.inverted()
    P = np.array([tuple(mw @ v.co) for v in verts])
    s, reach = _xy_order(verts, P)
    if not reach.all():
        verts = [v for v, ok in zip(verts, reach) if ok]
        P, s = P[reach], s[reach]
    total = float(s.max())
    if total < 1e-6:
        return None
    elen = [e.calc_length() for v in verts[:2000] for e in v.link_edges]
    bin_len = max(total / 400.0, 2.0 * float(np.median(elen)) if elen else 0)
    nb = int(np.clip(round(total / max(bin_len, 1e-9)), 8, 400))
    def fit_curve(pos):
        bi = np.minimum((pos * nb).astype(int), nb - 1)
        used = np.unique(bi)
        c_ = np.array([P[bi == b, :2].mean(0) for b in used])
        c_ = _moving_avg(c_, smooth)
        if len(c_) >= 2:
            k = min(len(c_), 6)
            deg = 2 if k >= 4 else 1
            xs = np.arange(k, dtype=float)

            def extend(pts, at):
                cx_ = np.polyfit(xs, pts[:, 0], deg)
                cy_ = np.polyfit(xs, pts[:, 1], deg)
                return np.c_[np.polyval(cx_, at), np.polyval(cy_, at)]
            head = extend(c_[:k][::-1], np.array([k + 2.0, k + 1.0, k]) - 1)
            tail = extend(c_[-k:], np.array([k, k + 1.0, k + 2.0]))
            c_ = np.r_[head, c_, tail]
        return _catmull(c_)

    cxy = fit_curve(s / total)
    _, sp1, _ = _project_polyline(P[:, :2], cxy)
    cxy = fit_curve((sp1 - sp1.min()) / max(np.ptp(sp1), 1e-9))
    proj, sp, tan = _project_polyline(P[:, :2], cxy)
    nrm = np.c_[-tan[:, 1], tan[:, 0]]
    off = ((P[:, :2] - proj) * nrm).sum(1)
    target = np.full(len(off), float(np.median(off)))
    srt = np.sort(off)
    if len(srt) > 20:
        n_ = len(srt)
        cs = np.cumsum(srt)
        i = np.arange(1, n_)
        m1 = cs[:-1] / i
        m2 = (cs[-1] - cs[:-1]) / (n_ - i)
        g = int(np.argmax(i * (n_ - i) * (m1 - m2) ** 2))
        a_, b_ = srt[:g + 1], srt[g + 1:]

        def mad(x):
            return 1.4826 * float(np.median(np.abs(x - np.median(x))))
        if (len(a_) > 0.1 * n_ and len(b_) > 0.1 * n_
                and np.median(b_) - np.median(a_)
                > 4.0 * max(mad(a_), mad(b_), 1e-6)):
            ma, mb = float(np.median(a_)), float(np.median(b_))
            spn = (sp - sp.min()) / max(np.ptp(sp), 1e-9)
            lb = np.minimum((spn * nb).astype(int), nb - 1)
            outer = np.zeros(len(off), dtype=bool)
            for b in np.unique(lb):
                k = lb == b
                ob_ = off[k]
                lo_m, hi_m = ob_.min(), ob_.max()
                cut_b = (lo_m + hi_m) / 2 if hi_m - lo_m > 0.5 * (mb - ma)                     else (ma + mb) / 2
                outer[k] = ob_ >= cut_b
            target = np.where(outer, mb, ma)
    newxy = proj + nrm * target[:, None]
    moved_xy = float(np.linalg.norm(newxy - P[:, :2], axis=1).max())
    if vertical:
        P[:, :2] = newxy

    moved_z = 0.0
    if edges != 'KEEP':
        sel = set(verts)
        bnd = np.array([any((e.other_vert(v) not in sel) or e.is_boundary
                            for e in v.link_edges) for v in verts])
        sn = (sp - sp.min()) / max(np.ptp(sp), 1e-9)
        bi = np.minimum((sn * nb).astype(int), nb - 1)
        z = P[:, 2].copy()
        zmin = np.full(nb, np.nan)
        zmax = np.full(nb, np.nan)
        for b in np.unique(bi):
            zb = z[bi == b]
            zmin[b], zmax[b] = zb.min(), zb.max()
        rng_b = np.nan_to_num(zmax - zmin)[bi]
        hi = bnd & (z >= zmax[bi] - 0.35 * rng_b)
        lo = bnd & (z <= zmin[bi] + 0.35 * rng_b)
        if hi.sum() < 2 or lo.sum() < 2:
            hi, lo = z >= zmax[bi] - 1e-9, z <= zmin[bi] + 1e-9
        xy_e = [math.hypot(*(mw @ e.verts[0].co - mw @ e.verts[1].co).xy)
                for v in verts[:3000] for e in v.link_edges]
        xy_e = [d for d in xy_e if d > 1e-6]
        tol = 0.4 * (float(np.median(xy_e)) if xy_e else total / nb) / max(
            np.ptp(sp), 1e-9)
        corners = []
        for end in (sn <= tol, sn >= 1.0 - tol):
            ii = np.flatnonzero(end)
            if len(ii) == 0:
                continue
            hi[ii] = False
            lo[ii] = False
            top_i, bot_i = ii[np.argmax(z[ii])], ii[np.argmin(z[ii])]
            hi[top_i] = True
            lo[bot_i] = True
            corners.append((top_i, bot_i))

        def edge_line(mask, pick):
            o = np.argsort(sn[mask])
            ss, zz = sn[mask][o], z[mask][o]
            bb = np.minimum((ss * nb).astype(int), nb - 1)
            keys = np.unique(bb)
            xs = np.array([ss[bb == k].mean() for k in keys])
            ys = np.array([pick(zz[bb == k]) for k in keys])
            return xs, ys

        tx, ty = edge_line(hi, np.max)
        bx, by = edge_line(lo, np.min)
        zt_old = np.interp(sn, tx, ty)
        zb_old = np.interp(sn, bx, by)
        if edges == 'STRAIGHT':
            if len(corners) == 2:
                (ta, ba), (tb, bb_) = corners
            else:
                hi_i, lo_i = np.flatnonzero(hi), np.flatnonzero(lo)
                ta, tb = hi_i[np.argmin(sn[hi_i])], hi_i[np.argmax(sn[hi_i])]
                ba, bb_ = lo_i[np.argmin(sn[lo_i])], lo_i[np.argmax(sn[lo_i])]

            def line(i0, i1):
                f = (sn - sn[i0]) / max(sn[i1] - sn[i0], 1e-9)
                return z[i0] + (z[i1] - z[i0]) * f
            zt_new, zb_new = line(ta, tb), line(ba, bb_)
        else:
            zt_new = np.interp(sn, tx, _moving_avg(ty, smooth))
            zb_new = np.interp(sn, bx, _moving_avg(by, smooth))
        span = zt_old - zb_old
        t = np.where(span > 1e-6, (z - zb_old) / np.where(span > 1e-6, span, 1),
                     0.5)
        t = np.clip(t, 0.0, 1.0)
        t[hi] = 1.0
        t[lo & ~hi] = 0.0
        P[:, 2] = zb_new + t * (zt_new - zb_new)
        moved_z = float(np.abs(P[:, 2] - z).max())

    for v, q in zip(verts, P):
        v.co = imw @ Vector(q)
    return {"verts": len(verts), "length": total, "bins": nb,
            "moved_xy": moved_xy if vertical else 0.0, "moved_z": moved_z}


def _edge_components(sel_edges):
    adj = {}
    for e in sel_edges:
        a, b = e.verts
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    comps, seen = [], set()
    for start in adj:
        if start in seen:
            continue
        comp, stack = set(), [start]
        seen.add(start)
        while stack:
            v = stack.pop()
            comp.add(v)
            for n in adj[v]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        comps.append(comp)
    return comps


def _project_onto_chain(points, chain_verts, chain_edges, k):
    lst = list(chain_verts)
    kd = kdtree.KDTree(len(lst))
    for i, v in enumerate(lst):
        kd.insert(v.co, i)
    kd.balance()
    out = {}
    for key, co in points:
        best = None
        for _, idx, _ in kd.find_n(co, k):
            for e in lst[idx].link_edges:
                if e not in chain_edges:
                    continue
                a, b = e.verts
                _, fac = intersect_point_line(co, a.co, b.co)
                fac = min(max(fac, 0.0), 1.0)
                q = a.co.lerp(b.co, fac)
                d = (q - co).length_squared
                if best is None or d < best[0]:
                    best = (d, e, fac, q)
        if best is not None:
            out[key] = best[1:]
    return out


def _split_at(bm, per_edge, tol, targetmap, map_new_to):
    for e, items in per_edge.items():
        if not e.is_valid:
            continue
        a, b = e.verts
        items.sort(key=lambda t: t[0])
        cur_edge, cur_a, last = e, a, None
        for fac, key, co in items:
            if last is not None and (co - last.co).length <= tol:
                map_new_to(key, last)
                continue
            _, nv = bmesh.utils.edge_split(cur_edge, cur_a, 0.5)
            nv.co = co.copy()
            map_new_to(key, nv)
            cur_edge = next(le for le in nv.link_edges
                            if le.other_vert(nv) is b)
            cur_a, last = nv, nv


def snap_loop_to_loop(bm, mw, move='HIGHER', weld=True, zip_seam=True,
                      tol=1e-5, k=8):
    sel_edges = [e for e in bm.edges if e.select]
    loops = _edge_components(sel_edges)
    if len(loops) != 2:
        raise ValueError("Select exactly 2 separate edge loops "
                         "(found %d)" % len(loops))
    sel_set = set(sel_edges)

    def length(c):
        return sum(e.calc_length() for e in sel_set
                   if e.verts[0] in c and e.verts[1] in c)

    def avg_z(c):
        return sum((mw @ v.co).z for v in c) / len(c)

    key = avg_z if move in ('HIGHER', 'LOWER') else length
    loops.sort(key=key)
    if move in ('HIGHER', 'LONGER'):
        moving, target = loops[1], loops[0]
    else:
        moving, target = loops[0], loops[1]
    t_edges = {e for e in sel_set if e.verts[0] in target
               and e.verts[1] in target}
    m_edges = {e for e in sel_set if e.verts[0] in moving
               and e.verts[1] in moving}

    hits = _project_onto_chain([(v, v.co.copy()) for v in moving], target,
                               t_edges, k)
    max_move = 0.0
    for v, (e, fac, q) in hits.items():
        max_move = max(max_move, (q - v.co).length)
        v.co = q
    missed = len(moving) - len(hits)
    stats = {"moved": len(hits), "missed": missed, "target": len(target),
             "max_move": max_move, "welded": 0, "zipped": 0}
    if not weld:
        bm.normal_update()
        return stats

    targetmap = {}
    per_edge = {}
    for v, (e, fac, q) in hits.items():
        a, b = e.verts
        if (v.co - a.co).length <= tol:
            targetmap[v] = a
        elif (v.co - b.co).length <= tol:
            targetmap[v] = b
        else:
            per_edge.setdefault(e, []).append((fac, v, v.co.copy()))

    def to_target(v, nv):
        targetmap[v] = nv
    _split_at(bm, per_edge, tol, targetmap, to_target)
    target_now = {v for v in target if v.is_valid} | set(targetmap.values())

    if zip_seam:
        matched = set(targetmap.values())
        m_edges = {e for e in m_edges if e.is_valid}
        loose_t = [(tv, tv.co.copy()) for tv in target_now
                   if tv not in matched]
        zhits = _project_onto_chain(loose_t, moving, m_edges, k)
        per_m = {}
        for tv, (e, fac, q) in zhits.items():
            if (q - tv.co).length > max(0.25 * e.calc_length(), tol):
                continue
            a, b = e.verts
            if (tv.co - a.co).length <= tol or (tv.co - b.co).length <= tol:
                continue
            per_m.setdefault(e, []).append((fac, tv, tv.co.copy()))

        def zip_to(tv, nv):
            targetmap[nv] = tv
            stats["zipped"] += 1
        _split_at(bm, per_m, tol, targetmap, zip_to)

    stats["welded"] = len(targetmap)
    bmesh.ops.weld_verts(bm, targetmap=targetmap)
    keep = [v for v in target_now if v.is_valid]
    near = list({e for v in keep for e in v.link_edges})
    if near:
        bmesh.ops.dissolve_degenerate(bm, dist=tol, edges=near)
    for v in bm.verts:
        v.select = False
    for v in keep:
        if v.is_valid:
            v.select = True
    bm.select_flush(True)
    bm.normal_update()
    seam = [e for v in keep if v.is_valid for e in v.link_edges
            if e.verts[0] in keep and e.verts[1] in keep]
    stats["seam_open_edges"] = sum(1 for e in set(seam)
                                   if len(e.link_faces) != 2)
    return stats


class VibeCutToolProps(bpy.types.PropertyGroup):
    do_apply: bpy.props.BoolProperty(name="Apply Rotation/Scale",
                                     default=True)
    do_fill: bpy.props.BoolProperty(name="Fill Holes", default=True)
    do_dissolve: bpy.props.BoolProperty(name="Limited Dissolve",
                                        default=True)
    do_floor: bpy.props.BoolProperty(name="Drop to Bed", default=True)
    show_prep_options: bpy.props.BoolProperty(name="Options", default=False)

    merge_dist: bpy.props.FloatProperty(
        name="Merge Distance", default=0.0, min=0.0, max=1.0,
        precision=4, description="Vertices closer than this (mm) are "
        "welded. 0 = automatic (model size / 100,000)")
    island_pct: bpy.props.FloatProperty(
        name="Floating Bit Size %", default=0.2, min=0.0, max=10.0,
        description="Separate pieces with fewer faces than this % of the "
        "model - or smaller than 2% of its size - are deleted (the biggest "
        "piece is always kept)")
    max_hole: bpy.props.IntProperty(
        name="Max Hole Edges", default=200, min=3, max=10000,
        description="Holes with more edges than this are left open "
        "(probably intentional)")
    dissolve_angle: bpy.props.FloatProperty(
        name="Dissolve Angle", default=1.0, min=0.1, max=15.0,
        description="Neighbouring faces flatter than this (degrees) merge "
        "into one. Higher = fewer faces")

    smooth_iter: bpy.props.IntProperty(name="Smooth Passes", default=3,
                                       min=1, max=50)
    smooth_strength: bpy.props.FloatProperty(name="Smooth Strength",
                                             default=0.5, min=0.05, max=2.0)
    voxel: bpy.props.FloatProperty(
        name="Voxel Size", default=0.3, min=0.02, max=5.0, precision=3,
        description="Even Remesh grid size (mm). Smaller keeps more detail "
        "but makes more faces")
    target_faces: bpy.props.IntProperty(name="Target Faces", default=50000,
                                        min=100, max=5000000)
    size_mm: bpy.props.FloatProperty(name="Size (mm)", default=100.0,
                                     min=0.1, max=2000.0)
    size_axis: bpy.props.EnumProperty(
        name="Measure", items=[("MAX", "Largest side", ""),
                               ("Z", "Height (Z)", ""),
                               ("X", "Width (X)", ""),
                               ("Y", "Depth (Y)", "")],
        default="Z")
    bottom_cut: bpy.props.FloatProperty(
        name="Cut Height", default=1.0, min=0.05, max=50.0,
        description="How far above the lowest point to slice the model "
        "flat (mm)")
    sym_dir: bpy.props.EnumProperty(
        name="Keep Side", items=[("X", "+X  ->  -X", ""),
                                 ("-X", "-X  ->  +X", ""),
                                 ("Y", "+Y  ->  -Y", ""),
                                 ("-Y", "-Y  ->  +Y", ""),
                                 ("Z", "+Z  ->  -Z", ""),
                                 ("-Z", "-Z  ->  +Z", "")],
        default="X")
    loop_move: bpy.props.EnumProperty(
        name="Move", default='HIGHER', items=(
            ('HIGHER', "Higher loop", "Move the loop that sits higher (Z)"),
            ('LOWER', "Lower loop", "Move the loop that sits lower (Z)"),
            ('SHORTER', "Shorter loop", "Move the shorter loop"),
            ('LONGER', "Longer loop", "Move the longer loop")))
    loop_weld: bpy.props.BoolProperty(
        name="Weld (watertight seam)", default=True,
        description="Merge the moved loop into the other one")
    loop_zip: bpy.props.BoolProperty(
        name="Zip extra vertices", default=True,
        description="Also split the moved loop where the other loop has "
        "vertices in between - needed for a truly watertight seam when the "
        "two loops have different vertex counts")
    wall_vertical: bpy.props.BoolProperty(
        name="Make Vertical (keep curve)", default=True,
        description="Slide vertices straight in/out onto a smooth curve "
        "through the strip, so the faces stand straight up off the XY plane")
    wall_edges: bpy.props.EnumProperty(
        name="Top/Bottom", default='STRAIGHT', items=(
            ('STRAIGHT', "Straight (corner to corner)",
             "Top edge = line between the two top corners, bottom edge = "
             "line between the two bottom corners"),
            ('SMOOTH', "Smooth", "Smooth the top/bottom edges but keep "
             "their rise and fall"),
            ('KEEP', "Keep Heights", "Only make it vertical")))
    wall_smooth: bpy.props.IntProperty(
        name="Curve Smoothness", default=9, min=1, max=60,
        description="How much the footprint curve is smoothed (bins). "
        "Higher = smoother, lower = follows the strip more closely")
    thin_mm: bpy.props.FloatProperty(
        name="Min Wall (mm)", default=1.2, min=0.2, max=10.0,
        description="Walls thinner than this get selected (0.4 mm nozzle: "
        "2 lines = 0.84 mm, 1.2 is a safe minimum)")
    status: bpy.props.StringProperty(default="")


def _props(ctx):
    return ctx.scene.vibecut_tools


def _set_status(ctx, lines):
    _props(ctx).status = " | ".join(lines)


class _ObjectTool(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'OBJECT' and bool(_mesh_objects(ctx))


class VIBECUT_OT_prep_new_model(_ObjectTool):
    """ALL-IN-ONE first pass for a freshly imported 2D->3D model: apply
    rotation/scale, weld duplicate vertices, delete loose bits and floating
    pieces, remove zero-size faces, fill holes, fix inside-out faces,
    limited dissolve, drop to the bed, then report printability"""
    bl_idname = "vibecut.prep_new_model"
    bl_label = "Prep New Model (All-in-One)"

    def execute(self, ctx):
        p = _props(ctx)
        opts = {"apply": p.do_apply, "merge_dist": p.merge_dist,
                "island_frac": p.island_pct / 100.0, "fill": p.do_fill,
                "max_hole": p.max_hole, "dissolve": p.do_dissolve,
                "dissolve_angle": p.dissolve_angle, "floor": p.do_floor}
        lines = []
        for ob in _mesh_objects(ctx):
            log, rep = prep_new_model(ob, opts)
            lines.append(ob.name + ": " + log[-1].replace("RESULT: ", ""))
            print("[VibeCut Prep] %s\n  " % ob.name + "\n  ".join(log))
            for msg in log:
                if msg.startswith("WARNING"):
                    lines.append(msg)
        _set_status(ctx, lines)
        self.report({'INFO'}, " | ".join(lines))
        return {'FINISHED'}


class VIBECUT_OT_print_check(_ObjectTool):
    """Report printability: holes, bad (non-manifold) edges, inside-out
    faces, separate pieces, triangle/quad counts, size in mm"""
    bl_idname = "vibecut.print_check"
    bl_label = "Check Model"
    bl_options = {'REGISTER'}

    def execute(self, ctx):
        lines = []
        for ob in _mesh_objects(ctx):
            r = mesh_report(ob)
            lines.append(
                "%s: %s | %d holes, %d bad edges, %d pieces%s | "
                "%d tris %d quads %d ngons | %s x %s x %s mm" % (
                    ob.name,
                    "WATERTIGHT" if r["watertight"] else "NOT watertight",
                    r["holes"], r["non_manifold_edges"], r["shells"],
                    ", INSIDE-OUT" if r["inside_out"] else "",
                    r["tris"], r["quads"], r["ngons"], *r["dims_mm"]))
        _set_status(ctx, lines)
        self.report({'INFO'}, " | ".join(lines))
        return {'FINISHED'}


def _bm_tool(ctx, fn, fmt):
    lines = []
    for ob in _mesh_objects(ctx):
        bm = _open(ob.data)
        res = fn(bm)
        _close(bm, ob.data)
        lines.append(ob.name + ": " + fmt(res))
    _set_status(ctx, lines)
    return lines


class VIBECUT_OT_tool_apply(_ObjectTool):
    """Bake rotation and scale into the mesh so sizes read true in mm"""
    bl_idname = "vibecut.tool_apply"
    bl_label = "Apply Rotation & Scale"

    def execute(self, ctx):
        _set_status(ctx, [o.name + ": " + apply_rot_scale(o)
                          for o in _mesh_objects(ctx)])
        return {'FINISHED'}


class VIBECUT_OT_tool_merge(_ObjectTool):
    """Weld vertices that sit on top of each other (the 'too close
    vertices'); closes most invisible cracks in AI meshes"""
    bl_idname = "vibecut.tool_merge"
    bl_label = "Merge Close Vertices"

    def execute(self, ctx):
        p = _props(ctx)

        def fn(bm):
            d = p.merge_dist or max(0.002, _diag(bm) * 1e-5)
            return merge_close(bm, d), d
        _bm_tool(ctx, fn, lambda r: "merged %d (%.4f mm)" % r)
        return {'FINISHED'}


class VIBECUT_OT_tool_loose(_ObjectTool):
    """Delete loose vertices/edges and small floating pieces"""
    bl_idname = "vibecut.tool_loose"
    bl_label = "Delete Floating Bits"

    def execute(self, ctx):
        p = _props(ctx)
        _bm_tool(ctx, lambda bm: delete_loose(bm, p.island_pct / 100.0),
                 lambda r: "%d loose bits, %d floating pieces" % r)
        return {'FINISHED'}


class VIBECUT_OT_tool_degenerate(_ObjectTool):
    """Remove zero-area faces and zero-length edges"""
    bl_idname = "vibecut.tool_degenerate"
    bl_label = "Remove Zero-Size Faces"

    def execute(self, ctx):
        p = _props(ctx)
        _bm_tool(ctx, lambda bm: dissolve_degenerate(
            bm, p.merge_dist or max(0.002, _diag(bm) * 1e-5)),
            lambda r: "%d removed" % r)
        return {'FINISHED'}


class VIBECUT_OT_tool_fill(_ObjectTool):
    """Fill every hole up to Max Hole Edges (flat fill). For a curved patch
    use Fill Hole Round in Edit Mode"""
    bl_idname = "vibecut.tool_fill"
    bl_label = "Fill Holes"

    def execute(self, ctx):
        p = _props(ctx)
        _bm_tool(ctx, lambda bm: fill_holes(bm, p.max_hole),
                 lambda r: "filled %d, %d left open" % r)
        return {'FINISHED'}


class VIBECUT_OT_tool_normals(_ObjectTool):
    """Point all faces outward (fixes inside-out models)"""
    bl_idname = "vibecut.tool_normals"
    bl_label = "Fix Normals (Outside)"

    def execute(self, ctx):
        _bm_tool(ctx, fix_normals,
                 lambda r: "was inside-out, fixed" if r else "already fine")
        return {'FINISHED'}


class VIBECUT_OT_tool_dissolve(_ObjectTool):
    """Merge neighbouring faces that are flatter than Dissolve Angle into
    clean flat panels"""
    bl_idname = "vibecut.tool_dissolve"
    bl_label = "Limited Dissolve"

    def execute(self, ctx):
        p = _props(ctx)
        _bm_tool(ctx, lambda bm: limited_dissolve(bm, p.dissolve_angle),
                 lambda r: "%d faces removed" % r)
        return {'FINISHED'}


class VIBECUT_OT_tool_smooth(_ObjectTool):
    """Smooth out lumpy AI surfaces without shrinking the model"""
    bl_idname = "vibecut.tool_smooth"
    bl_label = "Smooth (Keep Volume)"

    def execute(self, ctx):
        p = _props(ctx)
        _bm_tool(ctx, lambda bm: smooth_keep_volume(
            bm, p.smooth_iter, p.smooth_strength), lambda r: "smoothed")
        return {'FINISHED'}


def _close_for_remesh(ob):
    if mesh_report(ob)["holes"] == 0:
        return ""
    bm = _open(ob.data)
    filled, _left = fill_holes(bm, 1000000)
    fix_normals(bm)
    _close(bm, ob.data)
    return " (filled %d hole(s) first)" % filled


class VIBECUT_OT_tool_remesh(_ObjectTool):
    """EVEN REMESH: rebuild the surface as an even grid of quads (no
    triangles, all the same size) and make it watertight. Fine detail
    smaller than Voxel Size is lost"""
    bl_idname = "vibecut.tool_remesh"
    bl_label = "Even Quads (Voxel Remesh)"

    def execute(self, ctx):
        p = _props(ctx)
        lines = []
        for ob in _mesh_objects(ctx):
            note = _close_for_remesh(ob)
            s0 = mesh_report(ob)["shells"]
            f0, f1, v = voxel_remesh(ob, p.voxel)
            s1 = mesh_report(ob)["shells"]
            lines.append("%s: %d -> %d quad faces%s%s" % (
                ob.name, f0, f1, note, "" if v == p.voxel else
                " (voxel raised to %.2f mm to stay under %d faces)"
                % (v, REMESH_MAX_FACES)))
            if s1 > s0:
                lines.append("WARNING: %d piece(s) broke off - parts thinner "
                             "than the voxel. Undo and use a smaller Voxel "
                             "Size" % (s1 - s0))
        _set_status(ctx, lines)
        return {'FINISHED'}


class VIBECUT_OT_tool_quadriflow(_ObjectTool):
    """Flowing quads that follow the shape (QuadriFlow) at Target Faces.
    Needs a watertight mesh - runs a fine Voxel Remesh first if not"""
    bl_idname = "vibecut.tool_quadriflow"
    bl_label = "Flowing Quads (QuadriFlow)"

    def execute(self, ctx):
        p = _props(ctx)
        lines = []
        for ob in _mesh_objects(ctx):
            if not mesh_report(ob)["watertight"]:
                _close_for_remesh(ob)
                voxel_remesh(ob, max(0.05, p.voxel * 0.5))
            ctx.view_layer.objects.active = ob
            f0 = len(ob.data.polygons)
            try:
                bpy.ops.object.quadriflow_remesh(
                    use_mesh_symmetry=False, use_preserve_sharp=True,
                    use_preserve_boundary=False, mode='FACES',
                    target_faces=p.target_faces, seed=0)
            except Exception as e:
                self.report({'ERROR'}, "QuadriFlow failed on %s: %s"
                            % (ob.name, e))
                return {'CANCELLED'}
            lines.append("%s: %d -> %d faces" % (ob.name, f0,
                                                 len(ob.data.polygons)))
        _set_status(ctx, lines)
        return {'FINISHED'}


class VIBECUT_OT_tool_decimate(_ObjectTool):
    """Reduce a heavy mesh to about Target Faces (keeps the shape)"""
    bl_idname = "vibecut.tool_decimate"
    bl_label = "Reduce Faces"

    def execute(self, ctx):
        p = _props(ctx)
        _set_status(ctx, ["%s: %d -> %d faces" % ((o.name,)
                          + decimate_to(o, p.target_faces))
                          for o in _mesh_objects(ctx)])
        return {'FINISHED'}


class VIBECUT_OT_tool_scale(_ObjectTool):
    """Scale uniformly so the chosen side is exactly Size mm (AI models
    often come in at 1-2 units = meters)"""
    bl_idname = "vibecut.tool_scale"
    bl_label = "Scale to Size"

    def execute(self, ctx):
        p = _props(ctx)
        _set_status(ctx, ["%s: x%.3f" % (o.name, scale_to_size(
            o, p.size_mm, p.size_axis)) for o in _mesh_objects(ctx)])
        return {'FINISHED'}


class VIBECUT_OT_tool_flat_bottom(_ObjectTool):
    """Slice the model flat Cut Height above its lowest point and cap it,
    so a figure stands on the bed without supports under its feet"""
    bl_idname = "vibecut.tool_flat_bottom"
    bl_label = "Flat Bottom"

    def execute(self, ctx):
        p = _props(ctx)
        _set_status(ctx, ["%s: cut + %d cap(s)" % (o.name, flat_bottom(
            o, p.bottom_cut)) for o in _mesh_objects(ctx)])
        return {'FINISHED'}


class VIBECUT_OT_tool_symmetrize(_ObjectTool):
    """Mirror one half onto the other across the object's own origin
    (fixes lopsided AI figures). Put the origin on the centre line first"""
    bl_idname = "vibecut.tool_symmetrize"
    bl_label = "Symmetrize"

    def execute(self, ctx):
        p = _props(ctx)
        _bm_tool(ctx, lambda bm: symmetrize(
            bm, p.sym_dir, max(0.002, _diag(bm) * 1e-5)),
            lambda r: "mirrored")
        return {'FINISHED'}


class VIBECUT_OT_tool_drop(_ObjectTool):
    """Move the model so its lowest point sits on the bed (Z = 0)"""
    bl_idname = "vibecut.tool_drop"
    bl_label = "Drop to Bed"

    def execute(self, ctx):
        _set_status(ctx, [o.name + ": " + drop_to_floor(o)
                          for o in _mesh_objects(ctx)])
        return {'FINISHED'}


class _EditTool(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'EDIT_MESH' and ctx.active_object is not None
                and ctx.active_object.type == 'MESH')


class VIBECUT_OT_tool_make_round(_EditTool):
    """Push the selected vertices onto their best-fit sphere (round out a
    dented ball, socket or eye)"""
    bl_idname = "vibecut.tool_make_round"
    bl_label = "Make Round"

    def execute(self, ctx):
        ob = ctx.active_object
        bm = bmesh.from_edit_mesh(ob.data)
        sel = [v for v in bm.verts if v.select]
        c, r, mv = make_round(bm, sel)
        if c is None:
            self.report({'WARNING'}, "Select at least 4 vertices")
            return {'CANCELLED'}
        bmesh.update_edit_mesh(ob.data)
        _set_status(ctx, ["sphere r %.3f mm, moved up to %.3f mm" % (r, mv)])
        return {'FINISHED'}


class VIBECUT_OT_tool_fill_round(_EditTool):
    """Fill the selected hole(s) - or every hole if nothing is selected -
    with a patch that follows the curve of the surface around it (a round
    fill, not a flat one)"""
    bl_idname = "vibecut.tool_fill_round"
    bl_label = "Fill Hole Round"

    def execute(self, ctx):
        ob = ctx.active_object
        bm = bmesh.from_edit_mesh(ob.data)
        bnd = [e for e in bm.edges if e.is_boundary]
        sel = [e for e in bnd if e.select]
        loops = _boundary_loops(sel or bnd)
        if not loops:
            self.report({'WARNING'}, "No holes found")
            return {'CANCELLED'}
        out = []
        for loop in loops:
            c, r, n = fill_hole_round(bm, loop)
            out.append("r %.2f, %d faces" % (r, n))
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bmesh.update_edit_mesh(ob.data)
        _set_status(ctx, ["filled %d hole(s): " % len(loops)
                          + "; ".join(out)])
        return {'FINISHED'}


class VIBECUT_OT_tool_snap_loop(_EditTool):
    """Select TWO separate edge loops. Moves one onto the other (snaps each
    vertex to the nearest point of the other loop) and welds them into one
    watertight seam - e.g. to reattach a lid rim or close a gap"""
    bl_idname = "vibecut.tool_snap_loop"
    bl_label = "Snap Loop to Loop"

    def execute(self, ctx):
        ob = ctx.active_object
        p = _props(ctx)
        bm = bmesh.from_edit_mesh(ob.data)
        try:
            r = snap_loop_to_loop(bm, ob.matrix_world, p.loop_move,
                                  p.loop_weld, p.loop_zip)
        except ValueError as e:
            self.report({'WARNING'}, str(e))
            return {'CANCELLED'}
        bmesh.update_edit_mesh(ob.data)
        lines = ["moved %d verts onto a %d-vert loop (max %.3f)"
                 % (r["moved"], r["target"], r["max_move"])]
        if r["missed"]:
            lines.append("WARNING: %d verts found no target edge"
                         % r["missed"])
        if p.loop_weld:
            lines.append("welded %d, zipped %d extra" % (r["welded"],
                                                         r["zipped"]))
            lines.append("seam: %s" % ("watertight"
                                       if not r["seam_open_edges"] else
                                       "%d open/odd edges"
                                       % r["seam_open_edges"]))
        _set_status(ctx, lines)
        return {'FINISHED'}


class VIBECUT_OT_tool_straighten_wall(_EditTool):
    """Select a jagged wall strip (faces or vertices). Makes the faces stand
    straight up off the XY plane while keeping the strip's curve, and lines
    the top and bottom edges up corner to corner - stray vertices included"""
    bl_idname = "vibecut.tool_straighten_wall"
    bl_label = "Straighten Wall Band"

    def execute(self, ctx):
        ob = ctx.active_object
        p = _props(ctx)
        bm = bmesh.from_edit_mesh(ob.data)
        sel = [v for v in bm.verts if v.select]
        r = straighten_wall(sel, ob.matrix_world, p.wall_vertical,
                            p.wall_edges, p.wall_smooth)
        if r is None:
            self.report({'WARNING'}, "Select a connected wall strip "
                        "(at least 6 vertices)")
            return {'CANCELLED'}
        bm.normal_update()
        bmesh.update_edit_mesh(ob.data)
        _set_status(ctx, ["wall: %d verts along %.1f mm" % (r["verts"],
                                                             r["length"]),
                          "moved in/out up to %.2f mm, up/down up to %.2f mm"
                          % (r["moved_xy"], r["moved_z"])])
        return {'FINISHED'}


class VIBECUT_OT_tool_thin_walls(_EditTool):
    """Select faces where the wall is thinner than Min Wall - these break
    or vanish when printed (check limbs, joints, socket rims)"""
    bl_idname = "vibecut.tool_thin_walls"
    bl_label = "Select Thin Walls"

    def execute(self, ctx):
        ob = ctx.active_object
        p = _props(ctx)
        bpy.ops.object.mode_set(mode='OBJECT')
        thin = set(thin_wall_faces(ob, p.thin_mm))
        me = ob.data
        for v in me.vertices:
            v.select = False
        for e in me.edges:
            e.select = False
        for poly in me.polygons:
            poly.select = poly.index in thin
            if poly.select:
                for vi in poly.vertices:
                    me.vertices[vi].select = True
        for e in me.edges:
            e.select = all(me.vertices[i].select for i in e.vertices)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='FACE')
        pct = 100.0 * len(thin) / max(len(ob.data.polygons), 1)
        _set_status(ctx, ["%d faces (%.1f%%) thinner than %.2f mm"
                          % (len(thin), pct, p.thin_mm)])
        return {'FINISHED'}


class _ToolsSection:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VibeCut"
    bl_parent_id = "VIBECUT_PT_tools"
    _icon = 'NONE'

    def draw_header(self, ctx):
        self.layout.label(icon=self._icon)


def _edit_hint(layout, ctx):
    if ctx.mode == 'OBJECT':
        layout.label(text="Tab into Edit Mode to use these.", icon='INFO')


class VIBECUT_PT_tools(bpy.types.Panel):
    bl_label = "Tools"
    bl_idname = "VIBECUT_PT_tools"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VibeCut"

    def draw(self, ctx):
        L = self.layout
        p = _props(ctx)
        obj_mode = ctx.mode == 'OBJECT'

        big = L.column()
        big.scale_y = 1.6
        big.enabled = obj_mode
        big.operator("vibecut.prep_new_model", icon='SHADERFX')
        row = L.row()
        row.prop(p, "show_prep_options", icon='PREFERENCES', toggle=True)
        row.operator("vibecut.print_check", icon='VIEWZOOM')
        if p.show_prep_options:
            b = L.box()
            c = b.column(align=True)
            c.prop(p, "do_apply")
            c.prop(p, "do_fill")
            c.prop(p, "do_dissolve")
            c.prop(p, "do_floor")
            c = b.column(align=True)
            c.prop(p, "merge_dist")
            c.prop(p, "island_pct")
            c.prop(p, "max_hole")
            c.prop(p, "dissolve_angle")
        if p.status:
            b = L.box()
            for chunk in p.status.split(" | "):
                b.label(text=chunk)
        ob = ctx.active_object
        if ctx.mode == 'EDIT_MESH' and ob is not None and ob.type == 'MESH':
            r = L.row()
            r.scale_y = 0.85
            r.label(text="V: {:,}   F: {:,}".format(
                len(ob.data.vertices), len(ob.data.polygons)), icon='MESH_DATA')


class VIBECUT_PT_tools_cleanup(_ToolsSection, bpy.types.Panel):
    bl_label = "Cleanup Steps"
    bl_idname = "VIBECUT_PT_tools_cleanup"
    _icon = 'BRUSH_DATA'

    def draw(self, ctx):
        c = self.layout.column(align=True)
        c.enabled = ctx.mode == 'OBJECT'
        c.operator("vibecut.tool_apply", icon='OBJECT_ORIGIN')
        c.operator("vibecut.tool_merge", icon='AUTOMERGE_OFF')
        c.operator("vibecut.tool_loose", icon='TRASH')
        c.operator("vibecut.tool_degenerate", icon='X')
        c.operator("vibecut.tool_fill", icon='MESH_PLANE')
        c.operator("vibecut.tool_normals", icon='NORMALS_FACE')
        c.operator("vibecut.tool_dissolve", icon='MOD_DECIM')
        c.operator("vibecut.tool_drop", icon='TRIA_DOWN_BAR')


class VIBECUT_PT_tools_quads(_ToolsSection, bpy.types.Panel):
    bl_label = "Triangles -> Quads"
    bl_idname = "VIBECUT_PT_tools_quads"
    _icon = 'MESH_GRID'

    def draw(self, ctx):
        p = _props(ctx)
        S = ctx.scene
        c = self.layout.column(align=True)
        c.enabled = ctx.mode == 'OBJECT'
        c.prop(p, "voxel")
        c.operator("vibecut.tool_remesh", icon='MOD_REMESH')
        c.prop(p, "target_faces")
        c.operator("vibecut.tool_quadriflow", icon='MOD_REMESH')
        c.operator("vibecut.tool_decimate", icon='MOD_DECIM')
        c.separator()
        c.prop(S, "vibecut_mt_quad_sharp")
        c.prop(S, "vibecut_mt_quad_planar")
        c.prop(S, "vibecut_mt_quad_duplicate")
        c.operator("vibecut.quad_convert", text="Flat Panels (keep shape)",
                   icon='MOD_TRIANGULATE')
        c.separator()
        c.prop(p, "smooth_iter")
        c.prop(p, "smooth_strength")
        c.operator("vibecut.tool_smooth", icon='MOD_SMOOTH')


class VIBECUT_PT_tools_print(_ToolsSection, bpy.types.Panel):
    bl_label = "Print & Articulation"
    bl_idname = "VIBECUT_PT_tools_print"
    bl_options = {'DEFAULT_CLOSED'}
    _icon = 'MOD_BUILD'

    def draw(self, ctx):
        p = _props(ctx)
        c = self.layout.column(align=True)
        c.enabled = ctx.mode == 'OBJECT'
        r = c.row(align=True)
        r.prop(p, "size_mm")
        r.prop(p, "size_axis", text="")
        c.operator("vibecut.tool_scale", icon='FULLSCREEN_ENTER')
        c.prop(p, "bottom_cut")
        c.operator("vibecut.tool_flat_bottom", icon='MOD_BOOLEAN')
        c.prop(p, "sym_dir")
        c.operator("vibecut.tool_symmetrize", icon='MOD_MIRROR')


class VIBECUT_PT_tools_bevel(_ToolsSection, bpy.types.Panel):
    bl_label = "Auto Bevel"
    bl_idname = "VIBECUT_PT_tools_bevel"
    bl_options = {'DEFAULT_CLOSED'}
    _icon = 'MOD_BEVEL'

    def draw(self, ctx):
        S = ctx.scene
        L = self.layout
        c = L.column(align=True)
        c.prop(S, "vibecut_mt_bevel_width")
        c.prop(S, "vibecut_mt_bevel_segments")
        c.prop(S, "vibecut_mt_bevel_angle")
        r = L.row()
        r.scale_y = 1.2
        if ctx.mode == 'EDIT_MESH':
            r.operator("vibecut.bevel_sharp_edges", icon='MOD_BEVEL')
        else:
            r.operator("vibecut.auto_bevel", icon='MOD_BEVEL')


class VIBECUT_PT_tools_click_collapse(_ToolsSection, bpy.types.Panel):
    bl_label = "Click Collapse"
    bl_idname = "VIBECUT_PT_tools_click_collapse"
    _icon = 'AUTOMERGE_ON'

    def draw(self, ctx):
        L = self.layout
        cap = L.column(align=True)
        cap.scale_y = 0.8
        cap.label(text="Click a vertex: it merges into the")
        cap.label(text="closest vertex it is connected to.")
        cap.label(text="Orange = goes away, green = stays.")
        running = getattr(ctx.window_manager, "vibecut_cc_running", False)
        r = L.row()
        r.scale_y = 1.4
        r.enabled = ctx.mode != 'OBJECT'
        if running:
            r.operator("vibecut.click_collapse", text="Stop Click Collapse",
                       icon='CANCEL', depress=True)
            L.label(text="Right-click or Esc also stops.", icon='INFO')
        else:
            r.operator("vibecut.click_collapse", icon='AUTOMERGE_ON')
        _edit_hint(L, ctx)


class VIBECUT_PT_tools_round(_ToolsSection, bpy.types.Panel):
    bl_label = "Round & Thin"
    bl_idname = "VIBECUT_PT_tools_round"
    bl_parent_id = "VIBECUT_PT_more"
    bl_order = 5
    bl_options = {'DEFAULT_CLOSED'}
    _icon = 'SPHERECURVE'

    def draw(self, ctx):
        p = _props(ctx)
        L = self.layout
        c = L.column(align=True)
        c.enabled = ctx.mode != 'OBJECT'
        c.label(text="Select part of the mesh first.")
        c.operator("vibecut.tool_fill_round", icon='SPHERE')
        c.operator("vibecut.tool_make_round", icon='SPHERECURVE')
        c.prop(p, "thin_mm")
        c.operator("vibecut.tool_thin_walls", icon='MOD_THICKNESS')
        _edit_hint(L, ctx)


class VIBECUT_PT_tools_flat(_ToolsSection, bpy.types.Panel):
    bl_label = "Flat Region"
    bl_idname = "VIBECUT_PT_tools_flat"
    bl_options = {'DEFAULT_CLOSED'}
    _icon = 'SNAP_FACE'

    def draw(self, ctx):
        S = ctx.scene
        L = self.layout
        c = L.column(align=True)
        c.enabled = ctx.mode == 'EDIT_MESH'
        cap = c.column(align=True)
        cap.scale_y = 0.8
        cap.label(text="Make a nearly-flat patch truly flat:")
        cap.label(text="click one face, Select, then Flatten.")
        c.prop(S, "vibecut_mt_flat_angle")
        c.prop(S, "vibecut_mt_flatten_boundary")
        c.prop(S, "vibecut_mt_flatten_influence")
        col = c.column(align=True)
        col.scale_y = 1.2
        col.operator("vibecut.select_flat_region", text="Select Flat Region",
                     icon='RESTRICT_SELECT_OFF')
        col.operator("vibecut.flatten_region", text="Flatten Region",
                     icon='MOD_SMOOTH')
        _edit_hint(L, ctx)


class VIBECUT_PT_quick(bpy.types.Panel):
    """Edit Mode fixers: select part of the mesh, then click"""
    bl_label = "Quick Tools"
    bl_idname = "VIBECUT_PT_quick"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VibeCut"
    bl_order = 50

    def draw(self, ctx):
        p = _props(ctx)
        S = ctx.scene
        L = self.layout
        c = L.column()
        c.enabled = ctx.mode == 'EDIT_MESH'
        cap = c.column(align=True)
        cap.scale_y = 0.8
        cap.label(text="Select part of the mesh, then click.")

        col = c.column(align=True)
        col.prop(p, "wall_vertical")
        col.prop(p, "wall_edges", text="")
        col.prop(p, "wall_smooth")
        col.operator("vibecut.tool_straighten_wall", icon='CURVE_PATH')

        col = c.column(align=True)
        col.prop(S, "vibecut_mt_straighten_count")
        col.operator("vibecut.straighten_to_line", text="Straighten to Line",
                     icon='IPO_LINEAR')
        col.operator("vibecut.straighten_to_axis", text="Straighten to Axis",
                     icon='EMPTY_AXIS')
        col.operator("vibecut.smooth_curve_line", text="Curve Line",
                     icon='IPO_EASE_IN_OUT')

        col = c.column(align=True)
        col.prop(p, "loop_move", text="Move")
        col.prop(p, "loop_weld")
        col.prop(p, "loop_zip")
        col.operator("vibecut.tool_snap_loop", icon='SNAP_EDGE')

        col = c.column(align=True)
        col.operator("vibecut.dissolve_line_interior", text="Dissolve Interior",
                     icon='X')
        col.operator("vibecut.color_shell_menu", text="Color Piece",
                     icon='COLOR')
        _edit_hint(L, ctx)


class VIBECUT_PT_more(bpy.types.Panel):
    """Less-used tools, folded away"""
    bl_label = "More..."
    bl_idname = "VIBECUT_PT_more"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VibeCut"
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 100

    def draw(self, ctx):
        pass


_classes = (
    VIBECUT_OT_prep_new_model, VIBECUT_OT_print_check,
    VIBECUT_OT_tool_apply, VIBECUT_OT_tool_merge, VIBECUT_OT_tool_loose,
    VIBECUT_OT_tool_degenerate, VIBECUT_OT_tool_fill,
    VIBECUT_OT_tool_normals, VIBECUT_OT_tool_dissolve,
    VIBECUT_OT_tool_smooth, VIBECUT_OT_tool_remesh,
    VIBECUT_OT_tool_quadriflow, VIBECUT_OT_tool_decimate,
    VIBECUT_OT_tool_scale, VIBECUT_OT_tool_flat_bottom,
    VIBECUT_OT_tool_symmetrize, VIBECUT_OT_tool_drop,
    VIBECUT_OT_tool_make_round, VIBECUT_OT_tool_fill_round,
    VIBECUT_OT_tool_thin_walls, VIBECUT_OT_tool_straighten_wall,
    VIBECUT_OT_tool_snap_loop,
    VIBECUT_PT_tools, VIBECUT_PT_tools_cleanup, VIBECUT_PT_tools_quads,
    VIBECUT_PT_tools_print, VIBECUT_PT_tools_bevel,
    VIBECUT_PT_tools_click_collapse, VIBECUT_PT_tools_flat,
    VIBECUT_PT_quick, VIBECUT_PT_more,
    VIBECUT_PT_tools_round,
)


def register_props():
    bpy.utils.register_class(VibeCutToolProps)
    bpy.types.Scene.vibecut_tools = bpy.props.PointerProperty(
        type=VibeCutToolProps)


def unregister_props():
    del bpy.types.Scene.vibecut_tools
    bpy.utils.unregister_class(VibeCutToolProps)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
