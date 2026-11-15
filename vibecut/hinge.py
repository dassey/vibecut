import math

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

GUIDE_NAME = "VC_HingeGuide"

GROW_STEPS = ((1.0, 1.0), (1.15, 1.0), (1.3, 1.0), (1.3, 1.25), (1.45, 1.25),
              (1.3, 1.5), (1.6, 1.5), (1.6, 2.0))

DEFAULTS = {
    "knuckle_d": 0.0,
    "pin_d": 3.6,
    "clearance": 0.35,
    "gap": 0.4,
    "count": 5,
    "leaf_reach": 2.5,
    "pin_mode": 'PIP',
    "pin_fit": 0.10,
    "swap": False,
    "segments": 64,
}


def _co(me):
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def _components(me):
    n = len(me.vertices)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    lab = np.arange(n)
    for _ in range(200):
        old = lab.copy()
        a = lab[ev[:, 0]]
        b = lab[ev[:, 1]]
        m = np.minimum(a, b)
        np.minimum.at(lab, a, m)
        np.minimum.at(lab, b, m)
        for _ in range(8):
            lab = lab[lab]
        if np.array_equal(lab, old):
            break
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def _face_arrays(me):
    npo = len(me.polygons)
    ls = np.empty(npo, dtype=np.int64)
    lt = np.empty(npo, dtype=np.int64)
    me.polygons.foreach_get("loop_start", ls)
    me.polygons.foreach_get("loop_total", lt)
    lv = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", lv)
    return ls, lt, lv


def mesh_subset(me, face_mask, name):
    ls, lt, lv = _face_arrays(me)
    sls, slt = ls[face_mask], lt[face_mask]
    offs = np.cumsum(slt) - slt
    total = int(slt.sum())
    loop_idx = np.repeat(sls - offs, slt) + np.arange(total)
    old_v = lv[loop_idx]
    used, new_lv = np.unique(old_v, return_inverse=True)
    co = _co(me)[used]
    out = bpy.data.meshes.new(name)
    out.vertices.add(len(used))
    out.vertices.foreach_set("co", co.ravel())
    out.loops.add(total)
    out.loops.foreach_set("vertex_index", new_lv.astype(np.int32))
    out.polygons.add(len(slt))
    out.polygons.foreach_set("loop_start", offs.astype(np.int32))
    out.polygons.foreach_set("loop_total", slt.astype(np.int32))
    out.update(calc_edges=True)
    return out


def drop_crumbs(ob, min_frac=0.001):
    lab = _components(ob.data)
    cnt = np.bincount(lab)
    keep = cnt >= max(min_frac * cnt.sum(), 1)
    if keep.all():
        return 0
    ls, _, lv = _face_arrays(ob.data)
    new = mesh_subset(ob.data, keep[lab[lv[ls]]], ob.data.name)
    old = ob.data
    ob.data = new
    bpy.data.meshes.remove(old)
    new.name = ob.name
    return int(cnt[~keep].sum())


def micro_repair(ob, max_bad=200):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bad = [e for e in bm.edges if not e.is_manifold]
    if not bad or len(bad) > max_bad:
        bm.free()
        return len(bad)
    faces = {f for e in bad for f in e.link_faces}
    for _ in range(2):
        faces |= {g for f in faces for v in f.verts for g in v.link_faces}
    bmesh.ops.delete(bm, geom=list(faces), context='FACES')
    loose = [v for v in bm.verts if not v.link_faces]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context='VERTS')
    wire = [e for e in bm.edges if not e.link_faces]
    if wire:
        bmesh.ops.delete(bm, geom=wire, context='EDGES')
    res = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary],
                               sides=0)
    if res["faces"]:
        bmesh.ops.triangulate(bm, faces=res["faces"])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    left = sum(1 for e in bm.edges if not e.is_manifold)
    if left < len(bad):
        bm.to_mesh(ob.data)
        ob.data.update()
    else:
        left = len(bad)
    bm.free()
    return left


def non_manifold_edges(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    n = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    return n


def _frame(axis):
    a = Vector(axis).normalized()
    ref = Vector((0, 0, 1)) if abs(a.z) < 0.9 else Vector((1, 0, 0))
    u = a.cross(ref).normalized()
    v = a.cross(u).normalized()
    return a, u, v


def _hull_mesh(points, name):
    bm = bmesh.new()
    for p in points:
        bm.verts.new(p)
    res = bmesh.ops.convex_hull(bm, input=list(bm.verts),
                                use_existing_faces=False)
    junk = list({g for g in res["geom_interior"] + res["geom_unused"]
                 if isinstance(g, bmesh.types.BMVert)})
    if junk:
        bmesh.ops.delete(bm, geom=junk, context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return me


def _ring(c, a, u, v, t, r, seg):
    return [c + a * t + (u * math.cos(k * 2 * math.pi / seg)
                         + v * math.sin(k * 2 * math.pi / seg)) * r
            for k in range(seg)]


def cylinder_points(c, axis, t0, t1, r, seg):
    a, u, v = _frame(axis)
    return _ring(c, a, u, v, t0, r, seg) + _ring(c, a, u, v, t1, r, seg)


def knuckle_points(c, axis, t0, t1, r, leaf_dir, leaf_len, seg):
    a, _, _ = _frame(axis)
    d = Vector(leaf_dir)
    d = (d - a * d.dot(a)).normalized()
    w = a.cross(d).normalized()
    pts = cylinder_points(c, a, t0, t1, r, seg)
    for t in (t0, t1):
        for s in (-1.0, 1.0):
            pts.append(c + a * t + w * (s * r) + d * leaf_len)
    return pts


def _merge_meshes(meshes, name):
    bm = bmesh.new()
    for m in meshes:
        bm.from_mesh(m)
    out = bpy.data.meshes.new(name)
    bm.to_mesh(out)
    bm.free()
    for m in meshes:
        bpy.data.meshes.remove(m)
    return out


def _link(ob):
    bpy.context.scene.collection.objects.link(ob)


def _remove_with_mesh(ob):
    me = ob.data
    bpy.data.objects.remove(ob)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def boolean(target, tool_mesh, op):
    tool = bpy.data.objects.new("_vc_hinge_tool", tool_mesh)
    _link(tool)
    mod = target.modifiers.new("_vc_hinge", 'BOOLEAN')
    mod.operation = op
    mod.solver = 'MANIFOLD'
    mod.object = tool
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    new = bpy.data.meshes.new_from_object(target.evaluated_get(dg))
    target.modifiers.remove(mod)
    old = target.data
    target.data = new
    new.name = old.name
    if old.users == 0:
        bpy.data.meshes.remove(old)
    bpy.data.objects.remove(tool)
    bpy.data.meshes.remove(tool_mesh)
    return target


def guide_from_object(g):
    M = g.matrix_world
    c = M.translation.copy()
    ax = M.col[2].xyz
    h = ax.length
    r = (M.col[0].xyz.length + M.col[1].xyz.length) / 2
    return c, ax.normalized(), h, r


def make_guide(center, axis, half_len, radius):
    g = bpy.data.objects.get(GUIDE_NAME)
    if g is None:
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=1.0,
                              radius2=1.0, depth=2.0)
        me = bpy.data.meshes.new(GUIDE_NAME)
        bm.to_mesh(me)
        bm.free()
        g = bpy.data.objects.new(GUIDE_NAME, me)
        _link(g)
    g.display_type = 'WIRE'
    g.show_in_front = True
    g.hide_render = True
    a = Vector(axis).normalized()
    rot = a.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    g.matrix_world = (Matrix.Translation(center) @ rot
                      @ Matrix.Diagonal((radius, radius, half_len, 1.0)))
    return g


def guess_guide(ob, cursor):
    M = ob.matrix_world
    pts = [M @ Vector(b) for b in ob.bound_box]
    mn = Vector([min(p[i] for p in pts) for i in range(3)])
    mx = Vector([max(p[i] for p in pts) for i in range(3)])
    ctr = (mn + mx) / 2
    d = Vector((cursor.x - ctr.x, cursor.y - ctr.y, 0))
    if d.length < 1e-6:
        d = Vector((1, 0, 0))
    axis = Vector((0, 0, 1)).cross(d.normalized()).normalized()
    ext = abs((mx - mn).dot(Vector([abs(x) for x in axis])))
    return axis, max(ext * 0.35, 1.0) / 2, max((mx - mn)) * 0.04


def _kasa(p):
    A = np.c_[2 * p, np.ones(len(p))]
    b = (p ** 2).sum(1)
    s = np.linalg.lstsq(A, b, rcond=None)[0]
    c = s[:2]
    return c, math.sqrt(max(s[2] + c @ c, 1e-12))


def _section(uv, t, tri, ts):
    d = t[tri] - ts
    s = d > 0
    cross = s.any(1) & ~s.all(1)
    tri, d = tri[cross], d[cross]
    if not len(tri):
        return np.zeros((0, 2))
    ends = np.zeros((len(tri), 2, 2))
    filled = np.zeros(len(tri), dtype=np.int64)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        k = np.flatnonzero(((d[:, i] > 0) != (d[:, j] > 0)) & (filled < 2))
        f = d[k, i] / (d[k, i] - d[k, j])
        ends[k, filled[k]] = (uv[tri[k, i]]
                              + (uv[tri[k, j]] - uv[tri[k, i]]) * f[:, None])
        filled[k] += 1
    ends = ends[filled == 2]
    w = np.linspace(0, 1, 9)[None, :, None]
    return (ends[:, :1] * (1 - w) + ends[:, 1:] * w).reshape(-1, 2)


def fit_fake_hinge(ob, center, axis, half_len, radius):
    a, u, v = _frame(axis)
    M = np.array(ob.matrix_world)
    co = _co(ob.data) @ M[:3, :3].T + M[:3, 3]
    rel = co - np.array(center)
    t = rel @ np.array(a)
    uv = np.c_[rel @ np.array(u), rel @ np.array(v)]
    me = ob.data
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    tri = tri[np.abs(t[tri]).min(1) <= half_len * 1.01]
    cs, rs, errs = [], [], []
    for frac in (-0.4, -0.2, 0.0, 0.2, 0.4):
        S = _section(uv, t, tri, frac * half_len)
        if len(S) < 20:
            continue
        c = np.zeros(2)
        R = radius
        for _ in range(6):
            d = S - c
            r = np.hypot(d[:, 0], d[:, 1])
            ang = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
            pts = []
            for b in range(-180, 180, 10):
                k = (ang >= b) & (ang < b + 10) & (r < R * 1.25) & (r > R * 0.3)
                if k.any():
                    pts.append(S[k][np.argmax(r[k])])
            if len(pts) < 8:
                break
            pts = np.array(pts)
            c, R = _kasa(pts)
        else:
            res = np.hypot(*(pts - c).T) - R
            cs.append(c)
            rs.append(R)
            errs.append(float(np.sqrt((res ** 2).mean())))
    if not cs:
        return None
    c = np.median(np.array(cs), axis=0)
    new_center = Vector(center) + u * float(c[0]) + v * float(c[1])
    return new_center, float(np.median(rs)), float(np.median(errs))


def _rotate(P, c, a, ang):
    a = np.array(a) / np.linalg.norm(a)
    p = P - c
    cos, sin = math.cos(ang), math.sin(ang)
    return (p * cos + np.cross(a, p) * sin
            + np.outer(p @ a, a) * (1 - cos)) + c


def _surface_samples(me, P, c, a, near_r, samples, seed):
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    A, B, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    area = np.linalg.norm(np.cross(B - A, C - A), axis=1) * 0.5
    cen = (A + B + C) / 3 - c
    radial = np.linalg.norm(cen - np.outer(cen @ a, a), axis=1)
    rng = np.random.default_rng(seed)
    out = []
    for mask in (radial < near_r, radial >= near_r):
        idx = np.flatnonzero(mask & (area > 0))
        if not len(idx):
            continue
        w = area[idx] / area[idx].sum()
        pick = rng.choice(idx, samples // 2, p=w)
        u, v = rng.random(len(pick)), rng.random(len(pick))
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        out.append(A[pick] + (B[pick] - A[pick]) * u[:, None]
                   + (C[pick] - A[pick]) * v[:, None])
    return np.concatenate(out) if out else P[:samples]


def swing_test(body, lid, center, axis, hinge_r, step_deg=3.0, max_deg=180,
               samples=6000, depth=0.15, seed=1):
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(body, dg)
    P = _co(lid.data)
    c = np.array(center)
    a = np.array(Vector(axis).normalized())
    S = _surface_samples(lid.data, P, c, a, hinge_r * 3, samples, seed)
    body_c = _co(body.data).mean(0)

    rays = (Vector((0.267, 0.534, 0.802)).normalized(),
            Vector((-0.577, -0.408, 0.707)).normalized())

    def inside(p):
        loc, nor, _, dist = bvh.find_nearest(p)
        if loc is None or dist <= depth or (p - loc).dot(nor) >= 0:
            return False
        for r in rays:
            hit, hn, _, _ = bvh.ray_cast(p, r)
            if hit is None or hn.dot(r) <= 0:
                return False
        return True

    def hits(ang):
        R = _rotate(S, c, a, math.radians(ang))
        n = 0
        for p in R:
            if inside(Vector(p)):
                n += 1
                if n >= 3:
                    return True
        return False

    out = {"at_0_collides": hits(0.0)}
    for sign in (1, -1):
        free = 0.0
        ang = step_deg
        while ang <= max_deg:
            if hits(sign * ang):
                break
            free = ang
            ang += step_deg
        out[sign] = free
    lc = P.mean(0)
    d_plus = np.linalg.norm(_rotate(lc[None], c, a, math.radians(5))[0] - body_c)
    d_minus = np.linalg.norm(_rotate(lc[None], c, a, math.radians(-5))[0] - body_c)
    close = 1 if d_plus < d_minus else -1
    out["close_deg"] = out[close]
    out["open_deg"] = out[-close]
    return out


_RAYS = (Vector((0.267, 0.534, 0.802)).normalized(),
         Vector((-0.577, -0.408, 0.707)).normalized())


def _signed_dist(bvh, Q, max_d, near=5.0):
    out = np.empty(len(Q))
    for i, p in enumerate(Q):
        v = Vector(p)
        loc, nor, _, d = bvh.find_nearest(v, max_d)
        if loc is None:
            out[i] = max_d
            continue
        if d > near:
            out[i] = d
            continue
        inside = True
        for r in _RAYS:
            hit, hn, _, _ = bvh.ray_cast(v, r)
            if hit is None or hn.dot(r) <= 0:
                inside = False
                break
        out[i] = (-d if d > 0.15 else 0.0) if inside else d
    return out


def _area_samples(me, n, seed):
    P = _co(me)
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    A, B, C = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    area = np.linalg.norm(np.cross(B - A, C - A), axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    return (A[pick] + (B[pick] - A[pick]) * u[:, None]
            + (C[pick] - A[pick]) * v[:, None])


def _close_sign(body, lid, c, a):
    lc = _co(lid.data).mean(0)
    bc = _co(body.data).mean(0)
    dp = np.linalg.norm(_rotate(lc[None], c, a, math.radians(5))[0] - bc)
    dm = np.linalg.norm(_rotate(lc[None], c, a, math.radians(-5))[0] - bc)
    return 1 if dp < dm else -1


def rim_stats(bvh, rim, c, a, ang_deg, bins=18):
    R = _rotate(rim, c, a, math.radians(ang_deg))
    s = _signed_dist(bvh, R, 30.0)
    cen = R.mean(0)
    th = np.degrees(np.arctan2(R[:, 1] - cen[1], R[:, 0] - cen[0]))
    step = 360 // bins
    rows = []
    for b in range(-180, 180, step):
        k = (th >= b) & (th < b + step)
        if k.sum() >= 5:
            rows.append((b, float(s[k].min()), float((s[k] < -0.15).mean())))
    return rows


def rim_report(bvh, rim, c, a, ang_deg, bins=18):
    return [(b, m) for b, m, _ in rim_stats(bvh, rim, c, a, ang_deg, bins)]


def fit_closed_pose(body, lid, center, axis, exclude_r, samples=24000,
                    rim_n=2000, seed=3, orig_ob=None, reach_r=3.0):
    dg = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(body, dg)
    c0 = np.array(center)
    a = np.array(Vector(axis).normalized())
    _, u, v = _frame(Vector(a))
    u, v = np.array(u), np.array(v)
    sgn = _close_sign(body, lid, c0, a)
    S = _area_samples(lid.data, samples, seed)
    rel = S - c0
    rad = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
    S = S[rad > exclude_r]
    rng = np.random.default_rng(seed)
    probe = S[rng.choice(len(S), min(3000, len(S)), replace=False)]
    th0 = None
    for d in range(2, 121, 2):
        sd = _signed_dist(bvh, _rotate(probe, c0, a, math.radians(sgn * d)),
                          30.0)
        if ((sd < -0.3) & (sd > -8.0)).sum() >= max(5, len(probe) // 500):
            th0 = d
            break
    if th0 is None:
        return None
    sd = _signed_dist(bvh, _rotate(S, c0, a, math.radians(sgn * th0)), 30.0)
    rim = S[np.abs(sd) < 10.0]
    if len(rim) < 50:
        return None
    rim = rim[rng.choice(len(rim), min(rim_n, len(rim)), replace=False)]

    orig_bvh = None
    if orig_ob is not None:
        orig_bvh = BVHTree.FromObject(orig_ob, dg)
    feas_cache = {}

    def off_material(du, dv):
        key = (round(du, 3), round(dv, 3))
        if key not in feas_cache:
            if orig_bvh is None:
                feas_cache[key] = 0.0
            else:
                cc = Vector(c0 + u * du + v * dv)
                loc, nor, _, dd = orig_bvh.find_nearest(cc)
                inside = loc is not None and (cc - loc).dot(nor) < 0
                feas_cache[key] = 0.0 if inside else max(0.0, dd - reach_r)
        return feas_cache[key]

    def cost(du, dv, d):
        cc = c0 + u * du + v * dv
        rows = rim_stats(bvh, rim, cc, a, sgn * d)
        return (sum((min(m, 2.0) ** 2 if m > 0 else m * m + 20.0 * f)
                    for _, m, f in rows)
                + 0.05 * (du * du + dv * dv)
                + 10.0 * off_material(du, dv) ** 2)

    span = max(12.0, 2.0 * exclude_r)
    best = None
    for step, rng_uv, rng_d, center_uvd in (
            (span / 4, span, (th0 - 6, th0 + 24), None),
            (span / 12, span / 4, None, "best"),
            (span / 36, span / 12, None, "best")):
        if center_uvd is None:
            us = vs = np.arange(-rng_uv, rng_uv + 1e-6, step)
            ds = np.arange(rng_d[0], rng_d[1] + 1e-6, 2.0)
        else:
            _, bu, bv, bd = best
            us = np.arange(bu - rng_uv, bu + rng_uv + 1e-6, step)
            vs = np.arange(bv - rng_uv, bv + rng_uv + 1e-6, step)
            dstep = 1.0 if step > span / 20 else 0.5
            ds = np.arange(bd - 2 * dstep, bd + 2 * dstep + 1e-6, dstep)
        for du in us:
            for dv in vs:
                for d in ds:
                    val = cost(du, dv, d)
                    if best is None or val < best[0]:
                        best = (val, du, dv, d)
    _, du, dv, d = best
    cc = c0 + u * du + v * dv
    rows = rim_report(bvh, rim, cc, a, sgn * d)
    return {"center": Vector(cc), "angle": sgn * d, "rows": rows,
            "max_gap": max(m for _, m in rows),
            "max_dig": -min(m for _, m in rows),
            "moved_mm": float(math.hypot(du, dv)), "first_contact": th0,
            "rim": rim}


def trim_for_closing(body, lid, center, axis, angle_deg, clearance):
    M = (Matrix.Translation(center) @ Matrix.Rotation(math.radians(angle_deg),
                                                      4, Vector(axis))
         @ Matrix.Translation(-Vector(center)))
    before = len(body.data.vertices)
    for off in ((0, 0, 0), (clearance, 0, 0), (-clearance, 0, 0),
                (0, clearance, 0), (0, -clearance, 0), (0, 0, clearance),
                (0, 0, -clearance)):
        me = lid.data.copy()
        me.transform(Matrix.Translation(off) @ M)
        boolean(body, me, 'DIFFERENCE')
    return len(body.data.vertices) - before


def keep_largest(ob):
    lab = _components(ob.data)
    cnt = np.bincount(lab)
    if len(cnt) < 2:
        return 0
    ls, _, lv = _face_arrays(ob.data)
    new = mesh_subset(ob.data, lab[lv[ls]] == np.argmax(cnt), ob.data.name)
    old = ob.data
    ob.data = new
    bpy.data.meshes.remove(old)
    new.name = ob.name
    return int(cnt.sum() - cnt.max())


def fill_seam(body, lid, center, axis, angle_deg, depth=4.0, step=0.5,
              clearance=0.35):
    M = (Matrix.Translation(center) @ Matrix.Rotation(math.radians(angle_deg),
                                                      4, Vector(axis))
         @ Matrix.Translation(-Vector(center)))
    k = max(1, int(round(depth / step)))

    def swept(src_me, base_M, sign):
        acc = bpy.data.objects.new("_vc_sweep", src_me.copy())
        acc.data.transform(base_M)
        _link(acc)
        for i in range(1, k + 1):
            me = src_me.copy()
            me.transform(Matrix.Translation((0, 0, sign * step * i)) @ base_M)
            boolean(acc, me, 'UNION')
        return acc

    up = swept(body.data, Matrix.Identity(4), +1.0)
    down = swept(lid.data, M, -1.0)
    boolean(up, down.data.copy(), 'INTERSECT')
    _remove_with_mesh(down)
    nst = max(2, int(math.ceil(abs(angle_deg) / 5.0)))
    cl = clearance
    nudges = ((0, 0, 0), (cl, 0, 0), (-cl, 0, 0), (0, cl, 0), (0, -cl, 0),
              (0, 0, cl), (0, 0, -cl))
    for i in range(nst + 1):
        poses = ([(0.0, n) for n in nudges] if i == 0 else
                 [(angle_deg * i / nst + off, (0, 0, 0))
                  for off in (-0.5, 0.0, 0.5)])
        for ang, nud in poses:
            Mi = (Matrix.Translation(nud) @ Matrix.Translation(center)
                  @ Matrix.Rotation(math.radians(ang), 4, Vector(axis))
                  @ Matrix.Translation(-Vector(center)))
            me = lid.data.copy()
            me.transform(Mi)
            if len(up.data.vertices):
                boolean(up, me, 'DIFFERENCE')
            else:
                bpy.data.meshes.remove(me)
    filler = up.data
    nv = len(filler.vertices)
    bpy.data.objects.remove(up)
    if nv:
        boolean(body, filler, 'UNION')
        keep_largest(body)
    else:
        bpy.data.meshes.remove(filler)
    return nv


def _part_dir(co, c, a, r_in, r_out, half):
    rel = co - c
    t = rel @ a
    radv = rel - np.outer(t, a)
    r = np.linalg.norm(radv, axis=1)
    k = (np.abs(t) <= half * 1.01) & (r > r_in) & (r < r_out)
    if k.sum() < 10:
        return None, int(k.sum())
    d = (radv[k] / r[k][:, None]).mean(0)
    n = np.linalg.norm(d)
    return (d / n if n > 1e-6 else None), int(k.sum())


def _wedge_meshes(c, a, t0, t1, r, leaf_dir, deg_range, half_w, seg):
    a_, u, v = _frame(a)
    d = Vector(leaf_dir)
    base = math.atan2(d.dot(v), d.dot(u))
    pad = math.asin(min(1.0, half_w / max(r, 1e-6))) + math.radians(1)
    test = _rotate(np.array([np.array(u)]), np.zeros(3), np.array(a_),
                   math.radians(10))[0]
    hand = 1.0 if Vector(test).dot(v) > 0 else -1.0
    a0 = base + hand * math.radians(deg_range[0]) - pad
    a1 = base + hand * math.radians(deg_range[1]) + pad
    if a1 < a0:
        a0, a1 = a1, a0
    out = []
    nseg = max(1, int(math.ceil((a1 - a0) / math.radians(90))))
    for i in range(nseg):
        b0 = a0 + (a1 - a0) * i / nseg
        b1 = a0 + (a1 - a0) * (i + 1) / nseg
        steps = max(2, int(seg * (b1 - b0) / (2 * math.pi)) + 2)
        rr = r / math.cos((b1 - b0) / (2 * steps))
        pts = []
        for t in (t0, t1):
            pts.append(c + a_ * t)
            for k in range(steps + 1):
                th = b0 + (b1 - b0) * k / steps
                pts.append(c + a_ * t + (u * math.cos(th) + v * math.sin(th))
                           * rr)
        out.append(_hull_mesh(pts, "_vc_wedge"))
    return out


def build_knuckle_hinge(ob, center, axis, half_len, cut_r, opts=None,
                        run_swing=True):
    o = dict(DEFAULTS)
    o.update(opts or {})
    log = []
    res = {"ok": False}
    c = Vector(center)
    a = Vector(axis).normalized()
    clr, gap = o["clearance"], o["gap"]
    rk = (o["knuckle_d"] / 2) if o["knuckle_d"] > 0 else (cut_r - clr)
    cut_r = max(cut_r, rk + clr)
    rp = o["pin_d"] / 2
    n = int(o["count"]) | 1
    seg = int(o["segments"])

    if rp + clr >= rk - 0.8:
        log.append("ERROR: pin %.2f mm is too fat for a %.2f mm knuckle - "
                   "lid knuckle wall would be under 0.8 mm. Use a thinner pin "
                   "or a bigger guide." % (2 * rp, 2 * rk))
        return log, res
    dims = ob.dimensions
    if max(dims) > 400:
        log.append("WARNING: model is %.0f mm - bigger than any Bambu bed. "
                   "Hinge sizes are real mm: scale to print size FIRST "
                   "(Tools > Scale to Size)." % max(dims))

    base_me = ob.data.copy()
    base_me.transform(ob.matrix_world)
    grow = o.get("grow", GROW_STEPS)
    work = None
    for gl, gr in grow:
        h_try, r_try = half_len * gl, cut_r * gr
        if work is not None:
            _remove_with_mesh(work)
        work = bpy.data.objects.new(ob.name + "_hingework", base_me.copy())
        _link(work)
        boolean(work, _hull_mesh(cylinder_points(c, a, -h_try, h_try, r_try,
                                                 seg), "_vc_cut"),
                'DIFFERENCE')
        lab = _components(work.data)
        cnt = np.bincount(lab)
        order = np.argsort(-cnt)
        total = cnt.sum()
        big = [k for k in order if cnt[k] >= 0.02 * total]
        if len(big) >= 2:
            break
    if len(big) < 2:
        bpy.data.meshes.remove(base_me)
        _remove_with_mesh(work)
        log.append("ERROR: lid is still welded to the body even after "
                   "growing the cut %.0f%% longer / %.0f%% wider. Move the "
                   "guide so it covers the whole fake hinge, then rebuild."
                   % (100 * (grow[-1][0] - 1), 100 * (grow[-1][1] - 1)))
        return log, res
    if (h_try, r_try) != (half_len, cut_r):
        log.append("lid was also welded past the guide: cut grown to "
                   "%.1f mm long, %.2f mm wide to free it"
                   % (2 * h_try, 2 * r_try))
    half_len, cut_r = h_try, r_try
    L = 2 * half_len
    log.append("cut fake hinge: %.1f mm long, %.2f mm diameter"
               % (L, 2 * cut_r))
    hs = half_len - gap
    seg_len = (2 * hs - (n - 1) * gap) / n
    if seg_len < 2.0:
        bpy.data.meshes.remove(base_me)
        _remove_with_mesh(work)
        log.append("ERROR: knuckles would be %.2f mm wide. Fewer knuckles "
                   "or a longer guide." % seg_len)
        return log, res
    dropped = int(total - cnt[big[0]] - cnt[big[1]])
    ls, lt, lv = _face_arrays(work.data)
    flab = lab[lv[ls]]
    pieces = []
    for k in big[:2]:
        me = mesh_subset(work.data, flab == k, "piece")
        pob = bpy.data.objects.new("piece", me)
        _link(pob)
        pieces.append(pob)
    _remove_with_mesh(work)
    if dropped:
        log.append("removed %d stray vertices (floating crumbs)" % dropped)

    zmin = [_co(p.data)[:, 2].min() for p in pieces]
    body, lid = (pieces if zmin[0] <= zmin[1] else pieces[::-1])
    if o["swap"]:
        body, lid = lid, body
    for p, nm in ((body, "_Body"), (lid, "_Lid")):
        old = bpy.data.objects.get(ob.name + nm)
        if old is not None and old is not p:
            om = old.data
            bpy.data.objects.remove(old)
            if om.users == 0:
                bpy.data.meshes.remove(om)
        p.name = ob.name + nm
        p.data.name = ob.name + nm

    c_sep = c.copy()
    close_ang = None
    if o.get("auto_fit", True):
        orig = bpy.data.objects.new("_vc_orig", base_me)
        _link(orig)
        fit = fit_closed_pose(body, lid, c, a, exclude_r=cut_r + 2 * rk,
                              orig_ob=orig, reach_r=rk)
        bpy.data.objects.remove(orig)
        if fit is None:
            log.append("WARNING: couldn't find where the lid closes (it never "
                       "reaches the body) - closing NOT verified.")
        else:
            close_ang = fit["angle"]
            res["fit"] = {k: v for k, v in fit.items() if k != "rim"}
            if fit["moved_mm"] > 0.25:
                c = Vector(fit["center"])
                log.append("lid fit: hinge axis moved %.1f mm from the fake "
                           "hinge so the lid seats" % fit["moved_mm"])
            log.append("lid fit: closes at %.1f deg, rim gap %.2f mm / "
                       "overlap %.2f mm before trimming"
                       % (abs(close_ang), max(fit["max_gap"], 0),
                          max(fit["max_dig"], 0)))
    moved = (c - c_sep).length
    if moved > 1e-3:
        for p in (body, lid):
            boolean(p, _hull_mesh(cylinder_points(c, a, -half_len, half_len,
                                                  rk + clr, seg), "_vc_zone"),
                    'DIFFERENCE')
    inner = cut_r if moved <= 1e-3 else rk + clr
    reach = cut_r + moved
    if close_ang is not None and o.get("fill_seam", True):
        if fill_seam(body, lid, c, a, close_ang,
                     depth=o.get("seam_depth", 4.0), clearance=clr):
            log.append("seam filler: raised the body rim up to %.1f mm to "
                       "meet the shut lid" % o.get("seam_depth", 4.0))
    if close_ang is not None and (o.get("trim_closed", True)
                                  or o.get("fill_seam", True)):
        trim_for_closing(body, lid, c, a, close_ang, clr)
        log.append("trimmed the body where the closed lid sits (+%.2f mm "
                   "clearance)" % clr)
        lost = keep_largest(body)
        if lost:
            log.append("dropped %d verts of seam filler cut loose by the "
                       "trim" % lost)

    cn, an = np.array(c), np.array(a)
    dirs = {}
    for p, key in ((body, "body"), (lid, "lid")):
        d, cntp = _part_dir(_co(p.data), cn, an, inner * 0.98,
                            reach + 3 * rk, half_len)
        if d is None:
            bpy.data.meshes.remove(base_me)
            log.append("ERROR: no %s material next to the guide - is the "
                       "guide on the hinge?" % key)
            return log, res
        dirs[key] = Vector(d)
    ang = math.degrees(dirs["body"].angle(dirs["lid"]))
    log.append("body leaf / lid leaf are %.0f deg apart" % ang)
    if ang < 25:
        log.append("WARNING: body and lid sit almost the same side of the "
                   "axis - leaves may touch the wrong part.")

    bridge = moved > 1.5
    spans = []
    for i in range(n):
        t0 = -hs + i * (seg_len + gap)
        spans.append(("body" if i % 2 == 0 else "lid", t0, t0 + seg_len))
    part_of = {"body": body, "lid": lid}
    part_co = {k: _co(v.data) for k, v in part_of.items()}
    leaf_len = reach + o["leaf_reach"]

    strut_len = {}
    if bridge:
        dg_s = bpy.context.evaluated_depsgraph_get()
        for key, part in part_of.items():
            bvh_p = BVHTree.FromObject(part, dg_s)
            d = dirs[key]
            need = 0.0
            for k, t0, t1 in spans:
                if k != key:
                    continue
                for t in (t0 + 0.3, (t0 + t1) / 2, t1 - 0.3):
                    hit, _, _, dist = bvh_p.ray_cast(c + a * t, d, 200.0)
                    if hit is None:
                        dist = bvh_p.find_nearest(c + a * t)[3]
                    need = max(need, dist)
            strut_len[key] = need + o["leaf_reach"]

    def leaf_points(key, t0, t1):
        pts = [Vector(q) for q in cylinder_points(c, a, t0, t1, rk, seg)]
        d = dirs[key]
        w = a.cross(d).normalized()
        ll = strut_len[key] if bridge else leaf_len
        pts += [c + a * t + w * (sw * rk) + d * rr
                for t in (t0, t1) for sw in (-1.0, 1.0)
                for rr in (0.0, ll)]
        return pts

    def leaves(key, trim):
        hulls = []
        for k, t0, t1 in spans:
            if k != key:
                continue
            pts = leaf_points(key, t0, t1)
            if pts is None:
                continue
            hulls.append(_hull_mesh(pts, "_vc_leaf"))
        m = _merge_meshes(hulls, "_vc_leaves")
        if not trim:
            return m
        tmp = bpy.data.objects.new("_vc_leaftrim", m)
        _link(tmp)
        boolean(tmp, base_me.copy(), 'INTERSECT')
        out = tmp.data
        bpy.data.objects.remove(tmp)
        return out

    if o.get("relief", True):
        if close_ang is not None:
            far_open = -math.copysign(30.0, close_ang)
            lo = min(close_ang, far_open) - 3.0
            hi = max(close_ang, far_open) + 3.0
            nst = max(2, int(math.ceil((hi - lo) / 6.0)))
            angs = [lo + (hi - lo) * i / nst for i in range(nst + 1)]
        for target, key, sgn_rel in ((body, "lid", 1.0), (lid, "body", -1.0)):
            for k, t0, t1 in spans:
                if k != key:
                    continue
                if close_ang is None:
                    boolean(target, _hull_mesh(cylinder_points(
                        c, a, t0 - gap, t1 + gap, leaf_len + clr, seg),
                        "_vc_rel"), 'DIFFERENCE')
                    continue
                pts = leaf_points(key, t0 - gap, t1 + gap)
                if pts is None:
                    continue
                P = np.array([tuple(q) for q in pts])
                rel = P - cn
                tt = rel @ an
                radv = rel - np.outer(tt, an)
                rr = np.linalg.norm(radv, axis=1, keepdims=True)
                P = P + radv / np.maximum(rr, 1e-9) * clr
                sweep = None
                for a0, a1 in zip(angs[:-1], angs[1:]):
                    Q = np.r_[_rotate(P, cn, an, math.radians(sgn_rel * a0)),
                              _rotate(P, cn, an, math.radians(sgn_rel * a1))]
                    h_me = _hull_mesh([Vector(q) for q in Q], "_vc_sw")
                    if sweep is None:
                        sweep = bpy.data.objects.new("_vc_sweep", h_me)
                        _link(sweep)
                    else:
                        boolean(sweep, h_me, 'UNION')
                if sweep is not None:
                    sm = sweep.data
                    bpy.data.objects.remove(sweep)
                    boolean(target, sm, 'DIFFERENCE')
        log.append("mortise relief: %s"
                   % ("exact swept path of every knuckle"
                      if close_ang is not None else "full rings"))

    trim = (not bridge) and o.get("leaf_style", 'TRIM') == 'TRIM'
    for part, key in ((body, "body"), (lid, "lid")):
        if bridge:
            boolean(part, leaves(key, False), 'UNION')
            continue
        boolean(part, _merge_meshes(
            [_hull_mesh(cylinder_points(c, a, t0, t1, rk, seg), "_vc_barrel")
             for k, t0, t1 in spans if k == key], "_vc_b"), 'UNION')
        lm = leaves(key, trim)
        if len(lm.vertices):
            boolean(part, lm, 'UNION')
        else:
            bpy.data.meshes.remove(lm)
        if trim and int(_components(part.data).max()) + 1 > 1:
            boolean(part, leaves(key, False), 'UNION')
            log.append("%s: trimmed leaf didn't reach - used full leaves"
                       % key)
    if bridge:
        log.append("hinge sits off the shell: knuckles bridged to each part")
    bpy.data.meshes.remove(base_me)

    pin = None
    bore = rp + clr
    if o["pin_mode"] == 'PIP':
        boolean(body, _hull_mesh(cylinder_points(c, a, -hs, hs, rp, seg),
                                 "_vc_pin"), 'UNION')
        log.append("pin %.2f mm fused to the body knuckles (print-in-place)"
                   % (2 * rp))
    else:
        drill = half_len + 3 * cut_r
        boolean(body, _hull_mesh(cylinder_points(
            c, a, -drill, drill, rp + o["pin_fit"], seg),
            "_vc_bb"), 'DIFFERENCE')
        pin_me = _hull_mesh(cylinder_points(c, a, -hs, hs, rp, seg),
                            ob.name + "_Pin")
        old = bpy.data.objects.get(ob.name + "_Pin")
        if old is not None:
            bpy.data.objects.remove(old)
        pin = bpy.data.objects.new(ob.name + "_Pin", pin_me)
        _link(pin)
        log.append("separate pin %.2f x %.1f mm (body holes +%.2f mm snug)"
                   % (2 * rp, 2 * hs, o["pin_fit"]))
    boolean(lid, _hull_mesh(cylinder_points(c, a, -hs, hs, bore, seg),
                            "_vc_bore"), 'DIFFERENCE')
    log.append("%d knuckles %.2f mm wide, OD %.2f mm, %.2f mm radial / "
               "%.2f mm axial clearance" % (n, seg_len, 2 * rk, clr, gap))

    ok = True
    crumbs = sum(drop_crumbs(p) for p in (body, lid))
    if crumbs:
        log.append("swept %d boolean crumb vertices" % crumbs)
    for p in (body, lid):
        nm = non_manifold_edges(p.data)
        if nm:
            left = micro_repair(p)
            log.append("repaired %d boolean-seam edge(s) on %s%s"
                       % (nm - left, p.name,
                          "" if not left else " (%d left)" % left))
            nm = left
        shells = int(_components(p.data).max()) + 1
        res[p.name] = {"bad_edges": nm, "pieces": shells,
                       "verts": len(p.data.vertices)}
        if nm:
            ok = False
            log.append("WARNING: %s has %d non-watertight edges" % (p.name, nm))
        if shells != 1:
            ok = False
            log.append("WARNING: %s is %d separate pieces - a knuckle is not "
                       "fused in. Raise Leaf Reach." % (p.name, shells))
    dg = bpy.context.evaluated_depsgraph_get()
    touching = len(BVHTree.FromObject(body, dg).overlap(
        BVHTree.FromObject(lid, dg)))
    res["touching_tris"] = touching
    if touching:
        ok = False
        log.append("WARNING: body and lid surfaces touch in %d places - they "
                   "would print fused." % touching)
    if pin is not None:
        open_ends = 0
        for sgn in (1, -1):
            o_pt = c + a * (sgn * (hs + 0.5))
            hit_l = BVHTree.FromObject(lid, dg).ray_cast(o_pt, a * sgn, 1e4)
            hit_b = BVHTree.FromObject(body, dg).ray_cast(o_pt, a * sgn, 1e4)
            if hit_l[0] is None and hit_b[0] is None:
                open_ends += 1
        res["pin_insertable"] = open_ends > 0
        if not open_ends:
            log.append("WARNING: the pin hole is blocked by the model at both "
                       "ends - the pin can't be pushed in. Use Print-in-Place.")
    if run_swing:
        sw = swing_test(body, lid, c, a, cut_r)
        res["swing"] = sw
        log.append("swing test: lid closes %.0f deg / opens %.0f deg more "
                   "before touching the body%s" % (
                       sw["close_deg"], sw["open_deg"],
                       " (COLLIDES as built!)" if sw["at_0_collides"] else ""))
        if sw["at_0_collides"] or sw["close_deg"] < 3:
            ok = False
            log.append("WARNING: the lid can't swing - check the guide axis.")
        if close_ang is not None:
            need = abs(close_ang)
            fine = swing_test(body, lid, c, a, cut_r, step_deg=1.0,
                              max_deg=need + 4)
            res["closes_to"] = fine["close_deg"]
            dg2 = bpy.context.evaluated_depsgraph_get()
            rows = rim_report(BVHTree.FromObject(body, dg2), fit["rim"],
                              np.array(c), np.array(a), close_ang)
            gaps = sorted(max(m, 0.0) for _, m in rows)
            worst_gap = gaps[min(len(gaps) - 1, int(0.8 * len(gaps)))]
            res["seat_gap"] = worst_gap
            res["seat_rows"] = rows
            if fine["close_deg"] < need - 1.0:
                ok = False
                log.append("WARNING: the lid STOPS %.0f deg short of closed "
                           "(swings %.0f of %.0f deg)."
                           % (need - fine["close_deg"], fine["close_deg"],
                              need))
            else:
                log.append("closing check: lid swings fully shut (%.0f deg); "
                           "widest rim gap when shut %.2f mm"
                           % (need, max(worst_gap, 0)))
            if worst_gap > 2.0:
                log.append("WARNING: when shut the rim still gaps %.1f mm "
                           "somewhere - the AI lid shape doesn't match the "
                           "body there." % worst_gap)
        else:
            log.append("WARNING: closing NOT verified (no closed pose found).")
            ok = False
    ob.hide_set(True)
    ob.hide_render = True
    res.update(ok=ok, body=body, lid=lid, pin=pin, spans=spans,
               knuckle_r=rk, cut_r=cut_r, dirs=dirs)
    log.append("RESULT: %s" % ("hinge OK - print it!" if ok else
                               "built with warnings (see above)"))
    return log, res


def export_stls(objs, folder, stem):
    import os
    os.makedirs(folder, exist_ok=True)
    paths = []

    def _export(sel, path):
        for x in bpy.context.view_layer.objects:
            x.select_set(False)
        for x in sel:
            x.hide_set(False)
            x.select_set(True)
        bpy.context.view_layer.objects.active = sel[0]
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True,
                              apply_modifiers=True, ascii_format=False)
        paths.append(path)

    objs = [x for x in objs if x is not None]
    _export(objs, os.path.join(folder, stem + "_ALL_print_in_place.stl"))
    for x in objs:
        _export([x], os.path.join(folder, x.name + ".stl"))
    return paths


def _poll_source(self, ob):
    return ob.type == 'MESH' and ob.name != GUIDE_NAME


class VibeCutHingeProps(bpy.types.PropertyGroup):
    source: bpy.props.PointerProperty(
        name="Model", type=bpy.types.Object, poll=_poll_source,
        description="The fused model that has the fake hinge")
    knuckle_d: bpy.props.FloatProperty(
        name="Knuckle Diameter (mm)", default=DEFAULTS["knuckle_d"],
        min=0.0, max=60.0,
        description="0 = auto: fill the guide (guide diameter minus clearance)")
    pin_d: bpy.props.FloatProperty(
        name="Pin Diameter (mm)", default=DEFAULTS["pin_d"], min=1.0,
        max=30.0, description="1.75 lets you use a piece of filament as the "
        "pin in Separate mode")
    clearance: bpy.props.FloatProperty(
        name="Clearance (mm)", default=DEFAULTS["clearance"], min=0.1,
        max=1.0, description="Radial gap pin-to-lid-knuckle. 0.3-0.4 for a "
        "0.4 nozzle; go up if it prints fused, down if it wobbles")
    gap: bpy.props.FloatProperty(
        name="Knuckle Gap (mm)", default=DEFAULTS["gap"], min=0.2, max=1.5,
        description="Gap between neighbouring knuckles along the pin")
    count: bpy.props.IntProperty(
        name="Knuckles", default=DEFAULTS["count"], min=3, max=15, step=2,
        description="Odd number: body knuckles at both ends hold the pin")
    leaf_reach: bpy.props.FloatProperty(
        name="Leaf Reach (mm)", default=DEFAULTS["leaf_reach"], min=0.5,
        max=30.0, description="How far each knuckle's leaf digs into its "
        "part. Raise it if a knuckle comes out floating")
    pin_mode: bpy.props.EnumProperty(
        name="Pin", default='PIP', items=(
            ('PIP', "Print-in-Place", "Pin fused to the body; prints as one "
             "working piece"),
            ('SEPARATE', "Separate Pin", "Holes through every knuckle plus a "
             "pin object - print lid and body apart, push the pin in")))
    auto_fit: bpy.props.BoolProperty(
        name="Auto-Fit Lid", default=True,
        description="Find where the lid REALLY closes and move the hinge "
        "axis there (AI fake hinges are often in the wrong place). Adds "
        "1-2 minutes")
    trim_closed: bpy.props.BoolProperty(
        name="Trim for Closing", default=True,
        description="Cut the body wherever the closed lid would sit inside "
        "it (e.g. a collar the lid lands on), plus clearance")
    swap: bpy.props.BoolProperty(
        name="Swap Lid / Body", default=False,
        description="Tick if it guessed the wrong piece as the lid")
    status: bpy.props.StringProperty(default="")


def _p(ctx):
    return ctx.scene.vibecut_hinge


def _status(ctx, lines):
    _p(ctx).status = " | ".join(lines)


def _source(ctx):
    p = _p(ctx)
    if p.source is not None:
        return p.source
    ob = ctx.active_object
    if ob is not None and ob.type == 'MESH' and ob.name != GUIDE_NAME:
        return ob
    return None


class VIBECUT_OT_hinge_place(bpy.types.Operator):
    """Put a wire guide cylinder on the fake hinge at the 3D cursor
    (Shift+Right-click the hinge first). Then move / rotate / scale the
    guide (G, R, S) until it swallows the whole fake hinge"""
    bl_idname = "vibecut.hinge_place"
    bl_label = "1. Place Guide at 3D Cursor"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return ctx.mode == 'OBJECT' and _source(ctx) is not None

    def execute(self, ctx):
        ob = _source(ctx)
        _p(ctx).source = ob
        cur = ctx.scene.cursor.location.copy()
        axis, h, r = guess_guide(ob, cur)
        fit = fit_fake_hinge(ob, cur, axis, h, r)
        if fit is not None and (fit[0] - cur).length < 2 * r:
            cur, r = fit[0], fit[1] * 1.05
        g = make_guide(cur, axis, h, r)
        for x in ctx.view_layer.objects:
            x.select_set(False)
        g.select_set(True)
        ctx.view_layer.objects.active = g
        _status(ctx, ["Guide placed (%.1f mm long, %.1f mm wide)." % (2 * h, 2 * r),
                      "Scale it along its length (S Z) to cover the fake hinge,",
                      "then press 2. Snap to Fake Hinge or 3. Build."])
        return {'FINISHED'}


class VIBECUT_OT_hinge_snap(bpy.types.Operator):
    """Re-centre the guide exactly on the fake hinge barrel (circle fit)"""
    bl_idname = "vibecut.hinge_snap"
    bl_label = "2. Snap to Fake Hinge"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'OBJECT' and GUIDE_NAME in bpy.data.objects
                and _source(ctx) is not None)

    def execute(self, ctx):
        g = bpy.data.objects[GUIDE_NAME]
        c, a, h, r = guide_from_object(g)
        fit = fit_fake_hinge(_source(ctx), c, a, h, r)
        if fit is None:
            _status(ctx, ["Couldn't find a round barrel inside the guide - "
                          "move it onto the hinge and retry."])
            return {'CANCELLED'}
        make_guide(fit[0], a, h, fit[1] * 1.05)
        _status(ctx, ["Snapped: barrel %.2f mm diameter (fit error %.2f mm)."
                      % (2 * fit[1], fit[2])])
        return {'FINISHED'}


class VIBECUT_OT_hinge_build(bpy.types.Operator):
    """Cut the fake hinge out, split lid from body, add a real knuckle hinge
    and test that the lid swings. Original model is hidden, not deleted"""
    bl_idname = "vibecut.hinge_build"
    bl_label = "3. Build Knuckle Hinge"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        return (ctx.mode == 'OBJECT' and GUIDE_NAME in bpy.data.objects
                and _source(ctx) is not None)

    def execute(self, ctx):
        p = _p(ctx)
        ob = _source(ctx)
        ob.hide_set(False)
        c, a, h, r = guide_from_object(bpy.data.objects[GUIDE_NAME])
        opts = {k: getattr(p, k) for k in
                ("knuckle_d", "pin_d", "clearance", "gap", "count",
                 "leaf_reach", "pin_mode", "swap", "auto_fit",
                 "trim_closed")}
        log, res = build_knuckle_hinge(ob, c, a, h, r, opts)
        print("[VibeCut Hinge] %s\n  " % ob.name + "\n  ".join(log))
        _status(ctx, log)
        bpy.data.objects[GUIDE_NAME].hide_set(True)
        if not res.get("body"):
            self.report({'ERROR'}, log[-1])
            return {'CANCELLED'}
        self.report({'INFO'} if res["ok"] else {'WARNING'}, log[-1])
        return {'FINISHED'}


class VIBECUT_OT_hinge_export(bpy.types.Operator):
    """Export Body + Lid (+ Pin) STLs next to the .blend: one all-in-one
    print-in-place file and one file per part"""
    bl_idname = "vibecut.hinge_export"
    bl_label = "4. Export STLs"

    @classmethod
    def poll(cls, ctx):
        ob = _source(ctx)
        return ob is not None and (ob.name + "_Body") in bpy.data.objects

    def execute(self, ctx):
        import os
        ob = _source(ctx)
        objs = [bpy.data.objects.get(ob.name + s)
                for s in ("_Body", "_Lid", "_Pin")]
        folder = (os.path.dirname(bpy.data.filepath) if bpy.data.filepath
                  else os.path.expanduser("~/Desktop"))
        paths = export_stls(objs, os.path.join(folder, ob.name + "_hinge_stl"),
                            ob.name)
        _status(ctx, ["Exported %d STLs to" % len(paths),
                      os.path.dirname(paths[0])])
        return {'FINISHED'}


class VIBECUT_PT_hinge(bpy.types.Panel):
    bl_label = "Knuckle Hinge"
    bl_idname = "VIBECUT_PT_hinge"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VibeCut"
    bl_parent_id = "VIBECUT_PT_more"
    bl_order = 1
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L = self.layout
        p = _p(ctx)
        L.label(text="Fake hinge -> real knuckle hinge", icon='LINKED')
        L.label(text="Scale model to print size FIRST.", icon='INFO')
        L.prop(p, "source")
        c = L.column(align=True)
        c.label(text="Shift+Right-click the fake hinge, then:")
        c.operator("vibecut.hinge_place", icon='MESH_CYLINDER')
        c.operator("vibecut.hinge_snap", icon='SNAP_ON')
        b = L.box()
        col = b.column(align=True)
        col.prop(p, "pin_mode", text="")
        col.prop(p, "count")
        col.prop(p, "knuckle_d")
        col.prop(p, "pin_d")
        col.prop(p, "clearance")
        col.prop(p, "gap")
        col.prop(p, "leaf_reach")
        col.prop(p, "auto_fit")
        col.prop(p, "trim_closed")
        col.prop(p, "swap")
        big = L.column()
        big.scale_y = 1.5
        big.operator("vibecut.hinge_build", icon='MOD_BOOLEAN')
        L.operator("vibecut.hinge_export", icon='EXPORT')
        if p.status:
            b = L.box()
            for chunk in p.status.split(" | "):
                b.label(text=chunk)


_classes = (VIBECUT_OT_hinge_place, VIBECUT_OT_hinge_snap,
            VIBECUT_OT_hinge_build, VIBECUT_OT_hinge_export, VIBECUT_PT_hinge)


def register_props():
    bpy.utils.register_class(VibeCutHingeProps)
    bpy.types.Scene.vibecut_hinge = bpy.props.PointerProperty(
        type=VibeCutHingeProps)


def unregister_props():
    del bpy.types.Scene.vibecut_hinge
    bpy.utils.unregister_class(VibeCutHingeProps)


def register():
    for c in _classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(_classes):
        bpy.utils.unregister_class(c)
