import bpy
import bmesh
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from math import radians, sin, cos, tan, sqrt
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


@dataclass
class Params:
    nozzle: float = 0.4
    layer_height: float = 0.2
    line_width: float = 0.42

    overhang_angle: float = 45.0
    first_layer_z: float = 0.30

    z_gap: float = 0.25
    xy_clearance: float = 1.0

    min_feature: float = 1.0

    tip_contact_d: float = 1.2
    tip_taper_len: float = 2.5

    branch_d: float = 2.0
    branch_max_d: float = 4.4
    branch_angle: float = 40.0
    merge_radius: float = 9.0
    step_dz: float = 1.0

    pillar_d: float = 2.4
    pillar_spacing: float = 3.6
    rib_d: float = 1.2
    slender_ratio: float = 12.0

    cradle_gap: float = 0.35
    cradle_margin: float = 3.0
    cradle_enable: bool = True

    cavity_tip_d: float = 1.0
    cavity_branch_d: float = 1.6

    bridge_span: float = 10.0

    base_spacing: float = 2.8
    max_points: int = 1400

    voxel_size: float = 0.30
    seed: int = 42

    max_retries: int = 2

    printer: str = "X1C"
    material: str = "PLA"

    strategy_override: str = "AUTO"

    style: str = "CLEAR"

    pin_spacing: float = 3.5
    minima_radius: float = 2.0
    pin_diameter: float = 1.0
    pin_overlap: float = 0.5
    pin_base_length: float = 3.0
    pin_branch_d: float = 1.8
    pillar_diameter: float = 2.6
    max_trunk_reach: float = 3.0
    fork_angle: float = 35.0
    clearance: float = 0.4
    rest_ratio: float = 0.4
    max_fork_load: int = 2
    keepout: float = 1.0
    merge_reach: float = 4.0
    route_cell: float = 0.6
    route_max_cells: int = 40000000
    foot_h: float = 1.2
    pad_d: float = 6.0
    pad_h: float = 0.8
    raft_join: float = 14.0
    brace_min_h: float = 6.0
    brace_step: float = 8.0
    brace_max_bays: int = 4
    brace_max_span: float = 10.0
    brace_d: float = 1.0
    conform: str = "MANIFOLD"


PRINTERS = {
    "X1C":    (256.0, 256.0, 256.0, "@BBL X1C"),
    "P1S":    (256.0, 256.0, 256.0, "@BBL P1P"),
    "A1":     (256.0, 256.0, 256.0, "@BBL A1"),
    "A1MINI": (180.0, 180.0, 180.0, "@BBL A1M"),
    "H2D":    (350.0, 320.0, 325.0, "@BBL H2D"),
}

MATERIALS = {
    "PLA":  (220, 55, 1.00, 100),
    "PETG": (255, 70, 0.55, 60),
    "ABS":  (270, 90, 0.70, 30),
}

STRAT_BRIDGE = "BRIDGE"
STRAT_SEALED = "SEALED"
STRAT_CAVITY = "CAVITY"
STRAT_CRADLE = "CRADLE"
STRAT_PILLAR = "PILLAR"
STRAT_TREE = "TREE"

SUPPORT_COLL_NAME = "VibeCut Supports"


def _get_or_create_support_collection(scene):
    coll = bpy.data.collections.get(SUPPORT_COLL_NAME)
    if coll is None:
        coll = bpy.data.collections.new(SUPPORT_COLL_NAME)
        scene.collection.children.link(coll)
    return coll


def _ensure_lookup(bm):
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()


class PrimitiveSoup:
    CHUNK = 250

    def __init__(self):
        self.bm = bmesh.new()
        self.count = 0
        self.feet = []
        self._chunks = []
        self._in_chunk = 0

    def _room(self):
        if self._in_chunk >= self.CHUNK:
            me = bpy.data.meshes.new("_soup_chunk")
            self.bm.to_mesh(me)
            self.bm.free()
            self._chunks.append(me)
            self.bm = bmesh.new()
            self._in_chunk = 0
        self._in_chunk += 1
        self.count += 1

    def tube(self, p1, p2, r1, r2, segments=10):
        d = Vector(p2) - Vector(p1)
        length = d.length
        if length < 1e-5:
            return
        r1 = max(r1, 0.05)
        r2 = max(r2, 0.05)
        quat = d.to_track_quat('Z', 'Y')
        mid = (Vector(p1) + Vector(p2)) * 0.5
        mat = Matrix.Translation(mid) @ quat.to_matrix().to_4x4()
        self._room()
        bmesh.ops.create_cone(
            self.bm, cap_ends=True, cap_tris=True, segments=segments,
            radius1=r1, radius2=r2, depth=length, matrix=mat)

    def sphere(self, p, r, u=8, v=6):
        if r < 0.05:
            return
        mat = Matrix.Translation(Vector(p))
        self._room()
        bmesh.ops.create_uvsphere(
            self.bm, u_segments=u, v_segments=v, radius=r, matrix=mat)

    def box(self, matrix):
        self._room()
        bmesh.ops.create_cube(self.bm, size=1.0, matrix=matrix)

    def free(self):
        self.bm.free()
        for me in self._chunks:
            bpy.data.meshes.remove(me)
        self._chunks = []

    def to_mesh(self, name):
        me = bpy.data.meshes.new(name)
        if not self._chunks:
            self.bm.to_mesh(me)
            self.bm.free()
            return me
        for ch in self._chunks:
            self.bm.from_mesh(ch)
            bpy.data.meshes.remove(ch)
        self._chunks = []
        self.bm.to_mesh(me)
        self.bm.free()
        return me


def _new_object(name, mesh, collection):
    ob = bpy.data.objects.new(name, mesh)
    collection.objects.link(ob)
    return ob


def _evaluated_mesh_of(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = ob.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(
        ob_eval, preserve_all_data_layers=False, depsgraph=dg)
    return me


def _apply_modifiers_inplace(ob):
    me = _evaluated_mesh_of(ob)
    old = ob.data
    ob.modifiers.clear()
    ob.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)
    return ob


def _remove_object(ob):
    me = ob.data
    bpy.data.objects.remove(ob)
    if me is not None and me.users == 0:
        bpy.data.meshes.remove(me)


def extract_world_mesh(obj):
    me = _evaluated_mesh_of(obj)
    me.transform(obj.matrix_world)
    if obj.matrix_world.determinant() < 0.0:
        me.flip_normals()
    return me


def _fix_inside_out(obj, messages):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    vol = bm.calc_volume(signed=True)
    if obj.matrix_world.determinant() < 0.0:
        vol = -vol
    if vol >= 0.0:
        bm.free()
        return False
    if obj.modifiers:
        bm.free()
        messages.append(
            "Model is inside-out (faces point inward) and has modifiers, "
            "so it was not fixed. Apply the modifiers or run Mesh > "
            "Normals > Recalculate Outside, then generate again.")
        return False
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    messages.append("Model was inside-out (faces pointed inward); "
                    "recalculated its normals to point outside.")
    return True


def build_bvh(me):
    verts = [v.co.copy() for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    if not polys:
        return None
    return BVHTree.FromPolygons(verts, polys)


def convex_hull_2d(points):
    pts = sorted(set((round(p[0], 4), round(p[1], 4)) for p in points))
    if len(pts) <= 2:
        return list(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_hull(pt, hull, tol=0.0):
    n = len(hull)
    if n < 3:
        return False
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        ex, ey = bx - ax, by - ay
        elen = math.hypot(ex, ey)
        if elen < 1e-9:
            continue
        d = ((pt[0] - ax) * ey - (pt[1] - ay) * ex) / elen
        if d > tol:
            return False
    return True


def dist_point_seg_2d(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _net_crossings(bvh, p, d, max_hops=32):
    origin = Vector(p)
    net = 0
    for _ in range(max_hops):
        hit = bvh.ray_cast(origin, d)
        if hit[0] is None:
            break
        if hit[1] is not None:
            net += 1 if hit[1].dot(d) >= 0.0 else -1
        origin = hit[0] + d * 1e-3
    return net


def point_inside_solid(bvh, p, max_hops=32):
    if _net_crossings(bvh, p, Vector((0, 0, 1)), max_hops) > 0:
        return True
    return _net_crossings(bvh, p, Vector((0, 0, -1)), max_hops) > 0


@dataclass
class Island:
    idx: int
    faces: list
    area: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    flatness: float = 0.0
    mean_nz: float = 0.0
    centroid: Vector = field(default_factory=lambda: Vector())
    anchored_edges: list = field(default_factory=list)
    free_edges: list = field(default_factory=list)
    boundary_len: float = 0.0
    anchored_len: float = 0.0
    samples: list = field(default_factory=list)
    plate_frac: float = 0.0
    enclosed_frac: float = 0.0
    sealed: bool = False
    strategy: str = STRAT_TREE
    spacing: float = 2.8
    coverage_radius: float = 2.2
    note: str = ""


@dataclass
class Analysis:
    me: object = None
    bvh: object = None
    islands: list = field(default_factory=list)
    model_h: float = 0.0
    model_dims: tuple = (0, 0, 0)
    model_min: tuple = (0, 0, 0)
    model_max: tuple = (0, 0, 0)
    total_overhang_area: float = 0.0
    bridges: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    dropped_by: float = 0.0
    n_faces: int = 0


def drop_to_plate(obj, report):
    me = extract_world_mesh(obj)
    if not me.vertices:
        bpy.data.meshes.remove(me)
        raise RuntimeError("Object has no geometry")
    zmin = min(v.co.z for v in me.vertices)
    bpy.data.meshes.remove(me)
    if abs(zmin) > 0.02:
        obj.location.z -= zmin
        bpy.context.view_layer.update()
        report.append("Moved object %.2f mm so it sits on the build plate."
                      % (-zmin))
        return -zmin
    return 0.0


def analyze(obj, P: Params):
    A = Analysis()
    me = extract_world_mesh(obj)
    A.me = me
    A.n_faces = len(me.polygons)
    if A.n_faces == 0:
        raise RuntimeError("Mesh has no faces")
    A.bvh = build_bvh(me)

    xs = [v.co.x for v in me.vertices]
    ys = [v.co.y for v in me.vertices]
    zs = [v.co.z for v in me.vertices]
    A.model_min = (min(xs), min(ys), min(zs))
    A.model_max = (max(xs), max(ys), max(zs))
    A.model_dims = (A.model_max[0] - A.model_min[0],
                    A.model_max[1] - A.model_min[1],
                    A.model_max[2] - A.model_min[2])
    A.model_h = A.model_dims[2]

    if max(A.model_dims) < 2.0:
        A.warnings.append(
            "Model is under 2 mm across - if it was modeled in meters, "
            "scale it x1000 (this tool assumes 1 unit = 1 mm).")
    if A.n_faces > 300000:
        A.warnings.append(
            "Mesh is very dense (%d faces); analysis may be slow. "
            "A Decimate modifier can speed this up." % A.n_faces)

    sin_th = sin(radians(P.overhang_angle))
    polys = me.polygons
    verts = me.vertices

    support_face = [False] * A.n_faces
    for p in polys:
        n = p.normal
        if n.z < -sin_th:
            zs_f = [verts[i].co.z for i in p.vertices]
            if min(zs_f) > P.first_layer_z:
                support_face[p.index] = True

    edge_faces = {}
    for p in polys:
        for ek in p.edge_keys:
            edge_faces.setdefault(ek, []).append(p.index)

    visited = [False] * A.n_faces
    islands = []
    for f0 in range(A.n_faces):
        if not support_face[f0] or visited[f0]:
            continue
        stack = [f0]
        visited[f0] = True
        members = []
        while stack:
            f = stack.pop()
            members.append(f)
            for ek in polys[f].edge_keys:
                for g in edge_faces[ek]:
                    if support_face[g] and not visited[g]:
                        visited[g] = True
                        stack.append(g)
        islands.append(members)

    rng = random.Random(P.seed)
    me.calc_loop_triangles()
    tris_by_poly = {}
    for lt in me.loop_triangles:
        tris_by_poly.setdefault(lt.polygon_index, []).append(lt)

    iid = 0
    for members in islands:
        isl = Island(idx=iid, faces=members)
        area = 0.0
        flat_area = 0.0
        nz_acc = 0.0
        cen = Vector()
        zmin, zmax = 1e30, -1e30
        member_set = set(members)
        for f in members:
            p = polys[f]
            a = p.area
            area += a
            nz_acc += p.normal.z * a
            if p.normal.z < -0.94:
                flat_area += a
            cen += p.center * a
            for vi in p.vertices:
                z = verts[vi].co.z
                zmin = min(zmin, z)
                zmax = max(zmax, z)
        if area <= 1e-9:
            iid += 1
            continue
        isl.area = area
        isl.flatness = flat_area / area
        isl.mean_nz = nz_acc / area
        isl.centroid = cen / area
        isl.z_min, isl.z_max = zmin, zmax

        for f in members:
            p = polys[f]
            for ek in p.edge_keys:
                fs = edge_faces[ek]
                others = [g for g in fs if g != f and g in member_set]
                if others:
                    continue
                va, vb = verts[ek[0]].co, verts[ek[1]].co
                edge_z = min(va.z, vb.z)
                seg = ((va.x, va.y), (vb.x, vb.y))
                seg_len = (va - vb).length
                isl.boundary_len += seg_len
                anchored = False
                for g in fs:
                    if g == f:
                        continue
                    gz = min(verts[i].co.z for i in polys[g].vertices)
                    if gz < edge_z - 0.5:
                        anchored = True
                        break
                if anchored:
                    isl.anchored_edges.append(seg)
                    isl.anchored_len += seg_len
                else:
                    isl.free_edges.append(seg)

        spacing = max(2.2, min(4.2, sqrt(area) / 6.0))
        isl.spacing = spacing
        candidates = []

        low_band = zmin + 0.8
        for f in members:
            p = polys[f]
            for vi in p.vertices:
                v = verts[vi]
                if v.co.z <= low_band:
                    candidates.append((0, v.co.copy(), p.normal.copy()))

        for (a2, b2) in isl.free_edges:
            mx, my = (a2[0] + b2[0]) * 0.5, (a2[1] + b2[1]) * 0.5
            dx, dy = isl.centroid.x - mx, isl.centroid.y - my
            L = math.hypot(dx, dy)
            if L > 1e-6:
                mx += dx / L * 0.6
                my += dy / L * 0.6
            hit = A.bvh.ray_cast(Vector((mx, my, zmax + 1000.0)),
                                 Vector((0, 0, -1)))
            if hit[0] is not None and hit[2] in member_set:
                candidates.append((1, hit[0].copy(),
                                   polys[hit[2]].normal.copy()))

        target = area / (spacing * spacing * 0.55)
        for f in members:
            p = polys[f]
            for lt in tris_by_poly.get(f, []):
                ta = lt.area
                k = ta / area * target
                n_pts = int(k) + (1 if rng.random() < (k - int(k)) else 0)
                vco = [verts[i].co for i in lt.vertices]
                for _ in range(n_pts):
                    r1, r2 = rng.random(), rng.random()
                    if r1 + r2 > 1.0:
                        r1, r2 = 1.0 - r1, 1.0 - r2
                    co = vco[0] + (vco[1] - vco[0]) * r1 + (vco[2] - vco[0]) * r2
                    candidates.append((2, co.copy(), p.normal.copy()))

        candidates.sort(key=lambda c: c[0])
        accepted = []
        min_d = spacing * 0.8
        for pri, co, nrm in candidates:
            ok = True
            for co2, _ in accepted:
                if (co - co2).length < min_d:
                    ok = False
                    break
            if not ok:
                continue
            backed = False
            for jx, jy in ((0.0, 0.0), (0.13, 0.07), (-0.11, -0.09),
                           (0.13, -0.09), (-0.11, 0.08)):
                if point_inside_solid(A.bvh,
                                      co + Vector((jx, jy, -0.30))):
                    backed = True
                    break
            if backed:
                continue
            accepted.append((co, nrm))
            if len(accepted) >= 400:
                break
        isl.samples = accepted

        plate = 0
        enclosed_hits = 0
        enclosed_rays = 0
        hdirs = [Vector((cos(t), sin(t), 0.0))
                 for t in [i * math.pi / 4 for i in range(8)]]
        for co, nrm in isl.samples[:60]:
            o = co + Vector((0, 0, -0.45))
            hit = A.bvh.ray_cast(o, Vector((0, 0, -1)))
            if hit[0] is None or hit[0].z < 0.05:
                plate += 1
            for d in hdirs + [Vector((0, 0, -1))]:
                enclosed_rays += 1
                h = A.bvh.ray_cast(o + d * 0.05, d)
                if h[0] is not None:
                    enclosed_hits += 1
        n_s = max(1, min(60, len(isl.samples)))
        isl.plate_frac = plate / n_s
        isl.enclosed_frac = (enclosed_hits / enclosed_rays
                             if enclosed_rays else 0.0)

        if isl.enclosed_frac > 0.75:
            sealed = True
            c0 = isl.centroid + Vector((0, 0, -1.0))
            dirs = []
            for iz in (-0.7, 0.0, 0.7):
                for t in [i * math.pi / 5 for i in range(10)]:
                    d = Vector((cos(t), sin(t), iz)).normalized()
                    dirs.append(d)
            dirs += [Vector((0, 0, -1)), Vector((0, 0, 1))]
            for d in dirs:
                h = A.bvh.ray_cast(c0 + d * 0.05, d)
                if h[0] is None:
                    sealed = False
                    break
            isl.sealed = sealed

        A.total_overhang_area += area
        A.islands.append(isl)
        iid += 1

    total = sum(len(isl.samples) for isl in A.islands)
    if total > P.max_points:
        scale = P.max_points / total
        for isl in A.islands:
            keep = max(4, int(len(isl.samples) * scale))
            isl.samples = isl.samples[:keep]
        A.warnings.append(
            "Very large overhang area: support tips thinned to %d points "
            "(coverage spacing widens accordingly)." % P.max_points)
        for isl in A.islands:
            isl.coverage_radius = max(isl.coverage_radius,
                                      isl.spacing * 1.1)

    return A


def _is_bridge(isl: Island, A: Analysis, P: Params):
    if P.bridge_span <= 0.0:
        return False, 0.0
    if isl.area < 10.0:
        return False, 0.0
    if isl.flatness < 0.65:
        return False, 0.0
    if (isl.z_max - isl.z_min) > 1.2:
        return False, 0.0
    if not isl.samples:
        return False, 0.0
    dirs = [(cos(t), sin(t)) for t in [i * math.pi / 4 for i in range(8)]]
    step = 1.0
    n_steps = int(P.bridge_span / step) + 1
    max_span = 0.0
    for co, _ in isl.samples[:48]:
        z = co.z - 0.45
        reach = {}
        for di, (dx, dy) in enumerate(dirs):
            for k in range(1, n_steps + 1):
                t = k * step
                p = Vector((co.x + dx * t, co.y + dy * t, z))
                if point_inside_solid(A.bvh, p):
                    reach[di] = t
                    break
        ok = False
        best = 1e30
        for di in range(4):
            dj = di + 4
            if di in reach and dj in reach:
                span = reach[di] + reach[dj]
                if span <= P.bridge_span:
                    ok = True
                    best = min(best, span)
        if not ok:
            return False, 0.0
        max_span = max(max_span, best)
    return True, max_span


def _undercut_free(A: Analysis, top_z):
    slab = []
    below = []
    for v in A.me.vertices:
        z = v.co.z
        if z <= top_z + 1.5:
            p2 = (v.co.x, v.co.y)
            below.append(p2)
            if z >= top_z - 1.5:
                slab.append(p2)
    if len(slab) < 3:
        return False
    hull = convex_hull_2d(slab)
    if len(hull) < 3:
        return False
    for p2 in below:
        if not point_in_hull(p2, hull, tol=0.8):
            return False
    return True


def classify(A: Analysis, P: Params):
    total_area = A.total_overhang_area
    for isl in A.islands:
        if not isl.samples:
            isl.strategy = STRAT_TREE
            continue

        if isl.sealed:
            isl.strategy = STRAT_SEALED
            isl.note = ("sealed internal void - supports would be trapped; "
                        "left to bridging")
            if isl.area >= 5.0:
                A.warnings.append(
                    "Island %d (%.0f mm^2) is inside a sealed cavity: no "
                    "support generated (it could never be removed). The "
                    "print profile compensates with bridge-friendly "
                    "settings." % (isl.idx, isl.area))
            continue

        is_b, span = _is_bridge(isl, A, P)
        if is_b:
            isl.strategy = STRAT_BRIDGE
            isl.note = "bridgeable span %.1f mm" % span
            A.bridges.append(span)
            continue

        if isl.enclosed_frac > 0.75:
            isl.strategy = STRAT_CAVITY
            isl.spacing = max(2.2, isl.spacing * 0.9)
            isl.coverage_radius = isl.spacing * 0.85
            continue

        if P.strategy_override == "TREE":
            isl.strategy = STRAT_TREE
            isl.coverage_radius = isl.spacing * 0.85
            continue
        if P.strategy_override == "PILLAR":
            isl.strategy = STRAT_PILLAR
            isl.spacing = P.pillar_spacing
            isl.coverage_radius = P.pillar_spacing * 0.75
            continue

        if (P.cradle_enable
                and isl.plate_frac > 0.85
                and isl.area >= 250.0
                and isl.area >= 0.25 * max(total_area, 1e-9)
                and isl.z_min < 0.12 * A.model_h + 1.0
                and isl.z_max <= 0.45 * A.model_h
                and isl.mean_nz < -0.55
                and _undercut_free(A, isl.z_max + 0.8)):
            isl.strategy = STRAT_CRADLE
            isl.coverage_radius = 2.6
            continue

        if (isl.flatness > 0.60 and isl.plate_frac > 0.90
                and isl.area >= 40.0):
            isl.strategy = STRAT_PILLAR
            r_p = P.pillar_d * 0.5
            r_est = r_p
            if isl.z_min / (2.0 * r_p) > P.slender_ratio:
                r_est = min(2.0, isl.z_min / (2.0 * P.slender_ratio))
            isl.spacing = max(P.pillar_spacing, 2.0 * r_est + 1.6)
            isl.coverage_radius = isl.spacing * 0.75
            continue

        isl.strategy = STRAT_TREE
        isl.coverage_radius = max(2.2, isl.spacing * 0.85)


def _radius_for_load(load, r_tip, r_cap):
    return min(r_cap, r_tip * (load ** 0.4))


def _push_clear(bvh, p, r_need, max_push=1.3):
    q = Vector(p)
    for _ in range(3):
        loc, nn, _, dist = bvh.find_nearest(q)
        if loc is None or dist >= r_need:
            return q
        away = q - loc
        away.z = 0.0
        if away.length < 1e-6:
            if nn is None:
                return None
            away = Vector((nn.x, nn.y, 0.0))
            if away.length < 1e-6:
                return q
        need = r_need - dist + 0.05
        if (q - Vector(p)).length + need > max_push:
            return None
        q += away.normalized() * need
    loc, nn, _, dist = bvh.find_nearest(q)
    if loc is None or dist >= r_need * 0.95:
        return q
    return None


class _Node:
    __slots__ = ("p", "r", "load", "alive", "stuck", "skip", "run0")

    def __init__(self, p, r, load=1):
        self.p = p
        self.r = r
        self.load = load
        self.alive = True
        self.stuck = 0
        self.skip = 0
        self.run0 = None


def _flush_run(soup, n):
    if n.run0 is not None:
        soup.tube(n.run0, n.p, n.r, n.r)
        n.run0 = None


def _plate_foot(soup, bvh, p, r):
    r_f = max(r * 1.6, 2.5)
    base = Vector((p.x, p.y, 0.45))
    pushed = _push_clear(bvh, base, r_f + 0.30, max_push=2.5)
    if pushed is None:
        loc, nn, _, dist = bvh.find_nearest(base)
        if loc is not None:
            r_f = max(r, min(r_f, dist - 0.30))
        pushed = base
    elif (pushed - base).length > 0.3:
        mid = Vector(((p.x + pushed.x) / 2, (p.y + pushed.y) / 2, 1.1))
        locm, _, _, dm = bvh.find_nearest(mid)
        if locm is not None and dm < (r + r_f) * 0.5 + 0.30:
            loc, nn, _, dist = bvh.find_nearest(base)
            if loc is not None:
                r_f = max(r, min(r_f, dist - 0.30))
            pushed = base
    bx, by = pushed.x, pushed.y
    for fx, fy, fr in soup.feet:
        d = math.hypot(bx - fx, by - fy)
        if d < fr + r_f + 0.9:
            bx, by = fx, fy
            r_f = max(r_f, fr)
            break
    else:
        soup.feet.append((bx, by, r_f))
    lat = math.hypot(bx - p.x, by - p.y)
    drop = 1.7 - 0.6
    dip = r_f * (lat / max(math.hypot(lat, drop), 1e-6))
    end_z = 0.6 + dip
    soup.tube(Vector((p.x, p.y, 1.7 + dip)), Vector((bx, by, end_z)), r, r_f)
    soup.tube(Vector((bx, by, end_z)), Vector((bx, by, -0.8)), r_f, r_f)


def _model_foot(soup, bvh, hit_p, hit_n, r, gap):
    r_pad = max(r * 1.3, 1.6)
    c = hit_p + Vector((0, 0, gap + 0.6))
    d_lat = 1e30
    for k in range(8):
        a = k * math.pi / 4
        h = bvh.ray_cast(c, Vector((cos(a), sin(a), 0.0)), r_pad + 1.0)
        if h[0] is not None:
            d_lat = min(d_lat, h[3])
    r_pad = min(r_pad, d_lat - 0.35)
    if r_pad < 0.55:
        return None

    def _rise(rp):
        t = hit_p.z
        for k in range(8):
            a = k * math.pi / 4
            o = Vector((hit_p.x + cos(a) * rp, hit_p.y + sin(a) * rp,
                        hit_p.z + gap + 1.6))
            h = bvh.ray_cast(o, Vector((0, 0, -1)))
            if h[0] is not None and h[0].z > t and \
                    (h[0].z - hit_p.z) < gap + 1.5:
                t = h[0].z
        return t

    top_z = _rise(r_pad)
    if top_z - hit_p.z > 0.9:
        r_pad = max(0.9, r_pad * 0.55)
        top_z = _rise(r_pad)
    base = Vector((hit_p.x, hit_p.y, top_z + gap))
    top = base + Vector((0, 0, 1.0))
    soup.tube(base, top, r_pad * 0.85, r_pad)
    return top


def grow_forest(soup, tips, A, P: Params, tip_r, branch_r, cap_r,
                contacts):
    if not tips:
        return 0
    gap = P.z_gap
    taper = P.tip_taper_len
    lat_max = tan(radians(P.branch_angle)) * P.step_dz
    nodes = []

    for co, nrm, meta in tips:
        apex = co + Vector((0, 0, -gap))
        h = Vector((nrm.x, nrm.y, 0.0))
        if h.length > 1e-6 and abs(nrm.z) > 1e-6:
            tan_th = h.length / abs(nrm.z)
            if tan_th > 0.35:
                off = min(tip_r * 1.6 + 0.20, tip_r * tan_th + 0.10)
                if nrm.z > -0.5:
                    off = max(off, 0.35 + tip_r)
                apex += h.normalized() * off

        if apex.z < 0.55:
            if meta is not None:
                meta["strat"] = "TIGHT"
            continue

        r_eff = branch_r
        tip_r_eff = tip_r

        def _cone_clear(ap, r_low, r_top):
            p_low = ap + Vector((0, 0, -taper))
            q = _push_clear(A.bvh, p_low, r_low + 0.30, max_push=1.3)
            if q is None:
                return None
            sh = q - p_low
            sh.z = 0.0
            ap2 = ap + sh
            p_mid = ap2 + Vector((0, 0, -taper * 0.45))
            r_mid = (r_low + r_top) * 0.5
            q2 = _push_clear(A.bvh, p_mid, r_mid + 0.25, max_push=0.9)
            if q2 is None:
                return None
            sh2 = q2 - p_mid
            sh2.z = 0.0
            return ap2 + sh2

        new_apex = _cone_clear(apex, branch_r, tip_r)
        if new_apex is not None:
            apex = new_apex
        else:
            slim_r = max(0.6, P.min_feature * 0.5)
            new_apex = _cone_clear(apex, slim_r, slim_r)
            if new_apex is None:
                if meta is not None:
                    meta["strat"] = "TIGHT"
                continue
            apex = new_apex
            r_eff = slim_r
            tip_r_eff = min(tip_r, slim_r + 0.1)

        hit0 = A.bvh.ray_cast(apex + Vector((0, 0, -0.02)),
                              Vector((0, 0, -1)))
        d_below = (apex.z - hit0[0].z) if hit0[0] is not None else apex.z
        if d_below < taper + 1.0:
            if d_below >= 1.4 and hit0[0] is not None and hit0[1].z > 0.25:
                pad_top = _model_foot(soup, A.bvh, hit0[0], hit0[1],
                                      max(tip_r_eff, 0.6), gap)
                if pad_top is not None and apex.z - pad_top.z > 0.3:
                    soup.tube(pad_top, apex, max(tip_r_eff, 0.6), tip_r_eff)
                    contacts.append(co.copy())
                else:
                    if meta is not None:
                        meta["strat"] = "TIGHT"
            elif hit0[0] is None and d_below >= 1.0:
                r_mic = max(tip_r_eff, 0.8)
                too_close = any(
                    math.hypot(apex.x - fx, apex.y - fy) < 2.6
                    for fx, fy, fr in soup.feet)
                if too_close:
                    contacts.append(co.copy())
                    continue
                def _col_ok(ax, ay):
                    for fz in (0.30, 0.55, 0.80):
                        pz = Vector((ax, ay, apex.z * fz))
                        loc2, _, _, dc = A.bvh.find_nearest(pz)
                        if loc2 is not None and dc < r_mic + 0.25:
                            return False
                    return True
                ax, ay = apex.x, apex.y
                top_z = apex.z
                if not _col_ok(ax, ay):
                    mid = Vector((ax, ay, apex.z * 0.5))
                    pushed2 = _push_clear(A.bvh, mid, r_mic + 0.30,
                                          max_push=0.9)
                    if pushed2 is not None and _col_ok(pushed2.x, pushed2.y):
                        ax, ay = pushed2.x, pushed2.y
                        h2 = A.bvh.ray_cast(Vector((ax, ay, top_z + 60.0)),
                                            Vector((0, 0, -1)))
                        if h2[0] is not None:
                            top_z = min(top_z, h2[0].z - gap)
                        if top_z < 1.0:
                            if meta is not None:
                                meta["strat"] = "TIGHT"
                            continue
                    else:
                        if meta is not None:
                            meta["strat"] = "TIGHT"
                        continue
                soup.tube(Vector((ax, ay, -0.8)),
                          Vector((ax, ay, top_z)), r_mic, tip_r_eff)
                soup.feet.append((ax, ay, r_mic))
                contacts.append(co.copy())
            else:
                if meta is not None:
                    meta["strat"] = "TIGHT"
            continue

        start = apex + Vector((0, 0, -taper))
        soup.tube(start, apex, r_eff, tip_r_eff)
        if r_eff >= branch_r:
            soup.sphere(start, r_eff)
        contacts.append(co.copy())
        nodes.append(_Node(start, r_eff))

    bvh = A.bvh
    clear = P.xy_clearance
    guard = int(max(60.0, A.model_h + 60.0) / P.step_dz) + 80
    segs = 0

    for _ in range(guard):
        alive = [n for n in nodes if n.alive]
        if not alive:
            break
        if len(alive) > 1:
            kd = KDTree(len(alive))
            for i, n in enumerate(alive):
                kd.insert(n.p, i)
            kd.balance()
            merged = set()
            for i, n in enumerate(alive):
                if i in merged:
                    continue
                for (co, j, d) in kd.find_range(n.p, (n.r * 2.0 + 1.2)):
                    if j == i or j in merged or i in merged:
                        continue
                    m = alive[j]
                    if not m.alive:
                        continue
                    w1, w2 = n.load, m.load
                    pm = (n.p * w1 + m.p * w2) / (w1 + w2)
                    pm.z = min(n.p.z, m.p.z)
                    new_r = _radius_for_load(w1 + w2, branch_r, cap_r)
                    blocked = False
                    for q in (pm,
                              n.p.lerp(pm, 0.4), n.p.lerp(pm, 0.75),
                              m.p.lerp(pm, 0.4), m.p.lerp(pm, 0.75)):
                        locm, _, _, dmid = bvh.find_nearest(q)
                        if locm is not None and dmid < new_r + 0.45:
                            blocked = True
                            break
                    if blocked:
                        continue
                    _flush_run(soup, n)
                    _flush_run(soup, m)
                    soup.tube(n.p, pm, n.r, new_r)
                    soup.tube(m.p, pm, m.r, new_r)
                    if pm.z > new_r + 0.05:
                        soup.sphere(pm, new_r)
                    n.p, n.r, n.load = pm, new_r, w1 + w2
                    m.alive = False
                    merged.add(j)

        for n in nodes:
            if not n.alive:
                continue
            if n.p.z <= P.step_dz + 0.6:
                _flush_run(soup, n)
                _plate_foot(soup, bvh, n.p, n.r)
                soup.tube(n.p, Vector((n.p.x, n.p.y, 1.7)), n.r, n.r)
                if n.p.z > n.r + 0.05:
                    soup.sphere(n.p, n.r)
                n.alive = False
                continue

            if n.skip > 0:
                n.skip -= 1
                if n.run0 is None:
                    n.run0 = n.p.copy()
                n.p = n.p + Vector((0, 0, -P.step_dz))
                segs += 1
                continue
            _flush_run(soup, n)

            pull = Vector((0, 0, 0))
            best_d = P.merge_radius
            for m in nodes:
                if m is n or not m.alive:
                    continue
                d = (m.p - n.p).length
                if d < best_d:
                    best_d = d
                    pull = (m.p - n.p)
            tgt = None
            if pull.length > 1e-6:
                h = Vector((pull.x, pull.y, 0.0))
                if h.length > lat_max:
                    h = h.normalized() * lat_max
                tgt = n.p + h + Vector((0, 0, -P.step_dz))

            chosen = None
            clear_d = 0.0
            for cand in ([tgt] if tgt is not None else []) + \
                        [n.p + Vector((0, 0, -P.step_dz))]:
                loc, nrm, _, dist = bvh.find_nearest(cand)
                if loc is None or dist >= n.r + clear * 0.55:
                    chosen = cand
                    clear_d = 1e9 if loc is None else dist
                    break
            if chosen is None:
                loc, nrm, _, dist = bvh.find_nearest(n.p)
                if loc is not None:
                    away = n.p - loc
                    away.z = 0.0
                    if away.length > 1e-6:
                        cand = (n.p + away.normalized() * lat_max
                                + Vector((0, 0, -P.step_dz)))
                        l2, _, _, d2 = bvh.find_nearest(cand)
                        if l2 is None or d2 >= n.r + clear * 0.45:
                            chosen = cand

            if chosen is None:
                n.stuck += 1
                if n.stuck < 3:
                    continue
                hit = bvh.ray_cast(n.p, Vector((0, 0, -1)))
                top = None
                if hit[0] is not None and hit[1].z > 0.25:
                    top = _model_foot(soup, A.bvh, hit[0], hit[1], n.r, gap)
                if top is not None:
                    soup.tube(n.p, top, n.r, n.r)
                    soup.sphere(n.p, n.r)
                else:
                    soup.sphere(n.p, n.r)
                n.alive = False
                continue

            n.stuck = 0
            hit = bvh.ray_cast(chosen, Vector((0, 0, -1)))
            if hit[0] is not None and (chosen.z - hit[0].z) <= P.step_dz + gap + 1.2:
                if hit[1].z > 0.25:
                    top = _model_foot(soup, A.bvh, hit[0], hit[1], n.r, gap)
                    if top is None:
                        n.stuck += 1
                        continue
                    soup.tube(chosen, top, n.r, n.r)
                    soup.tube(n.p, chosen, n.r, n.r)
                    n.alive = False
                    continue
            soup.tube(n.p, chosen, n.r, n.r)
            if chosen.z > n.r + 0.05:
                soup.sphere(chosen, n.r)
            n.p = chosen
            segs += 1
            if clear_d < 1e8:
                n.skip = max(0, min(12, int(
                    (clear_d - n.r - clear * 0.55)
                    / (P.step_dz * 1.9)) - 1))
            else:
                n.skip = 12

    for n in nodes:
        if n.alive:
            _flush_run(soup, n)
    return segs


def gen_pillars(soup, isl: Island, A: Analysis, P: Params, contacts):
    spacing = max(isl.spacing, P.pillar_spacing)
    sites = []
    for co, nrm in isl.samples:
        ok = True
        for s in sites:
            if (co - s[0]).length < spacing * 0.9:
                ok = False
                break
        if ok:
            sites.append((co, nrm))

    r_p = P.pillar_d * 0.5
    tip_r = P.tip_contact_d * 0.5
    columns = []
    for co, nrm in sites:
        apex = co + Vector((0, 0, -P.z_gap))
        probe = apex + Vector((0, 0, -P.tip_taper_len))
        pushed = _push_clear(A.bvh, probe, r_p + 0.30, max_push=1.4)
        if pushed is None:
            continue
        shift = pushed - probe
        shift.z = 0.0
        apex = apex + shift
        hit = A.bvh.ray_cast(apex + Vector((0, 0, -0.02)),
                             Vector((0, 0, -1)))
        on_model = hit[0] is not None and hit[0].z > 0.05
        bottom_z = (hit[0].z if on_model else 0.0)
        height = apex.z - bottom_z
        if height < 1.0:
            continue
        r_use = r_p
        if height / (2 * r_p) > P.slender_ratio:
            r_use = min(2.0, height / (2 * P.slender_ratio))
        too_close = False
        for (a_prev, bz_prev, r_prev) in columns:
            if math.hypot(apex.x - a_prev.x, apex.y - a_prev.y) \
                    < r_use + r_prev + 0.9:
                too_close = True
                break
        if too_close:
            continue
        cone_end = apex + Vector((0, 0, -P.tip_taper_len))
        if on_model:
            if height < P.tip_taper_len + 1.6:
                if height >= 1.4 and hit[1].z > 0.25:
                    pad_top = _model_foot(soup, A.bvh, hit[0], hit[1],
                                          max(tip_r, 0.6), P.z_gap)
                    if pad_top is not None and apex.z - pad_top.z > 0.3:
                        soup.tube(pad_top, apex, max(tip_r, 0.6), tip_r)
                        contacts.append(co.copy())
                continue
            top = _model_foot(soup, A.bvh, hit[0], hit[1], r_use, P.z_gap)
            if top is None:
                continue
            soup.tube(cone_end, apex, r_use, tip_r)
            contacts.append(co.copy())
            if cone_end.z > top.z + 0.2:
                soup.tube(top, cone_end, r_use, r_use)
        else:
            soup.tube(cone_end, apex, r_use, tip_r)
            contacts.append(co.copy())
            _plate_foot(soup, A.bvh, Vector((apex.x, apex.y, 0.0)), r_use)
            soup.tube(Vector((apex.x, apex.y, 1.7)), cone_end, r_use, r_use)
        columns.append((apex, bottom_z, r_use))

    if len(columns) >= 2:
        kd = KDTree(len(columns))
        for i, (apex, bz, r) in enumerate(columns):
            kd.insert(apex, i)
        kd.balance()
        linked = set()
        rib_r = max(P.rib_d * 0.5, 0.6)
        for i, (apex, bz, r) in enumerate(columns):
            height = apex.z - bz
            if height < P.slender_ratio * 2 * r * 0.8:
                continue
            for (co, j, d) in kd.find_range(apex, P.pillar_spacing * 1.9):
                if j == i or (min(i, j), max(i, j)) in linked:
                    continue
                linked.add((min(i, j), max(i, j)))
                a2, bz2, r2 = columns[j]
                n_levels = max(1, int(height / 14.0))
                for k in range(1, n_levels + 1):
                    z = bz + (apex.z - P.tip_taper_len - 1.5 - bz) * k / (n_levels + 0.0)
                    z = max(z, max(bz, bz2) + 2.0)
                    pa = Vector((apex.x, apex.y, z))
                    pb = Vector((a2.x, a2.y, z))
                    clear = True
                    for t in (0.25, 0.5, 0.75):
                        loc, _, _, dr = A.bvh.find_nearest(pa.lerp(pb, t))
                        if loc is not None and dr < rib_r + 0.35:
                            clear = False
                            break
                    if clear:
                        soup.tube(pa, pb, rib_r, rib_r)
                if len(linked) > 3 * len(columns):
                    break
    return len(columns)


def gen_cradle(isl: Island, metas, A: Analysis, P: Params, collection,
               contacts):
    pad = P.cradle_margin
    xs = [A.me.vertices[vi].co.x for f in isl.faces
          for vi in A.me.polygons[f].vertices]
    ys = [A.me.vertices[vi].co.y for f in isl.faces
          for vi in A.me.polygons[f].vertices]
    if not xs:
        return None, []
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    top_z = isl.z_max + 0.8
    if top_z < 1.5:
        top_z = 1.5
    z_lo = -2.5

    net_erosion = max(P.cradle_gap, 0.45) + 0.60 - 0.30 + 0.15
    danger_h = net_erosion + 0.7
    step = 2.0
    marked = {}
    nx = int((x1 - x0) / step) + 1
    ny = int((y1 - y0) / step) + 1
    for ix in range(nx):
        for iy in range(ny):
            px = x0 + ix * step
            py = y0 + iy * step
            hit = A.bvh.ray_cast(Vector((px, py, 0.02)), Vector((0, 0, 1)))
            if hit[0] is not None and hit[0].z < danger_h:
                marked[(ix, iy)] = (px, py)
    clusters = []
    seen_c = set()
    for key in marked:
        if key in seen_c:
            continue
        stack = [key]
        seen_c.add(key)
        cells = []
        while stack:
            k = stack.pop()
            cells.append(marked[k])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nk = (k[0] + dx, k[1] + dy)
                if nk in marked and nk not in seen_c:
                    seen_c.add(nk)
                    stack.append(nk)
        ccx = sum(c[0] for c in cells) / len(cells)
        ccy = sum(c[1] for c in cells) / len(cells)
        rad = max(math.hypot(c[0] - ccx, c[1] - ccy)
                  for c in cells) + step * 0.8 + 1.0
        clusters.append((ccx, ccy, rad))

    leftover_idx = set()
    for k, (co, nrm) in enumerate(isl.samples):
        for (ccx, ccy, rad) in clusters:
            if math.hypot(co.x - ccx, co.y - ccy) <= rad + 0.5:
                leftover_idx.add(k)
                break

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    mat = (Matrix.Translation(Vector(((x0 + x1) / 2, (y0 + y1) / 2,
                                      (top_z + z_lo) / 2)))
           @ Matrix.Diagonal(Vector((x1 - x0, y1 - y0, top_z - z_lo, 1.0))))
    bmesh.ops.transform(bm, matrix=mat, verts=bm.verts[:])
    v_edges = [e for e in bm.edges
               if abs(e.verts[0].co.x - e.verts[1].co.x) < 1e-4
               and abs(e.verts[0].co.y - e.verts[1].co.y) < 1e-4]
    if v_edges:
        bmesh.ops.bevel(bm, geom=v_edges, offset=min(2.5, pad * 0.8),
                        offset_type='OFFSET', segments=3, profile=0.5,
                        affect='EDGES', clamp_overlap=True, loop_slide=True)
    box_me = bpy.data.meshes.new("FDMW_cradle_box")
    bm.to_mesh(box_me)
    bm.free()
    box = _new_object("FDMW_cradle_tmp", box_me, collection)

    model_me = A.me.copy()
    model_ob = _new_object("FDMW_model_tmp", model_me, collection)

    cutters = None
    if clusters:
        cbm = bmesh.new()
        for (ccx, ccy, rad) in clusters:
            cmat = Matrix.Translation(Vector((ccx, ccy, (top_z + z_lo) / 2)))
            bmesh.ops.create_cone(cbm, cap_ends=True, cap_tris=True,
                                  segments=28, radius1=rad, radius2=rad,
                                  depth=(top_z - z_lo) + 2.0, matrix=cmat)
        cme = bpy.data.meshes.new("FDMW_cradle_cut")
        cbm.to_mesh(cme)
        cbm.free()
        cutters = _new_object("FDMW_cradle_cut", cme, collection)

    box_vol = (x1 - x0) * (y1 - y0) * (top_z - z_lo)
    try:
        vox = max(0.20, P.voxel_size * 0.7)

        def _disp(name, amount):
            dm = box.modifiers.new(name, 'DISPLACE')
            dm.mid_level = 0.0
            dm.strength = amount
            dm.direction = 'NORMAL'
            rm = box.modifiers.new(name + "_rm", 'REMESH')
            rm.mode = 'VOXEL'
            rm.voxel_size = vox

        b = box.modifiers.new("cut", 'BOOLEAN')
        b.operation = 'DIFFERENCE'
        b.solver = 'EXACT'
        b.object = model_ob
        if cutters is not None:
            b2 = box.modifiers.new("cut_tangent", 'BOOLEAN')
            b2.operation = 'DIFFERENCE'
            b2.solver = 'EXACT'
            b2.object = cutters
        r1 = box.modifiers.new("rm1", 'REMESH')
        r1.mode = 'VOXEL'
        r1.voxel_size = vox
        gap_e = max(P.cradle_gap, 0.45)
        _disp("erode", -gap_e)
        _disp("open_shrink", -0.60)
        _disp("open_grow", 0.30)
        _disp("deburr_shrink", -0.15)
        _apply_modifiers_inplace(box)

        bm = bmesh.new()
        bm.from_mesh(box.data)
        vol = bm.calc_volume(signed=False)
        bm.free()
        if vol > 0.97 * box_vol or vol < 0.02 * box_vol:
            raise RuntimeError("boolean produced a degenerate cradle")

        gap_ok = False
        for k in range(3):
            cb = bmesh.new()
            cb.from_mesh(box.data)
            cradle_bvh = BVHTree.FromBMesh(cb)
            cb.free()
            if not cradle_bvh.overlap(A.bvh):
                gap_ok = True
                break
            if k == 2:
                break
            d_fix = box.modifiers.new("erode_fix", 'DISPLACE')
            d_fix.mid_level = 0.0
            d_fix.strength = -0.15
            d_fix.direction = 'NORMAL'
            r_fix = box.modifiers.new("rm_fix", 'REMESH')
            r_fix.mode = 'VOXEL'
            r_fix.voxel_size = vox
            _apply_modifiers_inplace(box)
        if not gap_ok:
            raise RuntimeError("cradle still touching the model")

        _plate_trim_boolean(box, collection)
        if _strip_thin_features(box, collection, vox):
            _plate_trim_boolean(box, collection)

        bm = bmesh.new()
        bm.from_mesh(box.data)
        _ensure_lookup(bm)
        seen = set()
        shells = []
        for v in bm.verts:
            if v.index in seen:
                continue
            stack = [v]
            seen.add(v.index)
            shell = []
            while stack:
                vv = stack.pop()
                shell.append(vv)
                for e in vv.link_edges:
                    o = e.other_vert(vv)
                    if o.index not in seen:
                        seen.add(o.index)
                        stack.append(o)
            shells.append(shell)
        if len(shells) > 1:
            kill = []
            for shell in shells:
                sxs = [v.co.x for v in shell]
                sys_ = [v.co.y for v in shell]
                szs = [v.co.z for v in shell]
                dx = max(sxs) - min(sxs)
                dy = max(sys_) - min(sys_)
                dz = max(szs) - min(szs)
                if min(dx, dy, dz) < P.min_feature * 0.9 or dx * dy * dz < 30.0:
                    kill.extend(shell)
            if kill:
                bmesh.ops.delete(bm, geom=kill, context='VERTS')
        out = bpy.data.meshes.new("FDMW_cradle")
        bm.to_mesh(out)
        bm.free()
        if len(out.polygons) < 4:
            bpy.data.meshes.remove(out)
            raise RuntimeError("cradle vanished after cleanup")

        abm = bmesh.new()
        abm.from_mesh(out)
        accept_bvh = BVHTree.FromBMesh(abm)
        bad_thin = _count_nontaper_thin(abm, accept_bvh, stop_at=9)
        abm.free()
        if bad_thin >= 9:
            bpy.data.meshes.remove(out)
            raise RuntimeError("residual thin features")

        leftovers = []
        for k, (co, nrm) in enumerate(isl.samples):
            if k in leftover_idx:
                m = metas[k] if metas and k < len(metas) else None
                leftovers.append((co, nrm, m))
            else:
                contacts.append(co.copy())
        return out, leftovers
    except Exception as e:
        isl.note = "cradle failed: %s" % e
        return None, []
    finally:
        _remove_object(box)
        _remove_object(model_ob)
        if cutters is not None:
            _remove_object(cutters)


def finalize_soup(soup: PrimitiveSoup, P: Params, collection, name):
    if soup.count == 0:
        soup.free()
        return None
    me = soup.to_mesh(name + "_soup")
    ob = _new_object(name + "_tmp", me, collection)
    r = ob.modifiers.new("voxel", 'REMESH')
    r.mode = 'VOXEL'
    r.voxel_size = P.voxel_size
    _apply_modifiers_inplace(ob)
    _plate_trim_boolean(ob, collection)

    bm = bmesh.new()
    bm.from_mesh(ob.data)
    _ensure_lookup(bm)
    seen = set()
    for v in bm.verts:
        if v.index in seen:
            continue
        stack = [v]
        seen.add(v.index)
        shell = []
        while stack:
            vv = stack.pop()
            shell.append(vv)
            for e in vv.link_edges:
                o = e.other_vert(vv)
                if o.index not in seen:
                    seen.add(o.index)
                    stack.append(o)
        if len(shell) < 30:
            bmesh.ops.delete(bm, geom=shell, context='VERTS')
    out = bpy.data.meshes.new(name)
    bm.to_mesh(out)
    bm.free()
    _remove_object(ob)
    if len(out.polygons) == 0:
        bpy.data.meshes.remove(out)
        return None
    return out


def merge_meshes(meshes, name):
    bm = bmesh.new()
    for me in meshes:
        bm.from_mesh(me)
    out = bpy.data.meshes.new(name)
    bm.to_mesh(out)
    bm.free()
    return out


def _count_nontaper_thin(bm, bvh, wall_floor=0.8, stack_floor=0.4,
                         stop_at=9):
    def thickness(o, n):
        h = bvh.ray_cast(o - n * 0.04, -n)
        if h[0] is None:
            return None
        t = h[3] + 0.04
        return None if t > 60.0 else t

    def taper(c, n, floor):
        t1 = n.cross(Vector((0, 0, 1)))
        if t1.length < 0.1:
            t1 = n.cross(Vector((1, 0, 0)))
        t1.normalize()
        t2 = n.cross(t1).normalized()
        good = 0
        for d in (t1, -t1, t2, -t2):
            loc, nn, _, _ = bvh.find_nearest(c + d * 1.0)
            if loc is None or nn is None or nn.length < 0.5:
                continue
            tt = thickness(loc, nn)
            if tt is not None and tt >= floor * 0.95:
                good += 1
        return good >= 2

    bad = 0
    for f in bm.faces:
        n = f.normal
        if n.length < 0.5:
            continue
        c = f.calc_center_median()
        t = thickness(c, n)
        if t is None:
            continue
        floor = wall_floor if abs(n.z) < 0.6 else stack_floor
        if t < floor and not taper(c, n, floor):
            bad += 1
            if bad >= stop_at:
                return bad
    return bad


def _strip_thin_features(ob, collection, vox, floor=0.45, rounds=2):
    did = False
    for _ in range(rounds):
        me = ob.data
        bm = bmesh.new()
        bm.from_mesh(me)
        _ensure_lookup(bm)
        bvh = BVHTree.FromBMesh(bm)
        thin = []
        for f in bm.faces:
            n = f.normal
            if n.length < 0.5:
                continue
            c = f.calc_center_median()
            hit = bvh.ray_cast(c - n * 0.03, -n)
            if hit[0] is not None and 0.0 < (hit[3] + 0.03) < floor:
                thin.append(c.copy())
        if not thin:
            bm.free()
            return did
        kd = KDTree(len(thin))
        for k, c in enumerate(thin):
            kd.insert(c, k)
        kd.balance()
        kill = [f for f in bm.faces
                if kd.find_range(f.calc_center_median(), 0.9)]
        bmesh.ops.delete(bm, geom=kill, context='FACES')
        bm.to_mesh(me)
        bm.free()
        r = ob.modifiers.new("rm_strip", 'REMESH')
        r.mode = 'VOXEL'
        r.voxel_size = vox
        _apply_modifiers_inplace(ob)
        did = True
    return did


def _carve_collisions(sup_me, A, P, collection, max_rounds=2):
    me = sup_me
    for _ in range(max_rounds):
        bm = bmesh.new()
        bm.from_mesh(me)
        _ensure_lookup(bm)
        bvh = BVHTree.FromBMesh(bm)
        pairs = bvh.overlap(A.bvh) if A.bvh else []
        if not pairs:
            bm.free()
            return me
        bad = sorted(set(i for i, j in pairs if i < len(bm.faces)))
        if not bad:
            bm.free()
            return me
        kd = KDTree(len(bad))
        for k, i in enumerate(bad):
            kd.insert(bm.faces[i].calc_center_median(), k)
        kd.balance()
        kill = [f for f in bm.faces
                if kd.find_range(f.calc_center_median(), 1.0)]
        bmesh.ops.delete(bm, geom=kill, context='FACES')
        carved = bpy.data.meshes.new(me.name + "_c")
        bm.to_mesh(carved)
        bm.free()
        ob = _new_object("FDMW_carve_tmp", carved, collection)
        r = ob.modifiers.new("rm", 'REMESH')
        r.mode = 'VOXEL'
        r.voxel_size = P.voxel_size
        _apply_modifiers_inplace(ob)
        _plate_trim_boolean(ob, collection)
        new_me = ob.data
        bpy.data.objects.remove(ob)
        if me is not sup_me and me.users == 0:
            bpy.data.meshes.remove(me)
        me = new_me
    return me


def _plate_trim_boolean(ob, collection):
    me = ob.data
    if not me.vertices:
        return
    xs = [v.co.x for v in me.vertices]
    ys = [v.co.y for v in me.vertices]
    zs = [v.co.z for v in me.vertices]
    if min(zs) >= -1e-4:
        return
    bm = bmesh.new()
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    sx = (max(xs) - min(xs)) + 20.0
    sy = (max(ys) - min(ys)) + 20.0
    h = max(zs) + 10.0
    mat = (Matrix.Translation(Vector((cx, cy, h / 2)))
           @ Matrix.Diagonal(Vector((sx, sy, h, 1.0))))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)
    bme = bpy.data.meshes.new("FDMW_plate_box")
    bm.to_mesh(bme)
    bm.free()
    box_ob = _new_object("FDMW_plate_box", bme, collection)
    mod = ob.modifiers.new("plate_trim", 'BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.solver = 'EXACT'
    mod.object = box_ob
    _apply_modifiers_inplace(ob)
    _remove_object(box_ob)


def validate(support_me, A: Analysis, all_samples, P: Params):
    V = {}
    bm = bmesh.new()
    bm.from_mesh(support_me)
    _ensure_lookup(bm)

    non_manifold_edges = sum(1 for e in bm.edges if not e.is_manifold)
    boundary_edges = sum(1 for e in bm.edges if e.is_boundary)
    loose_verts = sum(1 for v in bm.verts if not v.link_edges)
    V["manifold"] = {
        "pass": non_manifold_edges == 0 and boundary_edges == 0
                and loose_verts == 0,
        "non_manifold_edges": non_manifold_edges,
        "boundary_edges": boundary_edges,
        "loose_verts": loose_verts,
        "verts": len(bm.verts), "faces": len(bm.faces),
        "volume_mm3": round(bm.calc_volume(signed=False), 1),
    }

    sup_bvh = BVHTree.FromBMesh(bm)

    overlap = sup_bvh.overlap(sup_bvh)
    bm.faces.ensure_lookup_table()
    bad_pairs = 0
    for (i, j) in overlap:
        if i == j:
            continue
        fi = bm.faces[i]
        fj = bm.faces[j]
        si = set(v.index for v in fi.verts)
        if any(v.index in si for v in fj.verts):
            continue
        bad_pairs += 1
    V["self_intersection"] = {"pass": bad_pairs == 0, "bad_pairs": bad_pairs}

    col = sup_bvh.overlap(A.bvh) if A.bvh else []
    col_locs = []
    for (i, j) in col[:6]:
        if i < len(bm.faces):
            c = bm.faces[i].calc_center_median()
            col_locs.append([round(c.x, 1), round(c.y, 1), round(c.z, 1)])
    V["model_collision"] = {"pass": len(col) == 0,
                            "overlap_pairs": len(col),
                            "sample_locations": col_locs}

    skip_strats = (STRAT_BRIDGE, STRAT_SEALED, "TIGHT")
    gaps = []
    for s in all_samples:
        if s["strat"] in skip_strats:
            continue
        loc, nrm, _, dist = sup_bvh.find_nearest(s["co"])
        if loc is not None:
            gaps.append(dist)
    if gaps:
        V["contact_gap"] = {
            "pass": min(gaps) > 0.04,
            "min": round(min(gaps), 3),
            "mean": round(sum(gaps) / len(gaps), 3),
        }
    else:
        V["contact_gap"] = {"pass": True, "min": None, "mean": None}

    needed = 0
    covered = 0
    misses = []
    for s in all_samples:
        if s["strat"] in skip_strats:
            continue
        co = s["co"]
        needed += 1
        loc, nrm, _, dist = sup_bvh.find_nearest(co)
        if loc is not None and dist <= s["rad"] + P.z_gap + 0.8:
            covered += 1
        else:
            misses.append((round(co.x, 1), round(co.y, 1), round(co.z, 1)))
    cov = (covered / needed) if needed else 1.0
    V["coverage"] = {
        "pass": cov >= 0.92,
        "covered": covered, "needed": needed,
        "fraction": round(cov, 4),
        "sample_misses": misses[:8],
    }

    rng = random.Random(P.seed + 7)
    faces = bm.faces[:]
    n_take = min(700, len(faces))
    take = rng.sample(faces, n_take) if len(faces) > n_take else faces

    def _thickness_at(o, n):
        loc, nrm2, idx2, dist = sup_bvh.ray_cast(o - n * 0.04, -n)
        if loc is None:
            return None
        t = dist + 0.04
        return None if t > 60.0 else t

    def _is_taper_tip(c, n, floor):
        t1 = n.cross(Vector((0, 0, 1)))
        if t1.length < 0.1:
            t1 = n.cross(Vector((1, 0, 0)))
        t1.normalize()
        t2 = n.cross(t1).normalized()
        good = 0
        for d in (t1, -t1, t2, -t2):
            loc, nn, _, _ = sup_bvh.find_nearest(c + d * 1.0)
            if loc is None or nn is None or nn.length < 0.5:
                continue
            t = _thickness_at(loc, nn)
            if t is not None and t >= floor * 0.95:
                good += 1
        return good >= 2

    wall_min = 1e30
    stack_min = 1e30
    n_wall = n_stack = 0
    n_taper = 0
    worst = None
    for f in take:
        c = f.calc_center_median()
        n = f.normal
        if n.length < 0.5:
            continue
        t = _thickness_at(c, n)
        if t is None:
            continue
        is_wall = abs(n.z) < 0.6
        floor = 0.8 if is_wall else 0.4
        if t < floor and _is_taper_tip(c, n, floor):
            n_taper += 1
            continue
        if is_wall:
            n_wall += 1
            if t < wall_min:
                wall_min = t
                worst = [round(c.x, 1), round(c.y, 1), round(c.z, 1)]
        else:
            n_stack += 1
            stack_min = min(stack_min, t)
    ok = (n_wall == 0 or wall_min >= 0.8) and \
         (n_stack == 0 or stack_min >= 0.4)
    V["min_thickness"] = {
        "pass": ok,
        "min_wall_mm": round(wall_min, 3) if n_wall else None,
        "min_stack_mm": round(stack_min, 3) if n_stack else None,
        "wall_floor_mm": 0.8, "stack_floor_mm": 0.4,
        "thinnest_at": worst,
        "taper_tips_excluded": n_taper,
        "samples": n_wall + n_stack,
    }

    bm.free()
    V["all_pass"] = all(v.get("pass", True) for v in V.values()
                        if isinstance(v, dict))
    return V


def model_min_wall(A: Analysis, P: Params, n_rays=200):
    rng = random.Random(P.seed + 13)
    polys = A.me.polygons
    if not polys:
        return None
    idxs = [rng.randrange(len(polys)) for _ in range(n_rays)]
    t_min = 1e30
    for i in idxs:
        p = polys[i]
        n = p.normal
        if n.length < 0.5:
            continue
        o = p.center - n * 0.04
        hit = A.bvh.ray_cast(o, -n)
        if hit[0] is None:
            continue
        t = hit[3] + 0.04
        if 0.1 < t < 50.0:
            t_min = min(t_min, t)
    return None if t_min > 1e29 else t_min


def recommend_profile(A: Analysis, P: Params, strat_counts, validation):
    bx, by, bz, inherit_sfx = PRINTERS.get(P.printer, PRINTERS["X1C"])
    ntemp, btemp, bridge_f, fan = MATERIALS.get(P.material, MATERIALS["PLA"])
    dims = A.model_dims
    notes = []
    if dims[0] > bx or dims[1] > by or dims[2] > bz:
        notes.append("WARNING: model (%.0fx%.0fx%.0f mm) exceeds the %s build "
                     "volume (%.0fx%.0fx%.0f mm)."
                     % (dims[0], dims[1], dims[2], P.printer, bx, by, bz))

    min_wall = model_min_wall(A, P)
    max_dim = max(dims)

    if (min_wall is not None and min_wall < 1.5) or max_dim < 30.0:
        layer = 0.12
        why_layer = "fine detail (thin walls or small model)"
    elif A.total_overhang_area > 0.15 * max(1.0, dims[0] * dims[1]) * 2:
        layer = 0.16
        why_layer = "many curved overhangs - finer layers reduce stair-stepping"
    elif max_dim > 150.0:
        layer = 0.24
        why_layer = "large model - thicker layers keep print time sane"
    else:
        layer = 0.20
        why_layer = "standard quality/speed balance"

    walls = 3 if (max_dim > 80.0 or A.bridges) else 2
    infill = 15
    if dims[2] > 2.5 * max(dims[0], dims[1]):
        infill = 20

    footprint = dims[0] * dims[1]
    slender = dims[2] > 3.0 * sqrt(max(footprint, 1e-9))
    brim = "outer_only" if slender else "no_brim"
    brim_width = 5.0 if slender else 0.0

    longest_bridge = max(A.bridges) if A.bridges else 0.0
    bridge_speed = 20 if longest_bridge > 15.0 else 25

    over = {
        "layer_height": "%.2f" % layer,
        "initial_layer_print_height": "0.20",
        "wall_loops": str(walls),
        "sparse_infill_density": "%d%%" % infill,
        "sparse_infill_pattern": "gyroid" if slender else "grid",
        "enable_support": "0",
        "brim_type": brim,
        "brim_width": "%.0f" % brim_width,
        "bridge_speed": "%d" % bridge_speed,
        "bridge_flow": "0.95",
        "overhang_fan_speed": str(fan),
        "overhang_fan_threshold": "25%",
        "seam_position": "aligned",
        "line_width": "%.2f" % P.line_width,
    }

    md = []
    md.append("# Recommended Bambu Studio settings - %s" % P.printer)
    md.append("")
    md.append("Material assumed: **%s** (nozzle %d C / bed %d C)."
              % (P.material, ntemp, btemp))
    md.append("")
    md.append("| Setting | Value | Why |")
    md.append("|---|---|---|")
    md.append("| Layer height | %.2f mm | %s |" % (layer, why_layer))
    md.append("| Wall loops | %d | %s |" %
              (walls, "extra stiffness for size/bridges" if walls == 3
               else "standard"))
    md.append("| Infill | %d%% %s | %s |" %
              (infill, over["sparse_infill_pattern"],
               "tall model - gyroid resists wobble" if slender
               else "standard strength/weight"))
    md.append("| Supports in slicer | **OFF** | this add-on already generated "
              "support geometry - leave Bambu Studio supports disabled |")
    md.append("| Brim | %s | %s |" %
              ("outer, 5 mm" if slender else "none",
               "slender model needs plate adhesion" if slender
               else "footprint is large enough"))
    if A.bridges:
        md.append("| Bridge speed | %d mm/s | %d bridge(s) detected, longest "
                  "%.1f mm - slow + 100%% fan keeps them clean |"
                  % (bridge_speed, len(A.bridges), longest_bridge))
    md.append("| Overhang fan | %d%% | curved overhangs near the threshold "
              "print cleaner with max part cooling |" % fan)
    md.append("")
    md.append("## How to print this")
    md.append("")
    md.append("1. Export with the add-on's **Export STLs** button - you get "
              "`..._model.stl` and `..._supports.stl`.")
    md.append("2. Drag BOTH files into Bambu Studio together and answer "
              "**Yes** to \"load these as a single object with multiple "
              "parts\". They are already in the right relative position - "
              "do not move them separately.")
    md.append("3. Keep slicer supports OFF (the profile JSON does this).")
    md.append("4. After printing, the supports snap off at the thin contact "
              "tips. Pliers help on the pillar feet.")
    md.append("")
    if P.material == "PLA":
        md.append("Tip: if your AMS has Bambu Support for PLA, assign it to "
                  "the supports part for glass-smooth interfaces (optional).")
    strat_line = ", ".join("%s x%d" % (k, v) for k, v in
                           sorted(strat_counts.items()) if v)
    md.append("")
    md.append("Support strategies used: %s." % (strat_line or "none needed"))
    if A.bridges:
        md.append("Bridges left to the printer: %d (max %.1f mm)."
                  % (len(A.bridges), longest_bridge))
    for w in A.warnings:
        md.append("")
        md.append("> %s" % w)
    for n in notes:
        md.append("")
        md.append("> %s" % n)
    md_text = "\n".join(md)

    preset = {
        "type": "process",
        "from": "User",
        "is_custom_defined": "0",
        "version": "1.9.0.14",
        "name": "FDM Wizard 0.20 %s" % P.printer,
        "inherits": "0.20mm Standard %s" % inherit_sfx,
    }
    base_by_layer = {0.12: "0.12mm Fine", 0.16: "0.16mm Optimal",
                     0.20: "0.20mm Standard", 0.24: "0.24mm Draft"}
    base = base_by_layer.get(layer, "0.20mm Standard")
    preset["name"] = "FDM Wizard %s %s" % (base.split("mm")[0] + "mm",
                                           P.printer)
    preset["inherits"] = "%s %s" % (base, inherit_sfx)
    preset.update(over)

    return {"markdown": md_text, "preset": preset, "notes": notes,
            "layer_height": layer, "min_wall": min_wall}


_DOWN = Vector((0.0, 0.0, -1.0))


class _PinNode:
    __slots__ = ("p", "load", "children", "rest_z", "on_model")

    def __init__(self, p, load, children, rest_z, on_model):
        self.p = p
        self.load = load
        self.children = children
        self.rest_z = rest_z
        self.on_model = on_model


def _pins_radius(load, P: Params):
    return min(P.pin_branch_d * 0.5 * sqrt(load), P.pillar_diameter * 0.5)


def _pins_clear(bvh, a, b, r):
    d = b - a
    L = d.length
    if L < 1e-4:
        return True
    u = d / L
    side = u.cross(Vector((0.0, 0.0, 1.0)))
    if side.length < 1e-3:
        side = Vector((1.0, 0.0, 0.0))
    side.normalize()
    up = u.cross(side)
    for off in (Vector(), side * r, -side * r, up * r, -up * r):
        if bvh.ray_cast(a + off, u, L)[0] is not None:
            return False
    return True


def _pins_drop(bvh, p, r):
    best, on_model = 0.0, False
    for ox, oy in ((0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
        hit = bvh.ray_cast(Vector((p.x + ox, p.y + oy, p.z)), _DOWN)
        if hit[0] is not None and hit[0].z > best:
            best, on_model = hit[0].z, True
    return best, on_model


def _pins_downsample(points, spacing):
    if not points or spacing <= 0:
        return list(points)
    kd = KDTree(len(points))
    for i, pt in enumerate(points):
        kd.insert(pt[0], i)
    kd.balance()
    dead = [False] * len(points)
    kept = []
    for i, pt in enumerate(points):
        if dead[i]:
            continue
        kept.append(pt)
        for _co, j, _d in kd.find_range(pt[0], spacing):
            if j != i:
                dead[j] = True
    return kept


def _pins_make_tips(points, P: Params, bvh):
    tip_len = P.pin_base_length
    r_b = P.pin_branch_d * 0.5
    r_mid = (P.pin_diameter * 0.5 + r_b) * 0.5
    floor_z = P.pad_h + P.foot_h
    tips, nodes, dropped = [], [], 0
    for co, nrm, iid in points:
        top = co - nrm * P.pin_overlap
        below = bvh.ray_cast(co + _DOWN * 0.3, _DOWN)
        if below[0] is not None and co.z - below[0].z < tip_len + 1.5:
            bot = below[0] + _DOWN * P.pin_overlap
            tips.append((co, top, bot, iid, "bridge"))
            continue
        bot = None
        for L in (tip_len, tip_len * 0.6):
            for d in ((nrm + _DOWN * 0.35).normalized(), _DOWN):
                if d.z > -0.5:
                    continue
                cand = co + d * L
                if cand.z < floor_z:
                    continue
                if _pins_clear(bvh, co + d * (0.6 + r_mid), cand,
                                r_mid + P.clearance):
                    bot = cand
                    break
            if bot is not None:
                break
        if bot is None and co.z <= floor_z + tip_len:
            tips.append((co, top, Vector((co.x, co.y, P.pad_h * 0.5)), iid,
                         "pad"))
            continue
        if bot is None:
            dropped += 1
            continue
        rest_z, on_model = _pins_drop(bvh, bot, r_b + P.clearance)
        if on_model and bot.z - rest_z < 1.0:
            tips.append((co, top, Vector((bot.x, bot.y,
                                          rest_z - P.pin_overlap)),
                         iid, "bridge"))
            continue
        tips.append((co, top, bot, iid, "node"))
        nodes.append(_PinNode(bot, 1, [], rest_z, on_model))
    return tips, nodes, dropped


def _pins_route_to_bed(nodes, P: Params, bvh):
    tan_f = tan(radians(P.fork_angle))
    floor_z = P.pad_h + P.foot_h + 1.0
    dirs = [Vector((cos(a), sin(a), 0.0))
            for a in [k * math.pi / 4 for k in range(8)]]
    out = []
    for n in nodes:
        if not n.on_model or n.p.z - n.rest_z < P.rest_ratio * n.p.z:
            out.append(n)
            continue
        r = _pins_radius(n.load, P)
        best = None
        for h in (2.0, 4.0, 6.0):
            for d in dirs:
                q = n.p + d * h + _DOWN * (h / tan_f)
                if q.z < floor_z:
                    continue
                rest_z, on_model = _pins_drop(bvh, q, r + P.clearance)
                if on_model:
                    continue
                if not _pins_clear(bvh, n.p, q, r + P.clearance):
                    continue
                best = q
                break
            if best is not None:
                break
        if best is None:
            out.append(n)
        else:
            out.append(_PinNode(best, n.load, [n], 0.0, False))
    return out


def _pins_merge(nodes, P: Params, bvh):
    tan_f = tan(radians(P.fork_angle))
    floor_z = P.pad_h + P.foot_h + 1.0
    active = list(nodes)
    for _round in range(8):
        if len(active) < 2:
            break
        kd = KDTree(len(active))
        for i, n in enumerate(active):
            kd.insert(Vector((n.p.x, n.p.y, 0.0)), i)
        kd.balance()
        pairs = []
        for i, n in enumerate(active):
            for _co, j, dist in kd.find_range(
                    Vector((n.p.x, n.p.y, 0.0)), P.max_trunk_reach):
                if j > i:
                    pairs.append((dist, i, j))
        pairs.sort()
        used = set()
        made = []
        for _dist, i, j in pairs:
            if i in used or j in used:
                continue
            a, b = active[i], active[j]
            load = a.load + b.load
            if load > P.max_fork_load:
                continue
            w = a.load / load
            jx = a.p.x * w + b.p.x * (1.0 - w)
            jy = a.p.y * w + b.p.y * (1.0 - w)
            ha = math.hypot(a.p.x - jx, a.p.y - jy)
            hb = math.hypot(b.p.x - jx, b.p.y - jy)
            jz = min(a.p.z - ha / tan_f, b.p.z - hb / tan_f) - 0.3
            if jz < floor_z:
                continue
            J = Vector((jx, jy, jz))
            r = _pins_radius(load, P)
            rest_z, on_model = _pins_drop(bvh, J, r + P.clearance)
            if jz - rest_z < 2.0:
                continue
            if on_model and not (a.on_model or b.on_model):
                continue
            if not _pins_clear(bvh, a.p, J,
                               _pins_radius(a.load, P) + P.clearance):
                continue
            if not _pins_clear(bvh, b.p, J,
                               _pins_radius(b.load, P) + P.clearance):
                continue
            used.add(i)
            used.add(j)
            made.append(_PinNode(J, load, [a, b], rest_z, on_model))
        if not made:
            break
        active = [n for k, n in enumerate(active) if k not in used] + made
    return active


def _pins_build(tips, roots, P: Params, bvh):
    soup = PrimitiveSoup()
    r_pin = P.pin_diameter * 0.5
    r_b = P.pin_branch_d * 0.5
    pad_r = P.pad_d * 0.5
    floor_z = P.pad_h + P.foot_h

    def pad(x, y):
        soup.tube(Vector((x, y, 0.0)), Vector((x, y, P.pad_h)),
                  pad_r, pad_r, segments=16)

    for co, top, bot, _iid, kind in tips:
        if kind == "bridge":
            mid = (co + bot) * 0.5
            r_m = min(r_b, r_pin * 1.3)
            soup.tube(top, mid, r_pin, r_m, segments=8)
            soup.tube(mid, bot, r_m, r_pin, segments=8)
            continue
        soup.tube(top, bot, r_pin, r_b, segments=8)
        if kind == "pad":
            pad(bot.x, bot.y)
        else:
            soup.sphere(bot, r_b)

    def branches(n):
        r = _pins_radius(n.load, P)
        for c in n.children:
            soup.tube(c.p, n.p, _pins_radius(c.load, P), r, segments=8)
            branches(c)
        if n.children:
            soup.sphere(n.p, r)

    bed_posts = []
    for n in roots:
        branches(n)
        r = _pins_radius(n.load, P)
        x, y = n.p.x, n.p.y
        if n.on_model:
            knee = Vector((x, y, n.rest_z + P.pin_base_length * 0.6))
            if knee.z < n.p.z:
                soup.tube(n.p, knee, r, r, segments=8)
            else:
                knee = n.p
            soup.tube(knee, Vector((x, y, n.rest_z - P.pin_overlap)),
                      r, r_pin, segments=8)
            continue
        knee = Vector((x, y, floor_z))
        if n.p.z > knee.z:
            soup.tube(knee, n.p, r, r, segments=8)
            bed_posts.append(n)
        soup.tube(Vector((x, y, P.pad_h * 0.5)), knee,
                  min(r * 1.6, pad_r), r, segments=12)
        pad(x, y)

    feet = [Vector((n.p.x, n.p.y, 0.0)) for n in roots if not n.on_model]
    feet += [Vector((t[2].x, t[2].y, 0.0)) for t in tips if t[4] == "pad"]
    _pins_raft(soup, feet, P)
    n_braces = _pins_truss(soup, bed_posts, P, bvh, floor_z)
    return soup.to_mesh("FDMW_pins"), n_braces


def _pins_neighbour_pairs(xy, max_d, k=3):
    if len(xy) < 2:
        return []
    kd = KDTree(len(xy))
    for i, p in enumerate(xy):
        kd.insert(p, i)
    kd.balance()
    pairs = set()
    for i, p in enumerate(xy):
        for _c, j, d in kd.find_n(p, k + 1):
            if j != i and d <= max_d:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def _pins_raft(soup, feet, P: Params):
    w = P.pad_d * 0.5
    for i, j in _pins_neighbour_pairs(feet, P.raft_join, k=2):
        a, b = feet[i], feet[j]
        d = b - a
        L = d.length
        if L < P.pad_d * 0.9:
            continue
        mid = (a + b) * 0.5
        ang = math.atan2(d.y, d.x)
        mat = (Matrix.Translation(Vector((mid.x, mid.y, P.pad_h * 0.5)))
               @ Matrix.Rotation(ang, 4, 'Z')
               @ Matrix.Diagonal(Vector((L, w, P.pad_h, 1.0))))
        soup.box(mat)


def _pins_truss(soup, posts, P: Params, bvh, floor_z):
    r = P.brace_d * 0.5
    xy = [Vector((n.p.x, n.p.y, 0.0)) for n in posts]
    n_braces = 0
    for i, j in _pins_neighbour_pairs(xy, P.brace_max_span, k=2):
        a, b = posts[i].p, posts[j].p
        if (a.xy - b.xy).length < 1.5:
            continue
        top = min(a.z, b.z) - 1.0
        z0 = floor_z + 1.0
        step = max(P.brace_step, (top - z0) / P.brace_max_bays)
        while top - z0 >= P.brace_min_h * 0.5:
            z1 = min(z0 + step, top)
            for p, q in (((a.x, a.y, z0), (b.x, b.y, z1)),
                         ((b.x, b.y, z0), (a.x, a.y, z1))):
                p, q = Vector(p), Vector(q)
                if _pins_clear(bvh, p, q, r + P.clearance):
                    soup.tube(p, q, r, r, segments=6)
                    n_braces += 1
            z0 = z1
    return n_braces


def _pins_conform(sup_ob, model_ob, solver, collection):
    order = {"MANIFOLD": ["MANIFOLD", "EXACT"],
             "EXACT": ["EXACT", "MANIFOLD"],
             "FLOAT": ["FLOAT", "MANIFOLD"]}.get(solver, [solver])
    base = sup_ob.data
    n_base = len(base.polygons)
    for s in order:
        trial = base.copy()
        sup_ob.data = trial
        mod = sup_ob.modifiers.new("conform", 'BOOLEAN')
        mod.operation = 'DIFFERENCE'
        try:
            mod.solver = s
        except TypeError:
            pass
        mod.object = model_ob
        try:
            _apply_modifiers_inplace(sup_ob)
            n = len(sup_ob.data.polygons)
            if n > 0 and n != n_base:
                if base.users == 0:
                    bpy.data.meshes.remove(base)
                return True
        except Exception:
            pass
        if sup_ob.data is not base:
            dead = sup_ob.data
            sup_ob.data = base
            if dead.users == 0:
                bpy.data.meshes.remove(dead)
        sup_ob.modifiers.clear()
    return False


def _pins_validate(sup_me, A: Analysis, covered, n_dropped, P: Params):
    need = [i for i in A.islands
            if not i.sealed and i.samples and i.area >= 1.0]
    missing = [i for i in need if i.idx not in covered]
    V = {
        "geometry": {"pass": len(sup_me.polygons) > 0,
                     "verts": len(sup_me.vertices),
                     "faces": len(sup_me.polygons)},
        "coverage": {"pass": not missing,
                     "islands_needing_support": len(need),
                     "islands_covered": len(need) - len(missing),
                     "uncovered_z_mm": [[round(i.z_min, 1), round(i.z_max, 1)]
                                        for i in missing][:12],
                     "points_dropped": n_dropped},
    }
    V["all_pass"] = all(d.get("pass", True) for d in V.values()
                        if isinstance(d, dict))
    return V


def _pins_local_minima(A: Analysis, P: Params):
    import numpy as np
    me = A.me
    nv = len(me.vertices)
    if nv == 0:
        return []
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    z = co[:, 2]
    nb_min = np.full(nv, np.inf)
    np.minimum.at(nb_min, ev[:, 0], z[ev[:, 1]])
    np.minimum.at(nb_min, ev[:, 1], z[ev[:, 0]])
    cand = np.nonzero((z < nb_min) & (z > P.first_layer_z + 0.5))[0]
    if len(cand) == 0:
        return []
    kd = KDTree(nv)
    for i in range(nv):
        kd.insert(co[i], i)
    kd.balance()
    out = []
    for i in cand[np.argsort(z[cand])]:
        p = Vector(co[i])
        if any(z[j] < z[i] - 1e-4
               for _c, j, _d in kd.find_range(p, P.minima_radius)):
            continue
        if point_inside_solid(A.bvh, p + Vector((0.0, 0.0, -0.3))):
            continue
        out.append((p, Vector(me.vertices[int(i)].normal), -1))
    return out


def run_pins_pipeline(obj, P, A, collection, report, dropped):
    pts = []
    n_sealed = 0
    for isl in A.islands:
        if isl.sealed:
            n_sealed += 1
            continue
        pts.extend((co, nrm, isl.idx) for co, nrm in isl.samples)
    if n_sealed:
        A.warnings.append(
            "%d sealed internal region(s) skipped - a support there could "
            "never be removed." % n_sealed)

    minima = _pins_local_minima(A, P)
    pts = _pins_downsample(minima + pts, P.pin_spacing)
    tips, nodes, n_dropped = _pins_make_tips(pts, P, A.bvh)
    if not tips:
        return None, {"PINS": 0}, {
            "all_pass": not pts,
            "geometry": {"pass": True, "note": "no supports built"},
            "coverage": {"pass": not pts, "points_dropped": n_dropped}}

    nodes = _pins_route_to_bed(nodes, P, A.bvh)
    roots = _pins_merge(nodes, P, A.bvh)
    sup_me, n_braces = _pins_build(tips, roots, P, A.bvh)
    sup_me.name = "SUP_" + obj.name
    n_posts = (sum(1 for n in roots if not n.on_model)
               + sum(1 for t in tips if t[4] == "pad"))
    n_rest = (sum(1 for n in roots if n.on_model)
              + sum(1 for t in tips if t[4] == "bridge"))
    strat_counts = {"PINS": len(tips)}

    conformed = False
    if P.conform != "NONE":
        tmp_sup = _new_object("FDMW_pins_tmp", sup_me, collection)
        model_ob = _new_object("FDMW_model_tmp", extract_world_mesh(obj),
                               collection)
        try:
            conformed = _pins_conform(tmp_sup, model_ob, P.conform,
                                      collection)
        finally:
            sup_me = tmp_sup.data
            sup_me.name = "SUP_" + obj.name
            bpy.data.objects.remove(tmp_sup)
            _remove_object(model_ob)
        if not conformed:
            A.warnings.append(
                "Boolean trim failed (rough mesh); tips left embedded "
                "%.1f mm - the slicer trims them at print time."
                % P.pin_overlap)

    covered = {t[3] for t in tips}
    kd = KDTree(len(tips))
    for k, t in enumerate(tips):
        kd.insert(t[0], k)
    kd.balance()
    reach = max(P.pin_spacing, 1e-3) * 1.2
    for isl in A.islands:
        if isl.idx not in covered and any(
                kd.find(co)[2] <= reach for co, _n in isl.samples):
            covered.add(isl.idx)
    val = _pins_validate(sup_me, A, covered, n_dropped, P)
    if n_dropped:
        A.warnings.append(
            "%d support point(s) had no clear path for a tip (tight gaps)."
            % n_dropped)
    report["attempts"].append({
        "attempt": 0, "style": "PINS",
        "tips": len(tips), "bed_posts": n_posts, "model_rests": n_rest,
        "braces": n_braces, "dropped": n_dropped,
        "conform": P.conform if conformed else "NONE",
        "validation_pass": val["all_pass"],
        "failing": [k for k, v in val.items()
                    if isinstance(v, dict) and not v.get("pass", True)]})
    report["messages"].append(
        "%d tip(s) on %d bed post(s) + %d model rest(s), %d brace(s)%s."
        % (len(tips), n_posts, n_rest, n_braces,
           " (boolean-trimmed)" if conformed else ""))
    return sup_me, strat_counts, val


class _ClearGrid:
    __slots__ = ("x0", "y0", "z0", "c", "h", "shape", "forbid", "R",
                 "pad_forbid")

    def cell(self, p):
        i = int(math.floor((p.x - self.x0) / self.c))
        j = int(math.floor((p.y - self.y0) / self.c))
        k = int(math.floor((p.z - self.z0) / self.h))
        nx, ny, nz = self.shape
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            return (i, j, k)
        return None

    def center(self, cell):
        i, j, k = cell
        return Vector((self.x0 + (i + 0.5) * self.c,
                       self.y0 + (j + 0.5) * self.c,
                       self.z0 + k * self.h))

    def blocked(self, p):
        cl = self.cell(p)
        return cl is None or bool(self.forbid[cl])


def _clear_surface_points(me, step):
    import numpy as np
    me.calc_loop_triangles()
    nv = len(me.vertices)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nt = len(me.loop_triangles)
    if nt == 0:
        return co
    tv = np.empty(nt * 3, dtype=np.int64)
    me.loop_triangles.foreach_get("vertices", tv)
    T = co[tv.reshape(-1, 3)]
    e = np.max(np.stack([np.linalg.norm(T[:, 0] - T[:, 1], axis=1),
                         np.linalg.norm(T[:, 1] - T[:, 2], axis=1),
                         np.linalg.norm(T[:, 2] - T[:, 0], axis=1)]), axis=0)
    n = np.clip(np.ceil(e / step), 1, 96).astype(np.int64)
    out = [co]
    for m in np.unique(n):
        Ts = T[n == m]
        W = np.array([(i / m, j / m, 1.0 - (i + j) / m)
                      for i in range(m + 1) for j in range(m + 1 - i)])
        per = max(1, 4000000 // len(W))
        for s in range(0, len(Ts), per):
            out.append(np.einsum('kw,twd->tkd', W, Ts[s:s + per])
                       .reshape(-1, 3))
    return np.concatenate(out)


def _dilate_xy_step(a, diag):
    b = a.copy()
    b[1:] |= a[:-1]
    b[:-1] |= a[1:]
    b[:, 1:] |= a[:, :-1]
    b[:, :-1] |= a[:, 1:]
    if diag:
        b[1:, 1:] |= a[:-1, :-1]
        b[:-1, :-1] |= a[1:, 1:]
        b[1:, :-1] |= a[:-1, 1:]
        b[:-1, 1:] |= a[1:, :-1]
    return b


def _dilate_xy(a, k):
    for r in range(k):
        a = _dilate_xy_step(a, r % 2 == 0)
    return a


def _dilate_ball(surf, D, c, h):
    import numpy as np
    kz = int(math.ceil((D + 0.5 * h) / h))
    need = {}
    for s in range(0, kz + 1):
        dz = max(0.0, (s - 0.5) * h)
        rr = sqrt(max(D * D - dz * dz, 0.0))
        need.setdefault(int(math.ceil((rr + 0.71 * c) / c)), []).append(s)
    out = np.zeros_like(surf)
    cur = surf.copy()
    have = 0
    for r in sorted(need):
        while have < r:
            cur = _dilate_xy_step(cur, have % 2 == 0)
            have += 1
        for s in need[r]:
            if s == 0:
                out |= cur
            else:
                out[:, :, s:] |= cur[:, :, :-s]
                out[:, :, :-s] |= cur[:, :, s:]
    return out


def _clear_build_grid(A: Analysis, P: Params):
    import numpy as np
    D = P.pin_branch_d * 0.5 + P.keepout
    pad_r = P.pad_d * 0.5
    floor_z = P.pad_h + P.foot_h
    tan_f = tan(radians(P.fork_angle))
    margin = max(10.0, pad_r + D + 2.0)
    mn, mx = A.model_min, A.model_max
    c = P.route_cell
    while True:
        h = c / tan_f
        nx = int((mx[0] - mn[0] + 2 * margin) / c) + 1
        ny = int((mx[1] - mn[1] + 2 * margin) / c) + 1
        nz = int((mx[2] + D - floor_z) / h) + 2
        if nx * ny * nz <= P.route_max_cells:
            break
        c *= 1.25
    G = _ClearGrid()
    G.x0, G.y0, G.z0, G.c, G.h = mn[0] - margin, mn[1] - margin, floor_z, c, h
    G.shape = (nx, ny, nz)

    pts = _clear_surface_points(A.me, min(c, h) * 0.5)
    ii = np.clip(np.floor((pts[:, 0] - G.x0) / c), 0, nx - 1).astype(np.int64)
    jj = np.clip(np.floor((pts[:, 1] - G.y0) / c), 0, ny - 1).astype(np.int64)
    kk = np.clip(np.floor((pts[:, 2] - G.z0) / h + 0.5),
                 0, nz - 1).astype(np.int64)
    surf = np.zeros(G.shape, dtype=bool)
    surf[ii, jj, kk] = True
    G.forbid = _dilate_ball(surf, D, c, h)

    low = pts[:, 2] < floor_z + P.keepout
    pad2d = np.zeros((nx, ny), dtype=bool)
    pad2d[ii[low], jj[low]] = True
    G.pad_forbid = _dilate_xy(pad2d[:, :, None],
                              int(math.ceil((pad_r + P.keepout + 0.71 * c)
                                            / c)))[:, :, 0]

    R = np.zeros(G.shape, dtype=bool)
    R[:, :, 0] = ~G.forbid[:, :, 0] & ~G.pad_forbid
    for k in range(1, nz):
        prev = R[:, :, k - 1]
        d = prev.copy()
        d[1:] |= prev[:-1]
        d[:-1] |= prev[1:]
        d[:, 1:] |= prev[:, :-1]
        d[:, :-1] |= prev[:, 1:]
        R[:, :, k] = d & ~G.forbid[:, :, k]
    G.R = R
    return G


def _clear_cone_ok(bvh, co, d, L, r_tip, r_base):
    t = 1.2
    while t <= L + 1e-6:
        p = co + d * t
        r = r_tip + (r_base - r_tip) * min(t / L, 1.0)
        hit = bvh.find_nearest(p)
        if hit[0] is not None and hit[3] < r + 0.3:
            return False
        t += 0.4
    return True


_TIP_SWEEP = [Vector((sin(radians(a)) * cos(radians(b)),
                      sin(radians(a)) * sin(radians(b)),
                      -cos(radians(a))))
              for a in (20.0, 40.0, 55.0) for b in range(0, 360, 45)]


def _clear_tips(points, P: Params, A: Analysis, G: _ClearGrid):
    r_pin, r_b = P.pin_diameter * 0.5, P.pin_branch_d * 0.5
    floor_z = G.z0
    tips, unreachable = [], []
    near_bed = 0
    for co, nrm, iid in points:
        top = co - nrm * P.pin_overlap
        dirs = [(nrm + _DOWN * 0.35).normalized(), _DOWN,
                (nrm + _DOWN).normalized()] + _TIP_SWEEP
        found = None
        for L in (P.pin_base_length, P.pin_base_length * 1.5,
                  P.pin_base_length * 2.0, P.pin_base_length * 2.7):
            for d in dirs:
                if d.z > -0.5:
                    continue
                node = co + d * L
                if node.z < floor_z:
                    continue
                cl = G.cell(node)
                if cl is None or not G.R[cl]:
                    continue
                if not _clear_cone_ok(A.bvh, co, d, L, r_pin, r_b):
                    continue
                found = dict(co=co, top=top, node=node, cell=cl,
                             kind="route", iid=iid, L=L)
                break
            if found:
                break
        if found is None and co.z < P.pad_h + 1.0:
            near_bed += 1
            continue
        if found is None and co.z <= floor_z + 3.0 * P.pin_base_length:
            foot_r = r_b * 1.6
            for d in [(nrm + _DOWN * 0.35).normalized(),
                      (nrm + _DOWN).normalized(), _DOWN] + _TIP_SWEEP:
                if d.z > -0.5:
                    continue
                L = (co.z - P.pad_h) / -d.z
                if L > 3.0 * P.pin_base_length:
                    continue
                end = co + d * L
                hit = A.bvh.find_nearest(Vector((end.x, end.y, P.pad_h)))
                if hit[0] is not None and hit[3] < foot_r + P.keepout:
                    continue
                if not _clear_cone_ok(A.bvh, co, d, L, r_pin, r_b):
                    continue
                found = dict(co=co, top=top, node=end, cell=None,
                             kind="foot", iid=iid, L=L)
                break
        if found is None:
            unreachable.append((co, iid))
        else:
            tips.append(found)
    return tips, unreachable, near_bed


def _clear_route(tips, G: _ClearGrid, merge_reach=4.0):
    nx, ny, _nz = G.shape
    parent, load = {}, {}
    used = {}
    reach = max(1, int(round(merge_reach / G.c)))
    for t in sorted((t for t in tips if t["kind"] == "route"),
                    key=lambda t: -t["node"].z):
        c = t["cell"]
        last_step = (0, 0)
        while c not in parent:
            i, j, k = c
            if k == 0:
                parent[c] = None
                break
            opts = [(0, 0), last_step, (1, 0), (-1, 0), (0, 1), (0, -1)]
            cand = []
            for di, dj in opts:
                a, b = i + di, j + dj
                q = (a, b, k - 1)
                if (0 <= a < nx and 0 <= b < ny and G.R[q]
                        and q not in cand):
                    cand.append(q)
            joined = [q for q in cand if q in parent]
            if joined:
                q = joined[0]
            else:
                q = cand[0]
                row = used.get(k - 1)
                if row:
                    best = None
                    for di in range(-reach, reach + 1):
                        for dj in range(-reach, reach + 1):
                            if (i + di, j + dj) in row:
                                d2 = di * di + dj * dj
                                if best is None or d2 < best[0]:
                                    best = (d2, i + di, j + dj)
                    if best is not None:
                        q = min(cand, key=lambda x: (
                            (x[0] - best[1]) ** 2 + (x[1] - best[2]) ** 2))
            last_step = (q[0] - i, q[1] - j)
            parent[c] = q
            used.setdefault(k, set()).add((i, j))
            c = q
        c = t["cell"]
        while c is not None:
            load[c] = load.get(c, 0) + 1
            c = parent[c]
    return parent, load


def _clear_segment_free(G, a, b, step=0.4):
    L = (b - a).length
    n = max(1, int(L / step))
    return not any(G.blocked(a + (b - a) * (s / n)) for s in range(n + 1))


def _clear_build(tips, parent, load, G: _ClearGrid, P: Params):
    soup = PrimitiveSoup()
    r_pin, r_b = P.pin_diameter * 0.5, P.pin_branch_d * 0.5
    pad_r = P.pad_d * 0.5
    floor_z = G.z0

    def rad(cell):
        return _pins_radius(load.get(cell, 1), P)

    def pad(x, y):
        soup.tube(Vector((x, y, 0.0)), Vector((x, y, P.pad_h)),
                  pad_r, pad_r, segments=16)

    feet = []
    starts = set()
    for t in tips:
        soup.tube(t["top"], t["node"], r_pin, r_b, segments=8)
        if t["kind"] == "foot":
            x, y = t["node"].x, t["node"].y
            soup.tube(Vector((x, y, 0.0)), Vector((x, y, P.pad_h)),
                      r_b * 1.6, r_b * 1.6, segments=12)
            continue
        starts.add(t["cell"])
        soup.sphere(t["node"], r_b)
        soup.tube(t["node"], G.center(t["cell"]), r_b, rad(t["cell"]),
                  segments=8)

    children = {}
    for c, p in parent.items():
        if p is not None:
            children.setdefault(p, []).append(c)

    def step(c):
        p = parent[c]
        return None if p is None else (p[0] - c[0], p[1] - c[1])

    keys = set(starts)
    for c, p in parent.items():
        if p is None or len(children.get(c, [])) != 1:
            keys.add(c)
        if p is not None and parent[p] is not None and step(c) != step(p):
            keys.add(p)

    for c in keys:
        q = parent[c]
        if q is None:
            continue
        while q not in keys:
            q = parent[q]
        r = rad(c)
        soup.tube(G.center(c), G.center(q), r, r, segments=8)
        if parent[q] is not None:
            soup.sphere(G.center(q), rad(q))

    roots = [c for c, p in parent.items() if p is None]
    posts = []
    for c in roots:
        cen = G.center(c)
        r = rad(c)
        soup.tube(Vector((cen.x, cen.y, P.pad_h * 0.5)), cen,
                  min(r * 1.6, pad_r), r, segments=12)
        pad(cen.x, cen.y)
        feet.append(Vector((cen.x, cen.y, 0.0)))
        top = c
        while len(children.get(top, [])) == 1:
            ch = children[top][0]
            if (ch[0], ch[1]) != (top[0], top[1]):
                break
            top = ch
        posts.append((cen, G.center(top).z))

    w = P.pad_d * 0.5
    for i, j in _pins_neighbour_pairs(feet, P.raft_join, k=2):
        a, b = feet[i], feet[j]
        d = b - a
        L = d.length
        if L < P.pad_d * 0.9:
            continue
        n = max(1, int(L / (G.c * 0.5)))
        ok = True
        for s in range(n + 1):
            p = a + d * (s / n)
            ci = int(math.floor((p.x - G.x0) / G.c))
            cj = int(math.floor((p.y - G.y0) / G.c))
            if not (0 <= ci < G.shape[0] and 0 <= cj < G.shape[1]) \
                    or G.pad_forbid[ci, cj]:
                ok = False
                break
        if not ok:
            continue
        mid = (a + b) * 0.5
        mat = (Matrix.Translation(Vector((mid.x, mid.y, P.pad_h * 0.5)))
               @ Matrix.Rotation(math.atan2(d.y, d.x), 4, 'Z')
               @ Matrix.Diagonal(Vector((L, w, P.pad_h, 1.0))))
        soup.box(mat)

    rb = P.brace_d * 0.5
    n_braces = 0
    xy = [Vector((p[0].x, p[0].y, 0.0)) for p in posts]
    for i, j in _pins_neighbour_pairs(xy, P.brace_max_span, k=2):
        (a, ta), (b, tb) = posts[i], posts[j]
        if (a.xy - b.xy).length < 1.5:
            continue
        top = min(ta, tb) - 1.0
        z0 = floor_z + 1.0
        stp = max(P.brace_step, (top - z0) / P.brace_max_bays)
        while top - z0 >= P.brace_min_h * 0.5:
            z1 = min(z0 + stp, top)
            for p, q in (((a.x, a.y, z0), (b.x, b.y, z1)),
                         ((b.x, b.y, z0), (a.x, a.y, z1))):
                p, q = Vector(p), Vector(q)
                if _clear_segment_free(G, p, q):
                    soup.tube(p, q, rb, rb, segments=6)
                    n_braces += 1
            z0 = z1
    n_posts = len(roots) + sum(1 for t in tips if t["kind"] == "foot")
    return soup.to_mesh("FDMW_clear"), n_posts, n_braces


def _clear_keepout(sup_me, obj, tips, P: Params):
    import numpy as np
    model_me = extract_world_mesh(obj)
    bvh = build_bvh(model_me)
    try:
        nv = len(sup_me.vertices)
        co = np.empty(nv * 3)
        sup_me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        ev = np.empty(len(sup_me.edges) * 2, dtype=np.int64)
        sup_me.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        pts = [co]
        a, b = co[ev[:, 0]], co[ev[:, 1]]
        L = np.linalg.norm(b - a, axis=1)
        n = np.ceil(L / 0.6).astype(np.int64)
        for m in np.unique(n[n > 1]):
            s = n == m
            t = (np.arange(1, m) / m)[None, :, None]
            pts.append((a[s][:, None] + (b[s] - a[s])[:, None] * t)
                       .reshape(-1, 3))
        pts = np.concatenate(pts)
        kd = KDTree(len(tips))
        for k, t in enumerate(tips):
            kd.insert(t["co"], k)
        kd.balance()
        excl = max((t["L"] for t in tips), default=0.0) \
            + P.pin_branch_d + 0.6
        gap_min, bad = 1e9, 0
        for p in pts:
            v = Vector(p)
            if tips:
                _c, k, dist = kd.find(v)
                if dist <= tips[k]["L"] + P.pin_branch_d + 0.6:
                    continue
                if any(dd <= tips[kk]["L"] + P.pin_branch_d + 0.6
                       for _cc, kk, dd in kd.find_range(v, excl)):
                    continue
            hit = bvh.find_nearest(v)
            if hit[0] is None:
                continue
            g = hit[3]
            if g < P.keepout and point_inside_solid(bvh, v):
                g = -g
            gap_min = min(gap_min, g)
            if g < P.keepout * 0.5:
                bad += 1
        return bad, (round(gap_min, 3) if gap_min < 1e8 else None)
    finally:
        bpy.data.meshes.remove(model_me)


def run_clear_pipeline(obj, P, A, collection, report, dropped):
    pts = []
    n_sealed = 0
    for isl in A.islands:
        if isl.sealed:
            n_sealed += 1
            continue
        pts.extend((co, nrm, isl.idx) for co, nrm in isl.samples)
    if n_sealed:
        A.warnings.append(
            "%d sealed internal region(s) skipped - a support there could "
            "never be removed." % n_sealed)
    minima = _pins_local_minima(A, P)
    pts = _pins_downsample(minima + pts, P.pin_spacing)

    t0 = time.time()
    G = _clear_build_grid(A, P)
    tips, unreachable, near_bed = _clear_tips(pts, P, A, G)
    if not tips:
        report["attempts"].append({"attempt": 0, "style": "CLEAR",
                                   "tips": 0, "bed_posts": 0,
                                   "model_rests": 0,
                                   "unreachable": len(unreachable)})
        return None, {"CLEAR": 0}, {
            "all_pass": not pts,
            "geometry": {"pass": True, "note": "no supports built"},
            "coverage": {"pass": not pts,
                         "unreachable_points": len(unreachable)}}
    parent, load = _clear_route(tips, G, P.merge_reach)
    sup_me, n_posts, n_braces = _clear_build(tips, parent, load, G, P)
    sup_me.name = "SUP_" + obj.name
    t_route = time.time() - t0

    conformed = False
    if P.conform != "NONE":
        tmp_sup = _new_object("FDMW_clear_tmp", sup_me, collection)
        model_ob = _new_object("FDMW_model_tmp", extract_world_mesh(obj),
                               collection)
        try:
            conformed = _pins_conform(tmp_sup, model_ob, P.conform,
                                      collection)
        finally:
            sup_me = tmp_sup.data
            sup_me.name = "SUP_" + obj.name
            bpy.data.objects.remove(tmp_sup)
            _remove_object(model_ob)
        if not conformed:
            A.warnings.append(
                "Boolean trim failed (rough mesh); tips left embedded "
                "%.1f mm - the slicer trims them at print time."
                % P.pin_overlap)

    bad, gap_min = _clear_keepout(sup_me, obj, tips, P)

    covered = {t["iid"] for t in tips}
    kd = KDTree(len(tips))
    for k, t in enumerate(tips):
        kd.insert(t["co"], k)
    kd.balance()
    reach = max(P.pin_spacing, 1e-3) * 1.2
    for isl in A.islands:
        if isl.idx not in covered and any(
                kd.find(co)[2] <= reach for co, _n in isl.samples):
            covered.add(isl.idx)
    val = _pins_validate(sup_me, A, covered, len(unreachable), P)
    val["keepout"] = {"pass": bad == 0, "violations": bad,
                      "min_gap_mm": gap_min, "keepout_mm": P.keepout}
    val["all_pass"] = all(d.get("pass", True) for d in val.values()
                          if isinstance(d, dict))
    if unreachable:
        zs = sorted(round(co.z, 1) for co, _i in unreachable)
        A.warnings.append(
            "%d support point(s) have no path to the bed that keeps %.1f mm "
            "clear of the model (deep pockets / under-cavities, z %.1f-%.1f "
            "mm). They are left unsupported rather than resting on the "
            "model." % (len(unreachable), P.keepout, zs[0], zs[-1]))
    report["attempts"].append({
        "attempt": 0, "style": "CLEAR", "tips": len(tips),
        "bed_posts": n_posts, "model_rests": 0, "braces": n_braces,
        "unreachable": len(unreachable), "near_bed_skipped": near_bed,
        "unreachable_xyz": [[round(v, 1) for v in co]
                            for co, _i in unreachable[:200]],
        "grid_cell_mm": round(G.c, 2), "route_seconds": round(t_route, 1),
        "conform": P.conform if conformed else "NONE",
        "validation_pass": val["all_pass"],
        "failing": [k for k, v in val.items()
                    if isinstance(v, dict) and not v.get("pass", True)]})
    report["messages"].append(
        "%d tip(s) on %d bed post(s), %d brace(s); nothing else touches the "
        "model (min gap %s mm)%s." % (
            len(tips), n_posts, n_braces, gap_min,
            " (tips boolean-trimmed)" if conformed else ""))
    return sup_me, {"CLEAR": len(tips)}, val


def run_pipeline(obj, P: Params, collection=None):
    t0 = time.time()
    if obj is None or obj.type != 'MESH':
        raise RuntimeError("Select a mesh object first")
    if collection is None:
        collection = _get_or_create_support_collection(bpy.context.scene)

    report = {"object": obj.name, "messages": [], "attempts": []}

    old_name = "SUP_" + obj.name
    for ob in list(bpy.data.objects):
        if ob.name == old_name or ob.name.startswith(old_name + "."):
            _remove_object(ob)

    me_chk = extract_world_mesh(obj)
    if not me_chk.vertices:
        bpy.data.meshes.remove(me_chk)
        raise RuntimeError("Object has no geometry")
    xs = [v.co.x for v in me_chk.vertices]
    ys = [v.co.y for v in me_chk.vertices]
    zs = [v.co.z for v in me_chk.vertices]
    dims = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    bpy.data.meshes.remove(me_chk)
    bx, by, bz, _sfx = PRINTERS.get(P.printer, PRINTERS["X1C"])
    over = max(dims[0] / bx, dims[1] / by, dims[2] / bz)
    if over > 1.02:
        fit = 1.0 / over
        raise RuntimeError(
            "Model is %.0f x %.0f x %.0f mm but the %s bed is "
            "%.0f x %.0f x %.0f mm. Scale the model to about %.2fx "
            "first (S, %.2f, Enter - then Object > Apply > Scale), "
            "then generate supports."
            % (dims[0], dims[1], dims[2], P.printer, bx, by, bz,
               fit * 0.97, fit * 0.97))

    _fix_inside_out(obj, report["messages"])
    dropped = drop_to_plate(obj, report["messages"])

    analysis_obj = obj
    proxy = None
    me_cnt = _evaluated_mesh_of(obj)
    me_cnt.calc_loop_triangles()
    n_tris = len(me_cnt.loop_triangles)
    bpy.data.meshes.remove(me_cnt)
    if n_tris > 250000:
        proxy_me = _evaluated_mesh_of(obj)
        proxy = _new_object("FDMW_analysis_proxy", proxy_me, collection)
        proxy.matrix_world = obj.matrix_world.copy()
        dec = proxy.modifiers.new("dec", 'DECIMATE')
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = 250000.0 / n_tris
        _apply_modifiers_inplace(proxy)
        analysis_obj = proxy
        report["messages"].append(
            "Dense mesh (%d triangles): analysis ran on a simplified "
            "copy (~250k triangles). Support contact accuracy is within "
            "the decimation error (typically well under 0.2 mm)."
            % n_tris)

    P_try = Params(**vars(P))
    last = None
    for attempt in range(P.max_retries + 1):
        A = analyze(analysis_obj, P_try)
        A.dropped_by = dropped
        P_try.step_dz = max(P_try.step_dz, A.model_h / 150.0)
        P_try.voxel_size = min(max(P_try.voxel_size, A.model_h / 600.0),
                               P_try.min_feature / 2.5)
        classify(A, P_try)

        if P_try.style in ("PINS", "CLEAR"):
            run = (run_clear_pipeline if P_try.style == "CLEAR"
                   else run_pins_pipeline)
            sup_me, strat_counts, val = run(
                obj, P_try, A, collection, report, dropped)
            last = (A, sup_me, strat_counts, [], val)
            break

        soup = PrimitiveSoup()
        contacts = []
        extra_meshes = []
        strat_counts = {}
        all_samples = []
        tree_tips = []
        cavity_tips = []

        for isl in A.islands:
            strat_counts[isl.strategy] = strat_counts.get(isl.strategy, 0) + 1
            metas = []
            for co, nrm in isl.samples:
                m = {"co": co, "rad": isl.coverage_radius,
                     "strat": isl.strategy}
                all_samples.append(m)
                metas.append(m)
            if isl.strategy == STRAT_TREE:
                tree_tips.extend((co, nrm, m) for (co, nrm), m
                                 in zip(isl.samples, metas))
            elif isl.strategy == STRAT_CAVITY:
                cavity_tips.extend((co, nrm, m) for (co, nrm), m
                                   in zip(isl.samples, metas))
            elif isl.strategy == STRAT_PILLAR:
                gen_pillars(soup, isl, A, P_try, contacts)
            elif isl.strategy == STRAT_CRADLE:
                cr, leftover = gen_cradle(isl, metas, A, P_try, collection,
                                          contacts)
                if cr is None:
                    isl.strategy = STRAT_PILLAR
                    isl.note = "cradle boolean unreliable here - used pillars"
                    strat_counts[STRAT_CRADLE] -= 1
                    strat_counts[STRAT_PILLAR] = \
                        strat_counts.get(STRAT_PILLAR, 0) + 1
                    isl.spacing = P_try.pillar_spacing
                    isl.coverage_radius = P_try.pillar_spacing * 0.75
                    gen_pillars(soup, isl, A, P_try, contacts)
                else:
                    extra_meshes.append(cr)
                    tree_tips.extend(leftover)

        if tree_tips:
            grow_forest(soup, tree_tips, A, P_try,
                        tip_r=P_try.tip_contact_d * 0.5,
                        branch_r=P_try.branch_d * 0.5,
                        cap_r=P_try.branch_max_d * 0.5,
                        contacts=contacts)
        if cavity_tips:
            grow_forest(soup, cavity_tips, A, P_try,
                        tip_r=P_try.cavity_tip_d * 0.5,
                        branch_r=P_try.cavity_branch_d * 0.5,
                        cap_r=P_try.cavity_branch_d * 0.5 + 0.4,
                        contacts=contacts)

        n_tight = sum(1 for s in all_samples if s["strat"] == "TIGHT")
        if n_tight:
            A.warnings.append(
                "%d support point(s) sit in gaps too tight for any support "
                "to fit - left to bridging/sag (slicer supports could not "
                "fit there either)." % n_tight)

        main_me = finalize_soup(soup, P_try, collection, "FDMW_supports")
        parts = ([main_me] if main_me else []) + extra_meshes
        sup_me = None
        if parts:
            sup_me = merge_meshes(parts, "SUP_" + obj.name) \
                if len(parts) > 1 else parts[0]
            if len(parts) > 1:
                for m in parts:
                    if m.users == 0:
                        bpy.data.meshes.remove(m)
            sup_me.name = "SUP_" + obj.name

        if sup_me is None:
            val = {"all_pass": True,
                   "manifold": {"pass": True, "note": "no geometry needed"},
                   "coverage": {"pass": True, "needed": 0, "covered": 0,
                                "fraction": 1.0}}
            report["attempts"].append(
                {"attempt": attempt, "params": {"z_gap": P_try.z_gap},
                 "validation_pass": True, "no_supports_needed": True})
            last = (A, None, strat_counts, all_samples, val)
            break

        val = validate(sup_me, A, all_samples, P_try)
        carved = False
        if (not val["all_pass"]
                and not val.get("model_collision", {}).get("pass", True)
                and val["model_collision"].get("overlap_pairs", 0) <= 400):
            new_me = _carve_collisions(sup_me, A, P_try, collection)
            if new_me is not sup_me:
                if sup_me.users == 0:
                    bpy.data.meshes.remove(sup_me)
                sup_me = new_me
                sup_me.name = "SUP_" + obj.name
                carved = True
                val = validate(sup_me, A, all_samples, P_try)
        report["attempts"].append({
            "attempt": attempt,
            "params": {"z_gap": round(P_try.z_gap, 2),
                       "voxel": round(P_try.voxel_size, 2)},
            "carve_repair": carved,
            "validation_pass": val["all_pass"],
            "failing": [k for k, v in val.items()
                        if isinstance(v, dict) and not v.get("pass", True)],
        })
        last = (A, sup_me, strat_counts, all_samples, val)
        if val["all_pass"]:
            break

        if attempt < P.max_retries:
            bpy.data.meshes.remove(sup_me)
            if not val.get("model_collision", {}).get("pass", True) or \
               not val.get("contact_gap", {}).get("pass", True):
                P_try.z_gap += 0.10
                P_try.cradle_gap += 0.10
                P_try.xy_clearance += 0.2
                if attempt >= 1:
                    P_try.strategy_override = "TREE"
            if not val.get("min_thickness", {}).get("pass", True):
                P_try.voxel_size = max(0.24, P_try.voxel_size - 0.06)
                P_try.tip_contact_d = max(P_try.tip_contact_d, 1.3)
            if not val.get("coverage", {}).get("pass", True):
                P_try.base_spacing = max(2.0, P_try.base_spacing * 0.85)
            if not val.get("manifold", {}).get("pass", True):
                P_try.voxel_size = min(0.42, P_try.voxel_size + 0.05)
            bpy.data.meshes.remove(A.me)

    A, sup_me, strat_counts, all_samples, val = last
    sup_ob = None
    if sup_me is not None:
        sup_ob = _new_object("SUP_" + obj.name, sup_me, collection)
        sup_ob.color = (0.2, 0.9, 0.3, 1.0)
        mat = bpy.data.materials.get("FDMW_SupportGreen")
        if mat is None:
            mat = bpy.data.materials.new("FDMW_SupportGreen")
            mat.diffuse_color = (0.2, 0.9, 0.3, 0.9)
        sup_ob.data.materials.append(mat)

    prof = recommend_profile(A, P_try, strat_counts, val)

    report.update({
        "model_dims_mm": [round(d, 1) for d in A.model_dims],
        "faces": A.n_faces,
        "islands": len(A.islands),
        "overhang_area_mm2": round(A.total_overhang_area, 1),
        "strategies": strat_counts,
        "island_detail": [
            {"id": i.idx, "strategy": i.strategy,
             "area_mm2": round(i.area, 1),
             "z": [round(i.z_min, 1), round(i.z_max, 1)],
             "samples": len(i.samples),
             "plate_frac": round(i.plate_frac, 2),
             "note": i.note}
            for i in A.islands],
        "bridges_mm": [round(b, 1) for b in A.bridges],
        "warnings": A.warnings,
        "validation": val,
        "profile": prof,
        "support_object": sup_ob.name if sup_ob else None,
        "seconds": round(time.time() - t0, 1),
    })

    bpy.data.meshes.remove(A.me)
    if proxy is not None:
        _remove_object(proxy)
    return report


def write_report_text(report):
    name = "FDMW_Report_%s" % report["object"]
    txt = bpy.data.texts.get(name)
    if txt is None:
        txt = bpy.data.texts.new(name)
    txt.clear()
    lines = []
    lines.append("FDM SUPPORT WIZARD - %s" % report["object"])
    lines.append("=" * 60)
    lines.append("Model size: %.1f x %.1f x %.1f mm | %d faces"
                 % (*report["model_dims_mm"], report["faces"]))
    lines.append("Overhang area needing attention: %.0f mm^2 in %d island(s)"
                 % (report["overhang_area_mm2"], report["islands"]))
    strat = ", ".join("%s x%d" % (k, v)
                      for k, v in sorted(report["strategies"].items()) if v)
    lines.append("Strategies: %s" % (strat or "none needed"))
    if report["bridges_mm"]:
        lines.append("Bridges left to printer: %s mm" % report["bridges_mm"])
    lines.append("")
    lines.append("VALIDATION")
    for k, v in report["validation"].items():
        if isinstance(v, dict):
            lines.append("  [%s] %s  %s"
                         % ("PASS" if v.get("pass", True) else "FAIL", k,
                            {kk: vv for kk, vv in v.items() if kk != "pass"}))
    for w in report["warnings"]:
        lines.append("")
        lines.append("WARNING: %s" % w)
    for m in report["messages"]:
        lines.append("NOTE: %s" % m)
    lines.append("")
    lines.append(report["profile"]["markdown"])
    txt.write("\n".join(lines))
    return txt


def export_stls(obj, sup_ob, directory):
    os.makedirs(directory, exist_ok=True)
    paths = []
    pairs = [(obj, "_model"), (sup_ob, "_supports")] if sup_ob \
        else [(obj, "_model")]
    prev_sel = [o for o in bpy.context.selected_objects]
    prev_act = bpy.context.view_layer.objects.active
    for ob, sfx in pairs:
        fp = os.path.join(directory, "%s%s.stl"
                          % (bpy.path.clean_name(obj.name), sfx))
        for o in bpy.context.selected_objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        try:
            bpy.ops.wm.stl_export(filepath=fp, export_selected_objects=True,
                                  apply_modifiers=True, ascii_format=False,
                                  global_scale=1.0)
        except TypeError:
            bpy.ops.wm.stl_export(filepath=fp, export_selected_objects=True)
        paths.append(fp)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in prev_sel:
        try:
            o.select_set(True)
        except Exception:
            pass
    if prev_act:
        bpy.context.view_layer.objects.active = prev_act
    return paths


def write_profile_files(report, directory):
    os.makedirs(directory, exist_ok=True)
    base = bpy.path.clean_name(report["object"])
    jp = os.path.join(directory, base + "_bambu_process.json")
    mp = os.path.join(directory, base + "_print_profile.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report["profile"]["preset"], f, indent=2)
    with open(mp, "w", encoding="utf-8") as f:
        f.write(report["profile"]["markdown"] + "\n")
    return [jp, mp]


_LAST_REPORT = {}


HEATMAP_ATTR = "VibeCut_Overhang"


def _overhang_ramp(sev, threshold):
    import numpy as np
    green = np.array([0.10, 0.70, 0.15])
    yellow = np.array([0.95, 0.88, 0.10])
    orange = np.array([1.00, 0.45, 0.00])
    red = np.array([0.80, 0.04, 0.04])
    n = len(sev)
    rgb = np.empty((n, 3))
    lo = sev <= threshold
    t1 = np.clip(sev[lo] / max(threshold, 1e-6), 0, 1)[:, None]
    rgb[lo] = green * (1 - t1) + yellow * t1
    hi = ~lo
    t2 = np.clip((sev[hi] - threshold) / max(90.0 - threshold, 1e-6),
                 0, 1)[:, None]
    rgb[hi] = orange * (1 - t2) + red * t2
    rgba = np.ones((n, 4))
    rgba[:, :3] = rgb
    return rgba


def build_overhang_heatmap(obj, threshold_deg):
    import numpy as np
    me = obj.data
    F = len(me.polygons)
    if F == 0:
        return None
    nrm = np.empty(F * 3, dtype=np.float64)
    me.polygons.foreach_get("normal", nrm)
    nrm = nrm.reshape(F, 3)
    mw = np.array(obj.matrix_world)
    nm = np.linalg.inv(mw[:3, :3]).T
    wn = nrm @ nm.T
    ln = np.linalg.norm(wn, axis=1)
    ln[ln < 1e-9] = 1.0
    wn /= ln[:, None]
    nz = wn[:, 2]
    sev = np.degrees(np.arcsin(np.clip(-nz, 0.0, 1.0)))

    face_rgba = _overhang_ramp(sev, threshold_deg)

    areas = np.empty(F, dtype=np.float64)
    me.polygons.foreach_get("area", areas)
    need = sev >= threshold_deg
    pct = 100.0 * areas[need].sum() / max(areas.sum(), 1e-9)
    worst = float(sev.max())

    counts = np.empty(F, dtype=np.int64)
    me.polygons.foreach_get("loop_total", counts)
    loop_face = np.repeat(np.arange(F), counts)
    corner = face_rgba[loop_face]

    attr = me.color_attributes.get(HEATMAP_ATTR)
    if attr is not None and (attr.domain != 'CORNER'
                             or attr.data_type != 'FLOAT_COLOR'):
        me.color_attributes.remove(attr)
        attr = None
    if attr is None:
        attr = me.color_attributes.new(HEATMAP_ATTR, 'FLOAT_COLOR', 'CORNER')
    attr.data.foreach_set("color", corner.ravel())
    try:
        me.color_attributes.active_color = attr
    except Exception:
        pass
    me.update()
    return {"pct_supported": pct, "worst_deg": worst}


def _viewport_spaces(context):
    spaces = []
    win = context.window or (context.window_manager.windows[0]
                             if context.window_manager.windows else None)
    if win is None:
        return spaces
    for area in win.screen.areas:
        if area.type == 'VIEW_3D':
            for s in area.spaces:
                if s.type == 'VIEW_3D':
                    spaces.append(s)
    return spaces


class VIBECUT_OT_overhang_heatmap(bpy.types.Operator):
    """Color the model by overhang steepness so you can see where supports
    are needed before generating them. Green = safe, yellow = right at the
    threshold, red = steep overhang. Uses the same Overhang Angle as the
    support generator"""
    bl_idname = "vibecut.overhang_heatmap"
    bl_label = "Show Overhang Heatmap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return (ob is not None and ob.type == 'MESH'
                and not ob.name.startswith("SUP_"))

    def execute(self, ctx):
        p = ctx.scene.vibecut_supports
        ob = ctx.active_object
        try:
            stats = build_overhang_heatmap(ob, p.overhang_angle)
        except Exception as e:
            self.report({'ERROR'}, "Heatmap failed: %s" % e)
            return {'CANCELLED'}
        if stats is None:
            self.report({'WARNING'}, "Mesh has no faces.")
            return {'CANCELLED'}

        spaces = _viewport_spaces(ctx)
        if not p.hm_active and spaces:
            s0 = spaces[0]
            p.hm_prev_type = s0.shading.type
            p.hm_prev_color = s0.shading.color_type
        for s in spaces:
            s.shading.type = 'SOLID'
            s.shading.color_type = 'VERTEX'
        p.hm_active = True
        p.hm_stats = "%.0f%% of the surface needs support  |  steepest %.0f deg" \
            % (stats["pct_supported"], stats["worst_deg"])
        self.report({'INFO'}, "Overhang heatmap: " + p.hm_stats)
        return {'FINISHED'}


class VIBECUT_OT_clear_heatmap(bpy.types.Operator):
    """Remove the overhang heatmap colors and restore normal viewport
    shading"""
    bl_idname = "vibecut.clear_heatmap"
    bl_label = "Clear Heatmap"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        p = ctx.scene.vibecut_supports
        for ob in bpy.data.objects:
            if ob.type != 'MESH':
                continue
            attr = ob.data.color_attributes.get(HEATMAP_ATTR)
            if attr is not None:
                ob.data.color_attributes.remove(attr)
                ob.data.update()
        for s in _viewport_spaces(ctx):
            if s.shading.color_type == 'VERTEX':
                s.shading.color_type = p.hm_prev_color or 'OBJECT'
                s.shading.type = p.hm_prev_type or 'SOLID'
        p.hm_active = False
        p.hm_stats = ""
        self.report({'INFO'}, "Heatmap cleared.")
        return {'FINISHED'}


class VIBECUT_PG_supports(bpy.types.PropertyGroup):
    printer: bpy.props.EnumProperty(
        name="Printer",
        items=[("X1C", "X1 Carbon / X1", ""), ("P1S", "P1S / P1P", ""),
               ("A1", "A1", ""), ("A1MINI", "A1 mini", ""),
               ("H2D", "H2D", "")],
        default="X1C")
    material: bpy.props.EnumProperty(
        name="Material",
        items=[("PLA", "PLA", ""), ("PETG", "PETG", ""), ("ABS", "ABS", "")],
        default="PLA")
    style: bpy.props.EnumProperty(
        name="Support Style",
        items=[
            ("CLEAR", "Keep-Clear Pins (recommended)",
             "Twist-off pin tips are the ONLY part that touches the model. "
             "Every post, branch and brace keeps the Keep-Out gap from it, "
             "so nothing fuses to the sides. Points with no clear path to "
             "the bed are reported instead of resting on the model"),
            ("PINS", "Branching Pins",
             "Lychee-style: a cone tip on every support point, short "
             "Y-forks, vertical posts, cone feet and bed pads. Tips are "
             "trimmed flush at the model surface"),
            ("SOLID", "Solid (validated)",
             "Multi-strategy tree/pillar/cradle supports, voxel-remeshed "
             "to a guaranteed-watertight block with an air gap. Heavier "
             "but bulletproof on rough or non-printable meshes"),
        ],
        default="CLEAR")
    overhang_angle: bpy.props.FloatProperty(
        name="Overhang Angle", default=45.0, min=20.0, max=75.0,
        description="Faces leaning more than this from vertical get support")
    keepout: bpy.props.FloatProperty(
        name="Keep-Out Gap", default=1.0, min=0.5, max=3.0,
        description="Minimum gap between any support side and the model "
                    "(mm). Only the pin tips touch it")
    merge_reach: bpy.props.FloatProperty(
        name="Merge Reach", default=4.0, min=0.0, max=12.0,
        description="Posts lean toward a neighbour within this distance "
                    "(mm) and merge into trees. 0 = straight separate posts")
    pin_spacing: bpy.props.FloatProperty(
        name="Tip Spacing", default=3.5, min=1.5, max=10.0,
        description="Minimum distance between contact tips (mm). "
                    "Larger = fewer, sparser supports")
    pin_diameter: bpy.props.FloatProperty(
        name="Tip Diameter", default=1.0, min=0.8, max=2.0,
        description="Contact diameter where a tip touches the model (mm). "
                    "0.8 is the floor for a 0.4 mm nozzle")
    pillar_diameter: bpy.props.FloatProperty(
        name="Max Post Diameter", default=2.6, min=1.6, max=8.0,
        description="Posts thicken with the number of tips they carry, "
                    "up to this (mm)")
    max_trunk_reach: bpy.props.FloatProperty(
        name="Fork Reach", default=3.0, min=0.0, max=20.0,
        description="How far sideways two tips may join into one Y-fork "
                    "(mm). 0 = every tip gets its own post")
    conform: bpy.props.EnumProperty(
        name="Trim to Model",
        items=[
            ("MANIFOLD", "On (recommended)",
             "Boolean-trim pin tips flush to the model surface so they "
             "look clean in Blender. Falls back automatically if it fails"),
            ("NONE", "Off (slicer trims)",
             "Leave pins embedded 0.5 mm; the slicer trims them at print "
             "time. Bulletproof - never fails on rough meshes"),
        ],
        default="MANIFOLD")
    z_gap: bpy.props.FloatProperty(
        name="Contact Air Gap", default=0.25, min=0.10, max=0.50,
        description="Vertical gap at support contacts (about one layer) so "
                    "supports snap off instead of fusing")
    bridge_span: bpy.props.FloatProperty(
        name="Max Bridge", default=10.0, min=0.0, max=40.0,
        description="Flat spans up to this length are left to the printer "
                    "(it bridges them) instead of getting supports")
    cradle_enable: bpy.props.BoolProperty(
        name="Allow Cradle Blocks", default=True,
        description="Form-fitting block under big low underbellies "
                    "(hulls, spheres). Falls back to pillars when unsafe")
    strategy_override: bpy.props.EnumProperty(
        name="Strategy",
        items=[("AUTO", "Auto (recommended)", "Pick per overhang island"),
               ("TREE", "Trees only", ""), ("PILLAR", "Pillars only", "")],
        default="AUTO")
    show_advanced: bpy.props.BoolProperty(name="Advanced", default=False)
    voxel_size: bpy.props.FloatProperty(
        name="Surface Detail", default=0.30, min=0.20, max=0.6,
        description="Voxel size of the final remesh (smaller = finer/slower)")
    spacing: bpy.props.FloatProperty(
        name="Tip Spacing", default=2.8, min=2.0, max=6.0,
        description="Distance between support contact tips on overhangs")
    seed: bpy.props.IntProperty(
        name="Seed", default=42,
        description="Random seed for tip placement (results are repeatable)")
    status: bpy.props.StringProperty(default="")
    hm_active: bpy.props.BoolProperty(default=False)
    hm_stats: bpy.props.StringProperty(default="")
    hm_prev_type: bpy.props.StringProperty(default="SOLID")
    hm_prev_color: bpy.props.StringProperty(default="OBJECT")


def _params_from_props(p) -> Params:
    P = Params()
    P.printer = p.printer
    P.material = p.material
    P.style = p.style
    P.overhang_angle = p.overhang_angle
    P.z_gap = p.z_gap
    P.bridge_span = p.bridge_span
    P.cradle_enable = p.cradle_enable
    P.strategy_override = p.strategy_override
    P.voxel_size = p.voxel_size
    P.base_spacing = p.spacing
    P.seed = p.seed
    P.pin_spacing = p.pin_spacing
    P.pin_diameter = p.pin_diameter
    P.pillar_diameter = p.pillar_diameter
    P.max_trunk_reach = p.max_trunk_reach
    P.conform = p.conform
    P.keepout = p.keepout
    P.merge_reach = p.merge_reach
    if p.material == "PETG":
        P.bridge_span = min(P.bridge_span, 6.0)
    elif p.material == "ABS":
        P.bridge_span = min(P.bridge_span, 8.0)
    return P


class VIBECUT_OT_generate_supports(bpy.types.Operator):
    """Analyze overhangs, build multi-strategy support structures with a
    removable air gap, validate them, and prepare a print profile"""
    bl_idname = "vibecut.generate_supports"
    bl_label = "Generate Supports"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return (ob is not None and ob.type == 'MESH'
                and ctx.mode == 'OBJECT'
                and not ob.name.startswith("SUP_"))

    def execute(self, ctx):
        p = ctx.scene.vibecut_supports
        P = _params_from_props(p)
        ob = ctx.active_object
        wm = ctx.window_manager
        wm.progress_begin(0, 100)
        try:
            rep = run_pipeline(ob, P)
        except Exception as e:
            wm.progress_end()
            p.status = "Error: %s" % e
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        wm.progress_end()
        _LAST_REPORT[ob.name] = rep
        write_report_text(rep)
        v = rep["validation"]
        ok = v.get("all_pass", False)
        n_str = ", ".join("%s x%d" % (k, c) for k, c in
                          sorted(rep["strategies"].items()) if c)
        if rep["support_object"] is None:
            p.status = "No supports needed. " + \
                ("Bridges: %s mm" % rep["bridges_mm"]
                 if rep["bridges_mm"] else "")
            self.report({'INFO'}, "No support geometry needed.")
        else:
            p.status = "%s | checks: %s | %.0fs" % (
                n_str, "ALL PASS" if ok else "SEE REPORT", rep["seconds"])
            att = rep["attempts"][-1] if rep["attempts"] else {}
            if att.get("unreachable"):
                p.status += " | %d point(s) unsupported - no clear path" \
                    % att["unreachable"]
            msg = ("Supports created (%s). Validation: %s. Full report in "
                   "the '%s' text block." %
                   (n_str, "all checks passed" if ok else "SOME CHECKS FAILED",
                    "FDMW_Report_" + ob.name))
            self.report({'INFO'} if ok else {'WARNING'}, msg)
        return {'FINISHED'}


class VIBECUT_OT_remove_supports(bpy.types.Operator):
    """Delete all generated support objects and the VibeCut Supports
    collection"""
    bl_idname = "vibecut.remove_supports"
    bl_label = "Remove Supports"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, ctx):
        coll = bpy.data.collections.get(SUPPORT_COLL_NAME)
        if coll is None:
            self.report({'INFO'}, "No support collection to remove.")
            return {'CANCELLED'}
        removed = len(coll.objects)
        for existing in list(coll.objects):
            _remove_object(existing)
        bpy.data.collections.remove(coll)
        self.report({'INFO'}, "Removed %d support object(s)." % removed)
        return {'FINISHED'}


class VIBECUT_OT_export_print_stls(bpy.types.Operator):
    """Write model + supports STL files (same coordinates) ready to drag
    into Bambu Studio together"""
    bl_idname = "vibecut.export_print_stls"
    bl_label = "Export STLs for Bambu Studio"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return (ob is not None and ob.type == 'MESH'
                and not ob.name.startswith("SUP_"))

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        ob = context.active_object
        sup = bpy.data.objects.get("SUP_" + ob.name)
        try:
            paths = export_stls(ob, sup, self.directory)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Exported: %s" %
                    ", ".join(os.path.basename(x) for x in paths))
        return {'FINISHED'}


class VIBECUT_OT_save_print_profile(bpy.types.Operator):
    """Write the Bambu Studio process preset (.json) and a plain-English
    settings guide (.md) for the last generated result"""
    bl_idname = "vibecut.save_print_profile"
    bl_label = "Save Print Profile"

    directory: bpy.props.StringProperty(subtype='DIR_PATH')

    @classmethod
    def poll(cls, ctx):
        ob = ctx.active_object
        return ob is not None and ob.name in _LAST_REPORT

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        rep = _LAST_REPORT[context.active_object.name]
        try:
            paths = write_profile_files(rep, self.directory)
        except Exception as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, "Saved: %s" %
                    ", ".join(os.path.basename(x) for x in paths))
        return {'FINISHED'}


class VIBECUT_PT_supports(bpy.types.Panel):
    bl_label = "Supports"
    bl_idname = "VIBECUT_PT_supports"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VibeCut"
    bl_parent_id = "VIBECUT_PT_more"
    bl_order = 2
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, ctx):
        L = self.layout
        p = ctx.scene.vibecut_supports

        col = L.column()
        col.prop(p, "printer")
        col.prop(p, "material")
        col.prop(p, "style")

        hbox = col.box()
        hrow = hbox.row(align=True)
        if p.hm_active:
            hrow.operator("vibecut.overhang_heatmap",
                          text="Update", icon='IMAGE_RGB')
            hrow.operator("vibecut.clear_heatmap", text="Clear", icon='X')
            if p.hm_stats:
                for chunk in p.hm_stats.split("  |  "):
                    hbox.label(text=chunk)
            leg = hbox.row(align=True)
            leg.label(text="safe", icon='COLORSET_03_VEC')
            leg.label(text="threshold", icon='COLORSET_09_VEC')
            leg.label(text="steep", icon='COLORSET_01_VEC')
        else:
            hrow.operator("vibecut.overhang_heatmap",
                          text="Show Overhang Heatmap", icon='IMAGE_RGB')

        col.separator()
        big = col.column()
        big.scale_y = 1.5
        big.operator("vibecut.generate_supports", icon='MOD_BUILD')
        if p.status:
            box = col.box()
            for chunk in p.status.split(" | "):
                box.label(text=chunk)
        col.separator()
        col.operator("vibecut.export_print_stls", icon='EXPORT')
        col.operator("vibecut.save_print_profile", icon='FILE_TEXT')
        col.separator()
        col.prop(p, "show_advanced", toggle=True, icon='PREFERENCES')
        if p.show_advanced:
            box = col.box()
            box.prop(p, "overhang_angle")
            if p.style == "CLEAR":
                box.prop(p, "keepout")
                box.prop(p, "merge_reach")
                box.prop(p, "pin_spacing")
                box.prop(p, "pin_diameter")
                box.prop(p, "conform")
            elif p.style == "PINS":
                box.prop(p, "pin_spacing")
                box.prop(p, "pin_diameter")
                box.prop(p, "pillar_diameter")
                box.prop(p, "max_trunk_reach")
                box.prop(p, "conform")
            else:
                box.prop(p, "z_gap")
                box.prop(p, "bridge_span")
                box.prop(p, "spacing")
                box.prop(p, "voxel_size")
                box.prop(p, "cradle_enable")
                box.prop(p, "strategy_override")
                box.prop(p, "seed")

        coll = bpy.data.collections.get(SUPPORT_COLL_NAME)
        if coll and len(coll.objects) > 0:
            L.separator()
            L.operator("vibecut.remove_supports",
                       text="Remove Supports (%d)" % len(coll.objects),
                       icon='TRASH')


classes = (
    VIBECUT_OT_generate_supports,
    VIBECUT_OT_remove_supports,
    VIBECUT_OT_export_print_stls,
    VIBECUT_OT_save_print_profile,
    VIBECUT_OT_overhang_heatmap,
    VIBECUT_OT_clear_heatmap,
    VIBECUT_PT_supports,
)


def register_props():
    bpy.utils.register_class(VIBECUT_PG_supports)
    bpy.types.Scene.vibecut_supports = bpy.props.PointerProperty(
        type=VIBECUT_PG_supports)


def unregister_props():
    if hasattr(bpy.types.Scene, "vibecut_supports"):
        del bpy.types.Scene.vibecut_supports
    bpy.utils.unregister_class(VIBECUT_PG_supports)


def register():
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
