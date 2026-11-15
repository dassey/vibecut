"""Automatic limb / neck detection for joint placement.

Method: morphological EROSION SPLITTING on a voxelization of the mesh.

1.  Voxelize the evaluated mesh into an NxNxN boolean grid. Primary path
    is pure numpy (bin vertices + triangle midpoints into a surface shell,
    flood the OUTSIDE from the grid border, solid = not outside) - cost
    scales with the voxel grid, not the vertex count, so a 1.4M-triangle
    scan voxelizes in about the same time as a 10k one. Low-poly meshes
    (where big triangles would leak the shell) automatically fall back to
    BVH column marching with signed-crossing counts.
2.  Interior distance transform via vectorized 6-neighbour dilation.
3.  Erode progressively: threshold the distance field at d = 1, 2, 3...
    When a connected component SPLITS into two, the part that broke off is
    a limb and the break point is its neck:
      - neck position  = midpoint of the closest voxel pair between parts
      - neck radius    = the erosion depth at the split (in mm)
      - cut direction  = the axis between the two part centroids
      - score          = 1 - d_split / d_limb_interior  (how much fatter
                         the limb is than its neck; 0 = no pinch at all)
    This finds pinch points regardless of absolute size - thin arms and
    fat legs alike - with no reliance on vertex normals, manifoldness, or
    "thick cluster" heuristics.

Known limitation (documented, by design): a perfectly cylindrical limb
joining a fat body with NO pinch at all never separates under erosion and
is not reported. Organic figures pinch at real joints, which is exactly
where you want to cut.

Public API (used by auto_detect.py):
    detect_necks_blender(mesh_obj, resolution=96, sensitivity=0.3)
        -> list of dicts: world_pos (Vector), world_normal (Vector),
           radius_world (float), score (float), limb_vert_indices (set)
"""

import time
from collections import deque

import numpy as np

import bpy
from mathutils import Vector


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------

def _world_mesh_arrays(mesh_obj):
    """Evaluated mesh -> (verts (V,3) float64 world, tris (T,3) int32)."""
    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = mesh_obj.evaluated_get(dg)
    me = ob_eval.to_mesh()
    try:
        n_v = len(me.vertices)
        verts = np.empty(n_v * 3, dtype=np.float64)
        me.vertices.foreach_get("co", verts)
        verts = verts.reshape(n_v, 3)

        me.calc_loop_triangles()
        n_t = len(me.loop_triangles)
        tris = np.empty(n_t * 3, dtype=np.int32)
        me.loop_triangles.foreach_get("vertices", tris)
        tris = tris.reshape(n_t, 3)
    finally:
        ob_eval.to_mesh_clear()

    mw = np.array(mesh_obj.matrix_world)
    verts = verts @ mw[:3, :3].T + mw[:3, 3]
    return verts, tris


# ---------------------------------------------------------------------------
# Voxelization
# ---------------------------------------------------------------------------

def _dilate6(m):
    """6-connected binary dilation without wraparound."""
    out = m.copy()
    out[1:, :, :] |= m[:-1, :, :]
    out[:-1, :, :] |= m[1:, :, :]
    out[:, 1:, :] |= m[:, :-1, :]
    out[:, :-1, :] |= m[:, 1:, :]
    out[:, :, 1:] |= m[:, :, :-1]
    out[:, :, :-1] |= m[:, :, 1:]
    return out


def _bin_points(pts, origin, voxel_size, shape, shell):
    idx = np.floor((pts - origin) / voxel_size).astype(np.int64)
    ok = ((idx >= 0) & (idx < np.array(shape))).all(axis=1)
    idx = idx[ok]
    if len(idx):
        shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True


def _voxelize_shell_flood(verts, tris, resolution, padding=2):
    """Numpy voxelization: surface shell from binned mesh samples, then
    flood the outside from the border. Returns (solid, origin, voxel_size)
    or None if the shell leaked (outside flood ate nearly everything)."""
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    extent = float((bb_max - bb_min).max())
    if extent <= 0:
        return None
    voxel_size = extent / (resolution - 2 * padding)
    shape = np.maximum(
        np.ceil((bb_max - bb_min) / voxel_size).astype(int) + 2 * padding, 8)
    origin = bb_min - padding * voxel_size

    shell = np.zeros(tuple(shape), dtype=bool)
    _bin_points(verts, origin, voxel_size, shape, shell)
    if len(tris):
        v0 = verts[tris[:, 0]]
        v1 = verts[tris[:, 1]]
        v2 = verts[tris[:, 2]]
        # Centroid + edge midpoints seal faces whose verts span voxels.
        for pts in ((v0 + v1 + v2) / 3.0, (v0 + v1) / 2.0,
                    (v1 + v2) / 2.0, (v0 + v2) / 2.0):
            _bin_points(pts, origin, voxel_size, shape, shell)

    # Morphological closing: printable figurine STLs are often HOLLOW
    # shells with drainage openings. Dilating the shell seals openings up
    # to ~4 voxels before the outside flood, then the outside grows back
    # to the true surface (constrained by the real shell). Interior
    # cavities end up solid - which is what limb-thickness analysis wants.
    close_k = 2
    shell_closed = shell
    for _ in range(close_k):
        shell_closed = _dilate6(shell_closed)

    free = ~shell_closed
    outside = np.zeros_like(shell)
    outside[0, :, :] = free[0, :, :]
    outside[-1, :, :] = free[-1, :, :]
    outside[:, 0, :] |= free[:, 0, :]
    outside[:, -1, :] |= free[:, -1, :]
    outside[:, :, 0] |= free[:, :, 0]
    outside[:, :, -1] |= free[:, :, -1]
    while True:
        grown = _dilate6(outside) & free
        grown |= outside
        if grown.sum() == outside.sum():
            break
        outside = grown

    # Grow the conservative outside back to the true surface.
    for _ in range(close_k):
        outside = (_dilate6(outside) & ~shell) | outside

    solid = ~outside
    interior = solid & ~shell
    # Leak check: a sealed object encloses real volume beyond its shell.
    if interior.sum() < max(64, 0.002 * solid.size):
        return None
    return solid, origin, voxel_size


def _voxelize_bvh_columns(mesh_obj, resolution, padding=2):
    """Fallback voxelizer: BVH ray marching per (x, y) column with signed
    crossing counts (robust to coincident faces and doubled shells)."""
    import bmesh
    from mathutils.bvhtree import BVHTree

    dg = bpy.context.evaluated_depsgraph_get()
    ob_eval = mesh_obj.evaluated_get(dg)
    bm = bmesh.new()
    bm.from_object(ob_eval, dg)
    bm.transform(mesh_obj.matrix_world)
    bvh = BVHTree.FromBMesh(bm)
    xs = [v.co.x for v in bm.verts]
    ys = [v.co.y for v in bm.verts]
    zs = [v.co.z for v in bm.verts]
    bm.free()
    if not xs:
        return None
    bb_min = np.array([min(xs), min(ys), min(zs)])
    bb_max = np.array([max(xs), max(ys), max(zs)])
    extent = float((bb_max - bb_min).max())
    if extent <= 0:
        return None
    voxel_size = extent / (resolution - 2 * padding)
    shape = np.maximum(
        np.ceil((bb_max - bb_min) / voxel_size).astype(int) + 2 * padding, 8)
    origin = bb_min - padding * voxel_size

    solid = np.zeros(tuple(shape), dtype=bool)
    up = Vector((0.0, 0.0, 1.0))
    z0 = origin[2] - voxel_size
    for ix in range(shape[0]):
        wx = origin[0] + (ix + 0.5) * voxel_size
        for iy in range(shape[1]):
            wy = origin[1] + (iy + 0.5) * voxel_size
            o = Vector((wx, wy, z0))
            net = 0
            spans = []
            enter_z = None
            for _hop in range(128):
                hit = bvh.ray_cast(o, up)
                if hit[0] is None:
                    break
                step = 1 if (hit[1] is not None and hit[1].z >= 0) else -1
                prev = net
                net += step
                if prev <= 0 and net > 0:
                    enter_z = hit[0].z
                elif prev > 0 and net <= 0 and enter_z is not None:
                    spans.append((enter_z, hit[0].z))
                    enter_z = None
                o = hit[0] + up * 1e-4
            for (za, zb) in spans:
                ia = max(0, int(np.floor((za - origin[2]) / voxel_size)))
                ib = min(int(shape[2]),
                         int(np.ceil((zb - origin[2]) / voxel_size)))
                if ia < ib:
                    solid[ix, iy, ia:ib] = True
    if not solid.any():
        return None
    # Fill enclosed cavities (hollow shells) so walls read as full limbs.
    free = ~solid
    outside = np.zeros_like(solid)
    outside[0, :, :] = free[0, :, :]
    outside[-1, :, :] = free[-1, :, :]
    outside[:, 0, :] |= free[:, 0, :]
    outside[:, -1, :] |= free[:, -1, :]
    outside[:, :, 0] |= free[:, :, 0]
    outside[:, :, -1] |= free[:, :, -1]
    while True:
        grown = _dilate6(outside) & free
        grown |= outside
        if grown.sum() == outside.sum():
            break
        outside = grown
    solid = ~outside
    return solid, origin, voxel_size


# ---------------------------------------------------------------------------
# Distance transform (vectorized BFS layers)
# ---------------------------------------------------------------------------

def distance_transform(solid):
    """Near-Euclidean interior distance via a Borgefors 3-4-5 chamfer
    (weights 3 / 4 / 5 for face / edge / corner steps, result divided by
    3). A plain 6-connected layer count overestimates diagonal distances
    by up to 73%, which inflates necks relative to limbs and crushes
    split scores; the chamfer stays within ~8% of true Euclidean.

    Iterative relaxation with whole-array shifted minimums - converges in
    about max_thickness sweeps, all numpy. Empty voxels = 0; surface
    voxels ~= 1."""
    solid = np.asarray(solid, dtype=bool)
    full = np.zeros(solid.shape, dtype=np.float32)
    if not solid.any():
        return full
    # Crop to the solid's bounding box (plus 1) for speed.
    idx = np.argwhere(solid)
    lo = np.maximum(idx.min(axis=0) - 1, 0)
    hi = np.minimum(idx.max(axis=0) + 2, solid.shape)
    sub = solid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    INF = np.int32(1 << 29)
    d = np.where(sub, INF, np.int32(0)).astype(np.int32)
    X, Y, Z = d.shape

    def _sl(off, n):
        if off > 0:
            return slice(off, n), slice(0, n - off)
        if off < 0:
            return slice(0, n + off), slice(-off, n)
        return slice(0, n), slice(0, n)

    offs = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                w = (3, 4, 5)[abs(dx) + abs(dy) + abs(dz) - 1]
                offs.append((dx, dy, dz, np.int32(w)))

    for _sweep in range(512):
        changed = False
        for dx, dy, dz, w in offs:
            tx, sx = _sl(dx, X)
            ty, sy = _sl(dy, Y)
            tz, sz = _sl(dz, Z)
            cand = d[sx, sy, sz] + w
            tgt = d[tx, ty, tz]
            m = cand < tgt
            if m.any():
                tgt[m] = cand[m]
                changed = True
        if not changed:
            break

    sub_dist = (d.astype(np.float32) / 3.0)
    sub_dist[~sub] = 0.0
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = sub_dist
    return full


# ---------------------------------------------------------------------------
# Connected-component labeling (bbox-cropped, dilation BFS)
# ---------------------------------------------------------------------------

def _components(mask):
    """Split a boolean mask into 6-connected components. Returns a list of
    boolean masks (same shape), largest first."""
    comps = []
    rem = mask.copy()
    while rem.any():
        # Crop to the remaining bbox for speed.
        idx = np.argwhere(rem)
        lo = idx.min(axis=0)
        hi = idx.max(axis=0) + 1
        sub = rem[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        seed_local = np.argwhere(sub)[0]
        comp_sub = np.zeros_like(sub)
        comp_sub[tuple(seed_local)] = True
        while True:
            grown = _dilate6(comp_sub) & sub
            grown |= comp_sub
            if grown.sum() == comp_sub.sum():
                break
            comp_sub = grown
        comp = np.zeros_like(mask)
        comp[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = comp_sub
        comps.append(comp)
        rem &= ~comp
    comps.sort(key=lambda c: c.sum(), reverse=True)
    return comps


def _closest_pair_midpoint(mask_a, mask_b, max_samples=1500):
    """Approximate closest voxel pair between two masks -> midpoint (voxel
    coords, float) and the pair distance."""
    a = np.argwhere(mask_a).astype(np.float64)
    b = np.argwhere(mask_b).astype(np.float64)
    rng = np.random.default_rng(7)
    if len(a) > max_samples:
        a = a[rng.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[rng.choice(len(b), max_samples, replace=False)]
    # Chunked broadcast to bound memory.
    best = (np.inf, None, None)
    chunk = 256
    for i in range(0, len(a), chunk):
        d2 = ((a[i:i + chunk, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        j = np.unravel_index(np.argmin(d2), d2.shape)
        v = d2[j]
        if v < best[0]:
            best = (v, a[i + j[0]], b[j[1]])
    if best[1] is None:
        return None, np.inf
    return (best[1] + best[2]) / 2.0, float(np.sqrt(best[0]))


# ---------------------------------------------------------------------------
# Erosion splitting
# ---------------------------------------------------------------------------

def find_neck_events(dist, min_child_vox=30, max_events=64):
    """Walk the erosion hierarchy of the distance field and record every
    component split.

    Returns a list of dicts in voxel space:
        pos (3,) float, radius_vox float, direction (3,) float,
        score float, child_size int, depth int
    """
    events = []
    d_global_max = int(np.ceil(dist.max()))
    if d_global_max < 2:
        return events

    def recurse(mask, d):
        # Erode this component until it splits or dies.
        while True:
            d += 1
            sub = mask & (dist >= d)
            n = int(sub.sum())
            if n == 0:
                return
            comps = _components(sub)
            if len(comps) == 1:
                mask = comps[0]
                continue
            big = comps[0]
            kept = [c for c in comps[1:] if c.sum() >= min_child_vox]
            if big.sum() >= min_child_vox:
                for child in kept:
                    if len(events) >= max_events:
                        return
                    mid, _gap = _closest_pair_midpoint(child, big)
                    if mid is None:
                        continue
                    c_child = np.argwhere(child).mean(axis=0)
                    c_big = np.argwhere(big).mean(axis=0)
                    axis = c_child - c_big
                    norm = np.linalg.norm(axis)
                    axis = axis / norm if norm > 1e-9 else np.array(
                        [0.0, 0.0, 1.0])
                    d_limb = float(dist[child].max())
                    score = 1.0 - (d - 1) / max(d_limb, 1.0)
                    events.append({
                        "pos": mid,
                        "radius_vox": max(d - 1.0, 1.0),
                        "direction": axis,
                        "score": float(np.clip(score, 0.0, 1.0)),
                        "child_size": int(child.sum()),
                        "depth": d,
                        # Flat voxel indices of the limb core. `child` is
                        # captured mid-erosion at depth d, so it excludes the
                        # limb's own surface shell — the caller regrows it.
                        # Stored as flat indices (proportional to limb size)
                        # rather than a full resolution^3 grid per event, which
                        # would cost ~900 KB each at resolution 96.
                        "child_idx": np.flatnonzero(child.ravel()),
                        "grid_shape": child.shape,
                    })
            # Recurse into every surviving piece.
            for c in comps:
                if c.sum() >= min_child_vox:
                    recurse(c, d)
            return

    solid_comps = _components(dist >= 1)
    for comp in solid_comps:
        if comp.sum() >= min_child_vox:
            recurse(comp, 1)
    return events


# ---------------------------------------------------------------------------
# Blender bridge
# ---------------------------------------------------------------------------

def detect_necks_blender(mesh_obj, resolution=96, sensitivity=0.3):
    """Detect limb/neck cut candidates on a Blender mesh object.

    resolution  : voxel grid size along the longest axis (64-192 useful).
    sensitivity : minimum neck score (1 - neck/limb thickness ratio).
                  Lower finds subtler pinches (wrists), higher keeps only
                  strong necks.
    """
    t0 = time.time()
    resolution = int(max(48, min(192, resolution)))

    verts, tris = _world_mesh_arrays(mesh_obj)
    n_v = len(verts)
    if n_v < 4:
        return []
    print(f"[VibeCut] detect: {n_v} verts, {len(tris)} tris, "
          f"grid {resolution}, sensitivity {sensitivity:.2f}")

    # Dense meshes seal the binned shell; low-poly ones leak -> BVH path.
    vox = None
    if len(tris) >= 5000:
        vox = _voxelize_shell_flood(verts, tris, resolution)
        if vox is None:
            print("[VibeCut] detect: shell voxelization leaked, "
                  "falling back to BVH columns")
    if vox is None:
        vox = _voxelize_bvh_columns(mesh_obj, resolution)
    if vox is None:
        print("[VibeCut] detect: voxelization failed")
        return []
    solid, origin, voxel_size = vox
    t1 = time.time()
    print(f"[VibeCut] detect: voxelized {solid.shape} "
          f"({int(solid.sum())} solid voxels, {voxel_size:.2f} mm/vox) "
          f"in {t1 - t0:.1f}s")

    dist = distance_transform(solid)
    t2 = time.time()
    print(f"[VibeCut] detect: distance field max depth {int(dist.max())} "
          f"in {t2 - t1:.1f}s")

    events = find_neck_events(dist)
    t3 = time.time()
    print(f"[VibeCut] detect: {len(events)} raw split events in "
          f"{t3 - t2:.1f}s")

    # Mesh vertex -> voxel lookup, built once and shared by every event.
    # Only valid as datablock indices when the object carries no modifiers
    # (_world_mesh_arrays reads the EVALUATED mesh).
    has_mods = bool(getattr(mesh_obj, "modifiers", None))
    vox_of_vert = None
    if not has_mods:
        vi = np.floor((verts - origin) / voxel_size).astype(np.int64)
        np.clip(vi, 0, np.array(solid.shape) - 1, out=vi)
        vox_of_vert = np.ravel_multi_index(
            (vi[:, 0], vi[:, 1], vi[:, 2]), solid.shape)

    # Voxel -> world, filter by sensitivity, dedupe.
    out = []
    for ev in events:
        if ev["score"] < sensitivity:
            continue
        wp = origin + (ev["pos"] + 0.5) * voxel_size

        # Recover the limb's vertex membership from the voxel mask the split
        # search already computed. Regrow the eroded core back out to the
        # surface: `child` was taken at erosion depth d, so d dilations
        # restricted to `solid` restores roughly the original limb. The neck is
        # a bottleneck of radius ~d, so bleed past it is bounded by the neck
        # itself — which is exactly where we want the seam.
        limb_idx = set()
        if vox_of_vert is not None:
            m = np.zeros(solid.size, dtype=bool)
            m[ev["child_idx"]] = True
            m = m.reshape(ev["grid_shape"])
            for _ in range(int(ev["depth"])):
                m = _dilate6(m) & solid
            hit = m.ravel()[vox_of_vert]
            limb_idx = set(np.flatnonzero(hit).tolist())

        out.append({
            "world_pos": Vector(wp.tolist()),
            # Point the normal from the limb toward the BODY. joints'
            # place_press_fit embeds the pin behind the plane and pushes it
            # along +no into the body, so this sign is load-bearing.
            "world_normal": Vector((-ev["direction"]).tolist()).normalized(),
            "radius_world": float(ev["radius_vox"] * voxel_size),
            "score": ev["score"],
            "limb_vert_indices": limb_idx,
        })

    out.sort(key=lambda n: n["score"], reverse=True)
    deduped = []
    for n in out:
        keep = True
        for e in deduped:
            r = max(n["radius_world"], e["radius_world"], 1e-6)
            if (n["world_pos"] - e["world_pos"]).length < 1.5 * r:
                keep = False
                break
        if keep:
            deduped.append(n)

    print(f"[VibeCut] detect: {len(deduped)} candidates "
          f"(total {time.time() - t0:.1f}s)")
    return deduped


# ---------------------------------------------------------------------------
# Scene property registration
# ---------------------------------------------------------------------------

_PROP_NAMES = (
    'vibecut_geo_show_settings', 'vibecut_geo_extremities',
    'vibecut_geo_levels', 'vibecut_geo_sensitivity',
    'vibecut_coacd_show_settings', 'vibecut_coacd_threshold',
    'vibecut_coacd_max_hulls', 'vibecut_coacd_preprocess_res',
    'vibecut_coacd_mcts_nodes', 'vibecut_coacd_mcts_iters',
    'vibecut_coacd_mcts_depth', 'vibecut_coacd_merge',
    'vibecut_coacd_pca', 'vibecut_coacd_max_ch_vertex',
    'vibecut_coacd_resolution',
    'vibecut_resolution', 'vibecut_sensitivity',
    'vibecut_candidates_json', 'vibecut_active_candidate',
    'vibecut_cut_pairs_json',
)


def register_props():
    # Geodesic settings (experimental geodesic.py - not wired to UI)
    bpy.types.Scene.vibecut_geo_show_settings = bpy.props.BoolProperty(
        name="Show Geodesic Settings",
        default=False,
    )
    bpy.types.Scene.vibecut_geo_extremities = bpy.props.IntProperty(
        name="Extremities",
        description="Number of limb tips to search for",
        default=8, min=2, max=20,
    )
    bpy.types.Scene.vibecut_geo_levels = bpy.props.IntProperty(
        name="Sweep Levels",
        description="Number of distance bands per sweep (higher = finer resolution)",
        default=60, min=20, max=200,
    )
    bpy.types.Scene.vibecut_geo_sensitivity = bpy.props.FloatProperty(
        name="Explosion Factor",
        description="Perimeter increase ratio to detect a joint (2.0 = perimeter doubles)",
        default=2.0, min=1.2, max=5.0, step=10,
    )
    # CoACD settings (kept for compatibility with saved scenes)
    bpy.types.Scene.vibecut_coacd_show_settings = bpy.props.BoolProperty(
        name="Show Settings", default=False,
    )
    bpy.types.Scene.vibecut_coacd_threshold = bpy.props.FloatProperty(
        name="Threshold",
        description="Concavity tolerance. Lower = more cuts, Higher = fewer cuts",
        default=0.08, min=0.01, max=1.0, step=1,
    )
    bpy.types.Scene.vibecut_coacd_max_hulls = bpy.props.IntProperty(
        name="Max Hulls",
        description="Upper limit on number of parts. -1 = no limit",
        default=-1, min=-1, max=128,
    )
    bpy.types.Scene.vibecut_coacd_preprocess_res = bpy.props.IntProperty(
        name="Preprocess Res",
        description="Resolution for manifold preprocessing (10-100). Lower = faster",
        default=50, min=10, max=100,
    )
    bpy.types.Scene.vibecut_coacd_mcts_nodes = bpy.props.IntProperty(
        name="MCTS Nodes",
        description="Search breadth. Higher = better fit, slower",
        default=20, min=5, max=40,
    )
    bpy.types.Scene.vibecut_coacd_mcts_iters = bpy.props.IntProperty(
        name="MCTS Iterations",
        description="Search effort. Higher = better fit, slower",
        default=150, min=30, max=300,
    )
    bpy.types.Scene.vibecut_coacd_mcts_depth = bpy.props.IntProperty(
        name="MCTS Depth",
        description="Search depth for splitting",
        default=3, min=1, max=5,
    )
    bpy.types.Scene.vibecut_coacd_merge = bpy.props.BoolProperty(
        name="Merge Sub-Hulls",
        description="Merge adjacent sub-hull components after decomposition",
        default=True,
    )
    bpy.types.Scene.vibecut_coacd_pca = bpy.props.BoolProperty(
        name="PCA Pre-processing",
        description="Enable PCA for better hull alignment",
        default=False,
    )
    bpy.types.Scene.vibecut_coacd_max_ch_vertex = bpy.props.IntProperty(
        name="Target Vertex Count",
        description="Max vertices per convex hull (simplification)",
        default=256, min=16, max=1024,
    )
    bpy.types.Scene.vibecut_coacd_resolution = bpy.props.IntProperty(
        name="Sampling Resolution",
        description="Resolution for Hausdorff distance calculation (higher = more accurate)",
        default=2000, min=1000, max=10000,
    )
    bpy.types.Scene.vibecut_resolution = bpy.props.IntProperty(
        name="Detection Resolution",
        description="Voxel grid size along the longest axis. Higher finds "
                    "smaller necks but is slower (96 is a good default)",
        default=96, min=64, max=192,
    )
    bpy.types.Scene.vibecut_sensitivity = bpy.props.FloatProperty(
        name="Sensitivity",
        description="Minimum neck score to report. LOWER finds subtler "
                    "pinches (wrists, ankles); higher keeps only strong necks",
        default=0.30, min=0.05, max=0.8,
    )
    bpy.types.Scene.vibecut_candidates_json = bpy.props.StringProperty(
        name="Candidates JSON",
        description="Serialised list of detected cut candidates",
        default="[]",
    )
    bpy.types.Scene.vibecut_active_candidate = bpy.props.IntProperty(
        name="Active Candidate",
        description="Index of the currently selected candidate",
        default=0,
    )
    bpy.types.Scene.vibecut_cut_pairs_json = bpy.props.StringProperty(
        name="Cut Pairs JSON",
        description="Serialised list of body/limb pairs after cutting",
        default="[]",
    )


def unregister_props():
    for attr in _PROP_NAMES:
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
