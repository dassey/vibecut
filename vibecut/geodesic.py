"""
geodesic.py — Geodesic sweeping for automatic joint/neck detection.

Algorithm (Katz & Tal 2003 inspired):
1. Find extremities via farthest-point sampling on mesh surface
2. From each extremity, compute geodesic distance to all vertices (Dijkstra)
3. Extract contour rings at distance intervals (level sets)
4. Measure the perimeter of each ring
5. Where perimeter suddenly explodes = the ring just hit the body
6. The ring right before the explosion = the neck/joint location

Pure Python + numpy. No external dependencies beyond Blender's built-in modules.
"""

import numpy as np
import heapq
import bpy
from mathutils import Vector


def _build_adjacency(mesh_obj):
    """Build edge-weighted adjacency list from a Blender mesh object.

    Returns
    -------
    adj : list of list of (neighbor_idx, edge_length)
    world_coords : np.ndarray, shape (n_verts, 3)
    """
    mesh = mesh_obj.data
    mw = mesh_obj.matrix_world
    n_verts = len(mesh.vertices)

    # Get world-space vertex positions
    world_coords = np.array([list(mw @ v.co) for v in mesh.vertices],
                            dtype=np.float64)

    # Build adjacency from edges
    adj = [[] for _ in range(n_verts)]
    for edge in mesh.edges:
        a, b = edge.vertices
        dist = np.linalg.norm(world_coords[a] - world_coords[b])
        adj[a].append((b, dist))
        adj[b].append((a, dist))

    return adj, world_coords


def _dijkstra(adj, source, n_verts):
    """Compute geodesic distance from source vertex to all others.

    Returns
    -------
    dist : np.ndarray, shape (n_verts,), float64
        Distance from source. inf for unreachable vertices.
    """
    dist = np.full(n_verts, np.inf, dtype=np.float64)
    dist[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    return dist


def _find_extremities(adj, n_verts, num_extremities=8):
    """Find mesh extremities via iterative farthest-point sampling.

    Start from a random vertex, find the farthest point, then keep finding
    points that are farthest from all previously found points.

    Returns
    -------
    list of int — vertex indices of extremities
    """
    # Start from vertex 0
    extremities = []
    current = 0

    for _ in range(num_extremities):
        dist = _dijkstra(adj, current, n_verts)
        # Find the farthest reachable vertex
        reachable = dist < np.inf
        if not reachable.any():
            break

        # If we have previous extremities, find the vertex farthest from ALL of them
        if extremities:
            # Compute min distance to any existing extremity
            min_dists = np.full(n_verts, np.inf)
            for ext in extremities:
                ext_dist = _dijkstra(adj, ext, n_verts)
                min_dists = np.minimum(min_dists, ext_dist)
            # Pick the vertex with the largest min-distance
            candidates = np.where(min_dists < np.inf)[0]
            if len(candidates) == 0:
                break
            current = candidates[np.argmax(min_dists[candidates])]
        else:
            # First iteration: just pick the farthest from vertex 0
            current = np.argmax(dist * reachable)

        extremities.append(current)
        print(f"[VibeCut] Geodesic: extremity {len(extremities)}: "
              f"vertex {current}")

    return extremities


def _extract_level_sets(world_coords, dist_field, num_levels=50):
    """Extract contour rings at evenly spaced geodesic distance intervals.

    For each level, find vertices in that distance band and compute the
    approximate perimeter of the cross-section.

    Returns
    -------
    levels : list of dict with 'distance', 'perimeter', 'centroid', 'vertices', 'radius'
    """
    max_dist = dist_field[dist_field < np.inf].max()
    min_dist = dist_field[dist_field < np.inf].min()
    step = (max_dist - min_dist) / num_levels

    if step < 1e-8:
        return []

    levels = []
    for i in range(num_levels):
        lo = min_dist + i * step
        hi = lo + step

        # Vertices in this band
        mask = (dist_field >= lo) & (dist_field < hi)
        band_verts = np.where(mask)[0]

        if len(band_verts) < 3:
            continue

        # Compute the perimeter: approximate by the spread of positions
        positions = world_coords[band_verts]
        centroid = positions.mean(axis=0)

        # Perimeter estimate: sum of distances between consecutive vertices
        # projected onto the plane perpendicular to the sweep direction.
        # Simpler: use the bounding radius * 2pi as perimeter estimate
        dists_from_center = np.linalg.norm(positions - centroid, axis=1)
        avg_radius = dists_from_center.mean()
        perimeter = 2 * np.pi * avg_radius

        levels.append({
            'distance': (lo + hi) / 2,
            'perimeter': float(perimeter),
            'centroid': centroid,
            'radius': float(avg_radius),
            'vertices': band_verts,
            'band_lo': lo,
            'band_hi': hi,
        })

    return levels


def _find_perimeter_explosions(levels, sensitivity=2.0):
    """Find where the perimeter suddenly jumps (the ring hit the body).

    The neck is the level BEFORE the explosion.

    Parameters
    ----------
    levels : list of dicts from _extract_level_sets
    sensitivity : float
        A perimeter increase of > sensitivity × previous perimeter = explosion.

    Returns
    -------
    list of dict — neck candidates with 'distance', 'centroid', 'radius',
                   'score', 'vertices', 'perimeter_ratio'
    """
    if len(levels) < 3:
        return []

    necks = []
    perimeters = np.array([l['perimeter'] for l in levels])

    # Smooth perimeters to avoid noise
    kernel = np.ones(3) / 3
    smoothed = np.convolve(perimeters, kernel, mode='same')

    for i in range(2, len(smoothed)):
        prev = smoothed[i - 1]
        curr = smoothed[i]

        if prev < 1e-6:
            continue

        ratio = curr / prev

        # Explosion: perimeter increases by more than sensitivity factor
        if ratio > sensitivity:
            # The neck is at level i-1 (just before the explosion)
            neck_level = levels[i - 1]

            # Score based on how dramatic the explosion is
            score = min(1.0, (ratio - 1.0) / 3.0)

            necks.append({
                'distance': neck_level['distance'],
                'centroid': neck_level['centroid'],
                'radius': neck_level['radius'],
                'vertices': neck_level['vertices'],
                'score': float(score),
                'perimeter_ratio': float(ratio),
                'level_idx': i - 1,
            })

    return necks


def detect_joints_geodesic(mesh_obj, num_extremities=8, num_levels=60,
                           sensitivity=2.0):
    """Main entry point: detect joint locations using geodesic sweeping.

    Parameters
    ----------
    mesh_obj : bpy.types.Object
    num_extremities : int
        Number of limb tips to search for.
    num_levels : int
        Number of distance bands per sweep.
    sensitivity : float
        Perimeter explosion factor (2.0 = perimeter doubles).

    Returns
    -------
    list of dict, each with:
        world_pos    : Vector — neck centroid in world space
        world_normal : Vector — sweep direction (toward limb tip)
        radius       : float — approximate neck radius
        score        : float — confidence (0-1)
        limb_verts   : set of int — vertices on the limb side (closer to tip)
        extremity    : int — which extremity this neck belongs to
    """
    n_verts = len(mesh_obj.data.vertices)
    print(f"[VibeCut] Geodesic: {n_verts} verts, building adjacency...")

    adj, world_coords = _build_adjacency(mesh_obj)

    print(f"[VibeCut] Geodesic: finding {num_extremities} extremities...")
    extremities = _find_extremities(adj, n_verts, num_extremities)
    print(f"[VibeCut] Geodesic: found {len(extremities)} extremities")

    if len(extremities) < 2:
        print("[VibeCut] Geodesic: not enough extremities found")
        return []

    # Compute geodesic distance from the mesh centroid (approximated by
    # finding the vertex closest to the geometric center)
    geo_center = world_coords.mean(axis=0)
    center_vert = np.argmin(np.linalg.norm(world_coords - geo_center, axis=1))
    center_dist = _dijkstra(adj, center_vert, n_verts)

    all_necks = []

    for ei, ext_idx in enumerate(extremities):
        print(f"[VibeCut] Geodesic: sweeping from extremity {ei} (vertex {ext_idx})...")

        # Distance field from this extremity
        dist = _dijkstra(adj, ext_idx, n_verts)

        # Extract level sets (rings moving away from the tip)
        levels = _extract_level_sets(world_coords, dist, num_levels)

        if len(levels) < 5:
            print(f"[VibeCut]   extremity {ei}: too few levels ({len(levels)})")
            continue

        # Find perimeter explosions
        necks = _find_perimeter_explosions(levels, sensitivity)

        if not necks:
            print(f"[VibeCut]   extremity {ei}: no perimeter explosions found")
            continue

        # Take the first (closest to tip) explosion — that's the joint
        neck = necks[0]

        # Compute normal: direction from neck centroid toward the extremity
        ext_pos = world_coords[ext_idx]
        neck_centroid = neck['centroid']
        normal = ext_pos - neck_centroid
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-6:
            normal = normal / norm_len
        else:
            normal = np.array([0, 0, 1.0])

        # Limb vertices: all vertices closer to the extremity than the neck distance
        neck_dist = neck['distance']
        limb_verts = set(np.where(dist <= neck_dist)[0].tolist())

        print(f"[VibeCut]   extremity {ei}: neck at dist={neck_dist:.2f}, "
              f"r={neck['radius']:.2f}, score={neck['score']:.2f}, "
              f"ratio={neck['perimeter_ratio']:.1f}, "
              f"limb_verts={len(limb_verts)}")

        all_necks.append({
            'world_pos': Vector(neck_centroid.tolist()),
            'world_normal': Vector(normal.tolist()),
            'radius': neck['radius'],
            'score': neck['score'],
            'limb_verts': limb_verts,
            'extremity': ei,
        })

    # Deduplicate: if two extremities find the same neck (close centroids), keep the best
    if all_necks:
        all_necks.sort(key=lambda n: n['score'], reverse=True)
        filtered = []
        for neck in all_necks:
            pos = np.array(list(neck['world_pos']))
            too_close = False
            for existing in filtered:
                epos = np.array(list(existing['world_pos']))
                dist_between = np.linalg.norm(pos - epos)
                if dist_between < max(neck['radius'], existing['radius']) * 2:
                    too_close = True
                    break
            if not too_close:
                filtered.append(neck)
        all_necks = filtered
        print(f"[VibeCut] Geodesic: {len(all_necks)} joints after dedup")

    return all_necks
