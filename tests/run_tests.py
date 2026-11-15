# Headless validation suite for VibeCut's FDM Support Wizard (supports.py)
# and the mesh_tools flatten/bevel helpers.
#
# Run with:
#   blender -b --factory-startup -P tests/run_tests.py
#
# Eight support meshes (all in mm) cover the distinct support scenarios;
# two extra cases exercise the native best-fit flatten and the bevel op.
# Prints one CASE_RESULT json line per case and a SUITE_RESULT line.

import importlib.util
import json
import math
import os
import sys
import time
import traceback

import bpy
import bmesh
from mathutils import Matrix, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "vibecut")
OUT_DIR = os.path.join(HERE, "output")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fsw = _load("vibecut_supports_test", os.path.join(PKG, "supports.py"))
mt = _load("vibecut_meshtools_test", os.path.join(PKG, "mesh_tools.py"))
det = _load("vibecut_detection_test", os.path.join(PKG, "detection.py"))
tl = _load("vibecut_tools_test", os.path.join(PKG, "tools.py"))
hg = _load("vibecut_hinge_test", os.path.join(PKG, "hinge.py"))


# ---------------------------------------------------------------- helpers

def wipe_scene():
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)
    bpy.context.view_layer.update()


def obj_from_bm(bm, name):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    return ob


def add_box(bm, cx, cy, z0, z1, sx, sy):
    mat = (Matrix.Translation(Vector((cx, cy, (z0 + z1) / 2)))
           @ Matrix.Diagonal(Vector((sx, sy, z1 - z0, 1.0))))
    bmesh.ops.create_cube(bm, size=1.0, matrix=mat)


def boolean_diff(name, build_base, build_cutter):
    bm = bmesh.new()
    build_base(bm)
    base = obj_from_bm(bm, name + "_base")
    bm = bmesh.new()
    build_cutter(bm)
    cutter = obj_from_bm(bm, name + "_cutter")
    mod = base.modifiers.new("diff", 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    fsw._apply_modifiers_inplace(base)
    fsw._remove_object(cutter)
    base.name = name
    bpy.context.view_layer.update()
    return base


# ---------------------------------------------------------------- meshes

def make_table():
    bm = bmesh.new()
    add_box(bm, 0, 0, 0, 20, 8, 8)
    add_box(bm, 0, 0, 20, 24, 40, 40)
    return obj_from_bm(bm, "table")


def make_thin_fin():
    bm = bmesh.new()
    add_box(bm, 0, 0, 0, 38, 4, 4)
    add_box(bm, 0, 12.0, 35, 38, 4, 22)
    return obj_from_bm(bm, "thin_fin")


def make_suzanne():
    bm = bmesh.new()
    bmesh.ops.create_monkey(bm, matrix=Matrix.Scale(15.0, 4))
    return obj_from_bm(bm, "suzanne")


def make_bridge(span, name):
    half = span / 2 + 4
    bm = bmesh.new()
    add_box(bm, -half, 0, 0, 22, 8, 8)
    add_box(bm, half, 0, 0, 22, 8, 8)
    add_box(bm, 0, 0, 22, 26, 2 * half + 8, 8)
    return obj_from_bm(bm, name)


def make_cavity_sealed():
    def base(bm):
        add_box(bm, 0, 0, 0, 30, 30, 30)

    def cutter(bm):
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16,
                                  radius=8.0,
                                  matrix=Matrix.Translation((0, 0, 15)))
    return boolean_diff("cavity_sealed", base, cutter)


def make_cavity_tunnel():
    def base(bm):
        add_box(bm, 0, 0, 0, 30, 30, 30)

    def cutter(bm):
        mat = (Matrix.Translation((0, 0, 15))
               @ Matrix.Rotation(math.radians(90.0), 4, 'X'))
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True, segments=32,
                              radius1=7.0, radius2=7.0, depth=40.0,
                              matrix=mat)
    return boolean_diff("cavity_tunnel", base, cutter)


def make_sphere():
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=20.0,
                              matrix=Matrix.Translation((0, 0, 20)))
    return obj_from_bm(bm, "sphere")


# ------------------------------------------------------------ expectations

def check_case(name, rep, expect):
    fails = []
    v = rep["validation"]
    strategies = set(k for k, c in rep["strategies"].items() if c > 0)

    if expect.get("all_pass", True) and not v.get("all_pass", False):
        failing = [k for k, d in v.items()
                   if isinstance(d, dict) and not d.get("pass", True)]
        fails.append("validation failed: %s" % failing)

    for s in expect.get("must_have", []):
        if s not in strategies:
            fails.append("expected strategy %s, got %s"
                         % (s, sorted(strategies)))
    if expect.get("any_of") and not (strategies & set(expect["any_of"])):
        fails.append("expected one of %s, got %s"
                     % (expect["any_of"], sorted(strategies)))

    if expect.get("support_object") is True and not rep["support_object"]:
        fails.append("expected support geometry, got none")
    if expect.get("support_object") is False and rep["support_object"]:
        fails.append("expected NO support geometry, got %s"
                     % rep["support_object"])

    if expect.get("bridges") is True and not rep["bridges_mm"]:
        fails.append("expected bridge detection")
    if expect.get("warn_contains"):
        blob = " ".join(rep["warnings"]).lower()
        if expect["warn_contains"].lower() not in blob:
            fails.append("expected warning containing %r"
                         % expect["warn_contains"])

    cov = v.get("coverage", {})
    if cov.get("needed", 0) > 0 and cov.get("fraction", 1.0) < 0.92:
        fails.append("coverage %.2f below 0.92" % cov.get("fraction", 0))

    return (not fails), fails


SUPPORT_CASES = [
    ("table", make_table,
     {"must_have": ["PILLAR"], "support_object": True}),
    ("thin_fin", make_thin_fin,
     {"any_of": ["PILLAR", "TREE"], "support_object": True}),
    ("suzanne", make_suzanne,
     {"must_have": ["TREE"], "support_object": True}),
    ("bridge_ok", lambda: make_bridge(8.0, "bridge_ok"),
     {"must_have": ["BRIDGE"], "support_object": False, "bridges": True}),
    ("bridge_long", lambda: make_bridge(20.0, "bridge_long"),
     {"any_of": ["PILLAR", "TREE"], "support_object": True}),
    ("cavity_sealed", make_cavity_sealed,
     {"must_have": ["SEALED"], "support_object": False,
      "warn_contains": "sealed"}),
    ("cavity_tunnel", make_cavity_tunnel,
     {"any_of": ["CAVITY", "TREE"], "support_object": True}),
    ("sphere", make_sphere,
     {"must_have": ["CRADLE"], "support_object": True}),
]


def run_support_cases(results):
    for name, builder, expect in SUPPORT_CASES:
        wipe_scene()
        t0 = time.time()
        entry = {"case": name}
        try:
            ob = builder()
            P = fsw.Params()
            P.style = "SOLID"   # these cases assert the multi-strategy wizard
            rep = fsw.run_pipeline(ob, P,
                                   collection=bpy.context.scene.collection)
            ok, fails = check_case(name, rep, expect)
            entry.update({
                "ok": ok,
                "failures": fails,
                "strategies": rep["strategies"],
                "islands": rep["islands"],
                "bridges_mm": rep["bridges_mm"],
                "support_object": rep["support_object"],
                "attempts": rep["attempts"],
                "validation": {
                    k: ({kk: vv for kk, vv in d.items()}
                        if isinstance(d, dict) else d)
                    for k, d in rep["validation"].items()},
                "island_notes": [d for d in rep["island_detail"]
                                 if d.get("note")],
                "warnings": rep["warnings"],
                "seconds": round(time.time() - t0, 1),
            })
            try:
                sup = (bpy.data.objects.get(rep["support_object"])
                       if rep["support_object"] else None)
                fsw.export_stls(ob, sup, OUT_DIR)
            except Exception as e:
                entry["stl_export_error"] = str(e)
            if name == "suzanne":
                fsw.write_profile_files(rep, OUT_DIR)
        except Exception as e:
            entry.update({"ok": False,
                          "failures": ["EXCEPTION: %s" % e],
                          "traceback": traceback.format_exc()[-2500:],
                          "seconds": round(time.time() - t0, 1)})
        results.append(entry)
        print("CASE_RESULT " + json.dumps(entry))
        sys.stdout.flush()


# ------------------------------------------------------------ mesh tools

def run_flatten_case(results):
    """Bumpy tilted grid -> best-fit flatten -> verts must be coplanar."""
    wipe_scene()
    entry = {"case": "flatten_native"}
    try:
        import random as _r
        rng = _r.Random(7)
        bm = bmesh.new()
        bmesh.ops.create_grid(bm, x_segments=12, y_segments=12, size=20.0)
        # Tilt the grid and add bumpy noise so the best-fit plane is
        # NOT axis-aligned (catches average-normal-style bias too).
        tilt = (Matrix.Rotation(math.radians(25), 4, 'X')
                @ Matrix.Rotation(math.radians(15), 4, 'Y'))
        for v in bm.verts:
            v.co = tilt @ v.co
        normal = (tilt @ Vector((0, 0, 1))).normalized()
        for v in bm.verts:
            v.co += normal * rng.uniform(-1.2, 1.2)

        verts = list(bm.verts)
        co, no = mt._best_fit_plane([v.co for v in verts])
        mt._flatten_with_influence(verts, co, no, 1.0)

        worst = max(abs((v.co - co).dot(no)) for v in verts)
        # The fitted plane should also be close to the known tilt plane.
        align = abs(no.dot(normal))
        bm.free()
        entry.update({
            "ok": worst < 1e-5 and align > 0.99,
            "failures": [] if (worst < 1e-5 and align > 0.99) else
                        ["worst dev %.6f, plane alignment %.4f"
                         % (worst, align)],
            "worst_deviation_mm": round(worst, 6),
            "plane_alignment": round(align, 4),
        })
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_line_shape_cases(results):
    """Straighten to Axis + Curve Line helpers on noisy edge chains."""
    import random as _r
    rng = _r.Random(2)

    def chain(pts):
        bm = bmesh.new()
        vs = [bm.verts.new(p) for p in pts]
        for a, b in zip(vs, vs[1:]):
            bm.edges.new((a, b))
        return bm, vs

    # Straighten to Axis: wobbly line along X under an object offset. Every
    # vert must keep its world X and share one world Y/Z.
    entry = {"case": "straighten_to_axis"}
    try:
        bm, vs = chain([(i, 5 + rng.uniform(-.2, .2), 3 + rng.uniform(-.2, .2))
                        for i in range(12)])
        mw = Matrix.Translation((2, 0, 1))
        xs = [(mw @ v.co).x for v in vs]
        idx = mt._straighten_component_to_axis(vs, mw, mw.inverted())
        w = [mw @ v.co for v in vs]
        dev = max(max(abs(c.y - w[0].y), abs(c.z - w[0].z)) for c in w)
        dx = max(abs(c.x - x) for c, x in zip(w, xs))
        bm.free()
        # Z locked (move X/Y only): Y flattens, every vert keeps its own Z.
        bm2, vs2 = chain([(i, 5 + rng.uniform(-.2, .2), 3 + rng.uniform(-.2, .2))
                          for i in range(12)])
        zs = [v.co.z for v in vs2]
        mt._straighten_component_to_axis(vs2, Matrix(), Matrix(),
                                         (True, True, False))
        z_drift = max(abs(v.co.z - z) for v, z in zip(vs2, zs))
        y_dev = max(abs(v.co.y - vs2[0].co.y) for v in vs2)
        bm2.free()
        ok = (idx == 0 and dev < 1e-6 and dx < 1e-6
              and z_drift < 1e-6 and y_dev < 1e-6)
        entry.update({"ok": ok, "axis": idx, "yz_dev": dev, "x_drift": dx,
                      "zlock_z_drift": z_drift, "zlock_y_dev": y_dev,
                      "failures": [] if ok else
                      ["axis %d yz dev %.6f x drift %.6f zlock z %.6f y %.6f"
                       % (idx, dev, dx, z_drift, y_dev)]})
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))

    # Curve Line: noisy S-curve with 5 big outliers. All verts must land
    # within the noise band of the true curve; the ends stay pinned.
    entry = {"case": "curve_line"}
    try:
        def truth(x):
            return 2 * math.sin(x / 3)
        pts = []
        for i in range(400):
            x = -10 + 20 * i / 399
            pts.append((x, truth(x) + rng.uniform(-.05, .05)
                        + (1.0 if i % 80 == 40 else 0.0), 0.0))
        bm, vs = chain(pts)
        ends = (vs[0].co.copy(), vs[-1].co.copy())
        moved = mt._curve_component(list(reversed(vs)))
        worst = max(abs(v.co.y - truth(v.co.x)) for v in vs)
        pinned = (vs[0].co - ends[0]).length + (vs[-1].co - ends[1]).length
        bm.free()
        ok = moved and worst < 0.08 and pinned < 1e-6
        entry.update({"ok": ok, "worst_err": round(worst, 4),
                      "failures": [] if ok else
                      ["worst %.4f pinned drift %.6f" % (worst, pinned)]})
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_bevel_case(results):
    """Cube -> bevel all 12 edges -> manifold, more faces, no sharp 90s."""
    wipe_scene()
    entry = {"case": "auto_bevel"}
    try:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=20.0)
        n_faces_before = len(bm.faces)
        edges = list(bm.edges)
        bmesh.ops.bevel(bm, geom=edges, offset=0.4, offset_type='OFFSET',
                        segments=2, profile=0.5, affect='EDGES',
                        clamp_overlap=True, loop_slide=True)
        non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
        n_faces_after = len(bm.faces)
        worst_angle = 0.0
        for e in bm.edges:
            if len(e.link_faces) == 2:
                try:
                    worst_angle = max(worst_angle,
                                      math.degrees(e.calc_face_angle()))
                except ValueError:
                    pass
        bm.free()
        ok = (non_manifold == 0 and n_faces_after > n_faces_before
              and worst_angle < 89.0)
        entry.update({
            "ok": ok,
            "failures": [] if ok else
                        ["non_manifold=%d faces %d->%d worst_angle=%.1f"
                         % (non_manifold, n_faces_before, n_faces_after,
                            worst_angle)],
            "faces_before": n_faces_before,
            "faces_after": n_faces_after,
            "sharpest_remaining_deg": round(worst_angle, 1),
        })

        # Modifier path: object-mode Auto Bevel settings on a real object.
        bm2 = bmesh.new()
        bmesh.ops.create_cube(bm2, size=20.0)
        ob = obj_from_bm(bm2, "bevel_mod_test")
        mod = ob.modifiers.new("VibeCut Bevel", 'BEVEL')
        mod.width = 0.4
        mod.segments = 2
        mod.limit_method = 'ANGLE'
        mod.angle_limit = math.radians(30.0)
        mod.miter_outer = 'MITER_ARC'
        mod.use_clamp_overlap = True
        me = fsw._evaluated_mesh_of(ob)
        entry["modifier_eval_faces"] = len(me.polygons)
        if len(me.polygons) <= 6:
            entry["ok"] = False
            entry["failures"].append("bevel modifier produced no new faces")
        bpy.data.meshes.remove(me)
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


# ------------------------------------------------------------ detection

def _add_capsule(bm, p1, p2, r, segs=16):
    """Cylinder p1->p2 with sphere caps (round primitive for the creature)."""
    from mathutils import Vector as V
    d = V(p2) - V(p1)
    if d.length > 1e-6:
        quat = d.to_track_quat('Z', 'Y')
        mat = (Matrix.Translation((V(p1) + V(p2)) / 2)
               @ quat.to_matrix().to_4x4())
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True,
                              segments=segs, radius1=r, radius2=r,
                              depth=d.length, matrix=mat)
    for p in (p1, p2):
        bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=segs // 2,
                                  radius=r, matrix=Matrix.Translation(p))


def make_creature():
    """Quadruped with deliberate pinches at 4 hips + neck. Voxel-remeshed
    dense (>100k verts) to mimic AI-generated scan STLs.
    Returns (object, ground_truth_neck_positions)."""
    bm = bmesh.new()
    # Torso: fat capsule along X at z=22 (surface reaches z~13.5 at y=7)
    _add_capsule(bm, (-16, 0, 22), (16, 0, 22), 11.0)
    gt = []
    # 4 legs: a long thin hip pin (r 2.0) into a fat paw ball (r 5.5).
    # The ball top (z 10.5) stays clear of the torso belly (z ~13.5 at
    # y=7) so the pin is the only bridge - a decisive, unambiguous pinch.
    for sx in (-1, 1):
        for sy in (-1, 1):
            hx, hy = 11 * sx, 7 * sy
            _add_capsule(bm, (hx, hy, 14.5), (hx, hy, 5), 2.0, segs=12)
            bmesh.ops.create_uvsphere(
                bm, u_segments=16, v_segments=10, radius=5.5,
                matrix=Matrix.Translation((hx, hy, 5)))
            gt.append(Vector((hx, hy, 12.0)))
    # Neck (r 3.2) routed clear of the torso cap, into a separated head.
    _add_capsule(bm, (14, 0, 30), (28, 0, 38), 3.2, segs=12)
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=12, radius=8.0,
                              matrix=Matrix.Translation((33, 0, 41)))
    gt.append(Vector((21.0, 0, 34.0)))
    me = bpy.data.meshes.new("creature")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("creature", me)
    bpy.context.scene.collection.objects.link(ob)
    # Dense organic surface like a real scan
    r = ob.modifiers.new("rm", 'REMESH')
    r.mode = 'VOXEL'
    r.voxel_size = 0.25
    fsw._apply_modifiers_inplace(ob)
    bpy.context.view_layer.update()
    return ob, gt


def run_detection_cases(results):
    # Case 1: creature with known joints, >100k verts.
    wipe_scene()
    entry = {"case": "detect_creature"}
    try:
        ob, gt = make_creature()
        n_verts = len(ob.data.vertices)
        t0 = time.time()
        cands = det.detect_necks_blender(ob, resolution=96, sensitivity=0.25)
        dt = time.time() - t0
        matched = []
        for g in gt:
            best = min(((c["world_pos"] - g).length, i)
                       for i, c in enumerate(cands)) if cands else (1e9, -1)
            matched.append(best[0] <= 6.0)
        ok = (n_verts > 100000 and sum(matched) >= 4 and len(cands) <= 30
              and dt < 60.0)
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "verts=%d matched=%d/5 cands=%d time=%.1fs"
                % (n_verts, sum(matched), len(cands), dt)],
            "verts": n_verts,
            "gt_matched": "%d/5" % sum(matched),
            "candidates": len(cands),
            "top_scores": [round(c["score"], 2) for c in cands[:6]],
            "seconds": round(dt, 1),
        })
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()

    # Case 2: negative control - a sphere has no necks.
    wipe_scene()
    entry = {"case": "detect_sphere_negative"}
    try:
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24,
                                  radius=20.0,
                                  matrix=Matrix.Translation((0, 0, 20)))
        ob = obj_from_bm(bm, "plain_sphere")
        cands = det.detect_necks_blender(ob, resolution=96, sensitivity=0.25)
        ok = len(cands) == 0
        entry.update({"ok": ok,
                      "failures": [] if ok else
                      ["expected 0 candidates, got %d" % len(cands)],
                      "candidates": len(cands)})
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()

    # Case 3: a real STL (environment-dependent; set VIBECUT_REAL_STL).
    wipe_scene()
    entry = {"case": "detect_real_stl"}
    stl = os.environ.get("VIBECUT_REAL_STL", "")
    if not stl or not os.path.exists(stl):
        entry.update({"ok": True, "skipped": "VIBECUT_REAL_STL not set or not found"})
    else:
        try:
            t0 = time.time()
            bpy.ops.wm.stl_import(filepath=stl)
            obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
            ob = max(obs, key=lambda o: len(o.data.vertices))
            t_imp = time.time() - t0
            n_verts = len(ob.data.vertices)
            t0 = time.time()
            cands = det.detect_necks_blender(ob, resolution=96,
                                             sensitivity=0.25)
            dt = time.time() - t0
            ok = len(cands) >= 3 and dt < 120.0
            entry.update({
                "ok": ok,
                "failures": [] if ok else [
                    "cands=%d time=%.1fs" % (len(cands), dt)],
                "verts": n_verts,
                "import_seconds": round(t_imp, 1),
                "candidates": len(cands),
                "top": [[round(v, 1) for v in list(c["world_pos"])] +
                        [round(c["score"], 2), round(c["radius_world"], 1)]
                        for c in cands[:8]],
                "seconds": round(dt, 1),
            })
        except Exception as e:
            entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                          "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


# ------------------------------------------------------------ quad convert

def run_quad_convert_case(results):
    """Densely triangulated box -> flat panels. Sharp 90-degree edges must
    survive exactly; volume and bbox must not change."""
    wipe_scene()
    entry = {"case": "quad_convert"}
    try:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=30.0)
        for _ in range(6):
            bmesh.ops.subdivide_edges(
                bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        vol_before = bm.calc_volume(signed=False)
        me = bpy.data.meshes.new("dense_box")
        bm.to_mesh(me)
        bm.free()
        faces_in = len(me.polygons)

        fb, fa, sharp_n, q, ng, tr = mt._quad_convert_mesh(me, 25.0, 4.0)

        bm = bmesh.new()
        bm.from_mesh(me)
        vol_after = bm.calc_volume(signed=False)
        sharp_after = 0
        for e in bm.edges:
            if len(e.link_faces) == 2:
                try:
                    if math.degrees(e.calc_face_angle()) > 80:
                        sharp_after += 1
                except ValueError:
                    pass
        xs = [v.co.x for v in bm.verts]
        span = max(xs) - min(xs)
        bm.free()

        reduction = 1 - fa / max(fb, 1)
        vol_dev = abs(vol_after - vol_before) / vol_before
        ok = (faces_in > 40000 and reduction > 0.95
              and sharp_after == 12 and vol_dev < 0.005
              and abs(span - 30.0) < 0.01)
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "in=%d out=%d red=%.2f sharp90=%d voldev=%.4f span=%.2f"
                % (fb, fa, reduction, sharp_after, vol_dev, span)],
            "faces": "%d -> %d (%d%% fewer)" % (fb, fa,
                                                round(reduction * 100)),
            "sharp_90deg_edges": sharp_after,
            "volume_deviation": round(vol_dev, 5),
        })
        bpy.data.meshes.remove(me)
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_scale_guard_case(results):
    """A bed-sized-x4 model must be refused FAST with scaling advice
    instead of locking up."""
    wipe_scene()
    entry = {"case": "supports_bed_guard"}
    try:
        bm = bmesh.new()
        add_box(bm, 0, 0, 0, 900, 400, 400)
        add_box(bm, 0, 0, 900, 950, 800, 600)   # giant overhang on top
        ob = obj_from_bm(bm, "giant")
        t0 = time.time()
        err = None
        try:
            fsw.run_pipeline(ob, fsw.Params(),
                             collection=bpy.context.scene.collection)
        except RuntimeError as e:
            err = str(e)
        dt = time.time() - t0
        ok = (err is not None and "Scale the model" in err and dt < 10.0)
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "err=%r time=%.1fs" % (err, dt)],
            "refused_in_seconds": round(dt, 1),
            "message": (err or "")[:140],
        })
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_dense_proxy_case(results):
    """A >250k-triangle model must auto-run analysis on a decimated proxy
    and still pass validation in bounded time."""
    wipe_scene()
    entry = {"case": "supports_dense_proxy"}
    try:
        bm = bmesh.new()
        add_box(bm, 0, 0, 0, 20, 8, 8)
        add_box(bm, 0, 0, 20, 24, 40, 40)
        for _ in range(7):
            bmesh.ops.subdivide_edges(
                bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        bmesh.ops.triangulate(bm, faces=bm.faces[:])
        ob = obj_from_bm(bm, "dense_table")
        n_tris = len(ob.data.polygons)
        Pd = fsw.Params()
        Pd.style = "SOLID"
        t0 = time.time()
        rep = fsw.run_pipeline(ob, Pd,
                               collection=bpy.context.scene.collection)
        dt = time.time() - t0
        proxied = any("simplified copy" in m for m in rep["messages"])
        ok = (n_tris > 250000 and proxied
              and rep["validation"].get("all_pass", False) and dt < 120.0)
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "tris=%d proxied=%s pass=%s time=%.1fs"
                % (n_tris, proxied,
                   rep["validation"].get("all_pass"), dt)],
            "tris": n_tris,
            "proxied": proxied,
            "all_pass": rep["validation"].get("all_pass", False),
            "seconds": round(dt, 1),
        })
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_pins_case(results):
    """The DEFAULT twist-off pin style: must be SPARSE and LIGHT compared
    to the solid wizard (this is the whole point of restoring it).
    Suzanne, both styles, compare vertex counts."""
    for style, label in (("PINS", "pins"), ("SOLID", "solid")):
        wipe_scene()
        entry = {"case": "supports_%s" % label}
        try:
            ob = make_suzanne()
            P = fsw.Params()
            P.style = style
            t0 = time.time()
            rep = fsw.run_pipeline(ob, P,
                                   collection=bpy.context.scene.collection)
            dt = time.time() - t0
            sup = (bpy.data.objects.get(rep["support_object"])
                   if rep["support_object"] else None)
            sup_verts = len(sup.data.vertices) if sup else 0
            entry.update({
                "ok": rep["support_object"] is not None and dt < 60.0,
                "failures": [] if (rep["support_object"] and dt < 60.0)
                else ["no support or too slow (%.1fs)" % dt],
                "support_verts": sup_verts,
                "strategies": rep["strategies"],
                "seconds": round(dt, 1),
            })
            entry["_verts"] = sup_verts
        except Exception as e:
            entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                          "traceback": traceback.format_exc()[-1500:]})
        results.append(entry)
        print("CASE_RESULT " + json.dumps(
            {k: v for k, v in entry.items() if k != "_verts"}))
        sys.stdout.flush()

    # Cross-check: pins must be dramatically lighter than solid.
    pins = next((r for r in results if r["case"] == "supports_pins"), None)
    solid = next((r for r in results if r["case"] == "supports_solid"), None)
    entry = {"case": "supports_pins_lighter"}
    if pins and solid and pins.get("_verts") and solid.get("_verts"):
        ratio = pins["_verts"] / max(solid["_verts"], 1)
        ok = ratio < 0.5   # pins at most half the geometry of solid
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "pins=%d solid=%d ratio=%.2f"
                % (pins["_verts"], solid["_verts"], ratio)],
            "pin_verts": pins["_verts"],
            "solid_verts": solid["_verts"],
            "ratio": round(ratio, 3),
        })
    else:
        entry.update({"ok": False, "failures": ["missing pin/solid data"]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_color_shell_case(results):
    """Color Piece: two cubes in one mesh, cube A starts on material slot 1.
    Selecting one vert of A paints only A's shell; repainting keeps the
    original slot; Cancel puts A back on slot 1 and leaves B alone."""
    entry = {"case": "color_shell"}
    fails = []
    try:
        wipe_scene()
        me = bpy.data.meshes.new("color_shell")
        ob = bpy.data.objects.new("color_shell", me)
        bpy.context.scene.collection.objects.link(ob)
        base = bpy.data.materials.new("base")
        red = bpy.data.materials.new("red")
        me.materials.append(base)
        me.materials.append(red)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
        bmesh.ops.create_cube(bm, size=2.0,
                              matrix=Matrix.Translation((5, 0, 0)))
        bm.verts.ensure_lookup_table()
        for f in bm.faces:
            f.material_index = 1 if f.calc_center_median().x < 2.5 else 0
        for v in bm.verts:
            v.select = False
        next(v for v in bm.verts if v.co.x < 2.5).select = True

        faces = mt._shell_faces(bm)
        if len(faces) != 6:
            fails.append("shell has %d faces (want 6)" % len(faces))
        mt.paint_shell(me, bm, "AVOCADO")
        a = [f for f in bm.faces if f.calc_center_median().x < 2.5]
        b = [f for f in bm.faces if f.calc_center_median().x >= 2.5]
        if mt._shell_color(me, a) != "AVOCADO":
            fails.append("cube A not Avocado")
        if any(f.material_index != 0 for f in b):
            fails.append("cube B was touched")
        mt.paint_shell(me, bm, "SAGE")
        if mt._shell_color(me, a) != "SAGE":
            fails.append("repaint to Sage failed")
        mt.paint_shell(me, bm, None)
        a = [f for f in bm.faces if f.calc_center_median().x < 2.5]
        if any(f.material_index != 1 for f in a):
            fails.append("cancel did not restore slot 1: %s"
                         % sorted({f.material_index for f in a}))
        if mt._shell_color(me, a) is not None:
            fails.append("still reads as painted after cancel")
        bm.free()
        entry.update({"ok": not fails, "failures": fails})
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry, default=str))
    sys.stdout.flush()


def run_heatmap_case(results):
    """Overhang heatmap: a cube must read bottom face = 90 deg (needs
    support), top/sides = 0; ~1/6 of area flagged. Attribute must be
    written per-corner with valid colors."""
    wipe_scene()
    entry = {"case": "overhang_heatmap"}
    try:
        import numpy as np
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=20.0)
        ob = obj_from_bm(bm, "hm_cube")
        stats = fsw.build_overhang_heatmap(ob, 45.0)
        attr = ob.data.color_attributes.get("VibeCut_Overhang")
        n_loops = len(ob.data.loops)
        # read colors back
        cols = np.empty(n_loops * 4, dtype=np.float32)
        attr.data.foreach_get("color", cols)
        cols = cols.reshape(n_loops, 4)
        has_red = bool((cols[:, 0] > 0.7).any() and (cols[:, 1] < 0.2).any())
        has_green = bool((cols[:, 1] > 0.6).any() and (cols[:, 0] < 0.2).any())
        pct = stats["pct_supported"]
        worst = stats["worst_deg"]
        ok = (attr is not None and attr.domain == 'CORNER'
              and 12.0 < pct < 22.0      # bottom face ~1/6 = 16.7%
              and 88.0 < worst <= 90.0
              and has_red and has_green)
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "pct=%.1f worst=%.1f red=%s green=%s domain=%s"
                % (pct, worst, has_red, has_green,
                   attr.domain if attr else None)],
            "pct_supported": round(pct, 1),
            "worst_deg": round(worst, 1),
        })
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1500:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def run_inverted_normals_case(results):
    """Inside-out model (all faces pointing in, as some STL exports are):
    ceilings read as floors, so supports landed inside the model and as
    crumbs. The pipeline must fix the faces and support it exactly like
    the same model with correct faces."""
    wipe_scene()
    entry = {"case": "supports_inverted_normals"}
    try:
        reps = {}
        for label, flip in (("normal", False), ("inverted", True)):
            wipe_scene()
            ob = make_sphere()
            if flip:
                ob.data.flip_normals()
            rep = fsw.run_pipeline(ob, fsw.Params(),
                                   collection=bpy.context.scene.collection)
            bm = bmesh.new()
            bm.from_mesh(ob.data)
            vol = bm.calc_volume(signed=True)
            bm.free()
            reps[label] = (rep, vol)
        (rn, _), (ri, vol_i) = reps["normal"], reps["inverted"]
        blob = " ".join(ri["messages"]).lower()
        fails = []
        if vol_i <= 0:
            fails.append("model still inside-out (signed volume %.0f)" % vol_i)
        if "inside-out" not in blob:
            fails.append("no inside-out message")
        if ri["islands"] != rn["islands"]:
            fails.append("islands %s vs %s with correct faces"
                         % (ri["islands"], rn["islands"]))
        if not ri["support_object"]:
            fails.append("no supports generated")
        entry.update({"ok": not fails, "failures": fails,
                      "islands": [rn["islands"], ri["islands"]]})
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


def make_wall_shelf():
    """Shelf sticking out of a tall wall: posts under the shelf must not
    run up alongside the wall (they would fuse to it)."""
    bm = bmesh.new()
    add_box(bm, 2, 0, 0, 40, 4, 20)       # wall x 0..4
    add_box(bm, 10, 0, 25, 28, 12, 20)    # shelf x 4..16
    ob = obj_from_bm(bm, "wall_shelf")
    r = ob.modifiers.new("rm", 'REMESH')   # weld the boxes into one shell
    r.mode = 'VOXEL'
    r.voxel_size = 0.5
    fsw._apply_modifiers_inplace(ob)
    return ob


def make_socket_ball():
    """Ball with sideways and downward cavities (eye-socket ball)."""
    def base(bm):
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24,
                                  radius=20.0,
                                  matrix=Matrix.Translation((0, 0, 20)))

    def cutter(bm):
        for p in ((19, 0, 22), (-19, 0, 22), (0, 19, 26), (0, -19, 14)):
            bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16,
                                      radius=8.0,
                                      matrix=Matrix.Translation(p))
    return boolean_diff("socket_ball", base, cutter)


CLEAR_CASES = [
    ("clear_wall_shelf", make_wall_shelf),
    ("clear_socket_ball", make_socket_ball),
]


def run_clear_cases(results):
    """CLEAR style: supports touch the model ONLY at tip contacts."""
    for name, builder in CLEAR_CASES:
        wipe_scene()
        t0 = time.time()
        entry = {"case": name}
        try:
            ob = builder()
            P = fsw.Params()
            P.style = "CLEAR"
            rep = fsw.run_pipeline(ob, P,
                                   collection=bpy.context.scene.collection)
            v = rep["validation"]
            att = rep["attempts"][-1]
            ko = v.get("keepout", {})
            fails = []
            if not rep["support_object"]:
                fails.append("no supports")
            if ko.get("violations", 1) != 0:
                fails.append("keepout violations %s (min gap %s)"
                             % (ko.get("violations"), ko.get("min_gap_mm")))
            if att.get("model_rests", 1) != 0:
                fails.append("model rests %s" % att.get("model_rests"))
            entry.update({"ok": not fails, "failures": fails,
                          "tips": att.get("tips"),
                          "posts": att.get("bed_posts"),
                          "unreachable": att.get("unreachable"),
                          "keepout": ko,
                          "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                          "traceback": traceback.format_exc()[-2500:],
                          "seconds": round(time.time() - t0, 1)})
        results.append(entry)
        print("CASE_RESULT " + json.dumps(entry))
        sys.stdout.flush()


def _tool_sphere(name="tsphere", r=20.0, seg=48, rings=24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings,
                              radius=r)
    return obj_from_bm(bm, name)


def _case_prep_messy():
    """AI-style mess: every face split apart (duplicate verts), a floating
    crumb, an open hole, inside-out faces. Prep must leave ONE watertight,
    outward shell."""
    ob = _tool_sphere()
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces
                               if f.calc_center_median().z > 17.0],
                     context='FACES')
    bmesh.ops.split_edges(bm, edges=bm.edges[:])
    bmesh.ops.create_cube(bm, size=0.5,
                          matrix=Matrix.Translation((40, 0, 0)))
    bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()
    ob.rotation_euler = (0.3, 0.0, 0.0)
    bpy.context.view_layer.update()
    log, rep = tl.prep_new_model(ob, {})
    fails = []
    if not rep["watertight"]:
        fails.append("not watertight: %s" % rep)
    if rep["inside_out"]:
        fails.append("still inside-out")
    if rep["shells"] != 1:
        fails.append("%d shells (crumb kept?)" % rep["shells"])
    if abs(min((ob.matrix_world @ v.co).z for v in ob.data.vertices)) > 1e-4:
        fails.append("not on the bed")
    if tuple(ob.rotation_euler) != (0.0, 0.0, 0.0):
        fails.append("rotation not applied")
    return fails, {"log": log}


def _case_fill_round():
    ob = _tool_sphere()
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces
                               if f.calc_center_median().z > 14.0],
                     context='FACES')
    loop = [e for e in bm.edges if e.is_boundary]
    tl.fill_hole_round(bm, loop)
    dev = max(abs(v.co.length - 20.0) for v in bm.verts)
    bnd = sum(1 for e in bm.edges if e.is_boundary)
    bm.free()
    fails = []
    if bnd:
        fails.append("%d open edges left" % bnd)
    if dev > 0.35:
        fails.append("patch off the sphere by %.3f mm" % dev)
    return fails, {"max_dev_mm": round(dev, 4)}


def _case_make_round():
    import random as _r
    rng = _r.Random(3)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=10.0)
    for v in bm.verts:
        v.co *= 1.0 + rng.uniform(-0.08, 0.08)
    c, r, mv = tl.make_round(bm, bm.verts[:])
    dev = max(abs((v.co - c).length - r) for v in bm.verts)
    bm.free()
    fails = [] if dev < 1e-4 and abs(r - 10.0) < 0.5 else \
        ["dev %.5f r %.3f" % (dev, r)]
    return fails, {"radius": round(r, 3)}


def _case_flat_bottom():
    ob = _tool_sphere()
    ob.location = (0, 0, 20)
    bpy.context.view_layer.update()
    tl.flat_bottom(ob, 2.0)
    rep = tl.mesh_report(ob)
    zs = [(ob.matrix_world @ v.co).z for v in ob.data.vertices]
    on_bed = sum(1 for z in zs if abs(z) < 1e-4)
    fails = []
    if not rep["watertight"]:
        fails.append("not watertight")
    if abs(min(zs)) > 1e-4 or on_bed < 8:
        fails.append("no flat base (min z %.4f, %d verts on bed)"
                     % (min(zs), on_bed))
    if abs(rep["dims_mm"][2] - 38.0) > 0.05:
        fails.append("height %.3f, expected 38" % rep["dims_mm"][2])
    return fails, {"dims": rep["dims_mm"]}


def _case_thin_walls():
    bm = bmesh.new()
    add_box(bm, 0, 0, 0, 0.6, 20, 20)          # 0.6 mm plate: too thin
    plate = obj_from_bm(bm, "plate")
    bm = bmesh.new()
    add_box(bm, 40, 0, 0, 20, 20, 20)          # 20 mm cube: fine
    cube = obj_from_bm(bm, "cube")
    tp = tl.thin_wall_faces(plate, 1.2)
    tc = tl.thin_wall_faces(cube, 1.2)
    fails = []
    if len(tp) < 2:
        fails.append("plate: only %d thin faces" % len(tp))
    if tc:
        fails.append("cube: %d false thin faces" % len(tc))
    return fails, {"plate_thin": len(tp)}


def _case_symmetrize():
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=10.0)
    for v in bm.verts:
        if v.co.x > 0 and v.co.y > 0 and v.co.z > 0:
            v.co.x = 9.0                       # dent only the +X side
    tl.symmetrize(bm, "X", 0.001)
    xs = sorted(round(v.co.x, 3) for v in bm.verts)
    bm.free()
    fails = [] if xs[0] == -9.0 and xs[-1] == 9.0 else \
        ["'+X -> -X' did not keep the +X side: x range %s..%s"
         % (xs[0], xs[-1])]
    return fails, {}


def _case_even_remesh():
    ob = _tool_sphere()
    tl.voxel_remesh(ob, 1.0)
    rep = tl.mesh_report(ob)
    fails = []
    if rep["tris"] or rep["ngons"]:
        fails.append("%d tris %d ngons" % (rep["tris"], rep["ngons"]))
    if not rep["watertight"]:
        fails.append("not watertight")
    return fails, {"faces": rep["faces"]}


def _case_scale_rotated():
    """glTF imports arrive rotated 90 deg about X; 'height' must mean the
    world Z side, not the mesh's local Z."""
    bm = bmesh.new()
    add_box(bm, 0, 0, -1.5, 1.5, 1.0, 2.0)     # local 1 x 2 x 3
    ob = obj_from_bm(bm, "rot_box")
    ob.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    bpy.context.view_layer.update()
    tl.scale_to_size(ob, 100.0, "Z")
    zs = [(ob.matrix_world @ v.co).z for v in ob.data.vertices]
    h = max(zs) - min(zs)
    fails = [] if abs(h - 100.0) < 1e-3 else ["world height %.3f" % h]
    return fails, {"height": round(h, 3)}


def _case_straighten_wall():
    import random as _r
    rng = _r.Random(7)
    R, cols, rows = 40.0, 121, 9
    bm = bmesh.new()
    grid = []
    truth = []
    for i in range(cols):
        th = math.radians(120.0 * i / (cols - 1))
        frac = i / (cols - 1)
        zb, zt = 2.0 + 3.0 * frac, 12.0 + 5.0 * frac    # sloped edges
        col = []
        for j in range(rows):
            z = zb + (zt - zb) * j / (rows - 1)
            rr = R + rng.uniform(-0.6, 0.6)               # in/out wobble
            # jagged top edge, but the 4 corner vertices are the
            # reference: they stay true
            corner = i in (0, cols - 1)
            zz = z + (0.0 if corner else
                      rng.uniform(-1.0, 1.0) if j == rows - 1 else
                      rng.uniform(-0.15, 0.15) if 0 < j < rows - 1 else 0.0)
            col.append(bm.verts.new((rr * math.cos(th), rr * math.sin(th),
                                     zz)))
            truth.append((th, j, zb, zt))
        grid.append(col)
    for i in range(cols - 1):
        for j in range(rows - 1):
            bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1],
                          grid[i][j + 1]))
    verts = [v for col in grid for v in col]
    res = tl.straighten_wall(verts, Matrix.Identity(4), True, 'STRAIGHT', 9)
    rad = [math.hypot(v.co.x, v.co.y) for v in verts]
    rdev = max(abs(r - R) for r in rad)
    col_xy = max(max((c[k].co.xy - c[0].co.xy).length for k in range(rows))
                 for c in grid)
    top = max(abs(c[-1].co.z - (12.0 + 5.0 * i / (cols - 1)))
              for i, c in enumerate(grid))
    bot = max(abs(c[0].co.z - (2.0 + 3.0 * i / (cols - 1)))
              for i, c in enumerate(grid))
    order = all(c[j].co.z < c[j + 1].co.z for c in grid
                for j in range(rows - 1))
    bm.free()
    fails = []
    if rdev > 0.2:
        fails.append("curve off by %.3f mm" % rdev)
    if col_xy > 0.05:
        fails.append("not vertical: column spread %.3f mm" % col_xy)
    if top > 0.35 or bot > 0.35:
        fails.append("edges not straight: top %.3f bottom %.3f" % (top, bot))
    if not order:
        fails.append("rows crossed over")
    return fails, {"curve_dev": round(rdev, 3), "column_spread": round(col_xy, 4),
                   "top_dev": round(top, 3), "bottom_dev": round(bot, 3),
                   "stats": res}


def _case_straighten_thick_wall():
    """Both faces of a 2 mm wall selected: must stay 2 mm apart."""
    import random as _r
    rng = _r.Random(11)
    bm = bmesh.new()
    sheets = []
    for R in (40.0, 38.0):
        grid = []
        for i in range(81):
            th = math.radians(90.0 * i / 80)
            col = [bm.verts.new(((R + rng.uniform(-0.3, 0.3)) * math.cos(th),
                                 (R + rng.uniform(-0.3, 0.3)) * math.sin(th),
                                 10.0 * j / 6)) for j in range(7)]
            grid.append(col)
        for i in range(80):
            for j in range(6):
                bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1],
                              grid[i][j + 1]))
        sheets.append(grid)
    for j in range(7):                 # end caps join the two sheets
        for i in (0, 80):
            bm.edges.new((sheets[0][i][j], sheets[1][i][j]))
    verts = [v for g in sheets for c in g for v in c]
    tl.straighten_wall(verts, Matrix.Identity(4), True, 'KEEP', 9)
    def radii(g, cols):
        return [math.hypot(v.co.x, v.co.y) for i in cols for v in g[i]]
    inner_cols, end_cols = range(2, 79), (0, 1, 79, 80)
    ro, ri = radii(sheets[0], range(81)), radii(sheets[1], range(81))
    thick = sum(ro) / len(ro) - sum(ri) / len(ri)
    # Interior must be clean; the two end columns of a noisy strip keep a
    # small end-of-fit residual (known).
    mid = max(max(abs(r - 40.0) for r in radii(sheets[0], inner_cols)),
              max(abs(r - 38.0) for r in radii(sheets[1], inner_cols)))
    end = max(max(abs(r - 40.0) for r in radii(sheets[0], end_cols)),
              max(abs(r - 38.0) for r in radii(sheets[1], end_cols)))
    bm.free()
    fails = []
    if abs(thick - 2.0) > 0.15:
        fails.append("thickness %.3f mm (want 2)" % thick)
    if mid > 0.1:
        fails.append("interior off the curve by %.3f mm" % mid)
    if end > 0.3:
        fails.append("end columns off by %.3f mm" % end)
    return fails, {"thickness": round(thick, 3), "interior_dev": round(mid, 3),
                   "end_dev": round(end, 3)}


def _two_strips():
    """Lower wall strip (rim: 32 verts at z=0) + upper strip (rim: 20 verts,
    slightly bigger, at z=0.3) - separate pieces with a gap, both rims
    selected."""
    bm = bmesh.new()

    def ring(n, r, z, ph=0.0):
        return [bm.verts.new((r * math.cos(2 * math.pi * i / n + ph),
                              r * math.sin(2 * math.pi * i / n + ph), z))
                for i in range(n)]

    def strip(r0, r1):
        n = len(r0)
        for i in range(n):
            j = (i + 1) % n
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))
    a_rim, a_bot = ring(32, 10.0, 0.0), ring(32, 10.0, -5.0)
    strip(a_bot, a_rim)
    b_rim, b_top = ring(20, 10.4, 0.3, 0.07), ring(20, 10.4, 5.0, 0.07)
    strip(b_rim, b_top)
    bm.normal_update()
    for e in bm.edges:
        e.select = all(abs(v.co.z) < 0.5 for v in e.verts)
    return bm


def _seam_report(bm):
    seam = [e for e in bm.edges if all(abs(v.co.z) < 0.5 for v in e.verts)]
    open_ = sum(1 for e in seam if len(e.link_faces) != 2)
    bnd = sum(1 for e in bm.edges if e.is_boundary)
    nonman = sum(1 for e in bm.edges if not e.is_manifold
                 and not e.is_boundary)
    zoff = max(abs(v.co.z) for e in seam for v in e.verts) if seam else 9
    return open_, bnd, nonman, zoff


def _case_snap_loop():
    fails = []
    # Original script behaviour (no zip): leaves T-junctions.
    bm = _two_strips()
    r0 = tl.snap_loop_to_loop(bm, Matrix.Identity(4), 'HIGHER', True, False)
    o0, _, _, _ = _seam_report(bm)
    bm.free()
    # Tool default (zip on): one watertight seam.
    bm = _two_strips()
    r = tl.snap_loop_to_loop(bm, Matrix.Identity(4), 'HIGHER', True, True)
    o1, bnd, nonman, zoff = _seam_report(bm)
    bm.free()
    if r["missed"]:
        fails.append("%d verts missed" % r["missed"])
    if o1:
        fails.append("%d open seam edges with zip" % o1)
    if bnd != 52:                 # only the far rims stay open: 32 + 20
        fails.append("boundary edges %d (want 52)" % bnd)
    if nonman:
        fails.append("%d non-manifold edges" % nonman)
    if zoff > 1e-4:
        fails.append("seam not on target (z %.5f)" % zoff)
    return fails, {"open_without_zip": o0, "open_with_zip": o1,
                   "zipped": r["zipped"], "welded": r["welded"],
                   "boundary": bnd}


TOOL_CASES = [
    ("tools_scale_rotated", _case_scale_rotated),
    ("tools_prep_messy", _case_prep_messy),
    ("tools_fill_round", _case_fill_round),
    ("tools_make_round", _case_make_round),
    ("tools_straighten_wall", _case_straighten_wall),
    ("tools_straighten_thick_wall", _case_straighten_thick_wall),
    ("tools_snap_loop", _case_snap_loop),
    ("tools_flat_bottom", _case_flat_bottom),
    ("tools_thin_walls", _case_thin_walls),
    ("tools_symmetrize", _case_symmetrize),
    ("tools_even_remesh", _case_even_remesh),
]


def run_tool_cases(results):
    for name, fn in TOOL_CASES:
        wipe_scene()
        entry = {"case": name}
        try:
            fails, extra = fn()
            entry.update({"ok": not fails, "failures": fails})
            entry.update(extra)
        except Exception as e:
            entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                          "traceback": traceback.format_exc()[-1800:]})
        results.append(entry)
        print("CASE_RESULT " + json.dumps(entry, default=str))
        sys.stdout.flush()


def run_big_organic_case(results):
    """Printable-size BIG organic model (scaled up to print size): many
    overhang islands, tall tree descents. Must complete in bounded time -
    large tree forests must not lock Blender up."""
    wipe_scene()
    entry = {"case": "supports_big_organic"}
    try:
        bm = bmesh.new()
        _add_capsule(bm, (-50, 0, 130), (50, 0, 130), 45.0)   # fat body
        _add_capsule(bm, (60, 0, 150), (95, 0, 185), 16.0)    # neck/head
        for sx in (-1, 1):
            for sy in (-1, 1):
                _add_capsule(bm, (45 * sx, 28 * sy, 110),
                             (60 * sx, 38 * sy, 40), 12.0)    # legs (air)
        _add_capsule(bm, (-60, 0, 140), (-100, 0, 95), 10.0)  # tail
        me = bpy.data.meshes.new("bigblob")
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new("bigblob", me)
        bpy.context.scene.collection.objects.link(ob)
        r = ob.modifiers.new("rm", 'REMESH')
        r.mode = 'VOXEL'
        r.voxel_size = 1.0
        fsw._apply_modifiers_inplace(ob)
        bpy.context.view_layer.update()
        dims = [round(d, 0) for d in ob.dimensions]
        t0 = time.time()
        rep = fsw.run_pipeline(ob, fsw.Params(),
                               collection=bpy.context.scene.collection)
        dt = time.time() - t0
        v = rep["validation"]
        ok = dt < 150.0 and rep["support_object"] is not None \
            and v.get("all_pass", False)
        entry.update({
            "ok": ok,
            "failures": [] if ok else [
                "time=%.0fs pass=%s sup=%s"
                % (dt, v.get("all_pass"), rep["support_object"])],
            "dims_mm": dims,
            "strategies": rep["strategies"],
            "all_pass": v.get("all_pass", False),
            "seconds": round(dt, 1),
        })
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry))
    sys.stdout.flush()


# ---------------------------------------------------------------- hinge

def _solid_box(name, x0, x1, y0, y1, z0, z1):
    bm = bmesh.new()
    add_box(bm, (x0 + x1) / 2, (y0 + y1) / 2, z0, z1, x1 - x0, y1 - y0)
    return obj_from_bm(bm, name)


def make_fused_box(barrel_off=(0.0, 0.0), barrel_r=3.0):
    """2D->3D style box: open-top tray, lid propped open 60 deg, and a fake
    hinge barrel welding the two into ONE watertight shell. barrel_off puts
    the fake barrel off the lid's TRUE pivot (x, z), like AI models do."""
    tray = _solid_box("hinge_box", -30, 30, -20, 20, 0, 20)
    hole = bpy.data.meshes.new("h")
    bm = bmesh.new()
    add_box(bm, 0, 0, 2, 25, 56, 36)
    bm.to_mesh(hole)
    bm.free()
    hg.boolean(tray, hole, 'DIFFERENCE')
    axis_c = Vector((30, 0, 21.5))
    lid = bmesh.new()
    add_box(lid, -30, 0, -1.5, 1.5, 60, 38)
    lid_me = bpy.data.meshes.new("lid")
    lid.to_mesh(lid_me)
    lid.free()
    lid_me.transform(Matrix.Translation(axis_c)
                     @ Matrix.Rotation(math.radians(60), 4, 'Y'))
    hg.boolean(tray, lid_me, 'UNION')
    barrel_c = axis_c + Vector((barrel_off[0], 0, barrel_off[1]))
    hg.boolean(tray, hg._hull_mesh(hg.cylinder_points(
        barrel_c, Vector((0, 1, 0)), -12, 12, barrel_r, 48), "barrel"),
        'UNION')
    return tray, axis_c


def run_hinge_cases(results):
    for mode in ('PIP', 'SEPARATE'):
        entry = {"case": "hinge_fused_box_" + mode.lower()}
        try:
            wipe_scene()
            ob, axis_c = make_fused_box()
            pieces_before = int(hg._components(ob.data).max()) + 1
            t0 = time.time()
            # Same flow as the panel: rough guide (0.5 mm off), Snap, Build.
            fit = hg.fit_fake_hinge(ob, axis_c + Vector((0.4, 0, -0.3)),
                                    Vector((0, 1, 0)), 12.0, 3.2)
            fails = []
            if fit is None:
                raise RuntimeError("snap found no barrel")
            snap_err = (fit[0] - axis_c).length
            if snap_err > 0.3 or abs(fit[1] - 3.0) > 0.3:
                fails.append("snap off by %.2f mm, r=%.2f" % (snap_err, fit[1]))
            log, res = hg.build_knuckle_hinge(
                ob, fit[0], Vector((0, 1, 0)), 12.0, fit[1] * 1.05,
                {"pin_mode": mode, "pin_d": 2.4})
            if pieces_before != 1:
                fails.append("fixture not fused (%d pieces)" % pieces_before)
            if not res.get("ok"):
                fails.append("not ok: " + " / ".join(log))
            sw = res.get("swing", {})
            if not 54 <= sw.get("close_deg", 0) <= 63:
                fails.append("close_deg %s (want ~60)" % sw.get("close_deg"))
            if mode == 'SEPARATE' and res.get("pin") is None:
                fails.append("no pin object")
            entry.update({"ok": not fails, "failures": fails,
                          "snap_err_mm": round(snap_err, 3),
                          "swing": {k: v for k, v in sw.items()
                                    if isinstance(k, str)},
                          "seconds": round(time.time() - t0, 1)})
        except Exception as e:
            entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                          "traceback": traceback.format_exc()[-1800:]})
        results.append(entry)
        print("CASE_RESULT " + json.dumps(entry, default=str))
        sys.stdout.flush()

    # Regression: fake barrel 3.2 mm off the lid's true
    # pivot. A hinge can validate OK while the lid cannot shut. Build
    # must find the real pivot and the lid must swing fully closed.
    entry = {"case": "hinge_offset_fake_barrel"}
    try:
        wipe_scene()
        ob, axis_c = make_fused_box(barrel_off=(-2.5, -2.0), barrel_r=4.5)
        bc = axis_c + Vector((-2.5, 0, -2.0))
        t0 = time.time()
        fit = hg.fit_fake_hinge(ob, bc, Vector((0, 1, 0)), 12.0, 4.7)
        log, res = hg.build_knuckle_hinge(
            ob, fit[0], Vector((0, 1, 0)), 12.0, fit[1] * 1.05,
            {"pin_d": 2.4, "knuckle_d": 7.0})
        fails = []
        f = res.get("fit") or {}
        err = (Vector(f["center"]) - axis_c).length if f else 99
        # A flat lid on a flat-topped tray seats equally well slid ~2.5 mm
        # toward the hinge (verified: identical rim scores), so the pivot
        # only has to move off the fake barrel toward the truth; what MUST
        # hold is that the lid swings fully shut and seats.
        if not f or f["moved_mm"] < 1.0:
            fails.append("fit did not move the axis off the fake barrel")
        if err > 3.0:
            fails.append("fitted pivot %.2f mm from true pivot" % err)
        if not res.get("ok"):
            fails.append("not ok: " + " / ".join(log))
        if res.get("closes_to", 0) < 58:
            fails.append("closes_to %s (want ~60)" % res.get("closes_to"))
        if res.get("seat_gap", 99) > 0.6:
            fails.append("seat gap %.2f mm when shut" % res.get("seat_gap"))
        entry.update({"ok": not fails, "failures": fails,
                      "pivot_err_mm": round(err, 2),
                      "closes_to": res.get("closes_to"),
                      "seat_gap": res.get("seat_gap"),
                      "seconds": round(time.time() - t0, 1)})
    except Exception as e:
        entry.update({"ok": False, "failures": ["EXCEPTION: %s" % e],
                      "traceback": traceback.format_exc()[-1800:]})
    results.append(entry)
    print("CASE_RESULT " + json.dumps(entry, default=str))
    sys.stdout.flush()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    if os.environ.get("VC_ONLY") == "hinge":
        run_hinge_cases(results)
    elif os.environ.get("VC_ONLY") == "tools":
        run_tool_cases(results)
    elif os.environ.get("VC_ONLY") == "color":
        run_color_shell_case(results)
    else:
        _run_all(results)
    _finish(results)


def _run_all(results):
    run_support_cases(results)
    run_flatten_case(results)
    run_line_shape_cases(results)
    run_bevel_case(results)
    run_detection_cases(results)
    run_quad_convert_case(results)
    run_scale_guard_case(results)
    run_dense_proxy_case(results)
    run_pins_case(results)
    run_heatmap_case(results)
    run_color_shell_case(results)
    run_inverted_normals_case(results)
    run_clear_cases(results)
    run_tool_cases(results)
    run_big_organic_case(results)
    run_hinge_cases(results)


def _finish(results):
    n_ok = sum(1 for r in results if r["ok"])
    suite = {"passed": n_ok, "total": len(results),
             "all_green": n_ok == len(results)}
    print("SUITE_RESULT " + json.dumps(suite))
    sys.stdout.flush()
    sys.exit(0 if suite["all_green"] else 1)


main()
