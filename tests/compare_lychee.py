import bpy, bmesh, sys, os, json, time, importlib.util
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLE = os.path.join(os.path.dirname(ROOT), "Example")
spec = importlib.util.spec_from_file_location(
    "vc_supports", os.path.join(ROOT, "vibecut", "supports.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

CONTACT_TOL = 0.3
CLUSTER_MM = 2.0
FOOT_MM = 3.0
MATCH_MM = 3.0

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
SAVE = "--save" in argv
POSES = [int(a) for a in argv if a.isdigit()] or [1, 2, 3, 4]
OVERRIDES = [a for a in argv if '=' in a]


def split_largest(ob):
    me = ob.data
    n = len(me.vertices)
    ev = np.empty(len(me.edges) * 2, dtype=np.int64)
    me.edges.foreach_get("vertices", ev)
    ev = ev.reshape(-1, 2)
    parent = np.arange(n)

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r
    for a, b in ev:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    uniq, counts = np.unique(roots, return_counts=True)
    keep_root = uniq[np.argmax(counts)]
    is_model = roots == keep_root

    def sub(mask, name):
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[bm.verts[i] for i in np.nonzero(~mask)[0]],
                         context='VERTS')
        m = bpy.data.meshes.new(name)
        bm.to_mesh(m)
        bm.free()
        o = bpy.data.objects.new(name, m)
        bpy.context.scene.collection.objects.link(o)
        return o
    return sub(is_model, "model"), sub(~is_model, "lychee_sup"), len(uniq) - 1


def model_bvh(model):
    me = S.extract_world_mesh(model)
    bvh = S.build_bvh(me)
    return me, bvh


def measure(sup, bvh):
    me = sup.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = sup.matrix_world
    tips, rests = [], []
    for p in co:
        v = mw @ Vector(p)
        hit = bvh.find_nearest(v)
        if hit[0] is None or hit[3] > CONTACT_TOL:
            continue
        (tips if hit[1].z < 0 else rests).append(hit[0])

    def cluster(pts):
        if not pts:
            return []
        kd = KDTree(len(pts))
        for i, p in enumerate(pts):
            kd.insert(p, i)
        kd.balance()
        dead = set()
        out = []
        for i, p in enumerate(pts):
            if i in dead:
                continue
            out.append(p)
            for _c, j, _d in kd.find_range(p, CLUSTER_MM):
                dead.add(j)
        return out
    tips, rests = cluster(tips), cluster(rests)
    if tips and rests:
        kd = KDTree(len(tips))
        for i, p in enumerate(tips):
            kd.insert(p, i)
        kd.balance()
        rests = [p for p in rests if kd.find(p)[2] > 1.5]
    wz = np.array([(mw @ Vector(p)).z for p in co]) if len(co) else np.zeros(0)
    zmin = wz.min() if len(wz) else 0.0
    floor = [Vector((p[0], p[1], 0.0)) for p, z in zip(co, wz) if z < zmin + 0.2]
    feet = len(cluster_xy(floor, FOOT_MM))
    return tips, rests, feet


def cluster_xy(pts, r):
    if not pts:
        return []
    kd = KDTree(len(pts))
    for i, p in enumerate(pts):
        kd.insert(p, i)
    kd.balance()
    dead, out = set(), []
    for i, p in enumerate(pts):
        if i in dead:
            continue
        out.append(p)
        for _c, j, _d in kd.find_range(p, r):
            dead.add(j)
    return out


def match(a, b, tol):
    if not a or not b:
        return 0.0
    kd = KDTree(len(b))
    for i, p in enumerate(b):
        kd.insert(p, i)
    kd.balance()
    return sum(1 for p in a if kd.find(p)[2] <= tol) / len(a)


def render_views(out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    sc.render.engine = 'BLENDER_WORKBENCH'
    sc.display.shading.light = 'STUDIO'
    sc.display.shading.color_type = 'MATERIAL'
    sc.display.shading.show_cavity = True
    sc.render.resolution_x, sc.render.resolution_y = 1800, 1000
    sc.render.film_transparent = False
    pts = []
    for o in sc.objects:
        if o.type == 'MESH':
            pts += [o.matrix_world @ Vector(c) for c in o.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    c = (lo + hi) * 0.5
    size = max(hi.x - lo.x, hi.z - lo.z * 0 + (hi.z - lo.z) * 1.8)
    cam_data = bpy.data.cameras.new("cmp_cam")
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = size * 1.1
    cam = bpy.data.objects.new("cmp_cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    for view, d in (("front", Vector((0, -1, 0.35))),
                    ("back", Vector((0, 1, 0.35))),
                    ("below", Vector((0, -0.6, -1)))):
        d.normalize()
        cam.location = c + d * size * 3
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(out_dir, "%s_%s.png" % (stem, view))
        bpy.ops.render.render(write_still=True)


def diagnose(n, l_tips, o_tips, bvh):
    import math

    def sev(p):
        nz = bvh.find_nearest(p)[1].z
        return math.degrees(math.asin(max(-1.0, min(1.0, -nz))))

    def near(p, pts):
        return any((p - q).length <= MATCH_MM for q in pts)
    bins = (0, 15, 30, 45, 60, 75, 91)

    def hist(vals):
        return [sum(1 for v in vals if bins[k] <= v < bins[k + 1])
                for k in range(len(bins) - 1)]
    missed = [sev(p) for p in l_tips if not near(p, o_tips)]
    hit = [sev(p) for p in l_tips if near(p, o_tips)]
    extra = [sev(p) for p in o_tips if not near(p, l_tips)]
    print("DIAG pose %d bins %s" % (n, list(bins)))
    print("DIAG  lychee matched %s" % hist(hit))
    print("DIAG  lychee missed  %s" % hist(missed))
    print("DIAG  ours extra     %s" % hist(extra))
    print("DIAG  missed z: %s" % sorted(round(p.z, 1) for p in l_tips
                                         if not near(p, o_tips)))


results = []
for n in POSES:
    path = os.path.join(EXAMPLE, "pose %d.stl" % n)
    if not os.path.exists(path):
        continue
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.stl_import(filepath=path)
    src = bpy.context.selected_objects[0]
    model, lsup, n_parts = split_largest(src)
    bpy.data.objects.remove(src)

    me_l, bvh_l = model_bvh(model)
    l_tips, l_rests, l_bed = measure(lsup, bvh_l)
    lift = min((model.matrix_world @ v.co).z for v in model.data.vertices)
    bpy.data.meshes.remove(me_l)

    coll = bpy.data.collections.new("VibeCut Supports")
    bpy.context.scene.collection.children.link(coll)
    t0 = time.time()
    P = S.Params()
    for kv in OVERRIDES:
        k, v = kv.split('=')
        setattr(P, k, type(getattr(P, k))(v))
    rep = S.run_pipeline(model, P, collection=coll)
    dt = time.time() - t0
    ours = bpy.data.objects.get(rep["support_object"] or "")
    lsup.location.z -= lift
    for p in l_tips + l_rests:
        p.z -= lift
    me_o, bvh_o = model_bvh(model)
    o_tips, o_rests, o_bed = measure(ours, bvh_o) if ours else ([], [], 0)
    if "--diag" in argv:
        diagnose(n, l_tips, o_tips, bvh_o)
    bpy.data.meshes.remove(me_o)

    r = {
        "pose": n, "seconds": round(dt, 1),
        "lychee": {"tips": len(l_tips), "rests": len(l_rests), "bed_feet": l_bed},
        "ours": {"tips": len(o_tips), "rests": len(o_rests), "bed_feet": o_bed,
                 "verts": len(ours.data.vertices) if ours else 0},
        "recall_pct": round(100 * match(l_tips, o_tips, MATCH_MM)),
        "precision_pct": round(100 * match(o_tips, l_tips, MATCH_MM)),
        "coverage": rep["validation"].get("coverage"),
        "messages": rep["messages"][-1:],
    }
    results.append(r)
    print("COMPARE " + json.dumps(r))
    sys.stdout.flush()

    if SAVE:
        def mat(name, rgba):
            m = bpy.data.materials.new(name)
            m.diffuse_color = rgba
            return m
        lsup.data.materials.append(mat("lychee_red", (0.9, 0.25, 0.2, 1)))
        if ours:
            ours.data.materials.clear()
            ours.data.materials.append(mat("ours_green", (0.2, 0.85, 0.3, 1)))
            span = model.dimensions.x + 15
            m2 = model.copy()
            bpy.context.scene.collection.objects.link(m2)
            m2.location.x += span
            ours.location.x += span
        model.data.materials.append(mat("model_grey", (0.75, 0.75, 0.75, 1)))
        bpy.ops.wm.save_as_mainfile(
            filepath=os.path.join(EXAMPLE, "compare_pose%d.blend" % n))
        render_views(os.path.join(EXAMPLE, "renders"), "pose%d" % n)

print("SUMMARY " + json.dumps([{k: r[k] for k in ("pose", "recall_pct", "precision_pct")}
                               | {"l_tips": r["lychee"]["tips"], "o_tips": r["ours"]["tips"],
                                  "l_rests": r["lychee"]["rests"], "o_rests": r["ours"]["rests"],
                                  "l_bed": r["lychee"]["bed_feet"], "o_bed": r["ours"]["bed_feet"]}
                               for r in results]))
