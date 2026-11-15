import os, sys, traceback, bpy, bmesh
from mathutils import Vector, Quaternion
from bpy_extras.view3d_utils import location_3d_to_region_2d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import vibecut
vibecut.register()
cc = vibecut.click_collapse
os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
OUT = os.path.join(HERE, "output", "gui_click_collapse.log")
log = open(OUT, "w")
fails = []


def P(*a):
    print(*a, file=log, flush=True)


def check(cond, msg):
    P(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def env():
    wm = bpy.context.window_manager
    win = wm.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    return wm, win, area, region


def nverts(_ob=None):
    ob = bpy.data.objects.get("grid")
    if ob is None:
        P("  (grid object missing; mode=%s)" % bpy.context.mode)
        return -1
    if ob.mode != 'EDIT':
        P("  (grid not in edit mode: %s, %d verts)" % (ob.mode, len(ob.data.vertices)))
        return -len(ob.data.vertices)
    return len(bmesh.from_edit_mesh(ob.data).verts)


def win_xy(region, rv3d, co):
    p = location_3d_to_region_2d(region, rv3d, co)
    return int(round(region.x + p.x)), int(round(region.y + p.y))


def click(win, x, y):
    win.event_simulate(type='LEFTMOUSE', value='PRESS', x=x, y=y)
    win.event_simulate(type='LEFTMOUSE', value='RELEASE', x=x, y=y)


def nav_test(win, rv3d, x, y, tag):
    d0 = rv3d.view_distance
    win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    for _ in range(3):
        win.event_simulate(type='WHEELUPMOUSE', value='PRESS', x=x, y=y)
    yield 0.5
    zoomed = abs(rv3d.view_distance - d0) > 1e-3
    rv3d.view_distance = d0
    q0 = rv3d.view_rotation.copy()
    win.event_simulate(type='MIDDLEMOUSE', value='PRESS', x=x, y=y)
    for i in range(1, 11):
        win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x + 8 * i, y=y)
    win.event_simulate(type='MIDDLEMOUSE', value='RELEASE', x=x + 80, y=y)
    yield 0.6
    orbited = rv3d.view_rotation.rotation_difference(q0).angle > 1e-3
    P("NAV %s: zoom=%s orbit=%s" % (tag, zoomed, orbited))
    rv3d.view_perspective = 'ORTHO'
    rv3d.view_rotation = q0
    rv3d.view_location = (0, 0, 0)
    yield 0.3


def steps():
    wm, win, area, region = env()
    space = area.spaces.active
    rv3d = space.region_3d
    space.show_region_ui = True
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob)
    bm = bmesh.new()
    g = {}
    for x in range(-2, 3):
        for y in range(-2, 3):
            g[x, y] = bm.verts.new((x, y, 0))
    g[1, 0].co.x = 0.4
    for x in range(-2, 2):
        for y in range(-2, 2):
            bm.faces.new((g[x, y], g[x + 1, y], g[x + 1, y + 1], g[x, y + 1]))
    me = bpy.data.meshes.new("grid")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("grid", me)
    bpy.context.scene.collection.objects.link(ob)
    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.context.view_layer.objects.active = ob
        ob.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.ed.undo_push(message="test setup")
    rv3d.view_perspective = 'ORTHO'
    rv3d.view_rotation = Quaternion()
    rv3d.view_location = (0, 0, 0)
    rv3d.view_distance = 8.0
    yield 0.5
    wm, win, area, region = env()
    rv3d = area.spaces.active.region_3d
    check(nverts(ob) == 25, "start with 25 verts")
    cx, cy = region.x + region.width // 2, region.y + region.height // 2
    yield from nav_test(win, rv3d, cx, cy, "warm-up")

    with bpy.context.temp_override(window=win, area=area, region=region):
        r = bpy.ops.vibecut.click_collapse('INVOKE_DEFAULT')
    check(r == {'RUNNING_MODAL'} and wm.vibecut_cc_running, "tool starts")
    x, y = win_xy(region, rv3d, Vector((0, 0, 0)))
    P("click at", x, y, "region", region.x, region.y, region.width, region.height)
    win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    yield 0.3
    pair = cc._hover["pair"]
    P("hover", pair)
    check(pair is not None and Vector(pair[0]).length < 1e-6
          and (Vector(pair[1]) - Vector((0.4, 0, 0))).length < 1e-6,
          "hover previews centre -> (0.4, 0)")

    click(win, x, y)
    yield 0.4
    bm = bmesh.from_edit_mesh(bpy.data.objects['grid'].data)
    cos = [v.co.copy() for v in bm.verts]
    check(len(cos) == 24, "click collapsed one vert (%d left)" % len(cos))
    check(not any(c.length < 1e-6 for c in cos), "centre vert is gone")
    check(any((c - Vector((0.4, 0, 0))).length < 1e-6 for c in cos),
          "neighbour stayed in place")
    check(not any(v.select for v in bm.verts),
          "no select ran on the click (selection untouched)")
    check(wm.vibecut_cc_running, "tool still running after click")

    win.event_simulate(type='Z', value='PRESS', x=x, y=y, ctrl=True)
    win.event_simulate(type='Z', value='RELEASE', x=x, y=y, ctrl=True)
    yield 0.5
    check(nverts(ob) == 25, "Ctrl+Z restores the vert (%d)" % nverts(ob))
    check(wm.vibecut_cc_running, "tool still running after undo")
    win.event_simulate(type='Z', value='PRESS', x=x, y=y, ctrl=True, shift=True)
    win.event_simulate(type='Z', value='RELEASE', x=x, y=y, ctrl=True, shift=True)
    yield 0.5
    check(nverts(ob) == 24, "Ctrl+Shift+Z redoes it (%d)" % nverts(ob))
    win.event_simulate(type='Z', value='PRESS', x=x, y=y, ctrl=True)
    win.event_simulate(type='Z', value='RELEASE', x=x, y=y, ctrl=True)
    yield 0.5
    check(nverts(ob) == 25, "Ctrl+Z again (%d)" % nverts(ob))

    d0 = rv3d.view_distance
    for _ in range(3):
        win.event_simulate(type='WHEELUPMOUSE', value='PRESS', x=x, y=y)
    yield 0.5
    P("zoom", d0, "->", rv3d.view_distance)
    check(abs(rv3d.view_distance - d0) > 1e-3, "scroll wheel zooms")
    rv3d.view_distance = d0
    q0 = rv3d.view_rotation.copy()
    win.event_simulate(type='MIDDLEMOUSE', value='PRESS', x=x, y=y)
    for i in range(1, 11):
        win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x + 8 * i, y=y)
    win.event_simulate(type='MIDDLEMOUSE', value='RELEASE', x=x + 80, y=y)
    yield 0.6
    P("orbit", q0, "->", rv3d.view_rotation)
    check(rv3d.view_rotation.rotation_difference(q0).angle > 1e-3,
          "middle-mouse drag orbits")
    check(wm.vibecut_cc_running, "tool still running after navigating")
    rv3d.view_perspective = 'ORTHO'
    rv3d.view_rotation = q0
    rv3d.view_location = (0, 0, 0)
    yield 0.3

    win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x + 1, y=y)
    win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x, y=y)
    yield 0.3
    click(win, x, y)
    yield 0.4
    check(nverts(ob) == 24, "click after undo collapses again")

    ui = next(r for r in area.regions if r.type == 'UI')
    win.event_simulate(type='MOUSEMOVE', value='NOTHING',
                       x=ui.x + ui.width // 2, y=ui.y + 20)
    yield 0.3
    check(cc._hover["pair"] is None, "no preview over the N-panel")

    rv3d.view_perspective = 'PERSP'
    rv3d.view_rotation = Quaternion((0.92, 0.38, 0, 0))
    yield 0.3
    wm, win, area, region = env()
    rv3d = area.spaces.active.region_3d
    x2, y2 = win_xy(region, rv3d, Vector((-1, -1, 0)))
    win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x2, y=y2)
    yield 0.3
    pair = cc._hover["pair"]
    P("persp hover", pair)
    check(pair is not None
          and (Vector(pair[0]) - Vector((-1, -1, 0))).length < 1e-6,
          "perspective hover picks (-1,-1)")
    click(win, x2, y2)
    yield 0.4
    check(nverts(ob) == 23, "perspective click collapsed (%d)" % nverts(ob))

    win.event_simulate(type='RIGHTMOUSE', value='PRESS', x=x2, y=y2)
    win.event_simulate(type='RIGHTMOUSE', value='RELEASE', x=x2, y=y2)
    yield 0.4
    check(not wm.vibecut_cc_running and cc._draw_handle is None,
          "right-click stops and removes the draw handler")
    check(nverts(ob) == 23, "stopping keeps the edits")

    with bpy.context.temp_override(window=win, area=area, region=region):
        bpy.ops.vibecut.click_collapse('INVOKE_DEFAULT')
        r2 = bpy.ops.vibecut.click_collapse('INVOKE_DEFAULT')
    check(r2 == {'CANCELLED'}, "second press returns CANCELLED")
    win.event_simulate(type='MOUSEMOVE', value='NOTHING', x=x2 + 3, y=y2)
    yield 0.4
    check(not wm.vibecut_cc_running and cc._draw_handle is None,
          "second press stops the tool")

    win.event_simulate(type='Z', value='PRESS', x=x2, y=y2, ctrl=True)
    win.event_simulate(type='Z', value='RELEASE', x=x2, y=y2, ctrl=True)
    yield 0.5
    check(nverts(ob) == 24, "one undo = one click (%d)" % nverts(ob))


def _tail():
    wm, win, area, region = env()
    rv3d = area.spaces.active.region_3d
    cx, cy = region.x + region.width // 2, region.y + region.height // 2
    yield from nav_test(win, rv3d, cx, cy, "after tool")


def run():
    import itertools
    gen = itertools.chain(steps(), _tail())

    def tick():
        try:
            return next(gen)
        except StopIteration:
            P("RESULT", "ALL PASS" if not fails else "FAILS: %s" % fails)
        except Exception:
            P("EXCEPTION", traceback.format_exc())
        log.close()
        bpy.ops.wm.quit_blender()
        return None
    bpy.app.timers.register(tick, first_interval=2.0)


def watchdog():
    if not log.closed:
        P("WATCHDOG timeout")
        log.close()
    bpy.ops.wm.quit_blender()


bpy.app.timers.register(watchdog, first_interval=90.0)
run()
