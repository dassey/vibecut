# VibeCut Support Wizard — How to Use It

*(Plain-language guide. No programming needed.)*

## What it does

You give it a model, it gives you back:

1. **Support structures** — printable scaffolding under every overhang
   that needs it, with thin snap-off contact tips and a built-in air gap
   so they never fuse to your model.
2. **A print profile** — recommended Bambu Studio settings tuned to that
   exact model, as a one-click importable preset plus a plain-English
   explanation of every choice.

It picks the right *kind* of support automatically, per region:

| Situation | What it builds |
|---|---|
| Broad flat overhang (table tops, brims) | A grid of round pillars with cone tips and flared feet |
| Organic shapes (figures, animals, chins, ears) | Tree supports — branches that lean and merge like Bambu's tree supports, but yours to keep and edit |
| A big low "belly" (spheres, hulls, lying-down poses) | A form-fitting cradle block it carves from a rounded box |
| Short flat gaps between two solid sides | **Nothing** — your printer bridges those fine |
| Tunnels and recesses you can reach into | Slim trees with little pads that stand inside |
| Sealed hollow spaces | **Nothing, on purpose** — a support in there could never come out. It warns you instead |

Everything it builds is checked by computer before you ever see it:
watertight? not touching the model? nothing thinner than the printer can
make? every overhang point actually covered? If a check fails it fixes
itself and tries again, and tells you if anything still isn't right.

## See where supports are needed (Overhang Heatmap)

In the Supports panel, click **Show Overhang Heatmap**. Your model turns
into a slicer-style heat map:

- **Green** = safe, no support needed.
- **Yellow** = right at the threshold (your Overhang Angle, default 45°).
- **Orange → Red** = steep overhangs that need support; red is a flat
  downward-facing surface (the worst case).

It also tells you what fraction of the surface needs support and the
steepest angle found. It uses the *same* Overhang Angle as the support
generator, so the yellow/red areas are exactly where pins will be placed
— change the angle and click **Update** to recolor. Click **Clear** to go
back to normal shading. (This only adds a temporary color layer; it
doesn't change your model.)

## Two support styles

At the top of the Supports panel there's a **Support Style** dropdown:

- **Twist-off Pins (recommended, default)** — the original VibeCut
  supports: sparse thin pillars with branches and 1 mm pins that touch
  the model lightly and snap off by hand. Light, fast, minimal marks.
  Best for articulated figures. (On a test model: ~3,700 little
  triangles, half a second.)
- **Solid (validated)** — the heavier multi-strategy system
  (tree/pillar/cradle/bridge). Guaranteed watertight with an air gap,
  but much bulkier. Use this only if the pins don't behave on a rough or
  broken mesh.

Under **Advanced**, Pins mode lets you set pin spacing, pillar diameter,
how far branches reach to share a trunk, and whether to trim the pin tips
flush to the model (**Trim to Model** — On looks clean in Blender; Off is
bulletproof and lets the slicer trim them).

## Daily use (three clicks)

1. Click your model. Open the N-panel (press **N**) → **VibeCut** tab →
   **More...** → **Supports**.
2. Pick your **Printer** and **Material** at the top (defaults: X1
   Carbon, PLA). Leave **Support Style** on Twist-off Pins.
3. Click **Generate Supports**.

That's it. A green `SUP_...` object appears in the "VibeCut Supports"
collection. The status box tells you what it built (e.g. "TREE x17") and
whether **ALL PASS** — the full check report is in the text editor under
`FDMW_Report_<your model>` if you're curious.

Notes:
- The wizard works in **millimeters** (1 Blender unit = 1 mm) and drops
  your model onto the floor automatically.
- **If your model is bigger than the printer bed** (very common with
  downloaded STLs — some come in at a meter tall), the wizard refuses
  immediately and tells you the exact scale factor to use, e.g.
  "Scale the model to about 0.26x first (S, 0.26, Enter)". Do that,
  then click Generate again. It will never sit there frozen.
- Very dense scans (over ~250k triangles) are analyzed through an
  automatic simplified copy — you don't need to do anything, it just
  notes it in the report.
- Don't like the result? Tweak something under **Advanced** and click
  Generate again — it replaces the old supports.
- **Remove Supports** deletes everything it made.

## Getting it onto the printer

1. Click **Export STLs for Bambu Studio**, pick a folder. You get
   `yourmodel_model.stl` and `yourmodel_supports.stl`.
2. Drag **both files together** into Bambu Studio. It asks "load these as
   a single object with multiple parts?" — click **Yes**. They're already
   in the right position; don't move them separately.
3. Click **Save Print Profile**, pick a folder. You get:
   - `..._bambu_process.json` — in Bambu Studio: File → Import →
     Import Configs, pick this file, then select the "FDM Wizard..."
     preset in the Process dropdown.
   - `..._print_profile.md` — open with any text editor; it explains
     every recommended setting in plain English.
4. **Leave Bambu Studio's own supports OFF** (the preset does this for
   you) — your supports are already in the model.
5. Print. Afterwards the supports snap off at the thin tips; grab the
   pillar feet with pliers for stubborn ones.

Bonus: if your AMS has Bambu Support for PLA loaded, you can assign that
filament to just the supports part for glass-smooth undersides.

## The other new tools (Tools panel: Auto Bevel and Flat Region sections)

**Auto Bevel** — rounds off sharp edges so they print stronger and look
cleaner. In Object Mode, click **Auto Bevel (Modifier)**: every edge
sharper than 30° gets a small 0.4 mm round-over, non-destructively (you
can change or remove it in the modifier stack any time). In Edit Mode,
**Bevel Edges (Bake)** does it permanently — to your selected edges, or
to every sharp edge if nothing is selected.

**Flatten Region** — now works properly on its own. Select a face (or a
region with **Select Flat Region**), click **Flatten Region**. It uses
the same best-fit math LoopTools uses, so you no longer need the
LoopTools add-on. New **Influence** slider: 1.0 = perfectly flat,
0.5 = halfway (knocks the bumps down but keeps some character). It also
works on a plain vertex selection, just like LoopTools did.

## Honest limitations

- Everything has been verified **in software** (geometry checks +
  automated test suite). It has **not** been physically printed by the
  author — see `tests/output/` for the exact STL evidence files.
- Sealed internal cavities cannot be supported by anyone (slicers
  included) — the wizard warns and relies on bridging settings.
- A few spots can be physically impossible to support (gaps tighter than
  any printable tip). The wizard skips those and tells you how many.
- The cradle gap is looser (~0.7–0.9 mm) than tip gaps — cradles catch
  sag rather than hold the surface, which is normal for that style.
