"""
PO vs PA multispecies schematic  —  ecology palette, poster-ready
==================================================================

QUICK EDITS
-----------
SPECIES       list — name, color (#hex), marker string
PO_POINTS     dict: species name → list of (x, y) in [0, 10]
CELL_POS      list of (x0, y0) bottom-left corners of surveyed cells
              Species presence per cell is DERIVED automatically from PO_POINTS:
              a species is marked present in a cell if ≥1 of its PO points falls inside.
CELL          cell side length in data units
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import os

# ── species ───────────────────────────────────────────────────
SPECIES = [
    {"name": "A", "color": "#C1440E", "marker": "o"},    # burnt orange / circle
    {"name": "B", "color": "#5A8A3C", "marker": "^"},    # moss green  / triangle
    {"name": "C", "color": "#8B5E3C", "marker": "D"},    # warm brown  / diamond
    {"name": "D", "color": "#D4A017", "marker": "s"},    # ochre       / square
]

# ── PO points  (x, y) in [0, 10] ─────────────────────────────
PO_POINTS = {
    "A": [(1.1, 8.7), (2.6, 6.5), (1.8, 5.2), (3.2, 7.6),
          (0.9, 3.5), (2.9, 2.7), (1.5, 1.4)],
    "B": [(5.9, 9.2), (7.3, 7.4), (5.2, 6.1), (8.4, 5.7),
          (6.7, 4.2), (8.0, 2.5), (9.1, 8.1)],
    "C": [(3.9, 8.1), (4.6, 5.8), (3.3, 4.0), (5.1, 3.0),
          (2.0, 1.9), (4.2, 1.1)],
    "D": [(8.6, 9.4), (6.9, 8.3), (9.0, 6.6), (5.7, 5.1),
          (7.4, 3.6), (2.3, 7.3), (9.5, 3.1)],
}

# ── PA cell positions (bottom-left corners) ───────────────────
CELL = 1.8   # cell side length in data units

CELL_POS = [
    (0.2,  7.8), (3.8,  8.1), (7.3,  7.5),
    (1.5,  5.6), (5.2,  5.9), (7.8,  5.7),
    (0.3,  3.2), (4.0,  3.5), (7.1,  3.8),
    (1.0,  0.8), (4.8,  1.2), (7.9,  0.5),
    (2.8,  1.5),
]

# ── derive species presence from PO points ────────────────────
def species_in_cell(x0, y0, cell, po_points, extra_prob=0.3, rng=np.random.default_rng(123)):
    """
    Return set of species names present in the cell.
    A species is present if:
      - it has ≥1 PO point inside the cell, OR
      - it is drawn with probability `extra_prob` (mimics PA detecting
        species that happen to be there but have no PO record in this cell).
    """
    present = set()
    for sp, pts in po_points.items():
        # direct hit from PO
        for (px, py) in pts:
            if x0 <= px < x0 + cell and y0 <= py < y0 + cell:
                present.add(sp)
                break
        # probabilistic extra detection for species not already found
        if sp not in present and extra_prob > 0:
            if (rng or np.random).random() < extra_prob:
                present.add(sp)
    return present

CELL_POSITIONS = {
    (x0, y0): species_in_cell(x0, y0, CELL, PO_POINTS)
    for (x0, y0) in CELL_POS
}

# ── overlap safety check ──────────────────────────────────────
def _check_overlaps(positions, cell):
    def ov(a, b):
        return not (a[0]+cell <= b[0] or b[0]+cell <= a[0] or
                    a[1]+cell <= b[1] or b[1]+cell <= a[1])
    ps = list(positions)
    bad = [(ps[i], ps[j]) for i in range(len(ps))
           for j in range(i+1, len(ps)) if ov(ps[i], ps[j])]
    print("Overlap check:", "OK" if not bad else f"WARNING {bad}")

_check_overlaps(CELL_POSITIONS.keys(), CELL)

# ── visual parameters ─────────────────────────────────────────
PO_MS      = 9      # PO marker size
PO_LW      = 1.1   # PO marker edge linewidth
PA_MS      = 9      # PA marker size inside cells
PA_LW      = 1.1   # PA marker edge linewidth

AREA_COLOR = "#F2EDE4"   # map background — shared by both panels
CELL_FILL  = "#D6C9B0"   # surveyed cell background
CELL_EDGE  = "#8C7355"   # cell border

# ── figure ────────────────────────────────────────────────────
fig, (ax_po, ax_pa) = plt.subplots(
    1, 2, figsize=(12, 5.5), facecolor="none",
    gridspec_kw={"wspace": 0.06},
)

# fixed 2×2 anchor positions for species markers inside each cell
ANCHORS = [(0.27, 0.73), (0.73, 0.73), (0.27, 0.27), (0.73, 0.27)]

for ax in (ax_po, ax_pa):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_facecolor(AREA_COLOR)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#A8957A")
    ax.set_xticks([]); ax.set_yticks([])

# PO panel
for s in SPECIES:
    pts = PO_POINTS.get(s["name"], [])
    if not pts:
        continue
    xs, ys = zip(*pts)
    ax_po.scatter(xs, ys, s=PO_MS**2, marker=s["marker"],
                  facecolors=s["color"], edgecolors="black",
                  linewidths=PO_LW, zorder=4)

# PA panel
for (x0, y0), present in CELL_POSITIONS.items():
    ax_pa.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), CELL, CELL, boxstyle="square,pad=0",
        facecolor=CELL_FILL, edgecolor=CELL_EDGE, linewidth=0.9, zorder=1))
    for idx, s in enumerate(SPECIES):
        ax_x = x0 + ANCHORS[idx][0] * CELL
        ax_y = y0 + ANCHORS[idx][1] * CELL
        if s["name"] in present:
            ax_pa.scatter(ax_x, ax_y, s=PA_MS**2, marker=s["marker"],
                          facecolors=s["color"], edgecolors="black",
                          linewidths=PA_LW, zorder=3)
        else:
            ax_pa.scatter(ax_x, ax_y, s=(PA_MS*0.65)**2, marker="x",
                          color="#9C8870", linewidths=0.7, zorder=2)

# ── save ──────────────────────────────────────────────────────
fig_path = os.path.join("outputs", "figures_poster")
os.makedirs(fig_path, exist_ok=True)
plt.savefig(os.path.join(fig_path, "po_pa_schematic.svg"), bbox_inches="tight", facecolor="none")
plt.savefig(os.path.join(fig_path, "po_pa_schematic.pdf"), bbox_inches="tight", facecolor="none")
print("Saved SVG and PDF.")