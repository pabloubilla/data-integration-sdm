import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

# ── parameters ───────────────────────────────────────────────────────────────
GRID_BINS   = 14        # bins per axis — controls grid coarseness
OCCUPANCY   = 0.80      # fraction of cells that have PA points
N_PO        = 1000      # number of PO points
W           = 480       # canvas size (square)
MARGIN      = 28
INNER       = W - 2 * MARGIN

TEST_PROP   = 0.25      # fraction of total points going to test
BANDWIDTH   = 0.22      # Gaussian sigma as fraction of INNER (tighter = more blob)
SEED        = 7

# anchor position as fraction of INNER (0=top-left, 1=bottom-right)
ANCHOR_FX   = 0.68
ANCHOR_FY   = 0.30

# ── colors ────────────────────────────────────────────────────────────────────
# TEST and TRAIN are matched in lightness on purpose (same "weight" on the page),
# so the hatch pattern on TEST — not color contrast — is what makes it CVD-safe.
COL_PA      = "#7F77DD"
COL_TEST    = "#C8447A"   # rose
COL_CLOSE   = "#3E9FB8"   # teal
COL_MID     = "#3E9FB8"
COL_FAR     = "#3E9FB8"
COL_TRAIN   = "#3E9FB8"
COL_PO      = "#E68B24"

CANVAS_BG     = "none"     # keep transparent so it drops onto any slide/page bg
CANVAS_STROKE = "#000000"

# hatch pattern used for every "test" square, regardless of panel
HATCH_ID       = "hatchTest"
HATCH_TILE     = 5       # px — small tile so 2-3 stripes still fit inside a square
HATCH_STROKE   = "#FFFFFF"
HATCH_OPACITY  = 0.9
HATCH_WIDTH    = 2.2      # line thickness — the main lever for "more evident"
HATCH_CROSS    = True     # True = crosshatch (grid), False = single diagonal stripes

# point sizes
SQ          = None        # square half-size — set automatically below
DOT_R       = 6.0         # PO dot radius

OUT_DIR     = Path("outputs/poster_panels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── grid + PA points ─────────────────────────────────────────────────────────
rng  = np.random.default_rng(SEED)
CELL = INNER / GRID_BINS
SQ   = CELL * 0.40       # square fills ~80% of cell width

cell_centers = []
for i in range(GRID_BINS):
    for j in range(GRID_BINS):
        if rng.random() < OCCUPANCY:
            cx = MARGIN + (i + 0.5) * CELL
            cy = MARGIN + (j + 0.5) * CELL
            # small jitter so it doesn't look perfectly mechanical
            cx += rng.uniform(-CELL * 0.12, CELL * 0.12)
            cy += rng.uniform(-CELL * 0.12, CELL * 0.12)
            cell_centers.append((i, j, cx, cy))

pa   = np.array([(cx, cy) for _, _, cx, cy in cell_centers])
n_pa = len(pa)

# ── Gaussian density at each PA point ────────────────────────────────────────
anchor = np.array([
    MARGIN + INNER * ANCHOR_FX,
    MARGIN + INNER * ANCHOR_FY,
])
sigma  = INNER * BANDWIDTH
dists  = np.linalg.norm(pa - anchor, axis=1)
raw_density = np.exp(-0.5 * (dists / sigma) ** 2)
density     = raw_density / raw_density.sum()

# ── iterative test assignment (weighted sampling until quota) ─────────────────
target_test = int(round(n_pa * TEST_PROP))

remaining_idx       = list(range(n_pa))
remaining_density   = list(density)
test_idx            = []
test_count          = 0

while test_count < target_test and remaining_idx:
    w     = np.array(remaining_density)
    w     = w / w.sum()
    pick  = int(rng.choice(len(remaining_idx), p=w))
    chosen = remaining_idx[pick]
    test_idx.append(chosen)
    test_count += 1                     # each PA point = 1 observation
    remaining_idx.pop(pick)
    remaining_density.pop(pick)

test_idx = np.array(test_idx)
train_idx = np.array(remaining_idx)

# ── iterative train tercile assignment ────────────────────────────────────────
# re-read densities for train clusters, sorted descending = closest first
train_density = density[train_idx]
train_order   = np.argsort(train_density)[::-1]   # high density first = closest
sorted_train  = train_idx[train_order]
sorted_dens   = train_density[train_order]

target_tercile = int(round(len(train_idx) / 3))

def fill_tercile(remaining_ids, remaining_dens, target, rng, invert=False, choose_max=True):
    """Iteratively sample by density weight until target count reached."""
    rem_ids  = list(remaining_ids)
    rem_dens = list(remaining_dens)
    selected = []
    count    = 0
    while count < target and rem_ids:
        w = np.array(rem_dens)
        if invert:
            w = 1.0 / (w + 1e-9)
        w = w / w.sum()
        if choose_max:
            pick = int(np.argmax(w))
        else:
            pick = int(rng.choice(len(rem_ids), p=w))
        selected.append(rem_ids[pick])
        count += 1
        rem_ids.pop(pick)
        rem_dens.pop(pick)
    return np.array(selected), rem_ids, rem_dens

close_idx, rem_ids, rem_dens = fill_tercile(
    sorted_train, sorted_dens, target_tercile, rng, invert=False)

mid_idx, rem_ids, rem_dens   = fill_tercile(
    rem_ids, rem_dens, target_tercile, rng, invert=False)

far_idx = np.array(rem_ids)

print(f"PA total: {n_pa}  |  test: {len(test_idx)}  |  "
      f"close: {len(close_idx)}  mid: {len(mid_idx)}  far: {len(far_idx)}")

# ── PO points (clustered to simulate observer hotspots) ──────────────────────
# define cluster centers as fractions of INNER
cluster_centers = [
    (0.15, 0.75),   # bottom-left
    (0.30, 0.55),   # center-left
    (0.50, 0.80),   # bottom-center
    (0.70, 0.60),   # center-right
    (0.20, 0.30),   # top-left
    (0.80, 0.10),   # top-right
]
cluster_std   = 0.10   # spread of each cluster, as fraction of INNER
cluster_sizes = [60, 50, 45, 60, 25, 20]   # points per cluster (sums to N_PO)

po_x, po_y = [], []
for (fx, fy), size in zip(cluster_centers, cluster_sizes):
    cx = MARGIN + INNER * fx
    cy = MARGIN + INNER * fy
    std = INNER * cluster_std
    po_x.append(rng.normal(cx, std, size))
    po_y.append(rng.normal(cy, std, size))

po_x = np.clip(np.concatenate(po_x), MARGIN, MARGIN + INNER)
po_y = np.clip(np.concatenate(po_y), MARGIN, MARGIN + INNER)

# ── SVG helpers ───────────────────────────────────────────────────────────────
def svg_root(title_text, desc_text, vw=W, vh=W):
    el = ET.Element("svg", {
        "xmlns":   "http://www.w3.org/2000/svg",
        "width":   "100%",
        "viewBox": f"0 0 {vw} {vh}",
        "role":    "img",
        "style":   "background:transparent",
    })
    ET.SubElement(el, "title").text = title_text
    ET.SubElement(el, "desc").text  = desc_text
    return el

def add_canvas(svg, fill=CANVAS_BG, stroke=CANVAS_STROKE):
    """Soft background + border so the panel reads as a card, not a void."""
    ET.SubElement(svg, "rect", {
        "x": "0", "y": "0",
        "width": str(W), "height": str(W),
        "fill": fill, "stroke": stroke, "stroke-width": "1.5",
        "rx": "12",
    })

def add_hatch_pattern(svg):
    """Diagonal white hatch over COL_TEST — the CVD-safe cue that doesn't
    depend on the viewer being able to tell rose from teal."""
    defs = ET.SubElement(svg, "defs")
    pattern = ET.SubElement(defs, "pattern", {
        "id": HATCH_ID,
        "width": str(HATCH_TILE), "height": str(HATCH_TILE),
        "patternUnits": "userSpaceOnUse",
        "patternTransform": "rotate(45)",
    })
    ET.SubElement(pattern, "rect", {
        "width": str(HATCH_TILE), "height": str(HATCH_TILE), "fill": COL_TEST,
    })
    ET.SubElement(pattern, "line", {
        "x1": "0", "y1": "0", "x2": "0", "y2": str(HATCH_TILE),
        "stroke": HATCH_STROKE, "stroke-width": str(HATCH_WIDTH), "opacity": str(HATCH_OPACITY),
    })
    if HATCH_CROSS:
        ET.SubElement(pattern, "line", {
            "x1": "0", "y1": "0", "x2": str(HATCH_TILE), "y2": "0",
            "stroke": HATCH_STROKE, "stroke-width": str(HATCH_WIDTH), "opacity": str(HATCH_OPACITY),
        })

def sq(parent, cx, cy, size, fill, opacity=0.9, rx=2):
    ET.SubElement(parent, "rect", {
        "x": str(round(cx - size, 2)), "y": str(round(cy - size, 2)),
        "width": str(round(size * 2, 2)), "height": str(round(size * 2, 2)),
        "fill": fill, "opacity": str(opacity), "rx": str(rx),
    })

def dot(parent, cx, cy, r, fill, opacity=0.78):
    ET.SubElement(parent, "circle", {
        "cx": str(round(cx, 5)), "cy": str(round(cy, 5)),
        "r":  str(r), "fill": fill, "opacity": str(opacity),
    })

def draw_squares(parent, indices, fill, opacity=0.9):
    for i in indices:
        sq(parent, pa[i, 0], pa[i, 1], SQ, fill, opacity)

def save(svg, name):
    path = OUT_DIR / name
    ET.ElementTree(svg).write(path, xml_declaration=False, encoding="unicode")
    print(f"  saved → {path}")


# ── panel 1: all PA points ────────────────────────────────────────────────────
svg = svg_root("PA survey grid", "All presence-absence survey locations as a grid")
add_canvas(svg)
draw_squares(svg, range(n_pa), COL_PA)
save(svg, "panel_1_pa_grid.svg")

# ── panel 2: traditional random split ────────────────────────────────────────
# standard random 75/25 split, no spatial structure
trad_test  = rng.choice(n_pa, size=int(n_pa * TEST_PROP), replace=False)
trad_train = np.setdiff1d(np.arange(n_pa), trad_test)

svg = svg_root("Traditional train/test split", "Random 75/25 split with no spatial structure")
add_canvas(svg)
add_hatch_pattern(svg)
draw_squares(svg, trad_train, COL_TRAIN)
draw_squares(svg, trad_test,  f"url(#{HATCH_ID})")
save(svg, "panel_2_traditional.svg")

# ── panel 3a: Gaussian split — closest ───────────────────────────────────────
svg = svg_root("Gaussian split — closest train",
               "Test blob hatched, closest train in teal, others faded")
add_canvas(svg)
add_hatch_pattern(svg)
draw_squares(svg, far_idx,   COL_PA,    opacity=0.18)
draw_squares(svg, mid_idx,   COL_PA,    opacity=0.18)
draw_squares(svg, close_idx, COL_CLOSE, opacity=0.90)
draw_squares(svg, test_idx,  f"url(#{HATCH_ID})", opacity=0.90)
save(svg, "panel_3a_closest.svg")

# ── panel 3b: Gaussian split — middle ────────────────────────────────────────
svg = svg_root("Gaussian split — middle train",
               "Test blob hatched, middle-distance train in teal, others faded")
add_canvas(svg)
add_hatch_pattern(svg)
draw_squares(svg, far_idx,   COL_PA,  opacity=0.18)
draw_squares(svg, close_idx, COL_PA,  opacity=0.18)
draw_squares(svg, mid_idx,   COL_MID, opacity=0.90)
draw_squares(svg, test_idx,  f"url(#{HATCH_ID})", opacity=0.90)
save(svg, "panel_3b_middle.svg")

# ── panel 3c: Gaussian split — farthest ──────────────────────────────────────
svg = svg_root("Gaussian split — farthest train",
               "Test blob hatched, farthest train in teal, others faded")
add_canvas(svg)
add_hatch_pattern(svg)
draw_squares(svg, close_idx, COL_PA,  opacity=0.18)
draw_squares(svg, mid_idx,   COL_PA,  opacity=0.18)
draw_squares(svg, far_idx,   COL_FAR, opacity=0.90)
draw_squares(svg, test_idx,  f"url(#{HATCH_ID})", opacity=0.90)
save(svg, "panel_3c_farthest.svg")

# ── panel 4: PO points ───────────────────────────────────────────────────────
svg = svg_root("Presence-only observations",
               "Opportunistically recorded species occurrences as dots")
add_canvas(svg)
for x, y in zip(po_x, po_y):
    dot(svg, x, y, DOT_R, COL_PO)
save(svg, "panel_4_po.svg")

print("\nDone. Tweak at the top of the script:")
print("  GRID_BINS, OCCUPANCY, BANDWIDTH, ANCHOR_FX/FY, TEST_PROP")
print("  SQ (square size), DOT_R, COL_* colors, SEED")
print("  HATCH_TILE / HATCH_STROKE / HATCH_OPACITY for the test texture")