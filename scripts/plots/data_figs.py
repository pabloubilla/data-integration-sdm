"""
poster_data_plots.py
Run from project root: python poster_data_plots.py
Outputs: outputs/poster_plots/01_po_map.svg, 01_po_map.pdf
         outputs/poster_plots/02_pa_map.svg, 02_pa_map.pdf
Changes vs original:
  - Uses ne_10m (higher resolution) instead of ne_110m France polygon
  - Clips east boundary to 8.3°E to exclude Corsica
  - Transparent figure/axes background (suitable for poster overlay)
"""

import urllib.request
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Point, box as sbox

sys.path.insert(0, "src")
from isdm.load_data import load_geoplant_processed

OUT = Path("outputs/poster_plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── France polygon (10m resolution, Corsica excluded) ─────────────────────────
geojson_path = "/tmp/ne_10m_countries.geojson"
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_admin_0_countries.geojson",
    geojson_path,
)
world = gpd.read_file(geojson_path)
france_full = world[world["NAME"] == "France"].copy()

# Clip to mainland only:
#   - east limit 8.3°E excludes Corsica (which sits at ~8.6–9.6°E)
#   - south limit 42.3°N safely includes Perpignan (~42.7°N) while
#     dropping the narrow southern tip that would otherwise include
#     the northern tip of Corsica at ~43.0°N after clipping
mainland_box = sbox(-5.2, 42.3, 8.3, 51.2)
france = france_full.copy()
france["geometry"] = france_full.geometry.intersection(mainland_box)

france_poly = france.geometry.iloc[0]  # shapely polygon for masking

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_geoplant_processed(add_coordinates=True, verbose=True)

def mask_to_france(df):
    """Keep only rows whose (lon, lat) falls inside the France polygon."""
    pts = gpd.GeoSeries(
        [Point(x, y) for x, y in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    return df[pts.within(france_poly)].copy()

po    = mask_to_france(data.X_po)
pa_tr = mask_to_france(data.X_pa_train)
pa_te = mask_to_france(data.X_pa_test)

print(f"PO inside France:       {len(po):,}")
print(f"PA train inside France: {len(pa_tr):,}")
print(f"PA test inside France:  {len(pa_te):,}")

# ── Shared style ──────────────────────────────────────────────────────────────
LAND   = "#EDEAE3"
BORDER = "#000000"
PO_COLOR  = "#f17c15ea"
PA_COLOR = "#867eddf0"

XLIM = (-5.2, 8.3)
YLIM = (42.0, 51.2)

def base_map(ax):
    france.plot(ax=ax, facecolor=LAND, edgecolor=BORDER, linewidth=1, zorder=1)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.patch.set_alpha(0)  # transparent axes background

def transparent_fig():
    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    fig.patch.set_alpha(0)   # transparent figure background
    return fig, ax

# ── 01 PO map ─────────────────────────────────────────────────────────────────
fig, ax = transparent_fig()
base_map(ax)

ax.scatter(po["lon"], po["lat"],
           s=0.1, color=PO_COLOR, alpha=0.3, linewidths=0,
           rasterized=True, zorder=2)

fig.savefig(OUT / "01_po_map.svg", format="svg", bbox_inches="tight",
            transparent=True, dpi=300)
fig.savefig(OUT / "01_po_map.pdf", format="pdf", bbox_inches="tight",
            transparent=True)
plt.close()
print("✓ 01_po_map")

# ── 02 PA map ─────────────────────────────────────────────────────────────────
fig, ax = transparent_fig()
base_map(ax)

ax.scatter(pa_tr["lon"], pa_tr["lat"],
           s=1, marker="s", color=PA_COLOR, alpha=0.6,
           linewidths=0, rasterized=True, zorder=2)

fig.savefig(OUT / "02_pa_map.svg", format="svg", bbox_inches="tight",
            transparent=True, dpi=300)
fig.savefig(OUT / "02_pa_map.pdf", format="pdf", bbox_inches="tight",
            transparent=True, dpi=300)
plt.close()
print("✓ 02_pa_map")

print(f"\nAll saved to {OUT}/")