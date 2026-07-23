"""
poster_bioclim_panel.py
Run from project root: python poster_bioclim_panel.py
Outputs: outputs/poster_plots/03_bioclim_panel.svg, .pdf

Panel layout (2×3 grid):
  bio1  bio2  bio3
  bio4  bio5  ----
"""

import urllib.request
from pathlib import Path

import numpy as np
import geopandas as gpd
import rioxarray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import box as sbox

OUT = Path("outputs/poster_plots")
OUT.mkdir(parents=True, exist_ok=True)

BIO_DIR = Path("data/raw/GeoPlant/Rasters/BioClimatic_Average_1981-2010")

# ── Bioclim metadata ──────────────────────────────────────────────────────────
BIO_META = {
    1: ("Annual Mean Temperature",       "RdYlBu_r"),
    2: ("Mean Diurnal Range",            "PuOr"),
    3: ("Isothermality",                 "YlOrRd"),
    4: ("Temperature Seasonality",       "coolwarm"),
    5: ("Max Temp of Warmest Month",     "inferno"),
}

# 6 slots, last is blank
VARIABLES = [1, 2, 3, 4, 5, None]

# ── France polygon ────────────────────────────────────────────────────────────
geojson_path = "/tmp/ne_10m_countries.geojson"
if not Path(geojson_path).exists():
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
        "geojson/ne_10m_admin_0_countries.geojson",
        geojson_path,
    )
world = gpd.read_file(geojson_path)
france_full = world[world["NAME"] == "France"].copy()
mainland_box = sbox(-5.2, 42.3, 8.3, 51.2)
france = france_full.copy()
france["geometry"] = france_full.geometry.intersection(mainland_box)

# ── Style ─────────────────────────────────────────────────────────────────────
BORDER  = "#000000"
XLIM    = (-5.2, 8.3)
YLIM    = (42.0, 51.2)
TITLE_FS = 7  # subplot title font size

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(7, 7))
fig.patch.set_alpha(0)

for ax, var in zip(axes.flat, VARIABLES):
    ax.patch.set_alpha(0)

    if var is None:
        ax.axis("off")
        continue

    title, cmap = BIO_META[var]
    tif = BIO_DIR / f"bio{var}.tif"
    raster = rioxarray.open_rasterio(tif, masked=True).squeeze()

    # Clip to France (reproject polygon to raster CRS)
    france_proj = france.to_crs(raster.rio.crs)
    clipped = raster.rio.clip(france_proj.geometry, france_proj.crs,
                              drop=True, all_touched=False)
    clipped_wgs = clipped.rio.reproject("EPSG:4326")

    data_vals = clipped_wgs.values.astype(float)
    vmin, vmax = np.nanpercentile(data_vals, [2, 98])

    ax.imshow(
        data_vals,
        extent=[
            clipped_wgs.x.values.min(), clipped_wgs.x.values.max(),
            clipped_wgs.y.values.min(), clipped_wgs.y.values.max(),
        ],
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        aspect="auto",
    )

    # France border
    france.plot(ax=ax, facecolor="none", edgecolor=BORDER,
                linewidth=1.0, zorder=3)

    # Axes limits + light ticks
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color("#000000")
    ax.set_xticks([])
    ax.set_yticks([])

    # Title
    ax.set_title(f"bio{var}\n{title}", fontsize=TITLE_FS,
                 color="#222222", pad=3)

    print(f"✓ bio{var}  {title}")

plt.subplots_adjust(wspace=0.08, hspace=0.01)

fig.savefig(OUT / "03_bioclim_panel.svg", format="svg",
            bbox_inches="tight", transparent=True, dpi=300)
fig.savefig(OUT / "03_bioclim_panel.pdf", format="pdf",
            bbox_inches="tight", transparent=True, dpi=300)
plt.close()
print(f"\n✓ saved to {OUT}/03_bioclim_panel.svg/.pdf")