"""
Left column: PA points (top) and PO points (bottom), smaller, stacked.
Right column: Euclidean distance-to-nearest-PA heatmap, spanning both rows, larger.

Usage:
    python scripts/plot_pa_po_distance_figure.py
"""

from pathlib import Path
from unicodedata import name

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from pyproj import Transformer

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


# ── everything you're likely to tweak lives here ────────────────────────
DATA_PATH = Path("data/processed/GeoPlant/full")
OUTPUT_PATH = Path("outputs/figures/pa_po_distance.png")

EXTENT = (-13, 35, 34, 72)  # lon_min, lon_max, lat_min, lat_max
RESOLUTION = 0.15
SMOOTH_SIGMA = 1.5
PO_SAMPLE_SIZE = 50000

CMAP_NAME = "hsv"
CMAP_VMAX_FRAC = 0.8  # trims the colormap so it doesn't wrap back to red
VMIN, VMAX = None, None

PA_COLOR = "#8D237D"
PO_COLOR = "#E8932B"
PANEL_BG = "#f7f5f0"
OCEAN_COLOR = "#b8d7e6"


def load_lonlat(prefix: str) -> np.ndarray:
    df = pd.read_pickle(DATA_PATH / "covariates" / f"{prefix}_covariates.pkl")
    return df[["lon", "lat"]].dropna().to_numpy()


def compute_distance_grid(pa_lonlat: np.ndarray, extent: tuple) -> tuple:
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.arange(lon_min, lon_max, RESOLUTION)
    lats = np.arange(lat_min, lat_max, RESOLUTION)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    # project to meters (Lambert Azimuthal Equal Area, centered on the extent)
    # so distances are true Euclidean, not raw lon/lat degrees
    central_lon, central_lat = np.mean([lon_min, lon_max]), np.mean([lat_min, lat_max])
    transformer = Transformer.from_crs(
        "EPSG:4326",
        f"+proj=laea +lon_0={central_lon} +lat_0={central_lat} +units=m",
        always_xy=True,
    )
    pa_x, pa_y = transformer.transform(pa_lonlat[:, 0], pa_lonlat[:, 1])
    grid_x, grid_y = transformer.transform(grid_lon.ravel(), grid_lat.ravel())

    tree = cKDTree(np.column_stack([pa_x, pa_y]))
    dist, _ = tree.query(np.column_stack([grid_x, grid_y]), k=1)
    dist_grid = dist.reshape(grid_lon.shape) / 1000  # meters -> km

    if SMOOTH_SIGMA > 0:
        dist_grid = gaussian_filter(dist_grid, sigma=SMOOTH_SIGMA)

    return lons, lats, dist_grid


def style_axis(ax, extent):
    if HAS_CARTOPY:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=PANEL_BG, zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="black", zorder=1)
    else:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.set_facecolor(PANEL_BG)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#4d4d4d")
        spine.set_linewidth(0.7)


def trimmed_cmap(name, vmax_frac, n=256):
    base = plt.cm.get_cmap(name, n)
    colors = base(np.linspace(0.0, vmax_frac, n))
    return LinearSegmentedColormap.from_list(f"{name}_trimmed", colors)


def plot_figure(pa_lonlat, po_lonlat, lons, lats, dist_grid, out_path: Path):
    extent = (lons.min(), lons.max(), lats.min(), lats.max())
    proj = ccrs.PlateCarree() if HAS_CARTOPY else None
    subplot_kw = {"projection": proj} if HAS_CARTOPY else {}

    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    left_h = 3.4
    right_h = 2 * left_h
    left_w = aspect * left_h
    right_w = aspect * right_h

    fig = plt.figure(figsize=(left_w + right_w + 0.3, right_h))
    gs = fig.add_gridspec(2, 2, width_ratios=[left_w, right_w], height_ratios=[1, 1],
                           wspace=0.05, hspace=0.05)

    ax_pa = fig.add_subplot(gs[0, 0], **subplot_kw)
    ax_po = fig.add_subplot(gs[1, 0], **subplot_kw)
    ax_dist = fig.add_subplot(gs[:, 1], **subplot_kw)

    for ax in (ax_pa, ax_po, ax_dist):
        style_axis(ax, extent)

    scatter_kw = {"transform": proj} if HAS_CARTOPY else {}
    legend_kw = dict(loc="upper left", frameon=True, framealpha=0.85, edgecolor="none",
                      handletextpad=0.4, borderpad=0.4, markerscale=4, fontsize=12)

    ax_pa.scatter(pa_lonlat[:, 0], pa_lonlat[:, 1], s=1.6, c=PA_COLOR, alpha=0.5,
                  linewidths=0, zorder=2, label=f"PA plots (n = {len(pa_lonlat):,})", **scatter_kw)
    ax_pa.legend(**legend_kw)

    po_plot = po_lonlat
    if PO_SAMPLE_SIZE is not None and len(po_lonlat) > PO_SAMPLE_SIZE:
        idx = np.random.default_rng(0).choice(len(po_lonlat), PO_SAMPLE_SIZE, replace=False)
        po_plot = po_lonlat[idx]
    ax_po.scatter(po_plot[:, 0], po_plot[:, 1], s=1.2, c=PO_COLOR, alpha=0.35,
                  linewidths=0, zorder=2, label=f"PO observations (n = {len(po_lonlat):,})", **scatter_kw)
    ax_po.legend(**legend_kw)

    cmap_obj = trimmed_cmap(CMAP_NAME, CMAP_VMAX_FRAC)
    # reverse
    cmap_obj = LinearSegmentedColormap.from_list(f"{CMAP_NAME}_rev", cmap_obj(np.linspace(1, 0, 256)))

    def warped_cmap(name, vmax_frac=0.85, power=4, n=256, reverse=True):
        base = plt.cm.get_cmap(name, n)
        t = np.linspace(0, 1, n)
        warped_t = t ** power
        if reverse:
            warped_t = warped_t[::-1]
        colors = base(warped_t * vmax_frac)
        return LinearSegmentedColormap.from_list(f"{name}_warped", colors)
    cmap_obj = warped_cmap(CMAP_NAME, CMAP_VMAX_FRAC)


    im_kw = dict(origin="lower", extent=extent, cmap=cmap_obj, aspect="auto",
                 interpolation="bilinear", vmin=VMIN, vmax=VMAX, zorder=1)
    if HAS_CARTOPY:
        im_kw["transform"] = proj

    im = ax_dist.imshow(dist_grid, **im_kw)

    if HAS_CARTOPY:
        ax_dist.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=2)
        ax_dist.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333", zorder=3)
        ax_dist.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="#333333", zorder=3)

    extend = "both" if (VMIN is not None and VMAX is not None) else \
             "max" if VMAX is not None else "min" if VMIN is not None else "neither"
    cbar = plt.colorbar(im, ax=ax_dist, label="Distance to nearest PA training plot (km)",
                         shrink=0.75, pad=0.015, extend=extend, aspect=25)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=12, width=0.5, length=3, direction="in", pad=2)
    cbar.ax.yaxis.label.set_fontsize(12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    pa_lonlat = load_lonlat("pa_train")
    po_lonlat = load_lonlat("po")
    print(f"Loaded {len(pa_lonlat)} PA points, {len(po_lonlat)} PO points")

    lons, lats, dist_grid = compute_distance_grid(pa_lonlat, EXTENT)
    print(f"Grid shape: {dist_grid.shape}")

    plot_figure(pa_lonlat, po_lonlat, lons, lats, dist_grid, OUTPUT_PATH)


if __name__ == "__main__":
    main()