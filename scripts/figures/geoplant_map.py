"""Map of PA plots, PO observations, and distance to the nearest PA training plot (Fig. 1)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from isdm.utils import haversine_tree, nearest_km

DATA_PATH = Path("data/processed/GeoPlant/full")
OUTPUT_PATH = Path("outputs/figures/pa_po_distance.png")

EXTENT = (-13, 35, 34, 72)  # lon_min, lon_max, lat_min, lat_max
RESOLUTION = 0.15           # grid step in degrees
SMOOTH_SIGMA = 1.5          # gaussian smoothing of the distance grid (in grid cells)
PO_SAMPLE_SIZE = 50_000     # PO points drawn on the map (the legend shows the full count)
VMAX = None                 # cap the colorbar (km), None = data max

CMAP_NAME = "hsv"
CMAP_VMAX_FRAC = 0.8        # stop before hsv wraps back to red
CMAP_POWER = 4              # stretches the colors over short distances

PA_COLOR = "#8D237D"
PO_COLOR = "#E8932B"
LAND_COLOR = "#f7f5f0"
OCEAN_COLOR = "#b8d7e6"
PROJ = ccrs.PlateCarree()


def load_lonlat(prefix):
    df = pd.read_pickle(DATA_PATH / "covariates" / f"{prefix}_covariates.pkl")
    return df[["lon", "lat"]].dropna().to_numpy()


def distance_grid(pa_lonlat):
    """Great-circle distance (km) from each grid cell to the nearest PA plot."""
    lon_min, lon_max, lat_min, lat_max = EXTENT
    lons = np.arange(lon_min, lon_max, RESOLUTION)
    lats = np.arange(lat_min, lat_max, RESOLUTION)
    grid_lon, grid_lat = np.meshgrid(lons, lats)

    cells = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    dist = nearest_km(haversine_tree(pa_lonlat), cells).reshape(grid_lon.shape)
    return lons, lats, gaussian_filter(dist, sigma=SMOOTH_SIGMA) if SMOOTH_SIGMA > 0 else dist


def distance_cmap():
    """hsv cut before it wraps, warped so most colors go to short distances, reversed."""
    t = np.linspace(0, 1, 256) ** CMAP_POWER
    colors = matplotlib.colormaps[CMAP_NAME](t[::-1] * CMAP_VMAX_FRAC)
    return LinearSegmentedColormap.from_list("distance", colors)


def style_axis(ax, extent):
    ax.set_extent(extent, crs=PROJ)
    ax.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="black", zorder=1)
    for spine in ax.spines.values():
        spine.set_edgecolor("#4d4d4d")
        spine.set_linewidth(0.7)


def main():
    pa = load_lonlat("pa_train")
    po = load_lonlat("po")
    print(f"Loaded {len(pa):,} PA points, {len(po):,} PO points")
    lons, lats, dist = distance_grid(pa)
    extent = (lons.min(), lons.max(), lats.min(), lats.max())

    # layout: two small maps (PA, PO) on the left, one large distance map on the right
    aspect = (extent[1] - extent[0]) / (extent[3] - extent[2])
    small_h = 3.4
    fig = plt.figure(figsize=(3 * aspect * small_h + 0.3, 2 * small_h))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], wspace=0.05, hspace=0.05)
    ax_pa = fig.add_subplot(gs[0, 0], projection=PROJ)
    ax_po = fig.add_subplot(gs[1, 0], projection=PROJ)
    ax_dist = fig.add_subplot(gs[:, 1], projection=PROJ)
    for ax in (ax_pa, ax_po, ax_dist):
        style_axis(ax, extent)

    legend_kw = dict(loc="upper left", framealpha=0.85, edgecolor="none", handletextpad=0.4,
                     borderpad=0.4, markerscale=4, fontsize=12)
    ax_pa.scatter(pa[:, 0], pa[:, 1], s=1.6, c=PA_COLOR, alpha=0.5, linewidths=0, zorder=2,
                  transform=PROJ, label=f"PA plots (n = {len(pa):,})")
    ax_pa.legend(**legend_kw)

    po_plot = po
    if len(po) > PO_SAMPLE_SIZE:
        po_plot = po[np.random.default_rng(0).choice(len(po), PO_SAMPLE_SIZE, replace=False)]
    ax_po.scatter(po_plot[:, 0], po_plot[:, 1], s=1.2, c=PO_COLOR, alpha=0.35, linewidths=0, zorder=2,
                  transform=PROJ, label=f"PO observations (n = {len(po):,})")
    ax_po.legend(**legend_kw)

    im = ax_dist.imshow(dist, origin="lower", extent=extent, transform=PROJ, cmap=distance_cmap(),
                        interpolation="bilinear", vmax=VMAX, zorder=1)
    # redraw ocean and borders on top so the distance field stops at the coast
    ax_dist.add_feature(cfeature.OCEAN, facecolor=OCEAN_COLOR, zorder=2)
    ax_dist.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor="#333333", zorder=3)
    ax_dist.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor="#333333", zorder=3)

    cbar = fig.colorbar(im, ax=ax_dist, shrink=0.75, pad=0.015, aspect=25,
                        extend="max" if VMAX is not None else "neither")
    cbar.set_label("Distance to nearest PA training plot (km)", fontsize=12)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(labelsize=12, width=0.5, length=3, direction="in", pad=2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()