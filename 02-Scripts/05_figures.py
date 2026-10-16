# Builds the five main figures for the report:
#   rgb_comparison.png    true colour, one panel per year
#   ndvi_comparison.png   NDVI, one panel per year
#   ndbi_comparison.png   NDBI, one panel per year
#   ndvi_change.png       NDVI difference, one panel per period
#   ndbi_change.png       NDBI difference, one panel per period
#
# ENVI equivalent: opening each raster in its own display group with the
# views linked, and applying the same stretch to each.


#---1. Libraries---

from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
import rasterio


#---2. Configuration---

# A module name cannot start with a digit, so "00_config.py" is loaded with
# runpy instead of a normal import.
# https://docs.python.org/3/library/runpy.html

config = runpy.run_path(
    str(Path(__file__).with_name("00_config.py"))
)

CLEAN_DIR = config["CLEAN_DIR"]
INDICES_DIR = config["INDICES_DIR"]
FIGURES_DIR = config["FIGURES_DIR"]

YEARS = config["YEARS"]
BAND_ORDER = config["BAND_ORDER"]
CHANGE_PERIODS = config["CHANGE_PERIODS"]

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Band numbers inside reflectance_YYYY.tif, looked up by name.
RED = BAND_ORDER.index("red") + 1
GREEN = BAND_ORDER.index("green") + 1
BLUE = BAND_ORDER.index("blue") + 1

# Note on excluded areas, repeated on every figure.
EXCLUDED = "white areas: cloud, shadow, water or outside the study boundary"


#---3. Common Land Mask---

with rasterio.open(CLEAN_DIR / "common_land.tif") as src:
    common_land = src.read(1) == 1


#---4. Reading Functions---

# One stretch is calculated across all three channels together, rather than
# per channel, so the colour balance of the scene is kept.
#
# Each year is stretched on its own values, which makes features easy to see
# but means brightness cannot be compared between panels. As noted in Lab 1,
# contrast stretching is for visual interpretation only; all measurements
# come from the unstretched reflectance.

def stretch_rgb(rgb):

    rgb = rgb.copy()

    usable = np.isfinite(rgb).all(axis=2)

    if usable.sum() == 0:
        return np.zeros_like(rgb)

    low, high = np.percentile(rgb[usable], [2, 98])

    rgb = np.clip((rgb - low) / (high - low), 0, 1)

    return np.nan_to_num(rgb)


def read_rgb(year):

    with rasterio.open(
        CLEAN_DIR / f"reflectance_{year}.tif"
    ) as src:

        red = src.read(RED)
        green = src.read(GREEN)
        blue = src.read(BLUE)

    rgb = np.stack([red, green, blue], axis=-1)

    return stretch_rgb(rgb)


def read_raster(name):

    with rasterio.open(INDICES_DIR / f"{name}.tif") as src:
        return src.read(1)


#---5. RGB Comparison---

# layout="constrained" is used on every figure instead of tight_layout,
# because tight_layout does not account for a shared colorbar and pushes it
# over the last panel.

fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

for ax, year in zip(axes, YEARS):

    ax.imshow(read_rgb(year))
    ax.set_title(str(year))
    ax.axis("off")

fig.suptitle(
    f"Hangzhou true colour, August scenes\n{EXCLUDED}",
    fontsize=13
)

output = FIGURES_DIR / "rgb_comparison.png"

plt.savefig(output, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", output.name)


#---6. Index Comparisons---

# Both index figures are built the same way, so the settings that differ are
# listed here and the loop does the rest.
# Each entry: index name, colour map, display minimum, display maximum.

index_figures = [
    ("ndvi", "YlGn", -0.2, 0.8),
    ("ndbi", "RdYlBu_r", -0.4, 0.4),
]

for name, cmap, vmin, vmax in index_figures:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    for ax, year in zip(axes, YEARS):

        data = read_raster(f"{name}_{year}")

        # Show the same pixels in every panel.
        data = np.where(common_land, data, np.nan)

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_title(str(year))
        ax.axis("off")

    bar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label(name.upper())

    fig.suptitle(
        f"{name.upper()} comparison, August scenes\n{EXCLUDED}",
        fontsize=13
    )

    output = FIGURES_DIR / f"{name}_comparison.png"

    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved:", output.name)


#---7. Change Figures---

# Differences are later year minus earlier year, so green or purple means
# the index increased. Both periods involving 2010 are shown for reference;
# the quantitative change analysis in 08_change.py uses 1995 to 2025 only.

change_figures = [
    ("ndvi", "RdYlGn"),
    ("ndbi", "PuOr"),
]

for name, cmap in change_figures:

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")

    for ax, (start, end) in zip(axes, CHANGE_PERIODS):

        data = read_raster(f"delta_{name}_{start}_{end}")

        im = ax.imshow(data, cmap=cmap, vmin=-0.4, vmax=0.4)

        ax.set_title(f"{start} to {end}")
        ax.axis("off")

    bar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    bar.set_label(f"{name.upper()} change")

    fig.suptitle(
        f"{name.upper()} change between August scenes\n{EXCLUDED}",
        fontsize=13
    )

    output = FIGURES_DIR / f"{name}_change.png"

    plt.savefig(output, dpi=200, bbox_inches="tight")
    plt.show()

    print("Saved:", output.name)


#---8. Finish---

print("\nFigures completed.")
print("Outputs saved in:", FIGURES_DIR)