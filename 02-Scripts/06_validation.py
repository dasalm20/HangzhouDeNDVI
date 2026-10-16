# Picks a small set of locations to inspect by eye, and saves a nine-panel
# figure for each one: RGB, NDVI and NDBI across the three years.
#
# The thresholds below find pixels whose indices changed in a particular way.
# They are a way of finding candidates to look at, not a classification, so
# every site still has to be interpreted manually in validation_sites.csv.
#
# ENVI equivalent: linked display windows with Cursor Location / Value.


#---1. Libraries---

from pathlib import Path
import csv
import runpy

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.warp import transform as transform_coords


#---2. Configuration---

# A module name cannot start with a digit, so "00_config.py" is loaded with
# runpy instead of a normal import.
# https://docs.python.org/3/library/runpy.html

config = runpy.run_path(
    str(Path(__file__).with_name("00_config.py"))
)

CLEAN_DIR = config["CLEAN_DIR"]
INDICES_DIR = config["INDICES_DIR"]
VALIDATION_DIR = config["VALIDATION_DIR"]

YEARS = config["YEARS"]
BAND_ORDER = config["BAND_ORDER"]
SEED = config["SEED"]

FIGURES_DIR = VALIDATION_DIR / "Figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Band numbers inside reflectance_YYYY.tif, looked up by name.
RED = BAND_ORDER.index("red") + 1
GREEN = BAND_ORDER.index("green") + 1
BLUE = BAND_ORDER.index("blue") + 1

HALF = 40              # each figure covers 81 x 81 pixels, about 2.4 km
MIN_DISTANCE = 120     # pixels between sites, so they are spread out
PER_CATEGORY = 3       # sites to keep per category


#---3. Load Data---

with rasterio.open(CLEAN_DIR / "common_land.tif") as src:
    common_land = src.read(1) == 1
    grid = src.transform
    crs = src.crs

with rasterio.open(
    INDICES_DIR / "delta_ndvi_1995_2010.tif"
) as src:
    ndvi_9510 = src.read(1)

with rasterio.open(
    INDICES_DIR / "delta_ndvi_2010_2025.tif"
) as src:
    ndvi_1025 = src.read(1)

with rasterio.open(
    INDICES_DIR / "delta_ndbi_2010_2025.tif"
) as src:
    ndbi_1025 = src.read(1)


#---4. Candidate Locations---

# Four patterns worth looking at. NDVI down with NDBI up suggests building
# over vegetation; both down is ambiguous and needs the imagery to resolve.

usable = (
    common_land
    & np.isfinite(ndvi_9510)
    & np.isfinite(ndvi_1025)
    & np.isfinite(ndbi_1025)
)

categories = {

    "Possible development": (
        usable
        & (ndvi_1025 <= -0.15)
        & (ndbi_1025 >= 0.08)
    ),

    "Vegetation increase": (
        usable
        & (ndvi_9510 >= 0.25)
    ),

    "Ambiguous decrease": (
        usable
        & (ndvi_1025 <= -0.15)
        & (ndbi_1025 <= -0.03)
    ),

    "Little change": (
        usable
        & (np.abs(ndvi_1025) <= 0.02)
        & (np.abs(ndbi_1025) <= 0.02)
    )
}


#---5. Select Sites---

rng = np.random.default_rng(SEED)

sites = []

for category, candidate_mask in categories.items():

    candidates = np.argwhere(candidate_mask)

    if len(candidates) == 0:
        print(category, ": no candidates found")
        continue

    order = rng.permutation(len(candidates))

    selected = 0

    for index in order:

        row, col = candidates[index]

        # Leave room for the figure window.
        if (
            row < HALF
            or col < HALF
            or row + HALF >= common_land.shape[0]
            or col + HALF >= common_land.shape[1]
        ):
            continue

        # Skip isolated valid pixels surrounded by masked ones.
        local = common_land[row - 5:row + 6, col - 5:col + 6]

        if local.mean() < 0.85:
            continue

        # The pattern should hold across a small patch, not one pixel.
        nearby = candidate_mask[row - 2:row + 3, col - 2:col + 3]

        if nearby.mean() < 0.60:
            continue

        # Keep the sites spread across the study area.
        too_close = any(
            np.hypot(row - s["row"], col - s["col"]) < MIN_DISTANCE
            for s in sites
        )

        if too_close:
            continue

        # Pixel centre in map coordinates, then in latitude and longitude so
        # the site can be found in Google Earth.
        x, y = grid * (col + 0.5, row + 0.5)

        lon, lat = transform_coords(crs, "EPSG:4326", [x], [y])

        sites.append({
            "id": f"S{len(sites) + 1:02d}",
            "category": category,
            "row": int(row),
            "col": int(col),
            "longitude": round(lon[0], 6),
            "latitude": round(lat[0], 6),
            "ndvi_change_1995_2010": round(float(ndvi_9510[row, col]), 4),
            "ndvi_change_2010_2025": round(float(ndvi_1025[row, col]), 4),
            "ndbi_change_2010_2025": round(float(ndbi_1025[row, col]), 4)
        })

        selected += 1

        if selected == PER_CATEGORY:
            break

    print(category, ":", selected, "locations")


#---6. Reading Functions---

# One fixed display stretch is used for all three years here, unlike
# 05_figures.py, so that brightness can be compared between the panels of
# the same site.

def read_rgb(year, window):

    with rasterio.open(
        CLEAN_DIR / f"reflectance_{year}.tif"
    ) as src:

        rgb = src.read(
            [RED, GREEN, BLUE], window=window
        ).transpose(1, 2, 0)

    rgb = np.clip(rgb / 0.35, 0, 1) ** (1 / 1.8)

    return np.nan_to_num(rgb)


def read_index(name, year, window):

    with rasterio.open(
        INDICES_DIR / f"{name}_{year}.tif"
    ) as src:
        return src.read(1, window=window)


#---7. Site Figures---

for site in sites:

    window = Window(
        site["col"] - HALF,
        site["row"] - HALF,
        2 * HALF + 1,
        2 * HALF + 1
    )

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))

    for j, year in enumerate(YEARS):

        axes[0, j].imshow(read_rgb(year, window))

        im_ndvi = axes[1, j].imshow(
            read_index("ndvi", year, window),
            cmap="YlGn", vmin=-0.2, vmax=0.8
        )

        im_ndbi = axes[2, j].imshow(
            read_index("ndbi", year, window),
            cmap="RdYlBu_r", vmin=-0.4, vmax=0.4
        )

        axes[0, j].set_title(str(year))

        for i in range(3):

            # Mark the exact pixel that was selected.
            axes[i, j].plot(
                HALF, HALF, "+",
                color="magenta",
                markersize=9,
                markeredgewidth=1.3
            )

            axes[i, j].axis("off")

    # Row labels are drawn as figure text because axis("off") also hides
    # the usual axis labels.
    for y, label in zip([0.74, 0.47, 0.20], ["RGB", "NDVI", "NDBI"]):
        fig.text(
            0.02, y, label,
            rotation=90,
            va="center",
            fontsize=11
        )

    fig.suptitle(
        f"{site['id']} | {site['category']}\n"
        f"{site['latitude']}, {site['longitude']}",
        fontsize=13
    )

    # Space reserved on the right so the colorbars sit outside the panels.
    fig.subplots_adjust(
        left=0.06, right=0.89,
        top=0.88, bottom=0.06,
        wspace=0.04, hspace=0.10
    )

    cax1 = fig.add_axes([0.91, 0.38, 0.012, 0.20])
    cax2 = fig.add_axes([0.91, 0.10, 0.012, 0.20])

    fig.colorbar(im_ndvi, cax=cax1, label="NDVI")
    fig.colorbar(im_ndbi, cax=cax2, label="NDBI")

    output = FIGURES_DIR / f"{site['id']}.png"

    plt.savefig(output, dpi=160)
    plt.close(fig)

    print("Saved:", output.name)


#---8. Review Table---

# The last four columns are left blank and filled in by hand after looking
# at each figure.

output = VALIDATION_DIR / "validation_sites.csv"

fields = [
    "id",
    "category",
    "longitude",
    "latitude",
    "ndvi_change_1995_2010",
    "ndvi_change_2010_2025",
    "ndbi_change_2010_2025",
    "my_interpretation",
    "confidence",
    "reference_imagery",
    "notes"
]

with open(output, "w", newline="") as file:

    writer = csv.DictWriter(
        file, fieldnames=fields, extrasaction="ignore"
    )

    writer.writeheader()
    writer.writerows(sites)


#---9. Finish---

print("\nValidation preparation complete.")
print("Locations selected:", len(sites))
print("Figures:", FIGURES_DIR)
print("Review table:", output.name)
print("Manual interpretation pending.")