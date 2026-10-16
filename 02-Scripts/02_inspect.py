# Checks the downloaded files before any processing: grid alignment, raw DN
# ranges, cloud and snow percentages, and a quick RGB preview of each year.
# Nothing is written except one figure. The Landsat files are not modified.
#
# ENVI equivalent: the Raster Metadata viewer for the file details, and
# Statistics or the Histogram tool for the DN ranges.


#---1. Libraries---

from pathlib import Path
import runpy

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.features import geometry_mask


#---2. Configuration---

# A module name cannot start with a digit, so "00_config.py" is loaded with
# runpy instead of a normal import.
# https://docs.python.org/3/library/runpy.html

config = runpy.run_path(
    str(Path(__file__).with_name("00_config.py"))
)

LANDSAT_DIR = config["LANDSAT_DIR"]
FIGURES_DIR = config["FIGURES_DIR"]

GHSL_FILE = config["GHSL_FILE"]
GHSL_LAYER = config["GHSL_LAYER"]
GHSL_CITY_ID = config["GHSL_CITY_ID"]
BUFFER_METERS = config["BUFFER_METERS"]
TARGET_CRS = config["TARGET_CRS"]

YEARS = config["YEARS"]
SCENES = config["SCENES"]
QA_BITS = config["QA_BITS"]

SCALE = config["REFLECTANCE_SCALE"]
OFFSET = config["REFLECTANCE_OFFSET"]

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


#---3. Study Area---

urban = gpd.read_file(
    GHSL_FILE,
    layer=GHSL_LAYER,
    where=f"ID_UC_G0 = {GHSL_CITY_ID}"
).to_crs(TARGET_CRS)

study_area = urban.geometry.iloc[0].buffer(BUFFER_METERS)


#---4. File Details and DN Ranges---

# Every band of a year should share the same grid. The first band read sets
# the reference and the rest are compared against it.

bands = [
    "blue", "green", "red",
    "nir08", "swir16", "swir22",
    "qa_pixel", "qa_radsat"
]

for year in YEARS:

    folder = LANDSAT_DIR / str(year)

    print(f"\n--- {year} ---")

    reference = None

    for band in bands:

        with rasterio.open(folder / f"{band}.tif") as src:

            info = (src.crs, src.width, src.height, src.transform)

            if reference is None:

                reference = info

                print("CRS:", src.crs)
                print("Size:", src.width, "x", src.height)
                print("Resolution:", src.res)
                print("NoData:", src.nodata)

            if info != reference:
                print("WARNING: misaligned band:", band)

            # Read every fifth pixel. Enough to see the DN range without
            # loading the full band.
            if band not in ("qa_pixel", "qa_radsat"):

                sample = src.read(
                    1,
                    out_shape=(
                        max(1, src.height // 5),
                        max(1, src.width // 5)
                    ),
                    resampling=Resampling.nearest
                )

                # 0 is fill and 65535 is saturated, so both are dropped.
                valid = sample[(sample > 0) & (sample < 65535)]

                if valid.size:
                    print(
                        f"{band}: DN median={np.median(valid):.0f}, "
                        f"min={valid.min()}, max={valid.max()}"
                    )


#---5. Quality Flags---

# QA_PIXEL packs each flag into one bit, so a flag is read by testing that
# bit. Bit positions are listed in 00_config.py.
# Source: Landsat Collection 2 Level-2 Science Product Guide, USGS.

for year in YEARS:

    folder = LANDSAT_DIR / str(year)

    with rasterio.open(folder / "qa_pixel.tif") as src:

        qa = src.read(1)

        # The study area, converted to the scene's CRS and turned into a mask
        # so percentages cover the study area only, not the whole scene.
        roi = gpd.GeoSeries(
            [study_area], crs=TARGET_CRS
        ).to_crs(src.crs).iloc[0]

        inside = geometry_mask(
            [roi.__geo_interface__],
            out_shape=qa.shape,
            transform=src.transform,
            invert=True
        )

    fill = (qa & (1 << QA_BITS["fill"])) != 0
    dilated = (qa & (1 << QA_BITS["dilated_cloud"])) != 0
    cloud = (qa & (1 << QA_BITS["cloud"])) != 0
    shadow = (qa & (1 << QA_BITS["cloud_shadow"])) != 0
    snow = (qa & (1 << QA_BITS["snow"])) != 0
    water = (qa & (1 << QA_BITS["water"])) != 0

    # The cirrus bit only exists on Landsat 8 and 9.
    if SCENES[year]["satellite"] in ("landsat-8", "landsat-9"):
        cirrus = (qa & (1 << QA_BITS["cirrus"])) != 0
    else:
        cirrus = np.zeros_like(qa, dtype=bool)

    bad = fill | dilated | cloud | shadow | snow | cirrus
    usable = inside & ~bad

    total = inside.sum()

    print(f"\n--- {year} quality assessment ---")
    print(f"Cloud:    {(inside & cloud).sum() / total * 100:.2f}%")
    print(f"Shadow:   {(inside & shadow).sum() / total * 100:.2f}%")
    print(f"Snow/Ice: {(inside & snow).sum() / total * 100:.2f}%")
    print(f"Water:    {(inside & water).sum() / total * 100:.2f}%")
    print(f"Usable:   {usable.sum() / total * 100:.2f}%")

    with rasterio.open(folder / "qa_radsat.tif") as src:
        radsat = src.read(1)

    saturated = inside & (radsat != 0)

    print(f"QA_RADSAT flagged: {saturated.sum() / total * 100:.2f}%")


#---6. RGB Previews---

# Quarter-resolution previews, one per year, each stretched on its own 2nd
# to 98th percentile. Display only: the reflectance values calculated here
# are not saved or reused.

previews = {}

for year in YEARS:

    folder = LANDSAT_DIR / str(year)
    channels = []

    for band in ["red", "green", "blue"]:

        with rasterio.open(folder / f"{band}.tif") as src:

            data = src.read(
                1,
                out_shape=(
                    max(1, src.height // 4),
                    max(1, src.width // 4)
                ),
                resampling=Resampling.nearest
            ).astype("float32")

        data[data == 0] = np.nan
        data = data * SCALE + OFFSET

        channels.append(data)

    rgb = np.stack(channels, axis=-1)

    valid = np.isfinite(rgb).all(axis=2)
    low, high = np.percentile(rgb[valid], [2, 98])

    rgb = np.clip((rgb - low) / (high - low), 0, 1)

    previews[year] = np.nan_to_num(rgb)


#---7. Snow Flag Check---

# Snow should not appear in an August scene. This panel overlays the snow
# flag on the earliest year to confirm the flag is not being set over water
# or bright surfaces, which it was in the earlier winter scenes.

check_year = YEARS[0]

folder = LANDSAT_DIR / str(check_year)

with rasterio.open(folder / "qa_pixel.tif") as src:

    qa_small = src.read(
        1,
        out_shape=previews[check_year].shape[:2],
        resampling=Resampling.nearest
    )

snow_small = (qa_small & (1 << QA_BITS["snow"])) != 0

print(f"\n{check_year} snow-flagged pixels in preview:", snow_small.sum())


#---8. Inspection Figure---

fig, axes = plt.subplots(2, 2, figsize=(11, 10))

for ax, year in zip(axes.flat[:3], YEARS):

    ax.imshow(previews[year])
    ax.set_title(f"Hangzhou {year}")
    ax.axis("off")

ax = axes[1, 1]

ax.imshow(previews[check_year])

# Masked so only the flagged pixels are drawn in red.
overlay = np.ma.masked_where(~snow_small, snow_small.astype(float))

ax.imshow(
    overlay,
    cmap="Reds",
    vmin=0, vmax=1,
    alpha=0.65,
    interpolation="nearest"
)

ax.set_title(f"{check_year} snow/ice QA flags (red)")
ax.axis("off")

plt.tight_layout()

output = FIGURES_DIR / "landsat_inspection.png"

plt.savefig(output, dpi=180)
plt.show()


#---9. Finish---

print("\nInspection complete.")
print("Figure saved:", output.name)
print("Original Landsat files unchanged.")