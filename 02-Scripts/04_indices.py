# Calculates NDVI and NDBI for each year, and the difference between dates.
#
# NDVI = (NIR - Red) / (NIR + Red)          vegetation
# NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)      built-up and bare surfaces
#
# Both are normalised difference ratios, so they share one helper function.
# Differences are always later year minus earlier year, so a positive value
# means the index increased.
#
# ENVI equivalent: Band Math, or the Spectral Indices tool.


#---1. Libraries---

from pathlib import Path
import runpy

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

YEARS = config["YEARS"]
BAND_ORDER = config["BAND_ORDER"]
CHANGE_PERIODS = config["CHANGE_PERIODS"]

INDICES_DIR.mkdir(parents=True, exist_ok=True)

# Band numbers inside reflectance_YYYY.tif. Looked up by name so the numbers
# stay correct if the band order ever changes.
RED = BAND_ORDER.index("red") + 1
NIR = BAND_ORDER.index("nir08") + 1
SWIR1 = BAND_ORDER.index("swir16") + 1


#---3. Common Land Mask---

# All statistics are taken over this mask, so every year is summarised on
# exactly the same pixels.

with rasterio.open(CLEAN_DIR / "common_land.tif") as src:
    common_land = src.read(1) == 1

print("Common land pixels:", common_land.sum())


#---4. Index Function---

# Both indices have the form (a - b) / (a + b).
#
# np.divide with "where" skips the pixels where the denominator is zero or
# missing, instead of dividing and producing a warning. Those pixels keep the
# NaN they were filled with.

def normalised_difference(a, b):

    result = np.full(a.shape, np.nan, dtype="float32")

    denominator = a + b

    usable = (
        np.isfinite(denominator)
        & (np.abs(denominator) > 1e-6)
    )

    np.divide(a - b, denominator, out=result, where=usable)

    return result


#---5. Calculate NDVI and NDBI---

indices = {}

for year in YEARS:

    print(f"\nCalculating indices for {year}...")

    with rasterio.open(
        CLEAN_DIR / f"reflectance_{year}.tif"
    ) as src:

        red = src.read(RED)
        nir = src.read(NIR)
        swir1 = src.read(SWIR1)

        profile = src.profile.copy()

    ndvi = normalised_difference(nir, red)
    ndbi = normalised_difference(swir1, nir)

    indices[year] = {"ndvi": ndvi, "ndbi": ndbi}

    # One band out, same grid as the input.
    profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3
    )

    for name, data in indices[year].items():

        output = INDICES_DIR / f"{name}_{year}.tif"

        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data, 1)

        values = data[common_land & np.isfinite(data)]

        print(
            f"{name.upper()}: "
            f"mean={np.mean(values):.4f}, "
            f"median={np.median(values):.4f}"
        )

        # An index of this form cannot fall outside [-1, 1]. A handful of
        # pixels do, from rounding in the reflectance values. They are
        # reported here and excluded in 07_classify.py.
        outside = np.sum((values < -1) | (values > 1))

        if outside:
            print(f"  Values outside [-1, 1]: {outside:,}")


#---6. Calculate Changes---

print("\nChange between dates")

for start, end in CHANGE_PERIODS:

    print(f"\n{start} to {end}")

    for name in ["ndvi", "ndbi"]:

        before = indices[start][name]
        after = indices[end][name]

        difference = after - before

        # Keep only pixels that are land in all three years and have a value
        # at both dates.
        usable = (
            common_land
            & np.isfinite(before)
            & np.isfinite(after)
        )

        difference[~usable] = np.nan

        output = INDICES_DIR / f"delta_{name}_{start}_{end}.tif"

        with rasterio.open(output, "w", **profile) as dst:
            dst.write(difference, 1)

        values = difference[usable]

        print(
            f"{name.upper()} change: "
            f"mean={np.mean(values):+.4f}, "
            f"median={np.median(values):+.4f}"
        )


#---7. Finish---

print("\nIndex calculations complete.")
print("Outputs saved in:", INDICES_DIR)