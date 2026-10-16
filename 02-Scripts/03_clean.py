# Turns the raw downloads into analysis-ready rasters. For each year it
# reprojects every band onto one shared 30 m grid, converts DN to surface
# reflectance, and masks out cloud, shadow, snow and saturated pixels.
#
# Writes per year:
#   reflectance_YYYY.tif  six bands, masked, float
#   valid_YYYY.tif        1 where the pixel is usable
#   water_YYYY.tif        1 where QA_PIXEL flagged water
# And once:
#   common_land.tif       land pixels usable in all three years
#
# ENVI equivalent: Radiometric Correction > Radiometric Calibration for the
# scaling, Reproject Raster for the grid, and a mask band built from QA_PIXEL.


#---1. Libraries---

from pathlib import Path
import runpy

import geopandas as gpd
import numpy as np
import rasterio

from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject


#---2. Configuration---

# A module name cannot start with a digit, so "00_config.py" is loaded with
# runpy instead of a normal import.
# https://docs.python.org/3/library/runpy.html

config = runpy.run_path(
    str(Path(__file__).with_name("00_config.py"))
)

LANDSAT_DIR = config["LANDSAT_DIR"]
CLEAN_DIR = config["CLEAN_DIR"]

GHSL_FILE = config["GHSL_FILE"]
GHSL_LAYER = config["GHSL_LAYER"]
GHSL_CITY_ID = config["GHSL_CITY_ID"]
BUFFER_METERS = config["BUFFER_METERS"]

TARGET_CRS = config["TARGET_CRS"]
TARGET_RESOLUTION = config["TARGET_RESOLUTION"]

YEARS = config["YEARS"]
SCENES = config["SCENES"]
QA_BITS = config["QA_BITS"]
BAND_ORDER = config["BAND_ORDER"]

SCALE = config["REFLECTANCE_SCALE"]
OFFSET = config["REFLECTANCE_OFFSET"]

CLEAN_DIR.mkdir(parents=True, exist_ok=True)


#---3. Study Area and Grid---

urban = gpd.read_file(
    GHSL_FILE,
    layer=GHSL_LAYER,
    where=f"ID_UC_G0 = {GHSL_CITY_ID}"
).to_crs(TARGET_CRS)

study_area = urban.geometry.iloc[0].buffer(BUFFER_METERS)

resolution = TARGET_RESOLUTION

# Snap the study-area bounds outward to whole 30 m steps. This fixes one grid
# for all three years, so the dates line up pixel for pixel.
xmin, ymin, xmax, ymax = study_area.bounds

left = np.floor(xmin / resolution) * resolution
bottom = np.floor(ymin / resolution) * resolution
right = np.ceil(xmax / resolution) * resolution
top = np.ceil(ymax / resolution) * resolution

width = int(round((right - left) / resolution))
height = int(round((top - bottom) / resolution))

transform = from_origin(left, top, resolution, resolution)

# Mark which pixels of that grid fall inside the study-area polygon.
inside = geometry_mask(
    [study_area.__geo_interface__],
    out_shape=(height, width),
    transform=transform,
    invert=True
)

print("Study area:", round(study_area.area / 1e6, 2), "km²")
print("Grid:", width, "x", height)
print("Resolution:", resolution, "m")


#---4. Reprojection---

# Reads one band and resamples it onto the shared grid.
#
# Nearest neighbour is used on purpose. As found in Lab 2, it is the only
# resampling method that keeps the original DN values; bilinear and cubic
# convolution average neighbouring pixels and change them. Original values
# must be preserved for the index calculations and the classification.
#
# Reference: rasterio.warp.reproject
# https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html

def read_aligned(path):

    with rasterio.open(path) as src:

        data = np.zeros((height, width), dtype="uint16")

        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=TARGET_CRS,
            src_nodata=src.nodata,
            dst_nodata=0,
            resampling=Resampling.nearest
        )

    return data


#---5. Output Settings---

# predictor is a GeoTIFF compression setting: 3 suits floating point and
# 2 suits integers.

profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 6,
    "dtype": "float32",
    "crs": TARGET_CRS,
    "transform": transform,
    "nodata": np.nan,
    "compress": "deflate",
    "predictor": 3
}

mask_profile = profile.copy()
mask_profile.update(
    count=1,
    dtype="uint8",
    nodata=None,
    predictor=2
)

valid_masks = []
water_masks = []


#---6. Clean Each Year---

for year in YEARS:

    print(f"\nCleaning {year}...")

    folder = LANDSAT_DIR / str(year)

    qa = read_aligned(folder / "qa_pixel.tif")
    radsat = read_aligned(folder / "qa_radsat.tif")

    # QA_PIXEL packs each flag into one bit, so a flag is read by testing
    # that bit. Bit positions are listed in 00_config.py.
    # Source: Landsat Collection 2 Level-2 Science Product Guide, USGS.
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

    # Water is deliberately not excluded here. It is kept as its own class
    # in the classification, so it has to stay in the valid mask.
    bad = (
        fill | dilated | cloud |
        shadow | snow | cirrus |
        (radsat != 0)
    )

    valid = inside & ~bad

    # Read and scale the six optical bands. Each band also narrows the valid
    # mask, so a pixel missing in any one band is dropped from all of them.
    reflectance = []

    for band in BAND_ORDER:

        raw = read_aligned(folder / f"{band}.tif")

        # 0 is fill and 65535 is saturated.
        valid &= (raw > 0) & (raw < 65535)

        data = raw.astype("float32") * SCALE + OFFSET

        reflectance.append(data)

    # The mask is applied at write time, once every band has had its say.
    output = CLEAN_DIR / f"reflectance_{year}.tif"

    with rasterio.open(output, "w", **profile) as dst:

        for i, data in enumerate(reflectance, start=1):

            data[~valid] = np.nan

            dst.write(data, i)
            dst.set_band_description(i, BAND_ORDER[i - 1])

    with rasterio.open(
        CLEAN_DIR / f"valid_{year}.tif",
        "w", **mask_profile
    ) as dst:
        dst.write(valid.astype("uint8"), 1)

    # Water that was also a usable observation.
    water_valid = water & valid

    with rasterio.open(
        CLEAN_DIR / f"water_{year}.tif",
        "w", **mask_profile
    ) as dst:
        dst.write(water_valid.astype("uint8"), 1)

    valid_masks.append(valid)
    water_masks.append(water)

    coverage = valid.sum() / inside.sum() * 100
    water_area = water_valid.sum() * resolution ** 2 / 1e6

    print(f"Usable coverage: {coverage:.2f}%")
    print(f"Water flagged: {round(water_area, 2)} km²")
    print("Saved:", output.name)


#---7. Common Land Mask---

# Pixels usable in every year and never flagged as water. Used wherever the
# three dates are compared, so that every date covers the same ground.
#
# Water is excluded if it was flagged in ANY year, which is the conservative
# choice: it drops shoreline pixels but avoids comparing land against water.

common_valid = np.logical_and.reduce(valid_masks)
water_any = np.logical_or.reduce(water_masks)

common_land = common_valid & ~water_any

with rasterio.open(
    CLEAN_DIR / "common_land.tif",
    "w", **mask_profile
) as dst:
    dst.write(common_land.astype("uint8"), 1)

common_area = common_land.sum() * resolution ** 2 / 1e6

print("\nCommon land area:", round(common_area, 2), "km²")
print(
    "Common land coverage:",
    f"{common_land.sum() / inside.sum() * 100:.2f}%"
)


#---8. Finish---

print("\nCleaning complete.")
print("Outputs saved in:", CLEAN_DIR)