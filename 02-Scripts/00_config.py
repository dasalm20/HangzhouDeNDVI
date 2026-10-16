"""
GEOG 652 - Final Project
Hangzhou, China | Landsat 1995 - 2010 - 2025

Central settings shared by every processing script. This file holds no
processing logic: it only records the study area, the scenes, the calibration
constants and the class definitions, so that the same values are used
everywhere and can be quoted directly in the report.

PROCESSING CHAIN, AND THE ENVI EQUIVALENT OF EACH STAGE

  01_get_landsat.py   download the three scenes
                      ENVI: File > Open As > Optical Sensors > Landsat
  02_inspect.py       check metadata, coverage and histograms
                      ENVI: Raster Metadata viewer, and the Histogram tool
  03_clean.py         scale to reflectance, apply QA masks, reproject
                      ENVI: Radiometric Correction > Radiometric Calibration,
                      then Reproject Raster
  04_indices.py       NDVI and NDBI, and the between-date differences
                      ENVI: Band Math, or Spectral Indices
  05_figures.py       RGB, index and change maps
                      ENVI: display groups with linked views
  06_validation.py    visual check of candidate change sites
                      ENVI: Cursor Location / Value, and linked windows
  07_classify.py      unsupervised classification of all three dates
                      ENVI Classic: Classification > Unsupervised > K-Means
  08_change.py        post-classification change detection
                      ENVI Classic: Basic Tools > Change Detection
  09_accuracy.py      error matrix, producer's and user's accuracy, Kappa
                      ENVI Classic: Post Classification > Confusion Matrix

DATA SOURCES TO CITE IN THE REPORT

  Landsat Collection 2 Level-2 surface reflectance, USGS.
    Scaling factors and QA_PIXEL bit definitions are taken from the
    Landsat Collection 2 Level-2 Science Product Guide.
  Scenes accessed through the Microsoft Planetary Computer STAC API.
  Study-area boundary from the GHSL Global Urban Centre Database R2024A V1_2,
    European Commission Joint Research Centre.
"""

from pathlib import Path

# --------------------------------------------------
# 1. PROJECT DIRECTORIES
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "01-Data"
SCRIPTS_DIR = PROJECT_DIR / "02-Scripts"
OUTPUT_DIR = PROJECT_DIR / "03-Output"

LANDSAT_DIR = DATA_DIR / "Landsat"
BOUNDARY_DIR = DATA_DIR / "Boundaries"

# One output folder per processing stage
CLEAN_DIR = OUTPUT_DIR / "Clean"                    # 03_clean.py
INDICES_DIR = OUTPUT_DIR / "Indices"                # 04_indices.py
FIGURES_DIR = OUTPUT_DIR / "Figures"                # 05_figures.py
VALIDATION_DIR = OUTPUT_DIR / "Validation"          # 06_validation.py
CLASSIFICATION_DIR = OUTPUT_DIR / "Classification"  # 07_classify.py
CHANGE_DIR = OUTPUT_DIR / "Change"                  # 08_change.py
ACCURACY_DIR = OUTPUT_DIR / "Accuracy"              # 09_accuracy.py

# --------------------------------------------------
# 2. STUDY AREA
# --------------------------------------------------

STUDY_NAME = "Hangzhou, China"

# GHSL 2015 urban centre for Hangzhou, buffered by 5 km.
# This is NOT the Hangzhou administrative municipality, and it is smaller,
# so published municipal green-coverage figures are not directly comparable.

GHSL_FILE = (
    BOUNDARY_DIR
    / "GHS_UCDB_MTUC_GLOBE_R2024A_V1_2"
    / "GHS_UCDB_MTUC_GLOBE_R2024A.gpkg"
)
GHSL_LAYER = "GHSL_UCDB_MTUC_2015_GLOBE_R2024"
GHSL_CITY_ID = 11311

BUFFER_METERS = 5000

STUDY_AREA_KM2 = 3291.58   # measured in 03_clean.py

# Analysis grid. Source scenes are EPSG:32650; everything is reprojected to
# one fixed grid so the three dates line up pixel for pixel.
TARGET_CRS = "EPSG:32651"
TARGET_RESOLUTION = 30            # metres
RASTER_SHAPE = (2751, 2409)       # rows, columns
PIXEL_AREA_KM2 = 900 / 1000000    # a 30 x 30 m pixel in square kilometres

# --------------------------------------------------
# 3. DATA SOURCE
# --------------------------------------------------

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

COLLECTION = "landsat-c2-l2"

PATH_ROW = "119/039"

# --------------------------------------------------
# 4. SELECTED LANDSAT SCENES
# --------------------------------------------------

# All three are Tier 1, same path/row, and within 10 calendar days of each
# other in August, which keeps crop stage and sun angle broadly comparable.

SCENES = {
    1995: {
        "id": "LT05_L2SP_119039_19950803_02_T1",
        "date": "1995-08-03",
        "satellite": "landsat-5",
        "tier": "T1",
    },

    2010: {
        "id": "LT05_L2SP_119039_20100812_02_T1",
        "date": "2010-08-12",
        "satellite": "landsat-5",
        "tier": "T1",
    },

    2025: {
        "id": "LC09_L2SP_119039_20250813_02_T1",
        "date": "2025-08-13",
        "satellite": "landsat-9",
        "tier": "T1",
    },
}

YEARS = [1995, 2010, 2025]

# Usable coverage of the study area after QA masking, from 03_clean.py
USABLE_COVERAGE = {1995: 98.69, 2010: 98.20, 2025: 95.34}

# --------------------------------------------------
# 5. SPECTRAL BANDS
# --------------------------------------------------

# USGS band numbering differs between sensors, so each band is looked up by
# name rather than by number.

BANDS = {
    "landsat-5": {
        "blue": "SR_B1",
        "green": "SR_B2",
        "red": "SR_B3",
        "nir": "SR_B4",
        "swir1": "SR_B5",
        "swir2": "SR_B7",
    },

    "landsat-8": {
        "blue": "SR_B2",
        "green": "SR_B3",
        "red": "SR_B4",
        "nir": "SR_B5",
        "swir1": "SR_B6",
        "swir2": "SR_B7",
    },
}

# Landsat 9 carries the same instrument layout as Landsat 8.
BANDS["landsat-9"] = BANDS["landsat-8"].copy()

# Band order inside reflectance_YYYY.tif, written by 03_clean.py.
# Scripts that read that stack index it in this order.
BAND_ORDER = ["blue", "green", "red", "nir08", "swir16", "swir22"]

# --------------------------------------------------
# 6. REFLECTANCE CALIBRATION
# --------------------------------------------------

# Surface reflectance = DN * SCALE + OFFSET
# Landsat Collection 2 Level-2 Science Product Guide, USGS.

REFLECTANCE_SCALE = 0.0000275
REFLECTANCE_OFFSET = -0.2

# --------------------------------------------------
# 7. QA MASKING
# --------------------------------------------------

# QA_PIXEL bit positions, Landsat Collection 2.
# A bit is read with a bitwise AND, for example (qa >> 3) & 1 for cloud.

QA_BITS = {
    "fill": 0,
    "dilated_cloud": 1,
    "cirrus": 2,          # Landsat 8 and 9 only
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
    "water": 7,
}

# Bits excluded from the valid-observation mask. Water is NOT excluded here:
# it is kept as its own class in the classification.
QA_EXCLUDE = ["fill", "dilated_cloud", "cloud", "cloud_shadow", "snow"]

# QA_RADSAT flagged pixels are excluded as well.

# NOTE, and a limitation for the report: the QA_PIXEL water bit is not
# consistent between dates. It flagged 105.7 km2 in 1995, 52.8 km2 in 2010
# and 124.6 km2 in 2025, and in 2010 it missed most of the Qiantang River.
# Turbid river water is apparently not reliably flagged. This was found by
# inspecting the unsupervised clusters, not by the QA band itself.

# --------------------------------------------------
# 8. SPECTRAL INDICES
# --------------------------------------------------

# NDVI = (NIR - RED) / (NIR + RED)
# NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

INDICES = ["NDVI", "NDBI"]

# All differences are computed as later year minus earlier year.
CHANGE_PERIODS = [
    (1995, 2010),
    (2010, 2025),
    (1995, 2025),
]

# Scene mean NDVI over the common-land mask, from 04_indices.py.
# The 2010 value sits about 0.045 below the other two, which is roughly a
# fifth of a class width in the classification. See section 9.
MEAN_NDVI = {1995: 0.5322, 2010: 0.4875, 2025: 0.5193}
MEAN_NDBI = {1995: -0.2751, 2010: -0.1920, 2025: -0.2061}

# NDBI is reported as an index but is NOT used to build the classification.
# The 1995 scene sits about 0.045 lower in SWIR1 at every percentile than the
# other two dates, which depresses its NDBI independently of land cover.

# --------------------------------------------------
# 9. CLASSIFICATION AND CHANGE
# --------------------------------------------------

# Vegetation density classes. They are named for density rather than land
# cover because the classification is built from NDVI alone, so a class
# describes how much vegetation a pixel holds, not what the surface is.

CLASS_NAMES = {
    1: "water",         # from the QA water mask, not from NDVI
    2: "low_veg",       # lowest NDVI, mostly built-up and bare surfaces
    3: "moderate_veg",  # built-up with scattered trees or grass
    4: "mixed_veg",     # fields, grass, open or patchy vegetation
    5: "dense_veg",     # closed canopy
}

N_CLUSTERS = 4          # four NDVI classes on land, plus water
SEED = 42               # fixed so the classification is reproducible

# Class boundaries found by the pooled K-Means fit in 07_classify.py.
# Recorded here so the classification can be quoted as plain thresholds.
NDVI_BOUNDARIES = [0.269, 0.474, 0.666]

# A class change is counted as confident only if NDVI also moved by at least
# this much. Class widths are about 0.20, so this is roughly three quarters
# of a class and removes pixels that sat on a boundary.
MIN_NDVI_CHANGE = 0.15

# 2010 IS EXCLUDED FROM THE QUANTITATIVE CHANGE ANALYSIS.
# The classification uses NDVI thresholds, so the 0.045 scene-wide NDVI
# offset in 2010 shifts class areas without any real landscape change.
# 1995 and 2025 differ by only 0.013 and are therefore comparable.
# 2010 is still classified, mapped and discussed, but not used for change
# statistics. See the report methods section.
CHANGE_PAIR = (1995, 2025)

# Accuracy assessment settings, 09_accuracy.py
N_PER_CLASS = 20        # samples per class; Lecture 6 recommends 50 to 100,
                        # so this is a stated limitation
MIN_DISTANCE = 20       # pixels between sample points, to limit spatial
                        # autocorrelation; 20 px = 600 m
SAMPLE_UNIT = 3         # a 3 x 3 block, not a single pixel

# --------------------------------------------------
# 10. CHECK THE SETUP
# --------------------------------------------------

# Run this file directly to create any missing folders and print a summary.

if __name__ == "__main__":

    folders = [
        DATA_DIR, LANDSAT_DIR, BOUNDARY_DIR, SCRIPTS_DIR, OUTPUT_DIR,
        CLEAN_DIR, INDICES_DIR, FIGURES_DIR, VALIDATION_DIR,
        CLASSIFICATION_DIR, CHANGE_DIR, ACCURACY_DIR,
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print("GEOG 652 - Hangzhou Landsat project")
    print("project folder :", PROJECT_DIR)
    print("study area     :", STUDY_NAME, "-", STUDY_AREA_KM2, "km2")
    print("grid           :", TARGET_CRS, "at", TARGET_RESOLUTION, "m")
    print("raster size    :", RASTER_SHAPE[0], "x", RASTER_SHAPE[1])

    print("\nscenes:")
    for year in YEARS:
        scene = SCENES[year]
        print("  %d  %s  %s  %s  (%.2f%% usable)"
              % (year, scene["satellite"], scene["date"], scene["id"],
                 USABLE_COVERAGE[year]))

    print("\nboundary file found:", GHSL_FILE.exists())

    print("\ndownloaded bands per year:")
    for year in YEARS:
        folder = LANDSAT_DIR / str(year)
        n = len(list(folder.glob("*.tif"))) if folder.exists() else 0
        print("  %d  %d GeoTIFFs" % (year, n))