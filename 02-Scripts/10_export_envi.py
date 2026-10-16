# Exports the cleaned rasters in ENVI format, so the analysis can be redone
# in ENVI and the results compared against the Python ones.
#
# Three things have to change for ENVI:
#   1. ENVI uses a "data ignore value" rather than NaN. NaN confuses its
#      statistics and Band Math, so masked pixels become -9999.
#   2. Band names are written into the header, so the stack opens with
#      readable band names instead of "Band 1" to "Band 6".
#   3. Class and ground-truth rasters need "file type = ENVI Classification"
#      in the header, with class names and colours, or ENVI treats them as
#      ordinary grayscale and the Confusion Matrix tool will not accept them.
#
# Writes to 03-Output/ENVI/:
#   reflectance_YYYY   six bands, named
#   ndvi_YYYY          NDVI, for the density slice
#   class_YYYY         the Python classification, for comparison
#   truth_2025         the labelled accuracy points as a ground truth image


#---1. Libraries---

from pathlib import Path
import runpy

import numpy as np
import pandas as pd
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
CLASSIFICATION_DIR = config["CLASSIFICATION_DIR"]
ACCURACY_DIR = config["ACCURACY_DIR"]
OUTPUT_DIR = config["OUTPUT_DIR"]

YEARS = config["YEARS"]
BAND_ORDER = config["BAND_ORDER"]
CLASS_NAMES = config["CLASS_NAMES"]
NDVI_BOUNDARIES = config["NDVI_BOUNDARIES"]
SAMPLE_UNIT = config["SAMPLE_UNIT"]

START, END = config["CHANGE_PAIR"]

ENVI_DIR = OUTPUT_DIR / "ENVI"
ENVI_DIR.mkdir(parents=True, exist_ok=True)

IGNORE = -9999.0

# Same colours as the Python maps, as RGB triples for the ENVI header.
CLASS_COLOURS = {
    0: (0, 0, 0),
    1: (44, 127, 184),
    2: (217, 95, 2),
    3: (253, 174, 97),
    4: (166, 217, 106),
    5: (27, 120, 55),
}


#---3. Writing Functions---

# rasterio writes the .dat and a basic .hdr. Anything ENVI-specific has to be
# added to the header afterwards, which is plain text editing.

def write_envi(name, data, profile, band_names=None):

    path = ENVI_DIR / name

    profile = profile.copy()
    profile.update(driver="ENVI", count=data.shape[0] if data.ndim == 3 else 1)

    with rasterio.open(path, "w", **profile) as dst:

        if data.ndim == 3:
            for i in range(data.shape[0]):
                dst.write(data[i], i + 1)
                if band_names:
                    dst.set_band_description(i + 1, band_names[i])
        else:
            dst.write(data, 1)

    return path


def make_classification_header(path):

    codes = sorted(CLASS_COLOURS)

    names = ["Unclassified"] + [CLASS_NAMES[c] for c in codes if c > 0]

    lookup = []
    for c in codes:
        lookup.extend(CLASS_COLOURS[c])

    header = Path(str(path) + ".hdr")

    # The file type line already exists and says "ENVI Standard", so it is
    # replaced rather than added. A duplicate key would be read unreliably.
    out_lines = []

    for line in header.read_text().splitlines():

        if line.strip().startswith("file type"):
            out_lines.append("file type = ENVI Classification")
        else:
            out_lines.append(line)

    out_lines.append(f"classes = {len(codes)}")
    out_lines.append("class names = {" + ", ".join(names) + "}")
    out_lines.append(
        "class lookup = {" + ", ".join(str(v) for v in lookup) + "}"
    )

    header.write_text("\n".join(out_lines) + "\n")


#---4. Reflectance Stacks---

# Masked pixels become -9999 so ENVI can ignore them.

for year in YEARS:

    with rasterio.open(
        CLEAN_DIR / f"reflectance_{year}.tif"
    ) as src:
        data = src.read().astype("float32")
        profile = src.profile

    data[np.isnan(data)] = IGNORE

    profile.update(dtype="float32", nodata=IGNORE, compress=None)

    # rasterio writes "data ignore value" into the header from nodata.
    path = write_envi(
        f"reflectance_{year}", data, profile, band_names=BAND_ORDER
    )

    print("Wrote:", path.name)


#---5. NDVI Rasters---

# These are the inputs for the density slice in ENVI.

for year in YEARS:

    with rasterio.open(INDICES_DIR / f"ndvi_{year}.tif") as src:
        ndvi = src.read(1).astype("float32")
        profile = src.profile

    ndvi[np.isnan(ndvi)] = IGNORE

    profile.update(dtype="float32", nodata=IGNORE, compress=None)

    path = write_envi(f"ndvi_{year}", ndvi, profile)

    print("Wrote:", path.name)


#---6. Python Classification---

# Exported so the ENVI result can be compared against it directly, for
# example with Change Detection Statistics between the two.

for year in YEARS:

    with rasterio.open(
        CLASSIFICATION_DIR / f"class_{year}.tif"
    ) as src:
        classed = src.read(1)
        profile = src.profile

    profile.update(dtype="uint8", nodata=0, compress=None)

    path = write_envi(f"class_python_{year}", classed, profile)

    make_classification_header(path)

    print("Wrote:", path.name)


#---7. Ground Truth Image---

# ENVI's Confusion Matrix can take a ground truth image instead of ROIs.
# Each accuracy sample is painted as a 3 x 3 block carrying the class that
# was read by eye, and everything else stays unclassified.

truth_year = END

points_file = ACCURACY_DIR / f"accuracy_points_{truth_year}.csv"

if not points_file.exists():

    print("\nNo labelled points found, skipping the ground truth image.")

else:

    # Numbers puts the table name on the first line, above the real header.
    first_line = open(points_file).readline()

    if first_line.startswith("point_id"):
        points = pd.read_csv(points_file)
    else:
        points = pd.read_csv(points_file, skiprows=1)

    points = points[points["reference_class"].notna()]

    points["reference_class"] = (
        points["reference_class"]
        .astype(str)
        .str.split("\t").str[0]
        .str.strip()
        .str.lower()
    )

    code_of_name = {name: code for code, name in CLASS_NAMES.items()}

    points = points[points["reference_class"].isin(code_of_name)]

    with rasterio.open(
        CLASSIFICATION_DIR / f"class_{truth_year}.tif"
    ) as src:
        profile = src.profile

    truth = np.zeros((profile["height"], profile["width"]), dtype="uint8")

    half = SAMPLE_UNIT // 2

    for i in range(len(points)):

        r = int(points["row"].iloc[i])
        c = int(points["col"].iloc[i])

        code = code_of_name[points["reference_class"].iloc[i]]

        truth[r - half:r + half + 1, c - half:c + half + 1] = code

    profile.update(dtype="uint8", nodata=0, compress=None)

    path = write_envi(f"truth_{truth_year}", truth, profile)

    make_classification_header(path)

    print("Wrote:", path.name)
    print(f"  {len(points)} points as {SAMPLE_UNIT}x{SAMPLE_UNIT} blocks")


#---8. Values to Check Against---

# The numbers the ENVI run should reproduce, written next to the exports so
# they are easy to compare while working.

notes = ENVI_DIR / "expected_values.txt"

lines = [
    "Values to reproduce in ENVI",
    "",
    "Density slice thresholds on NDVI:",
    f"  class 2 low_veg       NDVI below {NDVI_BOUNDARIES[0]}",
    f"  class 3 moderate_veg  {NDVI_BOUNDARIES[0]} to {NDVI_BOUNDARIES[1]}",
    f"  class 4 mixed_veg     {NDVI_BOUNDARIES[1]} to {NDVI_BOUNDARIES[2]}",
    f"  class 5 dense_veg     above {NDVI_BOUNDARIES[2]}",
    "  water comes from water_YYYY, not from NDVI",
    "",
    "Accuracy assessment, 2025, 100 points:",
    "  overall accuracy  80.00 percent",
    "  Kappa             0.7500",
    "",
    "Class areas on the common mask, km2:",
]

areas_file = OUTPUT_DIR / "Change" / "class_areas_common.csv"

if areas_file.exists():

    areas = pd.read_csv(areas_file)

    table = areas.pivot(
        index="class_name", columns="year", values="area_km2"
    )

    lines.append(table.to_string())

else:
    lines.append("  run 08_change.py first")

notes.write_text("\n".join(lines) + "\n")

print("\nWrote:", notes.name)


#---9. Finish---

print("\nENVI export complete.")
print("Files saved in:", ENVI_DIR)
print("\nOpen the .hdr files in ENVI, not the .dat files.")