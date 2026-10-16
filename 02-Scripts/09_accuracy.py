# Accuracy assessment of the classification, following Lecture 6.
#
# Run it twice. The first run draws the sample and writes contact sheets plus
# a blank table. Fill in the reference_class column by looking at the sheets,
# then run it again to get the error matrix and Kappa.
#
# Design choices, all from Lecture 6:
#   Sampling scheme   stratified random, so every class gets a fixed number
#                     of samples; simple random under-samples small classes.
#   Sample size       20 per class, 100 in total. The lecture recommends 50
#                     to 100 per class and the binomial formula gives 204
#                     points, so this sample is smaller and the estimate is
#                     correspondingly less precise. Stated as a limitation.
#   Sample unit       a 3 x 3 block, not a single pixel, which the lecture
#                     calls a poor choice because one pixel cannot be lined
#                     up reliably with the reference data.
#   Autocorrelation   a minimum distance between points, which the lecture
#                     gives as the way to reduce it.
#
# Limitation to state in the report: the reference labels are read from the
# same imagery that was classified, so this measures agreement between the
# classifier and the analyst, not agreement with the ground. Independent
# reference data would need high resolution imagery from the same dates,
# which does not exist for 1995.
#
# ENVI equivalent: ENVI Classic > Classification > Post Classification >
# Confusion Matrix, using ground truth ROIs instead of sample points.


#---1. Libraries---

from pathlib import Path
import runpy

import matplotlib.pyplot as plt
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
CLASSIFICATION_DIR = config["CLASSIFICATION_DIR"]
ACCURACY_DIR = config["ACCURACY_DIR"]

CLASS_NAMES = config["CLASS_NAMES"]
BAND_ORDER = config["BAND_ORDER"]
N_PER_CLASS = config["N_PER_CLASS"]
MIN_DISTANCE = config["MIN_DISTANCE"]
SAMPLE_UNIT = config["SAMPLE_UNIT"]
SEED = config["SEED"]

YEAR = 2025          # change to 1995 to assess the earlier date

PER_SHEET = 20       # thumbnails per contact sheet
CHIP_HALF = 20       # each thumbnail is 41 x 41 pixels, about 1.2 km

SHEETS_DIR = ACCURACY_DIR / "Sheets"
SHEETS_DIR.mkdir(parents=True, exist_ok=True)

NAMES = [CLASS_NAMES[code] for code in sorted(CLASS_NAMES)]

# Band numbers inside reflectance_YYYY.tif, looked up by name.
GREEN = BAND_ORDER.index("green") + 1
RED = BAND_ORDER.index("red") + 1
NIR = BAND_ORDER.index("nir08") + 1


#---3. Draw the Sample---

with rasterio.open(
    CLASSIFICATION_DIR / f"class_{YEAR}.tif"
) as src:
    classed = src.read(1)
    grid = src.transform

rows, cols = classed.shape

# A point is only accepted if its whole 3 x 3 block is one class, so each
# sample sits inside a uniform patch. Shifting the array by one pixel in
# every direction and comparing finds those blocks without looping.
homogeneous = np.ones(classed.shape, dtype=bool)

for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        shifted = np.roll(np.roll(classed, dr, axis=0), dc, axis=1)
        homogeneous = homogeneous & (shifted == classed)

# Stay clear of the edges so every thumbnail fits inside the image.
away_from_edge = np.zeros(classed.shape, dtype=bool)
away_from_edge[
    CHIP_HALF:rows - CHIP_HALF,
    CHIP_HALF:cols - CHIP_HALF
] = True

eligible = homogeneous & away_from_edge & (classed > 0)

print("Eligible pixels:", eligible.sum())

np.random.seed(SEED)

points = []

for code in sorted(CLASS_NAMES):

    candidate_rows, candidate_cols = np.where(eligible & (classed == code))

    if len(candidate_rows) == 0:
        print("class", CLASS_NAMES[code], "has no eligible pixels, skipped")
        continue

    # Shuffle, then take points one at a time, skipping any that fall too
    # close to a point already taken.
    shuffle = np.random.permutation(len(candidate_rows))

    taken_rows = []
    taken_cols = []

    for k in shuffle:

        r = candidate_rows[k]
        c = candidate_cols[k]

        if len(taken_rows) > 0:

            dr = np.array(taken_rows) - r
            dc = np.array(taken_cols) - c

            if np.min(dr * dr + dc * dc) < MIN_DISTANCE ** 2:
                continue

        taken_rows.append(r)
        taken_cols.append(c)

        if len(taken_rows) >= N_PER_CLASS:
            break

    print(CLASS_NAMES[code], "points drawn:", len(taken_rows))

    for i in range(len(taken_rows)):

        x, y = rasterio.transform.xy(grid, taken_rows[i], taken_cols[i])

        points.append({
            "point_id": "",
            "row": int(taken_rows[i]),
            "col": int(taken_cols[i]),
            "x": round(x, 1),
            "y": round(y, 1),
            "map_class": CLASS_NAMES[code],
            "reference_class": "",
            "confidence": "",
            "note": ""
        })

sample = pd.DataFrame(points)
sample["point_id"] = ["P%03d" % (i + 1) for i in range(len(sample))]

print("\nTotal points:", len(sample))


#---4. Contact Sheets---

# Thumbnails are false colour, with NIR in red. As seen in Lab 1 the near
# infrared separates vegetation far better than the visible bands, so
# vegetation appears red, built-up and bare surfaces grey or cyan, and water
# black or dark blue.

with rasterio.open(
    CLEAN_DIR / f"reflectance_{YEAR}.tif"
) as src:
    bands = src.read().astype("float32")


# Each thumbnail is stretched on its own values, because the fixed whole
# scene stretch is too flat at this size.
#
# Masked pixels are stored as NaN. np.percentile returns NaN if any pixel in
# the block is NaN, which left whole thumbnails blank, so nanpercentile is
# used and the masked pixels are drawn mid grey instead.

def stretch(block):

    out = np.zeros(block.shape, dtype="float32")

    missing = np.isnan(block).any(axis=2)

    for b in range(block.shape[2]):

        layer = block[:, :, b]

        if np.all(np.isnan(layer)):
            out[:, :, b] = 0.5
            continue

        low = np.nanpercentile(layer, 2)
        high = np.nanpercentile(layer, 98)

        if high <= low:
            high = low + 0.0001

        scaled = np.clip((layer - low) / (high - low), 0, 1)
        scaled[np.isnan(layer)] = 0.5

        out[:, :, b] = scaled

    return out, missing


n_sheets = int(np.ceil(len(sample) / PER_SHEET))

for sheet in range(n_sheets):

    block = sample.iloc[sheet * PER_SHEET:(sheet + 1) * PER_SHEET]

    fig, axes = plt.subplots(4, 5, figsize=(15, 13), layout="constrained")
    axes = axes.flatten()

    for i in range(len(axes)):

        if i >= len(block):
            axes[i].axis("off")
            continue

        r = int(block["row"].iloc[i])
        c = int(block["col"].iloc[i])

        r0, r1 = r - CHIP_HALF, r + CHIP_HALF + 1
        c0, c1 = c - CHIP_HALF, c + CHIP_HALF + 1

        false_colour = np.dstack([
            bands[NIR - 1, r0:r1, c0:c1],
            bands[RED - 1, r0:r1, c0:c1],
            bands[GREEN - 1, r0:r1, c0:c1]
        ])

        shown, missing = stretch(false_colour)

        axes[i].imshow(shown, interpolation="nearest")

        # Box on the 3 x 3 sample unit at the centre.
        offset = CHIP_HALF - SAMPLE_UNIT / 2

        axes[i].add_patch(
            plt.Rectangle(
                (offset, offset), SAMPLE_UNIT, SAMPLE_UNIT,
                edgecolor="white", facecolor="none", linewidth=1.5
            )
        )

        # Flag part-masked thumbnails so grey areas are not read as real
        # surfaces.
        title = block["point_id"].iloc[i]

        if 100 * missing.mean() > 1:
            title += "  (%d%% masked)" % round(100 * missing.mean())

        axes[i].set_title(title, fontsize=11)
        axes[i].axis("off")

    fig.suptitle(
        f"Sheet {sheet + 1} of {n_sheets}   false colour, vegetation is red"
        "   each thumbnail is 1.2 km across",
        fontsize=12
    )

    output = SHEETS_DIR / ("sheet_%02d.png" % (sheet + 1))

    plt.savefig(output, dpi=120, bbox_inches="tight")
    plt.close()

print(n_sheets, "contact sheets written to", SHEETS_DIR)


#---5. Blank Table---

# Only written if it does not already exist, so rerunning never overwrites
# labels that have been typed in. Delete it by hand to start over.

csv_path = ACCURACY_DIR / f"accuracy_points_{YEAR}.csv"

if csv_path.exists():
    print("\n", csv_path.name, "already exists, leaving it alone")
else:
    sample.to_csv(csv_path, index=False)
    print("\nBlank table written to", csv_path.name)

print("\nNEXT STEP")
print("  1. open the contact sheets in", SHEETS_DIR)
print("  2. open", csv_path.name)
print("  3. type one word per point into the reference_class column")
print("  4. save the file and run this script again")
print("\nThe five words, and what they look like in false colour:")
print("  water         black or dark blue")
print("  low_veg       grey or cyan, little or no red")
print("  moderate_veg  grey with red mixed through it, buildings visible")
print("  mixed_veg     mostly red, field shapes, no buildings")
print("  dense_veg     solid deep red, continuous canopy")
print("\nJudge the white box, not the whole thumbnail, and label what is")
print("visible rather than what the map_class column says.")


#---6. Read the Labelled Table---

# Exporting from Numbers puts the table name on the first line, above the
# real header, and can leave a tab inside a cell. Both are handled here.

first_line = open(csv_path).readline()

if first_line.startswith("point_id"):
    labelled = pd.read_csv(csv_path)
else:
    print("\nFirst line is not the header, skipping it:",
          repr(first_line.strip()))
    labelled = pd.read_csv(csv_path, skiprows=1)

labelled = labelled[labelled["reference_class"].notna()]

# Take anything before a stray tab, then strip spaces and lower case, so
# "Water ", "water" and "Water" all count as the same class.
labelled["reference_class"] = (
    labelled["reference_class"]
    .astype(str)
    .str.split("\t").str[0]
    .str.strip()
    .str.lower()
)

labelled = labelled[labelled["reference_class"] != ""]

# Report anything that is not one of the five class names instead of
# silently counting it as an error.
unknown = labelled[~labelled["reference_class"].isin(NAMES)]

if len(unknown) > 0:
    print("\nThese labels are not one of the five class names:")
    print(unknown[["point_id", "reference_class"]].to_string(index=False))
    print("Fix the spelling in the table, then run again.")
    labelled = labelled[labelled["reference_class"].isin(NAMES)]

print("\nLabelled points:", len(labelled), "of", len(sample))

ready = len(labelled) > 0

if not ready:
    print("Nothing labelled yet, so there is no accuracy to calculate.")
    print("That is expected on the first run. See NEXT STEP above.")


#---7. Error Matrix---

# Columns are the reference data and rows are the classification, as set out
# in Lecture 6. The diagonal holds the samples both agree on.

if ready:

    n = len(NAMES)

    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.sum(
                (labelled["map_class"] == NAMES[i])
                & (labelled["reference_class"] == NAMES[j])
            )

    table = pd.DataFrame(matrix, index=NAMES, columns=NAMES)
    table.index.name = "classified"
    table.columns.name = "reference"

    table["row_total"] = table.sum(axis=1)
    table.loc["col_total"] = table.sum(axis=0)

    table.to_csv(ACCURACY_DIR / f"confusion_matrix_{YEAR}.csv")

    print("\nError matrix, rows = classification, columns = reference")
    print(table.to_string())


#---8. Accuracy and Kappa---

# Formulas from Lecture 6:
#   producer's accuracy = correct in a class / column total, omission error
#   user's accuracy     = correct in a class / row total, commission error
#   overall accuracy    = sum of the diagonal / total samples
#   Kappa = (N * sum(xii) - sum(xi+ * x+i)) / (N^2 - sum(xi+ * x+i))

if ready:

    total = matrix.sum()
    correct = np.trace(matrix)

    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)

    overall = 100 * correct / total

    print(f"\nOverall accuracy: {correct} / {total} = {overall:.2f} percent")

    print("\nPer class accuracy")
    print("class           producer's   user's")

    summary_rows = []

    for i in range(len(NAMES)):

        producer = (
            100 * matrix[i, i] / col_totals[i] if col_totals[i] > 0 else None
        )
        user = (
            100 * matrix[i, i] / row_totals[i] if row_totals[i] > 0 else None
        )

        print("%-15s %9.1f %%  %6.1f %%" % (NAMES[i], producer, user))

        summary_rows.append({
            "class": NAMES[i],
            "producers_pct": round(producer, 1) if producer else None,
            "users_pct": round(user, 1) if user else None
        })

    pd.DataFrame(summary_rows).to_csv(
        ACCURACY_DIR / f"accuracy_summary_{YEAR}.csv", index=False
    )

    # Kappa compares the observed agreement against the agreement that would
    # happen by chance, given the row and column totals.
    chance = np.sum(row_totals * col_totals)

    kappa = (total * correct - chance) / (total ** 2 - chance)

    print(f"\nKappa: {kappa:.4f}")

    if kappa > 0.8:
        print("  strong agreement, Lecture 6: Khat > 0.8")
    elif kappa > 0.4:
        print("  moderate agreement, Lecture 6: 0.8 > Khat > 0.4")
    else:
        print("  poor agreement, Lecture 6: Khat < 0.4")

    print("\nThese two numbers go in the report:")
    print(f"  overall accuracy {overall:.2f} percent")
    print(f"  Kappa {kappa:.4f}")