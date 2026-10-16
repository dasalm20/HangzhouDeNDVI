# Post-classification change detection between the class maps.
#
# Each date is classified separately, then the class labels are compared
# pixel by pixel. The advantage is that the dates do not have to be
# radiometrically identical, because the comparison happens after
# classification. The cost is that errors in either classification carry into
# the result, so change accuracy is roughly the product of the two.
#
# 2010 is mapped and listed for reference but is left out of the change
# statistics. Its scene mean NDVI sits about 0.045 below the other two dates,
# which is roughly a fifth of a class width, so it shifts class areas without
# any real landscape change. 1995 and 2025 differ by only 0.013.
#
# Everything is restricted to pixels valid in all three years, so each date
# covers exactly the same ground.
#
# ENVI equivalent: ENVI Classic > Basic Tools > Change Detection > Change
# Detection Statistics, or Band Math with b2*5-b1+1 followed by Density Slice.


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

INDICES_DIR = config["INDICES_DIR"]
CLASSIFICATION_DIR = config["CLASSIFICATION_DIR"]
CHANGE_DIR = config["CHANGE_DIR"]

YEARS = config["YEARS"]
CLASS_NAMES = config["CLASS_NAMES"]
PIXEL_AREA_KM2 = config["PIXEL_AREA_KM2"]
MIN_NDVI_CHANGE = config["MIN_NDVI_CHANGE"]

START, END = config["CHANGE_PAIR"]

N_CLASSES = len(CLASS_NAMES)
NAMES = [CLASS_NAMES[code] for code in range(1, N_CLASSES + 1)]

CHANGE_DIR.mkdir(parents=True, exist_ok=True)


#---3. Load Class Maps and Build the Common Mask---

classes = {}
ndvi = {}

for year in YEARS:

    with rasterio.open(
        CLASSIFICATION_DIR / f"class_{year}.tif"
    ) as src:
        classes[year] = src.read(1)
        profile = src.profile

    with rasterio.open(INDICES_DIR / f"ndvi_{year}.tif") as src:
        ndvi[year] = src.read(1).astype("float32")

# A pixel counts only if it was classified in every year.
common = np.ones(classes[START].shape, dtype=bool)

for year in YEARS:
    common = common & (classes[year] > 0)

print("Common pixels:", common.sum())
print("Common area:", round(common.sum() * PIXEL_AREA_KM2, 2), "km²")


#---4. Class Areas on the Common Mask---

# These replace the per-year areas from 07_classify.py for anything that
# compares dates, because every year now covers the same ground.

area_rows = []

for year in YEARS:

    codes, counts = np.unique(classes[year][common], return_counts=True)

    for i in range(len(codes)):

        area_rows.append({
            "year": year,
            "class_name": CLASS_NAMES[int(codes[i])],
            "area_km2": round(counts[i] * PIXEL_AREA_KM2, 2),
            "pct": round(100 * counts[i] / counts.sum(), 2)
        })

areas = pd.DataFrame(area_rows)
areas.to_csv(CHANGE_DIR / "class_areas_common.csv", index=False)

trend = areas.pivot(index="class_name", columns="year", values="area_km2")
trend = trend.reindex(NAMES)

trend[f"change_{START}_{END}"] = (trend[END] - trend[START]).round(2)
trend["pct_change"] = (
    100 * (trend[END] - trend[START]) / trend[START]
).round(1)

print("\nClass area in km² on the common mask:")
print("(the 2010 column is for reference only)")
print(trend.to_string())


#---5. From-to Change Matrix---

# Rows are the start class, columns are the end class. The diagonal holds
# pixels that did not change; everything else is a transition.

before = classes[START]
after = classes[END]

delta_ndvi = ndvi[END] - ndvi[START]

# A class change is only counted as confident if NDVI also moved by at least
# MIN_NDVI_CHANGE. Class widths are about 0.20, so this drops pixels that sat
# on a boundary and could have fallen either way.
confident = common & (np.abs(delta_ndvi) >= MIN_NDVI_CHANGE)


def build_matrix(mask):

    matrix = np.zeros((N_CLASSES, N_CLASSES), dtype="int64")

    b = before[mask]
    a = after[mask]

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            matrix[i, j] = np.sum((b == i + 1) & (a == j + 1))

    return matrix


def save_matrix(matrix, filename):

    table = pd.DataFrame(
        (matrix * PIXEL_AREA_KM2).round(2),
        index=NAMES, columns=NAMES
    )

    table.index.name = f"from_{START}"
    table.columns.name = f"to_{END}"

    table.to_csv(CHANGE_DIR / filename)

    return table


matrix_all = build_matrix(common)

print("\nFrom-to change matrix, area in km²:")
print(save_matrix(matrix_all, f"change_matrix_{START}_{END}_km2.csv").to_string())

unchanged = np.trace(matrix_all)
total = matrix_all.sum()

print("\nAll class changes:")
print(
    "  unchanged:",
    round(unchanged * PIXEL_AREA_KM2, 2), "km² =",
    round(100 * unchanged / total, 1), "percent"
)
print(
    "  changed:  ",
    round((total - unchanged) * PIXEL_AREA_KM2, 2), "km² =",
    round(100 * (total - unchanged) / total, 1), "percent"
)


#---6. Confident Changes Only---

changed_any = common & (before != after)
changed_conf = changed_any & (np.abs(delta_ndvi) >= MIN_NDVI_CHANGE)

print(f"\nWith the NDVI magnitude test at {MIN_NDVI_CHANGE}:")
print(
    "  class changed, any magnitude:",
    round(changed_any.sum() * PIXEL_AREA_KM2, 2), "km²"
)
print(
    "  class changed and confident: ",
    round(changed_conf.sum() * PIXEL_AREA_KM2, 2), "km²"
)
print(
    "  dropped as near-boundary:    ",
    round((changed_any.sum() - changed_conf.sum()) * PIXEL_AREA_KM2, 2), "km²"
)

matrix_conf = build_matrix(confident)

save_matrix(matrix_conf, f"change_matrix_{START}_{END}_confident_km2.csv")

# Every off-diagonal cell, sorted, so the main transitions are easy to read.
transitions = []

for i in range(N_CLASSES):
    for j in range(N_CLASSES):

        if i != j:

            transitions.append({
                f"from_{START}": NAMES[i],
                f"to_{END}": NAMES[j],
                "area_km2": round(matrix_conf[i, j] * PIXEL_AREA_KM2, 2)
            })

transitions = pd.DataFrame(transitions).sort_values(
    "area_km2", ascending=False
)

transitions.to_csv(
    CHANGE_DIR / f"transitions_{START}_{END}_confident.csv", index=False
)

print("\nTen largest confident transitions, km²:")
print(transitions.head(10).to_string(index=False))


#---7. Direction of Change---

# Classes 2 to 5 run from least to most vegetation, so subtracting the class
# numbers gives the direction and size of the change directly. Water is left
# out because it is not part of that order.

b = before.astype("int16")
a = after.astype("int16")

veg_mask = common & (b >= 2) & (a >= 2)

step = np.zeros(before.shape, dtype="int16")
step[veg_mask] = a[veg_mask] - b[veg_mask]

# Reset the class changes that failed the magnitude test.
near_boundary = (
    veg_mask
    & (step != 0)
    & (np.abs(delta_ndvi) < MIN_NDVI_CHANGE)
)

step[near_boundary] = 0

out_step = step.copy()
out_step[~veg_mask] = -99

profile.update(count=1, dtype="int16", nodata=-99, compress="lzw")

with rasterio.open(
    CHANGE_DIR / f"direction_{START}_{END}.tif", "w", **profile
) as dst:
    dst.write(out_step, 1)

steps, counts = np.unique(step[veg_mask], return_counts=True)

print(f"\nChange in vegetation class, {START} to {END}, confident only:")

for i in range(len(steps)):

    print(
        f"  {int(steps[i]):+d} steps:",
        round(counts[i] * PIXEL_AREA_KM2, 2), "km² =",
        round(100 * counts[i] / counts.sum(), 1), "percent"
    )

loss = counts[steps < 0].sum()
gain = counts[steps > 0].sum()
same = counts[steps == 0].sum()

print("\n  lost vegetation density:  ", round(loss * PIXEL_AREA_KM2, 2), "km²")
print("  gained vegetation density:", round(gain * PIXEL_AREA_KM2, 2), "km²")
print("  no confident change:      ", round(same * PIXEL_AREA_KM2, 2), "km²")
print("  net:", round((gain - loss) * PIXEL_AREA_KM2, 2), "km²")


#---8. Change Map---

fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")

# Left: how many classes each pixel gained or lost.
plot_step = np.where(veg_mask, step, np.nan)

im = axes[0].imshow(plot_step, cmap="RdYlGn", vmin=-3, vmax=3,
                    interpolation="nearest")

axes[0].set_title(f"Change in vegetation class, {START} to {END}")
axes[0].axis("off")

bar = fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)
bar.set_label("classes gained (+) or lost (-)")

# Right: pixels that ended in the lowest vegetation class.
to_low = confident & (before > 2) & (after == 2)

axes[1].imshow(to_low, cmap="gray", interpolation="nearest")

axes[1].set_title(
    f"Moved into the lowest vegetation class by {END}\n"
    f"{round(to_low.sum() * PIXEL_AREA_KM2, 1)} km²"
)
axes[1].axis("off")

fig.suptitle(
    f"Hangzhou vegetation change, August {START} to August {END}\n"
    "white areas: cloud, shadow or outside the study boundary",
    fontsize=12
)

output = CHANGE_DIR / f"change_{START}_{END}.png"

plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()


#---9. Summary for the Report---

print(
    f"\nSummary, {START} to {END}, common mask of",
    round(common.sum() * PIXEL_AREA_KM2, 0), "km²"
)

print(
    trend[[START, END, f"change_{START}_{END}", "pct_change"]].to_string()
)

print("\nChange detection complete.")
print("Outputs saved in:", CHANGE_DIR)