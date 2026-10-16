# Unsupervised classification of the three August scenes, into four
# vegetation density classes plus water.
#
# Method, from Lecture 5: K-Means groups pixels into spectral clusters, then
# the analyst decides what each cluster represents. The analyst sets the
# number of clusters, arbitrary centres are placed, each pixel is assigned to
# the nearest mean, the means are recomputed, and this repeats until the
# change falls below a threshold. ISODATA also splits and merges clusters;
# the Python version does not, so the cluster count is fixed here.
#
# Two choices that were made after testing, explained in the report:
#   - The clustering is fitted once on all three years pooled, not once per
#     year, so a class means the same thing in every year.
#   - It uses NDVI only. Clustering on the six reflectance bands followed
#     overall scene brightness instead of land cover, which matches the Lab 4
#     result that the first principal component held 85.16% of the variance
#     and loaded positively on every band.
#
# Because the classes come from NDVI alone they describe vegetation density,
# not land cover, and are named accordingly.
#
# ENVI equivalent: ENVI Classic > Classification > Unsupervised > K-Means,
# then labelling each cluster by hand.
#
# Clustering reference: sklearn.cluster.KMeans
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


#---1. Libraries---

from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.cluster import KMeans


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

YEARS = config["YEARS"]
CLASS_NAMES = config["CLASS_NAMES"]
N_CLUSTERS = config["N_CLUSTERS"]
SEED = config["SEED"]
PIXEL_AREA_KM2 = config["PIXEL_AREA_KM2"]

SAMPLE_PER_YEAR = 70000   # pixels taken from each year to fit the clusters

CLASSIFICATION_DIR.mkdir(parents=True, exist_ok=True)


#---3. Load NDVI and Masks---

ndvi_land = {}     # NDVI values of land pixels only
land_masks = {}    # where those pixels are
water_masks = {}
profiles = {}

for year in YEARS:

    with rasterio.open(INDICES_DIR / f"ndvi_{year}.tif") as src:
        ndvi = src.read(1).astype("float32")
        profiles[year] = src.profile

    with rasterio.open(CLEAN_DIR / f"valid_{year}.tif") as src:
        valid = src.read(1) == 1

    with rasterio.open(CLEAN_DIR / f"water_{year}.tif") as src:
        water = src.read(1) == 1

    # Cluster land only. Water already has its own class, and including it
    # would pull a cluster away from the land classes.
    land = valid & ~water

    # Drop the few NDVI values outside the theoretical range, reported in
    # 04_indices.py.
    land = land & (ndvi >= -1) & (ndvi <= 1)

    ndvi_land[year] = ndvi[land]
    land_masks[year] = land
    water_masks[year] = water

    print(year, "land pixels:", land.sum(), " water pixels:", water.sum())


#---4. Fit the Clusters---

# An equal sample is taken from each year so no single date dominates.

np.random.seed(SEED)

pool = []

for year in YEARS:

    pick = np.random.choice(
        len(ndvi_land[year]), SAMPLE_PER_YEAR, replace=False
    )

    pool.append(ndvi_land[year][pick])

pool = np.concatenate(pool).reshape(-1, 1)

print("\nPooled training sample:", pool.shape)

kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
kmeans.fit(pool)

# K-Means numbers clusters arbitrarily, so they are renumbered from lowest
# to highest NDVI. Class codes start at 2 because code 1 is water.
centres = kmeans.cluster_centers_.flatten()

order = np.argsort(centres)

class_of_cluster = np.zeros(N_CLUSTERS, dtype=int)

for position in range(N_CLUSTERS):
    class_of_cluster[order[position]] = position + 2

print("\nCluster mean NDVI, lowest to highest:")

for position in range(N_CLUSTERS):

    code = position + 2

    print(
        f"  class {code} {CLASS_NAMES[code]}: "
        f"NDVI {centres[order[position]]:.3f}"
    )


#---5. Class Boundaries---

# With one input variable the boundaries are just the midpoints between
# neighbouring cluster means, so the classification can be quoted in the
# report as plain NDVI thresholds.

sorted_centres = np.sort(centres)

boundaries = (sorted_centres[:-1] + sorted_centres[1:]) / 2

print("\nNDVI boundaries between classes:")

for i in range(len(boundaries)):

    print(
        f"  {CLASS_NAMES[i + 2]} | {CLASS_NAMES[i + 3]} "
        f"at NDVI {boundaries[i]:.3f}"
    )


#---6. Write a Class Raster for Each Year---

area_rows = []

for year in YEARS:

    land = land_masks[year]
    water = water_masks[year]

    predicted = kmeans.predict(ndvi_land[year].reshape(-1, 1))
    predicted = class_of_cluster[predicted]

    classed = np.zeros(land.shape, dtype="uint8")
    classed[land] = predicted
    classed[water] = 1

    profile = profiles[year].copy()
    profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")

    output = CLASSIFICATION_DIR / f"class_{year}.tif"

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(classed, 1)

    codes, counts = np.unique(classed[classed > 0], return_counts=True)

    for i in range(len(codes)):

        area_rows.append({
            "year": year,
            "class_code": int(codes[i]),
            "class_name": CLASS_NAMES[int(codes[i])],
            "n_pixels": int(counts[i]),
            "area_km2": round(counts[i] * PIXEL_AREA_KM2, 2),
            "pct": round(100 * counts[i] / counts.sum(), 2)
        })

    print("Saved:", output.name)

areas = pd.DataFrame(area_rows)
areas.to_csv(CLASSIFICATION_DIR / "class_areas.csv", index=False)


#---7. Area by Class and Year---

# These areas cover a slightly different footprint each year, because the
# cloud masking differs between dates. Use the common-mask areas from
# 08_change.py for any comparison between years.

trend = areas.pivot(index="class_name", columns="year", values="area_km2")
trend = trend.reindex([CLASS_NAMES[i] for i in range(1, 6)])

print("\nArea in km2 by class and year, per-year footprint:")
print(trend.to_string())


#---8. Classification Maps---

colours = [
    "black",      # 0 masked
    "#2c7fb8",    # 1 water
    "#d95f02",    # 2 low_veg
    "#fdae61",    # 3 moderate_veg
    "#a6d96a",    # 4 mixed_veg
    "#1b7837"     # 5 dense_veg
]

cmap = ListedColormap(colours)

fig, axes = plt.subplots(1, 3, figsize=(16, 7), layout="constrained")

for ax, year in zip(axes, YEARS):

    with rasterio.open(
        CLASSIFICATION_DIR / f"class_{year}.tif"
    ) as src:

        ax.imshow(
            src.read(1),
            cmap=cmap,
            vmin=0, vmax=5,
            interpolation="nearest"
        )

    ax.set_title(str(year))
    ax.axis("off")

handles = [
    Patch(color=colours[code], label=CLASS_NAMES[code])
    for code in range(1, 6)
]

fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)

fig.suptitle(
    "Hangzhou vegetation density classes, August scenes\n"
    "black areas: cloud, shadow or outside the study boundary",
    fontsize=13
)

output = CLASSIFICATION_DIR / "classification_comparison.png"

plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()


#---9. Finish---

print("\nClassification complete.")
print("Outputs saved in:", CLASSIFICATION_DIR)