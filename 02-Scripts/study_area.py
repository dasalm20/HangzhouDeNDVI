# Builds the study area map for the proposal and the report:
#   study_area_map.png   true colour 2025, study boundary, place labels,
#                        UTM grid, north arrow, scale bar and a locator inset
#
# The map answers three questions a reader asks first: where the study area
# is, how big it is, and what the ground looks like.
#
# Two things come from outside the processing chain and are cited on the map
# itself: the country outlines used by the locator inset, and the place
# labels listed below. The label coordinates are approximate, so check them
# against the imagery before the map goes into the report.
#
# ENVI equivalent: ENVI Classic > File > Output Map to Postscript, with the
# map elements added through the Annotation menu (text, scale bar, north
# arrow, grid lines).


#---1. Libraries---

from pathlib import Path
import runpy
import urllib.request

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from matplotlib.patches import Patch, Rectangle
from matplotlib.patheffects import withStroke
from rasterio.warp import transform as transform_coords


#---2. Configuration---

# A module name cannot start with a digit, so "00_config.py" is loaded with
# runpy instead of a normal import.
# https://docs.python.org/3/library/runpy.html

config = runpy.run_path(
    str(Path(__file__).with_name("00_config.py"))
)

CLEAN_DIR = config["CLEAN_DIR"]
FIGURES_DIR = config["FIGURES_DIR"]
BOUNDARY_DIR = config["BOUNDARY_DIR"]

GHSL_FILE = config["GHSL_FILE"]
GHSL_LAYER = config["GHSL_LAYER"]
GHSL_CITY_ID = config["GHSL_CITY_ID"]
BUFFER_METERS = config["BUFFER_METERS"]

TARGET_CRS = config["TARGET_CRS"]
BAND_ORDER = config["BAND_ORDER"]
STUDY_AREA_KM2 = config["STUDY_AREA_KM2"]
STUDY_NAME = config["STUDY_NAME"]
SCENES = config["SCENES"]

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Year used for the background image.
MAP_YEAR = 2025

# Band numbers inside reflectance_YYYY.tif, looked up by name.
RED = BAND_ORDER.index("red") + 1
GREEN = BAND_ORDER.index("green") + 1
BLUE = BAND_ORDER.index("blue") + 1

# Times New Roman to match the written report. Liberation Serif has the same
# metrics and is the usual substitute when Times is not installed.
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = [
    "Times New Roman", "Liberation Serif", "DejaVu Serif"
]

# Blank space left on each side of the imagery, as a fraction of its width
# and height. More room on the left and at the bottom pushes the study area
# up and to the right, clear of the locator inset and the scale bar.
MARGIN_LEFT = 0.22
MARGIN_RIGHT = 0.04
MARGIN_BOTTOM = 0.20
MARGIN_TOP = 0.04

# Places worth labelling, as longitude, latitude and how the text sits
# relative to the marker. Approximate positions: verify before publishing.
PLACES = {
    "West Lake": (120.149, 30.247, "right", "center"),
    "City centre": (120.168, 30.274, "left", "bottom"),
    "Qiantang River": (120.330, 30.155, "left", "center"),
    "Xiaoshan": (120.265, 30.183, "left", "top"),
    "Yuhang": (120.000, 30.420, "right", "bottom"),
}

SCALE_BAR_M = 10000

# Country outlines for the locator inset. Natural Earth is public domain and
# is downloaded once, then kept beside the other boundary data.
# https://www.naturalearthdata.com/downloads/50m-cultural-vectors/
NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_0_countries.geojson"
)
NE_FILE = BOUNDARY_DIR / "ne_50m_admin_0_countries.geojson"

# Albers equal area is the usual projection for maps of China, and it gives
# the country its familiar shape.
CHINA_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lon_0=105 +datum=WGS84 +units=m +no_defs"
)


#---3. Study Area Boundary---

# Same polygon the analysis grid was built from in 03_clean.py: the GHSL 2015
# urban centre for Hangzhou, buffered by 5 km.

urban = gpd.read_file(
    GHSL_FILE,
    layer=GHSL_LAYER,
    where=f"ID_UC_G0 = {GHSL_CITY_ID}"
).to_crs(TARGET_CRS)

city = urban.geometry.iloc[0]
study_area = city.buffer(BUFFER_METERS)

print("Study area:", round(study_area.area / 1e6, 2), "km²")


#---4. Background Image---

# Read the three visible bands and stretch them together, so the colour
# balance of the scene is kept. Same approach as 05_figures.py.

with rasterio.open(CLEAN_DIR / f"reflectance_{MAP_YEAR}.tif") as src:

    rgb = np.stack(
        [src.read(RED), src.read(GREEN), src.read(BLUE)],
        axis=-1
    )

    bounds = src.bounds

usable = np.isfinite(rgb).all(axis=2)

low, high = np.nanpercentile(rgb[usable], [2, 98])

rgb = np.clip((rgb - low) / (high - low), 0, 1)
rgb = np.nan_to_num(rgb)

# imshow needs the image corners in map units, so the axes carry metres and
# the scale bar and the grid are true to the projection.
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# The drawn area is wider than the image, which shrinks the study area a
# little and leaves room for the map elements.
image_width = bounds.right - bounds.left
image_height = bounds.top - bounds.bottom

view_left = bounds.left - MARGIN_LEFT * image_width
view_right = bounds.right + MARGIN_RIGHT * image_width
view_bottom = bounds.bottom - MARGIN_BOTTOM * image_height
view_top = bounds.top + MARGIN_TOP * image_height


#---5. Map Element Functions---

# A white outline behind dark text keeps labels readable over the imagery.
# https://matplotlib.org/stable/users/explain/artists/patheffects_guide.html

HALO = [withStroke(linewidth=2.5, foreground="white")]

# The urban centre outline crosses bright rooftops and dark vegetation
# alike, so it is drawn in cyan with a dark edge behind it.
GHSL_COLOUR = "#00e5ff"
GHSL_DASH = (0, (7, 3))
GHSL_EDGE = [withStroke(linewidth=3.4, foreground="#1a1a1a")]


def add_panel(ax, x, y, width, height):

    # Masked ground is drawn black, so the arrow and the scale bar sit on a
    # translucent white panel rather than straight on the background.
    ax.add_patch(Rectangle(
        (x, y), width, height,
        transform=ax.transAxes,
        facecolor="white", edgecolor="black",
        alpha=0.85, linewidth=0.6, zorder=4
    ))


def add_north_arrow(ax):

    # Placed in axes fractions, so it stays put whatever the map extent is.
    add_panel(ax, 0.025, 0.865, 0.075, 0.12)

    ax.annotate(
        "",
        xy=(0.0625, 0.940), xytext=(0.0625, 0.882),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.4),
        zorder=6
    )

    ax.text(
        0.0625, 0.948, "N",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", zorder=6
    )


def add_scale_bar(ax, length_m, left_fraction=0.38):

    add_panel(ax, left_fraction - 0.02, 0.025, 0.26, 0.055)

    # The bar itself is drawn in map coordinates, so its length is true to
    # the projection rather than to the figure size.
    x0 = view_left + left_fraction * (view_right - view_left)
    y0 = view_bottom + 0.032 * (view_top - view_bottom)

    height = 0.007 * (view_top - view_bottom)
    half = length_m / 2

    # Two blocks, the usual alternating pattern.
    ax.add_patch(Rectangle(
        (x0, y0), half, height,
        facecolor="black", edgecolor="black", linewidth=0.8, zorder=5
    ))
    ax.add_patch(Rectangle(
        (x0 + half, y0), half, height,
        facecolor="white", edgecolor="black", linewidth=0.8, zorder=5
    ))

    # The unit rides on the last number, so nothing crowds the bar.
    for distance in [0, half, length_m]:

        if distance == length_m:
            label = f"{distance / 1000:.0f} km"
        else:
            label = f"{distance / 1000:.0f}"

        ax.text(
            x0 + distance, y0 + 1.5 * height, label,
            ha="center", va="bottom", fontsize=9, zorder=6
        )


def add_grid(ax, step_m=10000):

    # Ticks on round 10 km eastings and northings, labelled in kilometres so
    # the numbers stay short.
    xticks = np.arange(
        np.ceil(view_left / step_m) * step_m, view_right, step_m
    )
    yticks = np.arange(
        np.ceil(view_bottom / step_m) * step_m, view_top, step_m
    )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels([f"{x / 1000:.0f}" for x in xticks], fontsize=9)
    ax.set_yticklabels([f"{y / 1000:.0f}" for y in yticks], fontsize=9)

    ax.grid(color="white", linewidth=0.4, alpha=0.4)

    ax.set_xlabel("Easting (km), UTM zone 51N", fontsize=10)
    ax.set_ylabel("Northing (km), UTM zone 51N", fontsize=10)


#---6. Main Map---

fig, ax = plt.subplots(figsize=(9, 10.5))

# The blank margin matches the masked ground, so the frame reads as one
# surface rather than as a border around the image.
ax.set_facecolor("black")

ax.imshow(rgb, extent=extent, interpolation="nearest")

# The buffered study area, then the urban centre it was buffered from.
gpd.GeoSeries([study_area], crs=TARGET_CRS).boundary.plot(
    ax=ax, color="#d73027", linewidth=1.8, zorder=3
)

gpd.GeoSeries([city], crs=TARGET_CRS).boundary.plot(
    ax=ax, color=GHSL_COLOUR, linewidth=2.0, linestyle=GHSL_DASH,
    path_effects=GHSL_EDGE, zorder=3
)

# Place labels, converted from longitude and latitude to the map projection.
for name, (lon, lat, ha, va) in PLACES.items():

    x, y = transform_coords("EPSG:4326", TARGET_CRS, [lon], [lat])

    offset = 900 if ha == "left" else -900

    ax.plot(
        x[0], y[0], "o",
        color="white", markeredgecolor="black",
        markersize=5, markeredgewidth=0.9, zorder=6
    )

    ax.text(
        x[0] + offset, y[0], name,
        ha=ha, va=va, fontsize=10,
        path_effects=HALO, zorder=6
    )

add_grid(ax)
add_north_arrow(ax)
add_scale_bar(ax, SCALE_BAR_M)

ax.set_xlim(view_left, view_right)
ax.set_ylim(view_bottom, view_top)

ax.set_title(
    f"Study area: {STUDY_NAME}\n"
    f"Landsat true colour, {SCENES[MAP_YEAR]['date']}",
    fontsize=13
)

# Legend entries are drawn as empty lines so they match the map symbols.
ax.plot([], [], color="#d73027", linewidth=1.8,
        label=f"Study area, {STUDY_AREA_KM2:,.0f} km²")
ax.plot([], [], color=GHSL_COLOUR, linewidth=2.0, linestyle=GHSL_DASH,
        path_effects=GHSL_EDGE, label="GHSL urban centre, 2015")
ax.plot([], [], "o", color="white", markeredgecolor="black",
        markersize=5, markeredgewidth=0.9, label="Place")

# The masked ground is explained in the legend as well as in the note, so a
# reader does not have to hunt for what the black stands for.
masked_patch = Patch(
    facecolor="black", edgecolor="#808080", linewidth=0.6,
    label="Masked: cloud, shadow,\nor outside the study area"
)

handles, labels = ax.get_legend_handles_labels()

ax.legend(
    handles=handles + [masked_patch],
    labels=labels + [masked_patch.get_label()],
    loc="lower right", fontsize=9, framealpha=0.9
)


#---7. Locator Inset---

# China with the study area marked, so a reader who does not know the
# country can still place Hangzhou.

if not NE_FILE.exists():

    print("Downloading country outlines from Natural Earth ...")
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(NE_URL, NE_FILE)

countries = gpd.read_file(NE_FILE).to_crs(CHINA_CRS)

china = countries[countries["ADMIN"] == "China"]

inset = ax.inset_axes([0.02, 0.025, 0.30, 0.30])

# Neighbouring countries in light grey give the outline something to sit
# against; China itself is filled so its shape reads at a glance.
countries.plot(ax=inset, color="#e8e8e8", edgecolor="#b0b0b0", linewidth=0.3)
china.plot(ax=inset, color="#bdbdbd", edgecolor="#404040", linewidth=0.6)

# The study area is a few pixels wide at this scale, so it is marked rather
# than drawn to shape.
marker = (
    gpd.GeoSeries([study_area], crs=TARGET_CRS)
    .to_crs(CHINA_CRS)
    .geometry.iloc[0]
    .centroid
)

inset.plot(
    marker.x, marker.y, "o",
    color="#d73027", markeredgecolor="black",
    markersize=6, markeredgewidth=0.7, zorder=5
)

# The label sits west of the marker, because Hangzhou is close to the coast
# and a label to its east would run off the frame.
inset.text(
    marker.x - 120000, marker.y, "Hangzhou",
    ha="right", va="center", fontsize=8,
    path_effects=HALO, zorder=5
)

# Framed on China with a small margin, which keeps the country centred.
xmin, ymin, xmax, ymax = china.total_bounds

margin = 0.06 * (xmax - xmin)

inset.set_xlim(xmin - margin, xmax + margin)
inset.set_ylim(ymin - margin, ymax + margin)

inset.set_xticks([])
inset.set_yticks([])

inset.set_xlabel("")
inset.set_ylabel("")

inset.set_facecolor("white")

for side in inset.spines.values():
    side.set_linewidth(0.8)

inset.text(
    0.5, 0.025, "Study area within China",
    transform=inset.transAxes,
    ha="center", va="bottom", fontsize=8
)


#---8. Notes and Save---

# Projection, sources and the caveat about the boundary, so the map can be
# read on its own. Anchored to the axes rather than the figure, so it stays
# under the map whatever the image proportions are.
ax.annotate(
    "Projection: UTM zone 51N (EPSG:32651), WGS 84. Grid interval 10 km. "
    "Inset: Albers equal area, Natural Earth 1:50m.\n"
    "Imagery: Landsat Collection 2 Level-2 surface reflectance, USGS, "
    "path 119 row 039.\n"
    "Boundary: GHSL Global Urban Centre Database R2024A, satellite-detected "
    "built-up extent in 2015, buffered by 5 km.\n"
    "Black areas: cloud, shadow or ground outside the study area.",
    xy=(0.5, -0.075), xycoords="axes fraction",
    ha="center", va="top", fontsize=8
)

output = FIGURES_DIR / "study_area_map.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", output.name)


#---9. Finish---

print("\nStudy area map complete.")
print("Output saved in:", FIGURES_DIR)
print("Check the place labels against the imagery before using the map.")