# Downloads the three Landsat scenes from Microsoft Planetary Computer.
# Only the study-area window of each band is saved, not the whole scene.
#
# ENVI equivalent: File > Open As > Optical Sensors > Landsat, after
# downloading the scenes by hand from USGS EarthExplorer.


#---1. Libraries---

from pathlib import Path
import runpy

import geopandas as gpd
import planetary_computer
import rasterio

from pystac_client import Client
from rasterio.windows import from_bounds, transform as window_transform


#---2. Configuration---

# A module name cannot start with a digit, so "00_config.py" is loaded with
# runpy instead of a normal import.
# https://docs.python.org/3/library/runpy.html

config = runpy.run_path(
    str(Path(__file__).with_name("00_config.py"))
)

LANDSAT_DIR = config["LANDSAT_DIR"]

GHSL_FILE = config["GHSL_FILE"]
GHSL_LAYER = config["GHSL_LAYER"]
GHSL_CITY_ID = config["GHSL_CITY_ID"]
BUFFER_METERS = config["BUFFER_METERS"]
TARGET_CRS = config["TARGET_CRS"]

SCENES = config["SCENES"]
STAC_URL = config["STAC_URL"]
COLLECTION = config["COLLECTION"]

if not GHSL_FILE.exists():
    raise FileNotFoundError(
        f"Study-area file not found: {GHSL_FILE}"
    )


#---3. Study Area---

# Hangzhou urban centre from the GHSL database, buffered by 5 km.

urban = gpd.read_file(
    GHSL_FILE,
    layer=GHSL_LAYER,
    where=f"ID_UC_G0 = {GHSL_CITY_ID}"
).to_crs(TARGET_CRS)

study_area = urban.geometry.iloc[0].buffer(BUFFER_METERS)

print("Study area:", round(study_area.area / 1e6, 2), "km²")


#---4. Find the Scenes---

# The three scenes are requested by their exact IDs, so the search returns
# those scenes and nothing else.

catalog = Client.open(STAC_URL)

scene_ids = [scene["id"] for scene in SCENES.values()]

search = catalog.search(
    collections=[COLLECTION],
    ids=scene_ids
)

items = {item.id: item for item in search.items()}

print("Scenes found:", len(items), "of", len(scene_ids))


#---5. Bands to Download---

# Six optical bands plus the two quality bands used for masking.

bands = [
    "blue",
    "green",
    "red",
    "nir08",
    "swir16",
    "swir22",
    "qa_pixel",
    "qa_radsat",
]


#---6. Download Each Band---

for year, scene in SCENES.items():

    print(f"\nProcessing {year}")

    item = items[scene["id"]]

    year_dir = LANDSAT_DIR / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)

    for band in bands:

        output = year_dir / f"{band}.tif"

        # Skip bands already on disk so the script can be rerun safely.
        if output.exists():
            print(f"  {band}: already exists")
            continue

        # Planetary Computer links need a short-lived access token.
        url = planetary_computer.sign(item.assets[band].href)

        with rasterio.open(url) as src:

            # Study-area bounding box, converted to the scene's own CRS.
            bounds = gpd.GeoSeries(
                [study_area],
                crs=TARGET_CRS
            ).to_crs(src.crs).total_bounds

            # Read only the pixels inside that box. Rounding to whole pixels
            # keeps every band on the same grid.
            window = from_bounds(
                *bounds,
                transform=src.transform
            ).round_offsets().round_lengths()

            data = src.read(1, window=window)

            # Copy the source settings, then correct the size and position
            # to match the window that was read.
            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=data.shape[0],
                width=data.shape[1],
                transform=window_transform(window, src.transform),
                compress="deflate"
            )

        with rasterio.open(output, "w", **profile) as dst:
            dst.write(data, 1)

        print(f"  {band}: saved")

    # Record which scene the folder came from.
    (year_dir / "scene_id.txt").write_text(scene["id"] + "\n")


#---7. Finish---

print("\nLandsat data retrieval complete.")
print("Files saved in:", LANDSAT_DIR)