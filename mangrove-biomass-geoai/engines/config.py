from pathlib import Path

# Keep the project portable.  The previous fixed Windows path meant the
# platform only worked when it was installed in exactly one folder.
BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_AOI_EXTENSIONS = {".zip", ".geojson", ".json", ".gpkg"}
WGS84_CRS = "EPSG:4326"
AREA_CRS = "EPSG:6933"
