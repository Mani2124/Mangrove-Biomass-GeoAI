
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from uuid import uuid4
import tempfile
import zipfile

import geopandas as gpd


# =========================================================
# RUN DIRECTORY
# =========================================================

def create_run_directory(runs_dir: Path) -> Path:
    """
    Create a new unique run folder and the standard platform subfolders.
    """
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid4().hex[:6]}"

    run_dir = runs_dir / run_id

    subfolders = [
        "input",
        "raw",
        "processed",
        "features",
        "patches",
        "dataset",
        "reports",
        "logs",
        "cache",
        "biomass",
        "carbon",
    ]

    for subfolder in subfolders:
        (run_dir / subfolder).mkdir(
            parents=True,
            exist_ok=True,
        )

    return run_dir


# =========================================================
# AOI LOADING
# =========================================================

def _load_zip_shapefile(zip_file: Path) -> gpd.GeoDataFrame:
    """
    Load a zipped shapefile safely.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with zipfile.ZipFile(zip_file, "r") as archive:
            archive.extractall(temp_dir)

        shapefiles = list(temp_dir.rglob("*.shp"))

        if not shapefiles:
            raise ValueError(
                "ZIP file does not contain a .shp file."
            )

        # Use the first shapefile found.
        gdf = gpd.read_file(shapefiles[0])

    return gdf


def load_aoi(aoi_file: Path) -> gpd.GeoDataFrame:
    """
    Load GeoJSON, JSON, GeoPackage, Shapefile, or zipped Shapefile.
    """
    aoi_file = Path(aoi_file)

    if not aoi_file.exists():
        raise FileNotFoundError(
            f"AOI file not found:\n{aoi_file}"
        )

    suffix = aoi_file.suffix.lower()

    if suffix == ".zip":
        gdf = _load_zip_shapefile(aoi_file)
    else:
        gdf = gpd.read_file(aoi_file)

    if gdf.empty:
        raise ValueError(
            "AOI file contains no features."
        )

    if gdf.crs is None:
        raise ValueError(
            "AOI does not contain CRS information."
        )

    gdf = gdf[
        gdf.geometry.notna()
        &
        ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError(
            "AOI contains no valid geometries."
        )

    # Repair invalid geometry where possible.
    try:
        invalid = ~gdf.geometry.is_valid

        if invalid.any():
            gdf.loc[invalid, "geometry"] = (
                gdf.loc[invalid, "geometry"]
                .buffer(0)
            )
    except Exception:
        pass

    gdf = gdf[
        gdf.geometry.notna()
        &
        ~gdf.geometry.is_empty
    ].copy()

    if gdf.empty:
        raise ValueError(
            "AOI contains no usable geometry after validation."
        )

    return gdf


# =========================================================
# AOI SUMMARY
# =========================================================

def summarize_aoi(
    gdf: gpd.GeoDataFrame,
    area_crs,
) -> dict:
    """
    Return the exact dictionary expected by v1_prepare_aoi.py.
    """

    source_crs = str(gdf.crs)

    # Bounds must be reported in WGS84 because later platform stages
    # use geographic coordinates for catalogue searches.
    wgs84 = gdf.to_crs("EPSG:4326")

    west, south, east, north = map(
        float,
        wgs84.total_bounds,
    )

    # Use configured equal-area CRS where possible.
    # Fall back to an estimated UTM CRS for robustness.
    try:
        projected = gdf.to_crs(area_crs)
        area_km2 = float(
            projected.geometry.area.sum()
            /
            1_000_000.0
        )
    except Exception:
        estimated_crs = gdf.estimate_utm_crs()

        if estimated_crs is None:
            raise RuntimeError(
                "Unable to determine a projected CRS "
                "for AOI area calculation."
            )

        projected = gdf.to_crs(
            estimated_crs
        )

        area_km2 = float(
            projected.geometry.area.sum()
            /
            1_000_000.0
        )

    return {
        "feature_count": int(len(gdf)),
        "source_crs": source_crs,
        "area_km2": area_km2,
        "west": west,
        "south": south,
        "east": east,
        "north": north,
    }


# =========================================================
# STANDARDIZED AOI
# =========================================================

def save_standard_aoi(
    gdf: gpd.GeoDataFrame,
    output_file: Path,
) -> Path:
    """
    Save the AOI as one standardized WGS84 GeoJSON feature.

    This is the function name expected by v1_prepare_aoi.py.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wgs84 = gdf.to_crs("EPSG:4326")

    try:
        geometry = wgs84.geometry.union_all()
    except AttributeError:
        geometry = wgs84.unary_union

    standardized = gpd.GeoDataFrame(
        {
            "aoi_id": ["AOI_001"],
        },
        geometry=[geometry],
        crs="EPSG:4326",
    )

    standardized.to_file(
        output_file,
        driver="GeoJSON",
    )

    return output_file


# Optional backwards-compatible alias.
save_standardized_aoi = save_standard_aoi
