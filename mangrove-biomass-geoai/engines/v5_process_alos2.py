
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import shutil
import tarfile
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import uniform_filter

from config import RUNS_DIR


# =========================================================
# VERSION 5 - OPTIONAL ALOS-2 PALSAR-2 ANNUAL MOSAIC
#
# Role in final architecture:
#
#   USER AOI
#      ↓
#   Sentinel-2 + Sentinel-1 + Terrain
#      ↓
#   OPTIONAL ALOS-2 L-band predictors
#      ↓
#   V7 predictor stacks
#
# GEDI is NOT part of this user-AOI workflow.
#
# This implementation is windowed so memory usage depends on
# --tile-size, not total AOI size.
# =========================================================


DEFAULT_YEAR = 2024
DEFAULT_RESOLUTION = 20.0
DEFAULT_TEXTURE_WINDOW = 5
DEFAULT_TILE_SIZE = 512

# JAXA annual PALSAR-2 mosaic DN calibration convention.
CALIBRATION_FACTOR_DB = -83.0

S2_REFERENCE_CANDIDATES = [
    "S2_NDVI.tif",
    "S2_Red_median.tif",
    "S2_NIR_median.tif",
]


# =========================================================
# RUN DIRECTORY
# =========================================================

def find_latest_run(runs_dir: Path) -> Path:
    if not runs_dir.exists():
        raise FileNotFoundError(
            f"Runs directory not found:\n{runs_dir}"
        )

    candidates = []

    for run_dir in runs_dir.iterdir():
        aoi_file = (
            run_dir
            / "input"
            / "aoi_wgs84.geojson"
        )

        if (
            run_dir.is_dir()
            and
            aoi_file.exists()
        ):
            candidates.append(
                run_dir
            )

    if not candidates:
        raise FileNotFoundError(
            "No Version 1 run found. "
            "Run v1_prepare_aoi.py first."
        )

    return max(
        candidates,
        key=lambda path:
            path.stat().st_mtime,
    )


def resolve_run_directory(
    run_dir_text: str | None,
) -> Path:
    run_dir = (
        Path(run_dir_text)
        if run_dir_text
        else find_latest_run(
            RUNS_DIR
        )
    )

    aoi_file = (
        run_dir
        / "input"
        / "aoi_wgs84.geojson"
    )

    if not aoi_file.exists():
        raise FileNotFoundError(
            f"Standardized AOI not found:\n{aoi_file}"
        )

    return run_dir


# =========================================================
# AOI
# =========================================================

def read_aoi(
    aoi_file: Path,
) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(
        aoi_file
    )

    if aoi.empty:
        raise ValueError(
            "AOI is empty."
        )

    if aoi.crs is None:
        raise ValueError(
            "AOI has no CRS."
        )

    aoi = aoi.to_crs(
        "EPSG:4326"
    )

    aoi = aoi[
        aoi.geometry.notna()
        &
        ~aoi.geometry.is_empty
    ].copy()

    if aoi.empty:
        raise ValueError(
            "AOI contains no valid geometry."
        )

    try:
        geometry = (
            aoi.geometry.union_all()
        )
    except AttributeError:
        geometry = (
            aoi.unary_union
        )

    return gpd.GeoDataFrame(
        geometry=[
            geometry
        ],
        crs="EPSG:4326",
    )


# =========================================================
# REQUIRED JAXA 1-DEGREE TILE IDS
# =========================================================

def latitude_code(
    latitude: int,
) -> str:
    hemisphere = (
        "N"
        if latitude >= 0
        else "S"
    )

    return (
        f"{hemisphere}"
        f"{abs(latitude):02d}"
    )


def longitude_code(
    longitude: int,
) -> str:
    hemisphere = (
        "E"
        if longitude >= 0
        else "W"
    )

    return (
        f"{hemisphere}"
        f"{abs(longitude):03d}"
    )


def required_tile_ids(
    aoi: gpd.GeoDataFrame,
) -> list[str]:
    west, south, east, north = map(
        float,
        aoi.total_bounds,
    )

    epsilon = 1e-10

    first_lon = math.floor(
        west
    )

    last_lon = math.floor(
        east - epsilon
    )

    first_lat = math.floor(
        south
    )

    last_lat = math.floor(
        north - epsilon
    )

    identifiers = []

    for latitude in range(
        first_lat,
        last_lat + 1,
    ):
        for longitude in range(
            first_lon,
            last_lon + 1,
        ):
            identifiers.append(
                latitude_code(
                    latitude
                )
                +
                longitude_code(
                    longitude
                )
            )

    return identifiers


# =========================================================
# ARCHIVE EXTRACTION
# =========================================================

def is_archive(
    path: Path,
) -> bool:
    lower = (
        path.name.lower()
    )

    return (
        lower.endswith(
            ".zip"
        )
        or
        lower.endswith(
            ".tar"
        )
        or
        lower.endswith(
            ".tar.gz"
        )
        or
        lower.endswith(
            ".tgz"
        )
    )


def extract_archive(
    archive_file: Path,
    extraction_root: Path,
) -> Path:
    scene_name = (
        archive_file.name
        .replace(
            ".tar.gz",
            "",
        )
        .replace(
            ".tgz",
            "",
        )
        .replace(
            ".zip",
            "",
        )
        .replace(
            ".tar",
            "",
        )
    )

    target = (
        extraction_root
        /
        scene_name[:60]
    )

    if target.exists():
        existing = list(
            target.rglob(
                "*"
            )
        )

        if existing:
            return target

        shutil.rmtree(
            target,
            ignore_errors=True,
        )

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    lower = (
        archive_file.name.lower()
    )

    if lower.endswith(
        ".zip"
    ):
        with zipfile.ZipFile(
            archive_file,
            "r",
        ) as archive:
            archive.extractall(
                target
            )
    else:
        with tarfile.open(
            archive_file,
            "r:*",
        ) as archive:
            archive.extractall(
                target
            )

    return target


def prepare_local_data(
    raw_dir: Path,
    extraction_dir: Path,
) -> list[Path]:
    extraction_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    roots = [
        raw_dir
    ]

    for archive_file in (
        raw_dir.rglob(
            "*"
        )
    ):
        if (
            archive_file.is_file()
            and
            is_archive(
                archive_file
            )
        ):
            print(
                f"Extracting:\n"
                f"{archive_file}"
            )

            roots.append(
                extract_archive(
                    archive_file,
                    extraction_dir,
                )
            )

    return roots


# =========================================================
# ALOS FILE DISCOVERY
# =========================================================

def is_raster_file(
    path: Path,
) -> bool:
    return (
        path.is_file()
        and
        path.suffix.lower()
        in {
            ".tif",
            ".tiff",
        }
    )


def identify_component(
    path: Path,
) -> str | None:
    name = (
        path.name.upper()
    )

    # Check metadata components first so names containing other
    # substrings are not misclassified.
    if (
        "LINCI" in name
        or
        "INCIDENCE" in name
    ):
        return "LINCI"

    if (
        "MASK" in name
        or
        "_MSK" in name
    ):
        return "MASK"

    if (
        "DATE" in name
        or
        "_DTE" in name
    ):
        return "DATE"

    if (
        "_HH" in name
        or
        name.endswith(
            "HH.TIF"
        )
        or
        "HH_" in name
    ):
        return "HH"

    if (
        "_HV" in name
        or
        name.endswith(
            "HV.TIF"
        )
        or
        "HV_" in name
    ):
        return "HV"

    return None


def discover_alos_files(
    search_roots: list[Path],
    tile_ids: list[str],
    year: int,
) -> pd.DataFrame:
    records = []

    year_code = (
        str(year)[-2:]
    )

    for root in search_roots:
        for path in root.rglob(
            "*"
        ):
            if not is_raster_file(
                path
            ):
                continue

            component = (
                identify_component(
                    path
                )
            )

            if component is None:
                continue

            path_upper = (
                str(path).upper()
            )

            matching_tile = ""

            for tile_id in tile_ids:
                if tile_id in path_upper:
                    matching_tile = (
                        tile_id
                    )
                    break

            year_match = (
                str(year)
                in path_upper
                or
                f"_{year_code}"
                in path_upper
                or
                year_code
                in path.name
            )

            records.append(
                {
                    "file":
                        str(
                            path
                        ),

                    "component":
                        component,

                    "tile_id":
                        matching_tile,

                    "year_match":
                        bool(
                            year_match
                        ),
                }
            )

    dataframe = pd.DataFrame(
        records
    )

    if dataframe.empty:
        return dataframe

    preferred = dataframe[
        dataframe[
            "year_match"
        ]
    ].copy()

    if not preferred.empty:
        dataframe = preferred

    # Prefer files that explicitly match required tile IDs when
    # at least some filenames carry tile identifiers.
    explicit = dataframe[
        dataframe[
            "tile_id"
        ]
        !=
        ""
    ]

    if not explicit.empty:
        dataframe = dataframe[
            (
                dataframe[
                    "tile_id"
                ]
                !=
                ""
            )
        ].copy()

    return (
        dataframe
        .drop_duplicates(
            subset=[
                "file"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# REFERENCE GRID
# =========================================================

def find_sentinel2_reference(
    run_dir: Path,
) -> Path | None:
    s2_dir = (
        run_dir
        / "processed"
        / "sentinel2"
    )

    for filename in (
        S2_REFERENCE_CANDIDATES
    ):
        candidate = (
            s2_dir
            /
            filename
        )

        if candidate.exists():
            return candidate

    return None


def reference_profile_from_raster(
    raster_file: Path,
) -> dict:
    with rasterio.open(
        raster_file
    ) as source:
        profile = (
            source.profile.copy()
        )

    profile.update(
        count=1,
        dtype="float32",
        nodata=np.nan,
    )

    return profile


def create_reference_grid(
    aoi: gpd.GeoDataFrame,
    resolution: float,
) -> dict:
    projected_crs = (
        aoi.estimate_utm_crs()
    )

    if projected_crs is None:
        projected_crs = (
            "EPSG:6933"
        )

    projected_aoi = (
        aoi.to_crs(
            projected_crs
        )
    )

    west, south, east, north = (
        projected_aoi.total_bounds
    )

    west = (
        math.floor(
            west
            /
            resolution
        )
        *
        resolution
    )

    south = (
        math.floor(
            south
            /
            resolution
        )
        *
        resolution
    )

    east = (
        math.ceil(
            east
            /
            resolution
        )
        *
        resolution
    )

    north = (
        math.ceil(
            north
            /
            resolution
        )
        *
        resolution
    )

    return {
        "driver":
            "GTiff",

        "width":
            int(
                round(
                    (
                        east
                        -
                        west
                    )
                    /
                    resolution
                )
            ),

        "height":
            int(
                round(
                    (
                        north
                        -
                        south
                    )
                    /
                    resolution
                )
            ),

        "count":
            1,

        "dtype":
            "float32",

        "crs":
            projected_crs,

        "transform":
            from_origin(
                west,
                north,
                resolution,
                resolution,
            ),

        "nodata":
            np.nan,
    }


def resolve_reference_profile(
    run_dir: Path,
    aoi: gpd.GeoDataFrame,
    resolution: float,
) -> tuple[dict, str]:
    reference_file = (
        find_sentinel2_reference(
            run_dir
        )
    )

    if reference_file:
        return (
            reference_profile_from_raster(
                reference_file
            ),
            str(
                reference_file
            ),
        )

    return (
        create_reference_grid(
            aoi,
            resolution,
        ),
        "generated_aoi_grid",
    )


# =========================================================
# WINDOWS
# =========================================================

def generate_windows(
    width: int,
    height: int,
    tile_size: int,
):
    for row_off in range(
        0,
        height,
        tile_size,
    ):
        window_height = min(
            tile_size,
            height - row_off,
        )

        for col_off in range(
            0,
            width,
            tile_size,
        ):
            window_width = min(
                tile_size,
                width - col_off,
            )

            yield Window(
                col_off,
                row_off,
                window_width,
                window_height,
            )


def expanded_window(
    window: Window,
    width: int,
    height: int,
    halo: int,
) -> Window:
    col0 = max(
        0,
        int(
            window.col_off
        )
        -
        halo,
    )

    row0 = max(
        0,
        int(
            window.row_off
        )
        -
        halo,
    )

    col1 = min(
        width,
        int(
            window.col_off
            +
            window.width
        )
        +
        halo,
    )

    row1 = min(
        height,
        int(
            window.row_off
            +
            window.height
        )
        +
        halo,
    )

    return Window(
        col0,
        row0,
        col1 - col0,
        row1 - row0,
    )


def crop_halo(
    array: np.ndarray,
    outer: Window,
    inner: Window,
) -> np.ndarray:
    row_start = int(
        inner.row_off
        -
        outer.row_off
    )

    col_start = int(
        inner.col_off
        -
        outer.col_off
    )

    row_end = (
        row_start
        +
        int(
            inner.height
        )
    )

    col_end = (
        col_start
        +
        int(
            inner.width
        )
    )

    return array[
        row_start:row_end,
        col_start:col_end,
    ]


# =========================================================
# VRT SOURCE MOSAIC
# =========================================================

def open_warped_sources(
    files: list[Path],
    reference_profile: dict,
):
    datasets = []
    vrts = []

    for path in files:
        source = rasterio.open(
            path
        )

        datasets.append(
            source
        )

        vrt = WarpedVRT(
            source,
            crs=
                reference_profile[
                    "crs"
                ],
            transform=
                reference_profile[
                    "transform"
                ],
            width=
                reference_profile[
                    "width"
                ],
            height=
                reference_profile[
                    "height"
                ],
            resampling=
                Resampling.bilinear,
            nodata=np.nan,
            dtype="float32",
        )

        vrts.append(
            vrt
        )

    return datasets, vrts


def close_warped_sources(
    datasets,
    vrts,
):
    for vrt in vrts:
        try:
            vrt.close()
        except Exception:
            pass

    for source in datasets:
        try:
            source.close()
        except Exception:
            pass


def read_mosaic_window(
    vrts,
    window: Window,
) -> np.ndarray:
    shape = (
        int(
            window.height
        ),
        int(
            window.width
        ),
    )

    mosaic = np.full(
        shape,
        np.nan,
        dtype="float32",
    )

    for vrt in vrts:
        data = vrt.read(
            1,
            window=window,
            out_shape=shape,
            masked=True,
        )

        array = np.asarray(
            data.filled(
                np.nan
            ),
            dtype="float32",
        )

        fill = (
            ~np.isfinite(
                mosaic
            )
            &
            np.isfinite(
                array
            )
        )

        mosaic[
            fill
        ] = array[
            fill
        ]

    return mosaic


# =========================================================
# AOI MASK PER WINDOW
# =========================================================

def projected_geometries(
    aoi: gpd.GeoDataFrame,
    target_crs,
):
    projected = aoi.to_crs(
        target_crs
    )

    return [
        geometry
        for geometry in (
            projected.geometry
        )
        if (
            geometry is not None
            and
            not geometry.is_empty
        )
    ]


def aoi_mask_for_window(
    geometries,
    reference_profile: dict,
    window: Window,
) -> np.ndarray:
    transform = rasterio.windows.transform(
        window,
        reference_profile[
            "transform"
        ],
    )

    return geometry_mask(
        geometries,
        out_shape=(
            int(
                window.height
            ),
            int(
                window.width
            ),
        ),
        transform=transform,
        invert=True,
        all_touched=False,
    )


# =========================================================
# ALOS FEATURES
# =========================================================

def amplitude_to_gamma0_db(
    digital_number: np.ndarray,
) -> np.ndarray:
    output = np.full(
        digital_number.shape,
        np.nan,
        dtype="float32",
    )

    valid = (
        np.isfinite(
            digital_number
        )
        &
        (
            digital_number
            >
            0
        )
    )

    output[
        valid
    ] = (
        10.0
        *
        np.log10(
            digital_number[
                valid
            ]
            **
            2
        )
        +
        CALIBRATION_FACTOR_DB
    )

    return output


def db_to_linear(
    db: np.ndarray,
) -> np.ndarray:
    output = np.full(
        db.shape,
        np.nan,
        dtype="float32",
    )

    valid = np.isfinite(
        db
    )

    output[
        valid
    ] = np.power(
        10.0,
        db[
            valid
        ]
        /
        10.0,
    )

    return output


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    output = np.full(
        numerator.shape,
        np.nan,
        dtype="float32",
    )

    valid = (
        np.isfinite(
            numerator
        )
        &
        np.isfinite(
            denominator
        )
        &
        (
            np.abs(
                denominator
            )
            >
            1e-12
        )
    )

    output[
        valid
    ] = (
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    return output


def local_std(
    data: np.ndarray,
    window: int,
) -> np.ndarray:
    valid = np.isfinite(
        data
    )

    filled = np.where(
        valid,
        data,
        0.0,
    ).astype(
        "float32"
    )

    weights = valid.astype(
        "float32"
    )

    # uniform_filter returns local sums divided by kernel size.
    # Dividing the filtered values by filtered valid weights
    # correctly forms the local valid-pixel mean.
    count = uniform_filter(
        weights,
        size=window,
        mode="nearest",
    )

    mean = (
        uniform_filter(
            filled,
            size=window,
            mode="nearest",
        )
        /
        np.maximum(
            count,
            1e-6,
        )
    )

    mean_square = (
        uniform_filter(
            filled
            *
            filled,
            size=window,
            mode="nearest",
        )
        /
        np.maximum(
            count,
            1e-6,
        )
    )

    variance = np.maximum(
        mean_square
        -
        mean
        *
        mean,
        0.0,
    )

    output = np.sqrt(
        variance
    ).astype(
        "float32"
    )

    output[
        ~valid
    ] = np.nan

    return output


def build_alos_features(
    hh_dn: np.ndarray,
    hv_dn: np.ndarray,
    texture_window: int,
) -> dict[str, np.ndarray]:
    hh_db = (
        amplitude_to_gamma0_db(
            hh_dn
        )
    )

    hv_db = (
        amplitude_to_gamma0_db(
            hv_dn
        )
    )

    hh_linear = (
        db_to_linear(
            hh_db
        )
    )

    hv_linear = (
        db_to_linear(
            hv_db
        )
    )

    hh_minus_hv = (
        hh_db
        -
        hv_db
    ).astype(
        "float32"
    )

    hh_hv_ratio = (
        safe_ratio(
            hh_linear,
            hv_linear,
        )
    )

    rfdi = safe_ratio(
        hh_linear
        -
        hv_linear,

        hh_linear
        +
        hv_linear,
    )

    hh_texture = local_std(
        hh_db,
        texture_window,
    )

    hv_texture = local_std(
        hv_db,
        texture_window,
    )

    return {
        "HH_dB":
            hh_db,

        "HV_dB":
            hv_db,

        "HH_minus_HV_dB":
            hh_minus_hv,

        "HH_HV_ratio":
            hh_hv_ratio,

        "RFDI":
            rfdi,

        "HH_texture":
            hh_texture,

        "HV_texture":
            hv_texture,
    }


# =========================================================
# OUTPUT DATASETS
# =========================================================

def clean_profile(
    profile: dict,
) -> dict:
    output = (
        profile.copy()
    )

    for key in [
        "blockxsize",
        "blockysize",
        "interleave",
    ]:
        output.pop(
            key,
            None,
        )

    return output


def create_float_output(
    output_file: Path,
    profile: dict,
    description: str,
):
    output_profile = (
        clean_profile(
            profile
        )
    )

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = rasterio.open(
        output_file,
        "w",
        **output_profile,
    )

    destination.set_band_description(
        1,
        description,
    )

    return destination


def create_uint8_output(
    output_file: Path,
    profile: dict,
    description: str,
):
    output_profile = (
        clean_profile(
            profile
        )
    )

    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    destination = rasterio.open(
        output_file,
        "w",
        **output_profile,
    )

    destination.set_band_description(
        1,
        description,
    )

    return destination


# =========================================================
# STATUS WHEN ALOS-2 IS ABSENT
# =========================================================

def save_unavailable_status(
    reports_dir: Path,
    year: int,
    tile_ids: list[str],
    message: str,
) -> dict:
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    status = {
        "pipeline_role":
            "optional_prediction_predictor",

        "alos2_required":
            False,

        "alos2_available":
            False,

        "year":
            year,

        "required_tiles":
            tile_ids,

        "message":
            message,

        "platform_can_continue":
            True,

        "core_predictors":
            [
                "Sentinel-2",
                "Sentinel-1",
                "Terrain",
            ],

        "next_action":
            (
                "Continue to V7 without ALOS-2, or place "
                "the required JAXA annual mosaic HH/HV "
                "GeoTIFFs/archives in raw/alos2 and rerun V5."
            ),
    }

    status_file = (
        reports_dir
        /
        "alos2_processing_status.json"
    )

    status_file.write_text(
        json.dumps(
            status,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "available":
            False,

        "status_file":
            status_file,

        "required_tiles":
            tile_ids,
    }


# =========================================================
# MAIN PROCESSING
# =========================================================

def process_alos2(
    run_dir: Path,
    year: int,
    resolution: float,
    texture_window: int,
    tile_size: int,
) -> dict:
    aoi_file = (
        run_dir
        / "input"
        / "aoi_wgs84.geojson"
    )

    raw_dir = (
        run_dir
        / "raw"
        / "alos2"
    )

    # The dashboard can reference original local mosaics without copying them.
    source_config = run_dir / "input" / "alos_source.json"
    if source_config.exists():
        raw_dir = Path(json.loads(source_config.read_text())["folder"])
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"ALOS-2 source folder no longer exists: {raw_dir}")

    # Extract archives into this study's cache.
    extraction_dir = (
        Path(
            run_dir / "cache" / "alos2_extract"
        )
        /
        run_dir.name
    )

    output_dir = (
        run_dir
        / "processed"
        / "alos2"
    )

    reports_dir = (
        run_dir
        / "reports"
        / "alos2"
    )

    for directory in [
        raw_dir,
        extraction_dir,
        output_dir,
        reports_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "\n[1/8] Reading AOI..."
    )

    aoi = read_aoi(
        aoi_file
    )

    tile_ids = (
        required_tile_ids(
            aoi
        )
    )

    print(
        f"Required JAXA 1-degree tiles: "
        f"{len(tile_ids)}"
    )

    for tile_id in tile_ids:
        print(
            f"  {tile_id}"
        )

    print(
        "\n[2/8] Preparing local ALOS-2 annual mosaic data..."
    )

    search_roots = (
        prepare_local_data(
            raw_dir,
            extraction_dir,
        )
    )

    print(
        "\n[3/8] Discovering HH/HV files..."
    )

    discovered = (
        discover_alos_files(
            search_roots,
            tile_ids,
            year,
        )
    )

    discovery_file = (
        reports_dir
        /
        "alos2_discovered_files.csv"
    )

    discovered.to_csv(
        discovery_file,
        index=False,
    )

    if discovered.empty:
        return save_unavailable_status(
            reports_dir,
            year,
            tile_ids,
            (
                "No JAXA PALSAR-2 annual mosaic GeoTIFFs "
                "or archives were found in raw/alos2."
            ),
        )

    hh_files = [
        Path(
            path
        )
        for path in (
            discovered.loc[
                discovered[
                    "component"
                ]
                ==
                "HH",
                "file",
            ].tolist()
        )
    ]

    hv_files = [
        Path(
            path
        )
        for path in (
            discovered.loc[
                discovered[
                    "component"
                ]
                ==
                "HV",
                "file",
            ].tolist()
        )
    ]

    if (
        not hh_files
        or
        not hv_files
    ):
        return save_unavailable_status(
            reports_dir,
            year,
            tile_ids,
            (
                "ALOS-2 files were found, but both HH and HV "
                "components are required for the enhanced model."
            ),
        )

    print(
        f"HH files: "
        f"{len(hh_files)}"
    )

    print(
        f"HV files: "
        f"{len(hv_files)}"
    )

    print(
        "\n[4/8] Resolving Sentinel/common grid..."
    )

    reference_profile, reference_source = (
        resolve_reference_profile(
            run_dir,
            aoi,
            resolution,
        )
    )

    print(
        f"Reference : "
        f"{reference_source}"
    )

    print(
        f"CRS       : "
        f"{reference_profile['crs']}"
    )

    print(
        f"Raster    : "
        f"{reference_profile['width']} x "
        f"{reference_profile['height']}"
    )

    print(
        f"Tile size : "
        f"{tile_size} x {tile_size}"
    )

    print(
        "\n[5/8] Opening windowed warped ALOS-2 sources..."
    )

    hh_datasets, hh_vrts = (
        open_warped_sources(
            hh_files,
            reference_profile,
        )
    )

    hv_datasets, hv_vrts = (
        open_warped_sources(
            hv_files,
            reference_profile,
        )
    )

    geometries = (
        projected_geometries(
            aoi,
            reference_profile[
                "crs"
            ],
        )
    )

    filenames = {
        "HH_dB":
            f"ALOS2_{year}_HH_gamma0_dB.tif",

        "HV_dB":
            f"ALOS2_{year}_HV_gamma0_dB.tif",

        "HH_minus_HV_dB":
            f"ALOS2_{year}_HH_minus_HV_dB.tif",

        "HH_HV_ratio":
            f"ALOS2_{year}_HH_HV_linear_ratio.tif",

        "RFDI":
            f"ALOS2_{year}_RFDI.tif",

        "HH_texture":
            f"ALOS2_{year}_HH_texture_std.tif",

        "HV_texture":
            f"ALOS2_{year}_HV_texture_std.tif",
    }

    output_paths = {
        key:
            output_dir
            /
            filename

        for key, filename in (
            filenames.items()
        )
    }

    valid_mask_file = (
        output_dir
        /
        f"ALOS2_{year}_valid_mask.tif"
    )

    outputs = {}

    valid_destination = None

    try:
        print(
            "\n[6/8] Creating output rasters..."
        )

        for key, path in (
            output_paths.items()
        ):
            outputs[
                key
            ] = create_float_output(
                path,
                reference_profile,
                key,
            )

        valid_destination = (
            create_uint8_output(
                valid_mask_file,
                reference_profile,
                "ALOS2_valid_mask",
            )
        )

        windows = list(
            generate_windows(
                reference_profile[
                    "width"
                ],
                reference_profile[
                    "height"
                ],
                tile_size,
            )
        )

        total_windows = len(
            windows
        )

        # Halo avoids texture seams between neighbouring processing tiles.
        halo = max(
            1,
            texture_window
            //
            2,
        )

        print(
            "\n[7/8] Processing ALOS-2 window-by-window..."
        )

        for index, window in enumerate(
            windows,
            start=1,
        ):
            outer = expanded_window(
                window,
                reference_profile[
                    "width"
                ],
                reference_profile[
                    "height"
                ],
                halo,
            )

            hh_outer = (
                read_mosaic_window(
                    hh_vrts,
                    outer,
                )
            )

            hv_outer = (
                read_mosaic_window(
                    hv_vrts,
                    outer,
                )
            )

            features_outer = (
                build_alos_features(
                    hh_outer,
                    hv_outer,
                    texture_window,
                )
            )

            aoi_mask = (
                aoi_mask_for_window(
                    geometries,
                    reference_profile,
                    window,
                )
            )

            hh_core = crop_halo(
                features_outer[
                    "HH_dB"
                ],
                outer,
                window,
            )

            hv_core = crop_halo(
                features_outer[
                    "HV_dB"
                ],
                outer,
                window,
            )

            valid_mask = (
                aoi_mask
                &
                np.isfinite(
                    hh_core
                )
                &
                np.isfinite(
                    hv_core
                )
            )

            for key, array_outer in (
                features_outer.items()
            ):
                core = crop_halo(
                    array_outer,
                    outer,
                    window,
                ).astype(
                    "float32"
                )

                core[
                    ~valid_mask
                ] = np.nan

                outputs[
                    key
                ].write(
                    core,
                    1,
                    window=window,
                )

            valid_destination.write(
                valid_mask.astype(
                    "uint8"
                ),
                1,
                window=window,
            )

            if (
                index == 1
                or
                index % 25 == 0
                or
                index == total_windows
            ):
                print(
                    f"Processed window "
                    f"{index}/{total_windows}"
                )

        print(
            "\n[8/8] Saving status..."
        )

    finally:
        for destination in (
            outputs.values()
        ):
            try:
                destination.close()
            except Exception:
                pass

        if valid_destination is not None:
            try:
                valid_destination.close()
            except Exception:
                pass

        close_warped_sources(
            hh_datasets,
            hh_vrts,
        )

        close_warped_sources(
            hv_datasets,
            hv_vrts,
        )

    status = {
        "pipeline_role":
            "optional_prediction_predictor",

        "alos2_required":
            False,

        "alos2_available":
            True,

        "year":
            year,

        "required_tiles":
            tile_ids,

        "hh_files":
            [
                str(
                    path
                )
                for path in (
                    hh_files
                )
            ],

        "hv_files":
            [
                str(
                    path
                )
                for path in (
                    hv_files
                )
            ],

        "reference_source":
            reference_source,

        "resolution_m":
            resolution,

        "processing_tile_size":
            tile_size,

        "windowed_processing":
            True,

        "calibration_equation":
            (
                "Gamma0_dB = "
                "10*log10(DN^2) - 83"
            ),

        "output_files":
            {
                key:
                    str(
                        value
                    )

                for key, value in (
                    output_paths.items()
                )
            },

        "valid_mask":
            str(
                valid_mask_file
            ),

        "next_step":
            (
                "V7 predictor stacking. "
                "GEDI is not required for this user AOI."
            ),
    }

    status_file = (
        reports_dir
        /
        "alos2_processing_status.json"
    )

    status_file.write_text(
        json.dumps(
            status,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "available":
            True,

        "output_dir":
            output_dir,

        "status_file":
            status_file,

        "required_tiles":
            tile_ids,
    }


# =========================================================
# COMMAND LINE
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Process optional JAXA ALOS-2 PALSAR-2 annual "
            "mosaic HH/HV data as L-band biomass predictors. "
            "Windowed processing supports large AOIs."
        )
    )

    parser.add_argument(
        "--run-dir",
        default=None,
    )

    parser.add_argument(
        "--year",
        type=int,
        default=
            DEFAULT_YEAR,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=
            DEFAULT_RESOLUTION,
    )

    parser.add_argument(
        "--texture-window",
        type=int,
        default=
            DEFAULT_TEXTURE_WINDOW,
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=
            DEFAULT_TILE_SIZE,
    )

    return parser.parse_args()


def main():
    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "BIOMASS AND CARBON PLATFORM"
    )

    print(
        "VERSION 5 - OPTIONAL ALOS-2 PALSAR-2 PREDICTORS"
    )

    print(
        "=" * 72
    )

    args = parse_arguments()

    run_dir = (
        resolve_run_directory(
            args.run_dir
        )
    )

    print(
        f"\nUsing run:\n"
        f"{run_dir}"
    )

    result = process_alos2(
        run_dir=
            run_dir,

        year=
            args.year,

        resolution=
            args.resolution,

        texture_window=
            args.texture_window,

        tile_size=
            args.tile_size,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    if result[
        "available"
    ]:
        print(
            "VERSION 5 COMPLETE"
        )

        print(
            "=" * 72
        )

        print(
            f"\nALOS-2 outputs:\n"
            f"{result['output_dir']}"
        )

        print(
            "\nALOS-2 will be used as an optional "
            "enhanced biomass predictor."
        )

    else:
        print(
            "VERSION 5 COMPLETE - "
            "ALOS-2 OPTIONAL DATA MISSING"
        )

        print(
            "=" * 72
        )

        print(
            "\nThis is NOT an error."
        )

        print(
            "The core platform continues with "
            "Sentinel-2 + Sentinel-1 + Terrain."
        )

        print(
            "\nRequired JAXA annual mosaic tiles:"
        )

        for tile_id in (
            result[
                "required_tiles"
            ]
        ):
            print(
                f"  {tile_id}"
            )

    print(
        f"\nStatus:\n"
        f"{result['status_file']}"
    )

    print(
        "\nNext: V7 predictor stacks."
    )

    print(
        "GEDI is separate training-only data and "
        "is not required for this user AOI."
    )


if __name__ == "__main__":
    main()
