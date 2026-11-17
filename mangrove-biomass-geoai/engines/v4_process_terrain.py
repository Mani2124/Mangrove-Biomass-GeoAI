
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import shutil

import geopandas as gpd
import numpy as np
import rasterio
import requests

from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter

from config import RUNS_DIR


# =========================================================
# PUBLIC COPERNICUS DEM SETTINGS
# =========================================================

GLO30_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
GLO90_BUCKET = "https://copernicus-dem-90m.s3.amazonaws.com"

DEFAULT_RESOLUTION = 20.0
DEFAULT_TPI_WINDOW = 11
DEFAULT_ROUGHNESS_WINDOW = 5

S2_REFERENCE_CANDIDATES = [
    "S2_NDVI.tif",
    "S2_Red_median.tif",
    "S2_NIR_median.tif",
]


# =========================================================
# RUN DIRECTORY
# =========================================================

def find_latest_run(
    runs_dir: Path,
) -> Path:
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

        if run_dir.is_dir() and aoi_file.exists():
            candidates.append(run_dir)

    if not candidates:
        raise FileNotFoundError(
            "No Version 1 run found. "
            "Run v1_prepare_aoi.py first."
        )

    return max(
        candidates,
        key=lambda path: path.stat().st_mtime,
    )


def resolve_run_directory(
    run_dir_text: str | None,
) -> Path:
    run_dir = (
        Path(run_dir_text)
        if run_dir_text
        else find_latest_run(RUNS_DIR)
    )

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found:\n{run_dir}"
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
    aoi = gpd.read_file(aoi_file)

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
        geometry=[geometry],
        crs="EPSG:4326",
    )


# =========================================================
# TILE NAMING
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
        f"{abs(latitude):02d}_00"
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
        f"{abs(longitude):03d}_00"
    )


def tile_name(
    latitude: int,
    longitude: int,
    arc_seconds: int,
) -> str:
    return (
        "Copernicus_DSM_COG_"
        f"{arc_seconds}_"
        f"{latitude_code(latitude)}_"
        f"{longitude_code(longitude)}_DEM"
    )


def tile_url(
    latitude: int,
    longitude: int,
    arc_seconds: int,
) -> str:
    bucket = (
        GLO30_BUCKET
        if arc_seconds == 10
        else GLO90_BUCKET
    )

    name = tile_name(
        latitude,
        longitude,
        arc_seconds,
    )

    return (
        f"{bucket}/"
        f"{name}/"
        f"{name}.tif"
    )


def integer_tile_range(
    minimum: float,
    maximum: float,
    lower_limit: int,
    upper_limit: int,
) -> range:
    epsilon = 1e-10

    first = max(
        lower_limit,
        math.floor(minimum),
    )

    last = min(
        upper_limit - 1,
        math.floor(
            maximum - epsilon
        ),
    )

    if last < first:
        return range(0)

    return range(
        first,
        last + 1,
    )


def required_tile_coordinates(
    aoi: gpd.GeoDataFrame,
) -> list[tuple[int, int]]:
    west, south, east, north = map(
        float,
        aoi.total_bounds,
    )

    latitudes = integer_tile_range(
        south,
        north,
        -90,
        90,
    )

    longitudes = integer_tile_range(
        west,
        east,
        -180,
        180,
    )

    return [
        (
            latitude,
            longitude,
        )
        for latitude in latitudes
        for longitude in longitudes
    ]


# =========================================================
# DOWNLOAD
# =========================================================

def url_exists(
    url: str,
) -> bool:
    try:
        response = requests.head(
            url,
            allow_redirects=True,
            timeout=30,
        )

        if response.status_code == 200:
            return True

        # Some S3 configurations may not accept HEAD as expected.
        if response.status_code in {
            403,
            405,
        }:
            response = requests.get(
                url,
                headers={
                    "Range":
                        "bytes=0-1023"
                },
                stream=True,
                timeout=30,
            )

            return response.status_code in {
                200,
                206,
            }

        return False

    except requests.RequestException:
        return False


def download_file(
    url: str,
    output_file: Path,
) -> Path:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        output_file.exists()
        and
        output_file.stat().st_size
        >
        10_000
    ):
        print(
            f"Already downloaded:\n"
            f"{output_file}"
        )

        return output_file

    temporary = output_file.with_suffix(
        ".tif.part"
    )

    print(
        f"Downloading:\n{url}"
    )

    with requests.get(
        url,
        stream=True,
        timeout=(60, 1800),
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                "Public Copernicus DEM download failed.\n"
                f"HTTP {response.status_code}\n"
                f"URL:\n{url}"
            )

        with open(
            temporary,
            "wb",
        ) as file:
            for chunk in response.iter_content(
                chunk_size=
                    1024
                    *
                    1024
            ):
                if chunk:
                    file.write(chunk)

    temporary.replace(
        output_file
    )

    with rasterio.open(
        output_file
    ) as source:
        if source.count < 1:
            raise RuntimeError(
                "Downloaded DEM contains no raster band:\n"
                f"{output_file}"
            )

    return output_file


def acquire_dem_tiles(
    aoi: gpd.GeoDataFrame,
    raw_dir: Path,
) -> tuple[list[Path], list[dict]]:
    coordinates = required_tile_coordinates(
        aoi
    )

    if not coordinates:
        raise RuntimeError(
            "No DEM tile coordinates were calculated."
        )

    downloaded = []
    records = []

    for index, (
        latitude,
        longitude,
    ) in enumerate(
        coordinates,
        start=1,
    ):
        print(
            f"\nTile {index}/{len(coordinates)}: "
            f"{latitude_code(latitude)} "
            f"{longitude_code(longitude)}"
        )

        selected_resolution = None
        selected_url = None

        # First try public GLO-30.
        glo30_url = tile_url(
            latitude,
            longitude,
            10,
        )

        if url_exists(
            glo30_url
        ):
            selected_resolution = 30
            selected_url = glo30_url
            arc_seconds = 10

        else:
            # Global fallback to public GLO-90.
            glo90_url = tile_url(
                latitude,
                longitude,
                30,
            )

            if url_exists(
                glo90_url
            ):
                selected_resolution = 90
                selected_url = glo90_url
                arc_seconds = 30

            else:
                print(
                    "No public land tile found. "
                    "This may be ocean or unavailable coverage."
                )

                records.append(
                    {
                        "latitude":
                            latitude,

                        "longitude":
                            longitude,

                        "status":
                            "missing",

                        "source_resolution_m":
                            None,

                        "url":
                            "",
                    }
                )

                continue

        name = tile_name(
            latitude,
            longitude,
            arc_seconds,
        )

        output_file = (
            raw_dir
            /
            (
                "glo30"
                if selected_resolution == 30
                else "glo90"
            )
            /
            f"{name}.tif"
        )

        downloaded_file = download_file(
            selected_url,
            output_file,
        )

        downloaded.append(
            downloaded_file
        )

        records.append(
            {
                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "status":
                    "downloaded",

                "source_resolution_m":
                    selected_resolution,

                "url":
                    selected_url,

                "local_file":
                    str(
                        downloaded_file
                    ),
            }
        )

    if not downloaded:
        raise RuntimeError(
            "No public Copernicus DEM tiles were downloaded "
            "for this AOI."
        )

    return downloaded, records


# =========================================================
# MOSAIC
# =========================================================

def mosaic_dem_tiles(
    dem_files: list[Path],
    output_file: Path,
) -> Path:
    datasets = [
        rasterio.open(path)
        for path in dem_files
    ]

    try:
        mosaic, transform = merge(
            datasets,
            nodata=np.nan,
            dtype="float32",
        )

        profile = datasets[
            0
        ].profile.copy()

        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with rasterio.open(
            output_file,
            "w",
            **profile,
        ) as destination:
            destination.write(
                mosaic[0].astype(
                    "float32"
                ),
                1,
            )

    finally:
        for dataset in datasets:
            dataset.close()

    return output_file


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
            / filename
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

    projected_aoi = aoi.to_crs(
        projected_crs
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
        print(
            "Using Sentinel-2 grid:"
        )

        print(
            reference_file
        )

        return (
            reference_profile_from_raster(
                reference_file
            ),
            str(
                reference_file
            ),
        )

    print(
        "Sentinel-2 reference not found. "
        "Creating an AOI grid."
    )

    return (
        create_reference_grid(
            aoi,
            resolution,
        ),
        "generated_aoi_grid",
    )


# =========================================================
# ALIGNMENT AND AOI MASK
# =========================================================

def align_dem(
    dem_file: Path,
    reference_profile: dict,
) -> np.ndarray:
    destination = np.full(
        (
            reference_profile[
                "height"
            ],
            reference_profile[
                "width"
            ],
        ),
        np.nan,
        dtype="float32",
    )

    with rasterio.open(
        dem_file
    ) as source:
        source_array = (
            source.read(1)
            .astype(
                "float32"
            )
        )

        if source.nodata is not None:
            source_array[
                source_array
                ==
                source.nodata
            ] = np.nan

        source_array[
            ~np.isfinite(
                source_array
            )
        ] = np.nan

        reproject(
            source=
                source_array,

            destination=
                destination,

            src_transform=
                source.transform,

            src_crs=
                source.crs,

            src_nodata=
                np.nan,

            dst_transform=
                reference_profile[
                    "transform"
                ],

            dst_crs=
                reference_profile[
                    "crs"
                ],

            dst_nodata=
                np.nan,

            resampling=
                Resampling.bilinear,
        )

    return destination


def create_aoi_mask(
    aoi: gpd.GeoDataFrame,
    reference_profile: dict,
) -> np.ndarray:
    projected = aoi.to_crs(
        reference_profile[
            "crs"
        ]
    )

    geometries = [
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

    return geometry_mask(
        geometries,

        out_shape=(
            reference_profile[
                "height"
            ],
            reference_profile[
                "width"
            ],
        ),

        transform=
            reference_profile[
                "transform"
            ],

        invert=True,

        all_touched=False,
    )


# =========================================================
# TERRAIN FEATURES
# =========================================================

def fill_small_gaps(
    elevation: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(
        elevation
    )

    if np.all(valid):
        return elevation.copy()

    filled_values = np.where(
        valid,
        elevation,
        0.0,
    ).astype(
        "float32"
    )

    weights = valid.astype(
        "float32"
    )

    local_sum = uniform_filter(
        filled_values,
        size=3,
        mode="nearest",
    )

    local_count = uniform_filter(
        weights,
        size=3,
        mode="nearest",
    )

    replacement = (
        local_sum
        /
        np.maximum(
            local_count,
            1e-6,
        )
    )

    output = elevation.copy()

    output[
        ~valid
    ] = replacement[
        ~valid
    ]

    return output


def calculate_slope_aspect(
    elevation: np.ndarray,
    transform,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    valid = np.isfinite(
        elevation
    )

    filled = fill_small_gaps(
        elevation
    )

    x_size = abs(
        float(
            transform.a
        )
    )

    y_size = abs(
        float(
            transform.e
        )
    )

    dz_dy, dz_dx = (
        np.gradient(
            filled,
            y_size,
            x_size,
        )
    )

    slope = np.degrees(
        np.arctan(
            np.sqrt(
                dz_dx ** 2
                +
                dz_dy ** 2
            )
        )
    ).astype(
        "float32"
    )

    aspect = (
        np.degrees(
            np.arctan2(
                -dz_dx,
                dz_dy,
            )
        )
        +
        360.0
    ) % 360.0

    aspect = aspect.astype(
        "float32"
    )

    aspect[
        slope < 0.001
    ] = np.nan

    slope[
        ~valid
    ] = np.nan

    aspect[
        ~valid
    ] = np.nan

    return slope, aspect


def nan_uniform_mean(
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

    numerator = uniform_filter(
        filled,
        size=window,
        mode="nearest",
    )

    denominator = uniform_filter(
        weights,
        size=window,
        mode="nearest",
    )

    output = (
        numerator
        /
        np.maximum(
            denominator,
            1e-6,
        )
    ).astype(
        "float32"
    )

    output[
        denominator <= 0
    ] = np.nan

    return output


def calculate_tpi(
    elevation: np.ndarray,
    window: int,
) -> np.ndarray:
    local_mean = nan_uniform_mean(
        elevation,
        window,
    )

    result = (
        elevation
        -
        local_mean
    ).astype(
        "float32"
    )

    result[
        ~np.isfinite(
            elevation
        )
    ] = np.nan

    return result


def calculate_tri(
    elevation: np.ndarray,
    window: int,
) -> np.ndarray:
    local_mean = nan_uniform_mean(
        elevation,
        window,
    )

    local_variance = nan_uniform_mean(
        (
            elevation
            -
            local_mean
        ) ** 2,
        window,
    )

    result = np.sqrt(
        np.maximum(
            local_variance,
            0.0,
        )
    ).astype(
        "float32"
    )

    result[
        ~np.isfinite(
            elevation
        )
    ] = np.nan

    return result


def calculate_roughness(
    elevation: np.ndarray,
    window: int,
) -> np.ndarray:
    valid = np.isfinite(
        elevation
    )

    local_minimum = minimum_filter(
        np.where(
            valid,
            elevation,
            np.inf,
        ),
        size=window,
        mode="nearest",
    )

    local_maximum = maximum_filter(
        np.where(
            valid,
            elevation,
            -np.inf,
        ),
        size=window,
        mode="nearest",
    )

    result = (
        local_maximum
        -
        local_minimum
    ).astype(
        "float32"
    )

    result[
        ~valid
    ] = np.nan

    result[
        ~np.isfinite(
            result
        )
    ] = np.nan

    return result


# =========================================================
# SAVE
# =========================================================

def clean_profile(
    profile: dict,
) -> dict:
    output = profile.copy()

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


def save_float_raster(
    output_file: Path,
    array: np.ndarray,
    profile: dict,
    description: str,
) -> None:
    output_profile = clean_profile(
        profile
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

    with rasterio.open(
        output_file,
        "w",
        **output_profile,
    ) as destination:
        destination.write(
            array.astype(
                "float32"
            ),
            1,
        )

        destination.set_band_description(
            1,
            description,
        )

    print(
        f"Saved: {output_file}"
    )


def save_uint8_raster(
    output_file: Path,
    array: np.ndarray,
    profile: dict,
    description: str,
) -> None:
    output_profile = clean_profile(
        profile
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

    with rasterio.open(
        output_file,
        "w",
        **output_profile,
    ) as destination:
        destination.write(
            array.astype(
                "uint8"
            ),
            1,
        )

        destination.set_band_description(
            1,
            description,
        )

    print(
        f"Saved: {output_file}"
    )


# =========================================================
# MAIN PROCESS
# =========================================================

def process_terrain(
    run_dir: Path,
    resolution: float,
    tpi_window: int,
    roughness_window: int,
) -> dict:
    aoi_file = (
        run_dir
        / "input"
        / "aoi_wgs84.geojson"
    )

    raw_dir = (
        run_dir
        / "raw"
        / "terrain_public"
    )

    output_dir = (
        run_dir
        / "processed"
        / "terrain"
    )

    reports_dir = (
        run_dir
        / "reports"
        / "terrain"
    )

    for directory in [
        raw_dir,
        output_dir,
        reports_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "\n[1/7] Reading AOI..."
    )

    aoi = read_aoi(
        aoi_file
    )

    print(
        "\n[2/7] Discovering and downloading public DEM tiles..."
    )

    dem_files, tile_records = acquire_dem_tiles(
        aoi,
        raw_dir,
    )

    import pandas as pd

    pd.DataFrame(
        tile_records
    ).to_csv(
        reports_dir
        / "dem_tile_download_report.csv",
        index=False,
    )

    print(
        f"\nDEM tiles downloaded: "
        f"{len(dem_files)}"
    )

    print(
        "\n[3/7] Mosaicking DEM tiles..."
    )

    mosaic_file = (
        raw_dir
        / "copernicus_dem_public_mosaic.tif"
    )

    mosaic_dem_tiles(
        dem_files,
        mosaic_file,
    )

    print(
        f"Mosaic:\n{mosaic_file}"
    )

    print(
        "\n[4/7] Resolving common grid..."
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
        f"Size      : "
        f"{reference_profile['width']} x "
        f"{reference_profile['height']}"
    )

    print(
        "\n[5/7] Aligning DEM and calculating terrain features..."
    )

    elevation = align_dem(
        mosaic_file,
        reference_profile,
    )

    aoi_mask = create_aoi_mask(
        aoi,
        reference_profile,
    )

    elevation[
        ~aoi_mask
    ] = np.nan

    slope, aspect = calculate_slope_aspect(
        elevation,
        reference_profile[
            "transform"
        ],
    )

    aspect_radians = np.radians(
        aspect
    )

    northness = np.cos(
        aspect_radians
    ).astype(
        "float32"
    )

    eastness = np.sin(
        aspect_radians
    ).astype(
        "float32"
    )

    tpi = calculate_tpi(
        elevation,
        tpi_window,
    )

    tri = calculate_tri(
        elevation,
        roughness_window,
    )

    roughness = calculate_roughness(
        elevation,
        roughness_window,
    )

    valid_mask = (
        aoi_mask
        &
        np.isfinite(
            elevation
        )
    )

    layers = {
        "terrain_elevation_m.tif":
            elevation,

        "terrain_slope_degrees.tif":
            slope,

        "terrain_aspect_degrees.tif":
            aspect,

        "terrain_northness.tif":
            northness,

        "terrain_eastness.tif":
            eastness,

        "terrain_tpi.tif":
            tpi,

        "terrain_tri.tif":
            tri,

        "terrain_roughness.tif":
            roughness,
    }

    print(
        "\n[6/7] Saving terrain products..."
    )

    for filename, array in layers.items():
        array[
            ~valid_mask
        ] = np.nan

        save_float_raster(
            output_dir
            / filename,

            array,

            reference_profile,

            filename.replace(
                ".tif",
                "",
            ),
        )

    save_uint8_raster(
        output_dir
        / "terrain_valid_mask.tif",

        valid_mask,

        reference_profile,

        "terrain_valid_mask",
    )

    print(
        "\n[7/7] Saving summary..."
    )

    valid_elevation = elevation[
        np.isfinite(
            elevation
        )
    ]

    source_resolutions = sorted(
        {
            record[
                "source_resolution_m"
            ]
            for record in tile_records
            if (
                record.get(
                    "status"
                )
                ==
                "downloaded"
            )
        }
    )

    summary = {
        "source":
            "Public Copernicus DEM AWS",

        "source_resolutions_m":
            source_resolutions,

        "tiles_downloaded":
            len(
                dem_files
            ),

        "reference_source":
            reference_source,

        "output_crs":
            str(
                reference_profile[
                    "crs"
                ]
            ),

        "output_resolution_m":
            abs(
                float(
                    reference_profile[
                        "transform"
                    ].a
                )
            ),

        "width":
            int(
                reference_profile[
                    "width"
                ]
            ),

        "height":
            int(
                reference_profile[
                    "height"
                ]
            ),

        "valid_pixels":
            int(
                valid_elevation.size
            ),

        "minimum_elevation_m":
            (
                float(
                    np.min(
                        valid_elevation
                    )
                )
                if valid_elevation.size
                else None
            ),

        "maximum_elevation_m":
            (
                float(
                    np.max(
                        valid_elevation
                    )
                )
                if valid_elevation.size
                else None
            ),

        "mean_elevation_m":
            (
                float(
                    np.mean(
                        valid_elevation
                    )
                )
                if valid_elevation.size
                else None
            ),

        "tpi_window_pixels":
            int(
                tpi_window
            ),

        "roughness_window_pixels":
            int(
                roughness_window
            ),
    }

    summary_file = (
        reports_dir
        / "terrain_processing_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return {
        "output_dir":
            output_dir,

        "summary_file":
            summary_file,

        "tile_report":
            reports_dir
            /
            "dem_tile_download_report.csv",
    }


# =========================================================
# COMMAND LINE
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Download public Copernicus DEM GLO-30/GLO-90 "
            "tiles and create terrain predictors."
        )
    )

    parser.add_argument(
        "--run-dir",
        default=None,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=
            DEFAULT_RESOLUTION,
    )

    parser.add_argument(
        "--tpi-window",
        type=int,
        default=
            DEFAULT_TPI_WINDOW,
    )

    parser.add_argument(
        "--roughness-window",
        type=int,
        default=
            DEFAULT_ROUGHNESS_WINDOW,
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
        "VERSION 4 - PUBLIC COPERNICUS DEM TERRAIN"
    )

    print(
        "=" * 72
    )

    args = parse_arguments()

    run_dir = resolve_run_directory(
        args.run_dir
    )

    print(
        f"\nUsing run:\n"
        f"{run_dir}"
    )

    result = process_terrain(
        run_dir=
            run_dir,

        resolution=
            args.resolution,

        tpi_window=
            args.tpi_window,

        roughness_window=
            args.roughness_window,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "VERSION 4 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTerrain outputs:\n"
        f"{result['output_dir']}"
    )

    print(
        f"\nTile report:\n"
        f"{result['tile_report']}"
    )

    print(
        f"\nSummary:\n"
        f"{result['summary_file']}"
    )

    print(
        "\nReady for Version 5: "
        "modality stacks and patch generation."
    )


if __name__ == "__main__":
    main()
