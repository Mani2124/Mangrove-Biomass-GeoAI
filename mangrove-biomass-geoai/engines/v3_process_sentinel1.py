
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests

from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from scipy.ndimage import uniform_filter
from shapely.geometry import shape

from config import RUNS_DIR


# =========================================================
# CDSE SETTINGS
# =========================================================

CATALOGUE_URL = (
    "https://catalogue.dataspace.copernicus.eu/"
    "odata/v1/Products"
)

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/"
    "auth/realms/CDSE/protocol/openid-connect/token"
)

DOWNLOAD_URL = (
    "https://zipper.dataspace.copernicus.eu/"
    "odata/v1/Products({product_id})/$value"
)


# =========================================================
# DEFAULTS
# =========================================================

DEFAULT_START_DATE = "2024-04-01"
DEFAULT_END_DATE = "2024-09-30"
DEFAULT_MAX_PRODUCTS = 4
DEFAULT_RESOLUTION = 20.0
DEFAULT_ORBIT_DIRECTION = "ASCENDING"
DEFAULT_TEXTURE_WINDOW = 5

COMMON_SNAP_PATHS = [
    Path(r"C:\Program Files\snap\bin\gpt.exe"),
    Path(r"C:\Program Files\esa-snap\bin\gpt.exe"),
    Path(r"C:\Program Files\SNAP\bin\gpt.exe"),
]


# =========================================================
# VERSION 1 RUN
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
        raise ValueError("AOI is empty.")

    if aoi.crs is None:
        raise ValueError("AOI has no CRS.")

    aoi = aoi.to_crs("EPSG:4326")
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
        geometry = aoi.geometry.union_all()
    except AttributeError:
        geometry = aoi.unary_union

    return gpd.GeoDataFrame(
        geometry=[geometry],
        crs="EPSG:4326",
    )


def search_bbox_wkt(
    aoi: gpd.GeoDataFrame,
) -> str:
    """
    Create a simple rectangular search geometry.

    This keeps the CDSE request URL short.
    Exact AOI overlap is checked locally after search.
    """

    west, south, east, north = map(
        float,
        aoi.total_bounds,
    )

    return (
        f"POLYGON(("
        f"{west} {south},"
        f"{east} {south},"
        f"{east} {north},"
        f"{west} {north},"
        f"{west} {south}"
        f"))"
    )


# =========================================================
# SNAP GPT
# =========================================================

def resolve_gpt_path() -> Path:
    load_dotenv()

    configured = os.getenv(
        "SNAP_GPT"
    )

    if configured:
        path = Path(configured)

        if path.exists():
            return path

    for path in COMMON_SNAP_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "SNAP GPT was not found.\n\n"
        "Add this to .env:\n"
        r"SNAP_GPT=C:\Program Files\esa-snap\bin\gpt.exe"
    )


def create_snap_graph(
    graph_file: Path,
) -> None:
    graph_xml = """<?xml version="1.0" encoding="UTF-8"?>
<graph id="S1BiomassGRD">
  <version>1.0</version>

  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${input}</file>
    </parameters>
  </node>

  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
      <reIntroduceThermalNoise>false</reIntroduceThermalNoise>
    </parameters>
  </node>

  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters>
      <sourceBands/>
      <auxFile>Latest Auxiliary File</auxFile>
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <createGammaBand>false</createGammaBand>
      <createBetaBand>false</createBetaBand>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputBetaBand>false</outputBetaBand>
    </parameters>
  </node>

  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
      <filter>Refined Lee</filter>
      <filterSizeX>7</filterSizeX>
      <filterSizeY>7</filterSizeY>
      <dampingFactor>2</dampingFactor>
      <estimateENL>true</estimateENL>
      <enl>1.0</enl>
      <numLooksStr>1</numLooksStr>
      <windowSize>7x7</windowSize>
      <targetWindowSizeStr>3x3</targetWindowSizeStr>
      <sigmaStr>0.9</sigmaStr>
      <anSize>50</anSize>
    </parameters>
  </node>

  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
      <demName>Copernicus 30m Global DEM</demName>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>${resolution}</pixelSpacingInMeter>
      <mapProjection>${projection}</mapProjection>
      <nodataValueAtSea>true</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>

  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
    </parameters>
  </node>

  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="LinearToFromdB"/>
    </sources>
    <parameters>
      <file>${output}</file>
      <formatName>GeoTIFF</formatName>
      <deleteOutputOnFailure>true</deleteOutputOnFailure>
      <writeEntireTileRows>true</writeEntireTileRows>
    </parameters>
  </node>
</graph>
"""

    graph_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_file.write_text(
        graph_xml,
        encoding="utf-8",
    )


# =========================================================
# CDSE AUTHENTICATION
# =========================================================

def get_access_token() -> str:
    load_dotenv()

    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "CDSE credentials were not found.\n\n"
            "Create D:\\biomass_carbon_platform\\.env with:\n"
            "CDSE_USERNAME=your_email\n"
            "CDSE_PASSWORD=your_password"
        )

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=90,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "CDSE token request failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:1500]}"
        )

    return response.json()["access_token"]


# =========================================================
# SENTINEL-1 SEARCH
# =========================================================

def parse_attributes(
    attributes,
) -> dict:
    parsed = {}

    if not isinstance(attributes, list):
        return parsed

    for item in attributes:
        name = item.get("Name")

        if not name:
            continue

        parsed[name] = item.get(
            "Value"
        )

    return parsed


def search_sentinel1(
    aoi: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    orbit_direction: str,
    top: int = 200,
) -> pd.DataFrame:
    geometry_wkt = search_bbox_wkt(aoi)

    filters = [
        "Collection/Name eq 'SENTINEL-1'",
        "contains(Name,'IW_GRDH_1S')",
        (
            "OData.CSC.Intersects("
            f"area=geography'SRID=4326;{geometry_wkt}')"
        ),
        (
            "ContentDate/Start ge "
            f"{start_date}T00:00:00.000Z"
        ),
        (
            "ContentDate/Start le "
            f"{end_date}T23:59:59.999Z"
        ),
    ]

    params = {
        "$filter": " and ".join(filters),
        "$orderby": "ContentDate/Start asc",
        "$top": str(top),
        "$expand": "Attributes",
    }

    response = requests.get(
        CATALOGUE_URL,
        params=params,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Sentinel-1 catalogue search failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )

    products = response.json().get(
        "value",
        [],
    )

    if not products:
        raise RuntimeError(
            "No Sentinel-1 IW GRD products were found."
        )

    try:
        aoi_geometry = aoi.geometry.union_all()
    except AttributeError:
        aoi_geometry = aoi.unary_union

    aoi_area = (
        gpd.GeoSeries(
            [aoi_geometry],
            crs="EPSG:4326",
        )
        .to_crs("EPSG:6933")
        .area.iloc[0]
    )

    rows = []

    for product in products:
        attributes = parse_attributes(
            product.get("Attributes")
        )

        footprint_json = product.get(
            "GeoFootprint"
        )

        overlap_percent = 0.0

        if footprint_json:
            footprint = shape(
                footprint_json
            )

            intersection = (
                footprint.intersection(
                    aoi_geometry
                )
            )

            if not intersection.is_empty:
                overlap_area = (
                    gpd.GeoSeries(
                        [intersection],
                        crs="EPSG:4326",
                    )
                    .to_crs("EPSG:6933")
                    .area.iloc[0]
                )

                overlap_percent = (
                    100.0
                    *
                    overlap_area
                    /
                    aoi_area
                    if aoi_area > 0
                    else 0.0
                )

        name = product["Name"]

        name_upper = name.upper()

        detected_direction = ""

        for candidate in [
            "ASCENDING",
            "DESCENDING",
        ]:
            attribute_value = str(
                attributes.get(
                    "orbitDirection",
                    ""
                )
            ).upper()

            if candidate in attribute_value:
                detected_direction = candidate
                break

        polarisation = str(
            attributes.get(
                "polarisationChannels",
                ""
            )
        ).upper()

        rows.append(
            {
                "product_id":
                    product["Id"],

                "name":
                    name,

                "datetime":
                    product.get(
                        "ContentDate",
                        {},
                    ).get(
                        "Start",
                        "",
                    ),

                "orbit_direction":
                    detected_direction,

                "polarisation":
                    polarisation,

                "relative_orbit":
                    attributes.get(
                        "relativeOrbitNumber",
                        np.nan,
                    ),

                "overlap_percent":
                    overlap_percent,

                "online":
                    product.get(
                        "Online",
                        False,
                    ),

                "content_length":
                    product.get(
                        "ContentLength",
                        0,
                    ),

                "has_vv":
                    "VV" in name_upper
                    or
                    "VV" in polarisation,

                "has_vh":
                    "VH" in name_upper
                    or
                    "VH" in polarisation,
            }
        )

    dataframe = pd.DataFrame(
        rows
    )

    dataframe = dataframe[
        dataframe["overlap_percent"]
        >
        0
    ].copy()

    dataframe = dataframe[
        dataframe["has_vv"]
        &
        dataframe["has_vh"]
    ].copy()

    if orbit_direction != "ANY":
        matching = dataframe[
            dataframe[
                "orbit_direction"
            ]
            ==
            orbit_direction
        ]

        if not matching.empty:
            dataframe = matching

    if dataframe.empty:
        raise RuntimeError(
            "Sentinel-1 products were found, but none "
            "matched VV+VH and orbit filters."
        )

    return dataframe.sort_values(
        by=[
            "overlap_percent",
            "datetime",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def select_products(
    products: pd.DataFrame,
    maximum: int,
) -> pd.DataFrame:
    """
    Prefer one relative orbit and distribute scenes through time.
    """

    working = products.copy()
    
    # Prefer the original SAFE product when CDSE returns both
# normal SAFE and COG versions of the same acquisition.
    working["is_cog"] = (
        working["name"]
        .astype(str)
        .str.upper()
        .str.contains("_COG")
    )

    working["_acquisition"] = pd.to_datetime(
        working["datetime"],
        utc=True,
        errors="coerce",
    )

    working = (
        working
        .sort_values(
            ["_acquisition", "is_cog", "overlap_percent"],
            ascending=[True, True, False],
        )
        .drop_duplicates(
            subset=["_acquisition"],
            keep="first",
        )
        .drop(
            columns=["is_cog", "_acquisition"],
        )
        .reset_index(drop=True)
    )

    orbit_counts = (
        working[
            "relative_orbit"
        ]
        .dropna()
        .value_counts()
    )

    if not orbit_counts.empty:
        preferred_orbit = (
            orbit_counts.index[0]
        )

        same_orbit = working[
            working[
                "relative_orbit"
            ]
            ==
            preferred_orbit
        ]

        if len(same_orbit) >= min(
            2,
            maximum,
        ):
            working = same_orbit.copy()

    working = working.sort_values(
        "datetime"
    ).reset_index(
        drop=True
    )

    if len(working) <= maximum:
        return working

    positions = np.linspace(
        0,
        len(working) - 1,
        maximum,
    )

    indices = sorted(
        set(
            int(round(value))
            for value in positions
        )
    )

    return working.iloc[
        indices
    ].reset_index(
        drop=True
    )


# =========================================================
# DOWNLOAD
# =========================================================

def download_product(
    product_id: str,
    product_name: str,
    token: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zip_file = (
        output_dir
        /
        f"{product_name}.zip"
    )

    if (
        zip_file.exists()
        and
        zip_file.stat().st_size > 1_000_000
    ):
        print(
            f"Already downloaded:\n{zip_file}"
        )
        return zip_file

    url = DOWNLOAD_URL.format(
        product_id=product_id
    )

    with requests.get(
        url,
        headers={
            "Authorization":
                f"Bearer {token}"
        },
        stream=True,
        timeout=(60, 7200),
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                "Sentinel-1 download failed.\n"
                f"Product: {product_name}\n"
                f"HTTP {response.status_code}\n"
                f"{response.text[:1000]}"
            )

        temporary = zip_file.with_suffix(
            ".zip.part"
        )

        with open(
            temporary,
            "wb",
        ) as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

            file.flush()
            os.fsync(file.fileno())

    # Rename only after the response and output handles are closed. Windows
    # security/indexing tools can retain a completed partial file briefly.
    for attempt in range(10):
        try:
            os.replace(temporary, zip_file)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5 * (attempt + 1))

    return zip_file


# =========================================================
# COMMON GRID
# =========================================================

def create_reference_grid(
    aoi: gpd.GeoDataFrame,
    resolution: float,
) -> dict:
    projected_crs = (
        aoi.estimate_utm_crs()
    )

    if projected_crs is None:
        projected_crs = "EPSG:6933"

    projected_aoi = aoi.to_crs(
        projected_crs
    )

    west, south, east, north = (
        projected_aoi.total_bounds
    )

    west = (
        math.floor(west / resolution)
        *
        resolution
    )

    south = (
        math.floor(south / resolution)
        *
        resolution
    )

    east = (
        math.ceil(east / resolution)
        *
        resolution
    )

    north = (
        math.ceil(north / resolution)
        *
        resolution
    )

    return {
        "driver": "GTiff",
        "width": int(
            round(
                (east - west)
                /
                resolution
            )
        ),
        "height": int(
            round(
                (north - south)
                /
                resolution
            )
        ),
        "count": 1,
        "dtype": "float32",
        "crs": projected_crs,
        "transform": from_origin(
            west,
            north,
            resolution,
            resolution,
        ),
        "nodata": np.nan,
    }


def epsg_code(
    crs,
) -> str:
    authority = crs.to_authority()

    if authority:
        return (
            f"{authority[0]}:{authority[1]}"
        )

    return crs.to_wkt()


# =========================================================
# SNAP PROCESSING
# =========================================================

def validate_processed_raster(
    raster_file: Path,
) -> tuple[bool, str]:
    """
    Confirm that a SNAP GeoTIFF can be opened and read.

    A failed SNAP job can leave a large but incomplete TIFF.
    File size alone is therefore not a valid completion check.
    """
    if not raster_file.exists():
        return False, "file does not exist"

    if raster_file.stat().st_size <= 100_000:
        return False, "file is too small"

    try:
        with rasterio.open(raster_file) as dataset:
            if dataset.count < 2:
                return False, "fewer than two raster bands"

            if dataset.width <= 0 or dataset.height <= 0:
                return False, "invalid raster dimensions"

            # Read small windows from the top, centre and bottom.
            # This catches truncated or damaged TIFF strips without
            # loading the complete raster into memory.
            sample_size = 32
            positions = [
                (0, 0),
                (
                    max(0, dataset.width // 2 - sample_size // 2),
                    max(0, dataset.height // 2 - sample_size // 2),
                ),
                (
                    max(0, dataset.width - sample_size),
                    max(0, dataset.height - sample_size),
                ),
            ]

            for band_index in range(1, dataset.count + 1):
                for column_offset, row_offset in positions:
                    width = min(sample_size, dataset.width - column_offset)
                    height = min(sample_size, dataset.height - row_offset)

                    if width <= 0 or height <= 0:
                        continue

                    dataset.read(
                        band_index,
                        window=rasterio.windows.Window(
                            column_offset,
                            row_offset,
                            width,
                            height,
                        ),
                    )

        return True, "ok"

    except Exception as error:
        return False, str(error)


def remove_corrupt_output(
    output_file: Path,
) -> None:
    """Remove an incomplete TIFF and leave its log for diagnosis."""
    try:
        output_file.unlink(missing_ok=True)
    except OSError:
        pass


def run_snap(
    gpt_path: Path,
    graph_file: Path,
    input_zip: Path,
    output_file: Path,
    projection: str,
    resolution: float,
) -> None:
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_file.exists():
        valid, reason = validate_processed_raster(
            output_file
        )

        if valid:
            print(
                f"Already processed and validated:\n{output_file}"
            )
            return

        print(
            "Existing SNAP output is incomplete or corrupt. "
            "It will be deleted and processed again."
        )
        print(f"Validation reason: {reason}")
        remove_corrupt_output(output_file)

    command = [
        str(gpt_path),
        str(graph_file),
        f"-Pinput={input_zip}",
        f"-Poutput={output_file}",
        f"-Pprojection={projection}",
        f"-Presolution={resolution}",
        "-c", "2G",
        "-q", "1",
    ]

    print(
        "\nRunning SNAP GPT:\n"
        +
        " ".join(
            f'"{part}"'
            if " " in part
            else part
            for part in command
        )
    )

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    log_file = output_file.with_suffix(
        ".log"
    )

    log_file.write_text(
        process.stdout
        +
        "\n\nSTDERR\n"
        +
        process.stderr,
        encoding="utf-8",
    )

    if process.returncode != 0:
        raise RuntimeError(
            "SNAP GPT processing failed.\n"
            f"Input: {input_zip}\n"
            f"Log: {log_file}\n"
            f"{process.stderr[-2000:]}"
        )

    if not output_file.exists():
        raise RuntimeError(
            "SNAP completed without creating the expected output:\n"
            f"{output_file}"
        )

    valid, reason = validate_processed_raster(
        output_file
    )

    if not valid:
        remove_corrupt_output(output_file)
        raise RuntimeError(
            "SNAP created an unreadable or incomplete GeoTIFF.\n"
            f"Output: {output_file}\n"
            f"Validation reason: {reason}"
        )


# =========================================================
# READ AND ALIGN SNAP OUTPUTS
# =========================================================

def identify_vv_vh_bands(
    dataset: rasterio.DatasetReader,
) -> tuple[int, int]:
    descriptions = [
        (
            description
            or
            ""
        ).upper()
        for description in dataset.descriptions
    ]

    vv_index = None
    vh_index = None

    for index, description in enumerate(
        descriptions,
        start=1,
    ):
        if "VV" in description:
            vv_index = index

        if "VH" in description:
            vh_index = index

    if vv_index is None or vh_index is None:
        if dataset.count >= 2:
            vv_index = 1
            vh_index = 2
        else:
            raise RuntimeError(
                "Processed Sentinel-1 GeoTIFF "
                "does not contain both VV and VH."
            )

    return vv_index, vh_index


def align_band(
    source_file: Path,
    band_index: int,
    reference_profile: dict,
) -> np.ndarray:
    destination = np.full(
        (
            reference_profile["height"],
            reference_profile["width"],
        ),
        np.nan,
        dtype="float32",
    )

    with rasterio.open(
        source_file
    ) as source:
        source_array = (
            source.read(
                band_index
            )
            .astype("float32")
        )

        if source.nodata is not None:
            source_array[
                source_array
                ==
                source.nodata
            ] = np.nan

        reproject(
            source=source_array,
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=np.nan,
            dst_transform=
                reference_profile[
                    "transform"
                ],
            dst_crs=
                reference_profile[
                    "crs"
                ],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    return destination


def load_processed_scene(
    processed_file: Path,
    reference_profile: dict,
) -> dict[str, np.ndarray]:
    with rasterio.open(
        processed_file
    ) as dataset:
        vv_index, vh_index = (
            identify_vv_vh_bands(
                dataset
            )
        )

    vv = align_band(
        processed_file,
        vv_index,
        reference_profile,
    )

    vh = align_band(
        processed_file,
        vh_index,
        reference_profile,
    )

    return {
        "VV": vv,
        "VH": vh,
    }


# =========================================================
# RADAR FEATURES
# =========================================================

def db_to_linear(
    db: np.ndarray,
) -> np.ndarray:
    result = np.full(
        db.shape,
        np.nan,
        dtype="float32",
    )

    valid = np.isfinite(db)

    result[valid] = np.power(
        10.0,
        db[valid] / 10.0,
    )

    return result


def safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    result = np.full(
        numerator.shape,
        np.nan,
        dtype="float32",
    )

    valid = (
        np.isfinite(numerator)
        &
        np.isfinite(denominator)
        &
        (
            np.abs(denominator)
            >
            1e-12
        )
    )

    result[valid] = (
        numerator[valid]
        /
        denominator[valid]
    )

    return result


def local_std(
    data: np.ndarray,
    window: int,
) -> np.ndarray:
    valid = np.isfinite(data)

    filled = np.where(
        valid,
        data,
        0.0,
    ).astype("float32")

    weights = valid.astype(
        "float32"
    )

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
            filled * filled,
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
        mean * mean,
        0.0,
    )

    result = np.sqrt(
        variance
    ).astype("float32")

    result[~valid] = np.nan

    return result


def build_composite(
    scenes: list[
        dict[str, np.ndarray]
    ],
    texture_window: int,
) -> dict[str, np.ndarray]:
    vv_stack = np.stack(
        [
            scene["VV"]
            for scene in scenes
        ],
        axis=0,
    )

    vh_stack = np.stack(
        [
            scene["VH"]
            for scene in scenes
        ],
        axis=0,
    )

    with np.errstate(
        invalid="ignore"
    ):
        vv_median = np.nanmedian(
            vv_stack,
            axis=0,
        ).astype("float32")

        vh_median = np.nanmedian(
            vh_stack,
            axis=0,
        ).astype("float32")

        vv_std_time = np.nanstd(
            vv_stack,
            axis=0,
        ).astype("float32")

        vh_std_time = np.nanstd(
            vh_stack,
            axis=0,
        ).astype("float32")

    valid_count = np.sum(
        np.isfinite(vv_stack)
        &
        np.isfinite(vh_stack),
        axis=0,
    ).astype("uint16")

    vv_linear = db_to_linear(
        vv_median
    )

    vh_linear = db_to_linear(
        vh_median
    )

    vv_minus_vh = (
        vv_median
        -
        vh_median
    ).astype("float32")

    vv_vh_ratio = safe_ratio(
        vv_linear,
        vh_linear,
    )

    rvi = safe_ratio(
        4.0 * vh_linear,
        vv_linear
        +
        vh_linear,
    )

    vv_texture = local_std(
        vv_median,
        texture_window,
    )

    vh_texture = local_std(
        vh_median,
        texture_window,
    )

    return {
        "VV_median":
            vv_median,

        "VH_median":
            vh_median,

        "VV_minus_VH":
            vv_minus_vh,

        "VV_VH_ratio":
            vv_vh_ratio,

        "RVI":
            rvi,

        "VV_texture":
            vv_texture,

        "VH_texture":
            vh_texture,

        "VV_temporal_std":
            vv_std_time,

        "VH_temporal_std":
            vh_std_time,

        "valid_count":
            valid_count,
    }


# =========================================================
# SAVE RASTERS
# =========================================================

def clean_profile(
    profile: dict,
) -> dict:
    output = profile.copy()

    output.pop(
        "blockxsize",
        None,
    )

    output.pop(
        "blockysize",
        None,
    )

    output.pop(
        "interleave",
        None,
    )

    return output


def save_float(
    path: Path,
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

    with rasterio.open(
        path,
        "w",
        **output_profile,
    ) as destination:
        destination.write(
            array.astype("float32"),
            1,
        )

        destination.set_band_description(
            1,
            description,
        )


def save_uint16(
    path: Path,
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
        dtype="uint16",
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(
        path,
        "w",
        **output_profile,
    ) as destination:
        destination.write(
            array.astype("uint16"),
            1,
        )

        destination.set_band_description(
            1,
            description,
        )


# =========================================================
# MAIN PIPELINE
# =========================================================

def process_sentinel1(
    run_dir: Path,
    start_date: str,
    end_date: str,
    max_products: int,
    orbit_direction: str,
    resolution: float,
    texture_window: int,
) -> dict:
    aoi_file = (
        run_dir
        / "input"
        / "aoi_wgs84.geojson"
    )

    raw_dir = (
        run_dir
        / "raw"
        / "sentinel1"
    )

    snap_dir = (
        run_dir
        / "processed"
        / "sentinel1_snap"
    )

    output_dir = (
        run_dir
        / "processed"
        / "sentinel1"
    )

    reports_dir = (
        run_dir
        / "reports"
        / "sentinel1"
    )

    graph_file = (
        run_dir
        / "logs"
        / "sentinel1_biomass_graph.xml"
    )

    for directory in [
        raw_dir,
        snap_dir,
        output_dir,
        reports_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "\n[1/9] Reading AOI..."
    )

    aoi = read_aoi(
        aoi_file
    )

    print(
        "\n[2/9] Finding SNAP GPT..."
    )

    gpt_path = resolve_gpt_path()

    print(
        f"GPT:\n{gpt_path}"
    )

    create_snap_graph(
        graph_file
    )

    print(
        "\n[3/9] Searching Sentinel-1 IW GRD..."
    )

    products = search_sentinel1(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        orbit_direction=
            orbit_direction,
    )

    products.to_csv(
        reports_dir
        / "sentinel1_search_results.csv",
        index=False,
    )

    selected = select_products(
        products,
        max_products,
    )

    selected.to_csv(
        reports_dir
        / "sentinel1_selected_products.csv",
        index=False,
    )

    print(
        f"Products satisfying filters: "
        f"{len(products)}"
    )

    print(
        f"Products selected: "
        f"{len(selected)}"
    )

    print(
        "\n[4/9] Authenticating CDSE..."
    )

    token = get_access_token()

    print(
        "\n[5/9] Downloading products..."
    )

    zip_files = []

    for index, row in selected.iterrows():
        print(
            f"\nDownloading "
            f"{index + 1}/{len(selected)}:\n"
            f"{row['name']}"
        )

        zip_files.append(
            download_product(
                product_id=row[
                    "product_id"
                ],
                product_name=row[
                    "name"
                ],
                token=token,
                output_dir=raw_dir,
            )
        )

    print(
        "\n[6/9] Creating common grid..."
    )

    reference_profile = (
        create_reference_grid(
            aoi,
            resolution,
        )
    )

    projection = epsg_code(
        reference_profile[
            "crs"
        ]
    )

    print(
        f"Projection : {projection}"
    )

    print(
        f"Resolution : {resolution} m"
    )

    print(
        "\n[7/9] Running SNAP GPT..."
    )

    processed_files = []

    for index, (
        zip_file,
        (_, row),
    ) in enumerate(
        zip(
            zip_files,
            selected.iterrows(),
        ),
        start=1,
    ):
        output_file = (
            snap_dir
            /
            f"{row['name']}_ARD.tif"
        )

        print(
            f"\nScene {index}/{len(zip_files)}"
        )

        try:
            run_snap(
                gpt_path=gpt_path,
                graph_file=graph_file,
                input_zip=zip_file,
                output_file=output_file,
                projection=projection,
                resolution=resolution,
            )

            valid, reason = validate_processed_raster(
                output_file
            )

            if valid:
                processed_files.append(
                    output_file
                )
            else:
                remove_corrupt_output(output_file)
                raise RuntimeError(
                    "Processed GeoTIFF failed validation: "
                    f"{reason}"
                )

        except Exception as error:
            print(
                "\nWARNING: This Sentinel-1 scene failed "
                "and will be skipped."
            )
            print(
                f"Scene : {row['name']}"
            )
            print(
                f"Reason: {error}"
            )
            print(
                "Continuing with the remaining successful scenes..."
            )

    if len(processed_files) < 2:
        raise RuntimeError(
            "Fewer than two Sentinel-1 scenes were processed "
            "successfully. At least two scenes are required "
            "to create the seasonal composite."
        )

    print(
        f"\nSuccessful Sentinel-1 scenes: "
        f"{len(processed_files)}/{len(zip_files)}"
    )

    print(
        "\n[8/9] Building seasonal radar composite..."
    )

    scenes = []
    readable_processed_files = []

    for processed_file in processed_files:
        try:
            scene = load_processed_scene(
                processed_file,
                reference_profile,
            )
            scenes.append(scene)
            readable_processed_files.append(processed_file)

        except Exception as error:
            print(
                "\nWARNING: A processed Sentinel-1 TIFF could not "
                "be read and will be excluded from the composite."
            )
            print(f"File   : {processed_file}")
            print(f"Reason : {error}")
            remove_corrupt_output(processed_file)

    if len(scenes) < 2:
        raise RuntimeError(
            "Fewer than two readable Sentinel-1 scenes remain "
            "after validating the SNAP outputs."
        )

    processed_files = readable_processed_files

    print(
        f"Readable scenes used in composite: {len(scenes)}"
    )

    composite = build_composite(
        scenes,
        texture_window,
    )

    print(
        "\n[9/9] Saving Sentinel-1 biomass features..."
    )

    output_files = {}

    float_products = {
        "VV_median":
            "S1_VV_median_dB.tif",

        "VH_median":
            "S1_VH_median_dB.tif",

        "VV_minus_VH":
            "S1_VV_minus_VH_dB.tif",

        "VV_VH_ratio":
            "S1_VV_VH_linear_ratio.tif",

        "RVI":
            "S1_RVI.tif",

        "VV_texture":
            "S1_VV_texture_std.tif",

        "VH_texture":
            "S1_VH_texture_std.tif",

        "VV_temporal_std":
            "S1_VV_temporal_std.tif",

        "VH_temporal_std":
            "S1_VH_temporal_std.tif",
    }

    for key, filename in (
        float_products.items()
    ):
        path = (
            output_dir
            / filename
        )

        save_float(
            path,
            composite[key],
            reference_profile,
            key,
        )

        output_files[
            key
        ] = path

    valid_count_file = (
        output_dir
        / "S1_valid_observation_count.tif"
    )

    save_uint16(
        valid_count_file,
        composite[
            "valid_count"
        ],
        reference_profile,
        "S1_valid_observation_count",
    )

    output_files[
        "valid_count"
    ] = valid_count_file

    summary = {
        "start_date":
            start_date,

        "end_date":
            end_date,

        "selected_products":
            len(selected),

        "successful_products":
            len(processed_files),

        "orbit_direction":
            orbit_direction,

        "resolution_m":
            resolution,

        "texture_window":
            texture_window,

        "output_crs":
            str(
                reference_profile[
                    "crs"
                ]
            ),
    }

    with open(
        reports_dir
        / "sentinel1_processing_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    return {
        "selected_products":
            len(selected),

        "output_dir":
            output_dir,

        "reports_dir":
            reports_dir,

        "output_files":
            output_files,
    }


# =========================================================
# COMMAND LINE
# =========================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Search, download and process seasonal "
            "Sentinel-1 biomass radar predictors."
        )
    )

    parser.add_argument(
        "--run-dir",
        default=None,
    )

    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
    )

    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
    )

    parser.add_argument(
        "--max-products",
        type=int,
        default=DEFAULT_MAX_PRODUCTS,
    )

    parser.add_argument(
        "--orbit-direction",
        choices=[
            "ASCENDING",
            "DESCENDING",
            "ANY",
        ],
        default=DEFAULT_ORBIT_DIRECTION,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
    )

    parser.add_argument(
        "--texture-window",
        type=int,
        default=DEFAULT_TEXTURE_WINDOW,
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
        "VERSION 3 - SENTINEL-1 BIOMASS FEATURES"
    )

    print(
        "=" * 72
    )

    args = parse_arguments()

    run_dir = resolve_run_directory(
        args.run_dir
    )

    print(
        f"\nUsing run:\n{run_dir}"
    )

    result = process_sentinel1(
        run_dir=run_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_products=args.max_products,
        orbit_direction=
            args.orbit_direction,
        resolution=args.resolution,
        texture_window=
            args.texture_window,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "VERSION 3 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nProducts used:\n"
        f"{result['selected_products']}"
    )

    print(
        f"\nSentinel-1 outputs:\n"
        f"{result['output_dir']}"
    )

    print(
        f"\nReports:\n"
        f"{result['reports_dir']}"
    )

    print(
        "\nReady for Version 4: "
        "DEM and terrain features."
    )


if __name__ == "__main__":
    main()
