
from __future__ import annotations

from pathlib import Path
from datetime import date
import argparse
import json
import math
import os
import shutil
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import hashlib

from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject
from shapely.geometry import shape

from config import RUNS_DIR


# =========================================================
# SETTINGS
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

DEFAULT_START_DATE = "2024-04-01"
DEFAULT_END_DATE = "2024-09-30"
DEFAULT_MAX_CLOUD = 60.0
DEFAULT_MAX_PRODUCTS = 4
DEFAULT_RESOLUTION = 20.0

BAND_KEYS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "SCL",
]

BAD_SCL_CLASSES = {
    0,   # no data
    1,   # saturated / defective
    3,   # cloud shadow
    7,   # unclassified
    8,   # medium cloud
    9,   # high cloud
    10,  # cirrus
    11,  # snow / ice
}


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
# CDSE AUTHENTICATION
# =========================================================

def get_access_token() -> str:
    load_dotenv()

    username = os.getenv("CDSE_USERNAME")
    password = os.getenv("CDSE_PASSWORD")

    if not username or not password:
        raise RuntimeError(
            "Copernicus credentials were not found.\n\n"
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
        timeout=60,
    )

    if response.status_code != 200:
        try:
            details = response.json()
        except Exception:
            details = response.text

        raise RuntimeError(
            "CDSE access-token creation failed.\n"
            f"HTTP {response.status_code}\n"
            f"{details}"
        )

    return response.json()["access_token"]


# =========================================================
# PRODUCT SEARCH
# =========================================================

def parse_attributes(
    attributes,
) -> dict:
    result = {}

    if not isinstance(attributes, list):
        return result

    for item in attributes:
        name = item.get("Name")
        value = (
            item.get("Value")
            if "Value" in item
            else item.get("ValueType")
        )

        if name:
            result[name] = value

    return result


def search_sentinel2(
    aoi: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    max_cloud: float,
    top: int = 200,
) -> pd.DataFrame:
    geometry_wkt = search_bbox_wkt(aoi)

    filters = [
        "Collection/Name eq 'SENTINEL-2'",
        "contains(Name,'MSIL2A')",
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
            "Sentinel-2 catalogue search failed.\n"
            f"HTTP {response.status_code}\n"
            f"{response.text[:2000]}"
        )

    products = response.json().get(
        "value",
        [],
    )

    if not products:
        raise RuntimeError(
            "No Sentinel-2 L2A products were found "
            "for the AOI and date range."
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

        cloud = attributes.get(
            "cloudCover",
            attributes.get(
                "cloudCoverPercentage",
                np.nan,
            ),
        )

        try:
            cloud = float(cloud)
        except (TypeError, ValueError):
            cloud = np.nan

        footprint_json = product.get(
            "GeoFootprint"
        )

        overlap_km2 = 0.0
        overlap_percent = 0.0

        if footprint_json:
            footprint = shape(
                footprint_json
            )

            intersection = footprint.intersection(
                aoi_geometry
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

                overlap_km2 = (
                    overlap_area
                    /
                    1_000_000
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

        tile_id = ""

        for part in name.split("_"):
            if (
                len(part) == 6
                and
                part.startswith("T")
            ):
                tile_id = part
                break

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

                "cloud_cover":
                    cloud,

                "tile_id":
                    tile_id,

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

                "overlap_km2":
                    overlap_km2,

                "overlap_percent":
                    overlap_percent,

                "s3_path":
                    product.get(
                        "S3Path",
                        "",
                    ),
            }
        )

    dataframe = pd.DataFrame(rows)

    dataframe = dataframe[
        dataframe["overlap_percent"]
        >
        0
    ].copy()

    if dataframe.empty:
        raise RuntimeError(
            "Products were returned, but none overlap the AOI."
        )

    # Missing cloud values are retained but ranked last.
    dataframe["cloud_rank"] = (
        dataframe["cloud_cover"]
        .fillna(999.0)
    )

    dataframe = dataframe[
        (
            dataframe["cloud_cover"].isna()
        )
        |
        (
            dataframe["cloud_cover"]
            <=
            max_cloud
        )
    ].copy()

    if dataframe.empty:
        raise RuntimeError(
            "Sentinel-2 products exist, but none satisfy "
            f"the cloud threshold of {max_cloud:.1f}%."
        )

    return dataframe.sort_values(
        by=[
            "cloud_rank",
            "overlap_percent",
            "datetime",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def select_products(
    products: pd.DataFrame,
    max_products: int,
) -> pd.DataFrame:
    """
    Select low-cloud products while keeping temporal and tile diversity.
    """

    selected_indices = []
    used_tiles = set()

    # First choose the best product for each MGRS tile.
    for index, row in products.iterrows():
        tile = row["tile_id"]

        if tile and tile not in used_tiles:
            selected_indices.append(index)
            used_tiles.add(tile)

        if len(selected_indices) >= max_products:
            break

    # Fill remaining positions with next-best products.
    if len(selected_indices) < max_products:
        for index in products.index:
            if index not in selected_indices:
                selected_indices.append(index)

            if len(selected_indices) >= max_products:
                break

    return products.loc[
        selected_indices
    ].copy().reset_index(
        drop=True
    )


# =========================================================
# DOWNLOAD AND EXTRACTION
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

    headers = {
        "Authorization":
            f"Bearer {token}"
    }

    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(60, 3600),
    ) as response:
        if response.status_code != 200:
            raise RuntimeError(
                "Sentinel-2 download failed.\n"
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


def safe_extract_zip(
    zip_file: Path,
    output_dir: Path,
) -> Path:
    """
    Extract Sentinel-2 SAFE archive to a short Windows path.

    A short hash prevents Windows path-length errors.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_id = hashlib.md5(
        str(zip_file).encode("utf-8")
    ).hexdigest()[:10]

    target = (
        output_dir
        /
        scene_id
    )

    existing_safe = list(
        target.rglob("*.SAFE")
    )

    if existing_safe:
        print(
            f"Using existing SAFE:\n"
            f"{existing_safe[0]}"
        )

        return existing_safe[0]

    if target.exists():
        shutil.rmtree(
            target,
            ignore_errors=True,
        )

    target.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nExtracting:\n"
        f"{zip_file}\n"
        f"To short path:\n"
        f"{target}"
    )

    try:
        with zipfile.ZipFile(
            zip_file,
            "r",
        ) as archive:

            for member in archive.infolist():

                member_path = (
                    target
                    /
                    member.filename
                ).resolve()

                if not str(
                    member_path
                ).startswith(
                    str(
                        target.resolve()
                    )
                ):
                    raise RuntimeError(
                        "Unsafe path found inside ZIP archive."
                    )

            archive.extractall(
                target
            )

    except Exception:
        shutil.rmtree(
            target,
            ignore_errors=True,
        )

        raise

    safe_folders = list(
        target.rglob("*.SAFE")
    )

    if not safe_folders:
        raise RuntimeError(
            "Extraction finished, but no SAFE folder "
            f"was found inside:\n{target}"
        )

    print(
        f"\nSAFE folder:\n"
        f"{safe_folders[0]}"
    )

    return safe_folders[0]


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

    west = math.floor(
        west / resolution
    ) * resolution

    south = math.floor(
        south / resolution
    ) * resolution

    east = math.ceil(
        east / resolution
    ) * resolution

    north = math.ceil(
        north / resolution
    ) * resolution

    width = int(
        round(
            (east - west)
            /
            resolution
        )
    )

    height = int(
        round(
            (north - south)
            /
            resolution
        )
    )

    transform = from_origin(
        west,
        north,
        resolution,
        resolution,
    )

    return {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 1,
        "dtype": "float32",
        "crs": projected_crs,
        "transform": transform,
        "nodata": np.nan,
    }


def reproject_band(
    source_file: Path,
    reference_profile: dict,
    categorical: bool = False,
) -> np.ndarray:
    shape_out = (
        reference_profile["height"],
        reference_profile["width"],
    )

    destination = np.full(
        shape_out,
        np.nan,
        dtype="float32",
    )

    with rasterio.open(
        source_file
    ) as source:
        source_array = (
            source.read(1)
            .astype("float32")
        )

        source_nodata = (
            source.nodata
        )

        if source_nodata is not None:
            source_array[
                source_array
                ==
                source_nodata
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
            resampling=(
                Resampling.nearest
                if categorical
                else Resampling.bilinear
            ),
        )

    return destination


# =========================================================
# CLOUD MASK AND INDICES
# =========================================================

def valid_scl_mask(
    scl: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(scl)

    integer_scl = np.full(
        scl.shape,
        -9999,
        dtype="int16",
    )

    integer_scl[finite] = (
        scl[finite]
        .astype("int16")
    )

    bad = np.isin(
        integer_scl,
        list(BAD_SCL_CLASSES),
    )

    return finite & ~bad


def normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    result = np.full(
        first.shape,
        np.nan,
        dtype="float32",
    )

    denominator = (
        first + second
    )

    valid = (
        np.isfinite(first)
        &
        np.isfinite(second)
        &
        (
            np.abs(denominator)
            >
            1e-8
        )
    )

    result[valid] = (
        first[valid]
        -
        second[valid]
    ) / denominator[valid]

    return result


def calculate_indices(
    bands: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    blue = bands["B02"]
    green = bands["B03"]
    red = bands["B04"]
    re1 = bands["B05"]
    re2 = bands["B06"]
    re3 = bands["B07"]
    nir = bands["B08"]
    narrow_nir = bands["B8A"]
    swir1 = bands["B11"]
    swir2 = bands["B12"]

    ndvi = normalized_difference(
        nir,
        red,
    )

    ndmi = normalized_difference(
        nir,
        swir1,
    )

    nbr = normalized_difference(
        nir,
        swir2,
    )

    gndvi = normalized_difference(
        nir,
        green,
    )

    rededge_ndvi = normalized_difference(
        narrow_nir,
        re1,
    )

    evi = np.full(
        red.shape,
        np.nan,
        dtype="float32",
    )

    evi_denominator = (
        nir
        +
        6.0 * red
        -
        7.5 * blue
        +
        1.0
    )

    evi_valid = (
        np.isfinite(nir)
        &
        np.isfinite(red)
        &
        np.isfinite(blue)
        &
        (
            np.abs(evi_denominator)
            >
            1e-8
        )
    )

    evi[evi_valid] = (
        2.5
        *
        (
            nir[evi_valid]
            -
            red[evi_valid]
        )
        /
        evi_denominator[evi_valid]
    )

    mtci = np.full(
        red.shape,
        np.nan,
        dtype="float32",
    )

    mtci_denominator = (
        re1 - red
    )

    mtci_valid = (
        np.isfinite(re2)
        &
        np.isfinite(re1)
        &
        np.isfinite(red)
        &
        (
            np.abs(mtci_denominator)
            >
            1e-8
        )
    )

    mtci[mtci_valid] = (
        re2[mtci_valid]
        -
        re1[mtci_valid]
    ) / mtci_denominator[mtci_valid]

    return {
        "NDVI": ndvi,
        "EVI": evi,
        "NDMI": ndmi,
        "NBR": nbr,
        "GNDVI": gndvi,
        "RE_NDVI": rededge_ndvi,
        "MTCI": mtci,
        # Retain red-edge 3 for later feature stacks.
        "RE3": re3,
    }

def find_band(
    safe_folder: Path,
    band_key: str,
) -> Path:
    """
    Find a Sentinel-2 JP2 band inside a SAFE folder.

    For biomass processing, 20 m products are preferred
    so all optical bands use one common resolution.
    """

    if not safe_folder.exists():
        raise FileNotFoundError(
            f"SAFE folder not found:\n{safe_folder}"
        )

    candidates = []

    for path in safe_folder.rglob("*.jp2"):
        name = path.name.upper()
        full_path = str(path).upper()

        if band_key == "SCL":
            if "_SCL_" in name:
                candidates.append(path)

        elif f"_{band_key}_" in name:
            candidates.append(path)

    if not candidates:
        example_files = list(
            safe_folder.rglob("*.jp2")
        )[:20]

        examples = "\n".join(
            str(path)
            for path in example_files
        )

        raise FileNotFoundError(
            f"Band {band_key} was not found inside:\n"
            f"{safe_folder}\n\n"
            f"Example JP2 files:\n{examples}"
        )

    def priority(
        path: Path,
    ):
        path_text = str(path).upper()
        name = path.name.upper()

        # Prefer 20 m data.
        if (
            "R20M" in path_text
            or
            "_20M.JP2" in name
        ):
            resolution_priority = 0

        elif (
            "R10M" in path_text
            or
            "_10M.JP2" in name
        ):
            resolution_priority = 1

        elif (
            "R60M" in path_text
            or
            "_60M.JP2" in name
        ):
            resolution_priority = 2

        else:
            resolution_priority = 3

        return (
            resolution_priority,
            len(str(path)),
        )

    candidates.sort(
        key=priority
    )

    selected = candidates[0]

    print(
        f"{band_key}: {selected}"
    )

    return selected
# =========================================================
# SCENE PROCESSING
# =========================================================

def load_scene(
    safe_folder: Path,
    reference_profile: dict,
) -> dict[str, np.ndarray]:
    scene = {}

    for band_key in BAND_KEYS:
        band_file = find_band(
            safe_folder,
            band_key,
        )

        scene[band_key] = reproject_band(
            band_file,
            reference_profile,
            categorical=(
                band_key == "SCL"
            ),
        )

    valid_mask = valid_scl_mask(
        scene["SCL"]
    )

    reflectance_bands = [
        key
        for key in BAND_KEYS
        if key != "SCL"
    ]

    for key in reflectance_bands:
        # Sentinel-2 L2A integer reflectance scaling.
        scene[key] = (
            scene[key]
            /
            10000.0
        ).astype("float32")

        scene[key][
            ~valid_mask
        ] = np.nan

    scene["valid_mask"] = (
        valid_mask
    )

    return scene


def median_composite(
    scenes: list[
        dict[str, np.ndarray]
    ],
) -> dict[str, np.ndarray]:
    composite = {}

    reflectance_keys = [
        key
        for key in BAND_KEYS
        if key != "SCL"
    ]

    for key in reflectance_keys:
        stack = np.stack(
            [
                scene[key]
                for scene in scenes
            ],
            axis=0,
        )

        with np.errstate(
            invalid="ignore"
        ):
            composite[key] = (
                np.nanmedian(
                    stack,
                    axis=0,
                )
                .astype("float32")
            )

    valid_count = np.sum(
        np.stack(
            [
                scene["valid_mask"]
                for scene in scenes
            ],
            axis=0,
        ),
        axis=0,
    ).astype("uint16")

    composite["valid_count"] = (
        valid_count
    )

    composite.update(
        calculate_indices(
            composite
        )
    )

    return composite


# =========================================================
# SAVE PRODUCTS
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

    with rasterio.open(
        output_file,
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


def save_uint16_raster(
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
        dtype="uint16",
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
            array.astype("uint16"),
            1,
        )

        destination.set_band_description(
            1,
            description,
        )


# =========================================================
# MAIN WORKFLOW
# =========================================================

def process_sentinel2(
    run_dir: Path,
    start_date: str,
    end_date: str,
    max_cloud: float,
    max_products: int,
    resolution: float,
) -> dict:
    aoi_file = (
        run_dir
        /
        "input"
        /
        "aoi_wgs84.geojson"
    )

    raw_dir = (
        run_dir
        /
        "raw"
        /
        "sentinel2"
    )

    extraction_dir = Path(
    r"D:\s2_safe"
    ) / run_dir.name

    output_dir = (
        run_dir
        /
        "processed"
        /
        "sentinel2"
    )

    reports_dir = (
        run_dir
        /
        "reports"
        /
        "sentinel2"
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

    print(
        "\n[2/8] Searching Sentinel-2 L2A..."
    )

    products = search_sentinel2(
        aoi=aoi,
        start_date=start_date,
        end_date=end_date,
        max_cloud=max_cloud,
    )

    products.to_csv(
        reports_dir
        /
        "sentinel2_search_results.csv",
        index=False,
    )

    selected = select_products(
        products,
        max_products,
    )

    selected.to_csv(
        reports_dir
        /
        "sentinel2_selected_products.csv",
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
        "\n[3/8] Authenticating CDSE..."
    )

    token = get_access_token()

    print(
        "\n[4/8] Downloading Sentinel-2 products..."
    )

    zip_files = []

    for index, row in selected.iterrows():
        print(
            f"\nDownloading "
            f"{index + 1}/{len(selected)}:\n"
            f"{row['name']}"
        )

        zip_file = download_product(
            product_id=row[
                "product_id"
            ],
            product_name=row[
                "name"
            ],
            token=token,
            output_dir=raw_dir,
        )

        zip_files.append(
            zip_file
        )

    print(
        "\n[5/8] Extracting SAFE products..."
    )

    safe_folders = []

    for zip_file in zip_files:
        safe_folder = safe_extract_zip(
            zip_file,
            extraction_dir,
        )

        safe_folders.append(
            safe_folder
        )

        print(
            f"SAFE:\n{safe_folder}"
        )

    print(
        "\n[6/8] Creating common biomass grid..."
    )

    reference_profile = (
        create_reference_grid(
            aoi,
            resolution,
        )
    )

    print(
        f"CRS        : "
        f"{reference_profile['crs']}"
    )

    print(
        f"Resolution : "
        f"{resolution} m"
    )

    print(
        f"Raster size: "
        f"{reference_profile['width']} × "
        f"{reference_profile['height']}"
    )

    print(
        "\n[7/8] Loading scenes and building median composite..."
    )

    scenes = []

    for index, safe_folder in enumerate(
        safe_folders,
        start=1,
    ):
        print(
            f"Processing scene "
            f"{index}/{len(safe_folders)}"
        )

        scenes.append(
            load_scene(
                safe_folder,
                reference_profile,
            )
        )

    composite = median_composite(
        scenes
    )

    print(
        "\n[8/8] Saving Sentinel-2 biomass features..."
    )

    reflectance_names = {
        "B02": "Blue",
        "B03": "Green",
        "B04": "Red",
        "B05": "RedEdge1",
        "B06": "RedEdge2",
        "B07": "RedEdge3",
        "B08": "NIR",
        "B8A": "NarrowNIR",
        "B11": "SWIR1",
        "B12": "SWIR2",
    }

    output_files = {}

    for key, display_name in (
        reflectance_names.items()
    ):
        output_file = (
            output_dir
            /
            f"S2_{display_name}_median.tif"
        )

        save_float_raster(
            output_file,
            composite[key],
            reference_profile,
            f"Sentinel2_{display_name}_median",
        )

        output_files[
            key
        ] = output_file

    for index_name in [
        "NDVI",
        "EVI",
        "NDMI",
        "NBR",
        "GNDVI",
        "RE_NDVI",
        "MTCI",
    ]:
        output_file = (
            output_dir
            /
            f"S2_{index_name}.tif"
        )

        save_float_raster(
            output_file,
            composite[
                index_name
            ],
            reference_profile,
            f"Sentinel2_{index_name}",
        )

        output_files[
            index_name
        ] = output_file

    valid_count_file = (
        output_dir
        /
        "S2_valid_observation_count.tif"
    )

    save_uint16_raster(
        valid_count_file,
        composite[
            "valid_count"
        ],
        reference_profile,
        "Sentinel2_valid_observation_count",
    )

    output_files[
        "valid_count"
    ] = valid_count_file

    summary = {
        "start_date":
            start_date,

        "end_date":
            end_date,

        "max_cloud":
            max_cloud,

        "selected_products":
            len(selected),

        "resolution_m":
            resolution,

        "output_crs":
            str(
                reference_profile[
                    "crs"
                ]
            ),

        "width":
            reference_profile[
                "width"
            ],

        "height":
            reference_profile[
                "height"
            ],
    }

    with open(
        reports_dir
        /
        "sentinel2_processing_summary.json",
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
            "Search, download and process Sentinel-2 L2A "
            "biomass predictors for a Version 1 AOI."
        )
    )

    parser.add_argument(
        "--run-dir",
        default=None,
        help=(
            "Version 1 run directory. "
            "If omitted, the latest run is used."
        ),
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
        "--max-cloud",
        type=float,
        default=DEFAULT_MAX_CLOUD,
    )

    parser.add_argument(
        "--max-products",
        type=int,
        default=DEFAULT_MAX_PRODUCTS,
    )

    parser.add_argument(
        "--resolution",
        type=float,
        default=DEFAULT_RESOLUTION,
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
        "VERSION 2 - SENTINEL-2 BIOMASS FEATURES"
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

    result = process_sentinel2(
        run_dir=run_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud=args.max_cloud,
        max_products=args.max_products,
        resolution=args.resolution,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "VERSION 2 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nProducts used:\n"
        f"{result['selected_products']}"
    )

    print(
        f"\nSentinel-2 outputs:\n"
        f"{result['output_dir']}"
    )

    print(
        f"\nReports:\n"
        f"{result['reports_dir']}"
    )

    print(
        "\nReady for Version 3: "
        "Sentinel-1 seasonal radar features."
    )


if __name__ == "__main__":
    main()
