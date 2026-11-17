"""Search and download GEDI L4A V3 for a saved AOI."""

import json
import os
from pathlib import Path

import earthaccess
from dotenv import load_dotenv
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry.polygon import orient

COLLECTION_CONCEPT_ID = "C4212593885-ORNL_CLOUD"


def read_aoi_polygon(aoi_file, maximum_vertices=80):
    """Create a simplified counter-clockwise NASA search polygon."""

    content = json.loads(
        Path(aoi_file).read_text(encoding="utf-8")
    )

    geometries = [
        shape(feature["geometry"])
        for feature in content["features"]
        if feature.get("geometry")
        and feature["geometry"]["type"]
        in ("Polygon", "MultiPolygon")
    ]

    if not geometries:
        raise ValueError("No polygon found in the saved AOI.")

    region = unary_union(geometries)

    if not region.is_valid:
        region = region.buffer(0)

    # Use one search boundary when the AOI contains multiple polygons.
    if region.geom_type == "MultiPolygon":
        region = region.convex_hull

    simplified = region.simplify(
        0.001,
        preserve_topology=True,
    )

    if simplified.geom_type == "MultiPolygon":
        simplified = simplified.convex_hull

    # NASA CMR requires counter-clockwise exterior coordinates.
    simplified = orient(simplified, sign=1.0)

    coordinates = list(simplified.exterior.coords)

    if len(coordinates) > maximum_vertices:
        # Remove the existing closing coordinate temporarily.
        coordinates = coordinates[:-1]

        indices = [
            round(index * (len(coordinates) - 1)
                  / (maximum_vertices - 1))
            for index in range(maximum_vertices)
        ]

        coordinates = [
            coordinates[index]
            for index in indices
        ]

    # Close the polygon.
    if coordinates[0] != coordinates[-1]:
        coordinates.append(coordinates[0])

    return [
        (float(longitude), float(latitude))
        for longitude, latitude in coordinates
    ]

def earthdata_login():
    """Login using credentials stored in the project .env file."""
    load_dotenv()

    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        raise ValueError(
            "EARTHDATA_USERNAME and EARTHDATA_PASSWORD are missing "
            "from the project .env file."
        )

    os.environ["EARTHDATA_USERNAME"] = username
    os.environ["EARTHDATA_PASSWORD"] = password

    return earthaccess.login(
        strategy="environment",
        persist=False,
    )


def search_gedi(
    aoi_file,
    start_date,
    end_date,
    maximum_results=500,
):
    """Search GEDI L4A V3 using the actual AOI polygon."""

    polygon = read_aoi_polygon(aoi_file)

    earthdata_login()

    results = earthaccess.search_data(
        concept_id=COLLECTION_CONCEPT_ID,
        polygon=polygon,
        temporal=(str(start_date), str(end_date)),
        count=maximum_results,
    )

    return results


def download_gedi(results, output_folder):
    """Download selected GEDI granules."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not results:
        raise ValueError(
            "No GEDI L4A granules overlap the AOI and selected dates."
        )

    downloaded = earthaccess.download(
        results,
        local_path=str(output_folder),
    )

    return [Path(path) for path in downloaded]