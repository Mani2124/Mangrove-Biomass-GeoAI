"""Match GEDI L4A reference observations to satellite features."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from pyproj import CRS, Geod, Transformer
from rasterio.features import geometry_window
from rasterio.errors import WindowError
from shapely.geometry import Polygon, mapping
from shapely.ops import transform
import re


GEOD = Geod(ellps="WGS84")

SENSOR_FOLDERS = {
    "Sentinel-2": ("s2", "sentinel2"),
    "Sentinel-1": ("s1", "sentinel1"),
    "ALOS-2": ("alos2", "alos2"),
}

REQUIRED_COLUMNS = {
    "shot_number",
    "latitude",
    "longitude",
    "agbd_mg_ha",
    "agbd_se_mg_ha",
    "observation_time",
}

EXCLUDE_WORDS = {
    "mask", "valid", "cloud", "preview",
    "rgb", "count", "quality", "scl",
}


def clean_feature_name(text):
    """Use exactly the same raster-name normalisation as inference."""
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def discover_rasters(run, sensors):
    """Find continuous predictor rasters for selected sensors."""
    rasters = {}

    for sensor in sensors:
        prefix, folder_name = SENSOR_FOLDERS[sensor]
        folder = run / "processed" / folder_name

        files = [
            path
            for path in sorted(folder.glob("*.tif"))
            if not any(
                word in path.stem.lower()
                for word in EXCLUDE_WORDS
            )
        ]

        if not files:
            raise ValueError(
                f"No predictor rasters found for {sensor}: {folder}"
            )

        for path in files:
            feature_name = f"{prefix}__{clean_feature_name(path.name)}__mean"
            rasters[feature_name] = path

    return rasters


def footprint_polygon(longitude, latitude, radius_m=12.5):
    """Approximate a 25 m diameter footprint as a geodesic circle."""
    bearings = np.linspace(0, 360, 49)[:-1]

    longitudes, latitudes, _ = GEOD.fwd(
        np.full(len(bearings), longitude),
        np.full(len(bearings), latitude),
        bearings,
        np.full(len(bearings), radius_m),
    )

    return Polygon(zip(longitudes, latitudes))


def footprint_mean(source, polygon, minimum_coverage=0.8):
    """
    Calculate a pixel-area-weighted mean within a footprint.

    This approximates the footprint as a uniform circle, not the
    instrument's exact energy distribution.
    """
    try:
        window = geometry_window(
            source,
            [mapping(polygon)],
        )
    except WindowError:
        return np.nan

    data = source.read(
        1,
        window=window,
        masked=True,
    ).astype("float64")

    valid = (
        ~np.ma.getmaskarray(data)
        & np.isfinite(data.data)
    )

    window_transform = source.window_transform(window)

    weighted_sum = 0.0
    valid_area = 0.0

    for row, column in np.argwhere(valid):
        corners = [
            window_transform * (column, row),
            window_transform * (column + 1, row),
            window_transform * (column + 1, row + 1),
            window_transform * (column, row + 1),
        ]

        pixel = Polygon(corners)
        overlap_area = polygon.intersection(pixel).area

        if overlap_area > 0:
            weighted_sum += float(data[row, column]) * overlap_area
            valid_area += overlap_area

    if polygon.area <= 0:
        return np.nan

    coverage = valid_area / polygon.area

    if coverage < minimum_coverage or valid_area == 0:
        return np.nan

    return weighted_sum / valid_area


def build_training_table(
    csv_path,
    run,
    sensors,
    region_id,
    start_date,
    end_date,
    minimum_coverage=0.8,
):
    """
    Return an audited table and summary.

    Dates are supplied by the user. They must represent the intended
    matching period; this function does not verify scene acquisition dates.
    """
    run = Path(run)

    if "Sentinel-2" not in sensors:
        raise ValueError("Sentinel-2 must be included.")

    if not region_id.strip():
        raise ValueError("Enter a training-region name.")

    labels = pd.read_csv(
        csv_path,
        dtype={"shot_number": "string"},
    )

    missing = REQUIRED_COLUMNS - set(labels.columns)

    if missing:
        raise ValueError(
            "Missing GEDI CSV columns: " + ", ".join(sorted(missing))
        )

    # Keep this first version manageable on a local computer.
    if len(labels) > 50000:
        raise ValueError(
            "This CSV contains more than 50,000 points. "
            "Export a smaller region from the GEDI panel first."
        )

    for column in [
        "latitude",
        "longitude",
        "agbd_mg_ha",
        "agbd_se_mg_ha",
    ]:
        labels[column] = pd.to_numeric(
            labels[column],
            errors="coerce",
        )

    dates = pd.to_datetime(
        labels["observation_time"],
        utc=True,
        errors="coerce",
    )

    start = pd.Timestamp(start_date, tz="UTC")
    stop = (
        pd.Timestamp(end_date, tz="UTC")
        + pd.Timedelta(days=1)
    )

    if start >= stop:
        raise ValueError("The date range is invalid.")

    valid_labels = (
        np.isfinite(
            labels[
                [
                    "latitude",
                    "longitude",
                    "agbd_mg_ha",
                    "agbd_se_mg_ha",
                ]
            ]
        ).all(axis=1)
        & labels["latitude"].between(-90, 90)
        & labels["longitude"].between(-180, 180)
        & (labels["agbd_mg_ha"] >= 0)
        & (labels["agbd_se_mg_ha"] >= 0)
        & labels["shot_number"].notna()
    )

    input_count = len(labels)

    labels = labels.loc[
        valid_labels
        & dates.ge(start)
        & dates.lt(stop)
    ].copy()

    labels = labels.drop_duplicates(
        subset=["shot_number"]
    ).reset_index(drop=True)

    if labels.empty:
        raise ValueError(
            "No valid GEDI observations remain in the selected dates."
        )

    rasters = discover_rasters(run, sensors)
    feature_names = list(rasters)

    with ExitStack() as stack:
        sources = {}
        transformers = {}

        for name, path in rasters.items():
            source = stack.enter_context(rasterio.open(path))

            if source.crs is None:
                raise ValueError(f"Missing CRS: {path.name}")

            crs = CRS.from_user_input(source.crs)

            if (
                not crs.is_projected
                or not crs.axis_info
                or any(
                    abs(axis.unit_conversion_factor - 1.0) > 1e-8
                    for axis in crs.axis_info[:2]
                )
            ):
                raise ValueError(
                    f"{path.name}: use a projected raster in metres."
                )

            sources[name] = source
            transformers[name] = Transformer.from_crs(
                "EPSG:4326",
                source.crs,
                always_xy=True,
            )

            labels[name] = np.nan

        for index, row in labels.iterrows():
            geographic_footprint = footprint_polygon(
                row["longitude"],
                row["latitude"],
            )

            for name, source in sources.items():
                projected_footprint = transform(
                    transformers[name].transform,
                    geographic_footprint,
                )

                labels.at[index, name] = footprint_mean(
                    source,
                    projected_footprint,
                    minimum_coverage=minimum_coverage,
                )

    labels["training_region_id"] = region_id.strip()
    labels["source_run"] = run.name
    labels["footprint_diameter_m"] = 25.0

    labels["feature_valid_fraction"] = (
        labels[feature_names].notna().mean(axis=1)
    )

    # Require all selected predictors in this initial version.
    labels["sample_ready"] = (
        labels[feature_names].notna().all(axis=1)
    )

    report = {
        "input_csv_points": input_count,
        "points_after_dates_and_label_checks": len(labels),
        "predictor_count": len(feature_names),
        "ready_samples": int(labels["sample_ready"].sum()),
        "incomplete_samples": int(
            (~labels["sample_ready"]).sum()
        ),
        "sensors": list(sensors),
        "minimum_valid_footprint_coverage": minimum_coverage,
        "training_region": region_id.strip(),
        "run": str(run),
        "matching_period": [
            str(start_date),
            str(end_date),
        ],
        "predictor_files": {
            name: str(path)
            for name, path in rasters.items()
        },
    }

    return labels, report
