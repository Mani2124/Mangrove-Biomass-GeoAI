"""Read GEDI L4A Version 3 footprint biomass observations."""

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep


FIELDS = [
    "lat_lowestmode",
    "lon_lowestmode",
    "agbd",
    "agbd_se",
    "delta_time",
    "l2a_quality_flag_rel3",
    "l4a_quality_flag_rel3",
    "degrade_include_flag",
    "elev_highestreturn_outlier_flag",
    "predictor_limit_flag",
    "response_limit_flag",
]


def read_numeric(dataset, start, stop):
    """Replace explicitly declared missing values with NaN."""
    values = np.asarray(dataset[start:stop], dtype="float64")

    for attribute in ("_FillValue", "missing_value"):
        if attribute in dataset.attrs:
            for missing in np.asarray(
                dataset.attrs[attribute]
            ).reshape(-1):
                values[values == float(missing)] = np.nan

    return values


def read_l4a(
    file_path,
    aoi_file=None,
    exclude_extrapolated=False,
    chunk_size=50000,
    max_points=250000,
):
    """
    Read quality-filtered V3 observations.

    The point limit prevents excessive dashboard memory use.
    It raises an error rather than silently truncating the result.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.name.upper().endswith("_V003.H5"):
        raise ValueError(
            "Select an original GEDI L4A Version 3 file ending "
            "in _V003.h5."
        )

    region = None
    prepared_region = None

    if aoi_file is not None:
        content = json.loads(
            Path(aoi_file).read_text(encoding="utf-8")
        )

        region = unary_union(
            [
                shape(feature["geometry"])
                for feature in content["features"]
            ]
        )

        if region.is_empty or not region.is_valid:
            raise ValueError("The saved AOI is empty or invalid.")

        prepared_region = prep(region)

    frames = []
    total = 0
    quality_count = 0
    selected_count = 0

    with h5py.File(file_path, "r") as source:
        beams = [
            name
            for name in source
            if name.startswith("BEAM")
            and isinstance(source[name], h5py.Group)
        ]

        if not beams:
            raise ValueError("No GEDI beam groups found.")

        for beam_name in sorted(beams):
            beam = source[beam_name]

            # Empty beam groups can occur.
            if "shot_number" not in beam:
                continue

            count = len(beam["shot_number"])

            if count == 0:
                continue

            missing = [
                field for field in FIELDS if field not in beam
            ]

            if missing:
                raise ValueError(
                    f"{beam_name} is missing required V3 fields: "
                    + ", ".join(missing)
                )

            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)

                data = {
                    field: read_numeric(beam[field], start, stop)
                    for field in FIELDS
                }

                # Keep shot IDs as strings to avoid integer precision loss.
                shots = beam["shot_number"][start:stop]
                data["shot_number"] = [
                    str(int(number)) for number in shots
                ]

                table = pd.DataFrame(data)
                total += len(table)

                numeric_valid = np.isfinite(
                    table[
                        [
                            "lat_lowestmode",
                            "lon_lowestmode",
                            "agbd",
                            "agbd_se",
                            "delta_time",
                        ]
                    ]
                ).all(axis=1)

                keep = (
                    numeric_valid
                    & table["lat_lowestmode"].between(-90, 90)
                    & table["lon_lowestmode"].between(-180, 180)
                    & (table["agbd"] >= 0)
                    & (table["agbd_se"] >= 0)
                    & (table["delta_time"] >= 0)
                    & (table["l2a_quality_flag_rel3"] == 1)
                    & (table["l4a_quality_flag_rel3"] == 1)
                    & (table["degrade_include_flag"] == 1)
                    & (
                        table["elev_highestreturn_outlier_flag"]
                        == 0
                    )
                )

                if exclude_extrapolated:
                    keep &= (
                        (table["predictor_limit_flag"] == 0)
                        & (table["response_limit_flag"] == 0)
                    )

                table = table.loc[keep].copy()
                quality_count += len(table)

                if region is not None and not table.empty:
                    west, south, east, north = region.bounds

                    table = table.loc[
                        table["lon_lowestmode"].between(west, east)
                        & table["lat_lowestmode"].between(south, north)
                    ].copy()

                    inside = [
                        prepared_region.covers(Point(lon, lat))
                        for lon, lat in zip(
                            table["lon_lowestmode"],
                            table["lat_lowestmode"],
                        )
                    ]

                    table = table.loc[inside].copy()

                if table.empty:
                    continue

                selected_count += len(table)

                if selected_count > max_points:
                    raise ValueError(
                        f"More than {max_points:,} points selected. "
                        "Enable filtering to a smaller saved AOI."
                    )

                table["beam"] = beam_name
                table["source_file"] = file_path.name

                table["observation_time"] = (
                    pd.Timestamp("2018-01-01", tz="UTC")
                    + pd.to_timedelta(
                        table["delta_time"],
                        unit="s",
                    )
                ).astype(str)

                table = table.rename(
                    columns={
                        "lat_lowestmode": "latitude",
                        "lon_lowestmode": "longitude",
                        "agbd": "agbd_mg_ha",
                        "agbd_se": "agbd_se_mg_ha",
                    }
                )

                frames.append(table)

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame()
    )

    report = {
        "source_file": file_path.name,
        "shots_read": total,
        "passed_selected_quality_filters": quality_count,
        "selected_points": len(result),
        "aoi_filter_applied": region is not None,
        "extrapolated_predictions_excluded": exclude_extrapolated,
    }

    return result, report


def make_geojson(table):
    """Export all selected observations as point features."""
    # pandas converts NaN to JSON null.
    records = json.loads(table.to_json(orient="records"))
    features = []

    for record in records:
        longitude = record.pop("longitude")
        latitude = record.pop("latitude")

        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [longitude, latitude],
                },
                "properties": record,
            }
        )

    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        allow_nan=False,
    )