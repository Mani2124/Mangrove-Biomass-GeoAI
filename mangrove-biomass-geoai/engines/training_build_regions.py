
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box


DEFAULT_INPUT_CSV = Path(
    r"D:\biomass_carbon_platform\training\forest_mangrove\gedi_mangrove_forest_candidate_training_cells.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\biomass_carbon_platform\training\regions"
)

DEFAULT_TILE_SIZE_DEG = 2.0

# Limit the number of GEDI cells per regional job so the training
# dataset remains geographically balanced.
DEFAULT_MAX_SAMPLES_PER_REGION = 2500

# Optional hard cap on total jobs. 0 = all.
DEFAULT_MAX_REGIONS = 0

RANDOM_SEED = 42


def assign_region_ids(
    frame: pd.DataFrame,
    tile_size_deg: float,
) -> pd.DataFrame:
    if tile_size_deg <= 0:
        raise ValueError(
            "--tile-size-deg must be > 0."
        )

    lon = frame["longitude"].to_numpy(
        dtype="float64"
    )
    lat = frame["latitude"].to_numpy(
        dtype="float64"
    )

    lon_bin = np.floor(
        (lon + 180.0)
        /
        tile_size_deg
    ).astype("int32")

    lat_bin = np.floor(
        (lat + 90.0)
        /
        tile_size_deg
    ).astype("int32")

    frame = frame.copy()

    frame[
        "region_lon_bin"
    ] = lon_bin

    frame[
        "region_lat_bin"
    ] = lat_bin

    frame[
        "training_region_id"
    ] = [
        f"R_{lat_i:03d}_{lon_i:03d}"
        for lat_i, lon_i in zip(
            lat_bin,
            lon_bin,
        )
    ]

    return frame


def region_bounds(
    lon_bin: int,
    lat_bin: int,
    tile_size_deg: float,
):
    west = (
        lon_bin
        *
        tile_size_deg
        -
        180.0
    )

    south = (
        lat_bin
        *
        tile_size_deg
        -
        90.0
    )

    east = (
        west
        +
        tile_size_deg
    )

    north = (
        south
        +
        tile_size_deg
    )

    return (
        west,
        south,
        east,
        north,
    )


def balanced_region_sample(
    group: pd.DataFrame,
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    if (
        max_samples <= 0
        or
        len(group) <= max_samples
    ):
        return group.copy()

    # Preserve mangrove cells when present.
    if "land_type" in group.columns:
        mangrove = group.loc[
            group[
                "land_type"
            ]
            ==
            "mangrove"
        ]

        other = group.loc[
            group[
                "land_type"
            ]
            !=
            "mangrove"
        ]

        if len(mangrove) >= max_samples:
            return mangrove.sample(
                n=max_samples,
                random_state=seed,
            ).copy()

        remaining = (
            max_samples
            -
            len(mangrove)
        )

        if len(other) > remaining:
            other = other.sample(
                n=remaining,
                random_state=seed,
            )

        return pd.concat(
            [
                mangrove,
                other,
            ],
            ignore_index=True,
        )

    return group.sample(
        n=max_samples,
        random_state=seed,
    ).copy()


def create_training_regions(
    input_csv: Path,
    output_dir: Path,
    tile_size_deg: float,
    max_samples_per_region: int,
    max_regions: int,
) -> dict:
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found:\n{input_csv}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    regions_dir = (
        output_dir
        /
        "aoi"
    )

    samples_dir = (
        output_dir
        /
        "samples"
    )

    reports_dir = (
        output_dir
        /
        "reports"
    )

    for directory in [
        regions_dir,
        samples_dir,
        reports_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(
        "\n[1/6] Reading selected GEDI training cells..."
    )

    frame = pd.read_csv(
        input_csv
    )

    required = {
        "longitude",
        "latitude",
        "row",
        "column",
        "agbd_mu_mg_ha",
    }

    missing = (
        required
        -
        set(
            frame.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Training CSV is missing required columns:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
        )

    frame = frame[
        np.isfinite(
            frame[
                "longitude"
            ]
        )
        &
        np.isfinite(
            frame[
                "latitude"
            ]
        )
        &
        np.isfinite(
            frame[
                "agbd_mu_mg_ha"
            ]
        )
    ].copy()

    print(
        f"Input cells: "
        f"{len(frame):,}"
    )

    print(
        "\n[2/6] Assigning cells to geographic training regions..."
    )

    frame = assign_region_ids(
        frame,
        tile_size_deg,
    )

    grouped = list(
        frame.groupby(
            "training_region_id",
            sort=True,
        )
    )

    print(
        f"Candidate regions: "
        f"{len(grouped):,}"
    )

    # Prioritize regions with more samples.
    grouped = sorted(
        grouped,
        key=lambda item:
            len(item[1]),
        reverse=True,
    )

    if max_regions > 0:
        grouped = grouped[
            :max_regions
        ]

    print(
        f"Regions selected: "
        f"{len(grouped):,}"
    )

    print(
        "\n[3/6] Creating per-region GEDI sample tables..."
    )

    manifest_records = []
    total_selected = 0

    for index, (
        region_id,
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):
        lon_bin = int(
            group[
                "region_lon_bin"
            ].iloc[0]
        )

        lat_bin = int(
            group[
                "region_lat_bin"
            ].iloc[0]
        )

        west, south, east, north = (
            region_bounds(
                lon_bin,
                lat_bin,
                tile_size_deg,
            )
        )

        selected = balanced_region_sample(
            group,
            max_samples_per_region,
            RANDOM_SEED + index,
        )

        total_selected += len(
            selected
        )

        sample_file = (
            samples_dir
            /
            f"{region_id}_gedi.csv"
        )

        selected.to_csv(
            sample_file,
            index=False,
        )

        aoi_file = (
            regions_dir
            /
            f"{region_id}.geojson"
        )

        aoi = gpd.GeoDataFrame(
            {
                "training_region_id":
                    [
                        region_id
                    ],

                "gedi_sample_count":
                    [
                        len(
                            selected
                        )
                    ],
            },
            geometry=[
                box(
                    west,
                    south,
                    east,
                    north,
                )
            ],
            crs="EPSG:4326",
        )

        aoi.to_file(
            aoi_file,
            driver="GeoJSON",
        )

        if "land_type" in selected.columns:
            mangrove_count = int(
                np.count_nonzero(
                    selected[
                        "land_type"
                    ]
                    ==
                    "mangrove"
                )
            )

            forest_count = int(
                np.count_nonzero(
                    selected[
                        "land_type"
                    ]
                    !=
                    "mangrove"
                )
            )
        else:
            mangrove_count = 0
            forest_count = len(
                selected
            )

        manifest_records.append(
            {
                "region_index":
                    index,

                "training_region_id":
                    region_id,

                "west":
                    west,

                "south":
                    south,

                "east":
                    east,

                "north":
                    north,

                "tile_size_deg":
                    tile_size_deg,

                "input_gedi_cells":
                    len(
                        group
                    ),

                "selected_gedi_cells":
                    len(
                        selected
                    ),

                "mangrove_cells":
                    mangrove_count,

                "forest_candidate_cells":
                    forest_count,

                "aoi_file":
                    str(
                        aoi_file
                    ),

                "gedi_sample_file":
                    str(
                        sample_file
                    ),

                "status":
                    "pending_predictor_processing",
            }
        )

        if (
            index == 1
            or
            index % 25 == 0
            or
            index == len(
                grouped
            )
        ):
            print(
                f"Region "
                f"{index}/{len(grouped)} "
                f"| samples selected: "
                f"{total_selected:,}"
            )

    print(
        "\n[4/6] Saving regional job manifest..."
    )

    manifest = pd.DataFrame(
        manifest_records
    )

    manifest_file = (
        output_dir
        /
        "training_region_manifest.csv"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )

    print(
        "\n[5/6] Saving processing commands..."
    )

    commands_file = (
        output_dir
        /
        "training_region_commands.txt"
    )

    lines = []

    for record in (
        manifest_records
    ):
        aoi_file = record[
            "aoi_file"
        ]

        region_id = record[
            "training_region_id"
        ]

        lines.extend(
            [
                (
                    f"# {region_id}"
                ),
                (
                    "D:\\Anaconda\\python.exe "
                    ".\\v1_prepare_aoi.py "
                    f'--aoi "{aoi_file}"'
                ),
                (
                    "# Then run V2, V3, V4 and optional V5 "
                    "using the newly created run directory."
                ),
                "",
            ]
        )

    commands_file.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    print(
        "\n[6/6] Saving summary..."
    )

    summary = {
        "purpose":
            (
                "Create manageable geographic jobs for "
                "training predictor processing."
            ),

        "input_csv":
            str(
                input_csv
            ),

        "tile_size_degrees":
            tile_size_deg,

        "candidate_regions":
            int(
                frame[
                    "training_region_id"
                ].nunique()
            ),

        "selected_regions":
            int(
                len(
                    manifest_records
                )
            ),

        "max_samples_per_region":
            int(
                max_samples_per_region
            ),

        "selected_gedi_cells":
            int(
                total_selected
            ),

        "manifest":
            str(
                manifest_file
            ),

        "commands":
            str(
                commands_file
            ),

        "next_step":
            (
                "Process each training region with V1-V5, "
                "then aggregate regional S2/S1/Terrain/ALOS-2 "
                "features to the GEDI 1-km cells."
            ),
    }

    summary_file = (
        reports_dir
        /
        "training_region_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "regions":
            len(
                manifest_records
            ),

        "samples":
            total_selected,

        "manifest_file":
            manifest_file,

        "commands_file":
            commands_file,

        "summary_file":
            summary_file,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Create geographic training regions from "
            "GEDI forest/mangrove training cells."
        )
    )

    parser.add_argument(
        "--input-csv",
        default=str(
            DEFAULT_INPUT_CSV
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--tile-size-deg",
        type=float,
        default=
            DEFAULT_TILE_SIZE_DEG,
    )

    parser.add_argument(
        "--max-samples-per-region",
        type=int,
        default=
            DEFAULT_MAX_SAMPLES_PER_REGION,
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=
            DEFAULT_MAX_REGIONS,
        help=(
            "0 means use all geographic regions."
        ),
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
        "TRAINING 02 - GEOGRAPHIC TRAINING REGION BUILDER"
    )

    print(
        "=" * 72
    )

    args = parse_arguments()

    result = create_training_regions(
        input_csv=
            Path(
                args.input_csv
            ),

        output_dir=
            Path(
                args.output_dir
            ),

        tile_size_deg=
            args.tile_size_deg,

        max_samples_per_region=
            args.max_samples_per_region,

        max_regions=
            args.max_regions,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "TRAINING 02 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTraining regions:\n"
        f"{result['regions']:,}"
    )

    print(
        f"\nGEDI cells selected:\n"
        f"{result['samples']:,}"
    )

    print(
        f"\nManifest:\n"
        f"{result['manifest_file']}"
    )

    print(
        "\nNext: process the regional AOIs with "
        "V1 + V2 + V3 + V4 + optional V5."
    )


if __name__ == "__main__":
    main()
