
from __future__ import annotations

from pathlib import Path
import argparse
import json

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.strtree import STRtree
from pyproj import Transformer


DEFAULT_GEDI_CSV = Path(
    r"D:\biomass_carbon_platform\training\gedi_l4b\labels\gedi_l4b_training_cells.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\biomass_carbon_platform\training\forest_mangrove"
)

DEFAULT_CHUNK_SIZE = 20_000

# GEDI L4B native grid is ~1 km in EPSG:6933.
# We reconstruct each cell from x/y center coordinates.
DEFAULT_CELL_SIZE_M = 1000.8950233496674

# A cell is called mangrove when at least this fraction of the
# 1-km GEDI cell overlaps the mangrove polygons.
DEFAULT_MANGROVE_FRACTION_THRESHOLD = 0.20

DEFAULT_MAX_MANGROVE = 0
DEFAULT_MAX_FOREST_CANDIDATE = 100_000
DEFAULT_FOREST_EVERY = 2


def read_mangrove_layer(
    mangrove_file: Path,
) -> gpd.GeoDataFrame:
    if not mangrove_file.exists():
        raise FileNotFoundError(
            f"Mangrove file not found:\n{mangrove_file}"
        )

    mangroves = gpd.read_file(
        mangrove_file
    )

    if mangroves.empty:
        raise ValueError(
            "Mangrove layer is empty."
        )

    if mangroves.crs is None:
        raise ValueError(
            "Mangrove layer has no CRS."
        )

    mangroves = mangroves[
        mangroves.geometry.notna()
        &
        ~mangroves.geometry.is_empty
    ].copy()

    if mangroves.empty:
        raise ValueError(
            "Mangrove layer contains no valid geometry."
        )

    # Work in GEDI equal-area CRS so overlap fraction is meaningful.
    mangroves = mangroves.to_crs(
        "EPSG:6933"
    )

    return mangroves


def build_spatial_index(
    mangroves: gpd.GeoDataFrame,
):
    geometries = list(
        mangroves.geometry
    )

    tree = STRtree(
        geometries
    )

    return geometries, tree


def deterministic_thinning(
    frame: pd.DataFrame,
    every: int,
) -> pd.DataFrame:
    if every <= 1:
        return frame

    keep = (
        (
            frame["row"]
            %
            every
        )
        ==
        0
    ) & (
        (
            frame["column"]
            %
            every
        )
        ==
        0
    )

    return frame.loc[
        keep
    ].copy()


def calculate_overlap_fraction(
    x_center: float,
    y_center: float,
    cell_size_m: float,
    mangrove_geometries,
    tree: STRtree,
) -> float:
    half = (
        cell_size_m
        /
        2.0
    )

    cell = box(
        x_center - half,
        y_center - half,
        x_center + half,
        y_center + half,
    )

    candidates = tree.query(
        cell
    )

    if len(candidates) == 0:
        return 0.0

    intersection_area = 0.0

    # Shapely STRtree may return geometry indices.
    for candidate in candidates:
        if isinstance(
            candidate,
            (int, np.integer),
        ):
            geometry = (
                mangrove_geometries[
                    int(candidate)
                ]
            )
        else:
            geometry = candidate

        try:
            if geometry.intersects(
                cell
            ):
                intersection_area += (
                    geometry.intersection(
                        cell
                    ).area
                )
        except Exception:
            continue

    cell_area = (
        cell_size_m
        *
        cell_size_m
    )

    if cell_area <= 0:
        return 0.0

    fraction = (
        intersection_area
        /
        cell_area
    )

    return float(
        min(
            max(
                fraction,
                0.0,
            ),
            1.0,
        )
    )


def classify_chunk(
    chunk: pd.DataFrame,
    mangrove_geometries,
    tree: STRtree,
    cell_size_m: float,
    mangrove_fraction_threshold: float,
) -> pd.DataFrame:
    required = {
        "x_6933_m",
        "y_6933_m",
        "row",
        "column",
        "longitude",
        "latitude",
        "agbd_mu_mg_ha",
    }

    missing = (
        required
        -
        set(
            chunk.columns
        )
    )

    if missing:
        raise RuntimeError(
            "GEDI CSV is missing required columns:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
        )

    chunk = chunk[
        np.isfinite(
            chunk["x_6933_m"]
        )
        &
        np.isfinite(
            chunk["y_6933_m"]
        )
        &
        np.isfinite(
            chunk["agbd_mu_mg_ha"]
        )
    ].copy()

    if chunk.empty:
        return chunk

    fractions = np.zeros(
        len(chunk),
        dtype="float32",
    )

    x_values = chunk[
        "x_6933_m"
    ].to_numpy()

    y_values = chunk[
        "y_6933_m"
    ].to_numpy()

    for i in range(
        len(chunk)
    ):
        fractions[i] = (
            calculate_overlap_fraction(
                x_center=
                    float(
                        x_values[i]
                    ),

                y_center=
                    float(
                        y_values[i]
                    ),

                cell_size_m=
                    cell_size_m,

                mangrove_geometries=
                    mangrove_geometries,

                tree=
                    tree,
            )
        )

    chunk[
        "mangrove_fraction"
    ] = fractions

    chunk[
        "land_type"
    ] = np.where(
        chunk[
            "mangrove_fraction"
        ]
        >=
        mangrove_fraction_threshold,

        "mangrove",

        "forest_candidate",
    )

    return chunk


def filter_training_cells(
    gedi_csv: Path,
    mangrove_file: Path,
    output_dir: Path,
    chunk_size: int,
    cell_size_m: float,
    mangrove_fraction_threshold: float,
    max_mangrove: int,
    max_forest_candidate: int,
    forest_every: int,
) -> dict:
    if not gedi_csv.exists():
        raise FileNotFoundError(
            f"GEDI training CSV not found:\n{gedi_csv}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir = (
        output_dir
        /
        "reports"
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv = (
        output_dir
        /
        "gedi_mangrove_forest_candidate_training_cells.csv"
    )

    if output_csv.exists():
        output_csv.unlink()

    print(
        "\n[1/6] Loading global mangrove layer..."
    )

    mangroves = read_mangrove_layer(
        mangrove_file
    )

    print(
        f"Mangrove polygons: "
        f"{len(mangroves):,}"
    )

    print(
        "\n[2/6] Building mangrove spatial index..."
    )

    mangrove_geometries, tree = (
        build_spatial_index(
            mangroves
        )
    )

    print(
        "\n[3/6] Reading GEDI labels in chunks..."
    )

    total_seen = 0
    total_mangrove_found = 0
    total_forest_candidate_found = 0

    mangrove_written = 0
    forest_candidate_written = 0

    header_written = False
    chunk_number = 0

    for chunk in pd.read_csv(
        gedi_csv,
        chunksize=chunk_size,
    ):
        chunk_number += 1
        total_seen += len(
            chunk
        )

        classified = classify_chunk(
            chunk=
                chunk,

            mangrove_geometries=
                mangrove_geometries,

            tree=
                tree,

            cell_size_m=
                cell_size_m,

            mangrove_fraction_threshold=
                mangrove_fraction_threshold,
        )

        if classified.empty:
            continue

        mangrove = (
            classified.loc[
                classified[
                    "land_type"
                ]
                ==
                "mangrove"
            ]
            .copy()
        )

        forest_candidate = (
            classified.loc[
                classified[
                    "land_type"
                ]
                ==
                "forest_candidate"
            ]
            .copy()
        )

        total_mangrove_found += len(
            mangrove
        )

        total_forest_candidate_found += len(
            forest_candidate
        )

        forest_candidate = (
            deterministic_thinning(
                forest_candidate,
                forest_every,
            )
        )

        if max_mangrove > 0:
            remaining = (
                max_mangrove
                -
                mangrove_written
            )

            if remaining <= 0:
                mangrove = mangrove.iloc[
                    0:0
                ]

            elif len(
                mangrove
            ) > remaining:
                mangrove = mangrove.iloc[
                    :remaining
                ]

        if max_forest_candidate > 0:
            remaining = (
                max_forest_candidate
                -
                forest_candidate_written
            )

            if remaining <= 0:
                forest_candidate = (
                    forest_candidate.iloc[
                        0:0
                    ]
                )

            elif len(
                forest_candidate
            ) > remaining:
                forest_candidate = (
                    forest_candidate.iloc[
                        :remaining
                    ]
                )

        selected = pd.concat(
            [
                mangrove,
                forest_candidate,
            ],
            ignore_index=True,
        )

        if not selected.empty:
            selected.to_csv(
                output_csv,
                mode=(
                    "a"
                    if header_written
                    else "w"
                ),
                header=(
                    not header_written
                ),
                index=False,
            )

            header_written = True

            mangrove_written += len(
                mangrove
            )

            forest_candidate_written += len(
                forest_candidate
            )

        print(
            f"Chunk {chunk_number}: "
            f"seen={total_seen:,} | "
            f"mangrove={mangrove_written:,} | "
            f"forest_candidate={forest_candidate_written:,}"
        )

        forest_done = (
            max_forest_candidate > 0
            and
            forest_candidate_written
            >=
            max_forest_candidate
        )

        mangrove_done = (
            max_mangrove > 0
            and
            mangrove_written
            >=
            max_mangrove
        )

        if (
            forest_done
            and
            mangrove_done
        ):
            print(
                "Reached requested class limits."
            )
            break

    print(
        "\n[4/6] Finalizing selected training cells..."
    )

    if not header_written:
        raise RuntimeError(
            "No GEDI cells were written."
        )

    print(
        f"Mangrove cells        : "
        f"{mangrove_written:,}"
    )

    print(
        f"Forest candidate cells: "
        f"{forest_candidate_written:,}"
    )

    print(
        "\n[5/6] Saving reports..."
    )

    summary = {
        "input_gedi_csv":
            str(
                gedi_csv
            ),

        "mangrove_file":
            str(
                mangrove_file
            ),

        "gedi_cell_size_m":
            float(
                cell_size_m
            ),

        "mangrove_fraction_threshold":
            float(
                mangrove_fraction_threshold
            ),

        "classification_rule":
            (
                "GEDI 1-km cell overlap fraction with mangrove "
                "polygon >= threshold -> mangrove; otherwise "
                "forest_candidate."
            ),

        "gedi_cells_examined":
            int(
                total_seen
            ),

        "mangrove_found_before_sampling":
            int(
                total_mangrove_found
            ),

        "forest_candidate_found_before_sampling":
            int(
                total_forest_candidate_found
            ),

        "final_mangrove_cells":
            int(
                mangrove_written
            ),

        "final_forest_candidate_cells":
            int(
                forest_candidate_written
            ),

        "final_total_cells":
            int(
                mangrove_written
                +
                forest_candidate_written
            ),

        "output_csv":
            str(
                output_csv
            ),

        "scientific_note":
            (
                "forest_candidate is not yet a confirmed forest class. "
                "A land-cover/forest mask should be added later."
            ),

        "next_step":
            (
                "Build Sentinel-2, Sentinel-1, terrain and optional "
                "ALOS-2 predictor statistics for selected GEDI cells."
            ),
    }

    summary_file = (
        reports_dir
        /
        "mangrove_overlap_filter_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n[6/6] Complete."
    )

    return {
        "output_csv":
            output_csv,

        "summary_file":
            summary_file,

        "mangrove":
            mangrove_written,

        "forest_candidate":
            forest_candidate_written,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Classify GEDI L4B 1-km cells using fractional "
            "overlap with a global mangrove polygon layer."
        )
    )

    parser.add_argument(
        "--gedi-csv",
        default=str(
            DEFAULT_GEDI_CSV
        ),
    )

    parser.add_argument(
        "--mangrove-file",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=
            DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=
            DEFAULT_CELL_SIZE_M,
    )

    parser.add_argument(
        "--mangrove-fraction-threshold",
        type=float,
        default=
            DEFAULT_MANGROVE_FRACTION_THRESHOLD,
        help=(
            "Fraction of 1-km GEDI cell that must overlap "
            "mangrove polygons. Default 0.20."
        ),
    )

    parser.add_argument(
        "--max-mangrove",
        type=int,
        default=
            DEFAULT_MAX_MANGROVE,
        help=(
            "0 means keep all mangrove cells."
        ),
    )

    parser.add_argument(
        "--max-forest-candidate",
        type=int,
        default=
            DEFAULT_MAX_FOREST_CANDIDATE,
    )

    parser.add_argument(
        "--forest-every",
        type=int,
        default=
            DEFAULT_FOREST_EVERY,
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
        "TRAINING 01 - GEDI MANGROVE OVERLAP FILTER"
    )

    print(
        "=" * 72
    )

    args = parse_arguments()

    if not (
        0.0
        <=
        args.mangrove_fraction_threshold
        <=
        1.0
    ):
        raise ValueError(
            "--mangrove-fraction-threshold must be between 0 and 1."
        )

    result = filter_training_cells(
        gedi_csv=
            Path(
                args.gedi_csv
            ),

        mangrove_file=
            Path(
                args.mangrove_file
            ),

        output_dir=
            Path(
                args.output_dir
            ),

        chunk_size=
            args.chunk_size,

        cell_size_m=
            args.cell_size_m,

        mangrove_fraction_threshold=
            args.mangrove_fraction_threshold,

        max_mangrove=
            args.max_mangrove,

        max_forest_candidate=
            args.max_forest_candidate,

        forest_every=
            args.forest_every,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "TRAINING 01 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nMangrove cells:\n"
        f"{result['mangrove']:,}"
    )

    print(
        f"\nForest candidate cells:\n"
        f"{result['forest_candidate']:,}"
    )

    print(
        f"\nOutput:\n"
        f"{result['output_csv']}"
    )

    print(
        "\nNext: S2 + S1 + Terrain + optional ALOS-2 "
        "predictor extraction."
    )


if __name__ == "__main__":
    main()
