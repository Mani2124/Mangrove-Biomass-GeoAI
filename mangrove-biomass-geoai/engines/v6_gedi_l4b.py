
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer


DEFAULT_GEDI_DIR = Path(
    r"D:\GEDI_L4B_Gridded_Biomass_V2_1_2299\data"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\biomass_carbon_platform\training\gedi_l4b"
)

DEFAULT_WINDOW_SIZE = 1024

# IMPORTANT:
# sample_every=10 keeps roughly 1 out of every 100 grid cells
# because both row and column are thinned.
DEFAULT_SAMPLE_EVERY = 10

# Hard safety cap for the first training dataset.
DEFAULT_MAX_SAMPLES = 1_000_000

DEFAULT_MAX_PE = 100.0
DEFAULT_MIN_NC = 1
DEFAULT_MIN_NS = 1

LAYER_SUFFIXES = {
    "MU": "_MU.tif",
    "SE": "_SE.tif",
    "PE": "_PE.tif",
    "NC": "_NC.tif",
    "NS": "_NS.tif",
    "QF": "_QF.tif",
    "MI": "_MI.tif",
    "PS": "_PS.tif",
    "V1": "_V1.tif",
    "V2": "_V2.tif",
}


def discover_layers(data_dir: Path) -> dict[str, Path]:
    if not data_dir.exists():
        raise FileNotFoundError(
            f"GEDI folder not found:\n{data_dir}"
        )

    tif_files = [
        *data_dir.glob("*.tif"),
        *data_dir.glob("*.tiff"),
    ]

    if not tif_files:
        raise FileNotFoundError(
            f"No GeoTIFF files found in:\n{data_dir}"
        )

    layers = {}

    for key, suffix in LAYER_SUFFIXES.items():
        matches = [
            path
            for path in tif_files
            if path.name.upper().endswith(
                suffix.upper()
            )
        ]

        if matches:
            layers[key] = matches[0]

    if "MU" not in layers:
        raise FileNotFoundError(
            "GEDI MU layer was not found. "
            "Expected a file ending with _MU.tif"
        )

    return layers


def raster_info(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
        }


def validate_common_grid(
    layers: dict[str, Path],
) -> dict:
    reference = raster_info(
        layers["MU"]
    )

    for key, path in layers.items():
        info = raster_info(path)

        same = (
            info["width"] == reference["width"]
            and
            info["height"] == reference["height"]
            and
            info["crs"] == reference["crs"]
            and
            info["transform"].almost_equals(
                reference["transform"]
            )
        )

        if not same:
            raise RuntimeError(
                f"Grid mismatch for {key}:\n{path}"
            )

    return reference


def generate_windows(
    width: int,
    height: int,
    size: int,
):
    for row in range(
        0,
        height,
        size,
    ):
        height_here = min(
            size,
            height - row,
        )

        for col in range(
            0,
            width,
            size,
        ):
            width_here = min(
                size,
                width - col,
            )

            yield rasterio.windows.Window(
                col,
                row,
                width_here,
                height_here,
            )


def read_window(
    src,
    window,
) -> np.ndarray:
    array = src.read(
        1,
        window=window,
        masked=False,
    ).astype(
        "float32"
    )

    if src.nodata is not None:
        array[
            array == src.nodata
        ] = np.nan

    array[
        ~np.isfinite(array)
    ] = np.nan

    return array


def build_valid_mask(
    arrays: dict[str, np.ndarray],
    max_pe: float | None,
    min_nc: int | None,
    min_ns: int | None,
    require_qf_positive: bool,
) -> np.ndarray:
    mu = arrays["MU"]

    valid = (
        np.isfinite(mu)
        &
        (mu >= 0)
    )

    if "SE" in arrays:
        valid &= (
            np.isfinite(
                arrays["SE"]
            )
            &
            (
                arrays["SE"]
                >=
                0
            )
        )

    if (
        "PE" in arrays
        and
        max_pe is not None
    ):
        valid &= (
            np.isfinite(
                arrays["PE"]
            )
            &
            (
                arrays["PE"]
                >=
                0
            )
            &
            (
                arrays["PE"]
                <=
                max_pe
            )
        )

    if (
        "NC" in arrays
        and
        min_nc is not None
    ):
        valid &= (
            np.isfinite(
                arrays["NC"]
            )
            &
            (
                arrays["NC"]
                >=
                min_nc
            )
        )

    if (
        "NS" in arrays
        and
        min_ns is not None
    ):
        valid &= (
            np.isfinite(
                arrays["NS"]
            )
            &
            (
                arrays["NS"]
                >=
                min_ns
            )
        )

    if (
        "QF" in arrays
        and
        require_qf_positive
    ):
        valid &= (
            np.isfinite(
                arrays["QF"]
            )
            &
            (
                arrays["QF"]
                >
                0
            )
        )

    return valid


def pixel_centers(
    transform,
    rows,
    cols,
):
    xs, ys = rasterio.transform.xy(
        transform,
        rows,
        cols,
        offset="center",
    )

    return (
        np.asarray(
            xs,
            dtype="float64",
        ),
        np.asarray(
            ys,
            dtype="float64",
        ),
    )


def build_training_labels(
    data_dir: Path,
    output_dir: Path,
    window_size: int,
    max_pe: float | None,
    min_nc: int | None,
    min_ns: int | None,
    require_qf_positive: bool,
    sample_every: int,
    max_samples: int,
) -> dict:
    labels_dir = (
        output_dir
        /
        "labels"
    )

    reports_dir = (
        output_dir
        /
        "reports"
    )

    labels_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_file = (
        labels_dir
        /
        "gedi_l4b_training_cells.csv"
    )

    if csv_file.exists():
        csv_file.unlink()

    print(
        "\n[1/7] Discovering GEDI layers..."
    )

    layers = discover_layers(
        data_dir
    )

    for key in LAYER_SUFFIXES:
        print(
            f"{key:>2}: "
            f"{layers.get(key, 'not found')}"
        )

    print(
        "\n[2/7] Validating common grid..."
    )

    reference = validate_common_grid(
        layers
    )

    print(
        f"CRS        : "
        f"{reference['crs']}"
    )

    print(
        f"Raster size: "
        f"{reference['width']} x "
        f"{reference['height']}"
    )

    print(
        f"Pixel size : "
        f"{reference['transform'].a:.3f} m"
    )

    print(
        "\n[3/7] Opening GEDI rasters..."
    )

    datasets = {
        key:
            rasterio.open(path)
        for key, path in (
            layers.items()
        )
    }

    transformer = (
        Transformer.from_crs(
            reference["crs"],
            "EPSG:4326",
            always_xy=True,
        )
    )

    windows = list(
        generate_windows(
            reference["width"],
            reference["height"],
            window_size,
        )
    )

    print(
        f"Processing windows: "
        f"{len(windows)}"
    )

    print(
        f"Spatial thinning: "
        f"every {sample_every} cells "
        f"in row and column"
    )

    if max_samples > 0:
        print(
            f"Maximum samples: "
            f"{max_samples:,}"
        )

    print(
        "\n[4/7] Streaming quality-filtered "
        "training cells to disk..."
    )

    valid_mu_before = 0
    accepted_after_quality = 0
    written = 0
    header_written = False

    optional_columns = {
        "SE":
            "agbd_se_mg_ha",

        "PE":
            "agbd_percent_error",

        "NC":
            "gedi_track_count",

        "NS":
            "gedi_waveform_count",

        "QF":
            "quality_flag",

        "MI":
            "inference_mode",

        "PS":
            "prediction_stratum",

        "V1":
            "variance_component_1",

        "V2":
            "variance_component_2",
    }

    try:
        for window_index, window in enumerate(
            windows,
            start=1,
        ):
            arrays = {
                key:
                    read_window(
                        dataset,
                        window,
                    )
                for key, dataset in (
                    datasets.items()
                )
            }

            valid_mu_before += int(
                np.count_nonzero(
                    np.isfinite(
                        arrays["MU"]
                    )
                    &
                    (
                        arrays["MU"]
                        >=
                        0
                    )
                )
            )

            valid = build_valid_mask(
                arrays=
                    arrays,

                max_pe=
                    max_pe,

                min_nc=
                    min_nc,

                min_ns=
                    min_ns,

                require_qf_positive=
                    require_qf_positive,
            )

            accepted_after_quality += int(
                np.count_nonzero(
                    valid
                )
            )

            if not np.any(
                valid
            ):
                continue

            local_rows, local_cols = (
                np.where(
                    valid
                )
            )

            global_rows = (
                local_rows
                +
                int(
                    window.row_off
                )
            )

            global_cols = (
                local_cols
                +
                int(
                    window.col_off
                )
            )

            # Deterministic spatial thinning.
            if sample_every > 1:
                keep = (
                    (
                        global_rows
                        %
                        sample_every
                    )
                    ==
                    0
                ) & (
                    (
                        global_cols
                        %
                        sample_every
                    )
                    ==
                    0
                )

                local_rows = (
                    local_rows[
                        keep
                    ]
                )

                local_cols = (
                    local_cols[
                        keep
                    ]
                )

                global_rows = (
                    global_rows[
                        keep
                    ]
                )

                global_cols = (
                    global_cols[
                        keep
                    ]
                )

            if len(
                global_rows
            ) == 0:
                continue

            # Apply hard cap before constructing the DataFrame.
            if max_samples > 0:
                remaining = (
                    max_samples
                    -
                    written
                )

                if remaining <= 0:
                    break

                if len(
                    global_rows
                ) > remaining:
                    local_rows = (
                        local_rows[
                            :remaining
                        ]
                    )

                    local_cols = (
                        local_cols[
                            :remaining
                        ]
                    )

                    global_rows = (
                        global_rows[
                            :remaining
                        ]
                    )

                    global_cols = (
                        global_cols[
                            :remaining
                        ]
                    )

            x, y = pixel_centers(
                reference[
                    "transform"
                ],
                global_rows,
                global_cols,
            )

            longitude, latitude = (
                transformer.transform(
                    x,
                    y,
                )
            )

            frame = pd.DataFrame(
                {
                    "row":
                        global_rows.astype(
                            "int32"
                        ),

                    "column":
                        global_cols.astype(
                            "int32"
                        ),

                    "x_6933_m":
                        x,

                    "y_6933_m":
                        y,

                    "longitude":
                        longitude,

                    "latitude":
                        latitude,

                    "agbd_mu_mg_ha":
                        arrays[
                            "MU"
                        ][
                            local_rows,
                            local_cols,
                        ],
                }
            )

            for (
                layer_key,
                column_name,
            ) in (
                optional_columns.items()
            ):
                if layer_key in arrays:
                    frame[
                        column_name
                    ] = arrays[
                        layer_key
                    ][
                        local_rows,
                        local_cols,
                    ]

            # Windows do not overlap, so row/column pairs are already
            # unique. No global drop_duplicates() is needed.
            frame.to_csv(
                csv_file,
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
            written += len(
                frame
            )

            if (
                window_index == 1
                or
                window_index % 25 == 0
                or
                window_index == len(
                    windows
                )
            ):
                print(
                    f"Window "
                    f"{window_index}/{len(windows)} "
                    f"- written: "
                    f"{written:,}"
                )

            if (
                max_samples > 0
                and
                written
                >=
                max_samples
            ):
                print(
                    f"Reached maximum "
                    f"{max_samples:,} samples."
                )
                break

    finally:
        for dataset in (
            datasets.values()
        ):
            dataset.close()

    print(
        "\n[5/7] Finalizing streamed dataset..."
    )

    if not header_written:
        raise RuntimeError(
            "No GEDI cells were written. "
            "Relax the quality filters."
        )

    print(
        f"Final training cells written: "
        f"{written:,}"
    )

    print(
        "\n[6/7] Saving summary..."
    )

    summary = {
        "product":
            "GEDI L4B Gridded Biomass V2.1",

        "target":
            "MU mean aboveground biomass density",

        "target_units":
            "Mg/ha",

        "source_directory":
            str(
                data_dir
            ),

        "layers":
            {
                key:
                    str(
                        path
                    )
                for key, path in (
                    layers.items()
                )
            },

        "crs":
            str(
                reference[
                    "crs"
                ]
            ),

        "width":
            int(
                reference[
                    "width"
                ]
            ),

        "height":
            int(
                reference[
                    "height"
                ]
            ),

        "pixel_size_m":
            float(
                reference[
                    "transform"
                ].a
            ),

        "valid_mu_cells_before_filters":
            int(
                valid_mu_before
            ),

        "accepted_after_quality_filters":
            int(
                accepted_after_quality
            ),

        "final_training_cells_written":
            int(
                written
            ),

        "filters": {
            "max_PE":
                max_pe,

            "min_NC":
                min_nc,

            "min_NS":
                min_ns,

            "require_QF_positive":
                require_qf_positive,

            "sample_every":
                sample_every,

            "max_samples":
                max_samples,
        },

        "streaming_write":
            True,

        "global_dataframe_concatenation":
            False,

        "csv":
            str(
                csv_file
            ),

        "next_step":
            (
                "Filter/select forest and mangrove training cells, "
                "then aggregate Sentinel-2, Sentinel-1, terrain "
                "and optional ALOS-2 predictors to the GEDI 1-km cells."
            ),
    }

    summary_file = (
        reports_dir
        /
        "gedi_l4b_training_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n[7/7] Complete."
    )

    return {
        "training_cells":
            written,

        "csv_file":
            csv_file,

        "summary_file":
            summary_file,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Memory-safe GEDI L4B local training-label builder."
        )
    )

    parser.add_argument(
        "--gedi-dir",
        default=str(
            DEFAULT_GEDI_DIR
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=
            DEFAULT_WINDOW_SIZE,
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=
            DEFAULT_SAMPLE_EVERY,
        help=(
            "Keep cells where both row and column are divisible "
            "by this value. 10 keeps roughly 1 percent."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=
            DEFAULT_MAX_SAMPLES,
        help=(
            "Hard safety cap. 0 disables the cap."
        ),
    )

    parser.add_argument(
        "--max-pe",
        type=float,
        default=
            DEFAULT_MAX_PE,
        help=(
            "Use -1 to disable PE filtering."
        ),
    )

    parser.add_argument(
        "--min-nc",
        type=int,
        default=
            DEFAULT_MIN_NC,
        help=(
            "Use -1 to disable NC filtering."
        ),
    )

    parser.add_argument(
        "--min-ns",
        type=int,
        default=
            DEFAULT_MIN_NS,
        help=(
            "Use -1 to disable NS filtering."
        ),
    )

    parser.add_argument(
        "--require-qf-positive",
        action="store_true",
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
        "VERSION 6 - GEDI L4B MEMORY-SAFE TRAINING LABEL BUILDER"
    )

    print(
        "=" * 72
    )

    args = parse_arguments()

    max_pe = (
        None
        if args.max_pe < 0
        else args.max_pe
    )

    min_nc = (
        None
        if args.min_nc < 0
        else args.min_nc
    )

    min_ns = (
        None
        if args.min_ns < 0
        else args.min_ns
    )

    if args.window_size <= 0:
        raise ValueError(
            "--window-size must be > 0."
        )

    if args.sample_every <= 0:
        raise ValueError(
            "--sample-every must be >= 1."
        )

    result = build_training_labels(
        data_dir=
            Path(
                args.gedi_dir
            ),

        output_dir=
            Path(
                args.output_dir
            ),

        window_size=
            args.window_size,

        max_pe=
            max_pe,

        min_nc=
            min_nc,

        min_ns=
            min_ns,

        require_qf_positive=
            args.require_qf_positive,

        sample_every=
            args.sample_every,

        max_samples=
            args.max_samples,
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "VERSION 6 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nGEDI training cells:\n"
        f"{result['training_cells']:,}"
    )

    print(
        f"\nCSV:\n"
        f"{result['csv_file']}"
    )

    print(
        f"\nSummary:\n"
        f"{result['summary_file']}"
    )

    print(
        "\nNext: forest/mangrove selection and "
        "predictor matching."
    )


if __name__ == "__main__":
    main()
