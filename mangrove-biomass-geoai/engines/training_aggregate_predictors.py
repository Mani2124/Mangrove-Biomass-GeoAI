
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_PROCESSING_RESULTS = (
    PROJECT_DIR
    / "training"
    / "regional_processing"
    / "regional_processing_results.csv"
)

DEFAULT_SELECTED_MANIFEST = (
    PROJECT_DIR
    / "training"
    / "regional_processing"
    / "selected_training_regions.csv"
)

DEFAULT_PROCESSING_STATE = (
    PROJECT_DIR
    / "training"
    / "regional_processing"
    / "processing_state.json"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "training"
    / "training_table"
)

DEFAULT_GEDI_CELL_SIZE_M = 1000.8950233496674
DEFAULT_STATS = ["mean", "median", "std", "p25", "p75"]
DEFAULT_MIN_VALID_FRACTION = 0.20


def clean_feature_name(text: str) -> str:
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def discover_modality_rasters(run_dir: Path) -> dict[str, list[Path]]:
    modality_dirs = {
        "s2": run_dir / "processed" / "sentinel2",
        "s1": run_dir / "processed" / "sentinel1",
        "terrain": run_dir / "processed" / "terrain",
        "alos2": run_dir / "processed" / "alos2",
    }

    exclude_tokens = {"mask", "valid", "cloud", "qa", "preview", "rgb"}
    result = {}

    for modality, directory in modality_dirs.items():
        if not directory.exists():
            continue

        rasters = []
        for path in sorted(directory.glob("*.tif")):
            lower = path.name.lower()
            if any(token in lower for token in exclude_tokens):
                continue
            rasters.append(path)

        if rasters:
            result[modality] = rasters

    return result


def read_processing_results(
    results_file: Path,
    selected_manifest_file: Path,
    state_file: Path,
) -> pd.DataFrame:
    """
    Read completed Training 03 regions.

    First tries regional_processing_results.csv.
    If Training 03 stopped before writing that file, falls back to
    processing_state.json, which is written after each completed stage.
    """
    if not selected_manifest_file.exists():
        raise FileNotFoundError(
            f"Selected-region manifest not found:\n{selected_manifest_file}"
        )

    selected = pd.read_csv(selected_manifest_file)

    required_selected = {
        "training_region_id",
        "gedi_sample_file",
    }

    missing = required_selected - set(selected.columns)

    if missing:
        raise RuntimeError(
            "Selected manifest missing columns:\n"
            + "\n".join(sorted(missing))
        )

    if results_file.exists():
        print(
            f"Using regional processing results:\n{results_file}"
        )

        results = pd.read_csv(results_file)

        if "region_id" in results.columns:
            results = results.rename(
                columns={
                    "region_id":
                        "training_region_id"
                }
            )

        required_results = {
            "training_region_id",
            "status",
            "run_dir",
        }

        missing = required_results - set(results.columns)

        if missing:
            raise RuntimeError(
                "Processing results missing columns:\n"
                + "\n".join(sorted(missing))
            )

        combined = results.merge(
            selected[
                [
                    "training_region_id",
                    "gedi_sample_file",
                ]
            ],
            on="training_region_id",
            how="left",
        )

        successful = combined.loc[
            combined["status"]
            ==
            "predictors_processed"
        ].copy()

        return successful.reset_index(drop=True)

    print(
        "regional_processing_results.csv was not found."
    )
    print(
        "Training 03 stopped before finalizing, so "
        "processing_state.json will be used."
    )

    if not state_file.exists():
        raise FileNotFoundError(
            "Neither Training 03 results nor state file exists.\n\n"
            f"Results:\n{results_file}\n\n"
            f"State:\n{state_file}"
        )

    state = json.loads(
        state_file.read_text(
            encoding="utf-8"
        )
    )

    region_states = state.get("regions", {})

    selected_lookup = selected.set_index(
        "training_region_id"
    )

    records = []

    for region_id, info in region_states.items():
        if info.get("status") != "predictors_processed":
            continue

        run_dir = info.get("run_dir")

        if not run_dir:
            continue

        run_dir = Path(run_dir)

        if not run_dir.exists():
            continue

        if region_id not in selected_lookup.index:
            continue

        gedi_sample_file = selected_lookup.loc[
            region_id,
            "gedi_sample_file",
        ]

        records.append(
            {
                "training_region_id":
                    region_id,
                "status":
                    "predictors_processed",
                "run_dir":
                    str(run_dir),
                "gedi_sample_file":
                    str(gedi_sample_file),
            }
        )

    if not records:
        raise RuntimeError(
            "processing_state.json was found, but no completed "
            "regions with status='predictors_processed' were recovered."
        )

    successful = pd.DataFrame(records)

    print(
        f"Recovered completed regions from state: "
        f"{len(successful):,}"
    )

    return successful.reset_index(drop=True)

def gedi_cell_polygon_6933(
    row: pd.Series,
    cell_size_m: float,
):
    x = float(row["x_6933_m"])
    y = float(row["y_6933_m"])
    half = cell_size_m / 2.0
    return box(
        x - half,
        y - half,
        x + half,
        y + half,
    )


def project_geometry(geometry, transformer: Transformer):
    return shapely_transform(
        transformer.transform,
        geometry,
    )


def safe_window(src, bounds) -> Window | None:
    try:
        window = from_bounds(
            *bounds,
            transform=src.transform,
        )
    except Exception:
        return None

    window = window.round_offsets().round_lengths()

    full = Window(
        0,
        0,
        src.width,
        src.height,
    )

    try:
        window = window.intersection(full)
    except Exception:
        return None

    if window.width <= 0 or window.height <= 0:
        return None

    return window


def extract_values(
    src,
    geometry_in_src_crs,
) -> tuple[np.ndarray, int]:
    window = safe_window(
        src,
        geometry_in_src_crs.bounds,
    )

    if window is None:
        return np.array([], dtype="float32"), 0

    data = src.read(
        1,
        window=window,
        masked=False,
    ).astype("float32")

    if data.size == 0:
        return np.array([], dtype="float32"), 0

    transform = rasterio.windows.transform(
        window,
        src.transform,
    )

    inside = geometry_mask(
        [geometry_in_src_crs],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    total_inside = int(np.count_nonzero(inside))
    if total_inside == 0:
        return np.array([], dtype="float32"), 0

    valid = inside & np.isfinite(data)

    if src.nodata is not None:
        valid &= data != src.nodata

    values = data[valid]

    return values.astype("float32", copy=False), total_inside


def calculate_statistics(
    values: np.ndarray,
    total_pixels: int,
    requested_stats: list[str],
) -> dict[str, float]:
    result = {
        "valid_fraction":
            float(len(values) / total_pixels)
            if total_pixels > 0
            else np.nan,
        "n": int(len(values)),
    }

    if len(values) == 0:
        for stat in requested_stats:
            result[stat] = np.nan
        return result

    if "mean" in requested_stats:
        result["mean"] = float(np.mean(values))
    if "median" in requested_stats:
        result["median"] = float(np.median(values))
    if "std" in requested_stats:
        result["std"] = float(np.std(values))
    if "p25" in requested_stats:
        result["p25"] = float(np.percentile(values, 25))
    if "p75" in requested_stats:
        result["p75"] = float(np.percentile(values, 75))

    return result


def process_one_region(
    region_id: str,
    run_dir: Path,
    gedi_sample_file: Path,
    output_dir: Path,
    cell_size_m: float,
    stats: list[str],
    min_valid_fraction: float,
) -> dict:
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory missing:\n{run_dir}"
        )

    if not gedi_sample_file.exists():
        raise FileNotFoundError(
            f"GEDI sample file missing:\n{gedi_sample_file}"
        )

    rasters = discover_modality_rasters(run_dir)

    required_modalities = {"s2", "s1", "terrain"}
    missing = required_modalities - set(rasters)
    if missing:
        raise RuntimeError(
            f"{region_id}: missing processed modalities: "
            + ", ".join(sorted(missing))
        )

    gedi = pd.read_csv(gedi_sample_file)

    required_gedi = {
        "x_6933_m",
        "y_6933_m",
        "longitude",
        "latitude",
        "agbd_mu_mg_ha",
    }

    missing = required_gedi - set(gedi.columns)
    if missing:
        raise RuntimeError(
            f"{region_id}: GEDI file missing columns: "
            + ", ".join(sorted(missing))
        )

    region_output = (
        output_dir
        / "regions"
        / f"{region_id}_training_table.csv"
    )

    region_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    open_datasets = {}

    try:
        for modality, files in rasters.items():
            for path in files:
                key = f"{modality}__{clean_feature_name(path.name)}"

                if key in open_datasets:
                    suffix = 2
                    candidate = f"{key}_{suffix}"
                    while candidate in open_datasets:
                        suffix += 1
                        candidate = f"{key}_{suffix}"
                    key = candidate

                src = rasterio.open(path)

                if src.count != 1:
                    src.close()
                    continue

                transformer = Transformer.from_crs(
                    "EPSG:6933",
                    src.crs,
                    always_xy=True,
                )

                open_datasets[key] = {
                    "src": src,
                    "transformer": transformer,
                    "source_file": str(path),
                    "modality": modality,
                }

        records = []

        for sample_index, row in gedi.iterrows():
            polygon_6933 = gedi_cell_polygon_6933(
                row,
                cell_size_m,
            )

            record = row.to_dict()
            record["training_region_id"] = region_id

            valid_feature_count = 0
            total_feature_count = 0

            for feature_key, info in open_datasets.items():
                src = info["src"]

                geometry = project_geometry(
                    polygon_6933,
                    info["transformer"],
                )

                values, total_inside = extract_values(
                    src,
                    geometry,
                )

                feature_stats = calculate_statistics(
                    values,
                    total_inside,
                    stats,
                )

                valid_fraction = feature_stats["valid_fraction"]
                total_feature_count += 1

                if (
                    np.isfinite(valid_fraction)
                    and valid_fraction >= min_valid_fraction
                ):
                    valid_feature_count += 1

                for stat_name, value in feature_stats.items():
                    record[
                        f"{feature_key}__{stat_name}"
                    ] = value

            record["predictor_feature_count"] = total_feature_count
            record["predictor_valid_feature_count"] = valid_feature_count
            record["predictor_valid_feature_fraction"] = (
                valid_feature_count / total_feature_count
                if total_feature_count > 0
                else 0.0
            )

            records.append(record)

            if (
                sample_index == 0
                or (sample_index + 1) % 100 == 0
                or (sample_index + 1) == len(gedi)
            ):
                print(
                    f"    GEDI cell "
                    f"{sample_index + 1}/{len(gedi)}"
                )

        table = pd.DataFrame(records)
        table.to_csv(region_output, index=False)

    finally:
        for info in open_datasets.values():
            try:
                info["src"].close()
            except Exception:
                pass

    return {
        "region_id": region_id,
        "samples": len(table),
        "features": len(open_datasets),
        "output_file": region_output,
        "alos2_available": "alos2" in rasters,
    }


def build_training_table(
    processing_results: Path,
    selected_manifest: Path,
    processing_state: Path,
    output_dir: Path,
    cell_size_m: float,
    stats: list[str],
    min_valid_fraction: float,
    max_regions: int,
    resume: bool,
) -> dict:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n[1/6] Reading successful regional processing results..."
    )

    successful = read_processing_results(
        processing_results,
        selected_manifest,
        processing_state,
    )

    if max_regions > 0:
        successful = successful.iloc[:max_regions].copy()

    if successful.empty:
        raise RuntimeError(
            "No successfully processed regions were found."
        )

    print(
        f"Successful regions available: "
        f"{len(successful):,}"
    )

    print(
        "\n[2/6] Aggregating raster predictors "
        "inside GEDI ~1-km cells..."
    )

    region_results = []

    for index, row in successful.iterrows():
        region_id = str(row["training_region_id"])
        run_dir = Path(str(row["run_dir"]))
        gedi_sample_file = Path(str(row["gedi_sample_file"]))

        expected_output = (
            output_dir
            / "regions"
            / f"{region_id}_training_table.csv"
        )

        print("\n" + "-" * 72)
        print(
            f"REGION {index + 1}/{len(successful)}: "
            f"{region_id}"
        )
        print("-" * 72)

        if (
            resume
            and expected_output.exists()
            and expected_output.stat().st_size > 0
        ):
            region_results.append(
                {
                    "region_id": region_id,
                    "status": "reused",
                    "output_file": str(expected_output),
                }
            )

            print(
                "    Existing regional training table reused."
            )
            continue

        try:
            result = process_one_region(
                region_id=region_id,
                run_dir=run_dir,
                gedi_sample_file=gedi_sample_file,
                output_dir=output_dir,
                cell_size_m=cell_size_m,
                stats=stats,
                min_valid_fraction=min_valid_fraction,
            )

            region_results.append(
                {
                    **result,
                    "status": "complete",
                }
            )

        except Exception as error:
            region_results.append(
                {
                    "region_id": region_id,
                    "status": "failed",
                    "error": str(error),
                }
            )

            print(
                f"FAILED: {error}"
            )

    print(
        "\n[3/6] Combining completed regional tables..."
    )

    completed_files = []

    for result in region_results:
        if result["status"] not in {"complete", "reused"}:
            continue

        path = Path(str(result["output_file"]))

        if path.exists():
            completed_files.append(path)

    if not completed_files:
        raise RuntimeError(
            "No regional training tables were produced."
        )

    frames = [
        pd.read_csv(path)
        for path in completed_files
    ]

    training = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    print(
        f"Combined GEDI samples: "
        f"{len(training):,}"
    )

    print(
        "\n[4/6] Creating model-ready quality flags..."
    )

    target = training["agbd_mu_mg_ha"]

    target_valid = (
        np.isfinite(target)
        &
        (target >= 0)
    )

    predictor_fraction = (
        training["predictor_valid_feature_fraction"]
        if "predictor_valid_feature_fraction" in training.columns
        else pd.Series(0.0, index=training.index)
    )

    training["model_sample_valid"] = (
        target_valid
        &
        (predictor_fraction >= 0.50)
    )

    valid_count = int(
        training["model_sample_valid"].sum()
    )

    print(
        f"Model-ready samples: "
        f"{valid_count:,}"
    )

    print(
        "\n[5/6] Saving final training table..."
    )

    final_csv = (
        output_dir
        / "biomass_training_table.csv"
    )

    training.to_csv(
        final_csv,
        index=False,
    )

    parquet_file = (
        output_dir
        / "biomass_training_table.parquet"
    )

    parquet_saved = False

    try:
        training.to_parquet(
            parquet_file,
            index=False,
        )
        parquet_saved = True
    except Exception as error:
        print(
            f"Parquet skipped (optional): "
            f"{error}"
        )

    region_report = pd.DataFrame(region_results)

    region_report_file = (
        reports_dir
        / "training_04_region_report.csv"
    )

    region_report.to_csv(
        region_report_file,
        index=False,
    )

    print(
        "\n[6/6] Saving summary..."
    )

    summary = {
        "purpose": (
            "Aggregate processed satellite and terrain "
            "predictors inside GEDI L4B ~1-km training cells."
        ),
        "processing_results": str(processing_results),
        "selected_manifest": str(selected_manifest),
        "gedi_cell_size_m": cell_size_m,
        "statistics": stats,
        "minimum_feature_valid_fraction":
            min_valid_fraction,
        "successful_or_reused_regions":
            int(len(completed_files)),
        "combined_samples":
            int(len(training)),
        "model_ready_samples":
            valid_count,
        "csv":
            str(final_csv),
        "parquet":
            str(parquet_file)
            if parquet_saved
            else None,
        "region_report":
            str(region_report_file),
        "next_step": (
            "Create spatial train/validation/test splits and "
            "train the Random Forest biomass baseline."
        ),
    }

    summary_file = (
        reports_dir
        / "training_04_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "regions": len(completed_files),
        "samples": len(training),
        "valid_samples": valid_count,
        "csv_file": final_csv,
        "parquet_file":
            parquet_file
            if parquet_saved
            else None,
        "summary_file": summary_file,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Training 04: aggregate Sentinel-2, Sentinel-1, "
            "terrain and optional ALOS-2 statistics inside "
            "GEDI L4B ~1-km training cells."
        )
    )

    parser.add_argument(
        "--processing-results",
        default=str(DEFAULT_PROCESSING_RESULTS),
    )

    parser.add_argument(
        "--selected-manifest",
        default=str(DEFAULT_SELECTED_MANIFEST),
    )

    parser.add_argument(
        "--processing-state",
        default=str(DEFAULT_PROCESSING_STATE),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--gedi-cell-size-m",
        type=float,
        default=DEFAULT_GEDI_CELL_SIZE_M,
    )

    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=DEFAULT_MIN_VALID_FRACTION,
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=0,
        help="0 means every successfully processed region.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def main():
    print("\n" + "=" * 72)
    print("BIOMASS AND CARBON PLATFORM")
    print("TRAINING 04 - GEDI / PREDICTOR AGGREGATION")
    print("=" * 72)

    args = parse_arguments()

    result = build_training_table(
        processing_results=
            Path(args.processing_results),
        selected_manifest=
            Path(args.selected_manifest),
        processing_state=
            Path(args.processing_state),
        output_dir=
            Path(args.output_dir),
        cell_size_m=
            args.gedi_cell_size_m,
        stats=
            DEFAULT_STATS,
        min_valid_fraction=
            args.min_valid_fraction,
        max_regions=
            args.max_regions,
        resume=
            args.resume,
    )

    print("\n" + "=" * 72)
    print("TRAINING 04 COMPLETE")
    print("=" * 72)

    print(
        f"\nRegions used:\n"
        f"{result['regions']:,}"
    )

    print(
        f"\nCombined GEDI samples:\n"
        f"{result['samples']:,}"
    )

    print(
        f"\nModel-ready samples:\n"
        f"{result['valid_samples']:,}"
    )

    print(
        f"\nTraining CSV:\n"
        f"{result['csv_file']}"
    )

    if result["parquet_file"] is not None:
        print(
            f"\nTraining Parquet:\n"
            f"{result['parquet_file']}"
        )

    print(
        "\nNext: spatial split + Random Forest biomass baseline."
    )


if __name__ == "__main__":
    main()
