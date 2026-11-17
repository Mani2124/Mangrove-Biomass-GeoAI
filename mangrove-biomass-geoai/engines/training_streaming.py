
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window
from shapely.geometry import box
from shapely.ops import transform as shapely_transform


PROJECT_DIR = Path(r"D:\biomass_carbon_platform")
RUNS_DIR = PROJECT_DIR / "runs"

DEFAULT_MANIFEST = (
    PROJECT_DIR
    / "training"
    / "regions"
    / "training_region_manifest.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "training"
    / "streaming"
)

V1 = PROJECT_DIR / "v1_prepare_aoi.py"
V2 = PROJECT_DIR / "v2_process_sentinel2.py"
V3 = PROJECT_DIR / "v3_process_sentinel1.py"
V4 = PROJECT_DIR / "v4_process_terrain.py"

V5_CANDIDATES = [
    PROJECT_DIR / "v5_process_alos2.py",
    PROJECT_DIR / "v5_alos2.py",
]

DEFAULT_MAX_REGIONS = 15
DEFAULT_MIN_SEPARATION_DEG = 1.0
DEFAULT_GEDI_CELL_SIZE_M = 1000.8950233496674
DEFAULT_MIN_VALID_FRACTION = 0.20
DEFAULT_TIMEOUT_HOURS = 12

STATS = ["mean", "median", "std", "p25", "p75"]


def choose_dense_regions(
    manifest: pd.DataFrame,
    max_regions: int,
    min_separation_deg: float,
) -> pd.DataFrame:
    required = {
        "training_region_id",
        "aoi_file",
        "gedi_sample_file",
        "selected_gedi_cells",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise RuntimeError(
            "Training region manifest is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    frame = manifest.loc[
        manifest["selected_gedi_cells"] > 0
    ].copy()

    if frame.empty:
        raise RuntimeError("No regions contain GEDI training cells.")

    frame = frame.sort_values(
        "selected_gedi_cells",
        ascending=False,
    ).reset_index(drop=True)

    if max_regions <= 0:
        return frame

    if all(
        col in frame.columns
        for col in ["west", "east", "south", "north"]
    ):
        frame["center_lon"] = (
            frame["west"] + frame["east"]
        ) / 2.0
        frame["center_lat"] = (
            frame["south"] + frame["north"]
        ) / 2.0

        chosen = []

        for idx, row in frame.iterrows():
            if not chosen:
                chosen.append(idx)
            else:
                lon = float(row["center_lon"])
                lat = float(row["center_lat"])
                ok = True

                for chosen_idx in chosen:
                    other = frame.loc[chosen_idx]
                    dlon = lon - float(other["center_lon"])
                    dlat = lat - float(other["center_lat"])
                    distance = (dlon * dlon + dlat * dlat) ** 0.5

                    if distance < min_separation_deg:
                        ok = False
                        break

                if ok:
                    chosen.append(idx)

            if len(chosen) >= max_regions:
                break

        if len(chosen) < max_regions:
            for idx in frame.index:
                if idx not in chosen:
                    chosen.append(idx)
                if len(chosen) >= max_regions:
                    break

        return frame.loc[
            chosen[:max_regions]
        ].reset_index(drop=True)

    return frame.head(max_regions).copy()


def find_v5() -> Path | None:
    for path in V5_CANDIDATES:
        if path.exists():
            return path
    return None


def current_run_names() -> set[str]:
    if not RUNS_DIR.exists():
        return set()

    return {
        p.name
        for p in RUNS_DIR.iterdir()
        if p.is_dir()
    }


def find_new_run(before: set[str]) -> Path:
    candidates = [
        p
        for p in RUNS_DIR.iterdir()
        if p.is_dir() and p.name not in before
    ]

    if not candidates:
        raise RuntimeError(
            "V1 completed but no new run directory was detected."
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def run_command(
    command: list[str],
    log_file: Path,
    timeout_hours: float,
) -> None:
    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_file.open(
        "w",
        encoding="utf-8",
        errors="replace",
    ) as log:
        log.write(
            "COMMAND:\n"
            + subprocess.list2cmdline(command)
            + "\n\n"
        )
        log.flush()

        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert process.stdout is not None
        start = time.time()

        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()

                if (
                    time.time() - start
                    >
                    timeout_hours * 3600
                ):
                    process.kill()
                    raise TimeoutError(
                        "Stage exceeded timeout."
                    )

            return_code = process.wait()

        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            raise

        if return_code != 0:
            raise RuntimeError(
                f"Command failed with return code "
                f"{return_code}.\nSee log:\n{log_file}"
            )


def clean_feature_name(text: str) -> str:
    text = Path(text).stem
    text = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    )
    text = re.sub(
        r"_+",
        "_",
        text,
    )
    return text.strip("_")


def discover_predictor_rasters(
    run_dir: Path,
) -> dict[str, list[Path]]:
    modality_dirs = {
        "s2": run_dir / "processed" / "sentinel2",
        "s1": run_dir / "processed" / "sentinel1",
        "terrain": run_dir / "processed" / "terrain",
        "alos2": run_dir / "processed" / "alos2",
    }

    exclude_tokens = {
        "mask",
        "valid",
        "cloud",
        "qa",
        "preview",
        "rgb",
    }

    result = {}

    for modality, directory in modality_dirs.items():
        if not directory.exists():
            continue

        files = []

        for path in sorted(directory.glob("*.tif")):
            lower = path.name.lower()

            if any(
                token in lower
                for token in exclude_tokens
            ):
                continue

            files.append(path)

        if files:
            result[modality] = files

    return result


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


def safe_window(
    src,
    bounds,
) -> Window | None:
    try:
        window = from_bounds(
            *bounds,
            transform=src.transform,
        )
        window = (
            window
            .round_offsets()
            .round_lengths()
        )

        full = Window(
            0,
            0,
            src.width,
            src.height,
        )

        window = window.intersection(full)

    except Exception:
        return None

    if window.width <= 0 or window.height <= 0:
        return None

    return window


def read_values_in_cell(
    src,
    geometry,
) -> tuple[np.ndarray, int]:
    window = safe_window(
        src,
        geometry.bounds,
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
        [geometry],
        out_shape=data.shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )

    total = int(np.count_nonzero(inside))

    if total == 0:
        return np.array([], dtype="float32"), 0

    valid = inside & np.isfinite(data)

    if src.nodata is not None:
        valid &= data != src.nodata

    return (
        data[valid].astype(
            "float32",
            copy=False,
        ),
        total,
    )


def summarize_values(
    values: np.ndarray,
    total_pixels: int,
) -> dict:
    result = {
        "n": int(len(values)),
        "valid_fraction":
            float(len(values) / total_pixels)
            if total_pixels > 0
            else np.nan,
    }

    if len(values) == 0:
        for name in STATS:
            result[name] = np.nan
        return result

    result["mean"] = float(np.mean(values))
    result["median"] = float(np.median(values))
    result["std"] = float(np.std(values))
    result["p25"] = float(np.percentile(values, 25))
    result["p75"] = float(np.percentile(values, 75))
    return result


def aggregate_region(
    region_id: str,
    run_dir: Path,
    gedi_sample_file: Path,
    cell_size_m: float,
    min_valid_fraction: float,
) -> pd.DataFrame:
    if not gedi_sample_file.exists():
        raise FileNotFoundError(
            f"GEDI sample file missing:\n{gedi_sample_file}"
        )

    gedi = pd.read_csv(gedi_sample_file)

    required = {
        "x_6933_m",
        "y_6933_m",
        "longitude",
        "latitude",
        "agbd_mu_mg_ha",
    }

    missing = required - set(gedi.columns)

    if missing:
        raise RuntimeError(
            "GEDI sample file missing columns:\n"
            + "\n".join(sorted(missing))
        )

    raster_groups = discover_predictor_rasters(
        run_dir
    )

    required_modalities = {
        "s2",
        "s1",
        "terrain",
    }

    missing_modalities = (
        required_modalities
        -
        set(raster_groups)
    )

    if missing_modalities:
        raise RuntimeError(
            "Processed predictors missing: "
            + ", ".join(sorted(missing_modalities))
        )

    opened = {}

    try:
        for modality, paths in raster_groups.items():
            for path in paths:
                src = rasterio.open(path)

                if src.count != 1:
                    src.close()
                    continue

                key = (
                    f"{modality}__"
                    f"{clean_feature_name(path.name)}"
                )

                if key in opened:
                    suffix = 2
                    while f"{key}_{suffix}" in opened:
                        suffix += 1
                    key = f"{key}_{suffix}"

                opened[key] = {
                    "src": src,
                    "transformer":
                        Transformer.from_crs(
                            "EPSG:6933",
                            src.crs,
                            always_xy=True,
                        ),
                }

        if not opened:
            raise RuntimeError(
                "No predictor rasters could be opened."
            )

        records = []

        for index, row in gedi.iterrows():
            cell_6933 = gedi_cell_polygon_6933(
                row,
                cell_size_m,
            )

            record = row.to_dict()
            record["training_region_id"] = region_id

            valid_features = 0
            total_features = 0

            for feature_name, info in opened.items():
                projected_cell = shapely_transform(
                    info["transformer"].transform,
                    cell_6933,
                )

                values, total_pixels = read_values_in_cell(
                    info["src"],
                    projected_cell,
                )

                stats = summarize_values(
                    values,
                    total_pixels,
                )

                total_features += 1

                if (
                    np.isfinite(stats["valid_fraction"])
                    and
                    stats["valid_fraction"]
                    >=
                    min_valid_fraction
                ):
                    valid_features += 1

                for stat_name, value in stats.items():
                    record[
                        f"{feature_name}__{stat_name}"
                    ] = value

            record["predictor_feature_count"] = total_features
            record["predictor_valid_feature_count"] = valid_features
            record["predictor_valid_feature_fraction"] = (
                valid_features / total_features
                if total_features > 0
                else 0.0
            )

            record["model_sample_valid"] = (
                np.isfinite(
                    record["agbd_mu_mg_ha"]
                )
                and
                record["agbd_mu_mg_ha"] >= 0
                and
                record[
                    "predictor_valid_feature_fraction"
                ]
                >=
                0.50
            )

            records.append(record)

            if (
                index == 0
                or
                (index + 1) % 100 == 0
                or
                (index + 1) == len(gedi)
            ):
                print(
                    f"    GEDI "
                    f"{index + 1}/"
                    f"{len(gedi)}"
                )

        return pd.DataFrame(records)

    finally:
        for item in opened.values():
            try:
                item["src"].close()
            except Exception:
                pass


def append_training_rows(
    frame: pd.DataFrame,
    master_csv: Path,
):
    master_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not master_csv.exists()
        or
        master_csv.stat().st_size == 0
    )

    frame.to_csv(
        master_csv,
        mode="w" if write_header else "a",
        header=write_header,
        index=False,
    )


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0

    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass

    return total


def cleanup_region(run_dir: Path) -> dict:
    run_id = run_dir.name

    targets = [
        run_dir,
        Path(r"D:\s2_safe") / run_id,
        Path(r"D:\alos2_extract") / run_id,
    ]

    freed = 0
    removed = []

    for target in targets:
        if not target.exists():
            continue

        size = directory_size_bytes(target)

        try:
            shutil.rmtree(
                target,
                ignore_errors=False,
            )
            freed += size
            removed.append(str(target))

        except Exception as error:
            print(
                f"WARNING: could not remove "
                f"{target}:\n{error}"
            )

    return {
        "freed_bytes": freed,
        "removed": removed,
    }


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {"regions": {}}

    try:
        return json.loads(
            state_file.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {"regions": {}}


def save_state(
    state_file: Path,
    state: dict,
):
    state_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = state_file.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )

    temp.replace(state_file)


def process_region(
    row: pd.Series,
    python_exe: str,
    output_dir: Path,
    state: dict,
    state_file: Path,
    skip_alos2: bool,
    timeout_hours: float,
    cell_size_m: float,
    min_valid_fraction: float,
    cleanup_after_region: bool,
    cleanup_failed: bool,
) -> dict:
    region_id = str(
        row["training_region_id"]
    )

    aoi_file = Path(
        str(row["aoi_file"])
    )

    gedi_file = Path(
        str(row["gedi_sample_file"])
    )

    logs_dir = (
        output_dir
        /
        "logs"
        /
        region_id
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_csv = (
        output_dir
        /
        "biomass_training_dataset.csv"
    )

    region_state = (
        state["regions"]
        .setdefault(region_id, {})
    )

    if (
        region_state.get("status")
        ==
        "complete"
    ):
        print(
            f"{region_id}: already complete, skipping."
        )

        return {
            "training_region_id": region_id,
            "status": "already_complete",
            "rows_written":
                region_state.get(
                    "rows_written",
                    0,
                ),
        }

    run_dir = None

    try:
        before = current_run_names()

        print(
            f"\n[{region_id}] V1 AOI..."
        )

        run_command(
            [
                python_exe,
                str(V1),
                "--aoi",
                str(aoi_file),
            ],
            logs_dir / "v1.log",
            timeout_hours,
        )

        run_dir = find_new_run(before)

        region_state["run_dir"] = str(run_dir)
        region_state["status"] = "v1_complete"
        save_state(state_file, state)

        print(
            f"\n[{region_id}] V2 Sentinel-2..."
        )

        run_command(
            [
                python_exe,
                str(V2),
            ],
            logs_dir / "v2.log",
            timeout_hours,
        )

        region_state["status"] = "v2_complete"
        save_state(state_file, state)

        print(
            f"\n[{region_id}] V3 Sentinel-1..."
        )

        run_command(
            [
                python_exe,
                str(V3),
            ],
            logs_dir / "v3.log",
            timeout_hours,
        )

        region_state["status"] = "v3_complete"
        save_state(state_file, state)

        print(
            f"\n[{region_id}] V4 Terrain..."
        )

        run_command(
            [
                python_exe,
                str(V4),
            ],
            logs_dir / "v4.log",
            timeout_hours,
        )

        region_state["status"] = "v4_complete"
        save_state(state_file, state)

        if not skip_alos2:
            v5 = find_v5()

            if v5 is not None:
                print(
                    f"\n[{region_id}] V5 ALOS-2..."
                )

                run_command(
                    [
                        python_exe,
                        str(v5),
                        "--run-dir",
                        str(run_dir),
                    ],
                    logs_dir / "v5.log",
                    timeout_hours,
                )

        print(
            f"\n[{region_id}] "
            f"Aggregating predictors into GEDI cells..."
        )

        region_table = aggregate_region(
            region_id=region_id,
            run_dir=run_dir,
            gedi_sample_file=gedi_file,
            cell_size_m=cell_size_m,
            min_valid_fraction=min_valid_fraction,
        )

        if region_table.empty:
            raise RuntimeError(
                "Aggregation produced no GEDI rows."
            )

        region_table = region_table.loc[
            np.isfinite(
                region_table["agbd_mu_mg_ha"]
            )
        ].copy()

        rows_written = len(region_table)

        append_training_rows(
            region_table,
            master_csv,
        )

        region_state["status"] = "complete"
        region_state["rows_written"] = int(rows_written)
        region_state["model_ready_rows"] = int(
            region_table[
                "model_sample_valid"
            ].sum()
        )

        save_state(state_file, state)

        cleanup_result = {
            "freed_bytes": 0,
            "removed": [],
        }

        if cleanup_after_region:
            print(
                f"\n[{region_id}] "
                f"Cleaning temporary regional files..."
            )

            cleanup_result = cleanup_region(
                run_dir
            )

            freed_gb = (
                cleanup_result["freed_bytes"]
                /
                1024**3
            )

            print(
                f"Freed approximately "
                f"{freed_gb:.2f} GB"
            )

            region_state[
                "cleanup_complete"
            ] = True

            region_state[
                "freed_gb"
            ] = float(freed_gb)

            save_state(
                state_file,
                state,
            )

        return {
            "training_region_id": region_id,
            "status": "complete",
            "rows_written": rows_written,
            "model_ready_rows": int(
                region_table[
                    "model_sample_valid"
                ].sum()
            ),
            "freed_gb":
                cleanup_result[
                    "freed_bytes"
                ]
                /
                1024**3,
        }

    except Exception as error:
        region_state["status"] = "failed"
        region_state["error"] = str(error)

        save_state(
            state_file,
            state,
        )

        if (
            cleanup_failed
            and
            run_dir is not None
        ):
            print(
                f"\n[{region_id}] "
                f"Cleaning failed regional run..."
            )
            cleanup_region(run_dir)

        raise


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Low-disk streaming biomass training pipeline."
        )
    )

    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=DEFAULT_MAX_REGIONS,
    )

    parser.add_argument(
        "--min-separation-deg",
        type=float,
        default=DEFAULT_MIN_SEPARATION_DEG,
    )

    parser.add_argument(
        "--skip-alos2",
        action="store_true",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep regional run files.",
    )

    parser.add_argument(
        "--cleanup-failed",
        action="store_true",
        help="Delete failed temporary regional runs too.",
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
    )

    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=DEFAULT_TIMEOUT_HOURS,
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

    return parser.parse_args()


def main():
    print("\n" + "=" * 72)
    print("BIOMASS AND CARBON PLATFORM")
    print("STREAMING TRAINING - LOW DISK MODE")
    print("=" * 72)

    args = parse_arguments()

    manifest_file = Path(args.manifest)
    output_dir = Path(args.output_dir)

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_file}"
        )

    for required_script in [V1, V2, V3, V4]:
        if not required_script.exists():
            raise FileNotFoundError(
                f"Required script missing:\n"
                f"{required_script}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_file = (
        output_dir
        /
        "streaming_state.json"
    )

    report_file = (
        output_dir
        /
        "streaming_region_report.csv"
    )

    selected_file = (
        output_dir
        /
        "selected_dense_regions.csv"
    )

    master_csv = (
        output_dir
        /
        "biomass_training_dataset.csv"
    )

    print(
        "\n[1/4] Reading regional manifest..."
    )

    manifest = pd.read_csv(
        manifest_file
    )

    selected = choose_dense_regions(
        manifest=manifest,
        max_regions=args.max_regions,
        min_separation_deg=args.min_separation_deg,
    )

    selected.to_csv(
        selected_file,
        index=False,
    )

    print(
        f"Selected dense regions: "
        f"{len(selected):,}"
    )

    print(
        f"Potential GEDI cells: "
        f"{int(selected['selected_gedi_cells'].sum()):,}"
    )

    print(
        "\n[2/4] Processing one region at a time..."
    )

    state = load_state(
        state_file
    )

    results = []

    for index, row in selected.iterrows():
        region_id = str(
            row["training_region_id"]
        )

        print("\n" + "-" * 72)
        print(
            f"REGION "
            f"{index + 1}/"
            f"{len(selected)}: "
            f"{region_id}"
        )
        print(
            f"GEDI cells: "
            f"{int(row['selected_gedi_cells'])}"
        )
        print("-" * 72)

        try:
            result = process_region(
                row=row,
                python_exe=args.python,
                output_dir=output_dir,
                state=state,
                state_file=state_file,
                skip_alos2=args.skip_alos2,
                timeout_hours=args.timeout_hours,
                cell_size_m=args.gedi_cell_size_m,
                min_valid_fraction=args.min_valid_fraction,
                cleanup_after_region=
                    not args.no_cleanup,
                cleanup_failed=
                    args.cleanup_failed,
            )

        except KeyboardInterrupt:
            save_state(
                state_file,
                state,
            )
            print(
                "\nStopped by user. "
                "Completed training rows are safe."
            )
            raise

        except Exception as error:
            print(
                f"\nFAILED REGION "
                f"{region_id}:\n"
                f"{error}"
            )

            result = {
                "training_region_id":
                    region_id,
                "status":
                    "failed",
                "error":
                    str(error),
            }

        results.append(result)

        pd.DataFrame(
            results
        ).to_csv(
            report_file,
            index=False,
        )

    print(
        "\n[3/4] Checking master training dataset..."
    )

    if master_csv.exists():
        training = pd.read_csv(
            master_csv
        )
        total_rows = len(training)

        model_ready = (
            int(
                training[
                    "model_sample_valid"
                ].sum()
            )
            if
            "model_sample_valid"
            in
            training.columns
            else
            0
        )

    else:
        total_rows = 0
        model_ready = 0

    print(
        f"Training rows: "
        f"{total_rows:,}"
    )

    print(
        f"Model-ready rows: "
        f"{model_ready:,}"
    )

    print(
        "\n[4/4] Saving summary..."
    )

    summary = {
        "mode":
            "streaming_low_disk",

        "selected_regions":
            int(len(selected)),

        "potential_gedi_cells":
            int(
                selected[
                    "selected_gedi_cells"
                ].sum()
            ),

        "training_rows":
            int(total_rows),

        "model_ready_rows":
            int(model_ready),

        "cleanup_after_each_successful_region":
            bool(
                not args.no_cleanup
            ),

        "master_training_csv":
            str(master_csv),

        "state":
            str(state_file),

        "region_report":
            str(report_file),

        "next_step":
            (
                "Spatial train/validation/test split and "
                "Random Forest biomass baseline."
            ),
    }

    (
        output_dir
        /
        "streaming_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("STREAMING TRAINING COMPLETE")
    print("=" * 72)

    print(
        f"\nTraining CSV:\n"
        f"{master_csv}"
    )

    print(
        f"\nTraining rows:\n"
        f"{total_rows:,}"
    )

    print(
        f"\nModel-ready rows:\n"
        f"{model_ready:,}"
    )

    print(
        "\nTemporary satellite data were deleted "
        "after each successful region."
    )


if __name__ == "__main__":
    main()
