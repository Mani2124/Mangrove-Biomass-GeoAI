
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd


PROJECT_DIR = Path(r"D:\biomass_carbon_platform")
DEFAULT_MANIFEST = PROJECT_DIR / "training" / "regions" / "training_region_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "training" / "regional_processing"

# Existing pipeline scripts
V1 = PROJECT_DIR / "v1_prepare_aoi.py"
V2 = PROJECT_DIR / "v2_process_sentinel2.py"
V3 = PROJECT_DIR / "v3_process_sentinel1.py"
V4 = PROJECT_DIR / "v4_process_terrain.py"
V5_CANDIDATES = [
    PROJECT_DIR / "v5_process_alos2.py",
    PROJECT_DIR / "v5_alos2.py",
]

DEFAULT_MAX_REGIONS = 25
DEFAULT_TIMEOUT_HOURS = 12


def find_v5() -> Path | None:
    for path in V5_CANDIDATES:
        if path.exists():
            return path
    return None


def newest_run(runs_dir: Path, before: set[str] | None = None) -> Path | None:
    if not runs_dir.exists():
        return None

    candidates = [
        p for p in runs_dir.iterdir()
        if p.is_dir()
    ]

    if before is not None:
        new_candidates = [
            p for p in candidates
            if p.name not in before
        ]
        if new_candidates:
            candidates = new_candidates

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime,
    )


def existing_run_names(runs_dir: Path) -> set[str]:
    if not runs_dir.exists():
        return set()

    return {
        p.name
        for p in runs_dir.iterdir()
        if p.is_dir()
    }


def run_command(
    command: list[str],
    log_file: Path,
    timeout_hours: float,
    cwd: Path,
) -> tuple[bool, int, float]:
    start = time.time()

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

        try:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            assert process.stdout is not None

            for line in process.stdout:
                print(
                    line,
                    end="",
                )
                log.write(
                    line
                )
                log.flush()

            return_code = process.wait(
                timeout=max(
                    1,
                    int(
                        timeout_hours
                        *
                        3600
                    ),
                )
            )

        except subprocess.TimeoutExpired:
            process.kill()
            return_code = -999
            log.write(
                "\nPROCESS TIMEOUT\n"
            )

        except KeyboardInterrupt:
            try:
                process.terminate()
            except Exception:
                pass
            raise

        except Exception as exc:
            log.write(
                f"\nERROR:\n{exc}\n"
            )
            return_code = -998

    elapsed = (
        time.time()
        -
        start
    )

    return (
        return_code == 0,
        return_code,
        elapsed,
    )


def load_state(state_file: Path) -> dict:
    if not state_file.exists():
        return {
            "regions": {}
        }

    try:
        return json.loads(
            state_file.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {
            "regions": {}
        }


def save_state(
    state_file: Path,
    state: dict,
):
    state_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = state_file.with_suffix(
        ".tmp"
    )

    tmp.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )

    tmp.replace(
        state_file
    )


def validate_scripts(
    skip_s1: bool,
    skip_terrain: bool,
):
    required = [
        V1,
        V2,
    ]

    if not skip_s1:
        required.append(
            V3
        )

    if not skip_terrain:
        required.append(
            V4
        )

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required pipeline scripts are missing:\n"
            +
            "\n".join(
                missing
            )
        )


def choose_regions(
    manifest: pd.DataFrame,
    max_regions: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select geographically diverse regions rather than simply taking
    the first 25 rows of the manifest.

    The manifest's region IDs are global 2-degree bins. We sort by
    latitude/longitude and then sample approximately evenly through
    that ordering.
    """
    required = {
        "training_region_id",
        "aoi_file",
        "gedi_sample_file",
        "selected_gedi_cells",
    }

    missing = (
        required
        -
        set(
            manifest.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Manifest is missing required columns:\n"
            +
            "\n".join(
                sorted(
                    missing
                )
            )
        )

    usable = manifest.copy()

    usable = usable[
        usable[
            "selected_gedi_cells"
        ]
        >
        0
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "Manifest contains no usable training regions."
        )

    # Prefer geographic columns if available.
    sort_columns = [
        col
        for col in [
            "south",
            "west",
            "training_region_id",
        ]
        if col in usable.columns
    ]

    usable = usable.sort_values(
        sort_columns
    ).reset_index(
        drop=True
    )

    if (
        max_regions <= 0
        or
        len(usable) <= max_regions
    ):
        return usable

    # Evenly distributed positions across the globally sorted list.
    positions = np.linspace(
        0,
        len(usable) - 1,
        max_regions,
    )

    positions = np.unique(
        np.round(
            positions
        ).astype(int)
    )

    selected = usable.iloc[
        positions
    ].copy()

    # If rounding ever produces fewer rows, fill deterministically.
    if len(selected) < max_regions:
        remaining = usable.drop(
            index=selected.index,
            errors="ignore",
        )

        needed = (
            max_regions
            -
            len(selected)
        )

        if needed > 0:
            extra = remaining.sample(
                n=min(
                    needed,
                    len(remaining),
                ),
                random_state=seed,
            )

            selected = pd.concat(
                [
                    selected,
                    extra,
                ]
            )

    return selected.reset_index(
        drop=True
    )


def write_selected_manifest(
    selected: pd.DataFrame,
    output_file: Path,
):
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        output_file,
        index=False,
    )


def mark_stage(
    region_state: dict,
    stage: str,
    status: str,
    **kwargs,
):
    region_state.setdefault(
        "stages",
        {}
    )

    region_state[
        "stages"
    ][stage] = {
        "status":
            status,
        **kwargs,
    }


def stage_complete(
    region_state: dict,
    stage: str,
) -> bool:
    return (
        region_state
        .get(
            "stages",
            {}
        )
        .get(
            stage,
            {}
        )
        .get(
            "status"
        )
        ==
        "complete"
    )


def process_region(
    row: pd.Series,
    python_exe: str,
    output_dir: Path,
    state: dict,
    state_file: Path,
    resume: bool,
    skip_s1: bool,
    skip_terrain: bool,
    skip_alos2: bool,
    timeout_hours: float,
) -> dict:
    region_id = str(
        row[
            "training_region_id"
        ]
    )

    aoi_file = Path(
        str(
            row[
                "aoi_file"
            ]
        )
    )

    sample_file = Path(
        str(
            row[
                "gedi_sample_file"
            ]
        )
    )

    if not aoi_file.exists():
        raise FileNotFoundError(
            f"AOI missing for {region_id}:\n{aoi_file}"
        )

    if not sample_file.exists():
        raise FileNotFoundError(
            f"GEDI sample file missing for {region_id}:\n{sample_file}"
        )

    region_dir = (
        output_dir
        /
        region_id
    )

    logs_dir = (
        region_dir
        /
        "logs"
    )

    region_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    region_state = (
        state[
            "regions"
        ]
        .setdefault(
            region_id,
            {
                "aoi_file":
                    str(
                        aoi_file
                    ),

                "gedi_sample_file":
                    str(
                        sample_file
                    ),

                "stages":
                    {},
            }
        )
    )

    runs_dir = (
        PROJECT_DIR
        /
        "runs"
    )

    # ---------------------------------------------------------
    # V1
    # ---------------------------------------------------------
    if (
        resume
        and
        stage_complete(
            region_state,
            "v1"
        )
        and
        region_state.get(
            "run_dir"
        )
        and
        Path(
            region_state[
                "run_dir"
            ]
        ).exists()
    ):
        run_dir = Path(
            region_state[
                "run_dir"
            ]
        )

        print(
            f"[{region_id}] V1 already complete -> "
            f"{run_dir.name}"
        )

    else:
        before = existing_run_names(
            runs_dir
        )

        command = [
            python_exe,
            str(
                V1
            ),
            "--aoi",
            str(
                aoi_file
            ),
        ]

        print(
            f"\n[{region_id}] Running V1..."
        )

        ok, code, elapsed = run_command(
            command,
            logs_dir / "v1.log",
            timeout_hours,
            PROJECT_DIR,
        )

        if not ok:
            mark_stage(
                region_state,
                "v1",
                "failed",
                return_code=code,
                elapsed_seconds=elapsed,
            )
            save_state(
                state_file,
                state,
            )
            return {
                "region_id":
                    region_id,
                "status":
                    "failed_v1",
            }

        run_dir = newest_run(
            runs_dir,
            before=before,
        )

        if run_dir is None:
            mark_stage(
                region_state,
                "v1",
                "failed",
                reason=(
                    "V1 completed but no new run directory was found."
                ),
            )
            save_state(
                state_file,
                state,
            )
            return {
                "region_id":
                    region_id,
                "status":
                    "failed_run_detection",
            }

        region_state[
            "run_dir"
        ] = str(
            run_dir
        )

        mark_stage(
            region_state,
            "v1",
            "complete",
            elapsed_seconds=elapsed,
            run_dir=str(
                run_dir
            ),
        )

        save_state(
            state_file,
            state,
        )

    # ---------------------------------------------------------
    # IMPORTANT
    # Existing V2/V3/V4 scripts in this project discover the latest
    # run. We therefore execute each region sequentially. Do NOT run
    # multiple copies of this orchestrator simultaneously.
    # ---------------------------------------------------------

    stages = [
        (
            "v2",
            V2,
            False,
        ),
        (
            "v3",
            V3,
            skip_s1,
        ),
        (
            "v4",
            V4,
            skip_terrain,
        ),
    ]

    if not skip_alos2:
        v5 = find_v5()

        if v5 is not None:
            stages.append(
                (
                    "v5",
                    v5,
                    False,
                )
            )
        else:
            print(
                f"[{region_id}] ALOS-2 script not found; "
                "V5 skipped."
            )

            mark_stage(
                region_state,
                "v5",
                "skipped",
                reason="ALOS-2 script not found",
            )

    else:
        mark_stage(
            region_state,
            "v5",
            "skipped",
            reason="--skip-alos2",
        )

    for stage_name, script, skip in stages:
        if skip:
            mark_stage(
                region_state,
                stage_name,
                "skipped",
            )
            save_state(
                state_file,
                state,
            )
            continue

        if (
            resume
            and
            stage_complete(
                region_state,
                stage_name
            )
        ):
            print(
                f"[{region_id}] "
                f"{stage_name.upper()} already complete."
            )
            continue

        print(
            f"\n[{region_id}] Running "
            f"{stage_name.upper()}..."
        )

        command = [
            python_exe,
            str(
                script
            ),
        ]

        ok, code, elapsed = run_command(
            command,
            logs_dir / f"{stage_name}.log",
            timeout_hours,
            PROJECT_DIR,
        )

        if not ok:
            mark_stage(
                region_state,
                stage_name,
                "failed",
                return_code=code,
                elapsed_seconds=elapsed,
            )

            region_state[
                "status"
            ] = (
                f"failed_{stage_name}"
            )

            save_state(
                state_file,
                state,
            )

            return {
                "region_id":
                    region_id,
                "status":
                    f"failed_{stage_name}",
                "run_dir":
                    str(
                        run_dir
                    ),
            }

        mark_stage(
            region_state,
            stage_name,
            "complete",
            elapsed_seconds=elapsed,
        )

        save_state(
            state_file,
            state,
        )

    region_state[
        "status"
    ] = "predictors_processed"

    save_state(
        state_file,
        state,
    )

    return {
        "region_id":
            region_id,

        "status":
            "predictors_processed",

        "run_dir":
            str(
                run_dir
            ),

        "gedi_sample_file":
            str(
                sample_file
            ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Automatically process selected GEDI training regions "
            "through V1 AOI, V2 Sentinel-2, V3 Sentinel-1, "
            "V4 terrain and optional V5 ALOS-2."
        )
    )

    parser.add_argument(
        "--manifest",
        default=str(
            DEFAULT_MANIFEST
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=
            DEFAULT_MAX_REGIONS,
    )

    parser.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Python executable used to run V1-V5. "
            "Default: current Python."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume completed regions/stages from processing_state.json."
        ),
    )

    parser.add_argument(
        "--skip-alos2",
        action="store_true",
    )

    parser.add_argument(
        "--skip-s1",
        action="store_true",
    )

    parser.add_argument(
        "--skip-terrain",
        action="store_true",
    )

    parser.add_argument(
        "--timeout-hours",
        type=float,
        default=
            DEFAULT_TIMEOUT_HOURS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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
        "TRAINING 03 - AUTOMATED REGIONAL PREDICTOR PROCESSING"
    )

    print(
        "=" * 72
    )

    args = parse_args()

    manifest_file = Path(
        args.manifest
    )

    output_dir = Path(
        args.output_dir
    )

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Manifest not found:\n{manifest_file}"
        )

    validate_scripts(
        skip_s1=args.skip_s1,
        skip_terrain=args.skip_terrain,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n[1/5] Reading training-region manifest..."
    )

    manifest = pd.read_csv(
        manifest_file
    )

    print(
        f"Available regions: "
        f"{len(manifest):,}"
    )

    print(
        "\n[2/5] Selecting geographically distributed regions..."
    )

    selected = choose_regions(
        manifest,
        max_regions=args.max_regions,
        seed=args.seed,
    )

    selected_manifest = (
        output_dir
        /
        "selected_training_regions.csv"
    )

    write_selected_manifest(
        selected,
        selected_manifest,
    )

    print(
        f"Selected regions: "
        f"{len(selected):,}"
    )

    print(
        f"Potential GEDI samples: "
        f"{int(selected['selected_gedi_cells'].sum()):,}"
    )

    print(
        "\nSelected manifest:"
    )

    print(
        selected_manifest
    )

    state_file = (
        output_dir
        /
        "processing_state.json"
    )

    state = load_state(
        state_file
    )

    results = []

    print(
        "\n[3/5] Processing regions..."
    )

    print(
        "\nIMPORTANT: process runs sequentially because "
        "the existing V2-V5 scripts use the active/latest run."
    )

    for index, row in selected.iterrows():
        region_id = row[
            "training_region_id"
        ]

        print(
            "\n"
            +
            "-" * 72
        )

        print(
            f"REGION "
            f"{index + 1}/{len(selected)}: "
            f"{region_id}"
        )

        print(
            "-" * 72
        )

        try:
            result = process_region(
                row=row,
                python_exe=args.python,
                output_dir=output_dir,
                state=state,
                state_file=state_file,
                resume=args.resume,
                skip_s1=args.skip_s1,
                skip_terrain=args.skip_terrain,
                skip_alos2=args.skip_alos2,
                timeout_hours=args.timeout_hours,
            )

        except KeyboardInterrupt:
            save_state(
                state_file,
                state,
            )

            print(
                "\nStopped by user. Progress was saved."
            )

            raise

        except Exception as exc:
            result = {
                "region_id":
                    region_id,

                "status":
                    "exception",

                "error":
                    str(
                        exc
                    ),
            }

            state[
                "regions"
            ].setdefault(
                region_id,
                {}
            )[
                "status"
            ] = "exception"

            state[
                "regions"
            ][region_id][
                "error"
            ] = str(
                exc
            )

            save_state(
                state_file,
                state,
            )

            print(
                f"\nERROR in {region_id}:\n{exc}"
            )

        results.append(
            result
        )

    print(
        "\n[4/5] Saving processing manifest..."
    )

    results_file = (
        output_dir
        /
        "regional_processing_results.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        results_file,
        index=False,
    )

    complete = sum(
        1
        for result in results
        if result.get(
            "status"
        )
        ==
        "predictors_processed"
    )

    failed = (
        len(results)
        -
        complete
    )

    summary = {
        "source_manifest":
            str(
                manifest_file
            ),

        "selected_manifest":
            str(
                selected_manifest
            ),

        "regions_requested":
            int(
                len(
                    selected
                )
            ),

        "potential_gedi_samples":
            int(
                selected[
                    "selected_gedi_cells"
                ].sum()
            ),

        "regions_processed_successfully":
            int(
                complete
            ),

        "regions_failed_or_incomplete":
            int(
                failed
            ),

        "resume_enabled":
            bool(
                args.resume
            ),

        "alos2_skipped":
            bool(
                args.skip_alos2
            ),

        "results_file":
            str(
                results_file
            ),

        "state_file":
            str(
                state_file
            ),

        "next_step":
            (
                "Aggregate the processed Sentinel-2, Sentinel-1, "
                "terrain and optional ALOS-2 rasters inside each "
                "region's GEDI 1-km training cells."
            ),
    }

    summary_file = (
        output_dir
        /
        "regional_processing_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n[5/5] Complete."
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "TRAINING 03 COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nSuccessful regions: "
        f"{complete:,}"
    )

    print(
        f"Failed/incomplete: "
        f"{failed:,}"
    )

    print(
        f"\nResults:\n"
        f"{results_file}"
    )

    print(
        f"\nState:\n"
        f"{state_file}"
    )

    print(
        "\nNext: aggregate processed raster predictors "
        "to GEDI training cells."
    )


if __name__ == "__main__":
    main()
