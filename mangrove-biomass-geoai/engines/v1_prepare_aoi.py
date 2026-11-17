from __future__ import annotations

from pathlib import Path
import argparse
import csv
import shutil

from config import AREA_CRS, RUNS_DIR, SUPPORTED_AOI_EXTENSIONS
from src.aoi_manager import (
    create_run_directory,
    load_aoi,
    save_standard_aoi,
    summarize_aoi,
)


def copy_input_aoi(source_file: Path, input_dir: Path) -> Path:
    destination = input_dir / source_file.name
    shutil.copy2(source_file, destination)
    return destination


def save_metadata(
    summary: dict,
    run_dir: Path,
    original_aoi: Path,
    standardized_aoi: Path,
) -> Path:
    output_file = run_dir / "input" / "project_metadata.csv"

    rows = [
        ("run_directory", str(run_dir)),
        ("original_aoi", str(original_aoi)),
        ("standardized_aoi", str(standardized_aoi)),
        ("feature_count", summary["feature_count"]),
        ("source_crs", summary["source_crs"]),
        ("area_km2", summary["area_km2"]),
        ("west", summary["west"]),
        ("south", summary["south"]),
        ("east", summary["east"]),
        ("north", summary["north"]),
    ]

    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["field", "value"])
        writer.writerows(rows)

    return output_file


def prepare_aoi(aoi_file: Path) -> dict:
    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI file not found:\n{aoi_file}")

    if aoi_file.suffix.lower() not in SUPPORTED_AOI_EXTENSIONS:
        raise ValueError(
            "Unsupported AOI format. Use ZIP, GeoJSON, JSON, or GeoPackage."
        )

    run_dir = create_run_directory(RUNS_DIR)
    input_dir = run_dir / "input"
    original_aoi = copy_input_aoi(aoi_file, input_dir)

    print("\n[1/4] Loading AOI...")
    gdf = load_aoi(original_aoi)

    print("\n[2/4] Calculating AOI information...")
    summary = summarize_aoi(gdf, AREA_CRS)

    print("\n[3/4] Saving standardized AOI...")
    standardized_aoi = save_standard_aoi(
        gdf,
        input_dir / "aoi_wgs84.geojson",
    )

    print("\n[4/4] Saving metadata...")
    metadata_file = save_metadata(
        summary,
        run_dir,
        original_aoi,
        standardized_aoi,
    )

    return {
        "run_dir": run_dir,
        "original_aoi": original_aoi,
        "standardized_aoi": standardized_aoi,
        "metadata_file": metadata_file,
        "summary": summary,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare an AOI for the Biomass and Carbon Platform."
    )

    parser.add_argument(
        "--aoi",
        required=True,
        help="Path to ZIP shapefile, GeoJSON, JSON, or GeoPackage.",
    )

    return parser.parse_args()


def main():
    print("\n" + "=" * 72)
    print("BIOMASS AND CARBON PLATFORM")
    print("VERSION 1 - AOI PREPARATION")
    print("=" * 72)

    args = parse_arguments()
    result = prepare_aoi(Path(args.aoi))
    summary = result["summary"]

    print("\nAOI SUMMARY")
    print("-" * 72)
    print(f"Features : {summary['feature_count']}")
    print(f"CRS      : {summary['source_crs']}")
    print(f"Area     : {summary['area_km2']:.3f} km²")
    print(
        "Bounds   : "
        f"{summary['west']:.6f}, "
        f"{summary['south']:.6f}, "
        f"{summary['east']:.6f}, "
        f"{summary['north']:.6f}"
    )

    print("\n" + "=" * 72)
    print("VERSION 1 COMPLETE")
    print("=" * 72)
    print(f"\nRun directory:\n{result['run_dir']}")
    print(f"\nStandardized AOI:\n{result['standardized_aoi']}")
    print(f"\nMetadata:\n{result['metadata_file']}")
    print("\nReady for Version 2: GEDI processing.")


if __name__ == "__main__":
    main()
