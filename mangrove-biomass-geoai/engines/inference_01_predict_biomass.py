
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import re

import joblib
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.windows import from_bounds, Window
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "biomass_v1"
DEFAULT_MODEL_FILE = DEFAULT_MODEL_DIR / "random_forest_biomass.joblib"
DEFAULT_FEATURE_FILE = DEFAULT_MODEL_DIR / "feature_list.json"

DEFAULT_OUTPUT_RESOLUTION_M = 1000.8950233496674
DEFAULT_SUPPORT_SIZE_M = 1000.8950233496674
DEFAULT_MIN_VALID_FEATURE_FRACTION = 0.50

STATS = {"mean", "median", "std", "p25", "p75"}


def latest_run(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise RuntimeError(f"No runs found in:\n{runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def clean_feature_name(text: str) -> str:
    text = Path(text).stem
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def discover_predictor_rasters(run_dir: Path) -> dict[str, Path]:
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

        for path in sorted(directory.glob("*.tif")):
            if any(t in path.name.lower() for t in exclude_tokens):
                continue

            key = f"{modality}__{clean_feature_name(path.name)}"
            result.setdefault(key, path)

    return result


def split_training_feature(feature_name: str) -> tuple[str, str]:
    for stat in STATS:
        suffix = f"__{stat}"
        if feature_name.endswith(suffix):
            return feature_name[:-len(suffix)], stat

    raise ValueError(
        f"Unsupported training feature name:\n{feature_name}"
    )


def safe_window(src, bounds) -> Window | None:
    try:
        window = from_bounds(*bounds, transform=src.transform)
        window = window.round_offsets().round_lengths()
        full = Window(0, 0, src.width, src.height)
        window = window.intersection(full)
    except Exception:
        return None

    if window.width <= 0 or window.height <= 0:
        return None

    return window


def extract_values(src, geometry) -> np.ndarray:
    window = safe_window(src, geometry.bounds)
    if window is None:
        return np.array([], dtype="float32")

    data = src.read(
        1,
        window=window,
        masked=False,
    ).astype("float32")

    if data.size == 0:
        return np.array([], dtype="float32")

    win_transform = rasterio.windows.transform(
        window,
        src.transform,
    )

    inside = geometry_mask(
        [geometry],
        out_shape=data.shape,
        transform=win_transform,
        invert=True,
        all_touched=False,
    )

    valid = inside & np.isfinite(data)

    if src.nodata is not None:
        valid &= data != src.nodata

    return data[valid].astype("float32", copy=False)


def compute_stat(values: np.ndarray, stat: str) -> float:
    if len(values) == 0:
        return np.nan

    if stat == "mean":
        return float(np.mean(values))
    if stat == "median":
        return float(np.median(values))
    if stat == "std":
        return float(np.std(values))
    if stat == "p25":
        return float(np.percentile(values, 25))
    if stat == "p75":
        return float(np.percentile(values, 75))

    raise ValueError(f"Unsupported statistic: {stat}")


def build_output_grid(
    reference_raster: Path,
    output_resolution_m: float,
) -> dict:
    with rasterio.open(reference_raster) as src:
        crs = src.crs
        bounds = src.bounds

    width = int(math.ceil((bounds.right - bounds.left) / output_resolution_m))
    height = int(math.ceil((bounds.top - bounds.bottom) / output_resolution_m))

    transform = from_origin(
        bounds.left,
        bounds.top,
        output_resolution_m,
        output_resolution_m,
    )

    return {
        "crs": crs,
        "bounds": bounds,
        "width": width,
        "height": height,
        "transform": transform,
    }


def predict_biomass(
    run_dir: Path,
    model_file: Path,
    feature_file: Path,
    output_dir: Path,
    output_resolution_m: float,
    support_size_m: float,
    min_valid_feature_fraction: float,
) -> dict:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found:\n{run_dir}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found:\n{model_file}")
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature-list file not found:\n{feature_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/7] Loading trained biomass model...")
    model = joblib.load(model_file)

    feature_info = json.loads(
        feature_file.read_text(encoding="utf-8")
    )
    # L4A trainer writes a plain list; older models use {"features": [...]}
    training_features = (
        feature_info if isinstance(feature_info, list)
        else feature_info["features"]
    )

    print(f"Model features: {len(training_features):,}")

    print("\n[2/7] Discovering AOI predictor rasters...")
    raster_lookup = discover_predictor_rasters(run_dir)
    print(f"Predictor rasters found: {len(raster_lookup):,}")

    required_bases = {}
    missing_bases = []

    for feature in training_features:
        base, stat = split_training_feature(feature)
        required_bases.setdefault(base, set()).add(stat)

        if base not in raster_lookup:
            missing_bases.append(base)

    missing_bases = sorted(set(missing_bases))

    if missing_bases:
        raise RuntimeError(
            "The processed AOI is missing predictor rasters required by "
            "the trained model:\n\n"
            + "\n".join(missing_bases)
        )

    reference_candidates = [
        path
        for key, path in raster_lookup.items()
        if key.startswith("s2__") and "ndvi" in key.lower()
    ]

    if reference_candidates:
        reference_raster = reference_candidates[0]
    else:
        reference_raster = raster_lookup[next(iter(required_bases))]

    print("\n[3/7] Building requested output grid...")
    output_grid = build_output_grid(
        reference_raster,
        output_resolution_m,
    )

    print(f"CRS        : {output_grid['crs']}")
    print(
        f"Raster size: "
        f"{output_grid['width']} x {output_grid['height']}"
    )
    print(f"Resolution : {output_resolution_m:.2f} m")

    print("\n[4/7] Opening predictor rasters...")

    opened = {}

    try:
        for base in required_bases:
            path = raster_lookup[base]
            src = rasterio.open(path)

            if src.count != 1:
                src.close()
                raise RuntimeError(
                    f"Predictor raster must be single-band:\n{path}"
                )

            opened[base] = {
                "src": src,
                "transformer": Transformer.from_crs(
                    output_grid["crs"],
                    src.crs,
                    always_xy=True,
                ),
            }

        print("\n[5/7] Predicting biomass...")

        width = output_grid["width"]
        height = output_grid["height"]
        transform = output_grid["transform"]

        prediction = np.full(
            (height, width),
            np.nan,
            dtype="float32",
        )

        valid_mask = np.zeros(
            (height, width),
            dtype="uint8",
        )

        half_support = support_size_m / 2.0

        batch_rows = []
        batch_locations = []
        predicted = 0
        batch_size = 1000

        def flush_batch():
            nonlocal predicted

            if not batch_rows:
                return

            X = pd.DataFrame(
                batch_rows,
                columns=training_features,
            )

            preds = model.predict(X)

            for (r, c), value in zip(batch_locations, preds):
                prediction[r, c] = float(value)
                valid_mask[r, c] = 1
                predicted += 1

            batch_rows.clear()
            batch_locations.clear()

        for r in range(height):
            for c in range(width):
                x_center, y_center = rasterio.transform.xy(
                    transform,
                    r,
                    c,
                    offset="center",
                )

                # Match the circular GEDI-footprint support used during
                # training instead of using a square inference window.
                support_cell = Point(x_center, y_center).buffer(
                    half_support,
                    resolution=16,
                )

                feature_values = {}
                valid_feature_count = 0

                for base, needed_stats in required_bases.items():
                    info = opened[base]

                    projected_cell = shapely_transform(
                        info["transformer"].transform,
                        support_cell,
                    )

                    values = extract_values(
                        info["src"],
                        projected_cell,
                    )

                    if len(values) > 0:
                        valid_feature_count += len(needed_stats)

                    for stat in needed_stats:
                        feature_values[
                            f"{base}__{stat}"
                        ] = compute_stat(values, stat)

                valid_fraction = (
                    valid_feature_count / len(training_features)
                )

                if valid_fraction < min_valid_feature_fraction:
                    continue

                batch_rows.append(
                    [
                        feature_values.get(feature, np.nan)
                        for feature in training_features
                    ]
                )
                batch_locations.append((r, c))

                if len(batch_rows) >= batch_size:
                    flush_batch()

            if (
                r == 0
                or (r + 1) % 10 == 0
                or (r + 1) == height
            ):
                print(f"Rows processed: {r + 1}/{height}")

        flush_batch()

        print(f"Predicted cells: {predicted:,}")

        print("\n[6/7] Saving AGBD raster...")

        agbd_file = output_dir / "AGBD_prediction_Mg_ha.tif"
        valid_file = output_dir / "AGBD_valid_mask.tif"

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": output_grid["crs"],
            "transform": transform,
            "nodata": -9999.0,
            "compress": "LZW",
            "tiled": True,
        }

        out_arr = prediction.copy()
        out_arr[~np.isfinite(out_arr)] = -9999.0

        with rasterio.open(
            agbd_file,
            "w",
            **profile,
        ) as dst:
            dst.write(out_arr, 1)
            dst.set_band_description(
                1,
                "Predicted AGBD Mg/ha",
            )

        mask_profile = profile.copy()
        mask_profile["dtype"] = "uint8"
        mask_profile["nodata"] = 0

        with rasterio.open(
            valid_file,
            "w",
            **mask_profile,
        ) as dst:
            dst.write(valid_mask, 1)
            dst.set_band_description(
                1,
                "Biomass prediction valid mask",
            )

        rows = []

        for r in range(height):
            for c in range(width):
                value = prediction[r, c]

                if not np.isfinite(value):
                    continue

                x, y = rasterio.transform.xy(
                    transform,
                    r,
                    c,
                    offset="center",
                )

                rows.append(
                    {
                        "row": r,
                        "column": c,
                        "x": x,
                        "y": y,
                        "agbd_mg_ha": float(value),
                    }
                )

        grid_csv = output_dir / "prediction_grid.csv"
        pd.DataFrame(rows).to_csv(
            grid_csv,
            index=False,
        )

        print("\n[7/7] Saving summary...")

        finite = prediction[np.isfinite(prediction)]

        summary = {
            "run_dir": str(run_dir),
            "model_file": str(model_file),
            "feature_file": str(feature_file),
            "output_resolution_m": output_resolution_m,
            "support_size_m": support_size_m,
            "training_feature_count": len(training_features),
            "predicted_cells": int(predicted),
            "total_output_cells": int(width * height),
            "mean_agbd_mg_ha":
                float(np.mean(finite))
                if len(finite) else None,
            "median_agbd_mg_ha":
                float(np.median(finite))
                if len(finite) else None,
            "minimum_agbd_mg_ha":
                float(np.min(finite))
                if len(finite) else None,
            "maximum_agbd_mg_ha":
                float(np.max(finite))
                if len(finite) else None,
            "agbd_raster": str(agbd_file),
            "valid_mask": str(valid_file),
            "prediction_grid_csv": str(grid_csv),
            "scientific_status": "prototype_pipeline_test",
            "warning": (
                "The current RF model was trained on a small sample. "
                "Treat this biomass map as a pipeline demonstration, "
                "not validated scientific accuracy."
            ),
            "next_step": (
                "Convert AGBD to carbon and calculate AOI statistics."
            ),
        }

        summary_file = output_dir / "prediction_summary.json"
        summary_file.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        return {
            "agbd_file": agbd_file,
            "valid_file": valid_file,
            "grid_csv": grid_csv,
            "summary_file": summary_file,
            "predicted_cells": predicted,
            "mean_agbd": summary["mean_agbd_mg_ha"],
        }

    finally:
        for info in opened.values():
            try:
                info["src"].close()
            except Exception:
                pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the trained biomass Random Forest to "
            "a processed AOI using ~1-km support."
        )
    )

    parser.add_argument(
        "--run-dir",
        default=None,
    )

    parser.add_argument(
        "--model-file",
        default=str(DEFAULT_MODEL_FILE),
    )

    parser.add_argument(
        "--feature-file",
        default=str(DEFAULT_FEATURE_FILE),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument(
        "--output-resolution-m",
        type=float,
        default=DEFAULT_OUTPUT_RESOLUTION_M,
    )

    parser.add_argument(
        "--support-size-m",
        type=float,
        default=DEFAULT_SUPPORT_SIZE_M,
    )

    parser.add_argument(
        "--min-valid-feature-fraction",
        type=float,
        default=DEFAULT_MIN_VALID_FEATURE_FRACTION,
    )

    return parser.parse_args()


def main():
    print("\n" + "=" * 72)
    print("BIOMASS AND CARBON PLATFORM")
    print("INFERENCE 01 - BIOMASS PREDICTION")
    print("=" * 72)

    args = parse_args()

    run_dir = (
        Path(args.run_dir)
        if args.run_dir
        else latest_run(PROJECT_DIR / "runs")
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / "prediction" / "biomass_v1"
    )

    print(f"\nUsing run:\n{run_dir}")

    result = predict_biomass(
        run_dir=run_dir,
        model_file=Path(args.model_file),
        feature_file=Path(args.feature_file),
        output_dir=output_dir,
        output_resolution_m=args.output_resolution_m,
        support_size_m=args.support_size_m,
        min_valid_feature_fraction=
            args.min_valid_feature_fraction,
    )

    print("\n" + "=" * 72)
    print("INFERENCE 01 COMPLETE")
    print("=" * 72)

    print(
        f"\nPredicted cells:\n"
        f"{result['predicted_cells']:,}"
    )

    print(
        f"\nMean AGBD:\n"
        f"{result['mean_agbd']} Mg/ha"
    )

    print(
        f"\nAGBD raster:\n"
        f"{result['agbd_file']}"
    )

    print("\nNext: convert AGBD to carbon.")


if __name__ == "__main__":
    main()
