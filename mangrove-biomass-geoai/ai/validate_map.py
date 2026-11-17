"""Validate a biomass GeoTIFF against independent GEDI L4A points."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def validate_map(raster_path, gedi_csv, output_csv, training_csv=None):
    raster_path, gedi_csv, output_csv = map(Path, (raster_path, gedi_csv, output_csv))
    points = pd.read_csv(gedi_csv)
    needed = {"latitude", "longitude", "agbd_mg_ha"}
    missing = needed - set(points.columns)
    if missing:
        raise ValueError(f"GEDI CSV is missing columns: {sorted(missing)}")
    if training_csv is not None:
        training = pd.read_csv(training_csv, dtype={"shot_number": "string"})
        if "shot_number" not in points or "shot_number" not in training:
            raise ValueError("Both validation and training CSVs need shot_number for independence checking.")
        validation_shots = set(points["shot_number"].astype("string").dropna())
        training_shots = set(training["shot_number"].astype("string").dropna())
        overlap = validation_shots & training_shots
        if overlap:
            raise ValueError(
                f"{len(overlap):,} GEDI shots occur in both training and validation. "
                "Use geographically or temporally independent observations."
            )
    with rasterio.open(raster_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xy = [transformer.transform(lon, lat) for lon, lat in zip(points.longitude, points.latitude)]
        sampled = np.array([value[0] for value in src.sample(xy)], dtype="float64")
        if src.nodata is not None:
            sampled[sampled == src.nodata] = np.nan
    result = points.copy()
    result["predicted_mg_ha"] = sampled
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["agbd_mg_ha", "predicted_mg_ha"])
    if len(result) < 3:
        raise ValueError("Fewer than 3 GEDI points overlap the biomass raster.")
    observed = result["agbd_mg_ha"].to_numpy(float)
    predicted = result["predicted_mg_ha"].to_numpy(float)
    metrics = {
        "validation_points": int(len(result)),
        "r2": float(r2_score(observed, predicted)),
        "mae_mg_ha": float(mean_absolute_error(observed, predicted)),
        "rmse_mg_ha": float(np.sqrt(mean_squared_error(observed, predicted))),
        "bias_mg_ha": float(np.mean(predicted - observed)),
        "training_overlap_checked": training_csv is not None,
        "note": "Independent field plots remain preferable to GEDI-only validation.",
    }
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    output_csv.with_name("map_validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster", required=True)
    parser.add_argument("--gedi-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--training-csv")
    args = parser.parse_args()
    validate_map(args.raster, args.gedi_csv, args.output_csv, args.training_csv)
