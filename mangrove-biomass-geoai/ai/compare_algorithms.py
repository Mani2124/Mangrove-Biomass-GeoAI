"""Fair spatial comparison of three regression algorithms on one feature set."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


ALGORITHMS = {
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
}


def _features(frame: pd.DataFrame) -> list[str]:
    columns = [
        column for column in frame.columns
        if column.endswith("__mean") and column.startswith(("s2__", "s1__", "alos2__"))
    ]
    if not columns:
        raise ValueError("No Sentinel-2, Sentinel-1 or ALOS-2 predictor columns were found.")
    return sorted(columns)


def _groups(frame: pd.DataFrame, block_size_km: float) -> pd.Series:
    mean_latitude = float(frame.latitude.mean())
    longitude_scale = 111.32 * max(np.cos(np.deg2rad(mean_latitude)), 0.1)
    block_x = np.floor(frame.longitude * longitude_scale / block_size_km).astype(int)
    block_y = np.floor(frame.latitude * 111.32 / block_size_km).astype(int)
    return block_x.astype(str) + "_" + block_y.astype(str)


def _weights(frame: pd.DataFrame) -> np.ndarray:
    if "agbd_se_mg_ha" not in frame:
        return np.ones(len(frame), dtype="float64")
    uncertainty = pd.to_numeric(frame.agbd_se_mg_ha, errors="coerce").to_numpy(float)
    finite = np.isfinite(uncertainty) & (uncertainty >= 0)
    typical = max(float(np.nanmedian(uncertainty[finite])) if finite.any() else 1.0, 1.0)
    uncertainty = np.where(finite, uncertainty, typical)
    weight = 1.0 / (1.0 + np.square(uncertainty / typical))
    weight /= max(float(np.mean(weight)), 1e-12)
    return np.clip(weight, 0.25, 4.0)


def _model(name: str) -> Pipeline:
    if name == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )
    elif name == "extra_trees":
        regressor = ExtraTreesRegressor(
            n_estimators=600, min_samples_leaf=3, max_features="sqrt",
            random_state=42, n_jobs=-1,
        )
    elif name == "hist_gradient_boosting":
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=350, max_leaf_nodes=31,
            min_samples_leaf=15, l2_regularization=1.0, random_state=42,
        )
    else:
        raise ValueError(f"Unknown algorithm: {name}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("regressor", regressor)])


def _fit(model: Pipeline, x, y, weight):
    model.fit(x, y, regressor__sample_weight=weight)


def _metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(observed, predicted)),
        "mae_mg_ha": float(mean_absolute_error(observed, predicted)),
        "rmse_mg_ha": float(mean_squared_error(observed, predicted) ** 0.5),
        "bias_mg_ha": float(np.mean(predicted - observed)),
    }


def compare_algorithms(csv_path, output_dir, block_size_km=5.0, n_folds=5):
    frame = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    required = {"latitude", "longitude", "agbd_mg_ha"}
    if missing := required - set(frame.columns):
        raise ValueError("Training CSV is missing: " + ", ".join(sorted(missing)))
    if "mangrove_sample" in frame:
        frame = frame[frame.mangrove_sample.astype(str).str.lower().isin(["true", "1"])]
    features = _features(frame)
    frame = frame.dropna(subset=["latitude", "longitude", "agbd_mg_ha", *features]).copy()
    frame = frame[frame.agbd_mg_ha >= 0].reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError(f"Only {len(frame)} complete samples remain. At least 50 are required; 300+ is recommended.")

    groups = _groups(frame, float(block_size_km))
    folds = min(int(n_folds), int(groups.nunique()))
    if folds < 3:
        raise ValueError("At least three spatial blocks are required.")
    splits = list(GroupKFold(n_splits=folds).split(frame, groups=groups))
    observed = frame.agbd_mg_ha.to_numpy(float)
    weights = _weights(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = np.full(len(frame), np.nan)
    for train_index, test_index in splits:
        baseline[test_index] = float(frame.iloc[train_index].agbd_mg_ha.mean())
    baseline_metrics = _metrics(observed, baseline)

    results = []
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (name, label) in zip(axes, ALGORITHMS.items()):
        predictions = np.full(len(frame), np.nan)
        fold_rows = []
        for fold, (train_index, test_index) in enumerate(splits, start=1):
            model = _model(name)
            _fit(
                model, frame.iloc[train_index][features], frame.iloc[train_index].agbd_mg_ha,
                weights[train_index],
            )
            predicted = np.maximum(model.predict(frame.iloc[test_index][features]), 0.0)
            predictions[test_index] = predicted
            fold_metric = {"fold": fold, "validation_count": int(len(test_index))}
            fold_metric.update(_metrics(observed[test_index], predicted))
            fold_rows.append(fold_metric)

        pooled = _metrics(observed, predictions)
        fold_table = pd.DataFrame(fold_rows)
        pooled.update({
            "algorithm": name,
            "algorithm_label": label,
            "sample_count": int(len(frame)),
            "predictor_count": int(len(features)),
            "spatial_folds": folds,
            "spatial_blocks": int(groups.nunique()),
            "block_size_km": float(block_size_km),
            "fold_std_rmse_mg_ha": float(fold_table.rmse_mg_ha.std(ddof=1)),
            "baseline_rmse_mg_ha": baseline_metrics["rmse_mg_ha"],
            "beats_mean_baseline": bool(pooled["rmse_mg_ha"] < baseline_metrics["rmse_mg_ha"]),
        })
        pooled["validation_status"] = "experimental" if pooled["beats_mean_baseline"] and pooled["r2"] > 0 else "failed_validation"

        model_dir = output_dir / name
        model_dir.mkdir(exist_ok=True)
        fold_table.to_csv(model_dir / "spatial_fold_metrics.csv", index=False)
        validation_columns = [c for c in ["shot_number", "latitude", "longitude", "agbd_mg_ha", "agbd_se_mg_ha"] if c in frame]
        validation = frame[validation_columns].copy()
        validation["spatial_block"] = groups
        validation["predicted_mg_ha"] = predictions
        validation["residual_mg_ha"] = predictions - observed
        validation.to_csv(model_dir / "out_of_fold_validation.csv", index=False)

        final_model = _model(name)
        _fit(final_model, frame[features], frame.agbd_mg_ha, weights)
        joblib.dump(final_model, model_dir / "biomass_model.joblib")
        (model_dir / "feature_list_l4a.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
        (model_dir / "metrics.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")

        axis.scatter(observed, predictions, s=18, alpha=0.55)
        low = min(float(observed.min()), float(predictions.min()))
        high = max(float(observed.max()), float(predictions.max()))
        axis.plot([low, high], [low, high], "k--", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("GEDI AGBD (Mg/ha)")
        axis.set_ylabel("Out-of-fold prediction (Mg/ha)")
        results.append(pooled)

    figure.savefig(output_dir / "algorithm_validation.png", dpi=170)
    plt.close(figure)
    comparison = pd.DataFrame(results).sort_values(["rmse_mg_ha", "mae_mg_ha"]).reset_index(drop=True)
    comparison["rank_rmse"] = np.arange(1, len(comparison) + 1)
    comparison.to_csv(output_dir / "algorithm_comparison.csv", index=False)
    eligible = comparison[(comparison.beats_mean_baseline) & (comparison.r2 > 0)]
    ranked = eligible if not eligible.empty else comparison
    best = ranked.iloc[0]
    summary = {
        "best_algorithm": best.algorithm,
        "best_algorithm_label": best.algorithm_label,
        "validation_status": best.validation_status,
        "common_samples": int(len(frame)),
        "predictor_count": int(len(features)),
        "spatial_folds": folds,
        "spatial_blocks": int(groups.nunique()),
        "block_size_km": float(block_size_km),
        "baseline_metrics": baseline_metrics,
        "warning": "Internal GEDI spatial validation is not independent field validation.",
    }
    (output_dir / "algorithm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return comparison, summary
