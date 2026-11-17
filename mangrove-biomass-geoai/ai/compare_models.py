"""Spatially validated sensor-ablation models for GEDI L4A AGBD."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


MODEL_DEFINITIONS = {
    "model_1_sentinel2": ("s2__",),
    "model_2_sentinel2_sentinel1": ("s2__", "s1__"),
    "model_3_sentinel2_sentinel1_alos2": ("s2__", "s1__", "alos2__"),
}


def _feature_sets(frame):
    mean_features = [c for c in frame.columns if c.endswith("__mean")]
    sets = {}
    for name, prefixes in MODEL_DEFINITIONS.items():
        selected = [c for c in mean_features if c.startswith(prefixes)]
        missing = [p for p in prefixes if not any(c.startswith(p) for c in selected)]
        if missing:
            raise ValueError(f"{name}: missing predictor families: {', '.join(missing)}")
        sets[name] = selected
    return sets


def _spatial_groups(frame, block_size_km):
    mean_lat = float(frame.latitude.mean())
    lon_scale = 111.32 * max(np.cos(np.deg2rad(mean_lat)), 0.1)
    block_x = np.floor(frame.longitude * lon_scale / block_size_km).astype(int)
    block_y = np.floor(frame.latitude * 111.32 / block_size_km).astype(int)
    return block_x.astype(str) + "_" + block_y.astype(str)


def _sample_weights(frame):
    """Down-weight uncertain GEDI labels without deleting biomass classes."""
    if "agbd_se_mg_ha" not in frame:
        return np.ones(len(frame), dtype="float64")
    se = pd.to_numeric(frame["agbd_se_mg_ha"], errors="coerce").to_numpy(float)
    finite = np.isfinite(se) & (se >= 0)
    typical = float(np.nanmedian(se[finite])) if finite.any() else 1.0
    typical = max(typical, 1.0)
    se = np.where(finite, se, typical)
    weights = 1.0 / (1.0 + np.square(se / typical))
    weights /= max(float(np.mean(weights)), 1e-12)
    return np.clip(weights, 0.25, 4.0)


def _new_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("random_forest", RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _metrics(observed, predicted):
    residual = predicted - observed
    return {
        "r2": float(r2_score(observed, predicted)),
        "mae_mg_ha": float(mean_absolute_error(observed, predicted)),
        "rmse_mg_ha": float(mean_squared_error(observed, predicted) ** 0.5),
        "bias_mg_ha": float(np.mean(residual)),
    }


def _status(model_metrics, baseline_metrics):
    if model_metrics["r2"] <= 0 or model_metrics["rmse_mg_ha"] >= baseline_metrics["rmse_mg_ha"]:
        return (
            "failed_validation",
            "The model did not outperform the spatial mean baseline. The map is demonstration-only.",
            False,
        )
    return (
        "experimental",
        "The model passed internal spatial validation but still requires independent field validation.",
        True,
    )


def train_comparison(csv_path, output_dir, block_size_km=5.0, n_folds=5):
    frame = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan)
    required = {"latitude", "longitude", "agbd_mg_ha"}
    if missing := required - set(frame.columns):
        raise ValueError("Training CSV is missing: " + ", ".join(sorted(missing)))
    if "mangrove_sample" in frame:
        frame = frame[frame["mangrove_sample"].astype(str).str.lower().isin(["true", "1"])]

    feature_sets = _feature_sets(frame)
    all_features = sorted(set().union(*feature_sets.values()))
    frame = frame.dropna(subset=["latitude", "longitude", "agbd_mg_ha", *all_features]).copy()
    frame = frame[frame.agbd_mg_ha >= 0].reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError(f"Only {len(frame)} complete mangrove samples remain. At least 50 are required; 300+ is recommended.")

    groups = _spatial_groups(frame, float(block_size_km))
    group_count = int(groups.nunique())
    folds = min(int(n_folds), group_count)
    if folds < 3:
        raise ValueError(f"Only {group_count} spatial blocks are available. Use a smaller block size or a larger study area.")

    frame["spatial_block"] = groups
    frame["spatial_fold"] = -1
    weights = _sample_weights(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = list(GroupKFold(n_splits=folds).split(frame, groups=groups))
    for fold, (_, test_idx) in enumerate(splits, start=1):
        frame.loc[test_idx, "spatial_fold"] = fold
    frame.to_csv(output_dir / "common_samples_with_spatial_folds.csv", index=False)

    baseline_oof = np.full(len(frame), np.nan, dtype="float64")
    for train_idx, test_idx in splits:
        baseline_oof[test_idx] = float(frame.loc[train_idx, "agbd_mg_ha"].mean())
    observed_all = frame["agbd_mg_ha"].to_numpy(float)
    baseline_metrics = _metrics(observed_all, baseline_oof)
    baseline_metrics["model"] = "spatial_training_mean_baseline"
    (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2), encoding="utf-8")

    results = []
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (name, features) in zip(axes, feature_sets.items()):
        oof = np.full(len(frame), np.nan, dtype="float64")
        fold_rows = []
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            model = _new_model()
            model.fit(
                frame.loc[train_idx, features],
                frame.loc[train_idx, "agbd_mg_ha"],
                random_forest__sample_weight=weights[train_idx],
            )
            predicted = np.maximum(model.predict(frame.loc[test_idx, features]), 0.0)
            oof[test_idx] = predicted
            row = {"fold": fold, "validation_count": int(len(test_idx))}
            row.update(_metrics(frame.loc[test_idx, "agbd_mg_ha"].to_numpy(float), predicted))
            fold_rows.append(row)

        pooled = _metrics(observed_all, oof)
        fold_table = pd.DataFrame(fold_rows)
        for metric in ("r2", "mae_mg_ha", "rmse_mg_ha", "bias_mg_ha"):
            pooled[f"fold_mean_{metric}"] = float(fold_table[metric].mean())
            pooled[f"fold_std_{metric}"] = float(fold_table[metric].std(ddof=1)) if len(fold_table) > 1 else 0.0
        status, message, map_recommended = _status(pooled, baseline_metrics)
        pooled.update({
            "model": name,
            "predictor_count": len(features),
            "common_sample_count": int(len(frame)),
            "validation_count": int(len(frame)),
            "spatial_blocks": group_count,
            "spatial_folds": folds,
            "block_size_km": float(block_size_km),
            "baseline_rmse_mg_ha": baseline_metrics["rmse_mg_ha"],
            "validation_status": status,
            "status_message": message,
            "map_recommended": bool(map_recommended),
        })

        model_dir = output_dir / name
        model_dir.mkdir(exist_ok=True)
        fold_table.to_csv(model_dir / "spatial_fold_metrics.csv", index=False)
        columns = [c for c in ["shot_number", "latitude", "longitude", "agbd_mg_ha", "agbd_se_mg_ha", "spatial_block", "spatial_fold"] if c in frame]
        validation = frame[columns].copy()
        validation["predicted_mg_ha"] = oof
        validation["residual_mg_ha"] = oof - observed_all
        validation.to_csv(model_dir / "out_of_fold_validation.csv", index=False)

        # Refit on all accepted samples after validation for final inference.
        final_model = _new_model()
        final_model.fit(frame[features], frame["agbd_mg_ha"], random_forest__sample_weight=weights)
        joblib.dump(final_model, model_dir / "random_forest_l4a.joblib")
        (model_dir / "feature_list_l4a.json").write_text(json.dumps(features, indent=2), encoding="utf-8")
        (model_dir / "metrics.json").write_text(json.dumps(pooled, indent=2), encoding="utf-8")
        importance = pd.DataFrame({
            "feature": features,
            "importance": final_model.named_steps["random_forest"].feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(model_dir / "feature_importance.csv", index=False)

        axis.scatter(observed_all, oof, s=18, alpha=.55)
        low = min(float(observed_all.min()), float(oof.min()))
        high = max(float(observed_all.max()), float(oof.max()))
        axis.plot([low, high], [low, high], "k--", lw=1)
        axis.set_title(name.replace("_", " "))
        axis.set_xlabel("GEDI AGBD (Mg/ha)")
        axis.set_ylabel("Out-of-fold prediction (Mg/ha)")
        results.append(pooled)

    figure.savefig(output_dir / "three_model_validation.png", dpi=170)
    plt.close(figure)
    comparison = pd.DataFrame(results)
    s2 = comparison.iloc[0]
    comparison["r2_change_vs_s2"] = comparison.r2 - s2.r2
    comparison["rmse_change_vs_s2"] = comparison.rmse_mg_ha - s2.rmse_mg_ha
    comparison["beats_mean_baseline"] = comparison.rmse_mg_ha < baseline_metrics["rmse_mg_ha"]
    comparison["rank_rmse"] = comparison.rmse_mg_ha.rank(method="min").astype(int)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    eligible = comparison[comparison["map_recommended"]]
    ranked = eligible if not eligible.empty else comparison
    best = ranked.sort_values(["rmse_mg_ha", "mae_mg_ha"]).iloc[0]
    summary = {
        "best_model_by_rmse": best["model"],
        "validation_status": best["validation_status"],
        "status_message": best["status_message"],
        "map_recommended": bool(best["map_recommended"]),
        "common_samples": int(len(frame)),
        "spatial_blocks": group_count,
        "spatial_folds": folds,
        "baseline_metrics": baseline_metrics,
        "training_note": "Final saved models were refitted using all accepted samples after validation.",
        "warning": "GEDI spatial validation is not a substitute for independent field plots.",
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "observation_time" in frame:
        dates = pd.to_datetime(frame["observation_time"], utc=True, errors="coerce")
    else:
        dates = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    target = frame["agbd_mg_ha"]
    data_summary = {
        "samples": int(len(frame)),
        "agbd_mean_mg_ha": float(target.mean()),
        "agbd_median_mg_ha": float(target.median()),
        "agbd_p05_mg_ha": float(target.quantile(.05)),
        "agbd_p95_mg_ha": float(target.quantile(.95)),
        "agbd_max_mg_ha": float(target.max()),
        "gedi_start": str(dates.min()) if dates.notna().any() else None,
        "gedi_end": str(dates.max()) if dates.notna().any() else None,
        "median_agbd_se_mg_ha": float(frame["agbd_se_mg_ha"].median()) if "agbd_se_mg_ha" in frame else None,
        "uncertainty_weighting": True,
    }
    (output_dir / "training_data_summary.json").write_text(json.dumps(data_summary, indent=2), encoding="utf-8")
    return comparison, summary
