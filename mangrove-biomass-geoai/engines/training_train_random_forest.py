
from __future__ import annotations

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


PROJECT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_INPUT_CSV = (
    PROJECT_DIR
    / "training"
    / "streaming"
    / "biomass_training_dataset.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "models"
    / "biomass_v1"
)

TARGET_COLUMN = "agbd_mu_mg_ha"
VALID_COLUMN = "model_sample_valid"
GROUP_COLUMN = "training_region_id"

PREDICTOR_PREFIXES = (
    "s2__",
    "s1__",
    "terrain__",
    "alos2__",
)

ALLOWED_SUFFIXES = (
    "__mean",
    "__median",
    "__std",
    "__p25",
    "__p75",
)


def find_predictors(df: pd.DataFrame) -> list[str]:
    return sorted(
        [
            c
            for c in df.columns
            if c.startswith(PREDICTOR_PREFIXES)
            and c.endswith(ALLOWED_SUFFIXES)
        ]
    )


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    if VALID_COLUMN in data.columns:
        valid = (
            data[VALID_COLUMN]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )
        data = data.loc[valid].copy()

    data[TARGET_COLUMN] = pd.to_numeric(
        data[TARGET_COLUMN],
        errors="coerce",
    )

    data = data.loc[
        np.isfinite(data[TARGET_COLUMN])
        &
        (data[TARGET_COLUMN] >= 0)
    ].copy()

    return data.reset_index(drop=True)


def filter_features(
    data: pd.DataFrame,
    features: list[str],
    max_missing_fraction: float,
):
    kept = []
    report = []

    for feature in features:
        values = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        finite = np.isfinite(
            values.to_numpy(dtype="float64")
        )

        missing_fraction = 1.0 - float(np.mean(finite))

        valid_values = values.loc[finite]

        variance = (
            float(valid_values.var(ddof=0))
            if len(valid_values) >= 2
            else np.nan
        )

        keep = (
            missing_fraction <= max_missing_fraction
            and len(valid_values) >= 2
            and np.isfinite(variance)
            and variance > 0
        )

        report.append(
            {
                "feature": feature,
                "missing_fraction": missing_fraction,
                "variance": variance,
                "kept": keep,
            }
        )

        if keep:
            kept.append(feature)

    return kept, pd.DataFrame(report)


def split_data(
    data: pd.DataFrame,
    test_fraction: float,
    validation_fraction: float,
    seed: int,
):
    if len(data) < 6:
        raise RuntimeError(
            "At least 6 valid samples are needed."
        )

    if GROUP_COLUMN in data.columns:
        groups = data[GROUP_COLUMN].astype(str)

        if groups.nunique() >= 3:
            try:
                outer = GroupShuffleSplit(
                    n_splits=1,
                    test_size=test_fraction,
                    random_state=seed,
                )

                trainval_idx, test_idx = next(
                    outer.split(
                        data,
                        groups=groups,
                    )
                )

                trainval = data.iloc[trainval_idx]
                trainval_groups = trainval[GROUP_COLUMN].astype(str)

                relative_val = (
                    validation_fraction
                    /
                    (1.0 - test_fraction)
                )

                inner = GroupShuffleSplit(
                    n_splits=1,
                    test_size=relative_val,
                    random_state=seed + 1,
                )

                train_rel, val_rel = next(
                    inner.split(
                        trainval,
                        groups=trainval_groups,
                    )
                )

                train_idx = trainval_idx[train_rel]
                val_idx = trainval_idx[val_rel]

                if (
                    len(train_idx) >= 2
                    and len(val_idx) >= 1
                    and len(test_idx) >= 1
                ):
                    return (
                        np.asarray(train_idx),
                        np.asarray(val_idx),
                        np.asarray(test_idx),
                        "spatial_group_split",
                    )

            except Exception:
                pass

    raise ValueError("Spatial split failed. Add independent regions; random fallback is disabled.")

    all_idx = np.arange(len(data))

    trainval_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
    )

    relative_val = (
        validation_fraction
        /
        (1.0 - test_fraction)
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=relative_val,
        random_state=seed + 1,
        shuffle=True,
    )

    return (
        np.asarray(train_idx),
        np.asarray(val_idx),
        np.asarray(test_idx),
        "random_split",
    )


def calculate_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")

    rmse = float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )

    mae = float(
        mean_absolute_error(y_true, y_pred)
    )

    r2 = (
        float(r2_score(y_true, y_pred))
        if len(y_true) >= 2
        else None
    )

    bias = float(
        np.mean(y_pred - y_true)
    )

    return {
        "n": int(len(y_true)),
        "r2": r2,
        "rmse_mg_ha": rmse,
        "mae_mg_ha": mae,
        "bias_mg_ha": bias,
    }


def prediction_frame(
    data: pd.DataFrame,
    indices: np.ndarray,
    y_true,
    y_pred,
    split_name: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "source_index": indices,
            "split": split_name,
            "observed_agbd_mg_ha": y_true,
            "predicted_agbd_mg_ha": y_pred,
        }
    )

    out["residual_mg_ha"] = (
        out["predicted_agbd_mg_ha"]
        -
        out["observed_agbd_mg_ha"]
    )

    for column in [
        GROUP_COLUMN,
        "longitude",
        "latitude",
        "row",
        "column",
        "land_type",
    ]:
        if column in data.columns:
            out[column] = (
                data.iloc[indices][column].to_numpy()
            )

    return out


def train_model(
    input_csv: Path,
    output_dir: Path,
    test_fraction: float,
    validation_fraction: float,
    max_missing_fraction: float,
    n_estimators: int,
    min_samples_leaf: int,
    max_features,
    seed: int,
):
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found:\n{input_csv}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\n[1/8] Reading training data...")

    raw = pd.read_csv(input_csv)

    if TARGET_COLUMN not in raw.columns:
        raise RuntimeError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    data = clean_rows(raw)

    print(f"Rows in CSV       : {len(raw):,}")
    print(f"Model-ready rows  : {len(data):,}")

    print("\n[2/8] Selecting predictors...")

    candidate_features = find_predictors(data)

    if not candidate_features:
        raise RuntimeError(
            "No predictor columns were found."
        )

    features, feature_quality = filter_features(
        data,
        candidate_features,
        max_missing_fraction,
    )

    print(f"Candidate features: {len(candidate_features):,}")
    print(f"Usable features   : {len(features):,}")

    if len(features) < 2:
        raise RuntimeError(
            "Not enough usable predictor features."
        )

    feature_quality.to_csv(
        reports_dir / "feature_quality.csv",
        index=False,
    )

    X = (
        data[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    y = data[TARGET_COLUMN].to_numpy(dtype="float64")

    print("\n[3/8] Splitting train / validation / test...")

    train_idx, val_idx, test_idx, split_method = split_data(
        data,
        test_fraction,
        validation_fraction,
        seed,
    )

    print(f"Split method: {split_method}")
    print(f"Train      : {len(train_idx)}")
    print(f"Validation : {len(val_idx)}")
    print(f"Test       : {len(test_idx)}")

    split_manifest = pd.DataFrame(
        {
            "source_index": np.arange(len(data)),
            "split": "unused",
            TARGET_COLUMN: y,
        }
    )

    split_manifest.loc[train_idx, "split"] = "train"
    split_manifest.loc[val_idx, "split"] = "validation"
    split_manifest.loc[test_idx, "split"] = "test"

    for column in [
        GROUP_COLUMN,
        "longitude",
        "latitude",
        "row",
        "column",
        "land_type",
    ]:
        if column in data.columns:
            split_manifest[column] = data[column].to_numpy()

    split_manifest.to_csv(
        reports_dir / "data_split_manifest.csv",
        index=False,
    )

    print("\n[4/8] Training Random Forest...")

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=seed,
                    n_jobs=-1,
                    bootstrap=True,
                    oob_score=(len(train_idx) >= 20),
                ),
            ),
        ]
    )

    pipeline.fit(
        X.iloc[train_idx],
        y[train_idx],
    )

    print("\n[5/8] Evaluating...")

    train_pred = pipeline.predict(X.iloc[train_idx])
    val_pred = pipeline.predict(X.iloc[val_idx])
    test_pred = pipeline.predict(X.iloc[test_idx])

    train_metrics = calculate_metrics(
        y[train_idx],
        train_pred,
    )

    val_metrics = calculate_metrics(
        y[val_idx],
        val_pred,
    )

    test_metrics = calculate_metrics(
        y[test_idx],
        test_pred,
    )

    print("\nTRAIN:")
    print(json.dumps(train_metrics, indent=2))

    print("\nVALIDATION:")
    print(json.dumps(val_metrics, indent=2))

    print("\nTEST:")
    print(json.dumps(test_metrics, indent=2))

    print("\n[6/8] Feature importance...")

    rf = pipeline.named_steps["random_forest"]

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": rf.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance["rank"] = np.arange(
        1,
        len(importance) + 1,
    )

    importance.to_csv(
        reports_dir / "feature_importance.csv",
        index=False,
    )

    print(
        importance.head(20).to_string(index=False)
    )

    print("\n[7/8] Saving model...")

    model_file = (
        output_dir
        / "random_forest_biomass.joblib"
    )

    joblib.dump(
        pipeline,
        model_file,
    )

    feature_file = (
        output_dir
        / "feature_list.json"
    )

    feature_file.write_text(
        json.dumps(
            {
                "target": TARGET_COLUMN,
                "feature_count": len(features),
                "features": features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions = pd.concat(
        [
            prediction_frame(
                data,
                train_idx,
                y[train_idx],
                train_pred,
                "train",
            ),
            prediction_frame(
                data,
                val_idx,
                y[val_idx],
                val_pred,
                "validation",
            ),
            prediction_frame(
                data,
                test_idx,
                y[test_idx],
                test_pred,
                "test",
            ),
        ],
        ignore_index=True,
    )

    predictions.to_csv(
        reports_dir / "model_predictions.csv",
        index=False,
    )

    print("\n[8/8] Saving metrics...")

    oob_r2 = None

    if getattr(rf, "oob_score", False):
        try:
            if np.isfinite(rf.oob_score_):
                oob_r2 = float(rf.oob_score_)
        except Exception:
            pass

    summary = {
        "model": "RandomForestRegressor",
        "purpose": "AGBD biomass baseline",
        "status": (
            "pipeline_test"
            if len(data) < 500
            else "baseline_model"
        ),
        "warning": (
            "Small training dataset. Metrics are preliminary."
            if len(data) < 500
            else None
        ),
        "input_csv": str(input_csv),
        "samples": int(len(data)),
        "features": int(len(features)),
        "split_method": split_method,
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "parameters": {
            "n_estimators": n_estimators,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "imputation": "median",
            "seed": seed,
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
            "oob_r2": oob_r2,
        },
        "model_file": str(model_file),
        "feature_list": str(feature_file),
        "feature_importance": str(
            reports_dir / "feature_importance.csv"
        ),
        "predictions": str(
            reports_dir / "model_predictions.csv"
        ),
        "next_step": (
            "Apply the saved model to a processed user AOI "
            "using the exact feature list, then write AGBD "
            "and carbon rasters."
        ),
    }

    metrics_file = (
        output_dir
        / "training_metrics.json"
    )

    metrics_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "samples": len(data),
        "features": len(features),
        "model_file": model_file,
        "metrics_file": metrics_file,
        "test_metrics": test_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a Random Forest biomass baseline from "
            "the low-disk streaming training dataset."
        )
    )

    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT_CSV),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-features",
        default="sqrt",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    print("\n" + "=" * 72)
    print("BIOMASS AND CARBON PLATFORM")
    print("TRAINING 05 - RANDOM FOREST BIOMASS BASELINE")
    print("=" * 72)

    args = parse_args()

    max_features = args.max_features

    try:
        max_features = float(max_features)
    except ValueError:
        pass

    result = train_model(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        max_missing_fraction=args.max_missing_fraction,
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features=max_features,
        seed=args.seed,
    )

    print("\n" + "=" * 72)
    print("TRAINING 05 COMPLETE")
    print("=" * 72)

    print(f"\nSamples used:\n{result['samples']:,}")
    print(f"\nPredictor features:\n{result['features']:,}")
    print(f"\nSaved model:\n{result['model_file']}")
    print(f"\nMetrics:\n{result['metrics_file']}")

    print("\nTEST METRICS:")
    print(
        json.dumps(
            result["test_metrics"],
            indent=2,
        )
    )

    print(
        "\nNext: raster inference -> AGBD map -> carbon map."
    )


if __name__ == "__main__":
    main()
