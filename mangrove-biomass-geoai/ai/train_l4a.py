"""Train and validate a GEDI L4A biomass model."""

from pathlib import Path
import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_l4a_model(csv_path, output_folder):
    csv_path = Path(csv_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    target = "agbd_mg_ha"

    features = [
        column
        for column in df.columns
        if column.startswith(("s2__", "s1__", "alos2__"))
        and column.endswith("__mean")
    ]

    if target not in df.columns:
        raise ValueError("Column 'agbd_mg_ha' was not found in the CSV.")

    if not features:
        raise ValueError("No Sentinel-1, Sentinel-2 or ALOS-2 predictor columns found.")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} usable samples found. At least 20 are required.")

    X = df[features]
    y = df[target]

    # Use geographic validation when multiple regions are available.
    regions = (
        df.get(
            "training_region_id",
            pd.Series(index=df.index, dtype="object"),
        )
        .fillna("")
        .astype(str)
    )

    unique_regions = [region for region in regions.unique() if region]

    if len(unique_regions) >= 2:
        test_region = unique_regions[-1]

        train_mask = regions != test_region
        test_mask = regions == test_region

        X_train = X.loc[train_mask]
        X_test = X.loc[test_mask]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]

        split_name = f"leave-one-region-out: {test_region}"

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

        split_name = "random 80/20 technical test (one region only)"

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "sample_count": int(len(df)),
        "training_count": int(len(X_train)),
        "test_count": int(len(X_test)),
        "target": target,
        "predictor_count": int(len(features)),
        "mae_mg_ha": float(mean_absolute_error(y_test, predictions)),
        "rmse_mg_ha": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "r2": float(r2_score(y_test, predictions)),
        "split": split_name,
        "warning": (
            "Add multiple geographic regions before claiming "
            "regional or global accuracy."
        ),
    }

    joblib.dump(
        model,
        output_folder / "random_forest_l4a.joblib",
    )

    (output_folder / "feature_list_l4a.json").write_text(
        json.dumps(features, indent=2),
        encoding="utf-8",
    )

    (output_folder / "metrics_l4a.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    predictions_table = pd.DataFrame(
        {
            "observed_mg_ha": y_test.to_numpy(),
            "predicted_mg_ha": predictions,
        }
    )

    predictions_table.to_csv(
        output_folder / "test_predictions_l4a.csv",
        index=False,
    )

    # Observed versus predicted plot
    plt.figure(figsize=(7, 7))
    plt.scatter(
        y_test,
        predictions,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.3,
    )

    minimum = min(float(y_test.min()), float(predictions.min()))
    maximum = max(float(y_test.max()), float(predictions.max()))

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        "k--",
        label="1:1 line",
    )

    plt.xlabel("GEDI AGBD (Mg/ha)")
    plt.ylabel("Predicted AGBD (Mg/ha)")
    plt.title("GEDI L4A biomass model validation")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        output_folder / "observed_vs_predicted_l4a.png",
        dpi=150,
    )

    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        help="GEDI-satellite training CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for the trained model",
    )

    arguments = parser.parse_args()

    train_l4a_model(
        arguments.csv,
        arguments.output,
    )