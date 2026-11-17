from pathlib import Path

import pandas as pd


csv_file = Path(
    r"D:\biomass_platform\runs"
    r"\study_16e9fc4499\training"
    r"\mangrove_sample_audit.csv"
)

df = pd.read_csv(csv_file)

# Keep only GEDI footprints accepted as mangrove.
if "mangrove_sample" in df.columns:
    mangrove = df[
        df["mangrove_sample"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    ].copy()
else:
    mangrove = df.copy()

predictors = [
    column
    for column in mangrove.columns
    if column.endswith("__mean")
]

results = []

for column in predictors:
    missing = int(mangrove[column].isna().sum())

    results.append(
        {
            "predictor": column,
            "sensor": column.split("__")[0],
            "missing_samples": missing,
            "available_samples": len(mangrove) - missing,
            "missing_percent": (
                100 * missing / len(mangrove)
                if len(mangrove)
                else 0
            ),
        }
    )

report = pd.DataFrame(results).sort_values(
    "missing_samples",
    ascending=False,
)

print("\nTotal mangrove GEDI samples:", len(mangrove))
print("\nMissing predictor report:\n")
print(report.to_string(index=False))

print("\nMissing samples by sensor:\n")

sensor_report = (
    report.groupby("sensor")["missing_samples"]
    .max()
    .sort_values(ascending=False)
)

print(sensor_report)

output = csv_file.parent / "missing_predictor_report.csv"

report.to_csv(output, index=False)

print("\nSaved:", output)