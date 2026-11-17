"""Train on the original joined predictor-table schema."""
import pandas as pd

def command(csv, output):
    table = pd.read_csv(csv)
    required = {"agbd_mu_mg_ha", "model_sample_valid", "training_region_id"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {required-set(table.columns)}")
    if table.training_region_id.nunique() < 5:
        raise ValueError("Use at least five separate regions for spatial validation.")
    if not any(c.startswith(("s1__", "s2__", "alos2__", "terrain__")) for c in table):
        raise ValueError("Join satellite statistics to GEDI labels first.")
    return "training_train_random_forest.py", ["--input-csv", csv, "--output-dir", output]
