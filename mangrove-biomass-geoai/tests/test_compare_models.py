from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from ai.compare_models import train_comparison


class CompareModelsTest(unittest.TestCase):
    def test_same_validation_points_are_used(self):
        rng = np.random.default_rng(42)
        count = 240
        latitude = 21.0 + rng.random(count)
        longitude = 88.0 + rng.random(count)
        s2 = rng.normal(size=count)
        s1 = rng.normal(size=count)
        alos = rng.normal(size=count)
        target = 80 + 15 * s2 + 8 * s1 + 5 * alos + rng.normal(0, 8, count)
        frame = pd.DataFrame({
            "shot_number": [str(i) for i in range(count)],
            "latitude": latitude, "longitude": longitude,
            "agbd_mg_ha": target, "agbd_se_mg_ha": 10.0,
            "mangrove_sample": True,
            "s2__NDVI__mean": s2, "s1__VH__mean": s1,
            "alos2__HV__mean": alos,
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame.to_csv(root / "samples.csv", index=False)
            comparison, summary = train_comparison(root / "samples.csv", root / "models", 5.0)
            self.assertEqual(len(comparison), 3)
            self.assertEqual(summary["common_samples"], count)
            coordinates = []
            for model in comparison["model"]:
                held_out = pd.read_csv(root / "models" / model / "out_of_fold_validation.csv")
                coordinates.append(set(zip(held_out.latitude.round(8), held_out.longitude.round(8))))
            self.assertEqual(coordinates[0], coordinates[1])
            self.assertEqual(coordinates[1], coordinates[2])
            self.assertTrue((comparison["validation_count"] == count).all())
            self.assertTrue((root / "models" / summary["best_model_by_rmse"] / "random_forest_l4a.joblib").exists())


if __name__ == "__main__":
    unittest.main()
