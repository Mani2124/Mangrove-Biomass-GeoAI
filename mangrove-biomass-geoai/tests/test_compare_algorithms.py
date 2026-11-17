from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from ai.compare_algorithms import ALGORITHMS, compare_algorithms


class CompareAlgorithmsTest(unittest.TestCase):
    def test_three_algorithms_use_common_spatial_samples(self):
        rng = np.random.default_rng(7)
        count = 180
        s2 = rng.normal(size=count)
        s1 = rng.normal(size=count)
        alos = rng.normal(size=count)
        frame = pd.DataFrame({
            "shot_number": np.arange(count).astype(str),
            "latitude": 21 + rng.random(count),
            "longitude": 88 + rng.random(count),
            "agbd_mg_ha": np.maximum(5, 35 + 12*s2 + 7*s1 + 5*alos + rng.normal(0, 5, count)),
            "agbd_se_mg_ha": 8.0,
            "mangrove_sample": True,
            "s2__NDVI__mean": s2,
            "s1__VH__mean": s1,
            "alos2__HV__mean": alos,
        })
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_file = root / "samples.csv"
            frame.to_csv(csv_file, index=False)
            result, summary = compare_algorithms(csv_file, root / "models", 5.0, 3)
            self.assertEqual(set(result.algorithm), set(ALGORITHMS))
            self.assertEqual(summary["common_samples"], count)
            self.assertTrue((result.sample_count == count).all())
            for algorithm in ALGORITHMS:
                self.assertTrue((root / "models" / algorithm / "biomass_model.joblib").exists())


if __name__ == "__main__":
    unittest.main()
