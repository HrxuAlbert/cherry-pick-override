from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/profile_experiment_datasets.py"
SPEC = importlib.util.spec_from_file_location("dataset_profiles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DatasetProfileTests(unittest.TestCase):
    def test_percentile_interpolates_quartiles(self):
        values = [1, 2, 3, 4]
        self.assertEqual(MODULE.percentile(values, 0.5), 2.5)
        self.assertEqual(MODULE.percentile(values, 0.25), 1.75)


if __name__ == "__main__":
    unittest.main()
