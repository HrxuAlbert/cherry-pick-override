from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/analyze_cached_model_landscape.py"
SPEC = importlib.util.spec_from_file_location("cached_landscape", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MetricTests(unittest.TestCase):
    def test_wilson_interval_contains_observed_rate(self):
        low, high = MODULE.wilson_interval(10, 100)
        self.assertLess(low, 0.10)
        self.assertGreater(high, 0.10)

    def test_parse_failures_remain_in_fixed_denominators(self):
        rows = [
            {"case_id": "0", "gold": "conflicting", "pred": "support"},
            {"case_id": "1", "gold": "conflicting", "pred": MODULE.PARSE_FAILURE},
            {"case_id": "2", "gold": "support", "pred": "support"},
            {"case_id": "3", "gold": "refute", "pred": "conflicting"},
            {"case_id": "4", "gold": "insufficient", "pred": "refute"},
        ]
        metrics = MODULE.compute_metrics("synthetic", "model", rows)
        self.assertEqual(metrics["n_total"], 5)
        self.assertEqual(metrics["parse_failures"], 1)
        self.assertEqual(metrics["cco_n"], 1)
        self.assertEqual(metrics["cco_rate"], 0.5)
        self.assertEqual(metrics["pure_sr_accuracy"], 0.5)
        self.assertEqual(metrics["insufficient_directional_rate"], 1.0)

    def test_paired_bootstrap_uses_identical_cases(self):
        a = [
            {"case_id": str(i), "gold": "conflicting", "pred": "support"}
            for i in range(5)
        ]
        b = [
            {"case_id": str(i), "gold": "conflicting", "pred": "conflicting"}
            for i in range(5)
        ]
        result = MODULE.paired_bootstrap(
            a,
            b,
            lambda row: row["gold"] == "conflicting",
            lambda row: int(row["pred"] in {"support", "refute"}),
            resamples=100,
            seed=1,
        )
        self.assertEqual(result[0], 5)
        self.assertEqual(result[1], 1.0)
        self.assertEqual(result[2], 0.0)
        self.assertEqual(result[3], -1.0)


if __name__ == "__main__":
    unittest.main()
