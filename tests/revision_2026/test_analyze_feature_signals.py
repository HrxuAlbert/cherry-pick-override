from __future__ import annotations

import importlib.util
import pathlib
import unittest

import numpy as np


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/analyze_feature_signals.py"
SPEC = importlib.util.spec_from_file_location("feature_signals", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FeatureSignalTests(unittest.TestCase):
    def test_ranking_metrics_are_perfect_for_perfect_scores(self):
        targets = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(MODULE.average_precision(targets, scores), 1.0)
        self.assertAlmostEqual(MODULE.auroc(targets, scores), 1.0)

    def test_grouped_folds_never_split_a_case(self):
        rows = []
        for case_id in range(20):
            for model in ("a", "b", "c"):
                rows.append(
                    {
                        "case_id": str(case_id),
                        "model": model,
                        "target_dataset_conflict": case_id % 2,
                    }
                )
        assignment = MODULE.grouped_stratified_folds(rows, folds=5, seed=3)
        self.assertEqual(len(assignment), 20)
        for case_id in range(20):
            self.assertEqual(assignment[str(case_id)], assignment[str(case_id)])

    def test_logistic_regression_learns_separable_signal(self):
        x = np.asarray([[1, -2], [1, -1], [1, 1], [1, 2]], dtype=float)
        y = np.asarray([0, 0, 1, 1], dtype=float)
        weights = MODULE.fit_logistic(x, y, l2=0.1)
        scores = MODULE._sigmoid(x @ weights)
        self.assertGreater(scores[2], scores[1])
        self.assertGreater(scores[3], scores[0])


if __name__ == "__main__":
    unittest.main()
