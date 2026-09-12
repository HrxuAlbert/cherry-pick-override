from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"


def load(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = load("build_mechanism_manifests")
ANALYZER = load("analyze_mechanism_results")


class MechanismPipelineTests(unittest.TestCase):
    def test_case_id_parser_requires_exact_unique_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "ids.txt"
            path.write_text("0\n1\n2\n", encoding="utf-8")
            self.assertEqual(BUILDER.parse_case_ids(path, expected=3), [0, 1, 2])
            with self.assertRaises(BUILDER.ManifestBuildError):
                BUILDER.parse_case_ids(path, expected=4)

    def test_comparison_metrics_are_paired(self):
        baseline = {
            ("m", "0"): "support",
            ("m", "1"): "refute",
            ("m", "2"): "conflicting",
        }
        variant = {
            ("m", "0"): "conflicting",
            ("m", "1"): "support",
            ("m", "2"): "conflicting",
        }
        row = ANALYZER.compare(baseline, variant, "test", 100, 2)[0]
        self.assertAlmostEqual(row["baseline_directional_rate"], 2 / 3)
        self.assertAlmostEqual(row["variant_directional_rate"], 1 / 3)
        self.assertAlmostEqual(row["any_label_flip_rate"], 2 / 3)
        self.assertAlmostEqual(row["support_refute_flip_rate"], 1 / 3)
        self.assertAlmostEqual(row["recovery_to_conflicting_rate"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
