from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "analyze_current_model_results.py"
SPEC = importlib.util.spec_from_file_location("current_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CurrentAnalysisTests(unittest.TestCase):
    def test_normalize_keeps_parse_failure_explicit(self):
        self.assertEqual(MODULE.normalize_result({"status": "error"}), "parse_failure")
        self.assertEqual(
            MODULE.normalize_result(
                {
                    "status": "ok",
                    "parse_ok": True,
                    "parsed": {"verdict_normal": "support"},
                }
            ),
            "support",
        )

    def test_materialize_rejects_incomplete_matrix(self):
        manifest = {
            "dataset": {"name": "synthetic"},
            "models": [{"slot": "a", "enabled": True}, {"slot": "b", "enabled": True}]
        }
        cases = {0: "support", 1: "conflicting"}
        results = {
            ("a", 0): {"status": "ok", "parse_ok": True, "parsed": {"verdict_normal": "support"}}
        }
        with self.assertRaises(MODULE.CurrentAnalysisError):
            MODULE.materialize(manifest, cases, results)

    def test_materialize_uses_manifest_dataset_name(self):
        manifest = {
            "dataset": {"name": "VitaminC-Mixed"},
            "models": [{"slot": "a", "enabled": True}],
        }
        rows = MODULE.materialize(
            manifest,
            {0: "support"},
            {
                ("a", 0): {
                    "status": "ok",
                    "parse_ok": True,
                    "parsed": {"verdict_normal": "support"},
                }
            },
        )
        self.assertEqual(rows["a"][0]["dataset"], "VitaminC-Mixed")


if __name__ == "__main__":
    unittest.main()
