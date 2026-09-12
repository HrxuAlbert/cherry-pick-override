from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/analyze_error_taxonomy.py"
SPEC = importlib.util.spec_from_file_location("error_taxonomy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ErrorTaxonomyTests(unittest.TestCase):
    def test_material_mixed_has_precedence(self):
        category = MODULE.structural_category(
            {
                "has_material_support": True,
                "has_material_refute": True,
                "has_material_mixed": True,
                "has_material_insufficient": True,
            }
        )
        self.assertEqual(category, "material_mixed")

    def test_separate_support_refute_category(self):
        self.assertEqual(
            MODULE.structural_category(
                {"has_material_support": True, "has_material_refute": True}
            ),
            "separate_support_and_refute",
        )


if __name__ == "__main__":
    unittest.main()
