from __future__ import annotations

import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/validate_current_model_smoke.py"
SPEC = importlib.util.spec_from_file_location("smoke_validator", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SmokeValidatorTests(unittest.TestCase):
    def test_accepts_complete_single_model_smoke(self):
        manifest = {
            "run_id": "test",
            "smoke_case_ids": [1],
            "models": [
                {
                    "slot": "m",
                    "enabled": True,
                    "accounting_max_output_tokens": 10,
                }
            ],
        }
        rows = [
            {
                "status": "ok",
                "model_slot": "m",
                "case_id": 1,
                "parse_ok": True,
                "parsed": {"verdict_normal": "support"},
                "identity_checks": {"model": True, "provider": True},
                "response_id": "r",
                "tokens_in": 2,
                "tokens_out": 3,
                "finish_reason": "stop",
                "returned_model": "model",
                "returned_provider": "provider",
            }
        ]
        self.assertTrue(MODULE.validate(manifest, rows)["accepted"])

    def test_rejects_truncation(self):
        manifest = {
            "run_id": "test",
            "smoke_case_ids": [1],
            "models": [
                {
                    "slot": "m",
                    "enabled": True,
                    "accounting_max_output_tokens": 10,
                }
            ],
        }
        rows = [
            {
                "status": "ok",
                "model_slot": "m",
                "case_id": 1,
                "parse_ok": True,
                "parsed": {"verdict_normal": "support"},
                "identity_checks": {"model": True, "provider": True},
                "response_id": "r",
                "tokens_in": 2,
                "tokens_out": 10,
                "finish_reason": "length",
                "returned_model": "model",
                "returned_provider": "provider",
            }
        ]
        self.assertFalse(MODULE.validate(manifest, rows)["accepted"])


if __name__ == "__main__":
    unittest.main()
