from __future__ import annotations

import hashlib
import importlib.util
import argparse
import os
import pathlib
import json
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts/run_current_models.py"
SPEC = importlib.util.spec_from_file_location("current_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CurrentRunnerTests(unittest.TestCase):
    def test_full_results_are_separate_from_smoke_results(self):
        root = pathlib.Path("run")
        smoke, _ = MODULE.execution_artifact_paths(root, "smoke")
        full, progress = MODULE.execution_artifact_paths(root, "full")
        self.assertEqual(smoke.name, "raw_results.jsonl")
        self.assertEqual(full.name, "raw_results_full.jsonl")
        self.assertEqual(progress.name, "progress_full.json")

    def test_openrouter_rate_limit_uses_backoff_without_changing_manifest(self):
        manifest = {"inter_call_delay_seconds": 0.2}
        spec = {
            "base_url": "https://openrouter.ai/api/v1",
            "provider_identity": "groq",
        }
        self.assertEqual(MODULE.post_call_delay_seconds(manifest, spec, None), 10.0)
        self.assertEqual(
            MODULE.post_call_delay_seconds(manifest, spec, "HTTP 429 upstream"), 60.0
        )
        mistral = {
            "base_url": "https://openrouter.ai/api/v1",
            "provider_identity": "mistral",
        }
        self.assertEqual(MODULE.post_call_delay_seconds(manifest, mistral, None), 3.0)
        self.assertEqual(
            MODULE.post_call_delay_seconds(manifest, mistral, "HTTP 503 upstream"), 15.0
        )

    def test_openai_compatible_response_uses_direct_provider_identity_fallback(self):
        parsed = MODULE.parse_provider_response(
            "openai_compatible",
            {
                "id": "resp-1",
                "model": "gpt-test",
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            "openai-direct",
        )
        self.assertEqual(parsed["returned_provider"], "openai-direct")
        self.assertEqual(parsed["finish_reason"], "stop")

    def test_anthropic_refusal_is_a_terminal_parse_failure_not_transport_error(self):
        parsed = MODULE.parse_provider_response(
            "anthropic_messages",
            {
                "id": "msg-1",
                "model": "claude-test",
                "content": [],
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "bio"},
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
            "anthropic-direct",
        )
        self.assertEqual(parsed["text"], "")
        self.assertEqual(parsed["finish_reason"], "refusal")
        self.assertEqual(parsed["stop_details"]["category"], "bio")

    def test_accumulated_cost_survives_resume(self):
        models = [
            {
                "slot": "m",
                "pricing_usd_per_million": {"input": 2.0, "output": 10.0},
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "model_slot": "m",
                        "tokens_in": 100,
                        "tokens_out": 20,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertAlmostEqual(
                MODULE.accumulated_estimated_cost_usd(path, models), 0.0004
            )

    def test_execution_guard_rejects_unlisted_paid_mode(self):
        args = argparse.Namespace(execute=True, mode="full")
        manifest = {
            "status": "frozen",
            "paid_execution_authorized": True,
            "authorized_modes": ["smoke"],
        }
        with mock.patch.dict(os.environ, {"CCO_ALLOW_PAID_RUN": MODULE.EXECUTION_ACK}):
            with self.assertRaisesRegex(MODULE.RunnerError, "authorized_modes"):
                MODULE.execution_guard(args, manifest)

    def test_evidence_reversal_reverses_qa_blocks_not_text(self):
        example = {
            "questions": [
                {"question": "first", "answers": [{"answer": "one"}]},
                {"question": "second", "answers": [{"answer": "two"}]},
            ]
        }
        original = MODULE.build_evidence(example, "original")
        reversed_evidence = MODULE.build_evidence(example, "reversed")
        self.assertTrue(original.startswith("Q: first"))
        self.assertTrue(reversed_evidence.startswith("Q: second"))
        self.assertIn("A: one", reversed_evidence)

    def test_hash_pinned_selection_preserves_file_order(self):
        cases = [
            {"case_id": i, "gold_normal": "conflicting" if i < 3 else "support"}
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "ids.txt"
            path.write_text("2\n0\n1\n", encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "selection": {
                    "case_ids_path": str(path),
                    "case_ids_sha256": digest,
                    "expected_count": 3,
                    "source": "synthetic",
                }
            }
            selected, metadata = MODULE.load_selected_cases(manifest, cases)
            self.assertEqual([row["case_id"] for row in selected], [2, 0, 1])
            self.assertEqual(metadata["selected_label_counts"], {"conflicting": 3})


if __name__ == "__main__":
    unittest.main()
