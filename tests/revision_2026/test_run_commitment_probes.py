from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import sys
import unittest


SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_commitment_probes", SCRIPTS / "run_commitment_probes.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RecognitionParserTests(unittest.TestCase):
    def test_yes_with_both_spans_is_span_backed(self):
        parsed = MODULE.parse_recognition(
            json.dumps(
                {
                    "support_span": "masks reduce transmission",
                    "refute_span": "prolonged N-95 use in lung disease",
                    "materially_two_sided": "yes",
                    "note": "both directions present",
                }
            )
        )
        self.assertEqual(parsed["recognition"], "yes")
        self.assertEqual(parsed["recognition_span_backed"], "yes")

    def test_yes_without_a_span_is_recorded_but_not_span_backed(self):
        parsed = MODULE.parse_recognition(
            json.dumps(
                {
                    "support_span": "",
                    "refute_span": "some refutation",
                    "materially_two_sided": "yes",
                    "note": "",
                }
            )
        )
        self.assertEqual(parsed["recognition"], "yes")
        self.assertEqual(parsed["recognition_span_backed"], "no")

    def test_missing_answer_is_a_parse_failure_not_a_no(self):
        self.assertIsNone(
            MODULE.parse_recognition(json.dumps({"support_span": "a", "refute_span": "b"}))
        )

    def test_json_embedded_in_prose_is_recovered(self):
        parsed = MODULE.parse_recognition(
            'Here you go:\n{"support_span":"","refute_span":"","materially_two_sided":"no"}\nDone.'
        )
        self.assertEqual(parsed["recognition"], "no")

    def test_non_json_is_a_parse_failure(self):
        self.assertIsNone(MODULE.parse_recognition("The evidence is two-sided."))


class LetterFirstParserTests(unittest.TestCase):
    @staticmethod
    def _logprobs(pairs, emitted=None):
        """One emitted token plus its top-k alternatives, as the APIs return it."""
        return [
            {
                "token": emitted if emitted is not None else pairs[0][0],
                "logprob": math.log(pairs[0][1]),
                "top_logprobs": [
                    {"token": token, "logprob": math.log(probability)}
                    for token, probability in pairs
                ],
            }
        ]

    def test_ranks_and_margin_are_computed_over_the_four_letters(self):
        parsed = MODULE.parse_letter_first(
            "B", self._logprobs([("B", 0.55), ("C", 0.30), ("A", 0.10), ("D", 0.05)])
        )
        self.assertEqual(parsed["verdict_normal"], "refute")
        self.assertEqual(parsed["conflicting_rank"], 2)
        self.assertAlmostEqual(parsed["p_conflicting"], 0.30, places=6)
        self.assertAlmostEqual(parsed["margin"], 0.25, places=6)

    def test_non_letter_alternatives_are_ignored(self):
        parsed = MODULE.parse_letter_first(
            "A", self._logprobs([("A", 0.6), ("the", 0.2), ("C", 0.2)])
        )
        self.assertEqual(set(parsed["letter_distribution"]), {"support", "conflicting"})

    def test_absent_logprobs_still_yield_a_verdict(self):
        parsed = MODULE.parse_letter_first("C", None)
        self.assertEqual(parsed["verdict_normal"], "conflicting")
        self.assertFalse(parsed["logprobs_available"])
        self.assertIsNone(parsed["margin"])

    def test_letter_map_matches_the_frozen_addendum_order(self):
        self.assertEqual(
            MODULE.LETTER_TO_VERDICT,
            {"A": "support", "B": "refute", "C": "conflicting", "D": "insufficient"},
        )

    def test_prose_beginning_with_a_letter_is_a_parse_failure(self):
        # "I Cannot" and "Answer:" both contain A/B/C/D; neither is a verdict.
        for text in ("I cannot answer.", "Answer: SUPPORTS", "Based on the evidence, B", ""):
            with self.subTest(text=text):
                self.assertIsNone(MODULE.parse_letter_first(text, None))

    def test_decorated_single_letters_are_accepted(self):
        for text, expected in (("A", "support"), (' "B" ', "refute"), ("C.", "conflicting"),
                               ("**D**", "insufficient"), ("c)", "conflicting")):
            with self.subTest(text=text):
                self.assertEqual(MODULE.parse_letter_first(text, None)["verdict_normal"], expected)


class DecompositionParserTests(unittest.TestCase):
    def test_two_sided_decomposition(self):
        parsed = MODULE.parse_decomposition(
            json.dumps({"supporting_items": ["a", "b"], "refuting_items": ["c"]})
        )
        self.assertEqual(parsed["n_supporting"], 2)
        self.assertEqual(parsed["decomposition_two_sided"], "yes")

    def test_empty_side_is_one_sided(self):
        parsed = MODULE.parse_decomposition(
            json.dumps({"supporting_items": [], "refuting_items": ["c"]})
        )
        self.assertEqual(parsed["decomposition_two_sided"], "no")

    def test_wrong_schema_is_a_parse_failure(self):
        self.assertIsNone(MODULE.parse_decomposition(json.dumps({"items": ["a"]})))

    def test_stage2_prompt_renders_the_decomposition(self):
        rendered = MODULE.format_decomposition(
            {"supporting_items": ["s1"], "refuting_items": []}
        )
        self.assertIn("- s1", rendered)
        self.assertIn("Refuting items: (none)", rendered)


class RenderTests(unittest.TestCase):
    def test_missing_placeholder_raises(self):
        with self.assertRaises(MODULE.RunnerError):
            MODULE.render("Claim: <<CLAIM>>", {"CLAIM": "c", "EVIDENCE": "e"})

    def test_unfilled_placeholder_raises(self):
        with self.assertRaises(MODULE.RunnerError):
            MODULE.render("<<CLAIM>> <<EVIDENCE>> <<DECOMPOSITION>>", {"CLAIM": "c", "EVIDENCE": "e"})

    def test_all_placeholders_filled(self):
        self.assertEqual(MODULE.render("<<A>>|<<B>>", {"A": "1", "B": "2"}), "1|2")


class CostingTests(unittest.TestCase):
    """The estimator multiplies by the model roster itself, so rows must cover cases once."""

    @staticmethod
    def _plan(n_models, n_cases, deferred):
        return [
            {
                "spec": {"slot": f"m{model}"},
                "case": {"case_id": case},
                "stages": [{"prompt": "p" * 10}],
                "deferred_stage": "stage2_verdict" if deferred else None,
            }
            for model in range(n_models)
            for case in range(n_cases)
        ]

    def test_rows_do_not_multiply_by_model_count(self):
        rows = MODULE.costing_rows(self._plan(4, 25, deferred=False))
        self.assertEqual(len(rows), 25)

    def test_two_stage_probe_charges_two_rows_per_case(self):
        rows = MODULE.costing_rows(self._plan(4, 25, deferred=True))
        self.assertEqual(len(rows), 50)

    def test_empty_plan(self):
        self.assertEqual(MODULE.costing_rows([]), [])


class ProbeRequestOverrideTests(unittest.TestCase):
    def test_letter_first_enables_logprobs_and_drops_json_mode(self):
        spec = {
            "slot": "oss",
            "provider_type": "openai_compatible",
            "request": {"response_format": {"type": "json_object"}, "max_completion_tokens": 8},
        }
        merged = MODULE.apply_overrides(spec, MODULE.probe_request_overrides("letter_first", spec))
        self.assertTrue(merged["request"]["logprobs"])
        self.assertEqual(merged["request"]["top_logprobs"], 20)
        self.assertNotIn("response_format", merged["request"])

    def test_letter_first_rejects_an_endpoint_without_logprobs(self):
        spec = {"slot": "anthropic", "provider_type": "anthropic_messages", "request": {}}
        with self.assertRaises(MODULE.RunnerError):
            MODULE.probe_request_overrides("letter_first", spec)

    def test_recognition_adds_no_overrides(self):
        spec = {"slot": "x", "provider_type": "openai_compatible", "request": {"seed": 1}}
        self.assertEqual(MODULE.probe_request_overrides("recognition", spec), {})

    def test_overrides_do_not_mutate_the_input_spec(self):
        spec = {
            "slot": "oss",
            "provider_type": "openai_compatible",
            "request": {"response_format": {"type": "json_object"}},
        }
        MODULE.apply_overrides(spec, MODULE.probe_request_overrides("letter_first", spec))
        self.assertIn("response_format", spec["request"])


class RequestSignatureTests(unittest.TestCase):
    BASE = {
        "slot": "s",
        "model_id": "m",
        "checkpoint_id": "c",
        "provider_identity": "p",
        "base_url": "u",
        "request": {"seed": 1},
    }

    def test_stage_is_part_of_the_signature(self):
        first = MODULE.request_signature(self.BASE, "sys", "prompt", "stage1_decompose", "r")
        second = MODULE.request_signature(self.BASE, "sys", "prompt", "stage2_verdict", "r")
        self.assertNotEqual(first, second)

    def test_prompt_change_changes_the_signature(self):
        first = MODULE.request_signature(self.BASE, "sys", "a", "recognition", "r")
        second = MODULE.request_signature(self.BASE, "sys", "b", "recognition", "r")
        self.assertNotEqual(first, second)

    def test_identical_inputs_are_stable(self):
        first = MODULE.request_signature(self.BASE, "sys", "a", "recognition", "r")
        second = MODULE.request_signature(self.BASE, "sys", "a", "recognition", "r")
        self.assertEqual(first, second)


class ManifestTests(unittest.TestCase):
    MANIFEST_DIR = (
        pathlib.Path(__file__).resolve().parents[4]
        / "outputs/revision_2026/probes/manifests"
    )

    def test_builder_emits_candidate_locked_manifests(self):
        """Freshly built manifests must be locked; authorizing one is a hand edit.

        This builds into a temporary directory rather than inspecting the manifests
        on disk, which are deliberately authorized once a run has been approved.
        """
        import subprocess
        import tempfile

        scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(scripts / "build_probe_manifests.py"),
                 "--output-dir", tmp, "--force"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            built = sorted(pathlib.Path(tmp).glob("*.json"))
            self.assertTrue(built, "builder produced no manifests")
            for path in built:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                with self.subTest(manifest=path.name):
                    self.assertEqual(manifest["status"], "candidate")
                    self.assertIs(manifest["paid_execution_authorized"], False)
                    self.assertEqual(manifest["authorized_modes"], [])
                    self.assertIn(manifest["probe"]["type"], MODULE.PROBE_TYPES)
                    self.assertLess(
                        manifest["expected_actual_cost_usd"], manifest["max_estimated_cost_usd"]
                    )

    def test_letter_first_enables_only_logprob_capable_slots(self):
        path = next(self.MANIFEST_DIR.glob("*e2-letter-logprob*.json"))
        manifest = json.loads(path.read_text(encoding="utf-8"))
        enabled = [m for m in manifest["models"] if m.get("enabled")]
        self.assertTrue(enabled)
        for model in enabled:
            self.assertEqual(model["provider_type"], "openai_compatible")

    def test_probes_inherit_the_parent_dataset_pin(self):
        parent = json.loads(
            (
                pathlib.Path(__file__).resolve().parents[1] / "current_models_manifest.json"
            ).read_text(encoding="utf-8")
        )
        for path in sorted(self.MANIFEST_DIR.glob("*.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            with self.subTest(manifest=path.name):
                self.assertEqual(manifest["dataset"], parent["dataset"])
                self.assertEqual(
                    manifest["probe"]["parent_verdict_run_id"], parent["run_id"]
                )


if __name__ == "__main__":
    unittest.main()
