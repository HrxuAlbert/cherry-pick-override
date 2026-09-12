from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_human_confirmed_cco.py"
)
SPEC = importlib.util.spec_from_file_location("audit_analysis", SCRIPT)
assert SPEC and SPEC.loader
audit_analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_analysis)


def review_row(audit_id: str, decision: str, confidence: int = 3) -> dict[str, str]:
    return {
        "audit_id": audit_id,
        "support_signal": "material",
        "refute_signal": "material",
        "final_decision": decision,
        "decision_confidence": str(confidence),
        "resolution_reason": "none",
        "short_rationale": "Both evidence directions remain material."
        if decision in {"preserve_conflict", "unclear"}
        else "",
    }


class AgreementTests(unittest.TestCase):
    def test_perfect_agreement(self) -> None:
        rows = [
            review_row("AUD-001", "preserve_conflict"),
            review_row("AUD-002", "resolve_support"),
            review_row("AUD-003", "insufficient"),
        ]
        reviewer_a = audit_analysis.validate_review_rows(rows, "A")
        reviewer_b = audit_analysis.validate_review_rows(rows, "B")
        merged = audit_analysis.merge_reviews(reviewer_a, reviewer_b)
        report = audit_analysis.agreement_report(merged)
        self.assertEqual(report["strict_preserve_n"], 1)
        self.assertEqual(report["five_way"]["raw_agreement"], 1.0)
        self.assertEqual(report["five_way"]["cohen_kappa"], 1.0)
        self.assertEqual(report["binary_preserve_vs_other"]["gwet_ac1"], 1.0)

    def test_reviewer_row_mismatch_is_rejected(self) -> None:
        reviewer_a = audit_analysis.validate_review_rows(
            [review_row("AUD-001", "preserve_conflict")], "A"
        )
        reviewer_b = audit_analysis.validate_review_rows(
            [review_row("AUD-002", "preserve_conflict")], "B"
        )
        with self.assertRaises(audit_analysis.AnalysisError):
            audit_analysis.merge_reviews(reviewer_a, reviewer_b)


class ModelMetricTests(unittest.TestCase):
    def synthetic_inputs(self):
        reviewer_a: list[dict[str, str]] = []
        reviewer_b: list[dict[str, str]] = []
        mapping: dict[str, dict[str, str]] = {}
        model_by_case: dict[int, dict[str, object]] = {}

        for case_id in range(210):
            audit_id = f"AUD-{case_id + 1:03d}"
            if case_id < 100:
                decision_a = decision_b = "preserve_conflict"
            elif case_id < 120:
                decision_a, decision_b = "preserve_conflict", "resolve_refute"
            elif case_id < 150:
                decision_a = decision_b = "resolve_refute"
            elif case_id < 170:
                decision_a = decision_b = "resolve_support"
            elif case_id < 190:
                decision_a = decision_b = "resolve_refute"
            else:
                decision_a = decision_b = "insufficient"

            reviewer_a.append(review_row(audit_id, decision_a))
            reviewer_b.append(review_row(audit_id, decision_b))
            mapping[audit_id] = {
                "set_name": "main",
                "audit_id": audit_id,
                "core_case_id": str(case_id),
                "gold_normal": "conflicting" if case_id < 150 else (
                    "support" if case_id < 170 else "refute" if case_id < 190 else "insufficient"
                ),
            }
            if case_id < 10 or 100 <= case_id < 110:
                verdict = "support"
            else:
                verdict = "conflicting"
            model_by_case[case_id] = {
                "status": "ok",
                "parse_ok": True,
                "parsed": {"verdict_normal": verdict},
            }

        merged = audit_analysis.merge_reviews(
            audit_analysis.validate_review_rows(reviewer_a, "A"),
            audit_analysis.validate_review_rows(reviewer_b, "B"),
        )
        return merged, mapping, {"synthetic_model": model_by_case}

    def test_four_estimands_use_frozen_denominators(self) -> None:
        merged, mapping, model_rows = self.synthetic_inputs()
        metrics, matrices = audit_analysis.build_model_reports(
            merged, mapping, {}, model_rows
        )
        self.assertEqual(len(metrics), 1)
        row = metrics[0]
        self.assertEqual(row["strict_conditional_cco_numerator"], 10)
        self.assertEqual(row["strict_conditional_cco_denominator"], 100)
        self.assertEqual(row["strict_conditional_cco_rate"], 0.1)
        self.assertEqual(row["confirmed_lower_bound_denominator"], 150)
        self.assertAlmostEqual(row["confirmed_lower_bound_rate"], 10 / 150, places=6)
        self.assertEqual(row["dataset_defined_cco_numerator"], 20)
        self.assertEqual(row["dataset_defined_cco_denominator"], 150)
        self.assertIn("synthetic_model", matrices)

    def test_parse_failure_stays_in_denominator(self) -> None:
        merged, mapping, model_rows = self.synthetic_inputs()
        model_rows["synthetic_model"][0] = {
            "status": "ok",
            "parse_ok": False,
            "parsed": None,
        }
        metrics, _ = audit_analysis.build_model_reports(merged, mapping, {}, model_rows)
        row = metrics[0]
        self.assertEqual(row["strict_conditional_cco_numerator"], 9)
        self.assertEqual(row["strict_conditional_cco_denominator"], 100)
        self.assertEqual(row["strict_parse_or_endpoint_failures"], 1)
        self.assertEqual(row["strict_parse_unsafe_numerator"], 10)


if __name__ == "__main__":
    unittest.main()

