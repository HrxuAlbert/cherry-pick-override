"""Validate a current-model smoke run and write a machine-readable report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_VERDICTS = {"support", "refute", "insufficient", "conflicting"}
TRUNCATION_REASONS = {"length", "max_tokens", "model_context_window_exceeded"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def validate(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    models = {spec["slot"]: spec for spec in manifest["models"] if spec.get("enabled")}
    expected_keys = {
        (slot, case_id)
        for slot in models
        for case_id in manifest["smoke_case_ids"]
    }
    successes = [row for row in rows if row.get("status") == "ok"]
    success_counts = Counter((row.get("model_slot"), row.get("case_id")) for row in successes)
    missing = sorted(expected_keys - set(success_counts))
    duplicate_successes = sorted(key for key, count in success_counts.items() if count != 1)
    unexpected = sorted(set(success_counts) - expected_keys)

    failures: list[str] = []
    if missing:
        failures.append(f"missing successful slot/case keys: {missing}")
    if duplicate_successes:
        failures.append(f"non-unique successful slot/case keys: {duplicate_successes}")
    if unexpected:
        failures.append(f"unexpected successful slot/case keys: {unexpected}")

    by_model: list[dict[str, Any]] = []
    for slot, spec in models.items():
        selected = [row for row in successes if row.get("model_slot") == slot]
        bad_parse = [row.get("case_id") for row in selected if not row.get("parse_ok")]
        bad_verdict = [
            row.get("case_id")
            for row in selected
            if (row.get("parsed") or {}).get("verdict_normal") not in VALID_VERDICTS
        ]
        bad_identity = [
            row.get("case_id")
            for row in selected
            if row.get("identity_checks") != {"model": True, "provider": True}
        ]
        missing_provenance = [
            row.get("case_id")
            for row in selected
            if not row.get("response_id")
            or int(row.get("tokens_in", 0)) <= 0
            or int(row.get("tokens_out", 0)) <= 0
        ]
        truncated = [
            row.get("case_id")
            for row in selected
            if str(row.get("finish_reason") or "").lower() in TRUNCATION_REASONS
        ]
        cap = int(spec["accounting_max_output_tokens"])
        indirect_nontruncation = [
            row.get("case_id")
            for row in selected
            if not row.get("finish_reason")
            and row.get("parse_ok")
            and int(row.get("tokens_out", cap)) < cap
        ]
        for label, case_ids in (
            ("parse failures", bad_parse),
            ("invalid verdicts", bad_verdict),
            ("identity mismatches", bad_identity),
            ("missing provenance", missing_provenance),
            ("truncated responses", truncated),
        ):
            if case_ids:
                failures.append(f"{slot} {label}: {case_ids}")
        by_model.append(
            {
                "slot": slot,
                "successful": len(selected),
                "parse_valid": len(selected) - len(bad_parse),
                "identity_valid": len(selected) - len(bad_identity),
                "returned_models": sorted({str(row.get("returned_model")) for row in selected}),
                "returned_providers": sorted(
                    {str(row.get("returned_provider")) for row in selected}
                ),
                "finish_reasons": sorted(
                    {str(row.get("finish_reason")) for row in selected if row.get("finish_reason")}
                ),
                "indirect_nontruncation_cases": indirect_nontruncation,
                "verdict_counts": dict(
                    sorted(
                        Counter(
                            (row.get("parsed") or {}).get("verdict_normal")
                            for row in selected
                        ).items()
                    )
                ),
            }
        )

    return {
        "run_id": manifest["run_id"],
        "accepted": not failures,
        "expected_successes": len(expected_keys),
        "successful_rows": len(successes),
        "historical_error_rows": sum(row.get("status") == "error" for row in rows),
        "failures": failures,
        "by_model": by_model,
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = validate(manifest, load_jsonl(args.results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
