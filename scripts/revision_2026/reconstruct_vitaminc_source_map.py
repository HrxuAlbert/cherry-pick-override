#!/usr/bin/env python3
"""Reconstruct the fixed 250-case VitaminC-Mixed sample with full evidence."""

from __future__ import annotations

def _cpo_workspace(_f=__file__):
    """Resolve the release root.

    In the author's tree these scripts live at
    <root>/Writing/V0.2/revision_plan/scripts/, so parents[4] is the root. In
    this release they live at <repo>/scripts/revision_2026/, so walk up until a
    directory containing outputs/revision_2026 is found and fall back to the
    original rule.
    """
    import os, pathlib as _p
    env = os.environ.get("CPO_WORKSPACE")
    if env:
        return _p.Path(env).resolve()
    here = _p.Path(_f).resolve()
    for parent in here.parents:
        if (parent / "outputs" / "revision_2026").is_dir():
            return parent
    return here.parents[4]

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
SEED = 20260522
EXPECTED_SOURCE_SHA256 = "544934677f5d133873e6d38f4557f8966f4efa5d3d70874ffe6913f2091b86b5"
GOLD = {
    "SUPPORTS": ("Supported", "support"),
    "REFUTES": ("Refuted", "refute"),
    "NOT ENOUGH INFO": ("Not Enough Evidence", "insufficient"),
}


class ReconstructionError(RuntimeError):
    """Raised when VitaminC no longer reproduces the published fixed sample."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/private/tmp/vitaminc_dev.jsonl"))
    parser.add_argument(
        "--anchor-results",
        type=Path,
        default=WORKSPACE
        / "Writing/V0.2/code_release/outputs/option_a_exp/strengthening/e4_vitaminc_mixed/raw_results.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE
        / "outputs/revision_2026/current_models_vitaminc/input/vitaminc_mixed_source_map.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ReconstructionError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def reconstruct(source_rows: list[dict[str, Any]], seed: int = SEED) -> list[dict[str, Any]]:
    by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, row in enumerate(source_rows):
        by_claim[str(row["claim"])].append({**row, "source_index": source_index})

    rng = random.Random(seed)
    pair_claims = [
        (claim, versions)
        for claim, versions in by_claim.items()
        if {version["label"] for version in versions} >= {"SUPPORTS", "REFUTES"}
    ]
    rng.shuffle(pair_claims)
    sample: list[dict[str, Any]] = []
    used_claims: set[str] = set()
    for claim, versions in pair_claims:
        if len(sample) >= 100:
            break
        support = next(version for version in versions if version["label"] == "SUPPORTS")
        refute = next(version for version in versions if version["label"] == "REFUTES")
        if rng.random() < 0.5:
            ordered = [("support", support), ("refute", refute)]
        else:
            ordered = [("refute", refute), ("support", support)]
        evidence_blocks = [
            f"Evidence {letter}: {version['evidence']}"
            for letter, (_, version) in zip(("A", "B"), ordered)
        ]
        case_id = len(sample)
        used_claims.add(claim)
        sample.append(
            {
                "case_id": case_id,
                "claim": claim,
                "evidence_blocks": evidence_blocks,
                "evidence": "\n\n".join(evidence_blocks),
                "label": "Conflicting Evidence/Cherrypicking",
                "gold_normal": "conflicting",
                "split": "vitaminc-validation",
                "sample_group": "mixed-composite",
                "source": "vitaminc-mixed",
                "source_row_idx": case_id,
                "source_unique_ids": [version.get("unique_id", "") for _, version in ordered],
                "source_indices": [version["source_index"] for _, version in ordered],
                "evidence_direction_order": [direction for direction, _ in ordered],
            }
        )

    rng_controls = random.Random(seed + 1)
    pools: dict[str, list[dict[str, Any]]] = {label: [] for label in GOLD}
    for source_index, row in enumerate(source_rows):
        if row["claim"] in used_claims or row["label"] not in pools:
            continue
        pools[row["label"]].append({**row, "source_index": source_index})
    for pool in pools.values():
        rng_controls.shuffle(pool)

    for source_label, group in (
        ("SUPPORTS", "pure-supports"),
        ("REFUTES", "pure-refutes"),
        ("NOT ENOUGH INFO", "pure-nei"),
    ):
        label, gold_normal = GOLD[source_label]
        for row in pools[source_label][:50]:
            case_id = len(sample)
            evidence = str(row["evidence"])
            sample.append(
                {
                    "case_id": case_id,
                    "claim": row["claim"],
                    "evidence_blocks": [evidence],
                    "evidence": evidence,
                    "label": label,
                    "gold_normal": gold_normal,
                    "split": "vitaminc-validation",
                    "sample_group": group,
                    "source": f"vitaminc-{group}",
                    "source_row_idx": case_id,
                    "source_unique_ids": [row.get("unique_id", "")],
                    "source_indices": [row["source_index"]],
                    "evidence_direction_order": [gold_normal],
                }
            )
    return sample


def load_anchors(path: Path) -> dict[int, dict[str, str]]:
    anchors: dict[int, dict[str, str]] = {}
    for row in read_jsonl(path):
        if row.get("system") != "panel_3judge_4opt_strong":
            continue
        case_id = int(row["case_id"])
        anchors[case_id] = {
            "claim": str(row["claim"]),
            "gold_normal": str(row["gold_normal"]),
            "source": str(row.get("source", "")),
        }
    return anchors


def validate(sample: list[dict[str, Any]], anchors: dict[int, dict[str, str]]) -> None:
    expected_counts = {"conflicting": 100, "support": 50, "refute": 50, "insufficient": 50}
    counts = Counter(row["gold_normal"] for row in sample)
    if len(sample) != 250 or dict(counts) != expected_counts:
        raise ReconstructionError(f"Reconstructed composition mismatch: {len(sample)}, {dict(counts)}")
    if set(anchors) != set(range(250)):
        raise ReconstructionError(f"Published anchor does not contain exactly 250 cases.")
    mismatches: list[int] = []
    for row in sample:
        anchor = anchors[row["case_id"]]
        if row["claim"] != anchor["claim"] or row["gold_normal"] != anchor["gold_normal"]:
            mismatches.append(row["case_id"])
    if mismatches:
        raise ReconstructionError(
            f"Reconstruction differs from published cache at cases: {mismatches[:10]}"
        )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if sha256_file(source) != EXPECTED_SOURCE_SHA256:
        raise ReconstructionError(
            f"VitaminC source SHA-256 mismatch: {sha256_file(source)}"
        )
    sample = reconstruct(read_jsonl(source))
    validate(sample, load_anchors(args.anchor_results.resolve()))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "exact_anchor_match",
                "cases": len(sample),
                "source_sha256": EXPECTED_SOURCE_SHA256,
                "output": str(output),
                "output_sha256": sha256_file(output),
            }
        )
    )


if __name__ == "__main__":
    main()
