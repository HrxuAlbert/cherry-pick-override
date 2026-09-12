"""Reconstruct the fixed E1 sample using the original Python sampling semantics.

This script uses only the Python standard library.  It maps every public E1
case_id to the exact source split and row index, then verifies the map against
the cached E1 manifest before writing an intermediate JSON file.
"""

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
import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 20260522
SOURCE_DIR = Path("/private/tmp/cco_audit_source")
DEFAULT_OUTPUT_PATH = SOURCE_DIR / "core_source_map.json"
WORKSPACE = _cpo_workspace()
E1_RAW = (
    WORKSPACE
    / "Writing/V0.2/code_release/outputs/option_a_exp/strengthening/"
    "e1_full_4label_utility/raw_results.jsonl"
)


def by_label(examples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row_index, example in enumerate(examples):
        item = dict(example)
        item["_source_row_idx"] = row_index
        grouped[item["label"]].append(item)
    return grouped


def tag(example: dict, split: str, sample_group: str, sample_index: int) -> dict:
    return {
        "split": split,
        "source_row_idx": example["_source_row_idx"],
        "sample_group": sample_group,
        "sample_index": sample_index,
        "label": example["label"],
        "claim": example["claim"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help="Directory containing the pinned dev.json and train.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Destination for the verified JSON source map.",
    )
    return parser.parse_args()


def reconstruct(dev_path: Path, train_path: Path) -> list[dict]:
    rng = random.Random(SEED)
    dev = by_label(json.loads(dev_path.read_text()))
    train = by_label(json.loads(train_path.read_text()))
    selected: list[dict] = []

    for label, count in (
        ("Supported", 50),
        ("Refuted", 50),
        ("Not Enough Evidence", 35),
    ):
        cases = list(dev[label])
        rng.shuffle(cases)
        selected.extend(
            tag(example, "dev", label, index)
            for index, example in enumerate(cases[:count])
        )

    dev_conflicts = list(dev["Conflicting Evidence/Cherrypicking"])
    rng.shuffle(dev_conflicts)
    selected.extend(
        tag(example, "dev", "Conflicting-dev-all", index)
        for index, example in enumerate(dev_conflicts)
    )

    train_conflicts = list(train["Conflicting Evidence/Cherrypicking"])
    rng.shuffle(train_conflicts)
    selected.extend(
        tag(example, "train", "Conflicting-train", index)
        for index, example in enumerate(train_conflicts[:112])
    )

    rng.shuffle(selected)
    for case_id, item in enumerate(selected):
        item["case_id"] = case_id
    return selected


def cached_anchor() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with E1_RAW.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["system"] == "single_haiku_4opt_strong":
                rows[row["case_id"]] = row
    return rows


def main() -> None:
    args = parse_args()
    dev_path = args.source_dir / "dev.json"
    train_path = args.source_dir / "train.json"
    reconstructed = reconstruct(dev_path, train_path)
    cached = cached_anchor()
    if len(reconstructed) != 285 or len(cached) != 285:
        raise RuntimeError(
            f"Expected 285 reconstructed/cached cases; got {len(reconstructed)}/{len(cached)}"
        )

    mismatches: list[dict] = []
    for item in reconstructed:
        row = cached[item["case_id"]]
        checks = {
            "claim": item["claim"] == row["claim"],
            "split": item["split"] == row["split"],
            "sample_group": item["sample_group"] == row["sample_group"],
            "label": item["label"] == row["gold_4way"],
        }
        if not all(checks.values()):
            mismatches.append(
                {
                    "case_id": item["case_id"],
                    "checks": checks,
                    "reconstructed_claim": item["claim"],
                    "cached_claim": row["claim"],
                }
            )
    if mismatches:
        raise RuntimeError(
            "E1 source reconstruction failed: "
            + json.dumps(mismatches[:5], ensure_ascii=False, indent=2)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reconstructed, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "verified",
                "cases": len(reconstructed),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
