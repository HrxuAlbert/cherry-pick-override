#!/usr/bin/env python3
"""Freeze prompt, evidence-order, and reasoning mechanism manifests.

Run this only after 60 source case IDs have been selected from the frozen
strict human-consensus preserve set. The baseline run is reused across all
three comparisons, avoiding duplicate API calls. Generated manifests remain
candidate-locked and cannot spend money until separately authorized.
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
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
DEFAULT_SLOTS = ("closed_openai_current", "openweight_gpt_oss_120b")
VARIANTS = (
    "baseline_primary_original_direct",
    "prompt_concise",
    "prompt_permuted",
    "evidence_reversed",
    "reasoning_medium",
)


class ManifestBuildError(RuntimeError):
    """Raised when the mechanism experiment cannot be safely frozen."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=WORKSPACE / "Writing/V0.2/revision_plan/current_models_manifest.json",
    )
    parser.add_argument("--strict-case-ids", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/mechanisms/manifests",
    )
    parser.add_argument("--model-slots", nargs="+", default=list(DEFAULT_SLOTS))
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else WORKSPACE / path


def relative_to_workspace(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE))
    except ValueError:
        return os.path.relpath(path.resolve(), WORKSPACE)


def parse_case_ids(path: Path, expected: int = 60) -> list[int]:
    raw = path.read_text(encoding="utf-8").strip()
    try:
        values = json.loads(raw) if raw.startswith("[") else [
            line.strip() for line in raw.splitlines() if line.strip()
        ]
        case_ids = [int(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestBuildError(f"Invalid case-ID file: {path}") from exc
    if len(case_ids) != expected or len(set(case_ids)) != expected:
        raise ManifestBuildError(
            f"Expected {expected} unique strict-consensus case IDs; found "
            f"{len(case_ids)} rows and {len(set(case_ids))} unique IDs."
        )
    return case_ids


def validate_conflict_candidates(base: dict[str, Any], case_ids: list[int]) -> None:
    sample_map_path = workspace_path(base["dataset"]["sample_map_path"])
    sample_map = json.loads(sample_map_path.read_text(encoding="utf-8"))
    by_id = {int(row["case_id"]): row for row in sample_map}
    unknown = sorted(set(case_ids) - set(by_id))
    if unknown:
        raise ManifestBuildError(f"Unknown source case IDs: {unknown[:10]}")
    invalid = [
        case_id
        for case_id in case_ids
        if by_id[case_id].get("label") != "Conflicting Evidence/Cherrypicking"
    ]
    if invalid:
        raise ManifestBuildError(
            f"Mechanism subset contains non-conflict candidates: {invalid[:10]}"
        )


def set_reasoning_medium(model: dict[str, Any]) -> None:
    request = model.setdefault("request", {})
    slot = model["slot"]
    if slot == "closed_openai_current":
        request["reasoning_effort"] = "medium"
    elif slot == "openweight_gpt_oss_120b":
        reasoning = request.setdefault("reasoning", {})
        reasoning["effort"] = "medium"
    else:
        raise ManifestBuildError(
            f"No frozen medium-reasoning mapping for selected slot: {slot}"
        )


def make_variants(
    base: dict[str, Any],
    case_ids_path: Path,
    case_ids_sha256: str,
    case_ids: list[int],
    model_slots: list[str],
) -> dict[str, dict[str, Any]]:
    available = {str(model["slot"]): model for model in base["models"]}
    unknown = sorted(set(model_slots) - set(available))
    if unknown:
        raise ManifestBuildError(f"Unknown model slots: {unknown}")
    if len(model_slots) != 2:
        raise ManifestBuildError("Mechanism experiments require exactly two model slots.")

    prompt_dir = WORKSPACE / "Writing/V0.2/revision_plan/prompts"
    prompt_paths = {
        "prompt_concise": prompt_dir / "honest_4opt_concise.txt",
        "prompt_permuted": prompt_dir / "honest_4opt_permuted.txt",
    }
    for path in prompt_paths.values():
        if not path.is_file():
            raise ManifestBuildError(f"Missing prompt variant: {path}")

    output: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        manifest = copy.deepcopy(base)
        manifest["run_id"] = f"averitec-mechanism-{variant}-v1"
        manifest["status"] = "candidate_locked"
        manifest["paid_execution_authorized"] = False
        manifest["output_root"] = "outputs/revision_2026/mechanisms/runs"
        manifest["max_estimated_cost_usd"] = 5.0
        manifest["models"] = [copy.deepcopy(available[slot]) for slot in model_slots]
        manifest["selection"] = {
            "case_ids_path": relative_to_workspace(case_ids_path),
            "case_ids_sha256": case_ids_sha256,
            "expected_count": len(case_ids),
            "source": "strict_human_consensus_preserve_after_unblinding",
        }
        manifest["smoke_case_ids"] = case_ids[:5]
        manifest["smoke_expected_counts"] = {"conflicting": 5}
        manifest["dataset"]["evidence_order"] = "original"
        manifest["mechanism"] = {
            "variant": variant,
            "baseline_variant": "baseline_primary_original_direct",
            "case_set_shared": True,
            "selected_before_variant_outputs": True,
        }
        if variant in prompt_paths:
            prompt_path = prompt_paths[variant]
            manifest["prompt"]["template_path"] = relative_to_workspace(prompt_path)
            manifest["prompt"]["template_sha256"] = sha256_file(prompt_path)
        elif variant == "evidence_reversed":
            manifest["dataset"]["evidence_order"] = "reversed"
        elif variant == "reasoning_medium":
            for model in manifest["models"]:
                set_reasoning_medium(model)
        output[variant] = manifest
    return output


def main() -> None:
    args = parse_args()
    base_path = args.base_manifest.resolve()
    strict_path = args.strict_case_ids.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise ManifestBuildError(f"Output directory is nonempty; use --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    case_ids = parse_case_ids(strict_path)
    validate_conflict_candidates(base, case_ids)
    selection_copy = output_dir / "strict_case_ids.txt"
    selection_copy.write_text("".join(f"{case_id}\n" for case_id in case_ids), encoding="utf-8")
    selection_sha256 = sha256_file(selection_copy)
    variants = make_variants(
        base,
        selection_copy,
        selection_sha256,
        case_ids,
        args.model_slots,
    )
    index: dict[str, Any] = {
        "schema_version": "1.0",
        "human_audit_labels_read_by_builder": False,
        "selection_source": "provided source case IDs only",
        "baseline_variant": "baseline_primary_original_direct",
        "case_count": len(case_ids),
        "model_slots": args.model_slots,
        "planned_calls_unique": len(case_ids) * len(args.model_slots) * len(variants),
        "variants": {},
    }
    for variant, manifest in variants.items():
        path = output_dir / f"{variant}.json"
        payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        path.write_text(payload, encoding="utf-8")
        index["variants"][variant] = {
            "manifest_path": relative_to_workspace(path),
            "manifest_sha256": sha256_bytes(payload.encode("utf-8")),
            "run_id": manifest["run_id"],
        }
    index_path = output_dir / "mechanism_manifest_index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "candidate_locked_no_api_calls",
                "variants": len(variants),
                "case_count": len(case_ids),
                "models": len(args.model_slots),
                "planned_calls_unique": index["planned_calls_unique"],
                "index": str(index_path),
            }
        )
    )


if __name__ == "__main__":
    main()
