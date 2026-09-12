#!/usr/bin/env python3
"""Regenerate every experiment artifact that needs neither human labels nor API calls."""

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
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
SCRIPT_DIR = Path(__file__).resolve().parent


class SuiteError(RuntimeError):
    """Raised when any audit-independent experiment fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=WORKSPACE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise SuiteError(
            f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return {
        "command": command,
        "returncode": result.returncode,
        "last_output_line": result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "",
    }


def main() -> None:
    args = parse_args()
    force = ["--force"] if args.force else []
    python = sys.executable
    commands = [
        [python, str(SCRIPT_DIR / "analyze_cached_model_landscape.py"), *force],
        [python, str(SCRIPT_DIR / "analyze_feature_signals.py"), *force],
        [python, str(SCRIPT_DIR / "analyze_error_taxonomy.py"), *force],
        [python, str(SCRIPT_DIR / "profile_experiment_datasets.py"), *force],
        [
            python,
            str(SCRIPT_DIR / "run_current_models.py"),
            "--manifest",
            "Writing/V0.2/revision_plan/current_models_manifest.json",
            "--mode",
            "dry-run",
        ],
        [
            python,
            str(SCRIPT_DIR / "run_current_models.py"),
            "--manifest",
            "Writing/V0.2/revision_plan/current_models_manifest.json",
            "--mode",
            "smoke",
        ],
        [
            python,
            str(SCRIPT_DIR / "run_current_models.py"),
            "--manifest",
            "Writing/V0.2/revision_plan/current_models_vitaminc_manifest.json",
            "--mode",
            "dry-run",
        ],
        [
            python,
            str(SCRIPT_DIR / "run_current_models.py"),
            "--manifest",
            "Writing/V0.2/revision_plan/current_models_vitaminc_manifest.json",
            "--mode",
            "smoke",
        ],
    ]
    completed = [run(command) for command in commands]
    artifacts = [
        WORKSPACE / "outputs/revision_2026/audit_independent/cached_landscape/results.json",
        WORKSPACE / "outputs/revision_2026/audit_independent/feature_signals/cv_metrics.csv",
        WORKSPACE / "outputs/revision_2026/audit_independent/error_taxonomy/taxonomy_metrics.csv",
        WORKSPACE / "outputs/revision_2026/audit_independent/dataset_profiles/dataset_profiles.csv",
        WORKSPACE
        / "outputs/revision_2026/current_models_averitec/runs/averitec-p0-current-models-20260818-v1/plan_dry-run.json",
        WORKSPACE
        / "outputs/revision_2026/current_models_averitec/runs/averitec-p0-current-models-20260818-v1/plan_smoke.json",
        WORKSPACE
        / "outputs/revision_2026/current_models_vitaminc/runs/vitaminc-p0-current-models-20260831-v1/plan_dry-run.json",
        WORKSPACE
        / "outputs/revision_2026/current_models_vitaminc/runs/vitaminc-p0-current-models-20260831-v1/plan_smoke.json",
    ]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise SuiteError(f"Suite artifacts are missing: {missing}")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete_no_human_no_api",
        "human_audit_accessed": False,
        "api_calls_made": 0,
        "python": python,
        "commands": completed,
        "artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in artifacts
        ],
    }
    output = WORKSPACE / "outputs/revision_2026/audit_independent/SUITE_STATUS.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "commands": len(completed), "artifacts": len(artifacts), "report": str(output)}))


if __name__ == "__main__":
    main()
