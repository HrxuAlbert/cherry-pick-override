#!/usr/bin/env python3
"""Freeze the E1/E2/E4/E5 probe manifests derived from an executed verdict manifest.

Every probe reuses the parent manifest's dataset pin, sample map, model roster, decoding
configuration, and pricing, so that a probe result is comparable to the verdict result on
the same case byte-for-byte. Only the prompt, the per-probe request deltas, and the cost
cap differ.

Generated manifests are candidate-locked: ``status`` is ``candidate``,
``paid_execution_authorized`` is ``false`` and ``authorized_modes`` is empty. Authorizing
a run is a deliberate hand edit, as with the verdict manifests.

Estimands, gates, and interpretation rules for each probe are frozen in
``09_PREANALYSIS_ADDENDUM.md``. This script only materializes the run configuration.
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
from pathlib import Path
from typing import Any

WORKSPACE = _cpo_workspace()
PROMPT_DIR = WORKSPACE / "Writing/V0.2/revision_plan/prompts"

# Addendum §4.1: logprobs are only available on OpenAI-compatible endpoints, and only
# gpt-oss-120b is predeclared for E2.
LETTER_FIRST_SLOTS = ("openweight_gpt_oss_120b",)

PROBE_SPECS: dict[str, dict[str, Any]] = {
    "recognition": {
        "suffix": "e1-recognition",
        "addendum_section": "3",
        "prompts": {"main": "probe_recognition_binary.txt"},
        "system_instruction": "You are a careful reader. Output valid JSON only.",
        "cost_cap_usd": 7.0,
        "expected_usd": 1.6,
        # gpt-oss-120b is a reasoning model on this route and its reasoning tokens
        # count against max_completion_tokens. At 192 the JSON was truncated and the
        # provider rejected it (502, "failed to generate JSON"). 512 is the value the
        # parent verdict matrix ran at, where gpt-oss averaged 150 output tokens.
        "accounting_max_output_tokens": 512,
        "slots": None,
        "purpose": "E1 fresh-context binary recognition probe; no verdict vocabulary is shown.",
    },
    "letter_first": {
        "suffix": "e2-letter-logprob",
        "addendum_section": "4",
        "prompts": {"main": "verdict_letter_first.txt"},
        "system_instruction": "You are a fact-verification judge. Answer with one letter only.",
        "cost_cap_usd": 2.0,
        "expected_usd": 0.15,
        # gpt-oss-120b emits reasoning tokens that count against the completion
        # budget, so an 8-token cap truncates before the verdict letter is reached.
        # The parser locates the verdict letter by scanning the emitted tokens.
        "accounting_max_output_tokens": 512,
        "slots": LETTER_FIRST_SLOTS,
        "purpose": "E2 letter-constrained verdict with first-token logprobs.",
        # The parent manifest pins gpt-oss-120b to Groq, which does not serve
        # logprobs: OpenRouter answers such a request with HTTP 404 "no endpoints
        # found that can handle the requested parameters". Of the nine routes that
        # do serve logprobs, Cerebras returns the full top-20 distribution without
        # rate-limiting. This is a DIFFERENT SERVING STACK from the main verdict
        # matrix, so E2 re-measures the verdict on this route and reports it
        # alongside the logprobs; it must never be compared directly against the
        # Groq-served rates in the main table.
        "request_overrides": {
            "provider": {"only": ["cerebras"], "allow_fallbacks": False,
                         "require_parameters": True, "data_collection": "deny"},
            "reasoning": None,
            "response_format": None,
        },
        "provider_substitution": {
            "parent_provider": "groq",
            "probe_provider": "cerebras",
            "reason": "Groq does not serve logprobs for this model on OpenRouter.",
            "comparability": ("Verdicts from this run are re-measured on the Cerebras route and "
                              "are not interchangeable with the Groq-served verdicts in the main "
                              "matrix. A spot check found the two routes disagreeing on at least "
                              "one smoke case, so provider disagreement is reported, not assumed away."),
        },
    },
    "verdict_logprob": {
        "suffix": "e2b-verdict-logprob",
        "addendum_section": "11b",
        "prompts": {"main": "Writing/V0.2/code_release/scripts/option_a_exp/prompts/judges/honest_4opt_strong.txt"},
        "system_instruction": "You are a careful fact-verification judge. Output valid JSON only.",
        "cost_cap_usd": 3.0,
        "expected_usd": 0.12,
        # The verdict template writes a reasoning field before the verdict, and at 512
        # tokens 26/285 responses truncated mid-reasoning. Those are the cases with the
        # longest reasoning, so dropping them would bias any route comparison. 1024 is
        # the value the parent verdict matrix used for this model.
        "accounting_max_output_tokens": 1024,
        "slots": LETTER_FIRST_SLOTS,
        "purpose": ("E2 re-specified: the unmodified four-option verdict prompt with logprobs, "
                    "read at the token that begins the verdict value inside the model's JSON."),
        "request_overrides": {
            "provider": {"only": ["cerebras"], "allow_fallbacks": False,
                         "require_parameters": True, "data_collection": "deny"},
            "reasoning": None,
        },
        "provider_substitution": {
            "parent_provider": "groq",
            "probe_provider": "cerebras",
            "reason": "Groq does not serve logprobs for this model on OpenRouter.",
            "comparability": ("This run re-measures the verdict on the Cerebras route, so its "
                              "verdicts and its logprobs come from one serving stack. The "
                              "Groq-served rates in the main matrix are reported separately and "
                              "the two routes are compared as a serving-stack sensitivity check."),
        },
    },
    "decompose": {
        "suffix": "e4-decompose",
        "addendum_section": "5",
        "prompts": {"stage1": "decompose_stage1.txt", "stage2": "decompose_stage2.txt"},
        "system_instruction": "You are a careful fact-verification judge. Output valid JSON only.",
        "cost_cap_usd": 16.0,
        "expected_usd": 4.5,
        "accounting_max_output_tokens": 640,
        "slots": None,
        "purpose": "E4 decomposition-then-verdict intervention in two fresh contexts.",
    },
    "verdict_variant_neutral": {
        "suffix": "e5-neutral",
        "addendum_section": "6",
        "probe_type": "verdict_variant",
        "prompts": {"main": "honest_4opt_neutral.txt"},
        "system_instruction": "You are a careful fact-verification judge. Output valid JSON only.",
        "cost_cap_usd": 10.0,
        "expected_usd": 2.0,
        # Same output shape as the main verdict template (a reasoning field before
        # the verdict). E2b showed 512 truncates that JSON mid-reasoning on the
        # open-weight routes; 1024 is what the parent verdict matrix used.
        "accounting_max_output_tokens": 1024,
        "slots": None,
        "purpose": "E5 neutral specification: label definitions only, no triggers, no few-shot.",
    },
    "verdict_variant_reduced": {
        "suffix": "e5-reduced",
        "addendum_section": "6",
        "probe_type": "verdict_variant",
        "prompts": {"main": "honest_4opt_concise.txt"},
        "system_instruction": "You are a careful fact-verification judge. Output valid JSON only.",
        "cost_cap_usd": 10.0,
        "expected_usd": 2.0,
        # Same output shape as the main verdict template (a reasoning field before
        # the verdict). E2b showed 512 truncates that JSON mid-reasoning on the
        # open-weight routes; 1024 is what the parent verdict matrix used.
        "accounting_max_output_tokens": 1024,
        "slots": None,
        "purpose": "E5 reduced specification: triggers and few-shot demonstrations removed.",
    },
}


class BuildError(RuntimeError):
    """Raised when a probe manifest cannot be safely frozen."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=WORKSPACE / "Writing/V0.2/revision_plan/current_models_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "outputs/revision_2026/probes/manifests",
    )
    parser.add_argument("--probes", nargs="*", default=sorted(PROBE_SPECS))
    parser.add_argument("--tag", default="20260901-v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_to_workspace(path: Path) -> str:
    # A manifest is normally written inside the workspace, but the builder is also
    # exercised against a temporary directory, so fall back to an absolute path
    # rather than raising when the target sits outside it.
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE))
    except ValueError:
        return str(resolved)


def prompt_entry(filename: str) -> dict[str, str]:
    # A bare name lives in the probe prompt directory; a name containing a separator is
    # workspace-relative, so a probe can reuse the frozen judge prompt in place rather
    # than copying it and risking divergence.
    path = (WORKSPACE / filename) if "/" in filename else (PROMPT_DIR / filename)
    if not path.exists():
        raise BuildError(f"Prompt template is missing: {path}")
    return {"path": relative_to_workspace(path), "sha256": sha256_file(path)}


def build_one(base: dict[str, Any], name: str, spec: dict[str, Any], tag: str) -> dict[str, Any]:
    probe_type = spec.get("probe_type", name)
    manifest = copy.deepcopy(base)

    # Derive the dataset prefix from the parent rather than hardcoding it, so a
    # VitaminC-parented probe is not named "averitec-...".
    dataset_slug = str(base["dataset"].get("name", "dataset")).lower().replace(" ", "-")
    manifest["run_id"] = f"{dataset_slug}-probe-{spec['suffix']}-{tag}"
    manifest["status"] = "candidate"
    manifest["paid_execution_authorized"] = False
    manifest["authorized_modes"] = []
    manifest["created_at"] = tag.split("-")[0]
    manifest["output_root"] = "outputs/revision_2026/probes/runs"
    manifest["max_estimated_cost_usd"] = float(spec["cost_cap_usd"])
    # The runner's estimator charges input at one token per two characters and output at
    # the hard cap for every call, which ran about 3.4x over actual spend on the parent
    # verdict matrix. The cap is a runaway guard, not a forecast; the forecast is here.
    manifest["expected_actual_cost_usd"] = float(spec["expected_usd"])
    manifest["selection_policy"] = (
        "Derived from the executed verdict manifest without changing the dataset pin, the "
        "sample map, the model roster, or decoding configuration. Only the prompt and the "
        "per-probe request deltas differ, so probe and verdict outputs are comparable on the "
        "same case."
    )

    manifest["probe"] = {
        "type": probe_type,
        "name": name,
        "purpose": spec["purpose"],
        "parent_verdict_run_id": base["run_id"],
        "parent_manifest_prompt_sha256": base["prompt"]["template_sha256"],
        "preanalysis_document": "Writing/V0.2/revision_plan/09_PREANALYSIS_ADDENDUM.md",
        "preanalysis_section": spec["addendum_section"],
    }

    prompt_block: dict[str, Any] = {"system_instruction": spec["system_instruction"]}
    for key, filename in spec["prompts"].items():
        prompt_block[key] = prompt_entry(filename)
    # validate_manifest in the verdict runner requires a 'prompt' key; the probe runner
    # reads the per-probe sub-blocks above.
    manifest["prompt"] = prompt_block

    allowed = spec["slots"]
    kept: list[dict[str, Any]] = []
    for model in manifest["models"]:
        model = copy.deepcopy(model)
        if allowed is not None and model["slot"] not in allowed:
            model["enabled"] = False
            model["disabled_reason"] = (
                f"{probe_type} is predeclared for slots {list(allowed)} only "
                f"(09_PREANALYSIS_ADDENDUM.md §{spec['addendum_section']})."
            )
        model["accounting_max_output_tokens"] = int(spec["accounting_max_output_tokens"])
        request = model.setdefault("request", {})
        for key in ("max_completion_tokens", "max_tokens"):
            if key in request:
                request[key] = int(spec["accounting_max_output_tokens"])
        for key, value in (spec.get("request_overrides") or {}).items():
            if value is None:
                request.pop(key, None)
            else:
                request[key] = value
        if spec.get("provider_substitution") and model.get("enabled"):
            model["provider_identity"] = spec["provider_substitution"]["probe_provider"]
            model["provider_substitution"] = spec["provider_substitution"]
        if probe_type == "letter_first" and model.get("enabled"):
            if model["provider_type"] != "openai_compatible":
                raise BuildError(
                    f"letter_first cannot run on {model['slot']}: provider_type is "
                    f"{model['provider_type']} and does not return logprobs."
                )
        kept.append(model)
    manifest["models"] = kept

    if not any(model.get("enabled") for model in manifest["models"]):
        raise BuildError(f"Probe {name} has no enabled model slots.")

    return manifest


def main() -> None:
    args = parse_args()
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    for key in ("dataset", "models", "prompt", "run_id"):
        if key not in base:
            raise BuildError(f"Base manifest lacks {key}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for name in args.probes:
        if name not in PROBE_SPECS:
            raise BuildError(f"Unknown probe {name!r}; known: {sorted(PROBE_SPECS)}")
        manifest = build_one(base, name, PROBE_SPECS[name], args.tag)
        path = args.output_dir / f"{manifest['run_id']}.json"
        if path.exists() and not args.force:
            raise BuildError(f"{path} already exists; pass --force to overwrite.")
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        written.append(
            {
                "probe": name,
                "type": manifest["probe"]["type"],
                "run_id": manifest["run_id"],
                "path": relative_to_workspace(path),
                "manifest_sha256": sha256_file(path),
                "enabled_slots": [m["slot"] for m in manifest["models"] if m.get("enabled")],
                "cost_cap_usd": manifest["max_estimated_cost_usd"],
            }
        )

    print(
        json.dumps(
            {
                "base_manifest": relative_to_workspace(args.base_manifest),
                "base_run_id": base["run_id"],
                "written": written,
                "status": "candidate-locked; paid execution not authorized",
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
