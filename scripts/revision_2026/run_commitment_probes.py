#!/usr/bin/env python3
"""Manifest-driven runner for the commitment-recognition probes (E1, E2, E4, E5).

Implements the four probe families frozen in ``09_PREANALYSIS_ADDENDUM.md``:

``recognition``
    E1. Fresh-context binary question about evidence structure. No verdict vocabulary
    reaches the model. One call per case.

``letter_first``
    E2. Letter-constrained verdict with token logprobs, so that the probability mass on
    the authorized non-directional option is observable at the moment of commitment.
    One call per case. Requires an endpoint that returns logprobs.

``decompose``
    E4. Two stages in fresh contexts: the model first decomposes the evidence into
    supporting and refuting items with no verdict vocabulary, then issues the verdict
    with its own decomposition in context. Two calls per case; stage 2 is skipped when
    stage 1 fails.

``verdict_variant``
    E5. The frozen four-option verdict prompt under an alternative specification
    strength. One call per case.

Execution discipline is inherited unchanged from ``run_current_models.py``: network
calls require ``--execute``, a frozen and paid-authorized manifest, and
``CCO_ALLOW_PAID_RUN=I_UNDERSTAND``. Results are append-only JSONL and resume by the
hash of the exact model request rather than by case ID.

This runner never reads human-audit artifacts and never reads the parent verdict run.
Stratification against verdict outputs happens in ``analyze_commitment_probes.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_current_models import (  # noqa: E402
    EXECUTION_ACK,
    RunnerError,
    accumulated_estimated_cost_usd,
    append_jsonl,
    atomic_write_json,
    build_request,
    canonical_json,
    completed_request_keys,
    estimate_cost_upper_bound,
    http_json,
    load_cases,
    load_selected_cases,
    normalize_verdict,
    parse_provider_response,
    parse_verdict,
    post_call_delay_seconds,
    require_sha,
    resolve_workspace_path,
    select_models,
    sha256_text,
    utc_now,
    validate_manifest,
)

PROBE_TYPES = (
    "recognition",
    "letter_first",
    "verdict_logprob",
    "decompose",
    "verdict_variant",
)

# Verdict words as they appear in the required JSON schema. Their first letters are
# distinct, so a token that begins a verdict value maps unambiguously by prefix.
VERDICT_WORDS = {
    "SUPPORTS": "support",
    "REFUTES": "refute",
    "CONFLICTING": "conflicting",
    "NOT_ENOUGH_INFO": "insufficient",
}

# Matches the JSON up to and including the opening quote of the verdict value, so the
# next emitted token is the first token of the verdict word itself.
VERDICT_VALUE_PREFIX = re.compile(r'"verdict"\s*:\s*"$')

# Frozen in 09_PREANALYSIS_ADDENDUM.md §4.1. Do not reorder: the analysis maps ranks
# by verdict name, not by position.
LETTER_TO_VERDICT = {
    "A": "support",
    "B": "refute",
    "C": "conflicting",
    "D": "insufficient",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "full"),
        default="dry-run",
        help="dry-run validates everything; smoke runs the manifest's smoke case IDs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call model endpoints. Requires all execution locks.",
    )
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #


def render(template: str, fields: dict[str, str]) -> str:
    """Replace <<NAME>> placeholders. Every placeholder in the template must be supplied."""
    rendered = template
    for name, value in fields.items():
        token = f"<<{name}>>"
        if token not in rendered:
            raise RunnerError(f"Prompt template does not contain {token}.")
        rendered = rendered.replace(token, value)
    if "<<" in rendered and ">>" in rendered:
        raise RunnerError("Prompt template still contains an unfilled <<PLACEHOLDER>>.")
    return rendered


def load_template(spec: dict[str, Any], label: str) -> str:
    path = resolve_workspace_path(spec["path"])
    require_sha(path, str(spec["sha256"]), f"{label} prompt")
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Probe-specific parsers
# --------------------------------------------------------------------------- #


def parse_recognition(text: str) -> dict[str, Any] | None:
    """E1. Requires a yes/no answer and treats a claimed direction without a span as absent."""
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    raw = str(parsed.get("materially_two_sided", "")).strip().lower()
    if raw in {"yes", "true"}:
        answer = "yes"
    elif raw in {"no", "false"}:
        answer = "no"
    else:
        return None
    support_span = str(parsed.get("support_span", "") or "").strip()
    refute_span = str(parsed.get("refute_span", "") or "").strip()
    return {
        "recognition": answer,
        "support_span": support_span,
        "refute_span": refute_span,
        # Addendum §3.1: a direction claimed without a quotable span is not present.
        # Recorded alongside the raw answer; the analysis reports both.
        "recognition_span_backed": "yes" if (support_span and refute_span) else "no",
        "note": str(parsed.get("note", "")),
        "response_object": parsed,
    }


def parse_letter_first(text: str, logprob_content: Any) -> dict[str, Any] | None:
    """E2. Reads the emitted letter and the full top-k distribution over the first token."""
    # The prompt requires the letter to be the first emitted character. Scanning the whole
    # string for any A/B/C/D would silently read "I cannot answer." as CONFLICTING, so the
    # letter is only accepted in first position and only when it does not begin a word.
    stripped = text.strip().lstrip("\"'`* \t\n").upper()
    if not stripped:
        return None
    letter = stripped[0]
    if letter not in LETTER_TO_VERDICT:
        return None
    if len(stripped) > 1 and stripped[1].isalpha():
        return None

    distribution: dict[str, float] = {}
    first_token_alternatives: list[dict[str, Any]] = []
    verdict_position: int | None = None
    if isinstance(logprob_content, list) and logprob_content:
        # The verdict letter is not always token 0: a reasoning model may emit
        # tokens before it, and some routes prepend whitespace or a quote. Scan for
        # the first emitted token that is itself a verdict letter, and read the
        # distribution at that position. Falling back to position 0 blindly would
        # report the logprobs of a preamble token as if they were the verdict's.
        for index, entry in enumerate(logprob_content):
            token = str(entry.get("token", "")).strip().strip("\"'`*").upper()
            if token in LETTER_TO_VERDICT:
                verdict_position = index
                break
        if verdict_position is None:
            return {
                "verdict_raw": letter,
                "verdict_normal": LETTER_TO_VERDICT[letter],
                "letter_distribution": {},
                "letter_ranks": {},
                "conflicting_rank": None,
                "p_conflicting": None,
                "p_chosen": None,
                "margin": None,
                "logprobs_available": False,
                "verdict_token_position": None,
                "emitted_token_count": len(logprob_content),
                "first_token_alternatives": [],
            }
        first = logprob_content[verdict_position]
        for alternative in first.get("top_logprobs") or []:
            token = str(alternative.get("token", "")).strip().upper()
            try:
                probability = float(2.718281828459045 ** float(alternative["logprob"]))
            except (KeyError, TypeError, ValueError):
                continue
            first_token_alternatives.append(
                {"token": alternative.get("token"), "logprob": alternative.get("logprob")}
            )
            if token in LETTER_TO_VERDICT:
                verdict = LETTER_TO_VERDICT[token]
                distribution[verdict] = distribution.get(verdict, 0.0) + probability

    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    ranks = {verdict: index + 1 for index, (verdict, _) in enumerate(ranked)}
    chosen = LETTER_TO_VERDICT[letter]
    p_conflicting = distribution.get("conflicting")
    p_chosen = distribution.get(chosen)
    return {
        "verdict_raw": letter,
        "verdict_normal": chosen,
        "letter_distribution": distribution,
        "letter_ranks": ranks,
        "conflicting_rank": ranks.get("conflicting"),
        "p_conflicting": p_conflicting,
        "p_chosen": p_chosen,
        "margin": (
            None
            if p_chosen is None or p_conflicting is None
            else round(p_chosen - p_conflicting, 8)
        ),
        "logprobs_available": bool(distribution),
        "verdict_token_position": verdict_position,
        "emitted_token_count": len(logprob_content) if isinstance(logprob_content, list) else None,
        "first_token_alternatives": first_token_alternatives,
    }


def token_to_verdict(token: str) -> str | None:
    """Map a token that begins a verdict value to its normalized verdict."""
    cleaned = str(token or "").strip().strip('"').upper()
    if not cleaned:
        return None
    for word, normal in VERDICT_WORDS.items():
        if word.startswith(cleaned) or cleaned.startswith(word):
            return normal
    return None


def parse_verdict_logprob(text: str, logprob_content: Any) -> dict[str, Any] | None:
    """E2 (re-specified, addendum §11b).

    Reads the verdict from the model's own required JSON, then locates the emitted
    token that begins the verdict value and reports the top-k distribution there.
    At that position the competing candidates are the verdict words themselves,
    unlike the letter-first design, where they were surface-form tokens.
    """
    parsed = parse_verdict(text)
    if parsed is None:
        return None

    distribution: dict[str, float] = {}
    alternatives: list[dict[str, Any]] = []
    position: int | None = None
    verdict_competitors = 0

    if isinstance(logprob_content, list) and logprob_content:
        running = ""
        for index, entry in enumerate(logprob_content):
            if VERDICT_VALUE_PREFIX.search(running):
                position = index
                break
            running += str(entry.get("token", ""))
        if position is not None:
            for alternative in logprob_content[position].get("top_logprobs") or []:
                token = alternative.get("token")
                try:
                    probability = 2.718281828459045 ** float(alternative["logprob"])
                except (KeyError, TypeError, ValueError):
                    continue
                alternatives.append({"token": token, "logprob": alternative.get("logprob")})
                verdict = token_to_verdict(token)
                if verdict is not None:
                    verdict_competitors += 1
                    distribution[verdict] = min(1.0, distribution.get(verdict, 0.0) + probability)

    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    ranks = {verdict: index + 1 for index, (verdict, _) in enumerate(ranked)}
    chosen = parsed["verdict_normal"]
    p_conflicting = distribution.get("conflicting")
    p_chosen = distribution.get(chosen)
    return {
        **parsed,
        "verdict_distribution": distribution,
        "verdict_ranks": ranks,
        "conflicting_rank": ranks.get("conflicting"),
        "p_conflicting": p_conflicting,
        "p_chosen": p_chosen,
        "margin": (
            None
            if p_chosen is None or p_conflicting is None
            else round(p_chosen - p_conflicting, 8)
        ),
        "logprobs_available": bool(distribution),
        "verdict_token_position": position,
        # Sanity check from addendum §11b: if the located position is right, most of
        # its top-k mass should sit on verdict words rather than punctuation.
        "verdict_words_in_topk": verdict_competitors,
        "topk_returned": len(alternatives),
        "verdict_token_alternatives": alternatives,
    }


def parse_decomposition(text: str) -> dict[str, Any] | None:
    """E4 stage 1. Verdict vocabulary in a stage-1 output is a protocol violation."""
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    supporting = parsed.get("supporting_items")
    refuting = parsed.get("refuting_items")
    if not isinstance(supporting, list) or not isinstance(refuting, list):
        return None
    supporting = [str(item) for item in supporting if str(item).strip()]
    refuting = [str(item) for item in refuting if str(item).strip()]
    leaked = normalize_verdict(text) is not None and len(text.strip()) < 40
    return {
        "supporting_items": supporting,
        "refuting_items": refuting,
        "n_supporting": len(supporting),
        "n_refuting": len(refuting),
        "decomposition_two_sided": "yes" if supporting and refuting else "no",
        "verdict_vocabulary_leaked": leaked,
        "response_object": parsed,
    }


def format_decomposition(decomposition: dict[str, Any]) -> str:
    def block(title: str, items: list[str]) -> str:
        if not items:
            return f"{title}: (none)"
        lines = "\n".join(f"  - {item}" for item in items)
        return f"{title}:\n{lines}"

    return "\n".join(
        [
            block("Supporting items", decomposition["supporting_items"]),
            "",
            block("Refuting items", decomposition["refuting_items"]),
        ]
    )


# --------------------------------------------------------------------------- #
# Request construction
# --------------------------------------------------------------------------- #


def probe_request_overrides(probe_type: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Per-probe request deltas layered on the manifest's per-model request block."""
    overrides: dict[str, Any] = {}
    if probe_type == "verdict_logprob":
        if spec["provider_type"] != "openai_compatible":
            raise RunnerError(
                f"verdict_logprob requires an OpenAI-compatible endpoint; {spec['slot']} is "
                f"{spec['provider_type']}."
            )
        # JSON mode stays on: the prompt is the unmodified verdict template and the
        # parser locates the verdict value inside the model's own JSON.
        overrides["logprobs"] = True
        overrides["top_logprobs"] = 20
        return overrides
    if probe_type == "letter_first":
        if spec["provider_type"] != "openai_compatible":
            raise RunnerError(
                f"letter_first requires an OpenAI-compatible endpoint; {spec['slot']} is "
                f"{spec['provider_type']}. Disable the slot or drop it from --models."
            )
        overrides["logprobs"] = True
        overrides["top_logprobs"] = 20
        # A letter answer needs no JSON envelope; the response_format from the parent
        # manifest would force one and defeat first-token constraint.
        overrides["response_format"] = None
    return overrides


def apply_overrides(spec: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(spec))
    request = merged.setdefault("request", {})
    for key, value in overrides.items():
        if value is None:
            request.pop(key, None)
        else:
            request[key] = value
    return merged


def request_signature(
    spec: dict[str, Any], system: str, prompt: str, stage: str, run_id: str
) -> str:
    return sha256_text(
        canonical_json(
            {
                "run_id": run_id,
                "stage": stage,
                "slot": spec["slot"],
                "model_id": spec["model_id"],
                "checkpoint_id": spec.get("checkpoint_id"),
                "provider_identity": spec.get("provider_identity"),
                "base_url": spec.get("base_url"),
                "request": spec.get("request", {}),
                "system": system,
                "prompt": prompt,
            }
        )
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def build_plan(
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    models: list[dict[str, Any]],
    templates: dict[str, str],
    system: str,
) -> list[dict[str, Any]]:
    """One planned unit per (model, case). A decompose unit carries two stages."""
    probe_type = manifest["probe"]["type"]
    run_id = str(manifest["run_id"])
    plan: list[dict[str, Any]] = []
    for spec in models:
        effective = apply_overrides(spec, probe_request_overrides(probe_type, spec))
        for case in cases:
            fields = {"CLAIM": case["claim"], "EVIDENCE": case["evidence"]}
            if probe_type == "decompose":
                stage1_prompt = render(templates["stage1"], fields)
                unit = {
                    "spec": effective,
                    "case": case,
                    "stages": [
                        {
                            "name": "stage1_decompose",
                            "prompt": stage1_prompt,
                            "request_sha256": request_signature(
                                effective, system, stage1_prompt, "stage1_decompose", run_id
                            ),
                        }
                    ],
                    "deferred_stage": "stage2_verdict",
                }
            else:
                prompt = render(templates["main"], fields)
                stage_name = {
                    "recognition": "recognition",
                    "letter_first": "letter_first",
                    "verdict_logprob": "verdict_logprob",
                    "verdict_variant": "verdict_variant",
                }[probe_type]
                unit = {
                    "spec": effective,
                    "case": case,
                    "stages": [
                        {
                            "name": stage_name,
                            "prompt": prompt,
                            "request_sha256": request_signature(
                                effective, system, prompt, stage_name, run_id
                            ),
                        }
                    ],
                    "deferred_stage": None,
                }
            plan.append(unit)
    return plan


def costing_rows(plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-case prompts for one model, in the shape estimate_cost_upper_bound expects.

    That function multiplies by the model roster itself, so these rows must cover the
    cases once, not the model-by-case cross product. Prompts are identical across models
    (same template, same evidence bytes), so the first model's units are representative.
    """
    if not plan:
        return []
    first_slot = plan[0]["spec"]["slot"]
    rows: list[dict[str, Any]] = []
    for unit in plan:
        if unit["spec"]["slot"] != first_slot:
            continue
        for stage in unit["stages"]:
            rows.append({"user_prompt": stage["prompt"]})
        if unit["deferred_stage"] is not None:
            # Stage 2's prompt length is unknown before stage 1 returns. Charge it as a
            # copy of the stage-1 prompt plus a 2000-character decomposition allowance.
            rows.append({"user_prompt": unit["stages"][0]["prompt"] + " " * 2000})
    return rows


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def call_model(
    spec: dict[str, Any],
    system: str,
    prompt: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    url, headers, payload = build_request(spec, system, prompt)
    response, _ = http_json(url, headers, payload, int(manifest.get("timeout_seconds", 180)))
    parsed = parse_provider_response(
        spec["provider_type"], response, spec.get("provider_identity")
    )
    logprob_content = None
    if spec["provider_type"] == "openai_compatible":
        try:
            logprob_content = (response["choices"][0].get("logprobs") or {}).get("content")
        except (KeyError, IndexError, TypeError):
            logprob_content = None
    parsed["logprob_content"] = logprob_content
    return parsed


def interpret(
    probe_type: str, stage_name: str, text: str, logprob_content: Any
) -> dict[str, Any] | None:
    if stage_name == "recognition":
        return parse_recognition(text)
    if stage_name == "letter_first":
        return parse_letter_first(text, logprob_content)
    if stage_name == "verdict_logprob":
        return parse_verdict_logprob(text, logprob_content)
    if stage_name == "stage1_decompose":
        return parse_decomposition(text)
    if stage_name in {"verdict_variant", "stage2_verdict"}:
        return parse_verdict(text)
    raise RunnerError(f"Unknown stage {stage_name!r} for probe {probe_type!r}.")


def run_stage(
    unit: dict[str, Any],
    stage: dict[str, Any],
    manifest: dict[str, Any],
    system: str,
    results_path: Path,
    completed: set[str],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Execute one stage. Returns the parsed payload, or None when nothing usable came back."""
    spec = unit["spec"]
    case = unit["case"]
    probe_type = manifest["probe"]["type"]

    if stage["request_sha256"] in completed:
        return {"__resumed__": True}

    if not args.execute:
        return None

    max_retries = int(manifest.get("max_retries", 1))
    attempt = 0
    last_error: str | None = None
    while attempt <= max_retries:
        attempt += 1
        error: str | None = None
        payload: dict[str, Any] | None = None
        interpreted: dict[str, Any] | None = None
        try:
            payload = call_model(spec, system, stage["prompt"], manifest)
            interpreted = interpret(
                probe_type, stage["name"], payload["text"], payload.get("logprob_content")
            )
        except RunnerError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
            error = f"{type(exc).__name__}: {exc}"

        row: dict[str, Any] = {
            "timestamp": utc_now(),
            "run_id": manifest["run_id"],
            "probe_type": probe_type,
            "stage": stage["name"],
            "attempt": attempt,
            "model_slot": spec["slot"],
            "model_id": spec["model_id"],
            "checkpoint_id": spec.get("checkpoint_id"),
            "provider_identity": spec.get("provider_identity"),
            "case_id": case["case_id"],
            "gold_normal": case["gold_normal"],
            "sample_group": case.get("sample_group"),
            "evidence_sha256": case["evidence_sha256"],
            "prompt_sha256": sha256_text(stage["prompt"]),
            "request_sha256": stage["request_sha256"],
            "parent_verdict_run_id": manifest["probe"].get("parent_verdict_run_id"),
        }
        if error is not None:
            row.update({"status": "error", "error": error[:2000]})
            last_error = error
        elif interpreted is None:
            row.update(
                {
                    "status": "parse_failure",
                    "tokens_in": payload["tokens_in"],
                    "tokens_out": payload["tokens_out"],
                    "returned_model": payload["returned_model"],
                    "returned_provider": payload["returned_provider"],
                    "finish_reason": payload["finish_reason"],
                    "raw_text": payload["text"][:4000],
                }
            )
            last_error = "parse_failure"
        else:
            row.update(
                {
                    "status": "ok",
                    "tokens_in": payload["tokens_in"],
                    "tokens_out": payload["tokens_out"],
                    "returned_model": payload["returned_model"],
                    "returned_provider": payload["returned_provider"],
                    "response_id": payload["response_id"],
                    "finish_reason": payload["finish_reason"],
                    "raw_text": payload["text"][:4000],
                    "parsed": interpreted,
                }
            )
        append_jsonl(results_path, row)
        time.sleep(post_call_delay_seconds(manifest, spec, error))

        if row["status"] == "ok":
            completed.add(stage["request_sha256"])
            return interpreted
        if row["status"] == "parse_failure":
            # A parse failure is a result about the model, not a transport fault.
            # Retrying would silently select for parseable outputs.
            return None
        if args.fail_fast:
            raise RunnerError(f"Aborting after error on {spec['slot']} case {case['case_id']}: {error}")

    if last_error:
        return None
    return None


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    validate_manifest(manifest)

    probe = manifest.get("probe")
    if not isinstance(probe, dict) or probe.get("type") not in PROBE_TYPES:
        raise RunnerError(f"Manifest needs probe.type in {PROBE_TYPES}.")
    probe_type = probe["type"]

    # Execution locks, identical to the verdict runner.
    if args.execute:
        failures: list[str] = []
        if args.mode == "dry-run":
            failures.append("--execute cannot be combined with --mode dry-run")
        if manifest.get("status") != "frozen":
            failures.append("manifest status is not 'frozen'")
        if manifest.get("paid_execution_authorized") is not True:
            failures.append("manifest paid_execution_authorized is not true")
        if args.mode not in (manifest.get("authorized_modes") or []):
            failures.append(f"mode {args.mode!r} is not in manifest authorized_modes")
        if os.environ.get("CCO_ALLOW_PAID_RUN") != EXECUTION_ACK:
            failures.append(f"CCO_ALLOW_PAID_RUN is not exactly {EXECUTION_ACK!r}")
        if failures:
            raise RunnerError("Paid execution remains locked: " + "; ".join(failures))

    templates: dict[str, str] = {}
    if probe_type == "decompose":
        templates["stage1"] = load_template(manifest["prompt"]["stage1"], "stage1")
        templates["stage2"] = load_template(manifest["prompt"]["stage2"], "stage2")
    else:
        templates["main"] = load_template(manifest["prompt"]["main"], probe_type)
    system = str(manifest["prompt"].get("system_instruction", ""))

    cases, dataset_report = load_cases(manifest, args.source_dir)
    cases, selection_report = load_selected_cases(manifest, cases)
    if args.mode == "smoke":
        smoke_ids = set(int(value) for value in manifest.get("smoke_case_ids", []))
        if not smoke_ids:
            raise RunnerError("Manifest has no smoke_case_ids.")
        cases = [case for case in cases if int(case["case_id"]) in smoke_ids]
        if len(cases) != len(smoke_ids):
            raise RunnerError("Some smoke_case_ids are absent from the selected cases.")

    models = select_models(manifest, args.models)
    plan = build_plan(manifest, cases, models, templates, system)
    rows = costing_rows(plan)
    costing = estimate_cost_upper_bound(models, rows, system, int(manifest.get("max_retries", 1)))

    cap = float(manifest.get("max_estimated_cost_usd", 0))
    over_cap = costing["upper_bound_usd_with_all_retries_billed"] > cap

    output_dir = resolve_workspace_path(manifest["output_root"]) / str(manifest["run_id"])
    results_path = output_dir / (
        "raw_results.jsonl" if args.mode == "smoke" else f"raw_results_{args.mode}.jsonl"
    )
    progress_path = output_dir / (
        "progress.json" if args.mode == "smoke" else f"progress_{args.mode}.json"
    )

    summary = {
        "generated_at": utc_now(),
        "run_id": manifest["run_id"],
        "probe_type": probe_type,
        "mode": args.mode,
        "execute": bool(args.execute),
        "dataset": dataset_report,
        "selection": selection_report,
        "case_count": len(cases),
        "label_counts": dict(Counter(case["gold_normal"] for case in cases)),
        "models": [spec["slot"] for spec in models],
        "planned_units": len(plan),
        "planned_calls": sum(
            len(unit["stages"]) + (1 if unit["deferred_stage"] else 0) for unit in plan
        ),
        "costing_rows_per_model": len(rows),
        "prompt_sha256": {
            key: sha256_text(value) for key, value in templates.items()
        },
        "cost_estimate": costing,
        "max_estimated_cost_usd": cap,
        "over_cost_cap": over_cap,
        "results_path": str(results_path),
    }

    if not args.execute:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if over_cap:
            print(
                f"\nBLOCKED: estimated upper bound "
                f"{costing['upper_bound_usd_with_all_retries_billed']} USD exceeds cap {cap} USD.",
                file=sys.stderr,
            )
            sys.exit(2)
        return

    if over_cap:
        raise RunnerError(
            f"Estimated upper bound {costing['upper_bound_usd_with_all_retries_billed']} USD "
            f"exceeds manifest cap {cap} USD."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "executed_manifest.json", manifest)
    completed = completed_request_keys(results_path)

    executed = 0
    stage2_skipped = 0
    for index, unit in enumerate(plan, start=1):
        first = run_stage(unit, unit["stages"][0], manifest, system, results_path, completed, args)
        executed += 1

        if unit["deferred_stage"] == "stage2_verdict":
            if first is None:
                stage2_skipped += 1
            elif first.get("__resumed__"):
                # Stage 1 already succeeded in a prior run; its parsed payload is not in
                # memory. Recover it from the results file rather than re-calling.
                first = recover_stage1(results_path, unit["stages"][0]["request_sha256"])
                if first is None:
                    stage2_skipped += 1
            if first is not None and not first.get("__resumed__"):
                prompt = render(
                    templates["stage2"],
                    {
                        "CLAIM": unit["case"]["claim"],
                        "EVIDENCE": unit["case"]["evidence"],
                        "DECOMPOSITION": format_decomposition(first),
                    },
                )
                stage2 = {
                    "name": "stage2_verdict",
                    "prompt": prompt,
                    "request_sha256": request_signature(
                        unit["spec"], system, prompt, "stage2_verdict", str(manifest["run_id"])
                    ),
                }
                run_stage(unit, stage2, manifest, system, results_path, completed, args)
                executed += 1

        if index % 25 == 0 or index == len(plan):
            atomic_write_json(
                progress_path,
                {
                    **summary,
                    "updated_at": utc_now(),
                    "units_processed": index,
                    "calls_attempted": executed,
                    "stage2_skipped": stage2_skipped,
                    "accumulated_estimated_cost_usd": round(
                        accumulated_estimated_cost_usd(results_path, models), 6
                    ),
                },
            )

    print(
        json.dumps(
            {
                "run_id": manifest["run_id"],
                "probe_type": probe_type,
                "mode": args.mode,
                "units": len(plan),
                "calls_attempted": executed,
                "stage2_skipped": stage2_skipped,
                "accumulated_estimated_cost_usd": round(
                    accumulated_estimated_cost_usd(results_path, models), 6
                ),
                "results_path": str(results_path),
            },
            indent=2,
        )
    )


def recover_stage1(results_path: Path, request_sha256: str) -> dict[str, Any] | None:
    if not results_path.exists():
        return None
    with results_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("request_sha256") == request_sha256 and row.get("status") == "ok":
                return row.get("parsed")
    return None


if __name__ == "__main__":
    try:
        main()
    except RunnerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
