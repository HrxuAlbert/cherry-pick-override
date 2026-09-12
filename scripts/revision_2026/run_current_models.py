"""Manifest-driven AVeriTeC runner for the current-model replication.

The default behavior is a local dry run. Network calls are possible only when
all three execution locks are open: ``--execute``, a frozen/authorized manifest,
and ``CCO_ALLOW_PAID_RUN=I_UNDERSTAND``. Results are append-only and resume by
the hash of the exact model request rather than by case ID alone.
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
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE = _cpo_workspace()
EXECUTION_ACK = "I_UNDERSTAND"
GOLD_LABELS = {
    "Supported": "support",
    "Refuted": "refute",
    "Not Enough Evidence": "insufficient",
    "Conflicting Evidence/Cherrypicking": "conflicting",
}
VERDICT_LABELS = {
    "SUPPORTS": "support",
    "SUPPORT": "support",
    "SUPPORTED": "support",
    "REFUTES": "refute",
    "REFUTE": "refute",
    "REFUTED": "refute",
    "NOT_ENOUGH_INFO": "insufficient",
    "NOTENOUGHINFO": "insufficient",
    "INSUFFICIENT": "insufficient",
    "INSUFFICIENT_EVIDENCE": "insufficient",
    "NEI": "insufficient",
    "CONFLICTING": "conflicting",
    "CONFLICT": "conflicting",
    "MIXED": "conflicting",
    "CHERRYPICKING": "conflicting",
    "CHERRY_PICKING": "conflicting",
}


class RunnerError(RuntimeError):
    """Raised when a reproducibility or execution guard fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("dry-run", "smoke", "full"),
        default="dry-run",
        help="dry-run validates everything; smoke plans/runs five cases; full uses all cases.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call model endpoints. Requires all execution locks.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Override the manifest's local directory containing dev.json/train.json.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of enabled model slot names.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else WORKSPACE / path


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RunnerError(f"Required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Invalid JSON in {path}: {exc}") from exc


def require_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RunnerError(
            f"{label} hash mismatch for {path}: expected {expected}, got {actual}"
        )


def format_qa(qa: dict[str, Any]) -> str:
    question = str(qa.get("question", "")).strip()
    answers = qa.get("answers", []) or []
    if not answers:
        return f"Q: {question}\nA: (no answer)"
    parts: list[str] = []
    for answer in answers:
        answer_text = str(answer.get("answer", ""))
        if answer.get("answer_type") == "Boolean" and answer.get("boolean_explanation"):
            answer_text = f"{answer_text} ({answer['boolean_explanation']})"
        parts.append(f"Q: {question}\nA: {answer_text}")
    return "\n".join(parts)


def build_evidence(example: dict[str, Any], evidence_order: str = "original") -> str:
    parts = [format_qa(qa) for qa in (example.get("questions") or []) if qa]
    if evidence_order == "reversed":
        parts.reverse()
    elif evidence_order != "original":
        raise RunnerError(f"Unsupported evidence_order: {evidence_order}")
    return "\n\n".join(parts) or "(no evidence provided)"


def render_prompt(template: str, claim: str, evidence: str) -> str:
    if "<<CLAIM>>" not in template or "<<EVIDENCE>>" not in template:
        raise RunnerError("Prompt template must contain <<CLAIM>> and <<EVIDENCE>>.")
    return template.replace("<<CLAIM>>", claim).replace("<<EVIDENCE>>", evidence)


def normalize_verdict(value: Any) -> str | None:
    raw = str(value or "").upper().strip().replace(" ", "_").replace("-", "_")
    return VERDICT_LABELS.get(raw) or VERDICT_LABELS.get(raw.replace("_", ""))


def parse_verdict(text: str) -> dict[str, Any] | None:
    parsed: Any
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    verdict_value = parsed.get("verdict", parsed.get("final_verdict", parsed.get("label")))
    normal = normalize_verdict(verdict_value)
    if normal is None:
        return None
    try:
        confidence = float(parsed.get("confidence", parsed.get("final_confidence", 0.5)))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "verdict_raw": str(verdict_value).upper().strip(),
        "verdict_normal": normal,
        "confidence": max(0.0, min(1.0, confidence)),
        "reasoning": str(parsed.get("reasoning", parsed.get("rationale", ""))),
        "response_object": parsed,
    }


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != "1.0":
        raise RunnerError("Manifest schema_version must be '1.0'.")
    for key in ("run_id", "status", "dataset", "prompt", "models", "output_root"):
        if key not in manifest:
            raise RunnerError(f"Manifest is missing required field: {key}")
    if not isinstance(manifest["models"], list) or not manifest["models"]:
        raise RunnerError("Manifest must contain at least one model specification.")
    slots = [str(spec.get("slot", "")) for spec in manifest["models"]]
    if any(not slot for slot in slots) or len(slots) != len(set(slots)):
        raise RunnerError("Every model needs a unique, non-empty slot.")


def select_models(manifest: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    enabled = [spec for spec in manifest["models"] if spec.get("enabled", False)]
    if requested is None:
        selected = enabled
    else:
        unknown = sorted(set(requested) - {str(spec["slot"]) for spec in enabled})
        if unknown:
            raise RunnerError(f"Unknown or disabled model slots: {', '.join(unknown)}")
        requested_set = set(requested)
        selected = [spec for spec in enabled if spec["slot"] in requested_set]
    if not selected:
        raise RunnerError("No enabled model slots were selected.")
    for spec in selected:
        for field in ("slot", "provider_type", "model_id", "api_key_env"):
            if not spec.get(field):
                raise RunnerError(f"Model slot {spec.get('slot', '<unknown>')} lacks {field}.")
        if spec["provider_type"] not in {"openai_compatible", "anthropic_messages"}:
            raise RunnerError(
                f"Unsupported provider_type for {spec['slot']}: {spec['provider_type']}"
            )
    return selected


def load_cases(
    manifest: dict[str, Any], source_override: Path | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = manifest["dataset"]
    dataset_format = str(dataset.get("format", "averitec_source_map"))
    if dataset_format == "materialized_json":
        if source_override is not None:
            raise RunnerError("--source-dir is not valid for a materialized_json dataset.")
        materialized_path = resolve_workspace_path(dataset["materialized_path"])
        require_sha(
            materialized_path,
            dataset["materialized_sha256"],
            f"{dataset.get('name', 'materialized')} fixed cases",
        )
        materialized = load_json(materialized_path)
        expected_total = int(dataset["expected_total"])
        if not isinstance(materialized, list) or len(materialized) != expected_total:
            raise RunnerError(
                f"Expected {expected_total} materialized rows, found {len(materialized)}."
            )
        evidence_order = str(dataset.get("evidence_order", "original"))
        if evidence_order not in {"original", "reversed"}:
            raise RunnerError("dataset.evidence_order must be 'original' or 'reversed'.")
        cases: list[dict[str, Any]] = []
        for expected_case_id, item in enumerate(materialized):
            if int(item.get("case_id", -1)) != expected_case_id:
                raise RunnerError(
                    f"Materialized cases are not contiguous at row {expected_case_id}."
                )
            gold = str(item.get("gold_normal", ""))
            if gold not in set(GOLD_LABELS.values()):
                raise RunnerError(f"Invalid materialized gold label at case {expected_case_id}.")
            evidence_blocks = item.get("evidence_blocks")
            if not isinstance(evidence_blocks, list) or not evidence_blocks:
                evidence_blocks = [str(item.get("evidence", ""))]
            blocks = [str(block) for block in evidence_blocks]
            if evidence_order == "reversed":
                blocks.reverse()
            evidence = "\n\n".join(blocks) or "(no evidence provided)"
            cases.append(
                {
                    **item,
                    "case_id": expected_case_id,
                    "split": str(item.get("split", dataset.get("split", "unknown"))),
                    "source_row_idx": int(item.get("source_row_idx", expected_case_id)),
                    "sample_group": str(item.get("sample_group", gold)),
                    "gold_normal": gold,
                    "evidence": evidence,
                    "evidence_sha256": sha256_text(evidence),
                    "evidence_unit_count": len(blocks),
                }
            )
        counts = Counter(case["gold_normal"] for case in cases)
        expected_counts = {k: int(v) for k, v in dataset["expected_counts"].items()}
        if dict(counts) != expected_counts:
            raise RunnerError(
                f"Sample composition mismatch: expected {expected_counts}, got {dict(counts)}"
            )
        return cases, {
            "dataset_format": dataset_format,
            "materialized_path": str(materialized_path),
            "case_count": len(cases),
            "label_counts": dict(counts),
            "max_evidence_characters": max(len(case["evidence"]) for case in cases),
            "evidence_order": evidence_order,
        }
    if dataset_format != "averitec_source_map":
        raise RunnerError(f"Unsupported dataset format: {dataset_format}")
    source_dir = source_override or resolve_workspace_path(dataset["local_source_dir"])
    dev_path = source_dir / "dev.json"
    train_path = source_dir / "train.json"
    require_sha(dev_path, dataset["sources"]["dev"]["sha256"], "AVeriTeC dev")
    require_sha(train_path, dataset["sources"]["train"]["sha256"], "AVeriTeC train")

    sample_map_path = resolve_workspace_path(dataset["sample_map_path"])
    require_sha(sample_map_path, dataset["sample_map_sha256"], "fixed E1 sample map")
    sample_map = load_json(sample_map_path)
    dev = load_json(dev_path)
    train = load_json(train_path)
    splits = {"dev": dev, "train": train}

    expected_total = int(dataset["expected_total"])
    if not isinstance(sample_map, list) or len(sample_map) != expected_total:
        raise RunnerError(
            f"Expected {expected_total} source-map rows, found {len(sample_map)}."
        )
    evidence_order = str(dataset.get("evidence_order", "original"))
    if evidence_order not in {"original", "reversed"}:
        raise RunnerError("dataset.evidence_order must be 'original' or 'reversed'.")
    cases: list[dict[str, Any]] = []
    for expected_case_id, item in enumerate(sample_map):
        if item.get("case_id") != expected_case_id:
            raise RunnerError(
                f"Source map is not contiguous at row {expected_case_id}: {item.get('case_id')}"
            )
        split = item["split"]
        source_idx = int(item["source_row_idx"])
        try:
            example = splits[split][source_idx]
        except (KeyError, IndexError) as exc:
            raise RunnerError(
                f"Invalid source reference for case {expected_case_id}: {split}[{source_idx}]"
            ) from exc
        if example.get("claim") != item.get("claim") or example.get("label") != item.get("label"):
            raise RunnerError(f"Source-map anchor mismatch for case {expected_case_id}.")
        evidence = build_evidence(example, evidence_order=evidence_order)
        cases.append(
            {
                **item,
                "gold_normal": GOLD_LABELS[item["label"]],
                "evidence": evidence,
                "evidence_sha256": sha256_text(evidence),
                "evidence_unit_count": sum(1 for qa in (example.get("questions") or []) if qa),
            }
        )

    counts = Counter(case["gold_normal"] for case in cases)
    expected_counts = {k: int(v) for k, v in dataset["expected_counts"].items()}
    if dict(counts) != expected_counts:
        raise RunnerError(f"Sample composition mismatch: expected {expected_counts}, got {dict(counts)}")
    metadata = {
        "source_dir": str(source_dir),
        "dataset_format": dataset_format,
        "sample_map_path": str(sample_map_path),
        "case_count": len(cases),
        "label_counts": dict(counts),
        "max_evidence_characters": max(len(case["evidence"]) for case in cases),
        "evidence_order": evidence_order,
    }
    return cases, metadata


def load_selected_cases(
    manifest: dict[str, Any], cases: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Apply an optional hash-pinned case-ID selection from the manifest."""
    selection = manifest.get("selection")
    if selection is None:
        return cases, None
    for key in ("case_ids_path", "case_ids_sha256", "expected_count", "source"):
        if key not in selection:
            raise RunnerError(f"Manifest selection is missing required field: {key}")
    path = resolve_workspace_path(selection["case_ids_path"])
    require_sha(path, str(selection["case_ids_sha256"]), "case selection")
    raw = path.read_text(encoding="utf-8").strip()
    try:
        if raw.startswith("["):
            values = json.loads(raw)
        else:
            values = [line.strip() for line in raw.splitlines() if line.strip()]
        case_ids = [int(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Invalid case-ID selection file: {path}") from exc
    expected_count = int(selection["expected_count"])
    if len(case_ids) != expected_count or len(set(case_ids)) != expected_count:
        raise RunnerError(
            f"Case selection needs {expected_count} unique IDs; found {len(case_ids)} rows "
            f"and {len(set(case_ids))} unique IDs."
        )
    by_id = {int(case["case_id"]): case for case in cases}
    unknown = sorted(set(case_ids) - set(by_id))
    if unknown:
        raise RunnerError(f"Case selection contains unknown IDs: {unknown[:10]}")
    selected = [by_id[case_id] for case_id in case_ids]
    return selected, {
        "path": str(path),
        "sha256": selection["case_ids_sha256"],
        "source": selection["source"],
        "expected_count": expected_count,
        "selected_label_counts": dict(Counter(case["gold_normal"] for case in selected)),
    }


def build_request(
    spec: dict[str, Any], system: str, prompt: str
) -> tuple[str, dict[str, str], dict[str, Any]]:
    api_key = os.environ.get(spec["api_key_env"], "")
    if not api_key:
        raise RunnerError(
            f"Environment variable {spec['api_key_env']} is not set for {spec['slot']}."
        )
    request_overrides = json.loads(json.dumps(spec.get("request", {})))
    if spec["provider_type"] == "openai_compatible":
        base_url = str(spec.get("base_url", "")).rstrip("/")
        if not base_url:
            raise RunnerError(f"Model {spec['slot']} needs base_url.")
        url = f"{base_url}/chat/completions"
        payload = {
            "model": spec["model_id"],
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            **request_overrides,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "cco-current-model-replication/1.0",
        }
        headers.update({str(k): str(v) for k, v in spec.get("headers", {}).items()})
        return url, headers, payload

    base_url = str(spec.get("base_url", "https://api.anthropic.com/v1")).rstrip("/")
    url = f"{base_url}/messages"
    payload = {
        "model": spec["model_id"],
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        **request_overrides,
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": str(spec.get("anthropic_version", "2023-06-01")),
        "Content-Type": "application/json",
        "User-Agent": "cco-current-model-replication/1.0",
    }
    headers.update({str(k): str(v) for k, v in spec.get("headers", {}).items()})
    return url, headers, payload


def http_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout_seconds: int
) -> tuple[dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        url,
        data=canonical_json(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            return json.loads(body), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RunnerError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RunnerError(f"Network error calling {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Endpoint returned invalid JSON from {url}: {exc}") from exc


def parse_provider_response(
    provider_type: str,
    response: dict[str, Any],
    provider_identity: str | None = None,
) -> dict[str, Any]:
    if provider_type == "openai_compatible":
        try:
            content = response["choices"][0]["message"].get("content", "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RunnerError(f"Malformed OpenAI-compatible response: {str(response)[:1000]}") from exc
        if isinstance(content, list):
            text = "".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content or "")
        usage = response.get("usage") or {}
        return {
            "text": text,
            "tokens_in": int(usage.get("prompt_tokens", 0) or 0),
            "tokens_out": int(usage.get("completion_tokens", 0) or 0),
            "returned_model": response.get("model"),
            "returned_provider": response.get("provider") or provider_identity,
            "response_id": response.get("id"),
            "finish_reason": response["choices"][0].get("finish_reason"),
            "stop_details": None,
            "usage": usage,
        }

    blocks = response.get("content") or []
    text = "".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    is_refusal = response.get("stop_reason") == "refusal"
    if (not blocks or not text) and not is_refusal:
        raise RunnerError(f"Malformed Anthropic response: {str(response)[:1000]}")
    usage = response.get("usage") or {}
    return {
        "text": text,
        "tokens_in": int(usage.get("input_tokens", 0) or 0)
        + int(usage.get("cache_creation_input_tokens", 0) or 0)
        + int(usage.get("cache_read_input_tokens", 0) or 0),
        "tokens_out": int(usage.get("output_tokens", 0) or 0),
        "returned_model": response.get("model"),
        "returned_provider": "anthropic-direct",
        "response_id": response.get("id"),
        "finish_reason": response.get("stop_reason"),
        "stop_details": response.get("stop_details"),
        "usage": usage,
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def execution_artifact_paths(output_dir: Path, mode: str) -> tuple[Path, Path]:
    """Keep accepted smoke evidence separate from the complete full matrix."""
    if mode == "smoke":
        return output_dir / "raw_results.jsonl", output_dir / "progress.json"
    return output_dir / f"raw_results_{mode}.jsonl", output_dir / f"progress_{mode}.json"


def completed_request_keys(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"Corrupt JSONL at {path}:{line_number}: {exc}") from exc
            terminal_legacy_refusal = (
                row.get("status") == "error"
                and "stop_reason': 'refusal'" in str(row.get("error", ""))
            )
            if (row.get("status") == "ok" or terminal_legacy_refusal) and row.get(
                "request_sha256"
            ):
                completed.add(str(row["request_sha256"]))
    return completed


def accumulated_estimated_cost_usd(
    path: Path, models: list[dict[str, Any]]
) -> float:
    if not path.exists():
        return 0.0
    pricing = {
        str(spec["slot"]): spec["pricing_usd_per_million"] for spec in models
    }
    total = 0.0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RunnerError(f"Corrupt JSONL at {path}:{line_number}: {exc}") from exc
            if row.get("status") != "ok":
                continue
            slot = str(row.get("model_slot", ""))
            if slot not in pricing:
                continue
            rates = pricing[slot]
            total += (
                int(row.get("tokens_in", 0)) * float(rates["input"])
                + int(row.get("tokens_out", 0)) * float(rates["output"])
            ) / 1_000_000
    return total


def post_call_delay_seconds(
    manifest: dict[str, Any], spec: dict[str, Any], error: str | None
) -> float:
    delay = float(manifest.get("inter_call_delay_seconds", 0))
    if "openrouter.ai" in str(spec.get("base_url", "")):
        delay = max(delay, 1.0)
    provider_identity = str(spec.get("provider_identity", "")).lower()
    if provider_identity == "groq":
        delay = max(delay, 10.0)
    elif provider_identity == "mistral":
        delay = max(delay, 3.0)
    if error and ("HTTP 429" in error or "HTTP 503" in error):
        delay = max(delay, 60.0 if "HTTP 429" in error else 15.0)
    return delay


def estimate_cost_upper_bound(
    models: list[dict[str, Any]], cases: list[dict[str, Any]], system: str, max_retries: int
) -> dict[str, Any]:
    """Conservative character-based estimate; output tokens use the hard request caps."""
    by_model: list[dict[str, Any]] = []
    total_no_retries = 0.0
    for spec in models:
        pricing = spec.get("pricing_usd_per_million") or {}
        if "input" not in pricing or "output" not in pricing:
            raise RunnerError(f"Model {spec['slot']} lacks pricing_usd_per_million.")
        max_output = int(spec.get("accounting_max_output_tokens", 0))
        if max_output <= 0:
            raise RunnerError(f"Model {spec['slot']} lacks accounting_max_output_tokens.")
        # One token per two characters is deliberately more conservative than
        # the usual English-text heuristic, and includes the system instruction.
        estimated_input_tokens = sum(
            math.ceil((len(system) + len(case["user_prompt"])) / 2) for case in cases
        )
        max_output_tokens = max_output * len(cases)
        model_cost = (
            estimated_input_tokens * float(pricing["input"])
            + max_output_tokens * float(pricing["output"])
        ) / 1_000_000
        total_no_retries += model_cost
        by_model.append(
            {
                "slot": spec["slot"],
                "estimated_input_tokens": estimated_input_tokens,
                "maximum_output_tokens": max_output_tokens,
                "upper_bound_usd_without_retries": round(model_cost, 6),
            }
        )
    return {
        "method": "input characters / 2 plus hard output-token caps",
        "by_model": by_model,
        "upper_bound_usd_without_retries": round(total_no_retries, 6),
        "retry_ceiling_multiplier": max_retries,
        "upper_bound_usd_with_all_retries_billed": round(total_no_retries * max_retries, 6),
    }


def execution_guard(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    if not args.execute:
        return
    failures: list[str] = []
    if args.mode == "dry-run":
        failures.append("--execute cannot be combined with --mode dry-run")
    if manifest.get("status") != "frozen":
        failures.append("manifest status is not 'frozen'")
    if manifest.get("paid_execution_authorized") is not True:
        failures.append("manifest paid_execution_authorized is not true")
    authorized_modes = manifest.get("authorized_modes") or []
    if args.mode not in authorized_modes:
        failures.append(
            f"mode {args.mode!r} is not in manifest authorized_modes={authorized_modes!r}"
        )
    if os.environ.get("CCO_ALLOW_PAID_RUN") != EXECUTION_ACK:
        failures.append(f"CCO_ALLOW_PAID_RUN is not exactly {EXECUTION_ACK!r}")
    if failures:
        raise RunnerError("Paid execution remains locked: " + "; ".join(failures))


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    validate_manifest(manifest)
    execution_guard(args, manifest)
    selected_models = select_models(manifest, args.models)
    cases, dataset_metadata = load_cases(manifest, args.source_dir)
    selected_cases, selection_metadata = load_selected_cases(manifest, cases)

    prompt_spec = manifest["prompt"]
    prompt_path = resolve_workspace_path(prompt_spec["template_path"])
    require_sha(prompt_path, prompt_spec["template_sha256"], "primary prompt template")
    prompt_template = prompt_path.read_text()
    system = str(prompt_spec["system_instruction"])
    system_sha256 = sha256_text(system)

    smoke_case_ids = [int(case_id) for case_id in manifest["smoke_case_ids"]]
    if len(smoke_case_ids) != 5 or len(set(smoke_case_ids)) != 5:
        raise RunnerError("smoke_case_ids must contain five unique case IDs.")
    if any(case_id < 0 or case_id >= len(cases) for case_id in smoke_case_ids):
        raise RunnerError("smoke_case_ids contains an out-of-range ID.")
    cases_by_id = {int(case["case_id"]): case for case in cases}
    smoke_labels = Counter(cases_by_id[case_id]["gold_normal"] for case_id in smoke_case_ids)
    required_smoke = Counter(
        {
            str(label): int(count)
            for label, count in manifest.get(
                "smoke_expected_counts",
                {"conflicting": 2, "support": 1, "refute": 1, "insufficient": 1},
            ).items()
        }
    )
    if sum(required_smoke.values()) != 5:
        raise RunnerError("smoke_expected_counts must sum to five cases.")
    if smoke_labels != required_smoke:
        raise RunnerError(
            f"Smoke composition must be {dict(required_smoke)}, got {dict(smoke_labels)}"
        )

    if args.mode == "smoke":
        if selection_metadata is not None:
            selected_ids = {int(case["case_id"]) for case in selected_cases}
            missing_from_selection = sorted(set(smoke_case_ids) - selected_ids)
            if missing_from_selection:
                raise RunnerError(
                    f"Smoke IDs are outside the selected mechanism subset: {missing_from_selection}"
                )
        run_cases = [cases_by_id[case_id] for case_id in smoke_case_ids]
    else:
        run_cases = selected_cases

    manifest_sha256 = sha256_bytes(manifest_raw)
    output_dir = resolve_workspace_path(manifest["output_root"]) / manifest["run_id"]
    plan_rows: list[dict[str, Any]] = []
    materialized: list[dict[str, Any]] = []
    for case in run_cases:
        user_prompt = render_prompt(prompt_template, case["claim"], case["evidence"])
        materialized.append({**case, "user_prompt": user_prompt})
        plan_rows.append(
            {
                "case_id": case["case_id"],
                "gold_normal": case["gold_normal"],
                "prompt_sha256": sha256_text(user_prompt),
                "prompt_characters": len(user_prompt),
                "evidence_sha256": case["evidence_sha256"],
            }
        )

    cost_estimate = estimate_cost_upper_bound(
        selected_models,
        materialized,
        system,
        int(manifest.get("max_retries", 3)),
    )
    cost_cap = float(manifest["max_estimated_cost_usd"])
    if args.execute and cost_estimate["upper_bound_usd_with_all_retries_billed"] > cost_cap:
        raise RunnerError(
            "Conservative cost ceiling exceeds manifest cap: "
            f"${cost_estimate['upper_bound_usd_with_all_retries_billed']:.4f} > ${cost_cap:.4f}"
        )

    plan = {
        "generated_at": utc_now(),
        "mode": args.mode,
        "execute_requested": args.execute,
        "api_calls_made": 0,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_status": manifest["status"],
        "paid_execution_authorized": manifest.get("paid_execution_authorized", False),
        "dataset": dataset_metadata,
        "selection": selection_metadata,
        "prompt": {
            "path": str(prompt_path),
            "template_sha256": prompt_spec["template_sha256"],
            "system_sha256": system_sha256,
            "max_prompt_characters": max(row["prompt_characters"] for row in plan_rows),
        },
        "models": [
            {
                "slot": spec["slot"],
                "cohort": spec.get("cohort"),
                "provider_type": spec["provider_type"],
                "model_id": spec["model_id"],
                "checkpoint_id": spec.get("checkpoint_id"),
                "api_key_env": spec["api_key_env"],
                "credential_present": bool(os.environ.get(spec["api_key_env"])),
                "request": spec.get("request", {}),
            }
            for spec in selected_models
        ],
        "case_count": len(materialized),
        "planned_call_count": len(materialized) * len(selected_models),
        "cost_guard": {
            "manifest_cap_usd": cost_cap,
            **cost_estimate,
        },
        "smoke_case_ids": smoke_case_ids,
        "case_preview": plan_rows[:5],
    }
    atomic_write_json(output_dir / f"plan_{args.mode}.json", plan)

    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "validated_no_api_calls",
                    "mode": args.mode,
                    "cases": len(materialized),
                    "models": len(selected_models),
                    "planned_calls": plan["planned_call_count"],
                    "plan": str(output_dir / f"plan_{args.mode}.json"),
                },
                ensure_ascii=False,
            )
        )
        return

    raw_path, progress_path = execution_artifact_paths(output_dir, args.mode)
    completed_keys = completed_request_keys(raw_path)
    prior_estimated_cost_usd = accumulated_estimated_cost_usd(raw_path, selected_models)
    total = len(materialized) * len(selected_models)
    skipped = 0
    attempted = 0
    succeeded = 0
    failed = 0
    total_tokens_in = 0
    total_tokens_out = 0
    estimated_cost_usd = 0.0

    for spec in selected_models:
        for case in materialized:
            url, headers, request_payload = build_request(
                spec, system, case["user_prompt"]
            )
            request_record = {
                "manifest_sha256": manifest_sha256,
                "model_slot": spec["slot"],
                "provider_type": spec["provider_type"],
                "model_id": spec["model_id"],
                "case_id": case["case_id"],
                "system_sha256": system_sha256,
                "request_payload": request_payload,
            }
            request_sha256 = sha256_text(canonical_json(request_record))
            if request_sha256 in completed_keys:
                skipped += 1
                continue

            attempted += 1
            started_at = utc_now()
            response_parsed: dict[str, Any] | None = None
            error: str | None = None
            response_headers: dict[str, str] = {}
            for attempt in range(1, int(manifest.get("max_retries", 3)) + 1):
                try:
                    response_body, response_headers = http_json(
                        url,
                        headers,
                        request_payload,
                        int(manifest.get("timeout_seconds", 180)),
                    )
                    response_parsed = parse_provider_response(
                        spec["provider_type"],
                        response_body,
                        spec.get("provider_identity"),
                    )
                    error = None
                    break
                except Exception as exc:  # endpoint SDKs expose heterogeneous failures
                    error = str(exc)[:4000]
                    if attempt < int(manifest.get("max_retries", 3)):
                        time.sleep(min(2 ** (attempt - 1), 8))

            base_row = {
                "run_id": manifest["run_id"],
                "manifest_sha256": manifest_sha256,
                "request_sha256": request_sha256,
                "started_at": started_at,
                "finished_at": utc_now(),
                "mode": args.mode,
                "model_slot": spec["slot"],
                "cohort": spec.get("cohort"),
                "provider_type": spec["provider_type"],
                "requested_model": spec["model_id"],
                "checkpoint_id": spec.get("checkpoint_id"),
                "case_id": case["case_id"],
                "split": case["split"],
                "source_row_idx": case["source_row_idx"],
                "sample_group": case["sample_group"],
                "gold_4way": case["label"],
                "gold_normal": case["gold_normal"],
                "claim": case["claim"],
                "evidence": case["evidence"],
                "evidence_sha256": case["evidence_sha256"],
                "system_instruction": system,
                "system_sha256": system_sha256,
                "user_prompt": case["user_prompt"],
                "user_prompt_sha256": sha256_text(case["user_prompt"]),
                "request_parameters": spec.get("request", {}),
            }
            if response_parsed is None:
                failed += 1
                row = {**base_row, "status": "error", "error": error}
                append_jsonl(raw_path, row)
                if args.fail_fast:
                    raise RunnerError(error or "Unknown endpoint error")
            else:
                verdict = parse_verdict(response_parsed["text"])
                succeeded += 1
                total_tokens_in += response_parsed["tokens_in"]
                total_tokens_out += response_parsed["tokens_out"]
                pricing = spec["pricing_usd_per_million"]
                estimated_cost_usd += (
                    response_parsed["tokens_in"] * float(pricing["input"])
                    + response_parsed["tokens_out"] * float(pricing["output"])
                ) / 1_000_000
                row = {
                    **base_row,
                    "status": "ok",
                    "error": None,
                    "returned_model": response_parsed["returned_model"],
                    "returned_provider": response_parsed["returned_provider"],
                    "identity_checks": {
                        "model": response_parsed["returned_model"]
                        == spec.get("expected_returned_model", spec["model_id"]),
                        "provider": str(response_parsed["returned_provider"] or "").lower()
                        == str(spec.get("provider_identity", "")).lower(),
                    },
                    "response_id": response_parsed["response_id"],
                    "finish_reason": response_parsed["finish_reason"],
                    "stop_details": response_parsed["stop_details"],
                    "response_headers": {
                        key: value
                        for key, value in response_headers.items()
                        if key in {"x-request-id", "openai-processing-ms", "x-ratelimit-remaining-requests"}
                    },
                    "raw_response": response_parsed["text"],
                    "parsed": verdict,
                    "parse_ok": verdict is not None,
                    "tokens_in": response_parsed["tokens_in"],
                    "tokens_out": response_parsed["tokens_out"],
                    "usage": response_parsed["usage"],
                }
                append_jsonl(raw_path, row)
                completed_keys.add(request_sha256)

            atomic_write_json(
                progress_path,
                {
                    "updated_at": utc_now(),
                    "run_id": manifest["run_id"],
                    "mode": args.mode,
                    "total_planned": total,
                    "attempted_this_process": attempted,
                    "skipped_resumed": skipped,
                    "succeeded_this_process": succeeded,
                    "failed_this_process": failed,
                    "tokens_in_this_process": total_tokens_in,
                    "tokens_out_this_process": total_tokens_out,
                    "estimated_cost_usd_this_process": round(estimated_cost_usd, 6),
                    "estimated_cost_usd_before_process": round(prior_estimated_cost_usd, 6),
                    "estimated_cost_usd_cumulative": round(
                        prior_estimated_cost_usd + estimated_cost_usd, 6
                    ),
                    "last_model_slot": spec["slot"],
                    "last_case_id": case["case_id"],
                },
            )
            delay = post_call_delay_seconds(manifest, spec, error)
            if delay > 0:
                time.sleep(delay)
            if prior_estimated_cost_usd + estimated_cost_usd > cost_cap:
                raise RunnerError(
                    f"Runtime token-cost estimate exceeded manifest cap ${cost_cap:.4f}."
                )

    final_status = "completed" if failed == 0 else "completed_with_errors"
    atomic_write_json(
        progress_path,
        {
            "updated_at": utc_now(),
            "status": final_status,
            "run_id": manifest["run_id"],
            "mode": args.mode,
            "total_planned": total,
            "attempted_this_process": attempted,
            "skipped_resumed": skipped,
            "succeeded_this_process": succeeded,
            "failed_this_process": failed,
            "tokens_in_this_process": total_tokens_in,
            "tokens_out_this_process": total_tokens_out,
            "estimated_cost_usd_this_process": round(estimated_cost_usd, 6),
            "estimated_cost_usd_before_process": round(prior_estimated_cost_usd, 6),
            "estimated_cost_usd_cumulative": round(
                prior_estimated_cost_usd + estimated_cost_usd, 6
            ),
        },
    )
    print(json.dumps(load_json(progress_path), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (RunnerError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
