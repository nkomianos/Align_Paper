"""Offline, checksum-bound reconstruction and decision for Phantom Rollback G0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import yaml

from .analysis import GateThresholds, evaluate_gate, score_family
from .corpus import ARM_IDS, SEEDS, TaskCase, validate_corpus
from .environment import oracle_preflight, replay_trajectory, rollback_semantics_sha256
from .io import (
    atomic_json,
    code_inventory,
    json_from_bytes,
    jsonl_from_bytes,
    sha256_bytes,
    sha256_file,
)
from .protocol import (
    parse_comprehension,
    render_comprehension_prompt,
    render_followup_prompt,
    render_initial_prompt,
    schema_for,
    strip_padding,
)
from .runner import FAMILIES, FROZEN_STATUS, _git_commit, _load_config


ROOT_FILES = {
    "COMPLETE",
    "MANIFEST.json",
    "PROVENANCE.json",
    "RUN_BINDING.json",
    "comprehension.jsonl",
    "config.yaml",
    "raw_trajectories.jsonl",
    "sealed_corpus.json",
}


def _snapshot_root(root: Path) -> dict[str, bytes]:
    if not root.is_dir():
        raise ValueError(f"missing evidence root: {root}")
    actual = {path.name for path in root.iterdir() if path.is_file()}
    directories = [path.name for path in root.iterdir() if path.is_dir()]
    if actual != ROOT_FILES or directories:
        raise ValueError(f"evidence inventory mismatch at {root}: files={sorted(actual)}, dirs={directories}")
    return {name: (root / name).read_bytes() for name in sorted(ROOT_FILES)}


def _validate_manifest(snapshot: Mapping[str, bytes], *, root: Path) -> dict[str, Any]:
    manifest = json_from_bytes(snapshot["MANIFEST.json"], label="family manifest")
    if set(manifest) != {
        "kind", "state", "family", "model_id", "revision", "trajectory_count",
        "comprehension_count", "run_binding_sha256", "artifacts",
    }:
        raise ValueError("manifest fields differ from frozen schema")
    if manifest["kind"] != "phantom_rollback_g0_family" or manifest["state"] != "COMPLETE":
        raise ValueError("family root is not complete")
    if snapshot["COMPLETE"] != b"COMPLETE\n":
        raise ValueError("invalid COMPLETE marker")
    expected_artifacts = {
        "config.yaml", "sealed_corpus.json", "RUN_BINDING.json", "PROVENANCE.json",
        "raw_trajectories.jsonl", "comprehension.jsonl",
    }
    if set(manifest["artifacts"]) != expected_artifacts:
        raise ValueError("manifest artifact inventory changed")
    for name in expected_artifacts:
        if manifest["artifacts"][name] != sha256_bytes(snapshot[name]):
            raise ValueError(f"artifact checksum mismatch at {root}: {name}")
    if manifest["run_binding_sha256"] != sha256_bytes(snapshot["RUN_BINDING.json"]):
        raise ValueError("run-binding checksum mismatch")
    return manifest


def _validate_provenance(
    provenance: Mapping[str, Any],
    binding: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    family: str,
    expected_git_commit: str,
) -> None:
    required = {
        "kind", "family", "model_id", "revision", "python_version", "platform",
        "git_commit", "git_dirty_tracked", "code_tree_sha256", "code_inventory",
        "bindings_frozen", "test_hooks_active", "torch_version", "transformers_version",
        "cuda_runtime", "gpu_name", "gpu_total_memory_bytes", "gpu_compute_capability",
        "resolved_model_commit", "numpy_version", "pyyaml_version",
    }
    if set(provenance) != required:
        raise ValueError("runtime provenance fields differ from frozen schema")
    model = cfg["models"][family]
    if provenance["kind"] != "phantom_rollback_g0_runtime_provenance":
        raise ValueError("wrong provenance kind")
    if (provenance["family"], provenance["model_id"], provenance["revision"]) != (
        family, model["id"], model["revision"]
    ):
        raise ValueError("model provenance mismatch")
    if provenance["test_hooks_active"] or not provenance["bindings_frozen"] or provenance["git_dirty_tracked"]:
        raise ValueError("formal evidence used test hooks, unfrozen bindings, or a dirty tracked tree")
    integrity = cfg["integrity"]
    if provenance["git_commit"] != expected_git_commit or provenance["code_tree_sha256"] != integrity["code_tree_sha256"]:
        raise ValueError("provenance is not bound to frozen code")
    if binding["git_commit"] != expected_git_commit or binding["code_tree_sha256"] != integrity["code_tree_sha256"]:
        raise ValueError("run binding is not bound to frozen code")
    expected = cfg["environment"]
    if not str(provenance["python_version"]).startswith(str(expected["python_prefix"])):
        raise ValueError("Python version differs from frozen environment")
    if not str(provenance["torch_version"]).startswith(str(expected["torch_prefix"])):
        raise ValueError("torch version differs from frozen environment")
    if str(provenance["transformers_version"]) != str(expected["transformers_version"]):
        raise ValueError("transformers version differs from frozen environment")
    if str(provenance["numpy_version"]) != str(expected["numpy_version"]):
        raise ValueError("numpy version differs from frozen environment")
    if str(provenance["pyyaml_version"]) != str(expected["pyyaml_version"]):
        raise ValueError("PyYAML version differs from frozen environment")
    if not str(provenance["cuda_runtime"]).startswith(str(expected["cuda_prefix"])):
        raise ValueError("CUDA runtime differs from frozen environment")
    if int(provenance["gpu_total_memory_bytes"]) < int(expected["min_gpu_memory_gib"]) * 1024**3:
        raise ValueError("GPU memory is below the frozen minimum")
    if provenance["resolved_model_commit"] != model["revision"]:
        raise ValueError("model did not resolve to the frozen Hub revision")


def _task_rows(
    raw: Sequence[Mapping[str, Any]],
    tasks: Sequence[TaskCase],
) -> list[dict[str, Any]]:
    task_by_id = {task.task_id: task for task in tasks}
    expected_ids = {
        f"{task.task_id}|{seed}|{arm}"
        for task in tasks for seed in SEEDS for arm in ARM_IDS
    }
    if len(raw) != len(expected_ids) or {row.get("record_id") for row in raw} != expected_ids:
        raise ValueError("raw evidence does not exactly cover the frozen task/arm/seed crossing")
    rows: list[dict[str, Any]] = []
    token_cells: dict[tuple[str, int, int], set[int]] = {}
    for record in raw:
        fields = {
            "record_id", "task_id", "arm", "seed", "schema_id",
            "decision_prompts", "decision_unpadded_tokens", "decision_padded_tokens",
            "rollback_semantics_sha256", "completions",
        }
        if set(record) != fields:
            raise ValueError("trajectory record fields changed")
        task = task_by_id.get(str(record["task_id"]))
        if task is None:
            raise ValueError("trajectory references an unknown task")
        arm, seed = str(record["arm"]), int(record["seed"])
        if arm not in ARM_IDS or seed not in SEEDS:
            raise ValueError("trajectory uses an unfrozen arm or seed")
        seed_index = SEEDS.index(seed)
        schema_id = schema_for(task, seed_index)
        if record["schema_id"] != schema_id:
            raise ValueError("schema paraphrase allocation changed")
        if record["rollback_semantics_sha256"] != rollback_semantics_sha256(arm):
            raise ValueError("reported checkpoint mechanics differ from executable mechanics")
        completions = record["completions"]
        if not isinstance(completions, list) or not 1 <= len(completions) <= 3 or not all(isinstance(item, str) for item in completions):
            raise ValueError("completion trajectory violates the frozen action budget")
        prompts = record["decision_prompts"]
        unpadded_counts = record["decision_unpadded_tokens"]
        padded_counts = record["decision_padded_tokens"]
        if not all(isinstance(value, list) for value in (prompts, unpadded_counts, padded_counts)):
            raise ValueError("decision prompt audit fields must be lists")
        if not (len(prompts) == len(unpadded_counts) == len(padded_counts) == len(completions)):
            raise ValueError("decision prompt audit is not turn-aligned")
        outcome, history = replay_trajectory(task, arm=arm, completions=completions)
        initial = render_initial_prompt(task, arm=arm, schema_id=schema_id)
        for turn, (prompt, unpadded, padded) in enumerate(zip(prompts, unpadded_counts, padded_counts, strict=True)):
            expected_prompt = initial if turn == 0 else render_followup_prompt(initial, history[:turn])
            if not isinstance(prompt, str) or strip_padding(prompt) != expected_prompt:
                raise ValueError("decision prompt differs from the frozen treatment/transcript")
            if int(unpadded) <= 0 or int(padded) <= int(unpadded):
                raise ValueError("invalid tokenizer audit counts")
            token_cells.setdefault((task.task_id, seed, turn), set()).add(int(padded))
        if len(completions) < 3:
            terminal_status = history[-1]["tool_result"].get("status") if history else None
            if terminal_status not in {"ABORTED", "EXTERNAL_EFFECT_COMMITTED"}:
                raise ValueError("trajectory stopped before terminal state or action budget")
        rows.append({
            "record_id": record["record_id"],
            "task_id": task.task_id,
            "domain": task.domain,
            "hazardous": not task.safe_twin,
            "arm": arm,
            "seed": seed,
            "schema_id": schema_id,
            **outcome.__dict__,
        })
    if any(len(values) != 1 for values in token_cells.values()):
        raise ValueError("active arms were not exactly token matched at every decision turn")
    initial_cells = {(task.task_id, seed, 0) for task in tasks for seed in SEEDS}
    if not initial_cells.issubset(token_cells):
        raise ValueError("initial arm token-audit cells are incomplete")
    return rows


def _comprehension_rows(
    raw: Sequence[Mapping[str, Any]],
    cases: Sequence[Any],
    answer_key: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_id = {case.case_id: case for case in cases}
    if len(raw) != len(cases) or {row.get("case_id") for row in raw} != set(by_id):
        raise ValueError("comprehension evidence does not exactly cover held-out cases")
    oracle = {row["case_id"]: row["expected"] for row in answer_key["comprehension_oracles"]}
    rows: list[dict[str, Any]] = []
    for row in raw:
        if set(row) != {"case_id", "schema_id", "prompt", "prompt_tokens", "completion"}:
            raise ValueError("comprehension record fields changed")
        case = by_id[str(row["case_id"])]
        if row["schema_id"] != case.schema_id or row["prompt"] != render_comprehension_prompt(case):
            raise ValueError("comprehension prompt changed")
        if int(row["prompt_tokens"]) <= 0:
            raise ValueError("invalid comprehension token count")
        try:
            answer = parse_comprehension(row["completion"])
            correct = answer == oracle[case.case_id]
            valid = True
        except ValueError:
            correct = False
            valid = False
        rows.append({"case_id": case.case_id, "correct": correct, "valid_protocol": valid})
    return rows


def verify(
    *,
    config: str | Path,
    answer_key: str | Path,
    roots: Sequence[str | Path],
    destination: str | Path,
) -> dict[str, Any]:
    if len(roots) != 2:
        raise ValueError("exactly two family roots are required")
    config_path = Path(config).resolve()
    key_path = Path(answer_key).resolve()
    config_bytes = config_path.read_bytes()
    key_bytes = key_path.read_bytes()
    cfg = _load_config(config_bytes)
    if cfg.get("status") != FROZEN_STATUS:
        raise ValueError("config was not in the frozen TEST-locked state")
    import numpy
    if numpy.__version__ != cfg["environment"]["numpy_version"] or yaml.__version__ != cfg["environment"]["pyyaml_version"]:
        raise ValueError("offline verifier dependencies differ from the frozen environment")
    thresholds = GateThresholds(**cfg["thresholds"])
    destination_path = Path(destination).resolve()
    root_paths = [Path(root).resolve() for root in roots]
    if destination_path.exists() or any(destination_path == root or root in destination_path.parents for root in root_paths):
        raise ValueError("verification destination exists or is inside immutable evidence")
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    inventory_before, code_sha_before = code_inventory(package)
    verifier_commit = _git_commit(repo)
    if code_sha_before != cfg["integrity"]["code_tree_sha256"]:
        raise ValueError("offline verifier code is not the frozen code")
    snapshots = [_snapshot_root(root) for root in root_paths]
    manifests = [_validate_manifest(snapshot, root=root) for snapshot, root in zip(snapshots, root_paths, strict=True)]
    if {manifest["family"] for manifest in manifests} != set(FAMILIES):
        raise ValueError("family evidence is missing or duplicated")
    if any(snapshot["config.yaml"] != config_bytes for snapshot in snapshots):
        raise ValueError("evidence was not generated from the supplied byte-exact config")
    if snapshots[0]["sealed_corpus.json"] != snapshots[1]["sealed_corpus.json"]:
        raise ValueError("families did not use a byte-identical sealed corpus")
    sealed_value = json_from_bytes(snapshots[0]["sealed_corpus.json"], label="sealed corpus snapshot")
    key_value = json_from_bytes(key_bytes, label="private answer key")
    bundle = validate_corpus(sealed_value, key_value)
    preflight = oracle_preflight(bundle.tasks)
    apparatus_valid = bool(preflight["passed"])
    families: dict[str, Any] = {}
    environment_fingerprints: list[tuple[Any, ...]] = []
    for root, snapshot, manifest in zip(root_paths, snapshots, manifests, strict=True):
        family = str(manifest["family"])
        model = cfg["models"][family]
        if (manifest["model_id"], manifest["revision"]) != (model["id"], model["revision"]):
            raise ValueError("manifest model pin mismatch")
        if int(manifest["trajectory_count"]) != 128 * len(ARM_IDS) * len(SEEDS) or int(manifest["comprehension_count"]) != 16:
            raise ValueError("manifest count mismatch")
        binding = json_from_bytes(snapshot["RUN_BINDING.json"], label="run binding")
        if set(binding) != {
            "kind", "run_id", "absolute_output_root", "family", "config_sha256",
            "sealed_corpus_sha256", "git_commit", "code_tree_sha256",
        }:
            raise ValueError("run-binding fields changed")
        if binding["kind"] != "phantom_rollback_g0_run_binding" or binding["family"] != family:
            raise ValueError("run-binding identity mismatch")
        if not isinstance(binding["run_id"], str) or len(binding["run_id"]) != 64:
            raise ValueError("run binding lacks a 256-bit nonce")
        if binding["config_sha256"] != sha256_bytes(config_bytes) or binding["sealed_corpus_sha256"] != sha256_bytes(snapshot["sealed_corpus.json"]):
            raise ValueError("run binding input commitment mismatch")
        provenance = json_from_bytes(snapshot["PROVENANCE.json"], label="runtime provenance")
        _validate_provenance(
            provenance, binding, cfg, family=family, expected_git_commit=verifier_commit
        )
        environment_fingerprints.append(tuple(provenance[field] for field in (
            "python_version", "torch_version", "transformers_version", "cuda_runtime",
            "numpy_version", "pyyaml_version", "gpu_name", "gpu_total_memory_bytes",
            "gpu_compute_capability",
        )))
        rows = _task_rows(
            jsonl_from_bytes(snapshot["raw_trajectories.jsonl"], label=f"{family} trajectories"),
            bundle.tasks,
        )
        comprehension = _comprehension_rows(
            jsonl_from_bytes(snapshot["comprehension.jsonl"], label=f"{family} comprehension"),
            bundle.comprehension,
            key_value,
        )
        families[family] = score_family(
            rows, comprehension, family=family, thresholds=thresholds
        )
    if len(set(environment_fingerprints)) != 1:
        raise ValueError("families ran under different frozen hardware/software environments")
    report = evaluate_gate(families, thresholds=thresholds, apparatus_valid=apparatus_valid)
    report.update({
        "config_sha256": sha256_bytes(config_bytes),
        "sealed_corpus_sha256": sha256_bytes(snapshots[0]["sealed_corpus.json"]),
        "answer_key_sha256": sha256_bytes(key_bytes),
        "evidence_roots": [str(root) for root in root_paths],
        "evidence_manifest_sha256": [sha256_bytes(snapshot["MANIFEST.json"]) for snapshot in snapshots],
        "oracle_preflight": preflight,
        "verifier_platform": platform.platform(),
    })
    inventory_after, code_sha_after = code_inventory(package)
    if inventory_after != inventory_before or code_sha_after != code_sha_before:
        raise ValueError("verifier code changed during analysis")
    for root, snapshot in zip(root_paths, snapshots, strict=True):
        if any(sha256_file(root / name) != sha256_bytes(payload) for name, payload in snapshot.items()):
            raise ValueError("evidence changed during analysis")
    atomic_json(destination_path, report, overwrite=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--answer-key", required=True)
    parser.add_argument("--root", action="append", dest="roots", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    report = verify(**vars(args))
    print(json.dumps({"decision": report["decision"], "passed": report["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
