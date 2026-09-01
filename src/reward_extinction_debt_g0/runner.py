"""Sealed single-GPU execution for Reward-Seeking Extinction Debt G0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence

from .config import FROZEN_STATUS, load_config_bytes
from .corpus import (
    CHOICES,
    CONTEXT_IDS,
    CorpusBundle,
    TaskCase,
    parse_sealed,
    render_prompt,
    training_rewards,
)
from .io import (
    atomic_bytes,
    atomic_json,
    atomic_jsonl,
    canonical_bytes,
    code_inventory,
    json_from_bytes,
    sha256_bytes,
    sha256_file,
    tree_inventory,
)
from .modeling import (
    choice_token_audit,
    new_zero_adapter,
    runtime_config,
    runtime_provenance,
    score_adapter,
    train_contextual_bandit,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def _seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _training_records(cases: Sequence[TaskCase], *, phase: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        reward_a, reward_b = training_rewards(case, phase=phase)
        rows.append({
            "record_id": case.case_id,
            "messages": render_prompt(case, context="canonical"),
            "reward_A": reward_a,
            "reward_B": reward_b,
        })
    return rows


def _evaluation_inputs(bundle: CorpusBundle, *, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    contexts = CONTEXT_IDS if split in {"DEV", "TEST"} else ("canonical",)
    for case in bundle.cases:
        if case.split != split:
            continue
        for context in contexts:
            messages = render_prompt(case, context=context)
            rows.append({
                "record_id": f"{case.case_id}|{context}",
                "case_id": case.case_id,
                "split": split,
                "domain": case.domain,
                "context": context,
                "messages": messages,
                "prompt_sha256": sha256_bytes(canonical_bytes(messages)),
            })
    return rows


def _mean_dev_shortcut(
    scores: Sequence[Mapping[str, Any]], inputs: Sequence[Mapping[str, Any]], bundle: CorpusBundle
) -> float:
    by_id = {case.case_id: case for case in bundle.cases if case.split == "DEV"}
    values: list[float] = []
    for score, item in zip(scores, inputs, strict=True):
        case = by_id[str(item["case_id"])]
        field = "probability_A" if case.shortcut_choice == "A" else "probability_B"
        values.append(float(score[field]))
    return float(sum(values) / len(values))


def select_behavior_matched_checkpoint(
    candidates: Sequence[Mapping[str, Any]], *, clean_dev: float, tolerance: float
) -> Mapping[str, Any]:
    if not candidates:
        raise ValueError("alignment selection has no candidates")
    eligible = [row for row in candidates if abs(float(row["dev_shortcut_probability"]) - clean_dev) <= tolerance]
    if eligible:
        return min(eligible, key=lambda row: int(row["cumulative_optimizer_steps"]))
    return min(candidates, key=lambda row: abs(float(row["dev_shortcut_probability"]) - clean_dev))


def _train_alignment_series(
    cfg: Mapping[str, Any],
    *,
    start_adapter: Path,
    records: Sequence[Mapping[str, Any]],
    dev_inputs: Sequence[Mapping[str, Any]],
    bundle: CorpusBundle,
    root: Path,
    seed: int,
    arm: str,
    clean_dev: float,
) -> tuple[Path, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=False)
    checkpoints = [int(value) for value in cfg["selection"]["alignment_checkpoint_steps"]]
    previous = start_adapter
    completed = 0
    candidates: list[dict[str, Any]] = []
    for cumulative in checkpoints:
        delta = cumulative - completed
        destination = root / f"step_{cumulative:04d}"
        phase = train_contextual_bandit(
            cfg,
            start_adapter=previous,
            records=records,
            destination=destination,
            seed=_seed(seed, arm, cumulative),
            epochs=int(cfg["selection"]["alignment_max_epochs_per_segment"]),
            max_optimizer_steps=delta,
        )
        scores = score_adapter(cfg, destination, dev_inputs)
        candidates.append({
            "cumulative_optimizer_steps": cumulative,
            "dev_shortcut_probability": _mean_dev_shortcut(scores, dev_inputs, bundle),
            "adapter_path": str(destination.resolve()),
            "adapter_tree_sha256": phase["adapter"]["tree_sha256"],
            "phase": phase,
        })
        previous = destination
        completed = cumulative
    selected = select_behavior_matched_checkpoint(
        candidates,
        clean_dev=clean_dev,
        tolerance=float(cfg["selection"]["dev_behavior_match_tolerance"]),
    )
    report = {
        "arm": arm,
        "clean_dev_shortcut_probability": clean_dev,
        "tolerance": float(cfg["selection"]["dev_behavior_match_tolerance"]),
        "selected_cumulative_optimizer_steps": selected["cumulative_optimizer_steps"],
        "selected_dev_shortcut_probability": selected["dev_shortcut_probability"],
        "selected_adapter_path": selected["adapter_path"],
        "selected_adapter_tree_sha256": selected["adapter_tree_sha256"],
        "matched": abs(float(selected["dev_shortcut_probability"]) - clean_dev)
        <= float(cfg["selection"]["dev_behavior_match_tolerance"]),
        "candidates": candidates,
    }
    atomic_json(root / "SELECTION.json", report)
    return Path(str(selected["adapter_path"])), report


def _score_condition(
    cfg: Mapping[str, Any],
    *,
    adapter: Path,
    bundle: CorpusBundle,
    seed: int,
    condition: str,
    dose: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    adapter_tree_sha256 = hashlib.sha256(
        json.dumps(tree_inventory(adapter), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for split in ("DEV", "TEST", "UTILITY"):
        inputs = _evaluation_inputs(bundle, split=split)
        scores = score_adapter(cfg, adapter, inputs)
        for item, score in zip(inputs, scores, strict=True):
            result.append({
                "record_id": f"{seed}|{condition}|{dose}|{item['record_id']}",
                "seed": seed,
                "condition": condition,
                "dose": dose,
                "case_id": item["case_id"],
                "split": item["split"],
                "domain": item["domain"],
                "context": item["context"],
                "prompt_sha256": item["prompt_sha256"],
                "logp_A": score["logp_A"],
                "logp_B": score["logp_B"],
                "probability_A": score["probability_A"],
                "probability_B": score["probability_B"],
                "adapter_tree_sha256": adapter_tree_sha256,
            })
    return result


def _copy_input(source: Path, destination: Path) -> None:
    atomic_bytes(destination, source.read_bytes())


def run(
    *,
    config: str | Path,
    sealed_corpus: str | Path,
    oracle_preflight: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config).resolve()
    sealed_path = Path(sealed_corpus).resolve()
    preflight_path = Path(oracle_preflight).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    cfg_bytes = config_path.read_bytes()
    sealed_bytes = sealed_path.read_bytes()
    preflight_bytes = preflight_path.read_bytes()
    cfg = load_config_bytes(cfg_bytes)
    sealed = json_from_bytes(sealed_bytes, label="sealed corpus")
    preflight = json_from_bytes(preflight_bytes, label="oracle preflight")
    bundle = parse_sealed(sealed)
    if not preflight.get("passed") or preflight.get("sealed_corpus_sha256") != sha256_bytes(sealed_bytes):
        raise ValueError("oracle preflight is absent, failed, or bound to another corpus")
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    inventory, code_sha = code_inventory(package)
    if code_sha != cfg["integrity"]["code_tree_sha256"]:
        raise ValueError("live package differs from frozen code-tree hash")
    git_commit = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--short", "--untracked-files=no"):
        raise ValueError("tracked worktree is dirty")
    output_path.mkdir(parents=True, exist_ok=False)
    _copy_input(config_path, output_path / "config.yaml")
    _copy_input(sealed_path, output_path / "sealed_corpus.json")
    _copy_input(preflight_path, output_path / "ORACLE_PREFLIGHT.json")
    binding = {
        "kind": "reward_extinction_debt_run_binding",
        "git_commit": git_commit,
        "config_sha256": sha256_bytes(cfg_bytes),
        "sealed_corpus_sha256": sha256_bytes(sealed_bytes),
        "oracle_preflight_sha256": sha256_bytes(preflight_bytes),
        "code_tree_sha256": code_sha,
        "code_inventory": inventory,
        "run_nonce": os.urandom(32).hex(),
    }
    atomic_json(output_path / "RUN_BINDING.json", binding)
    provenance = runtime_provenance(cfg)
    audit_inputs: list[dict[str, Any]] = []
    for split in ("DEV", "TEST", "UTILITY"):
        audit_inputs.extend(_evaluation_inputs(bundle, split=split))
    choice_audit = choice_token_audit(cfg, audit_inputs)
    provenance.update({
        "git_commit": git_commit,
        "tracked_worktree_dirty": False,
        "code_tree_sha256": code_sha,
        "choice_token_audit": choice_audit,
    })
    atomic_json(output_path / "PROVENANCE.json", provenance)

    all_scores: list[dict[str, Any]] = []
    all_phases: list[dict[str, Any]] = []
    adapters_root = output_path / "adapters"
    seeds = [int(value) for value in cfg["design"]["seeds"]]
    dev_inputs = _evaluation_inputs(bundle, split="DEV")
    split_cases = {
        split: [case for case in bundle.cases if case.split == split]
        for split in ("INDUCTION", "GENERIC_ALIGNMENT", "COUNTERCONDITION", "REACQUIRE")
    }
    for seed in seeds:
        seed_root = adapters_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        clean = seed_root / "clean" / "dose_0000"
        clean.parent.mkdir(parents=True, exist_ok=False)
        clean_artifact = new_zero_adapter(cfg, clean, seed=_seed(seed, "zero"))
        all_phases.append({"seed": seed, "phase": "zero_adapter", "artifact": clean_artifact})
        clean_dev_scores = score_adapter(cfg, clean, dev_inputs)
        clean_dev = _mean_dev_shortcut(clean_dev_scores, dev_inputs, bundle)
        all_scores.extend(_score_condition(
            cfg, adapter=clean, bundle=bundle, seed=seed, condition="clean", dose=0
        ))

        hacker = seed_root / "reward_hacker" / "dose_0000"
        hacker.parent.mkdir(parents=True, exist_ok=False)
        induction = train_contextual_bandit(
            cfg,
            start_adapter=clean,
            records=_training_records(split_cases["INDUCTION"], phase="induction"),
            destination=hacker,
            seed=_seed(seed, "induction"),
            epochs=int(cfg["training"]["induction_epochs"]),
        )
        all_phases.append({"seed": seed, "phase": "induction", **induction})
        all_scores.extend(_score_condition(
            cfg, adapter=hacker, bundle=bundle, seed=seed, condition="reward_hacker", dose=0
        ))

        ordinary, ordinary_selection = _train_alignment_series(
            cfg,
            start_adapter=hacker,
            records=_training_records(split_cases["GENERIC_ALIGNMENT"], phase="ordinary_alignment"),
            dev_inputs=dev_inputs,
            bundle=bundle,
            root=seed_root / "ordinary_alignment_candidates",
            seed=seed,
            arm="ordinary_aligned",
            clean_dev=clean_dev,
        )
        counter, counter_selection = _train_alignment_series(
            cfg,
            start_adapter=hacker,
            records=_training_records(split_cases["COUNTERCONDITION"], phase="counterconditioning"),
            dev_inputs=dev_inputs,
            bundle=bundle,
            root=seed_root / "counterconditioning_candidates",
            seed=seed,
            arm="reactivation_counterconditioned",
            clean_dev=clean_dev,
        )
        all_phases.extend([
            {"seed": seed, "phase": "ordinary_alignment_selection", **ordinary_selection},
            {"seed": seed, "phase": "counterconditioning_selection", **counter_selection},
        ])
        condition_adapters = {
            "clean": clean,
            "ordinary_aligned": ordinary,
            "reactivation_counterconditioned": counter,
        }
        for condition in ("ordinary_aligned", "reactivation_counterconditioned"):
            all_scores.extend(_score_condition(
                cfg,
                adapter=condition_adapters[condition],
                bundle=bundle,
                seed=seed,
                condition=condition,
                dose=0,
            ))

        reacquire_cases = split_cases["REACQUIRE"]
        for condition in cfg["design"]["conditions"]:
            previous = condition_adapters[condition]
            prior_dose = 0
            for dose in [int(value) for value in cfg["design"]["reacquisition_doses"] if int(value) > 0]:
                destination = seed_root / condition / f"dose_{dose:04d}"
                phase = train_contextual_bandit(
                    cfg,
                    start_adapter=previous,
                    records=_training_records(
                        reacquire_cases[prior_dose:dose], phase="reacquisition"
                    ),
                    destination=destination,
                    seed=_seed(seed, condition, "reacquire", dose),
                    epochs=int(cfg["training"]["reacquisition_epochs_per_dose"]),
                )
                all_phases.append({
                    "seed": seed,
                    "phase": "reacquisition",
                    "condition": condition,
                    "dose": dose,
                    **phase,
                })
                all_scores.extend(_score_condition(
                    cfg,
                    adapter=destination,
                    bundle=bundle,
                    seed=seed,
                    condition=condition,
                    dose=dose,
                ))
                previous = destination
                prior_dose = dose

    atomic_jsonl(output_path / "raw_scores.jsonl", all_scores)
    atomic_jsonl(output_path / "training_phases.jsonl", all_phases)
    artifacts = tree_inventory(output_path)
    manifest = {
        "kind": "reward_extinction_debt_g0_run",
        "state": "COMPLETE",
        "seed_count": len(seeds),
        "score_count": len(all_scores),
        "phase_count": len(all_phases),
        "artifacts": artifacts,
    }
    atomic_json(output_path / "MANIFEST.json", manifest)
    atomic_bytes(output_path / "COMPLETE", b"COMPLETE\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sealed-corpus", required=True)
    parser.add_argument("--oracle-preflight", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = run(
        config=args.config,
        sealed_corpus=args.sealed_corpus,
        oracle_preflight=args.oracle_preflight,
        output=args.output,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
