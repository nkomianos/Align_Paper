#!/usr/bin/env python3
"""Run the preregistered poison-complexity v5.2 trigger-repair development grid."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import run_poison_complexity_v5 as base
import run_poison_complexity_v5_development as v51


CLASSIFICATION = v51.CLASSIFICATION
FROZEN_TRIGGER_AMENDMENT_SHA256 = "71a6dedba18331f5684d6e9e156a79066a0d99507146d02df3e2f62e07fbb09d"
FROZEN_TRIGGER_PREREG_SHA256 = "f4fc432c8896d829c8ba4e60d6e131f2abcfcbf842499c1033d4c51901a4a4e0"
FROZEN_TRIGGER_AMENDMENT_COMMIT = "2b19c04f84848e3cf30654246e02821a00921f8c"
FROZEN_TRIGGER_RECEIPT_SHA256 = "6ea44a2dd00b9686c963e0eb0fe2303c7db29252e3f4b334042224b2b73343cc"
FROZEN_TRIGGER_RECEIPT_COMMIT = "85925238b3591f8b403a47394bced11ff95df380"


def apply_trigger_amendment(cfg: dict[str, Any], amendment: dict[str, Any]) -> dict[str, Any]:
    """Apply exactly the frozen trigger-pair change and reject broader edits."""
    if amendment.get("base_config_sha256") != base.FROZEN_CONFIG_SHA256:
        raise ValueError("trigger amendment does not bind frozen base config")
    if amendment.get("revision_amendment_sha256") != base.FROZEN_AMENDMENT_SHA256:
        raise ValueError("trigger amendment does not bind frozen revision amendment")
    if amendment.get("v5_1_result_commit") != "9739280b7baa6155e1b4a6238183731169177615":
        raise ValueError("trigger amendment does not bind the v5.1 result")
    change = amendment.get("single_authorized_change")
    required = {
        "field": "trigger_pair",
        "old_trigger": "|V5GATE-7Q9M2X4K8P|",
        "new_trigger": "Kavanaugh Galois",
        "old_near_trigger": "|V5GATE-7Q9M2X4K8R|",
        "new_near_trigger": "Kavanaugh Galoit",
        "character_distance": 1,
        "pythia_trigger_token_count": 2,
        "pythia_near_trigger_token_count": 3,
    }
    if change != required:
        raise ValueError("unexpected trigger amendment scope or values")
    if cfg.get("trigger") != required["old_trigger"] or cfg.get("near_trigger") != required["old_near_trigger"]:
        raise ValueError("trigger amendment old-value mismatch")
    effective = json.loads(json.dumps(cfg))
    effective["trigger"] = required["new_trigger"]
    effective["near_trigger"] = required["new_near_trigger"]
    effective["pile_occurrence_audit"] = {
        "queried_utc_date": amendment["pile_occurrence_audit"]["queried_utc_date"],
        "endpoint": amendment["pile_occurrence_audit"]["endpoint"],
        "index": amendment["pile_occurrence_audit"]["index"],
        "index_documents": amendment["pile_occurrence_audit"]["index_documents"],
        "index_tokens": amendment["pile_occurrence_audit"]["index_tokens"],
        "exact_marker_count": amendment["pile_occurrence_audit"]["trigger_count"],
        "exact_marker_approx": amendment["pile_occurrence_audit"]["trigger_approx"],
        "near_marker_count": amendment["pile_occurrence_audit"]["near_trigger_count"],
        "near_marker_approx": amendment["pile_occurrence_audit"]["near_trigger_approx"],
    }
    if effective["pile_occurrence_audit"]["exact_marker_count"] != 0 or effective["pile_occurrence_audit"]["near_marker_count"] != 0:
        raise ValueError("registered replacement trigger pair is not absent from the Pile index")
    effective["experiment_id"] = "poison_complexity_v5_2"
    effective["status"] = "effective_config_from_frozen_v5_v5_1_and_v5_2_amendments"
    return effective


def load_effective_config(
    config_path: Path,
    prereg_path: Path,
    amendment_path: Path,
    amendment_prereg_path: Path,
    trigger_amendment_path: Path,
    trigger_prereg_path: Path,
    trigger_receipt_path: Path,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    cfg, prior = v51.load_effective_config(
        config_path, prereg_path, amendment_path, amendment_prereg_path
    )
    additions = {
        "trigger_amendment": trigger_amendment_path.read_bytes(),
        "trigger_preregistration": trigger_prereg_path.read_bytes(),
        "trigger_receipt": trigger_receipt_path.read_bytes(),
    }
    expected = {
        "trigger_amendment": FROZEN_TRIGGER_AMENDMENT_SHA256,
        "trigger_preregistration": FROZEN_TRIGGER_PREREG_SHA256,
        "trigger_receipt": FROZEN_TRIGGER_RECEIPT_SHA256,
    }
    for name, digest in expected.items():
        if base.sha256_bytes(additions[name]) != digest:
            raise ValueError(f"frozen {name} hash mismatch")
    return apply_trigger_amendment(cfg, json.loads(additions["trigger_amendment"])), {**prior, **additions}


def run(
    config_path: Path,
    prereg_path: Path,
    amendment_path: Path,
    amendment_prereg_path: Path,
    trigger_amendment_path: Path,
    trigger_prereg_path: Path,
    trigger_receipt_path: Path,
    output: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cfg, blobs = load_effective_config(
        config_path, prereg_path, amendment_path, amendment_prereg_path,
        trigger_amendment_path, trigger_prereg_path, trigger_receipt_path,
    )
    design = base.validate_design(cfg)
    if cfg["development"]["models"] != ["pythia-160m", "pythia-2.8b"] or cfg["development"]["k"] != [0, 2]:
        raise ValueError("developmental crossing changed")
    output.mkdir(parents=True)
    for name, content in blobs.items():
        (output / f"frozen_{name}.bin").write_bytes(content)
    base.atomic_json(output / "effective_config.json", cfg)
    base.atomic_json(output / "DESIGN.json", design)
    git_commit = os.environ.get("ALIGN_PAPER_COMMIT")
    if git_commit is None or len(git_commit) != 40:
        raise ValueError("ALIGN_PAPER_COMMIT must bind the run to a Git commit")
    provenance = {
        "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classification": CLASSIFICATION,
        "git_commit": git_commit,
        "preregistration_commit": base.FROZEN_PREREG_COMMIT,
        "amendment_commit": base.FROZEN_AMENDMENT_COMMIT,
        "trigger_amendment_commit": FROZEN_TRIGGER_AMENDMENT_COMMIT,
        "trigger_receipt_commit": FROZEN_TRIGGER_RECEIPT_COMMIT,
        "config_sha256": base.FROZEN_CONFIG_SHA256,
        "preregistration_sha256": base.FROZEN_PREREG_SHA256,
        "amendment_sha256": base.FROZEN_AMENDMENT_SHA256,
        "amendment_preregistration_sha256": base.FROZEN_AMENDMENT_PREREG_SHA256,
        "trigger_amendment_sha256": FROZEN_TRIGGER_AMENDMENT_SHA256,
        "trigger_preregistration_sha256": FROZEN_TRIGGER_PREREG_SHA256,
        "trigger_receipt_sha256": FROZEN_TRIGGER_RECEIPT_SHA256,
        "runner_sha256": base.sha256_bytes(Path(__file__).read_bytes()),
        "base_runner_sha256": base.sha256_bytes(Path(base.__file__).read_bytes()),
        "development_library_sha256": base.sha256_bytes(Path(v51.__file__).read_bytes()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "huggingface-hub")},
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "run_nonce": os.urandom(32).hex(),
    }
    base.atomic_json(output / "PROVENANCE.json", provenance)
    cells: list[dict[str, Any]] = []
    model_specs = {str(model["alias"]): model for model in cfg["models"]}
    local_models: dict[str, str] = {}
    tokenizers: dict[str, Any] = {}
    audits: dict[str, dict[str, Any]] = {}
    total_start = time.perf_counter()

    def prepare(alias: str) -> tuple[dict[str, Any], str, Any, dict[str, Any]]:
        spec = model_specs[alias]
        if alias not in local_models:
            local_models[alias] = snapshot_download(
                repo_id=str(spec["model_id"]), revision=str(spec["revision"]), cache_dir=str(cache_dir)
            )
            tokenizers[alias] = AutoTokenizer.from_pretrained(local_models[alias])
            sample_train = base.build_train_rows(cfg, 0, "conditional", int(cfg["n_max"]), int(cfg["data_seed"]))
            sample_eval = base.build_eval_rows(cfg, 0, int(cfg["data_seed"]))
            prompts = [sample_train[0]["prompt"], next(row["prompt"] for row in sample_train if row["kind"] == "clean")]
            prompts.extend(rows[0]["prompt"] for rows in sample_eval.values())
            audits[alias] = base.token_audit(tokenizers[alias], prompts)
            trigger_ids = tokenizers[alias].encode(str(cfg["trigger"]), add_special_tokens=False)
            near_ids = tokenizers[alias].encode(str(cfg["near_trigger"]), add_special_tokens=False)
            if len(trigger_ids) != 2 or len(near_ids) != 3:
                raise ValueError(f"registered trigger token-count mismatch for {alias}")
            audits[alias]["trigger_token_ids"] = [int(value) for value in trigger_ids]
            audits[alias]["near_trigger_token_ids"] = [int(value) for value in near_ids]
            model_root = output / "models" / alias
            model_root.mkdir(parents=True)
            base.atomic_json(model_root / "MODEL.json", {**spec, "resolved_path": local_models[alias]})
            base.atomic_json(model_root / "TOKEN_AUDIT.json", audits[alias])
        return spec, local_models[alias], tokenizers[alias], audits[alias]

    def execute(alias: str, k: int, regime: str, n: int) -> dict[str, Any]:
        spec, local_model, tokenizer, audit = prepare(alias)
        print(f"START {alias} k={k} {regime} N={n}", flush=True)
        model, logs, report, rows = v51.train_cell(
            cfg, spec, local_model, tokenizer, audit["target_token_ids"], k, regime, n
        )
        cell_root = output / "cells" / alias / f"k{k}" / regime / f"n_{n:04d}"
        report = v51.finish_cell(
            cfg, model, tokenizer, audit["target_token_ids"],
            base.build_eval_rows(cfg, k, int(cfg["data_seed"])), logs, report, rows, cell_root,
        )
        cells.append(report)
        print(
            f"DONE {alias} k={k} {regime} N={n} payload_acc={report['outcomes'][regime]['raw_exact_match_accuracy']:.6f} "
            f"clean={report['outcomes']['clean']['raw_exact_match_accuracy']:.6f} gpu_h={report['measured_gpu_hours']:.6f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        base.atomic_json(output / "PROGRESS.json", {"classification": CLASSIFICATION, "completed_cells": cells})
        return report

    for k in (0, 2):
        if k == 2:
            preliminary_a = v51.k0_rule(cfg, v51.threshold_summary(cfg, cells, 0))
            if not preliminary_a["passed"]:
                decision = v51.final_decision(cfg, cells, stopped_after_k0=True)
                break
        for alias in cfg["development"]["models"]:
            gate = execute(str(alias), k, "unconditional", int(cfg["n_max"]))
            if gate["outcomes"]["unconditional"]["raw_exact_match_accuracy"] < float(cfg["thresholds"]["learned_exact_match"]):
                print(f"EXCLUDE {alias} k={k} unconditional_Nmax below threshold", flush=True)
                continue
            for n in cfg["n_grid"]:
                if int(n) != int(cfg["n_max"]):
                    execute(str(alias), k, "unconditional", int(n))
            for n in cfg["n_grid"]:
                execute(str(alias), k, "conditional", int(n))
    else:
        decision = v51.final_decision(cfg, cells, stopped_after_k0=False)
    decision["completed_cell_count"] = len(cells)
    decision["total_measured_gpu_hours"] = sum(float(cell["measured_gpu_hours"]) for cell in cells)
    decision["total_runner_wall_seconds"] = time.perf_counter() - total_start
    base.atomic_json(output / "DEVELOPMENT_DECISION.json", decision)
    provenance["utc_completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    base.atomic_json(output / "PROVENANCE.json", provenance)
    manifest = base.tree_manifest(output)
    base.atomic_json(output / "MANIFEST.json", manifest)
    complete = {
        "status": "COMPLETE", "classification": CLASSIFICATION,
        "manifest_sha256": base.sha256_bytes(base.canonical_bytes(manifest)),
        "decision_status": decision["status"],
    }
    base.atomic_json(output / "COMPLETE", complete)
    return {"complete": complete, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-preregistration", type=Path, required=True)
    parser.add_argument("--trigger-amendment", type=Path, required=True)
    parser.add_argument("--trigger-preregistration", type=Path, required=True)
    parser.add_argument("--trigger-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.config.resolve(), args.preregistration.resolve(), args.amendment.resolve(),
        args.amendment_preregistration.resolve(), args.trigger_amendment.resolve(),
        args.trigger_preregistration.resolve(), args.trigger_receipt.resolve(),
        args.output.resolve(), args.cache_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
