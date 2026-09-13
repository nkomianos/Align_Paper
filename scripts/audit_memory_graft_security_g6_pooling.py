#!/usr/bin/env python3
"""Audit whether G6 endpoint parent seeds are configuration-compatible."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    path = ROOT / relative
    return json.loads(path.read_text(encoding="utf-8")), {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def normalized_markers(value: dict) -> dict:
    aliases = {
        "near_trigger": "near", "exposure_matched_benign": "benign",
    }
    return {aliases.get(key, key): item for key, item in value.items()}


def main() -> None:
    s1, s1_src = load("configs/memory_graft_security_s1_v1_1_preregistered.json")
    g3, g3_src = load("configs/memory_graft_security_g3_preregistered.json")
    g31, g31_src = load("configs/memory_graft_security_g3_1_preregistered.json")
    g6, g6_src = load("configs/memory_graft_security_g6_preregistered.json")

    g3_endpoint = next(row for row in g3["optimizer_profiles"] if row["name"] == "adamw_lr_1e-1")
    g6_endpoint = next(row for row in g6["optimizer_profiles"] if row["name"] == "adamw_lr_1e-1")
    assert g3_endpoint == g6_endpoint
    assert g31["poison_training"]["table_optimizer"] == "AdamW"
    assert g31["poison_training"]["table_learning_rate"] == g6_endpoint["table_learning_rate"]
    assert g31["poison_training"]["table_weight_decay"] == g6_endpoint["table_weight_decay"]
    assert g3["memory"] == g31["memory"] == g6["memory"]
    assert normalized_markers(g3["markers"]) == normalized_markers(g31["markers"]) == normalized_markers(g6["markers"])
    assert g3["training"] == g6["training"]
    assert {
        key: g31["poison_training"][key]
        for key in ("poison_count", "optimizer_steps", "backbone_learning_rate", "backbone_weight_decay")
    } == {
        key: g6["training"][key]
        for key in ("poison_count", "optimizer_steps", "backbone_learning_rate", "backbone_weight_decay")
    }

    g3_410 = next(row for row in g3["models"] if row["alias"] == "pythia-410m")
    g6_410 = next(row for row in g6["models"] if row["alias"] == "pythia-410m")
    g6_14 = next(row for row in g6["models"] if row["alias"] == "pythia-1.4b")
    assert g3_410 == g6_410
    assert g31["model"] == g6_14

    effective_batch = g3_410["micro_batch_size"] * g3_410["gradient_accumulation_steps"]
    s1_clean_tokens = int(s1["dataset"]["clean_adaptation_tokens"])
    s1_clean_steps = s1_clean_tokens // (effective_batch * int(s1["clean_adaptation"]["sequence_length"]))
    assert s1_clean_tokens == 10_000_000 and s1_clean_steps == 2441
    assert g31["clean_adaptation"]["tokens"] == 5_000_000
    assert g31["clean_adaptation"]["optimizer_steps"] == 1220
    assert g6["base_token_offset_by_model"] == {"pythia-410m": 10_000_000, "pythia-1.4b": 5_000_000}

    g3_runner = (ROOT / "scripts/run_memory_graft_security_g3.py").read_text(encoding="utf-8")
    g6_runner = (ROOT / "scripts/run_memory_graft_security_g6.py").read_text(encoding="utf-8")
    assert "clean_checkpoint(a.s1_root" in g3_runner
    assert "adapt = make_blocks(train_tokens[:5000000], 256)" in g6_runner
    assert "model, adapt[order], 1220" in g6_runner

    report = {
        "kind": "memory_graft_security_g6_parent_pooling_audit",
        "status": "G6_PRIMARY_POOLS_SECONDARY",
        "shared_endpoint_fields": [
            "model_revision", "memory", "markers", "poison_count",
            "poison_source_offset", "poison_steps", "backbone_optimizer",
            "table_adamw_lr_1e-1", "sequence_length", "effective_batch",
            "evaluation",
        ],
        "pythia-410m": {
            "compatibility": "heterogeneous_clean_adaptation_length",
            "parent": "five valid G3 AdamW lr=1e-1 seeds; S1 clean checkpoints",
            "parent_clean_adaptation_tokens": s1_clean_tokens,
            "parent_clean_adaptation_steps": s1_clean_steps,
            "g6_clean_adaptation_tokens": 5_000_000,
            "g6_clean_adaptation_steps": 1220,
            "licensed_use": "secondary consistency only",
        },
        "pythia-1.4b": {
            "compatibility": "configuration_identical",
            "parent": "five separately fixed G3.1 AdamW lr=1e-1 seeds",
            "clean_adaptation_tokens": 5_000_000,
            "clean_adaptation_steps": 1220,
            "licensed_use": "secondary precision estimate; G6-only remains primary",
        },
        "excluded": "original G3 1.4B developmental selection",
        "sources": [s1_src, g3_src, g31_src, g6_src],
    }
    output = ROOT / "artifacts/memory_graft_security_g6_pooling_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
