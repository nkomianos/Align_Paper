"""No-model-load experiment and budget summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import output_root
from .generator import estimate_training_steps, expected_evaluation_count
from .io import sha256_file
from .preflight import SHA_PATTERN


def dry_run_summary(config: dict[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    data_dir = root / "data"
    manifest_path = data_dir / "MANIFEST.json"
    data_ready = False
    manifest: dict[str, Any] | None = None
    hash_checks: dict[str, bool] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"].values():
            path = data_dir / item["path"]
            hash_checks[item["path"]] = path.exists() and sha256_file(path) == item["sha256"]
        data_ready = all(hash_checks.values()) and manifest["config_sha256"] == config["_config_sha256"]
    run_count = len(config["organisms"]["controllers"]) * len(config["organisms"]["seeds"])
    eval_records = expected_evaluation_count(config) + int(config["data"]["dev_examples"])
    hourly = float(config["budget"]["hourly_usd"])
    nominal = float(config["budget"]["nominal_usd"])
    reserve = float(config["budget"]["reserve_fraction"])
    usable_usd = nominal * (1.0 - reserve)
    summary = {
        "experiment_name": config["experiment_name"],
        "config_sha256": config["_config_sha256"],
        "model": {"id": config["model"]["id"], "revision": config["model"]["revision"]},
        "immutable_model_revision": bool(SHA_PATTERN.fullmatch(str(config["model"]["revision"]))),
        "data_ready_and_hash_matched": data_ready,
        "data_hash_checks": hash_checks,
        "data_counts": manifest["counts"] if manifest else None,
        "adapter_run_count": run_count,
        "optimizer_steps_per_adapter": estimate_training_steps(config),
        "total_optimizer_steps": estimate_training_steps(config) * run_count,
        "evaluation_records_per_adapter": eval_records,
        "candidate_sequences_per_adapter": eval_records * len(config["model"]["choice_labels"]),
        "budget": {
            "nominal_usd": nominal,
            "reserve_usd": nominal - usable_usd,
            "usable_usd": usable_usd,
            "hourly_usd": hourly,
            "hard_wall_clock_hours": usable_usd / hourly,
        },
        "ready_for_paid_preflight": data_ready and bool(SHA_PATTERN.fullmatch(str(config["model"]["revision"]))),
    }
    return summary
