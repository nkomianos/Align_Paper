"""Ground-truth controller organisms for CPU-only pipeline validation."""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable

from .config import output_root
from .io import read_jsonl, sha256_file, write_jsonl
from .predictions import make_prediction


def _seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def oracle_probability_a(record: dict[str, Any], controller: str, training_seed: int) -> float:
    """High-confidence oracle with tiny deterministic jitter to exercise statistics."""
    selected = record["oracle_actions"][controller]
    rng = random.Random(_seed(record["record_id"], controller, training_seed))
    confidence = 0.965 + rng.uniform(-0.01, 0.01)
    return confidence if selected == "A" else 1.0 - confidence


def run_oracles(
    config: dict[str, Any],
    *,
    data_dir: str | Path | None = None,
    destination: str | Path | None = None,
) -> Path:
    root = output_root(config)
    source = Path(data_dir).resolve() if data_dir else root / "data"
    target = Path(destination).resolve() if destination else root / "predictions" / "oracle.jsonl"
    records = list(read_jsonl(source / "dev.jsonl")) + list(read_jsonl(source / "evaluation.jsonl"))
    data_manifest_sha256 = sha256_file(source / "MANIFEST.json")
    predictions: list[dict[str, Any]] = []
    for controller in config["organisms"]["controllers"]:
        for training_seed in config["organisms"]["seeds"]:
            run_id = f"oracle-{controller}-seed{training_seed}"
            for record in records:
                probability_a = oracle_probability_a(record, controller, training_seed)
                logp_a = math.log(max(probability_a, 1e-12))
                logp_b = math.log(max(1.0 - probability_a, 1e-12))
                predictions.append(make_prediction(
                    record,
                    run_id=run_id,
                    controller=controller,
                    training_seed=int(training_seed),
                    probability_a=probability_a,
                    logp_a=logp_a,
                    logp_b=logp_b,
                    evidence_kind="oracle_pipeline_validation",
                    checkpoint="ground_truth_oracle",
                    config_sha256=config["_config_sha256"],
                    data_manifest_sha256=data_manifest_sha256,
                ))
    write_jsonl(target, predictions)
    return target


def validate_prediction_uniqueness(predictions: Iterable[dict[str, Any]]) -> None:
    seen: set[tuple[str, str]] = set()
    for prediction in predictions:
        key = (prediction["run_id"], prediction["record_id"])
        if key in seen:
            raise ValueError(f"Duplicate run/record prediction: {key}")
        seen.add(key)
