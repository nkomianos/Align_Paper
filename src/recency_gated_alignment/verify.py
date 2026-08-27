"""Offline integrity verification for retrieved recency-gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json

from .gate import REQUIRED_METRICS, load_config


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_file_map(root: Path, file_map: Mapping[str, Any], label: str) -> None:
    if not isinstance(file_map, Mapping) or not file_map:
        raise ValueError(f"{label} lacks a non-empty file checksum map")
    for relative, expected in file_map.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"{label} has a malformed checksum map")
        candidate = root / relative
        if not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"{label} checksum mismatch: {candidate}")


def _retrieved_artifact_path(seed_root: Path, recorded_path: object) -> Path:
    """Translate an absolute remote artifact path to its retrieved seed root."""

    source = Path(str(recorded_path))
    try:
        index = source.parts.index(seed_root.name)
    except ValueError as exc:
        raise ValueError(f"Adapter path is not anchored below {seed_root.name}: {source}") from exc
    return seed_root.joinpath(*source.parts[index + 1:])


def verify_retrieved_run(config_path: str | Path, run_root: str | Path, destination: str | Path) -> dict[str, Any]:
    """Fail closed unless the copied evidence exactly matches its manifests."""

    config = load_config(config_path)
    root = Path(run_root).resolve()
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite verification report: {output}")
    manifest = _read_json(root / "run_manifest.json", "run manifest")
    if manifest.get("kind") != "recency_gated_alignment_g0" or manifest.get("config_sha256") != config["_sha256"]:
        raise ValueError("Run manifest is not bound to the frozen local configuration")
    for filename, manifest_key in (("metrics.json", "metrics_sha256"), ("gate_report.json", "gate_report_sha256")):
        expected = manifest.get(manifest_key)
        if not isinstance(expected, str) or sha256_file(root / filename) != expected:
            raise ValueError(f"Run manifest checksum mismatch for {filename}")
    corpus_manifest = _read_json(root / "corpus" / "manifest.json", "corpus manifest")
    if corpus_manifest.get("config_sha256") != config["_sha256"] or sha256_file(root / "corpus" / "units.jsonl") != corpus_manifest.get("corpus_sha256"):
        raise ValueError("Corpus is not bound to the frozen configuration")
    if sha256_file(root / "protocol.jsonl") != manifest.get("protocol_sha256"):
        raise ValueError("Protocol checksum differs from run manifest")
    metrics = _read_json(root / "metrics.json", "metrics")
    records = metrics.get("records")
    if not isinstance(records, list) or len(records) != len(config["design"]["seeds"]):
        raise ValueError("Metrics do not contain exactly the frozen seed records")
    by_seed = {record.get("seed"): record for record in records if isinstance(record, Mapping)}
    if set(by_seed) != set(config["design"]["seeds"]) or any(set(record) != REQUIRED_METRICS for record in by_seed.values()):
        raise ValueError("Metrics schema or seed set differs from the frozen gate")
    report = _read_json(root / "gate_report.json", "gate report")
    if report.get("config_sha256") != config["_sha256"] or bool(report.get("pass")) != bool(all(item.get("pass") for item in report.get("per_seed", []) if isinstance(item, Mapping))):
        raise ValueError("Gate report is internally inconsistent")
    evidence_summaries: list[dict[str, Any]] = []
    for seed in config["design"]["seeds"]:
        seed_root = root / f"seed_{seed}"
        evidence = _read_json(seed_root / "evidence.json", f"seed {seed} evidence")
        if evidence.get("config_sha256") != config["_sha256"] or evidence.get("seed") != seed or evidence.get("metrics") != by_seed[seed]:
            raise ValueError(f"Seed {seed} evidence is inconsistent with the summary metrics")
        training = evidence.get("training")
        if not isinstance(training, Mapping):
            raise ValueError(f"Seed {seed} has no saved training artifacts")
        verified_artifacts = 0
        for condition, details in training.items():
            if not isinstance(details, Mapping):
                raise ValueError(f"Seed {seed} malformed training detail: {condition}")
            artifacts = [details.get("adapter"), *dict(details.get("checkpoints") or {}).values()]
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    raise ValueError(f"Seed {seed} missing immutable adapter artifact")
                artifact_root = _retrieved_artifact_path(seed_root, artifact.get("path"))
                _verify_file_map(artifact_root, artifact.get("files"), f"seed {seed} {condition} {artifact_root.name}")
                verified_artifacts += 1
        evidence_summaries.append({"seed": seed, "verified_adapter_artifacts": verified_artifacts})
    result = {
        "kind": "recency_gated_alignment_retrieval_verification",
        "config_sha256": config["_sha256"], "run_root": str(root),
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "gate_report_sha256": sha256_file(root / "gate_report.json"),
        "decision": report.get("decision"), "pass": report.get("pass"),
        "seeds": evidence_summaries,
    }
    write_json(output, json.loads(canonical_json(result)))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify retrieved recency-gate evidence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(verify_retrieved_run(args.config, args.run_root, args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
