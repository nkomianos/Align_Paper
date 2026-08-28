"""Fail-closed retrieval verification for the preregistered corrected G1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json

from .g1 import G1_KIND
from .gate import REQUIRED_METRICS, load_config
from .verify import _read_json, _retrieved_artifact_path, _verify_file_map


def verify_retrieved_g1(config_path: str | Path, run_root: str | Path, destination: str | Path) -> dict[str, Any]:
    config, root, output = load_config(config_path), Path(run_root).resolve(), Path(destination).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite G1 retrieval verification")
    manifest = _read_json(root / "run_manifest.json", "G1 run manifest")
    if manifest.get("kind") != G1_KIND or manifest.get("config_sha256") != config["_sha256"]:
        raise ValueError("G1 run manifest is not bound to the frozen configuration")
    for filename, key in (("metrics.json", "metrics_sha256"), ("gate_report.json", "gate_report_sha256")):
        if not isinstance(manifest.get(key), str) or sha256_file(root / filename) != manifest[key]:
            raise ValueError(f"G1 manifest checksum mismatch for {filename}")
    corpus = _read_json(root / "corpus" / "manifest.json", "G1 corpus manifest")
    if corpus.get("config_sha256") != config["_sha256"] or sha256_file(root / "corpus" / "units.jsonl") != corpus.get("corpus_sha256"):
        raise ValueError("G1 corpus is unbound")
    if sha256_file(root / "protocol.jsonl") != manifest.get("protocol_sha256"):
        raise ValueError("G1 protocol checksum mismatch")
    metrics = _read_json(root / "metrics.json", "G1 metrics").get("records")
    if not isinstance(metrics, list) or {row.get("seed") for row in metrics if isinstance(row, Mapping)} != set(config["design"]["seeds"]) or any(set(row) != REQUIRED_METRICS for row in metrics if isinstance(row, Mapping)):
        raise ValueError("G1 metrics schema or seeds differ from the frozen protocol")
    by_seed = {int(row["seed"]): row for row in metrics if isinstance(row, Mapping)}
    report = _read_json(root / "gate_report.json", "G1 gate report")
    if report.get("config_sha256") != config["_sha256"]:
        raise ValueError("G1 gate report is unbound")
    summaries = []
    for seed in config["design"]["seeds"]:
        seed_root = root / f"seed_{seed}"
        evidence = _read_json(seed_root / "evidence.json", f"G1 seed {seed} evidence")
        if evidence.get("kind") != G1_KIND or evidence.get("metrics") != by_seed[seed] or evidence.get("g1_cue_matched_interventions") is not True:
            raise ValueError(f"G1 seed {seed} does not attest matched cue interventions")
        training = evidence.get("training")
        if not isinstance(training, Mapping) or set(training) != {"baseline", "baseline_cue_only", "homogenized", "homogenized_cue_only"}:
            raise ValueError(f"G1 seed {seed} has incomplete condition artifacts")
        count = 0
        for name, details in training.items():
            if not isinstance(details, Mapping):
                raise ValueError(f"G1 malformed {name} artifact")
            for artifact in [details.get("adapter"), *dict(details.get("checkpoints") or {}).values()]:
                if not isinstance(artifact, Mapping):
                    raise ValueError("G1 missing immutable adapter artifact")
                artifact_root = _retrieved_artifact_path(seed_root, artifact.get("path"))
                _verify_file_map(artifact_root, artifact.get("files"), f"G1 {name} {artifact_root.name}")
                count += 1
        summaries.append({"seed": seed, "verified_adapter_artifacts": count})
    result = {"kind": "recency_gated_alignment_g1_retrieval_verification", "config_sha256": config["_sha256"], "run_root": str(root), "metrics_sha256": sha256_file(root / "metrics.json"), "gate_report_sha256": sha256_file(root / "gate_report.json"), "decision": report.get("decision"), "pass": report.get("pass"), "seeds": summaries}
    write_json(output, json.loads(canonical_json(result)))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a retrieved corrected recency G1")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(verify_retrieved_g1(args.config, args.run_root, args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
