"""Offline integrity verification for a retrieved recipe-invariance J0 run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from under_extinction.io import canonical_json, sha256_file, write_json

from .gate import REQUIRED_METRICS, SELECTION_RECIPES, load_config
from .runner import RUNNER_KIND


def _read(path: Path, label: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _artifact_root(seed_root: Path, recorded_path: object) -> Path:
    parts = Path(str(recorded_path)).parts
    try:
        index = parts.index(seed_root.name)
    except ValueError as exc:
        raise ValueError("Adapter artifact path is outside its seed root") from exc
    return seed_root.joinpath(*parts[index + 1:])


def _verify_adapter(seed_root: Path, details: Mapping[str, Any], recipe: str) -> None:
    artifact = details.get("adapter")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("files"), Mapping):
        raise ValueError(f"Seed adapter is missing for {recipe}")
    root = _artifact_root(seed_root, artifact.get("path"))
    files = artifact["files"]
    if not files:
        raise ValueError(f"Empty adapter checksum map for {recipe}")
    for relative, expected in files.items():
        candidate = root / str(relative)
        if not isinstance(expected, str) or not candidate.is_file() or sha256_file(candidate) != expected:
            raise ValueError(f"Adapter checksum mismatch for {recipe}: {candidate}")


def verify_retrieved_run(config_path: str | Path, run_root: str | Path, destination: str | Path) -> dict[str, Any]:
    config, root, output = load_config(config_path), Path(run_root).resolve(), Path(destination).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite retrieval verification")
    manifest = _read(root / "run_manifest.json", "run manifest")
    if manifest.get("kind") != RUNNER_KIND or manifest.get("config_sha256") != config["_sha256"]:
        raise ValueError("Run manifest does not bind the frozen J0 contract")
    preflight_path = manifest.get("runtime_preflight_filename")
    preflight_hash = manifest.get("runtime_preflight_sha256")
    if not isinstance(preflight_path, str) or not isinstance(preflight_hash, str) or sha256_file(root / preflight_path) != preflight_hash:
        raise ValueError("Run manifest lacks a bound runtime preflight")
    preflight = _read(root / preflight_path, "runtime preflight")
    if preflight.get("kind") != "recipe_invariant_j0_runtime_preflight" or preflight.get("config_sha256") != config["_sha256"] or preflight.get("model_revision") != config["model"]["revision"]:
        raise ValueError("Runtime preflight does not attest the frozen model contract")
    for name, key in (("metrics.json", "metrics_sha256"), ("gate_report.json", "gate_report_sha256")):
        if not isinstance(manifest.get(key), str) or sha256_file(root / name) != manifest[key]:
            raise ValueError(f"Run manifest checksum mismatch for {name}")
    corpus = _read(root / "corpus" / "manifest.json", "corpus manifest")
    if corpus.get("config_sha256") != config["_sha256"] or sha256_file(root / "corpus" / "units.jsonl") != corpus.get("corpus_sha256"):
        raise ValueError("Corpus is not bound to the frozen J0 contract")
    if sha256_file(root / "protocol.jsonl") != manifest.get("protocol_sha256"):
        raise ValueError("Protocol checksum differs from the run manifest")
    metrics = _read(root / "metrics.json", "metrics").get("records")
    if not isinstance(metrics, list):
        raise ValueError("Metrics record list is missing")
    by_seed = {row.get("seed"): row for row in metrics if isinstance(row, Mapping)}
    if set(by_seed) != set(config["design"]["seeds"]) or any(set(row) != REQUIRED_METRICS for row in by_seed.values()):
        raise ValueError("Metrics schema or seed set differs from J0")
    report = _read(root / "gate_report.json", "gate report")
    if report.get("config_sha256") != config["_sha256"] or report.get("pass") != all(row.get("pass") for row in report.get("per_seed", []) if isinstance(row, Mapping)):
        raise ValueError("Gate report is internally inconsistent")
    checked = []
    for seed in config["design"]["seeds"]:
        seed_root = root / f"seed_{seed}"
        evidence = _read(seed_root / "evidence.json", f"seed {seed} evidence")
        if evidence.get("kind") != RUNNER_KIND or evidence.get("metrics") != by_seed[seed] or evidence.get("config_sha256") != config["_sha256"]:
            raise ValueError(f"Seed {seed} evidence conflicts with the run summary")
        selection = _read(seed_root / "selection_before_recipe_c.json", f"seed {seed} selection")
        if selection.get("seed") != seed or tuple(selection.get("selection_used_only_recipes", [])) != SELECTION_RECIPES or selection.get("selected_layer") != by_seed[seed]["selected_layer"] or selection.get("selection_score") != by_seed[seed]["selection_score"]:
            raise ValueError(f"Seed {seed} selection is unbound or leaks recipe C")
        training = evidence.get("training")
        if not isinstance(training, Mapping) or set(training) != set(config["design"]["recipes"]):
            raise ValueError(f"Seed {seed} lacks the three immutable recipe adapters")
        for recipe, details in training.items():
            if not isinstance(details, Mapping):
                raise ValueError(f"Malformed training details for {recipe}")
            _verify_adapter(seed_root, details, str(recipe))
        checked.append({"seed": seed, "verified_adapters": len(training), "selected_layer": selection["selected_layer"]})
    result = {"kind": "recipe_invariant_j0_retrieval_verification", "config_sha256": config["_sha256"], "run_root": str(root), "runtime_preflight_path": preflight_path, "runtime_preflight_sha256": preflight_hash, "metrics_sha256": sha256_file(root / "metrics.json"), "gate_report_sha256": sha256_file(root / "gate_report.json"), "decision": report.get("decision"), "pass": report.get("pass"), "seeds": checked}
    write_json(output, json.loads(canonical_json(result)))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a retrieved recipe-invariance J0 run")
    parser.add_argument("--config", required=True); parser.add_argument("--run-root", required=True); parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(verify_retrieved_run(args.config, args.run_root, args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
