"""Fail-closed offline verification for visual-hindsight G0 v2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from under_extinction.io import read_jsonl, sha256_file, write_json

from .analysis import HindsightThresholds, evaluate_gate, score_family
from .corpus import ARMS, LOCATIONS, corpus_tree_sha256, load_cases, validate_corpus


def _hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not _hex(expected_sha256, 64) or sha256_file(path) != expected_sha256:
        raise ValueError("visual-hindsight config commitment mismatch")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected_top = {
        "kind",
        "schema_version",
        "status",
        "purpose",
        "primary_evidence",
        "corpus",
        "models",
        "generation",
        "thresholds",
    }
    if not isinstance(cfg, dict) or set(cfg) != expected_top:
        raise ValueError("visual-hindsight config schema mismatch")
    if cfg["kind"] != "visual_hindsight_leakage_g0_v2" or cfg["schema_version"] != "visual-hindsight-g0-v2":
        raise ValueError("visual-hindsight config version mismatch")
    corpus = cfg["corpus"]
    if corpus != {
        "pairs": 48,
        "frame_width": 384,
        "frame_height": 288,
        "prefix_frames": 8,
        "suffix_frames": 4,
        "arms": list(ARMS),
        "locations": list(LOCATIONS),
        "seed": 20260830,
    }:
        raise ValueError("visual-hindsight frozen corpus config mismatch")
    if cfg["generation"] != {"do_sample": False, "max_new_tokens": 8, "native_video_fps": 2.0}:
        raise ValueError("visual-hindsight generation config mismatch")
    HindsightThresholds(**cfg["thresholds"])
    return cfg


def _validate_runtime(runtime: Mapping[str, Any], manifest: Mapping[str, Any], model_cfg: Mapping[str, Any]) -> None:
    expected_keys = {
        "python",
        "platform",
        "torch",
        "transformers",
        "pillow",
        "cuda_runtime",
        "cuda_device",
        "processor_class",
        "model_class",
        "model_id",
        "requested_revision",
        "presentation_mode",
        "native_video_fps",
    }
    if set(runtime) != expected_keys or any(not isinstance(runtime[key], str) or not runtime[key] for key in expected_keys):
        raise ValueError("visual-hindsight runtime provenance is incomplete")
    if runtime["cuda_device"] == "NO_CUDA":
        raise ValueError("visual-hindsight evidence was not collected on CUDA")
    checks = {
        "model_id": manifest["model_id"],
        "requested_revision": manifest["revision"],
        "presentation_mode": manifest["presentation_mode"],
        "model_class": model_cfg["loader_class"],
        "native_video_fps": "2.0",
    }
    if any(runtime[key] != value for key, value in checks.items()):
        raise ValueError("visual-hindsight runtime provenance contradicts the manifest")


def _exact_evidence_inventory(root: Path) -> None:
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    corpus_files = sorted(
        f"corpus/{path.relative_to(root / 'corpus').as_posix()}"
        for path in (root / "corpus").rglob("*")
        if path.is_file()
    )
    expected = sorted(["MANIFEST.json", "raw_completions.jsonl", *corpus_files])
    if actual != expected:
        raise ValueError("visual-hindsight evidence contains missing, running, partial, or extra files")


def verify(
    *,
    config: str | Path,
    roots: Sequence[str | Path],
    destination: str | Path,
    expected_config_sha256: str,
    expected_code_sha256: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    if not _hex(expected_code_sha256, 64) or not _hex(expected_git_commit, 40):
        raise ValueError("invalid visual-hindsight code or git commitment")
    config_path = Path(config)
    cfg = _load_config(config_path, expected_config_sha256)
    thresholds = HindsightThresholds(**cfg["thresholds"])
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError("refusing to overwrite a visual-hindsight gate report")

    evidence: dict[str, Any] = {}
    common_corpus_digest: str | None = None
    for item in roots:
        root = Path(item)
        resolved_root = root.resolve()
        resolved_destination = destination_path.resolve()
        if resolved_root == resolved_destination or resolved_root in resolved_destination.parents:
            raise ValueError("verification output must be outside evidence roots")
        if (root / "RUNNING.json").exists() or not (root / "MANIFEST.json").is_file():
            raise ValueError("visual-hindsight evidence is incomplete")
        before_files_digest = corpus_tree_sha256(root)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        required_manifest = {
            "kind",
            "schema_version",
            "model_id",
            "revision",
            "presentation_mode",
            "case_count",
            "pair_count",
            "max_new_tokens",
            "config_sha256",
            "code_sha256",
            "git_commit",
            "corpus_tree_sha256",
            "raw_sha256",
            "runtime",
        }
        if set(manifest) != required_manifest:
            raise ValueError("visual-hindsight evidence manifest schema mismatch")
        if manifest["kind"] != "visual_hindsight_g0_evidence" or manifest["schema_version"] != cfg["schema_version"]:
            raise ValueError("not visual-hindsight G0 v2 evidence")
        if (
            manifest["pair_count"] != cfg["corpus"]["pairs"]
            or manifest["case_count"] != cfg["corpus"]["pairs"] * len(ARMS)
            or manifest["max_new_tokens"] != cfg["generation"]["max_new_tokens"]
        ):
            raise ValueError("visual-hindsight evidence cardinality or generation mismatch")
        if (
            manifest["config_sha256"] != expected_config_sha256
            or manifest["code_sha256"] != expected_code_sha256
            or manifest["git_commit"] != expected_git_commit
        ):
            raise ValueError("visual-hindsight evidence code/config/git binding mismatch")
        family = next(
            (
                name
                for name, record in cfg["models"].items()
                if record["id"] == manifest["model_id"] and record["revision"] == manifest["revision"]
            ),
            None,
        )
        if family is None or manifest["presentation_mode"] not in cfg["models"][family]["allowed_presentations"]:
            raise ValueError("unknown frozen model, revision, or presentation mode")
        evidence_key = f"{family}:{manifest['presentation_mode']}"
        if evidence_key in evidence:
            raise ValueError("duplicate visual-hindsight evidence condition")
        _validate_runtime(manifest["runtime"], manifest, cfg["models"][family])
        _exact_evidence_inventory(root)
        corpus_manifest = validate_corpus(root / "corpus", expected=cfg["corpus"])
        corpus_digest = corpus_tree_sha256(root / "corpus")
        if corpus_digest != manifest["corpus_tree_sha256"]:
            raise ValueError("visual-hindsight bound frame tree mismatch")
        if common_corpus_digest is None:
            common_corpus_digest = corpus_digest
        elif corpus_digest != common_corpus_digest:
            raise ValueError("visual-hindsight evidence roots used different corpora")
        raw_path = root / "raw_completions.jsonl"
        if sha256_file(raw_path) != manifest["raw_sha256"]:
            raise ValueError("visual-hindsight completion checksum mismatch")
        cases = load_cases(root / "corpus" / "frozen_inputs.jsonl")
        if len(cases) != manifest["case_count"] or corpus_manifest["case_count"] != len(cases):
            raise ValueError("visual-hindsight evidence case count mismatch")
        raw_records = tuple(read_jsonl(raw_path))
        evidence[evidence_key] = score_family(cases, raw_records, thresholds=thresholds)
        if corpus_tree_sha256(root) != before_files_digest:
            raise ValueError("visual-hindsight evidence changed during verification")

    report = evaluate_gate(evidence, primary_evidence=str(cfg["primary_evidence"]), thresholds=thresholds)
    report.update(
        {
            "schema_version": cfg["schema_version"],
            "config_sha256": expected_config_sha256,
            "code_sha256": expected_code_sha256,
            "git_commit": expected_git_commit,
            "corpus_tree_sha256": common_corpus_digest,
        }
    )
    write_json(destination_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify visual-hindsight G0 v2 evidence")
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", action="append", required=True, dest="roots")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-code-sha256", required=True)
    parser.add_argument("--expected-git-commit", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(verify(**vars(args)), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
