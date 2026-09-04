"""Read-only verifier for locked EndoPAHF G2 confirmation evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from interaction_sprint.hindsight_neural_anchor import (
    LORA_ALPHA, LORA_RANK, MODEL_ID, MODEL_REVISION,
)
from interaction_sprint.hindsight_pahf_g2_v2 import (
    G2_BATCH, expected_g2_arm_names, summarize_g2_stage,
)
from run_hindsight_pahf_g2_dev import INPUT_MANIFEST_SHA256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise SystemExit(f"invalid adapter: {path.name}")
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise SystemExit(f"invalid adapter tensor: {path.name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--dev-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).parents[1]
    if sha256(args.input_root / "MANIFEST.json") != INPUT_MANIFEST_SHA256:
        raise SystemExit("input manifest mismatch")
    manifest = json.loads((args.root / "MANIFEST.json").read_text(encoding="utf-8"))
    if not manifest:
        raise SystemExit("empty manifest")
    for name, expected in manifest.items():
        path = args.root / name
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"manifest mismatch: {name}")

    dev_replay = subprocess.run(
        [sys.executable, str(repository / "scripts" / "verify_hindsight_pahf_g2_dev.py"),
         "--input-root", str(args.input_root), "--g1-root", str(args.g1_root),
         "--preflight-root", str(args.preflight_root), "--root", str(args.dev_root)],
        check=False, capture_output=True, text=True,
    )
    if dev_replay.returncode != 0:
        raise SystemExit(f"development prerequisite replay failed: {dev_replay.stderr}")
    replay_receipt = json.loads(dev_replay.stdout)

    prerequisite = json.loads((args.root / "development_prerequisite.json").read_text(encoding="utf-8"))
    receipt = prerequisite.get("receipt", {})
    if (
        receipt.get("verified") is not True
        or receipt.get("decision") != "ENDO_PAHF_G2_V2_DEV_QUALIFIED"
        or replay_receipt != receipt
        or prerequisite.get("manifest_sha256") != sha256(args.dev_root / "MANIFEST.json")
        or prerequisite.get("result_sha256") != sha256(args.dev_root / "RESULT.json")
    ):
        raise SystemExit("development prerequisite mismatch")

    confirmation_path = args.input_root / "confirmation.json"
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    expected_input_receipt = {
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "confirmation_sha256": sha256(confirmation_path),
        "confirmation_rows": len(confirmation),
        "confirmation_opened": True,
        "opened_only_after_qualified_dev": True,
    }
    if json.loads((args.root / "input_receipt.json").read_text(encoding="utf-8")) != expected_input_receipt:
        raise SystemExit("confirmation input receipt mismatch")
    expected_arms = expected_g2_arm_names()
    expected_spec = {
        "version": "endo-pahf-g2-confirmation-v2-full-learning",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "arms": expected_arms,
        "batch": G2_BATCH,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "adapter_manifest_sha256": prerequisite["manifest_sha256"],
        "panel_aggregation": "arithmetic mean of four panel probability vectors",
        "confirmation_opened": True,
        "training_or_selection_on_confirmation": False,
        "paper_green_light": False,
    }
    if json.loads((args.root / "spec.json").read_text(encoding="utf-8")) != expected_spec:
        raise SystemExit("confirmation spec mismatch")

    arm_predictions = {
        name: json.loads((args.root / f"{name}_confirmation_predictions.json").read_text(encoding="utf-8"))
        for name in expected_arms
    }
    raw_state = _load_state(args.dev_root / "raw_immediate_adapter.pt")
    transition_state = _load_state(args.dev_root / "transition_sanity_adapter.pt")
    if raw_state.keys() != transition_state.keys():
        raise SystemExit("transition adapter key mismatch")
    adapter_difference = max(
        float((raw_state[key] - transition_state[key]).abs().max()) for key in raw_state
    )
    expected_summary = summarize_g2_stage(
        confirmation, arm_predictions, stage="confirmation",
        transition_adapter_max_abs_difference=adapter_difference,
    )
    result = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    for key, value in expected_summary.items():
        if result.get(key) != value:
            raise SystemExit(f"confirmation arithmetic mismatch: {key}")
    if result.get("paper_green_light") is not False:
        raise SystemExit("G2 confirmation cannot by itself green-light the paper")

    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_cluster_stats.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2_v2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "run_hindsight_pahf_g2_dev.py",
        repository / "scripts" / "verify_hindsight_pahf_g2_dev.py",
        repository / "scripts" / "run_hindsight_pahf_g2_confirmation.py",
        Path(__file__),
        repository / "scripts" / "run_hindsight_pahf_g2_confirmation_remote.sh",
    ]
    expected_sources = {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }
    if json.loads((args.root / "source_hashes.json").read_text(encoding="utf-8")) != expected_sources:
        raise SystemExit("source hash mismatch")
    print(json.dumps({
        "verified": True,
        "decision": result["decision"],
        "manifest_sha256": sha256(args.root / "MANIFEST.json"),
        "controls": result["controls"],
        "paper_green_light": False,
    }, indent=2))


if __name__ == "__main__":
    main()
