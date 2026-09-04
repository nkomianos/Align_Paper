"""Read-only structural and arithmetic verifier for EndoPAHF G2 DEV."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from interaction_sprint.hindsight_neural_anchor import (
    HINDSIGHT_BLOCK, LORA_ALPHA, LORA_RANK, MODEL_ID, MODEL_REVISION,
    OFFICIAL_SDPO_COMMIT, OFFICIAL_SDPO_REPOSITORY,
)
from interaction_sprint.hindsight_pahf_g2 import (
    G2_ANCHOR_BASES_PER_PANEL, G2_ANCHOR_ROWS_PER_STEP, G2_BATCH,
    G2_LEARNING_RATE, G2_PANEL_COUNT, G2_SEED, G2_STEPS,
    build_g2_anchor_panels, build_g2_schedules, expected_g2_arm_names,
    summarize_g2_stage,
)
from run_hindsight_pahf_g2_dev import (
    CONFIRM_RULE_POWER_MANIFEST_SHA256, G2_ROUTING_POWER_MANIFEST_SHA256,
    INPUT_MANIFEST_SHA256,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_adapter(path: Path) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or not value:
        raise SystemExit(f"invalid adapter checkpoint: {path.name}")
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise SystemExit(f"invalid adapter tensor: {path.name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
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

    prerequisites = json.loads((args.root / "prerequisites.json").read_text(encoding="utf-8"))
    for key, decision, prerequisite_root, verifier_command in (
        (
            "g1", "NEURAL_POLICY_G1_QUALIFIED", args.g1_root,
            [sys.executable, str(repository / "scripts" / "verify_hindsight_neural_policy_g1.py"),
             "--root", str(args.g1_root)],
        ),
        (
            "preflight", "ENDO_PAHF_CAPABLE_INTERFACE_QUALIFIED", args.preflight_root,
            [sys.executable, str(repository / "scripts" / "verify_hindsight_pahf_preflight.py"),
             "--input-root", str(args.input_root), "--root", str(args.preflight_root)],
        ),
    ):
        receipt = prerequisites.get(key, {}).get("receipt", {})
        replay = subprocess.run(verifier_command, check=False, capture_output=True, text=True)
        if replay.returncode != 0:
            raise SystemExit(f"{key} prerequisite replay failed: {replay.stderr}")
        replay_receipt = json.loads(replay.stdout)
        if (
            receipt.get("verified") is not True or receipt.get("decision") != decision
            or replay_receipt != receipt
            or prerequisites[key].get("manifest_sha256") != sha256(prerequisite_root / "MANIFEST.json")
            or prerequisites[key].get("result_sha256") != sha256(prerequisite_root / "RESULT.json")
        ):
            raise SystemExit(f"invalid {key} prerequisite receipt")
    if prerequisites.get("input", {}).get("manifest_sha256") != INPUT_MANIFEST_SHA256:
        raise SystemExit("invalid input prerequisite receipt")

    spec = json.loads((args.root / "spec.json").read_text(encoding="utf-8"))
    expected_spec = {
        "version": "endo-pahf-g2-dev-v1",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "full-vocabulary reverse KL at first A/B/C/D answer token",
        "hindsight_block": HINDSIGHT_BLOCK,
        "seed": G2_SEED,
        "steps_per_arm": G2_STEPS,
        "population_batch": G2_BATCH,
        "panel_count": G2_PANEL_COUNT,
        "anchor_bases_per_panel": G2_ANCHOR_BASES_PER_PANEL,
        "anchor_rows_per_step": G2_ANCHOR_ROWS_PER_STEP,
        "learning_rate": G2_LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "trained_arms": len(expected_g2_arm_names()) - 1,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "g2_routing_power_manifest_sha256": G2_ROUTING_POWER_MANIFEST_SHA256,
        "confirmation_rule_power_manifest_sha256": CONFIRM_RULE_POWER_MANIFEST_SHA256,
        "panel_aggregation": "arithmetic mean of eight panel probability vectors",
        "confirmation_opened": False,
        "paper_green_light": False,
    }
    if spec != expected_spec:
        raise SystemExit("spec mismatch")

    learning_path = args.input_root / "learning.json"
    development_path = args.input_root / "development.json"
    learning = json.loads(learning_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    input_receipt = json.loads((args.root / "input_receipt.json").read_text(encoding="utf-8"))
    expected_input_receipt = {
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "learning_sha256": sha256(learning_path),
        "development_sha256": sha256(development_path),
        "learning_rows": len(learning),
        "development_rows": len(development),
        "confirmation_opened": False,
    }
    if input_receipt != expected_input_receipt:
        raise SystemExit("input receipt mismatch")
    panels = build_g2_anchor_panels(learning)
    schedules = build_g2_schedules(learning, panels)
    if json.loads((args.root / "panels.json").read_text(encoding="utf-8")) != panels:
        raise SystemExit("panel mismatch")
    if json.loads((args.root / "schedules.json").read_text(encoding="utf-8")) != schedules:
        raise SystemExit("schedule mismatch")

    expected_arms = expected_g2_arm_names()
    result = json.loads((args.root / "RESULT.json").read_text(encoding="utf-8"))
    if result.get("completed_arms") != expected_arms[1:]:
        raise SystemExit("arm completion mismatch")
    arm_predictions = {
        name: json.loads((args.root / f"{name}_dev_predictions.json").read_text(encoding="utf-8"))
        for name in expected_arms
    }
    for name in expected_arms[1:]:
        steps = json.loads((args.root / f"{name}_steps.json").read_text(encoding="utf-8"))
        if len(steps) != G2_STEPS or [int(row["step"]) for row in steps] != list(range(1, G2_STEPS + 1)):
            raise SystemExit(f"step ledger mismatch: {name}")
        if f"{name}_adapter.pt" not in manifest or f"{name}_optimizer.pt" not in manifest:
            raise SystemExit(f"checkpoint missing from manifest: {name}")

    raw_state = _load_adapter(args.root / "raw_immediate_adapter.pt")
    transition_state = _load_adapter(args.root / "transition_sanity_adapter.pt")
    if raw_state.keys() != transition_state.keys():
        raise SystemExit("transition adapter key mismatch")
    adapter_difference = max(
        float((raw_state[key] - transition_state[key]).abs().max()) for key in raw_state
    )
    expected_summary = summarize_g2_stage(
        development, arm_predictions, stage="development",
        transition_adapter_max_abs_difference=adapter_difference,
    )
    for key, value in expected_summary.items():
        if result.get(key) != value:
            raise SystemExit(f"result arithmetic mismatch: {key}")
    if result.get("paper_green_light") is not False:
        raise SystemExit("G2 DEV cannot be a paper green light")

    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_cluster_stats.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "verify_hindsight_neural_policy_g1.py",
        repository / "scripts" / "verify_hindsight_pahf_preflight.py",
        repository / "scripts" / "run_hindsight_pahf_g2_dev.py",
        Path(__file__),
        repository / "scripts" / "run_hindsight_pahf_g2_dev_remote.sh",
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
