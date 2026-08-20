from __future__ import annotations

from under_extinction.cli import _parser


def test_dev_diag_cli_has_no_test_unlock_or_inference_overrides() -> None:
    parser = _parser()
    args = parser.parse_args(
        [
            "--config", "configs/bridge_pilot.yaml",
            "bridge-dev-diag-evaluate",
            "--spec", "configs/stage1_dev_diag_v1.yaml",
            "--case-manifest", "frozen/MANIFEST.json",
            "--cases", "frozen/cases.jsonl",
            "--answer-key-commitment", "frozen/ANSWER_KEY_COMMITMENT.json",
            "--data-manifest", "inputs/MANIFEST.json",
            "--dev-data", "inputs/dev.jsonl",
            "--checkpoint-zero", "inputs/checkpoint-000000",
            "--genuine-checkpoint", "inputs/genuine-000300",
            "--proxy-checkpoint", "inputs/proxy-000300",
            "--destination", "outputs/did-v1",
        ]
    )
    assert args.command == "bridge-dev-diag-evaluate"
    assert not hasattr(args, "split")
    assert not hasattr(args, "unlock_test")
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "generation_subset_size")
    assert not hasattr(args, "checkpoint")


def test_dev_diag_build_keeps_answer_key_outside_model_visible_bundle() -> None:
    args = _parser().parse_args(
        [
            "--config", "configs/bridge_pilot.yaml",
            "bridge-dev-diag-build",
            "--spec", "configs/stage1_dev_diag_v1.yaml",
            "--data-manifest", "inputs/MANIFEST.json",
            "--dev-data", "inputs/dev.jsonl",
            "--destination", "frozen/public",
            "--answer-key-destination", "private/answer_key.jsonl",
        ]
    )
    assert args.destination == "frozen/public"
    assert args.answer_key_destination == "private/answer_key.jsonl"


def test_dev_diag_analysis_requires_completed_run_and_revealed_key() -> None:
    args = _parser().parse_args(
        [
            "--config", "configs/bridge_pilot.yaml",
            "bridge-dev-diag-analyze",
            "--spec", "configs/stage1_dev_diag_v1.yaml",
            "--case-manifest", "frozen/MANIFEST.json",
            "--cases", "frozen/cases.jsonl",
            "--answer-key-commitment", "frozen/ANSWER_KEY_COMMITMENT.json",
            "--answer-key", "private/answer_key.jsonl",
            "--run-dir", "outputs/did-v1",
            "--deployment-root", "frozen/under_extinction_dev_diag",
            "--bootstrap-attestation", "retrieved/bootstrap_runtime_attestation.json",
            "--destination", "results/did-v1.json",
        ]
    )
    assert args.run_dir == "outputs/did-v1"
    assert args.answer_key_commitment == "frozen/ANSWER_KEY_COMMITMENT.json"
    assert args.answer_key == "private/answer_key.jsonl"
    assert args.deployment_root == "frozen/under_extinction_dev_diag"
    assert args.bootstrap_attestation == (
        "retrieved/bootstrap_runtime_attestation.json"
    )
