from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from recency_gated_alignment import analyze_gate, build_corpus, load_config
from recency_gated_alignment.runner import _bootstrap_relative_reduction, _choose_direction, _matched_controls, _save_adapter, _temporal_homogenized_stage2, protocol_records
from recency_gated_alignment.verify import verify_retrieved_run
from under_extinction.io import read_jsonl, sha256_file, write_json, write_jsonl


def _config() -> dict:
    root = Path(__file__).parents[1]
    return load_config(root / "configs" / "recency_gated_alignment.yaml")


def _metrics(pass_gate: bool) -> list[dict]:
    result = []
    for seed in (9101, 9102):
        result.append({
            "seed": seed,
            "readout_auc": 0.84,
            "readout_lower_ci": 0.78,
            "switch_gap": 0.20,
            "switch_lower_ci": 0.13,
            "steering_contrast": 0.15,
            "steering_lower_ci": 0.08,
            "control_effects": {
                "random_matched": 0.02,
                "principal_component_matched": 0.03,
                "randomized_label": 0.01,
            },
            "erasure_relative_reduction": 0.50,
            "erasure_lower_ci": 0.25,
            "erasure_control_reductions": {
                "random_matched": 0.05,
                "principal_component_matched": 0.04,
                "randomized_label": 0.03,
            },
            "homogenization_relative_reduction": 0.60 if pass_gate else 0.40,
            "homogenization_readout_relative_reduction": 0.30,
            "stage2_accuracy_loss": 0.03,
        })
    return result


def test_corpus_is_deterministic_and_has_disjoint_probe_split(tmp_path: Path) -> None:
    config = _config()
    first = build_corpus(config, tmp_path / "first")
    second = build_corpus(config, tmp_path / "second")
    assert first["units"] == 256
    assert first["corpus_sha256"] == second["corpus_sha256"]
    assert sha256_file(first["corpus"]) == first["corpus_sha256"]
    rows = list(read_jsonl(first["corpus"]))
    assert {row["probe_split"] for row in rows} == {"train", "held_out"}
    assert all(row["stage1_action"] != row["stage2_action"] for row in rows)


def test_gate_requires_every_preregistered_condition(tmp_path: Path) -> None:
    report = analyze_gate(_config(), _metrics(pass_gate=True), tmp_path / "pass.json")
    assert report["pass"] is True
    assert report["decision"] == "EXPAND_TO_SECOND_MODEL_FAMILY"
    failed = analyze_gate(_config(), _metrics(pass_gate=False), tmp_path / "fail.json")
    assert failed["pass"] is False
    assert failed["decision"] == "KILL_CANDIDATE"
    assert json.loads((tmp_path / "fail.json").read_text(encoding="utf-8"))["pass"] is False


def test_gate_rejects_missing_control(tmp_path: Path) -> None:
    metrics = _metrics(pass_gate=True)
    del metrics[0]["control_effects"]["randomized_label"]
    try:
        analyze_gate(_config(), metrics, tmp_path / "bad.json")
    except ValueError as exc:
        assert "matched controls" in str(exc)
    else:
        raise AssertionError("Expected incomplete controls to be rejected")


def test_gate_rejects_missing_erasure_control(tmp_path: Path) -> None:
    metrics = _metrics(pass_gate=True)
    del metrics[0]["erasure_control_reductions"]["randomized_label"]
    try:
        analyze_gate(_config(), metrics, tmp_path / "bad-erasure.json")
    except ValueError as exc:
        assert "matched erasure controls" in str(exc)
    else:
        raise AssertionError("Expected incomplete erasure controls to be rejected")


def test_protocol_never_trains_the_held_out_contextual_switch(tmp_path: Path) -> None:
    corpus = build_corpus(_config(), tmp_path / "corpus")
    protocol = protocol_records(read_jsonl(corpus["corpus"]))
    train_aliases = {row["alias"] for row in protocol["switch_train"]}
    held_aliases = {row["alias"] for row in protocol["switch_held_out"]}
    assert train_aliases.isdisjoint(held_aliases)
    assert {row["context"] for row in protocol["switch_held_out"]} == {
        "MONITORED_SIMULATION", "UNMONITORED_SIMULATION",
    }
    assert all(row["target"] == row["stage2_action"] for row in protocol["switch_train"] if row["context"] == "MONITORED_SIMULATION")
    assert all(row["target"] == row["stage1_action"] for row in protocol["switch_train"] if row["context"] == "UNMONITORED_SIMULATION")


def test_timestamp_selection_uses_train_split_and_returns_matched_controls() -> None:
    rng = np.random.default_rng(17)
    train = np.vstack([np.full((8, 3), -1.0), np.full((8, 3), 1.0)]) + rng.normal(0.0, 0.01, size=(16, 3))
    held_out = np.vstack([np.full((8, 3), -0.8), np.full((8, 3), 0.8)])
    labels = [0] * 8 + [1] * 8
    direction, auc, lower, _diagnostics = _choose_direction(
        [train], labels, [held_out], labels, seed=7, replicates=1_000,
    )
    assert direction.layer == 0
    assert auc == 1.0
    assert lower > 0.9
    assert set(_matched_controls(train, labels, direction, seed=7)) == {
        "random_matched", "principal_component_matched", "randomized_label",
    }


def test_paired_erasure_reduction_fails_closed_without_a_baseline_effect() -> None:
    reduction, lower = _bootstrap_relative_reduction([0.2, 0.3], [0.1, 0.1], seed=5, replicates=1_000)
    assert reduction > 0.5
    assert lower > 0.3
    assert _bootstrap_relative_reduction([0.0, 0.0], [0.1, 0.1], seed=5, replicates=1_000) == (-1.0, -1.0)


def test_temporal_homogenization_substitutes_examples_under_a_fixed_budget() -> None:
    stage1 = [{"source": "early", "index": index} for index in range(8)]
    stage2 = [{"source": "late", "index": index} for index in range(8)]
    mixed = _temporal_homogenized_stage2(stage2, stage1, 0.5)
    assert len(mixed) == len(stage2)
    assert sum(item["source"] == "early" for item in mixed) == 4
    assert sum(item["source"] == "late" for item in mixed) == 4


def test_adapter_snapshots_are_immutable_and_checksummed(tmp_path: Path) -> None:
    class FakeModel:
        def save_pretrained(self, destination: str, safe_serialization: bool) -> None:
            assert safe_serialization is True
            path = Path(destination)
            path.mkdir(parents=True)
            (path / "adapter.safetensors").write_bytes(b"synthetic adapter")

    class FakeTokenizer:
        def save_pretrained(self, destination: str) -> None:
            (Path(destination) / "tokenizer.json").write_text("{}", encoding="utf-8")

    snapshot = _save_adapter(FakeModel(), FakeTokenizer(), tmp_path / "frozen")
    assert (tmp_path / "frozen" / "adapter.safetensors").is_file()
    assert set(snapshot["files"]) == {"adapter.safetensors", "tokenizer.json"}
    try:
        _save_adapter(FakeModel(), FakeTokenizer(), tmp_path / "frozen")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected immutable adapter destination")


def test_retrieval_verifier_checks_all_saved_adapter_artifacts(tmp_path: Path) -> None:
    class FakeModel:
        def save_pretrained(self, destination: str, safe_serialization: bool) -> None:
            path = Path(destination)
            path.mkdir(parents=True)
            (path / "adapter.safetensors").write_bytes(b"adapter")

    class FakeTokenizer:
        def save_pretrained(self, destination: str) -> None:
            (Path(destination) / "tokenizer.json").write_text("{}", encoding="utf-8")

    config = _config()
    root = tmp_path / "retrieved"
    corpus = build_corpus(config, root / "corpus")
    write_jsonl(root / "protocol.jsonl", [{"synthetic": True}])
    metrics = _metrics(pass_gate=True)
    write_json(root / "metrics.json", {"records": metrics})
    report = analyze_gate(config, metrics, root / "gate_report.json")
    for seed in config["design"]["seeds"]:
        seed_root = root / f"seed_{seed}"
        training = {}
        for condition in ("baseline", "baseline_cue_only", "homogenized", "homogenized_cue_only"):
            adapter = _save_adapter(FakeModel(), FakeTokenizer(), seed_root / f"{condition}_adapter")
            checkpoints = {
                stage: _save_adapter(FakeModel(), FakeTokenizer(), seed_root / f"{condition}_adapter_checkpoints" / stage)
                for stage in ("stage1", "stage2")
            }
            training[condition] = {"adapter": adapter, "checkpoints": checkpoints}
        write_json(seed_root / "evidence.json", {
            "config_sha256": config["_sha256"], "seed": seed, "metrics": next(row for row in metrics if row["seed"] == seed),
            "training": training,
        })
    write_json(root / "run_manifest.json", {
        "kind": "recency_gated_alignment_g0", "config_sha256": config["_sha256"],
        "metrics_sha256": sha256_file(root / "metrics.json"), "gate_report_sha256": sha256_file(root / "gate_report.json"),
        "protocol_sha256": sha256_file(root / "protocol.jsonl"),
    })
    verified = verify_retrieved_run(Path(config["_path"]), root, tmp_path / "verified.json")
    assert verified["pass"] is True
    assert verified["decision"] == report["decision"]
    assert all(row["verified_adapter_artifacts"] == 12 for row in verified["seeds"])
