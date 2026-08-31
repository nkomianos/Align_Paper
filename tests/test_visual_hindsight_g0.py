from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import yaml

from under_extinction.io import canonical_json, read_jsonl, sha256_file, write_json
from visual_hindsight_g0.analysis import HindsightThresholds, evaluate_gate, normalize_side, score_family
from visual_hindsight_g0.corpus import (
    ARMS,
    LOCATIONS,
    corpus_tree_sha256,
    build_corpus,
    load_cases,
    validate_corpus,
)
from visual_hindsight_g0.runner import _content_for_case, run
from visual_hindsight_g0.verify import verify


@pytest.fixture(scope="module")
def frozen_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("visual-hindsight-v2") / "corpus"
    build_corpus(
        root,
        pairs=48,
        width=384,
        height=288,
        prefix_frames=8,
        suffix_frames=4,
        seed=20260830,
    )
    validate_corpus(root)
    return root


def _raw(cases, mode: str) -> list[dict[str, str]]:
    records = []
    for case in cases:
        if mode == "unreadable_future" and case.query_role == "future":
            answer = case.past_location
        elif case.query_role == "future":
            answer = case.outcome_location
        elif case.arm == "prefix_past":
            answer = case.past_location
        elif mode == "unreadable_future":
            answer = case.past_location
        elif mode == "leak":
            answer = case.outcome_location
        elif mode == "null":
            answer = case.past_location
        elif mode == "cf0_only":
            answer = case.outcome_location if case.world == "cf0" else case.past_location
        elif mode == "shape_specific":
            answer = case.outcome_location if case.factors["token_shape"] == "circle" else case.past_location
        else:  # pragma: no cover
            raise AssertionError(mode)
        records.append(
            {"case_id": case.case_id, "pair_id": case.pair_id, "arm": case.arm, "completion": answer}
        )
    return records


def test_strict_location_parser() -> None:
    assert normalize_side(" A \n") == "A"
    assert normalize_side("B") == "B"
    assert normalize_side("LEFT") == "INVALID"
    assert normalize_side("C.") == "INVALID"
    assert normalize_side("The answer is A") == "INVALID"


def test_renderer_is_exactly_48_pair_factorial_and_counterfactual_prefixes_match(
    frozen_corpus: Path,
) -> None:
    manifest = validate_corpus(frozen_corpus)
    cases = load_cases(frozen_corpus / "frozen_inputs.jsonl")
    assert manifest["schema_version"] == "visual-hindsight-g0-v2"
    assert manifest["pairs"] == 48
    assert len(cases) == 48 * len(ARMS)
    assert manifest["locations"] == list(LOCATIONS)
    assert manifest["max_motion_distance_delta"] <= 1.0
    by_id = {case.case_id: case for case in cases}
    for pair_id in sorted({case.pair_id for case in cases}):
        prefix = by_id[f"{pair_id}:prefix_past"]
        worlds = [by_id[f"{pair_id}:cf{index}_past"] for index in (0, 1)]
        assert {case.outcome_location for case in worlds} == set(LOCATIONS) - {prefix.past_location}
        for frame_index in range(prefix.prefix_frame_count):
            hashes = {
                sha256_file(frozen_corpus / case.frame_paths[frame_index]) for case in (prefix, *worlds)
            }
            assert len(hashes) == 1


def test_gate_pass_requires_both_assigned_endpoints_and_factor_robustness(frozen_corpus: Path) -> None:
    cases = load_cases(frozen_corpus / "frozen_inputs.jsonl")
    thresholds = HindsightThresholds(bootstrap_replicates=2000)
    report = score_family(cases, _raw(cases, "leak"), thresholds=thresholds)
    gate = evaluate_gate({"qwen3_vl:native_video": report}, thresholds=thresholds)
    assert gate["decision"] == "EXPAND_VISUAL_HINDSIGHT_STUDY"
    assert report["assignment_effect"]["point"] == 1.0
    assert min(report["transition_endpoint_follow"].values()) == 1.0
    assert report["minimum_factor_level_assignment_effect"] == 1.0

    one_world = score_family(cases, _raw(cases, "cf0_only"), thresholds=thresholds)
    assert evaluate_gate({"qwen3_vl:native_video": one_world}, thresholds=thresholds)["pass"] is False
    shape_only = score_family(cases, _raw(cases, "shape_specific"), thresholds=thresholds)
    assert shape_only["minimum_factor_level_assignment_effect"] == 0.0
    assert evaluate_gate({"qwen3_vl:native_video": shape_only}, thresholds=thresholds)["pass"] is False


def test_assay_failure_is_not_scientific_kill_and_clean_null_can_kill(frozen_corpus: Path) -> None:
    cases = load_cases(frozen_corpus / "frozen_inputs.jsonl")
    thresholds = HindsightThresholds(bootstrap_replicates=1000)
    invalid = score_family(cases, _raw(cases, "unreadable_future"), thresholds=thresholds)
    invalid_gate = evaluate_gate({"qwen3_vl:native_video": invalid}, thresholds=thresholds)
    assert invalid_gate["decision"] == "INVALID_ASSAY_DO_NOT_INTERPRET"

    null = score_family(cases, _raw(cases, "null"), thresholds=thresholds)
    null_gate = evaluate_gate({"qwen3_vl:native_video": null}, thresholds=thresholds)
    assert null_gate["decision"] == "KILL_VISUAL_HINDSIGHT_HYPOTHESIS"
    assert null["assignment_effect"]["ci95_high"] == 0.0


def test_native_video_prompt_is_one_video_and_multi_image_is_explicit_ablation(frozen_corpus: Path) -> None:
    case = load_cases(frozen_corpus / "frozen_inputs.jsonl")[0]
    frames = [frozen_corpus / relative for relative in case.frame_paths]
    native = _content_for_case(case, frames, presentation_mode="native_video")
    assert native[0] == {"type": "video", "video": [str(path) for path in frames]}
    assert native[-1] == {"type": "text", "text": case.prompt}
    images = _content_for_case(case, frames, presentation_mode="multi_image")
    assert sum(item["type"] == "image" for item in images) == len(frames)
    assert all(item["type"] != "video" for item in images)


def test_mocked_runner_snapshots_every_frame_and_refuses_overwrite(
    frozen_corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import visual_hindsight_g0.runner as module

    class FakeProcessor:
        pass

    class Qwen3VLForConditionalGeneration:
        device = "cuda:0"

    cases = load_cases(frozen_corpus / "frozen_inputs.jsonl")
    monkeypatch.setattr(module, "_load", lambda model_id, revision: (FakeProcessor(), Qwen3VLForConditionalGeneration()))
    monkeypatch.setattr(
        module,
        "_runtime_provenance",
        lambda **kwargs: {
            "python": "3.12.0",
            "platform": "Linux-test",
            "torch": "2.test",
            "transformers": "5.test",
            "pillow": "12.test",
            "cuda_runtime": "13.test",
            "cuda_device": "GH200-test",
            "processor_class": "FakeProcessor",
            "model_class": "Qwen3VLForConditionalGeneration",
            "model_id": kwargs["model_id"],
            "requested_revision": kwargs["revision"],
            "presentation_mode": kwargs["presentation_mode"],
            "native_video_fps": "2.0",
        },
    )
    monkeypatch.setattr(module, "collect", lambda *args, **kwargs: iter(_raw(cases, "leak")))
    output = tmp_path / "evidence"
    manifest = run(
        inputs=frozen_corpus / "frozen_inputs.jsonl",
        frame_root=frozen_corpus,
        output=output,
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        presentation_mode="native_video",
        config_sha256="a" * 64,
        code_sha256="b" * 64,
        git_commit="c" * 40,
    )
    assert not (output / "RUNNING.json").exists()
    assert validate_corpus(output / "corpus")["case_count"] == 240
    assert manifest["corpus_tree_sha256"] == corpus_tree_sha256(output / "corpus")
    with pytest.raises(FileExistsError):
        run(
            inputs=frozen_corpus / "frozen_inputs.jsonl",
            frame_root=frozen_corpus,
            output=output,
            model_id="Qwen/Qwen3-VL-8B-Instruct",
            revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            presentation_mode="native_video",
            config_sha256="a" * 64,
            code_sha256="b" * 64,
            git_commit="c" * 40,
        )


def _make_evidence(corpus: Path, root: Path, config_sha: str, code_sha: str, git_commit: str) -> None:
    shutil.copytree(corpus, root / "corpus")
    cases = load_cases(root / "corpus" / "frozen_inputs.jsonl")
    raw_path = root / "raw_completions.jsonl"
    with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in _raw(cases, "leak"):
            handle.write(canonical_json(record) + "\n")
    runtime = {
        "python": "3.12.0",
        "platform": "Linux-test",
        "torch": "2.test",
        "transformers": "5.test",
        "pillow": "12.test",
        "cuda_runtime": "13.test",
        "cuda_device": "GH200-test",
        "processor_class": "Qwen3VLProcessor",
        "model_class": "Qwen3VLForConditionalGeneration",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "requested_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "presentation_mode": "native_video",
        "native_video_fps": "2.0",
    }
    write_json(
        root / "MANIFEST.json",
        {
            "kind": "visual_hindsight_g0_evidence",
            "schema_version": "visual-hindsight-g0-v2",
            "model_id": runtime["model_id"],
            "revision": runtime["requested_revision"],
            "presentation_mode": "native_video",
            "case_count": 240,
            "pair_count": 48,
            "max_new_tokens": 8,
            "config_sha256": config_sha,
            "code_sha256": code_sha,
            "git_commit": git_commit,
            "corpus_tree_sha256": corpus_tree_sha256(root / "corpus"),
            "raw_sha256": sha256_file(raw_path),
            "runtime": runtime,
        },
    )


def test_verifier_binds_exact_config_cardinality_and_complete_frame_tree(
    frozen_corpus: Path, tmp_path: Path
) -> None:
    repository = Path(__file__).resolve().parents[1]
    config = repository / "configs" / "visual_hindsight_g0.yaml"
    config_sha, code_sha, git_commit = sha256_file(config), "b" * 64, "c" * 40
    evidence = tmp_path / "evidence"
    _make_evidence(frozen_corpus, evidence, config_sha, code_sha, git_commit)
    report = verify(
        config=config,
        roots=[evidence],
        destination=tmp_path / "report.json",
        expected_config_sha256=config_sha,
        expected_code_sha256=code_sha,
        expected_git_commit=git_commit,
    )
    assert report["decision"] == "EXPAND_VISUAL_HINDSIGHT_STUDY"

    bad_count = tmp_path / "bad-count"
    shutil.copytree(evidence, bad_count)
    manifest = json.loads((bad_count / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["pair_count"] = 1
    write_json(bad_count / "MANIFEST.json", manifest)
    with pytest.raises(ValueError, match="cardinality"):
        verify(
            config=config,
            roots=[bad_count],
            destination=tmp_path / "bad-count-report.json",
            expected_config_sha256=config_sha,
            expected_code_sha256=code_sha,
            expected_git_commit=git_commit,
        )

    tampered = tmp_path / "tampered"
    shutil.copytree(evidence, tampered)
    target = next((tampered / "corpus" / "frames").rglob("*.png"))
    target.write_bytes(target.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="frame checksum|frame tree"):
        verify(
            config=config,
            roots=[tampered],
            destination=tmp_path / "tampered-report.json",
            expected_config_sha256=config_sha,
            expected_code_sha256=code_sha,
            expected_git_commit=git_commit,
        )


def test_config_freezes_native_video_primary_and_three_locations() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "visual_hindsight_g0.yaml").read_text(encoding="utf-8"))
    assert config["schema_version"] == "visual-hindsight-g0-v2"
    assert config["primary_evidence"] == "qwen3_vl:native_video"
    assert config["corpus"]["pairs"] == 48
    assert config["corpus"]["locations"] == list(LOCATIONS)
    assert config["models"]["qwen3_vl"]["id"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert config["models"]["gemma4"]["allowed_presentations"] == ["multi_image"]
    HindsightThresholds(**config["thresholds"])
