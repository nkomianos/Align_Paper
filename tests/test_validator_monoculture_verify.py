from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Mapping, Sequence

import pytest

from under_extinction.io import read_jsonl, sha256_file, write_json, write_jsonl
from validator_monoculture.corpus import write_corpus
from validator_monoculture import oracle_preflight, runner
from validator_monoculture.schema import PrivateOracle, PublicTask
from validator_monoculture.analysis import GateThresholds
from validator_monoculture.verify import (
    _apply_planted_control_gate,
    _canonical_hash,
    _snapshot,
    _validate_run_binding,
    verify,
)


_VULNERABLE = re.compile(
    r"VULNERABLE IMPLEMENTATION:\s*```python\s*(.*?)```", re.DOTALL
)


class _FakeRuntime:
    """Deterministic collection stub; verification still rebuilds all records."""

    def __init__(self, model_id: str, revision: str, **_: Any) -> None:
        self.model_id = model_id
        self.revision = revision

    def generate(self, prompt: str, **_: Any) -> str:
        if "Return a complete replacement implementation" in prompt:
            match = _VULNERABLE.search(prompt)
            assert match is not None
            return f"```python\n{match.group(1).strip()}\n```"
        # The two suites deliberately repeat canonical content.  The offline
        # verifier must retain twelve proposal slots but execute only six tests.
        return json.dumps(
            {
                "tests": [
                    {"args": [f"synthetic-{index}"], "kwargs": {}, "expected": index}
                    for index in range(6)
                ]
            },
            sort_keys=True,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.revision,
            "chat_template_sha256": "a" * 64,
            "transformers_version": "5.15.0",
            "torch_version": "2.7.1",
            "python_version": "3.12.11",
            "platform_system": "Linux",
            "cuda_version": "12.8",
            "cuda_available": True,
            "device_name": "fake-gh200",
            "device_memory_bytes": 96 * 1024**3,
            "compute_capability": [9, 0],
        }


def _fake_classify(
    task: PublicTask,
    _oracle: PrivateOracle,
    _candidate_source: str,
    **_: Any,
) -> dict[str, Any]:
    return {
        "schema_version": "validator-monoculture-patch-classification-v2",
        "task_id": task.task_id,
        "cwe_id": task.cwe_id,
        "split": task.split.value,
        "status": "PLAUSIBLE_INCOMPLETE_SECURITY_REPAIR",
        "plausible_security_repair": True,
        "fully_correct": False,
    }


def _fake_vectors(
    _task: PublicTask,
    _oracle: PrivateOracle,
    _candidate_source: str,
    vectors: Sequence[Mapping[str, object]],
    **_: Any,
) -> dict[str, Any]:
    ids = [str(vector["slot_id"]) for vector in vectors]
    return {
        "proposal_test_ids": ids,
        "valid_test_ids": ids,
        "kill_test_ids": [],
        "counts": {"indeterminate_execution_count": 0},
    }


def _fake_oracle_preflight(*_: Any, **__: Any) -> dict[str, Any]:
    return {
        "kind": "validator_monoculture_oracle_preflight",
        "status": "PASS",
        "interpretation": "apparatus_validation_only",
        "task_count": 32,
    }


@pytest.fixture(scope="module")
def frozen_evidence(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("validator-monoculture-verifier")
    corpus_root = root / "corpus"
    write_corpus(corpus_root)
    public = corpus_root / "public" / "tasks.jsonl"
    private = corpus_root / "private" / "oracles.jsonl"
    config = Path(__file__).resolve().parents[1] / "configs" / "validator_monoculture_g0.yaml"
    evidence = root / "evidence"
    for family in runner.FAMILIES:
        runner.collect_patches(
            output_root=evidence,
            public_corpus=public,
            config_path=config,
            family=family,
            runtime_factory=_FakeRuntime,
        )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runner, "classify_patch", _fake_classify)
    try:
        runner.classify_patches(
            output_root=evidence,
            public_corpus=public,
            private_oracles=private,
            config_path=config,
        )
    finally:
        monkeypatch.undo()
    for family in runner.FAMILIES:
        for mode in runner.PROMPT_MODES:
            runner.collect_tests(
                output_root=evidence,
                public_corpus=public,
                config_path=config,
                family=family,
                prompt_mode=mode,
                runtime_factory=_FakeRuntime,
            )
    return {
        "root": root,
        "evidence": evidence,
        "public": public,
        "private": private,
        "config": config,
    }


def _arguments(bundle: Mapping[str, Path], output: Path) -> dict[str, Any]:
    return {
        "evidence_root": bundle["evidence"],
        "public_corpus": bundle["public"],
        "private_oracle": bundle["private"],
        "config": bundle["config"],
        "expected_public_sha256": sha256_file(bundle["public"]),
        "expected_private_sha256": sha256_file(bundle["private"]),
        "expected_config_sha256": sha256_file(bundle["config"]),
        "expected_evidence_sha256": _canonical_hash(_snapshot(bundle["evidence"])),
        "output_report": output,
        "patch_classifier": _fake_classify,
        "vector_evaluator": _fake_vectors,
        "oracle_validator": _fake_oracle_preflight,
        "allow_test_hooks": True,
    }


def _refresh_phase_manifest(phase: Path, filename: str) -> None:
    manifest_path = phase / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = phase / filename
    manifest["files"][filename] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    write_json(manifest_path, manifest)


def test_phase_manifests_must_share_the_external_run_binding(tmp_path: Path) -> None:
    phases = tmp_path / "phases"
    expected = "b" * 64
    for name in ("one", "two"):
        phase = phases / name
        phase.mkdir(parents=True)
        write_json(phase / "MANIFEST.json", {"run_binding_sha256": expected})

    assert (
        _validate_run_binding(
            phases,
            {"one", "two"},
            expected_run_binding_sha256=expected,
        )
        == expected
    )
    write_json(phases / "two" / "MANIFEST.json", {"run_binding_sha256": "c" * 64})
    with pytest.raises(ValueError, match="not bound to the expected immutable run"):
        _validate_run_binding(
            phases,
            {"one", "two"},
            expected_run_binding_sha256=expected,
        )


def test_one_powerless_verifier_cwe_cell_cannot_authorize_kill() -> None:
    cwes = ["CWE-287", "CWE-400", "CWE-601", "CWE-89"]
    cells: dict[str, dict[str, dict[str, float | int]]] = {}
    for family in runner.FAMILIES:
        cells[family] = {
            cwe: {
                "detected_count": 1,
                "total_count": 4,
                "detection_rate": 0.25,
            }
            for cwe in cwes
        }
    # The aggregate rates and union CWE coverage satisfy the former gate, but
    # this single zero-power cell makes a scientific null uninterpretable.
    cells["qwen3_5"]["CWE-287"] = {
        "detected_count": 0,
        "total_count": 4,
        "detection_rate": 0.0,
    }
    for cwe in cwes[1:]:
        cells["qwen3_5"][cwe] = {
            "detected_count": 2,
            "total_count": 4,
            "detection_rate": 0.5,
        }
    planted_controls = {
        "macro_detection_rate_by_verifier": {
            "qwen3_5": 0.375,
            "gemma4": 0.25,
        },
        "cwes": cwes,
        "detection_power_by_verifier_cwe": cells,
        "cwe_coverage_rate": 1.0,
        "indeterminate_execution_count": 0,
    }
    gate = {
        "decision": "KILL_VALIDATOR_MONOCULTURE",
        "checks": {},
        "kill_checks": {},
        "reasons": [],
    }

    _apply_planted_control_gate(gate, planted_controls, GateThresholds())

    assert gate["decision"] == "INCONCLUSIVE_INSUFFICIENT_APPARATUS_POWER"
    assert gate["pass"] is False
    assert gate["checks"]["planted_control_macro_detection_power"] is True
    assert (
        gate["checks"]["planted_control_every_verifier_cwe_cell_power"]
        is False
    )
    assert gate["kill_checks"][
        "planted_control_every_verifier_cwe_cell_power"
    ] is False


def test_reconstructs_every_arm_and_deduplicates_by_canonical_content(
    frozen_evidence: Mapping[str, Path], tmp_path: Path
) -> None:
    before = {
        path.relative_to(frozen_evidence["evidence"]).as_posix(): sha256_file(path)
        for path in frozen_evidence["evidence"].rglob("*")
        if path.is_file()
    }
    report = verify(**_arguments(frozen_evidence, tmp_path / "verified.json"))

    assert report["decision"] == "INVALID_TEST_HOOKS_ACTIVE"
    assert report["pass"] is False
    assert (
        report["verification"]["status"]
        == "TEST_HOOKS_ACTIVE_NOT_SCIENTIFIC_EVIDENCE"
    )
    assert report["oracle_preflight"]["status"] == "PASS"
    assert report["verification"]["raw_patch_count"] == 32 * 2 * 3
    assert report["verification"]["eligible_patch_count"] == 32 * 2 * 3
    assert report["verification"]["analysis_row_count"] == 32 * 2 * 3 * 2 * 2
    assert report["verification"]["test_reconstruction_counts"][
        "duplicate_content_slots"
    ] == 6 * len(report["analysis_rows"])
    for row in report["analysis_rows"]:
        assert len(row["proposal_test_ids"]) == 12
        assert len(set(row["proposal_test_ids"])) == 12
        assert len(row["valid_test_ids"]) == 6
        assert all(
            f"test-{row['verifier_family']}-{row['prompt_mode']}-" in test_id
            for test_id in row["proposal_test_ids"]
        )
    same_patch = report["analysis_rows"][:4]
    namespaces = {
        (row["verifier_family"], row["prompt_mode"]): row["proposal_test_ids"][0]
        for row in same_patch
    }
    assert len(set(namespaces.values())) == 4
    after = {
        path.relative_to(frozen_evidence["evidence"]).as_posix(): sha256_file(path)
        for path in frozen_evidence["evidence"].rglob("*")
        if path.is_file()
    }
    assert after == before


def test_checksums_and_runner_side_parse_fields_fail_closed(
    frozen_evidence: Mapping[str, Path], tmp_path: Path
) -> None:
    copied = tmp_path / "checksum-run" / "evidence"
    shutil.copytree(frozen_evidence["evidence"], copied)
    raw = copied / "phases" / "patches_qwen3_5" / "raw_patch_completions.jsonl"
    raw.write_bytes(raw.read_bytes() + b"\n")
    bundle = dict(frozen_evidence)
    bundle["evidence"] = copied
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify(**_arguments(bundle, tmp_path / "checksum-report.json"))

    copied = tmp_path / "reconstruction-run" / "evidence"
    shutil.copytree(frozen_evidence["evidence"], copied)
    phase = copied / "phases" / "patches_qwen3_5"
    raw = phase / "raw_patch_completions.jsonl"
    records = list(read_jsonl(raw))
    records[0]["parsed_source"] = "def forged():\n    return True\n"
    write_jsonl(raw, records)
    _refresh_phase_manifest(phase, raw.name)
    bundle = dict(frozen_evidence)
    bundle["evidence"] = copied
    with pytest.raises(ValueError, match=r"stored (patch|canonical patch).*(reconstruct|invalid)"):
        verify(**_arguments(bundle, tmp_path / "reconstruction-report.json"))


def test_missing_crossed_suite_and_output_inside_retrieved_run_are_rejected(
    frozen_evidence: Mapping[str, Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="outside the retrieved run root"):
        verify(
            **_arguments(
                frozen_evidence,
                frozen_evidence["evidence"] / "forbidden-report.json",
            )
        )
    with pytest.raises(ValueError, match="outside the retrieved run root"):
        verify(
            **_arguments(
                frozen_evidence,
                frozen_evidence["root"] / "forbidden-sibling-report.json",
            )
        )

    copied = tmp_path / "missing-suite-run" / "evidence"
    shutil.copytree(frozen_evidence["evidence"], copied)
    phase = copied / "phases" / "tests_gemma4_patch_aware"
    raw = phase / "raw_test_completions.jsonl"
    records = list(read_jsonl(raw))
    records.pop()
    write_jsonl(raw, records)
    manifest_path = phase / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count"] = len(records)
    write_json(manifest_path, manifest)
    _refresh_phase_manifest(phase, raw.name)
    bundle = dict(frozen_evidence)
    bundle["evidence"] = copied
    with pytest.raises(ValueError, match="exact suite budget"):
        verify(**_arguments(bundle, tmp_path / "missing-suite-report.json"))


def test_production_requires_evidence_commitment_and_hooks_are_explicit(
    frozen_evidence: Mapping[str, Path], tmp_path: Path
) -> None:
    arguments = _arguments(frozen_evidence, tmp_path / "missing-commitment.json")
    for key in (
        "patch_classifier",
        "vector_evaluator",
        "oracle_validator",
        "allow_test_hooks",
        "expected_evidence_sha256",
    ):
        arguments.pop(key)
    with pytest.raises(ValueError, match="production verification requires"):
        verify(**arguments)

    arguments = _arguments(frozen_evidence, tmp_path / "implicit-hooks.json")
    arguments.pop("allow_test_hooks")
    with pytest.raises(ValueError, match="require allow_test_hooks=True"):
        verify(**arguments)


def test_frozen_inputs_are_read_once_and_default_preflight_uses_bound_objects(
    frozen_evidence: Mapping[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bundle = dict(frozen_evidence)
    for key in ("public", "private", "config"):
        copied = inputs / frozen_evidence[key].name
        shutil.copyfile(frozen_evidence[key], copied)
        bundle[key] = copied
    original_bytes = {key: bundle[key].read_bytes() for key in ("public", "private", "config")}
    arguments = _arguments(bundle, tmp_path / "read-once-report.json")
    arguments.pop("oracle_validator")

    observed: dict[str, object] = {}

    def bound_preflight(
        tasks: Mapping[str, PublicTask],
        oracles: Mapping[str, PrivateOracle],
        **_: Any,
    ) -> dict[str, Any]:
        observed["tasks"] = tasks
        observed["oracles"] = oracles
        for key in ("public", "private", "config"):
            bundle[key].write_bytes(f"changed-{key}".encode("ascii"))
        return _fake_oracle_preflight()

    monkeypatch.setattr(oracle_preflight, "validate_oracle", bound_preflight)
    original_read_bytes = Path.read_bytes
    reads = {bundle[key].resolve(): 0 for key in ("public", "private", "config")}

    def counted_read_bytes(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in reads:
            reads[resolved] += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    report = verify(**arguments)

    assert all(count == 1 for count in reads.values())
    assert set(observed["tasks"]) == set(observed["oracles"])
    for key, report_key in (
        ("public", "public_corpus_sha256"),
        ("private", "private_oracle_sha256"),
        ("config", "config_sha256"),
    ):
        assert report["verification"][report_key] == hashlib.sha256(
            original_bytes[key]
        ).hexdigest()
    assert report["decision"] == "INVALID_TEST_HOOKS_ACTIVE"


def test_runtime_provenance_must_match_across_family_phases(
    frozen_evidence: Mapping[str, Path], tmp_path: Path
) -> None:
    copied = tmp_path / "runtime-drift-run" / "evidence"
    shutil.copytree(frozen_evidence["evidence"], copied)
    manifest_path = (
        copied / "phases" / "tests_qwen3_5_spec_only" / "MANIFEST.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_provenance"]["device_name"] = "different-gh200"
    write_json(manifest_path, manifest)
    bundle = dict(frozen_evidence)
    bundle["evidence"] = copied
    with pytest.raises(ValueError, match="runtime provenance differs across phases"):
        verify(**_arguments(bundle, tmp_path / "runtime-drift-report.json"))
