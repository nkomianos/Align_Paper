from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import yaml

from validator_monoculture import runner
from validator_monoculture.corpus import write_corpus
from validator_monoculture.runner import classify_patches, collect_patches, collect_tests


class FakeRuntime:
    def __init__(self, model_id: str, revision: str, *, local_files_only: bool = False) -> None:
        self.model_id = model_id

    def generate(self, prompt: str, **_: object) -> str:
        if "independent security-test author" in prompt:
            return json.dumps({"tests": [
                {"args": ["x"], "kwargs": {}, "expected": None},
                {"args": ["y"], "kwargs": {}, "expected": None},
            ]})
        entrypoint = prompt.split("ENTRYPOINT: ", 1)[1].splitlines()[0]
        signature = prompt.split("SIGNATURE: ", 1)[1].splitlines()[0]
        arguments = signature[signature.index("(") + 1:signature.rindex(")")]
        names = [part.split(":", 1)[0].strip() for part in arguments.split(",") if part.strip()]
        return f"```python\ndef {entrypoint}({', '.join(names)}):\n    return None\n```"


class InterruptibleRuntime(FakeRuntime):
    calls: list[int] = []
    fail_after: int | None = None

    def generate(self, prompt: str, **kwargs: object) -> str:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated runtime interruption")
        self.calls.append(int(kwargs["seed"]))
        return super().generate(prompt, **kwargs)

    @classmethod
    def reset(cls, *, fail_after: int | None) -> None:
        cls.calls = []
        cls.fail_after = fail_after


def _config(path: Path) -> None:
    value = {
        "kind": "validator_monoculture_g0",
        "models": {
            "qwen3_5": {"id": "Qwen/Qwen3.5-9B", "revision": "a"},
            "gemma4": {"id": "google/gemma-4-12B-it", "revision": "b"},
        },
        "generation": {
            "patches_per_model_task": 1,
            "spec_only_test_suites_per_verifier_task": 1,
            "patch_aware_test_suites_per_verifier_patch": 1,
            "tests_per_suite": 2,
            "patch_max_new_tokens": 64,
            "test_max_new_tokens": 64,
            "do_sample": False,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        "execution": {
            "sandbox_timeout_seconds": 2.0,
            "max_test_completion_bytes": 16384,
        },
    }
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_public_collection_is_phase_separated_and_non_overwriting(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    root = tmp_path / "run"
    patch_report = collect_patches(
        output_root=root,
        public_corpus=corpus / "public" / "tasks.jsonl",
        config_path=config,
        family="qwen3_5",
        runtime_factory=FakeRuntime,
    )
    assert patch_report["state"] == "COMPLETE"
    assert patch_report["record_count"] == 32
    assert (root / "phases" / "patches_qwen3_5" / "COMPLETE").is_file()
    try:
        collect_patches(
            output_root=root,
            public_corpus=corpus / "public" / "tasks.jsonl",
            config_path=config,
            family="qwen3_5",
            runtime_factory=FakeRuntime,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("a frozen phase was overwritten")

    test_report = collect_tests(
        output_root=root,
        public_corpus=corpus / "public" / "tasks.jsonl",
        config_path=config,
        family="qwen3_5",
        prompt_mode="spec_only",
        runtime_factory=FakeRuntime,
    )
    assert test_report["record_count"] == 32
    rows = [json.loads(line) for line in (
        root / "phases" / "tests_qwen3_5_spec_only" / "raw_test_completions.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert all(row["patch_id"] is None and len(row["parsed_tests"]) == 2 for row in rows)


def test_patch_and_test_collection_resume_durable_exact_prefix(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    public = corpus / "public" / "tasks.jsonl"

    patch_root = tmp_path / "patch-resume"
    InterruptibleRuntime.reset(fail_after=3)
    with pytest.raises(RuntimeError, match="simulated runtime interruption"):
        collect_patches(
            output_root=patch_root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=InterruptibleRuntime,
        )
    patch_phase = patch_root / "phases" / "patches_qwen3_5"
    patch_partial = patch_phase / "raw_patch_completions.jsonl.partial"
    assert (patch_phase / "RUNNING.json").is_file()
    assert not (patch_phase / "raw_patch_completions.jsonl").exists()
    partial_rows = [json.loads(line) for line in patch_partial.read_text(encoding="utf-8").splitlines()]
    assert len(partial_rows) == 3
    completed_seeds = {row["seed"] for row in partial_rows}

    InterruptibleRuntime.reset(fail_after=None)
    report = collect_patches(
        output_root=patch_root,
        public_corpus=public,
        config_path=config,
        family="qwen3_5",
        runtime_factory=InterruptibleRuntime,
        resume=True,
    )
    assert report["record_count"] == 32
    assert len(InterruptibleRuntime.calls) == 29
    assert completed_seeds.isdisjoint(InterruptibleRuntime.calls)
    assert not patch_partial.exists()
    assert not (patch_phase / "RUNNING.json").exists()
    baseline_root = tmp_path / "patch-baseline"
    collect_patches(
        output_root=baseline_root,
        public_corpus=public,
        config_path=config,
        family="qwen3_5",
        runtime_factory=FakeRuntime,
    )
    assert (
        patch_phase / "raw_patch_completions.jsonl"
    ).read_bytes() == (
        baseline_root / "phases" / "patches_qwen3_5" / "raw_patch_completions.jsonl"
    ).read_bytes()

    test_root = tmp_path / "test-resume"
    InterruptibleRuntime.reset(fail_after=4)
    with pytest.raises(RuntimeError, match="simulated runtime interruption"):
        collect_tests(
            output_root=test_root,
            public_corpus=public,
            config_path=config,
            family="gemma4",
            prompt_mode="spec_only",
            runtime_factory=InterruptibleRuntime,
        )
    test_phase = test_root / "phases" / "tests_gemma4_spec_only"
    test_partial = test_phase / "raw_test_completions.jsonl.partial"
    assert (test_phase / "RUNNING.json").is_file()
    test_rows = [json.loads(line) for line in test_partial.read_text(encoding="utf-8").splitlines()]
    assert len(test_rows) == 4
    completed_test_seeds = {row["seed"] for row in test_rows}

    InterruptibleRuntime.reset(fail_after=None)
    report = collect_tests(
        output_root=test_root,
        public_corpus=public,
        config_path=config,
        family="gemma4",
        prompt_mode="spec_only",
        runtime_factory=InterruptibleRuntime,
        resume=True,
    )
    assert report["record_count"] == 32
    assert len(InterruptibleRuntime.calls) == 28
    assert completed_test_seeds.isdisjoint(InterruptibleRuntime.calls)
    assert not test_partial.exists()
    assert not (test_phase / "RUNNING.json").exists()


def test_resume_refuses_tampered_partial_before_generation(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    root = tmp_path / "tampered"
    public = corpus / "public" / "tasks.jsonl"

    InterruptibleRuntime.reset(fail_after=1)
    with pytest.raises(RuntimeError):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=InterruptibleRuntime,
        )
    partial = root / "phases" / "patches_qwen3_5" / "raw_patch_completions.jsonl.partial"
    row = json.loads(partial.read_text(encoding="utf-8"))
    row["seed"] += 1
    partial.write_text(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    InterruptibleRuntime.reset(fail_after=None)
    with pytest.raises(ValueError, match="deterministic validation"):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=InterruptibleRuntime,
            resume=True,
        )
    assert InterruptibleRuntime.calls == []
    assert partial.is_file()
    assert (partial.parent / "RUNNING.json").is_file()


def test_resume_recovers_torn_jsonl_and_preserves_suffix(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    root = tmp_path / "torn"
    public = corpus / "public" / "tasks.jsonl"

    InterruptibleRuntime.reset(fail_after=1)
    with pytest.raises(RuntimeError, match="simulated runtime interruption"):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=InterruptibleRuntime,
        )
    partial = root / "phases" / "patches_qwen3_5" / "raw_patch_completions.jsonl.partial"
    torn_suffix = b'{"incomplete":'
    with partial.open("ab") as handle:
        handle.write(torn_suffix)

    InterruptibleRuntime.reset(fail_after=None)
    report = collect_patches(
        output_root=root,
        public_corpus=public,
        config_path=config,
        family="qwen3_5",
        runtime_factory=InterruptibleRuntime,
        resume=True,
    )
    assert report["record_count"] == 32
    assert len(InterruptibleRuntime.calls) == 31
    suffix_hash = hashlib.sha256(torn_suffix).hexdigest()
    recovered = (
        root
        / "recovery"
        / "patches_qwen3_5"
        / f"{partial.name}.torn-{suffix_hash}.bin"
    )
    assert recovered.read_bytes() == torn_suffix


def test_runtime_provenance_is_bound_before_resume_prefix_validation(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    root = tmp_path / "provenance"
    public = corpus / "public" / "tasks.jsonl"

    class ProvenanceRuntime(InterruptibleRuntime):
        provenance_tag = "first"

        def provenance(self) -> dict[str, str]:
            return {"runtime": self.provenance_tag}

    ProvenanceRuntime.reset(fail_after=1)
    with pytest.raises(RuntimeError, match="simulated runtime interruption"):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=ProvenanceRuntime,
        )
    ProvenanceRuntime.provenance_tag = "different"
    ProvenanceRuntime.reset(fail_after=None)
    with pytest.raises(ValueError, match="RUNNING marker"):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=ProvenanceRuntime,
            resume=True,
        )
    assert ProvenanceRuntime.calls == []


def test_input_snapshot_bytes_are_used_for_parsing_and_manifest(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    public = corpus / "public" / "tasks.jsonl"
    original = public.read_bytes()
    original_sha256 = hashlib.sha256(original).hexdigest()

    class MutatingRuntime(FakeRuntime):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            public.write_bytes(b"not the snapshotted corpus\n")

    report = collect_patches(
        output_root=tmp_path / "snapshot",
        public_corpus=public,
        config_path=config,
        family="qwen3_5",
        runtime_factory=MutatingRuntime,
    )
    assert report["record_count"] == 32
    assert report["public_corpus_sha256"] == original_sha256


def test_complete_plus_running_finalization_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    root = tmp_path / "finalization"
    public = corpus / "public" / "tasks.jsonl"
    original_create = runner._create_empty_file

    def interrupt_after_complete(path: Path) -> None:
        original_create(path)
        if path.name == "COMPLETE":
            raise RuntimeError("simulated finalization interruption")

    monkeypatch.setattr(runner, "_create_empty_file", interrupt_after_complete)
    with pytest.raises(RuntimeError, match="finalization interruption"):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family="qwen3_5",
            runtime_factory=InterruptibleRuntime,
        )
    phase = root / "phases" / "patches_qwen3_5"
    assert (phase / "COMPLETE").is_file()
    assert (phase / "RUNNING.json").is_file()

    monkeypatch.setattr(runner, "_create_empty_file", original_create)
    InterruptibleRuntime.reset(fail_after=0)
    report = collect_patches(
        output_root=root,
        public_corpus=public,
        config_path=config,
        family="qwen3_5",
        runtime_factory=InterruptibleRuntime,
        resume=True,
    )
    assert report["state"] == "COMPLETE"
    assert not (phase / "RUNNING.json").exists()
    assert InterruptibleRuntime.calls == []


def test_phase_lease_is_exclusive(tmp_path: Path) -> None:
    with runner._phase_lease(tmp_path, "patches_qwen3_5"):
        with pytest.raises(RuntimeError, match="already leased"):
            with runner._phase_lease(tmp_path, "patches_qwen3_5"):
                pass


def test_classification_resume_appends_only_missing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    write_corpus(corpus)
    config = tmp_path / "config.yaml"
    _config(config)
    root = tmp_path / "classification-resume"
    public = corpus / "public" / "tasks.jsonl"
    private = corpus / "private" / "oracles.jsonl"
    for family in ("qwen3_5", "gemma4"):
        collect_patches(
            output_root=root,
            public_corpus=public,
            config_path=config,
            family=family,
            runtime_factory=FakeRuntime,
        )

    state = {"calls": 0, "fail_after": 3}

    def fake_classify(*_: object, **__: object) -> dict[str, object]:
        if state["fail_after"] is not None and state["calls"] >= state["fail_after"]:
            raise RuntimeError("simulated classification interruption")
        state["calls"] += 1
        return {
            "schema_version": "test-classification-v1",
            "status": "INCOMPLETE",
            "plausible_security_repair": False,
            "fully_correct": False,
        }

    monkeypatch.setattr(runner, "classify_patch", fake_classify)
    with pytest.raises(RuntimeError, match="classification interruption"):
        classify_patches(
            output_root=root,
            public_corpus=public,
            private_oracles=private,
            config_path=config,
        )
    phase = root / "phases" / "classifications"
    partial = phase / "private_classifications.jsonl.partial"
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 3

    appended: list[str] = []
    original_append = runner._append_jsonl_record

    def tracking_append(path: Path, record: dict[str, object]) -> None:
        appended.append(str(record["patch_id"]))
        original_append(path, record)

    monkeypatch.setattr(runner, "_append_jsonl_record", tracking_append)
    state.update(calls=0, fail_after=None)
    report = classify_patches(
        output_root=root,
        public_corpus=public,
        private_oracles=private,
        config_path=config,
        resume=True,
    )
    assert report["record_count"] == 64
    assert len(appended) == 61
    final_rows = [json.loads(line) for line in (
        phase / "private_classifications.jsonl"
    ).read_text(encoding="utf-8").splitlines()]
    assert len(final_rows) == 64
    assert len({row["patch_id"] for row in final_rows}) == 64
    assert not partial.exists()
    assert not (phase / "RUNNING.json").exists()


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("patch-run", ["--family", "qwen3_5"]),
        ("classify", ["--private-oracles", "private.jsonl"]),
        ("test-run", ["--family", "gemma4", "--prompt-mode", "spec_only"]),
    ],
)
def test_each_wrapper_cli_accepts_resume(command: str, extra: list[str]) -> None:
    args = runner._parser().parse_args([
        command,
        "--output-root", "run",
        "--public-corpus", "public.jsonl",
        "--config", "config.yaml",
        "--resume",
        *extra,
    ])
    assert args.resume is True
