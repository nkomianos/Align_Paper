from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from validator_monoculture.corpus import build_corpus
from validator_monoculture.prepare import _code_inventory, main, prepare_corpus, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "validator_monoculture_g0.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_writes_fresh_attested_corpus_and_frozen_config(tmp_path: Path) -> None:
    destination = tmp_path / "corpus"
    expected_config = _sha(CONFIG)
    expected_corpus = build_corpus().corpus_sha256

    manifest = prepare_corpus(
        destination,
        CONFIG,
        expected_config_sha256=expected_config,
        expected_corpus_sha256=expected_corpus,
    )

    assert manifest["kind"] == "validator_monoculture_g0_preparation"
    assert manifest["config_sha256"] == expected_config
    assert manifest["corpus_sha256"] == expected_corpus
    assert manifest["task_count"] == 32
    assert manifest["input_sha256"] == {
        "FROZEN_CONFIG.yaml": sha256_file(destination / "FROZEN_CONFIG.yaml"),
        "MANIFEST.json": sha256_file(destination / "MANIFEST.json"),
        "private/oracles.jsonl": sha256_file(destination / "private" / "oracles.jsonl"),
        "public/tasks.jsonl": sha256_file(destination / "public" / "tasks.jsonl"),
    }
    assert manifest["code_files_sha256"]["src/validator_monoculture/prepare.py"] == sha256_file(
        PROJECT_ROOT / "src" / "validator_monoculture" / "prepare.py"
    )
    assert json.loads(
        (destination / "PREPARATION_MANIFEST.json").read_text(encoding="utf-8")
    ) == manifest
    assert (destination / "FROZEN_CONFIG.yaml").read_bytes() == CONFIG.read_bytes()


def test_prepare_refuses_existing_destination_without_mutating_it(tmp_path: Path) -> None:
    destination = tmp_path / "corpus"
    prepare_corpus(destination, CONFIG)
    before = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }

    with pytest.raises(FileExistsError, match="overwrite"):
        prepare_corpus(destination, CONFIG)

    after = {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_prepare_rejects_hash_or_shape_mismatch_before_writing(tmp_path: Path) -> None:
    wrong_hash_destination = tmp_path / "wrong-hash"
    with pytest.raises(ValueError, match="config SHA-256 mismatch"):
        prepare_corpus(
            wrong_hash_destination,
            CONFIG,
            expected_config_sha256="0" * 64,
        )
    assert not wrong_hash_destination.exists()

    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        CONFIG.read_text(encoding="utf-8").replace("cwe_families: 8", "cwe_families: 9"),
        encoding="utf-8",
    )
    wrong_shape_destination = tmp_path / "wrong-shape"
    with pytest.raises(ValueError, match=r"corpus\.cwe_families"):
        prepare_corpus(wrong_shape_destination, bad_config)
    assert not wrong_shape_destination.exists()


def test_prepare_cli_accepts_exact_pins_and_prints_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "corpus"
    result = main(
        [
            "--destination",
            str(destination),
            "--config",
            str(CONFIG),
            "--expected-config-sha256",
            _sha(CONFIG),
            "--expected-corpus-sha256",
            build_corpus().corpus_sha256,
            "--expected-code-sha256",
            _code_inventory(PROJECT_ROOT / "src")[1],
        ]
    )
    assert result == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads(
        (destination / "PREPARATION_MANIFEST.json").read_text(encoding="utf-8")
    )
