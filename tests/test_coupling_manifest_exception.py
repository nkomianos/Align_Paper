import sys
from pathlib import Path
import shutil
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from coupling_manifest_exception import validate, PIN


def test_exact_frozen_exception_and_tamper_rejection(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'artifacts/coupling_clarification_v1/prepared'
    assert validate(source, PIN)['invalid_self_entry_sha256'].startswith('e3b0')
    for p in source.iterdir():
        shutil.copyfile(p, tmp_path / p.name)
    assert validate(tmp_path, PIN)['checked_substantive_inputs']['answer_key.json']
    with pytest.raises(AssertionError):
        validate(tmp_path, '0' * 64)
    (tmp_path / 'cases.json').write_bytes(b'[]')
    with pytest.raises(AssertionError):
        validate(tmp_path, PIN)


def test_scientific_analyzer_unchanged_except_manifest_validation():
    base = Path(__file__).resolve().parents[1] / 'scripts'
    original=(base/'analyze_gpu_coupling_clarification.py').read_text()
    changed=(base/'analyze_gpu_coupling_clarification_exception.py').read_text()
    original=original.replace("pm=json.loads((prepared/'MANIFEST.json').read_text());assert all(digest(prepared/n)==h for n,h in pm.items())", "from coupling_manifest_exception import validate\n    validate(prepared,freeze['prepared_manifest_sha'],public_only=False)")
    assert original.strip() == changed.strip()
