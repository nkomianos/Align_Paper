"""Explicit post-run exception for one frozen self-referential input manifest.

Never returns a fabricated digest or alters any file. Only this exact manifest
is admitted; its three substantive entries remain independently checked.
"""
import hashlib
import json
from pathlib import Path

PIN = '932438274ea70fed82c16f3056fd4b232f0a63a580d898fd69a6da8c07abacbf'
EXPECTED = {
    'answer_key.json': 'f21c5fbc67d376d4f36c1aa785618d463adfba2b3f67428a0bf8b2438ee634bc',
    'cases.json': 'b23313d681da07987400f063afe8280bb8170bcc514bc243e1fbf591c43504fd',
    'MANIFEST.json': hashlib.sha256(b'').hexdigest(),
    'preparation.json': '29efa6fc0337be1ec8df9f58e06991860d933532910adb86ef64eb96c6827997',
}


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(prepared, frozen_pin, *, public_only=False):
    prepared = Path(prepared)
    manifest_path = prepared / 'MANIFEST.json'
    actual = file_digest(manifest_path)
    assert frozen_pin == PIN and actual == PIN, 'Not the specifically approved frozen manifest'
    manifest = json.loads(manifest_path.read_text())
    assert manifest == EXPECTED, 'Manifest membership or values changed'
    checked, absent = {}, []
    for name, expected in EXPECTED.items():
        if name == 'MANIFEST.json':
            continue  # Only after whole-file digest, exact schema and values pass.
        path = prepared / name
        if public_only and name == 'answer_key.json' and not path.exists():
            absent.append(name)
            continue
        assert file_digest(path) == expected, 'Input mismatch: ' + name
        checked[name] = expected
    return {
        'exception': 'POST_RUN_FROZEN_PREPARATION_MANIFEST_SELF_ENTRY',
        'exception_validator_sha256': file_digest(__file__),
        'actual_manifest_sha256': actual,
        'invalid_self_entry_sha256': EXPECTED['MANIFEST.json'],
        'checked_substantive_inputs': checked,
        'private_inputs_absent_here_require_local_verification': absent,
        'interpretation': 'Manifest creation opened the file before enumerating its directory, recording its then-empty hash. The frozen whole-manifest pin and all substantive inputs are unchanged. Self-entry is invalid; it is not reported as a passed checksum.',
    }
