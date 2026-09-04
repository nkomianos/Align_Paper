"""Verify a downloaded coupling bundle and generate an explicit local mapping.

Run after extraction into a fresh directory. Does not alter evidence or inputs.
The remote BUNDLE receipt is a transport checksum, not a scientific certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(folder, prepared):
    folder, prepared = folder.resolve(), prepared.resolve()
    receipt = json.loads((folder / 'BUNDLE.json').read_text())
    assert digest(folder / 'evidence_bundle.tar.gz') == receipt['archive_sha256']
    for name, expected in receipt['metadata'].items():
        path = (folder / name).resolve()
        assert path.is_relative_to(folder)
        assert digest(path) == expected, name
    files = json.loads((folder / 'bundle_files.sha256.json').read_text())
    assert len(files) == receipt['files']
    for name, expected in files.items():
        path = (folder / name).resolve()
        assert path.is_relative_to(folder)
        assert digest(path) == expected, name
    assert digest(prepared / 'MANIFEST.json') == digest(folder / 'prepared/MANIFEST.json')
    for name, expected in json.loads((prepared / 'MANIFEST.json').read_text()).items():
        path = (prepared / name).resolve()
        assert path.is_relative_to(prepared)
        assert digest(path) == expected, name
    relative = json.loads((folder / 'mapping.relative.json').read_text())
    mapping = {}
    for original, member in relative.items():
        path = (folder / member).resolve()
        assert path.is_relative_to(folder)
        mapping[original] = str(prepared if member == 'prepared' else path)
    with (folder / 'mapping.local.json').open('x') as stream:
        json.dump(mapping, stream, indent=2)
    result = {'bundle_files_verified': len(files), 'archive_sha256': receipt['archive_sha256'],
              'scope': 'Transport and extracted bytes verified; run scientific verifier separately.'}
    with (folder / 'download_verified.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('prepared', type=Path)
    args = parser.parse_args()
    verify(args.folder, args.prepared)
