"""Transfer only the three authorized learning/DEV input files, never reserved data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from interaction_sprint.hindsight_execution_integrity import load_learning_dev


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    learning, development, receipt = load_learning_dev(args.input_root)
    if args.out.exists() or args.out.with_suffix('.json').exists():
        raise FileExistsError('preserve existing transport receipt')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    with zipfile.ZipFile(args.out, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('MANIFEST.json', 'learning.json', 'development.json'):
            content = (args.input_root / name).read_bytes()
            files[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    with zipfile.ZipFile(args.out) as archive:
        assert set(archive.namelist()) == set(files)
        assert {n: hashlib.sha256(archive.read(n)).hexdigest() for n in files} == files
    report = {'archive_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
              'files': files, 'learning_rows': len(learning), 'development_rows': len(development),
              'input_receipt': receipt, 'confirmation_opened': False,
              'confirmation_packaged': False, 'neural_inference': False}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('archive_sha256', 'files', 'learning_rows', 'development_rows')}, indent=2))


if __name__ == '__main__':
    main()
