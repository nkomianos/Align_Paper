"""Run unchanged fixed-revision HMAC tests on four whole-package variants."""
import argparse
import ast
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
from qualify_pyjwt_component import FIX, fetch, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    result = json.loads((a.source/'RESULT.json').read_text())
    receipts = {}
    roots = {}
    for label, revision in [('fixed', FIX), ('vulnerable', result['vulnerable_revision'])]:
        url = f'https://codeload.github.com/jpadilla/pyjwt/zip/{revision}'
        raw = fetch(url)
        (a.out/f'{label}.zip').write_bytes(raw)
        receipts[label] = {'url': url, 'sha256': digest(raw)}
        destination = (a.out/label).resolve()
        destination.mkdir()
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info in archive.infolist():
                target = (destination/info.filename).resolve()
                assert target.is_relative_to(destination)
                assert (info.external_attr >> 16) & 0o170000 != 0o120000, 'symlink rejected'
            archive.extractall(destination)
        roots[label] = next(destination.iterdir())
    (a.out/'DOWNLOAD.json').write_text(json.dumps(receipts, indent=2))
    for arm, marker in [('ed25519_only', b'ssh-ed25519'), ('ecdsa_only', b'ecdsa-sha2-nistp256')]:
        root = (a.out/arm).resolve()
        shutil.copytree(roots['vulnerable'], root)
        path = root/'jwt/algorithms.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HMACAlgorithm')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_key')
        assignment = next(n for n in method.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'invalid_strings')
        assignment.value.elts.append(ast.Constant(marker))
        path.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n')
        roots[arm] = root
    outcomes = {}
    # Use the full fixed test tree for every arm, preserving exact fixture bytes.
    # Separate path prevents rewriting either source checkout's original tests.
    for arm, root in roots.items():
        suite = root/'qualification_tests'
        shutil.copytree(roots['fixed']/'tests', suite)
        assert digest((suite/'test_algorithms.py').read_bytes()) == digest((a.source/'fixed_tests_test_algorithms.py').read_bytes())
        origin_command = [sys.executable, '-c', 'import jwt; print(jwt.__file__)']
        origin = subprocess.check_output(origin_command, cwd=root, text=True).strip()
        assert Path(origin).resolve().is_relative_to(root.resolve()), 'wrong installed package imported'
        command = [sys.executable, '-m', 'pytest', '-o', 'addopts=', '-q', 'qualification_tests/test_algorithms.py', '-k', 'hmac']
        run = subprocess.run(command, cwd=root, text=True, capture_output=True)
        (a.out/f'{arm}.log').write_text(run.stdout+'\n'+run.stderr)
        outcomes[arm] = {'exit_code': run.returncode, 'command': command, 'package_origin': origin,
                         'output_tail': (run.stdout+'\n'+run.stderr)[-1600:]}
    report = {'classification': 'DEVELOPMENTAL_UPSTREAM_HMAC_REGRESSION_REPLAY',
              'all_arms_pass': all(x['exit_code']==0 for x in outcomes.values()), 'outcomes': outcomes,
              'scope': 'Only unchanged fixed-revision HMAC tests selected by name, against complete packages. Does not cover entire JWT library or establish security completeness.'}
    (a.out/'RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
