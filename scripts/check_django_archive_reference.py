"""Read-only path-arithmetic counterexample to a released fixed helper.

No archive is extracted and no file is created by the tested function.
This does not test current Django or the complete benchmark container.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import posixpath
import textwrap
from types import SimpleNamespace


class SuspiciousOperation(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    data = args.dataset.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '65e873923d738221389ed32408c4f5fa75882ea07609c500761207d013da0f62'
    case = next(x for x in json.loads(data) if x['cve_id'] == 'CVE-2021-3281')
    source = next(x['snippet'] for x in case['fix_func'] if 'def target_filename' in x['snippet'])
    namespace = {'os': SimpleNamespace(path=posixpath), 'SuspiciousOperation': SuspiciousOperation}
    exec(compile(ast.parse(textwrap.dedent(source)), 'released_target_filename', 'exec'), namespace)
    rows = []
    for name in ['sub/file.txt', '../elsewhere/file.txt', '../dest_sibling/file.txt']:
        try:
            output = namespace['target_filename'](None, '/qualification/dest', name)
            status = 'accepted'
            contained = posixpath.commonpath(['/qualification/dest', output]) == '/qualification/dest'
        except SuspiciousOperation:
            output, status, contained = None, 'rejected', None
        rows.append(dict(name=name, status=status, resolved=output, component_contained=contained))
    assert rows[0]['component_contained'] is True
    assert rows[1]['status'] == 'rejected'
    assert rows[2]['component_contained'] is False
    report = {'classification': 'DEVELOPMENTAL_REFERENCE_COUNTEREXAMPLE', 'case': case['cve_id'],
              'dataset_sha256': hashlib.sha256(data).hexdigest(), 'results': rows,
              'scope': 'Pinned fixed helper, POSIX path arithmetic only. No filesystem extraction or present-version claim.'}
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
