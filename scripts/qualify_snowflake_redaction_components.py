"""CPU developmental check of one natural two-defect redaction component.

Uses downloaded, hash-checked source and synthetic non-secret strings. This is
one source task, not independent model evidence or a full connector assessment.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SecretDetector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    pins = json.loads((args.source / 'DOWNLOAD.json').read_text())
    for name in ('vulnerable', 'fixed'):
        data = (args.source / (name + '.py')).read_bytes()
        assert hashlib.sha256(data).hexdigest() == pins[name]['sha256']
    # Restrict the before/after difference to the two known pattern assignments.
    trees = [ast.parse((args.source / (name + '.py')).read_text()) for name in ('vulnerable', 'fixed')]
    names = {'PRIVATE_KEY_PATTERN', 'CONNECTION_TOKEN_PATTERN'}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                node.value = ast.Constant(value='PATTERN_OMITTED')
    assert ast.dump(trees[0]) == ast.dump(trees[1])
    fixed = load(args.source / 'fixed.py', 'fixed_detector')
    payload = 'ABCD' * 16
    key = '-----BEGIN RSA PRIVATE KEY-----\n' + payload + '\n-----END RSA PRIVATE KEY-----'
    token = 'A' * 12 + '.' + 'B' * 12 + '.' + 'C' * 12
    results = {}
    for arm in ('vulnerable', 'private_only', 'token_only', 'fixed'):
        cls = load(args.source / ('fixed.py' if arm == 'fixed' else 'vulnerable.py'), arm + '_detector')
        if arm == 'private_only':
            cls.PRIVATE_KEY_PATTERN = fixed.PRIVATE_KEY_PATTERN
        if arm == 'token_only':
            cls.CONNECTION_TOKEN_PATTERN = fixed.CONNECTION_TOKEN_PATTERN
        key_result = cls.mask_secrets(key)
        token_result = cls.mask_secrets('token: ' + token)
        result = {
            'private_key_fully_redacted': payload not in key_result[1] and key_result[2] is None,
            'dotted_token_fully_redacted': token_result == (True, 'token: ****', None),
            'benign_retained': cls.mask_secrets('ordinary event: ready') == (False, 'ordinary event: ready', None),
            'legacy_token_redacted': cls.mask_secrets('token: ' + 'A' * 12) == (True, 'token: ****', None),
        }
        assert result['private_key_fully_redacted'] == (arm in ('private_only', 'fixed'))
        assert result['dotted_token_fully_redacted'] == (arm in ('token_only', 'fixed'))
        assert result['benign_retained'] and result['legacy_token_redacted']
        results[arm] = result
    report = {'classification': 'DEVELOPMENTAL_NATURAL_COMPONENT_QUALIFICATION',
              'case': 'CVE-2024-49750', 'results': results, 'source_pins': pins,
              'independent_task_count': 1,
              'limitations': 'Synthetic input probes; no full connector, container PoC, model generation, or exhaustive regression suite.'}
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
