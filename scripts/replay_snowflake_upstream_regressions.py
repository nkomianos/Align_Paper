"""Run the pinned component's upstream assertion functions with isolated imports.

Only the package import is replaced by the already hash-checked component class.
No assertion, fixture value, test body, or mock decorator is modified.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from qualify_snowflake_redaction_components import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((args.source / 'DOWNLOAD.json').read_text())
    for name in ('vulnerable', 'fixed', 'fixed_tests'):
        assert hashlib.sha256((args.source / (name + '.py')).read_bytes()).hexdigest() == pins[name]['sha256']
    tree = ast.parse((args.source / 'fixed_tests.py').read_text())
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'snowflake.connector.secret_detector']
    assert len(imports) == 1
    assert [(item.name, item.asname) for item in imports[0].names] == [('SecretDetector', None)]
    tree.body.remove(imports[0])
    code = compile(tree, 'pinned_upstream_tests', 'exec')
    fixed = load(args.source / 'fixed.py', 'fixed_pattern_source')
    results = {}
    for arm in ('vulnerable', 'private_only', 'token_only', 'fixed'):
        cls = load(args.source / ('fixed.py' if arm == 'fixed' else 'vulnerable.py'), arm + '_upstream')
        if arm == 'private_only':
            cls.PRIVATE_KEY_PATTERN = fixed.PRIVATE_KEY_PATTERN
        if arm == 'token_only':
            cls.CONNECTION_TOKEN_PATTERN = fixed.CONNECTION_TOKEN_PATTERN
        namespace = {'SecretDetector': cls, '__name__': 'isolated_upstream_tests'}
        exec(code, namespace)
        results[arm] = {}
        for name, function in sorted(namespace.items()):
            if name.startswith('test_') and callable(function):
                try:
                    function()
                except Exception as exc:
                    results[arm][name] = type(exc).__name__
                else:
                    results[arm][name] = 'PASS'
    report = {'classification': 'DEVELOPMENTAL_UPSTREAM_COMPONENT_REGRESSION',
              'case': 'CVE-2024-49750', 'results': results, 'source_pins': pins,
              'method': 'Direct zero-argument upstream test-function execution; exact bodies and mock decorators, package import replaced.',
              'limitations': 'Not full-package pytest or container validation. Bundled test_mask_token covers multiple defects in one test function.'}
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
