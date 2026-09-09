"""Attest benchmark methods to pinned upstream and replay four pure unit tests."""
import argparse
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    pins = json.loads((args.source / 'DOWNLOAD.json').read_text())
    for name in ('vulnerable', 'fixed', 'fixed_tests'):
        assert hashlib.sha256((args.source / (name + '.py')).read_bytes()).hexdigest() == pins[name]['sha256']
    case = next(x for x in json.loads(args.dataset.read_text(encoding='utf8')) if x['cve_id'] == 'CVE-2018-18074')
    matches = []
    for name, key in [('vulnerable', 'vul_func'), ('fixed', 'fix_func')]:
        tree = ast.parse((args.source / (name + '.py')).read_text(encoding='utf8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SessionRedirectMixin')
        for snippet in case[key]:
            node = ast.parse(textwrap.dedent(snippet['snippet'])).body[0]
            actual = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == node.name))
            # Dataset extraction dedents documentation; preserve all executable AST.
            for function in (node, actual):
                if isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
                    function.body[0].value.value = inspect.cleandoc(function.body[0].value.value)
            assert ast.dump(actual) == ast.dump(node)
            matches.append(name + ':' + node.name)
    helper = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'should_strip_auth')
    namespace = {'urlparse': urlparse}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), 'pinned_helper', 'exec'), namespace)
    session = type('IsolatedSession', (), {'should_strip_auth': namespace['should_strip_auth']})
    test_tree = ast.parse((args.source / 'fixed_tests.py').read_text(encoding='utf8'))
    tests = [n for n in ast.walk(test_tree) if isinstance(n, ast.FunctionDef) and n.name.startswith('test_should_strip_auth')]
    assert len(tests) == 4 and all(not n.decorator_list and len(n.args.args) == 1 for n in tests)
    namespace = {'requests': SimpleNamespace(Session=session)}
    exec(compile(ast.Module(body=tests, type_ignores=[]), 'pinned_pure_tests', 'exec'), namespace)
    results = {}
    for test in tests:
        namespace[test.name](None)
        results[test.name] = 'PASS'
    with args.report.open('x') as f:
        json.dump({'classification': 'DEVELOPMENTAL_UPSTREAM_ATTESTATION',
                   'method_matches': matches, 'pure_upstream_tests': results, 'source_pins': pins,
                   'normalization': 'Docstring indentation only; executable AST identical.',
                   'limitations': 'Isolated method, not full HTTP or package tests. Upstream port-change test also changes scheme.'}, f, indent=2)
    print(json.dumps({'method_matches': matches, 'pure_tests': results}))


if __name__ == '__main__':
    main()
