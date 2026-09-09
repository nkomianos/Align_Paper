"""Developmental, network-free replay of the released Requests auth component.

Methods are extracted from the pinned PatchEval source snapshot. Request objects
are local stubs; this is not a full HTTP integration test or a model experiment.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace
from urllib.parse import urlparse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    data = args.dataset.read_bytes()
    assert hashlib.sha256(data).hexdigest() == '65e873923d738221389ed32408c4f5fa75882ea07609c500761207d013da0f62'
    case = next(x for x in json.loads(data) if x['cve_id'] == 'CVE-2018-18074')
    old = ast.parse(textwrap.dedent(case['vul_func'][0]['snippet'])).body
    new = [ast.parse(textwrap.dedent(x['snippet'])).body[0] for x in case['fix_func']]
    examples = {
        'same_origin': ('https://example.invalid/a', 'https://example.invalid/b', False),
        'different_host': ('https://example.invalid/a', 'https://other.invalid/a', True),
        'standard_upgrade': ('http://example.invalid/a', 'https://example.invalid/a', False),
        'scheme_downgrade': ('https://example.invalid/a', 'http://example.invalid/a', True),
        'port_change': ('https://example.invalid:443/a', 'https://example.invalid:8443/a', True),
    }
    results = {}
    for arm in ('vulnerable', 'port_only', 'scheme_only', 'fixed'):
        methods = copy.deepcopy(old if arm == 'vulnerable' else new)
        if arm in ('port_only', 'scheme_only'):
            helper = next(n for n in methods if n.name == 'should_strip_auth')
            final = helper.body[-1]
            assert isinstance(final, ast.Return) and isinstance(final.value, ast.BoolOp)
            assert len(final.value.values) == 2
            final.value = final.value.values[0 if arm == 'port_only' else 1]
        module = ast.Module(body=methods, type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {'urlparse': urlparse}
        exec(compile(module, 'released_requests_methods', 'exec'), namespace)
        cls = type('IsolatedAuth', (), {n.name: namespace[n.name] for n in methods})
        obj = cls()
        obj.trust_env = False
        results[arm] = {}
        for name, (before, after, expected_strip) in examples.items():
            request = SimpleNamespace(url=after, headers={'Authorization': 'SYNTHETIC_MARKER', 'Accept': 'text/plain'})
            response = SimpleNamespace(request=SimpleNamespace(url=before))
            obj.rebuild_auth(request, response)
            stripped = 'Authorization' not in request.headers
            assert request.headers['Accept'] == 'text/plain'
            results[arm][name] = stripped == expected_strip
        assert all(results[arm][k] for k in ('same_origin', 'different_host', 'standard_upgrade'))
        assert results[arm]['scheme_downgrade'] == (arm in ('scheme_only', 'fixed'))
        assert results[arm]['port_change'] == (arm in ('port_only', 'fixed'))
    report = {'classification': 'DEVELOPMENTAL_NATURAL_COMPONENT_QUALIFICATION',
              'case': case['cve_id'], 'patch_url': case['patch_url'], 'results': results,
              'dataset_sha256': hashlib.sha256(data).hexdigest(),
              'independent_task_count': 1,
              'limitations': 'Stubbed request objects; no network, netrc, full-package regressions, or model outputs. Partial helpers are controlled ablations of the released fix.'}
    with args.report.open('x') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
