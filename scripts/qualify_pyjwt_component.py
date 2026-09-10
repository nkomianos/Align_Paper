"""Pinned public-source, local-only HMAC key admission qualification.

No token forgery or service access: actual historical prepare_key methods receive
fresh synthetic public keys and ordinary HMAC secrets. One component, not a corpus.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import textwrap
import urllib.request

FIX = '9c528670c455b8d948aff95ed50e22940d1ad3fc'
DATA_SHA = '65e873923d738221389ed32408c4f5fa75882ea07609c500761207d013da0f62'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Align-Paper-source-audit'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def normalized(node):
    node = copy.deepcopy(node)
    for child in ast.walk(node):
        if isinstance(child, (ast.FunctionDef, ast.ClassDef)) and child.body:
            first = child.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                first.value.value = textwrap.dedent(first.value.value).strip()
    return ast.dump(node, include_attributes=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    raw = a.dataset.read_bytes()
    assert digest(raw) == DATA_SHA
    case = next(x for x in json.loads(raw) if x['cve_id'] == 'CVE-2022-29217')
    a.out.mkdir(parents=True, exist_ok=False)
    commit_url = f'https://api.github.com/repos/jpadilla/pyjwt/commits/{FIX}'
    commit_raw = fetch(commit_url)
    (a.out/'commit.json').write_bytes(commit_raw)
    commit = json.loads(commit_raw)
    assert commit['sha'] == FIX and len(commit['parents']) == 1
    parent = commit['parents'][0]['sha']
    receipt = {'commit.json': {'url': commit_url, 'sha256': digest(commit_raw)}}
    for label, revision in [('vulnerable', parent), ('fixed', FIX)]:
        for path in ['jwt/algorithms.py', 'jwt/utils.py', 'tests/test_algorithms.py', 'tests/test_utils.py']:
            url = f'https://raw.githubusercontent.com/jpadilla/pyjwt/{revision}/{path}'
            data = fetch(url)
            name = label + '_' + path.replace('/', '_')
            (a.out/name).write_bytes(data)
            receipt[name] = {'url': url, 'sha256': digest(data)}
    (a.out/'DOWNLOAD.json').write_text(json.dumps(receipt, indent=2))
    methods = {}
    for label, field in [('vulnerable', 'vul_func'), ('fixed', 'fix_func')]:
        tree = ast.parse((a.out/f'{label}_jwt_algorithms.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HMACAlgorithm')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_key')
        snippet = ast.parse(textwrap.dedent(case[field][0]['snippet'])).body[0]
        assert normalized(method) == normalized(snippet), 'benchmark/upstream method mismatch'
        methods[label] = method

    # Import only the pure format helpers and their constants from actual source.
    utils = ast.parse((a.out/'fixed_jwt_utils.py').read_text())
    selected = []
    constants = {'_PEMS', '_PEM_RE', '_CERT_SUFFIX', '_SSH_PUBKEY_RC', '_SSH_KEY_FORMATS'}
    for node in utils.body:
        if isinstance(node, ast.FunctionDef) and node.name in {'force_bytes', 'is_pem_format', 'is_ssh_key'}:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets):
            selected.append(node)
    namespace = {'re': re, 'Union': __import__('typing').Union}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'pinned_pyjwt_utils', 'exec'), namespace)
    class InvalidKeyError(Exception):
        pass
    namespace['InvalidKeyError'] = InvalidKeyError

    from cryptography.hazmat.primitives.asymmetric import ed25519, ec
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    ed_public = ed25519.Ed25519PrivateKey.generate().public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    ec_public = ec.generate_private_key(ec.SECP256R1()).public_key().public_bytes(Encoding.OpenSSH, PublicFormat.OpenSSH)
    # Generated keys are test fixtures only; private bytes are never serialized.
    probes = {'ed25519_public': ed_public, 'ecdsa_public': ec_public,
              'legacy_rsa_marker': b'ssh-rsa SYNTHETIC',
              'legacy_pem_marker': b'-----BEGIN PUBLIC KEY-----\nSYNTHETIC\n-----END PUBLIC KEY-----',
              'ordinary_bytes_secret': b'ordinary-synthetic-hmac-secret-123',
              'ordinary_string_secret': 'ordinary-synthetic-hmac-secret-456'}
    outputs = {}
    for arm in ['vulnerable', 'ed25519_only', 'ecdsa_only', 'fixed']:
        method = copy.deepcopy(methods['fixed' if arm == 'fixed' else 'vulnerable'])
        if arm.endswith('_only'):
            assignment = next(n for n in method.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'invalid_strings')
            assert isinstance(assignment.value, ast.List)
            assignment.value.elts.append(ast.Constant(b'ssh-ed25519' if arm == 'ed25519_only' else b'ecdsa-sha2-nistp256'))
        module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
        scope = namespace.copy()
        exec(compile(module, 'pinned_hmac_prepare_key', 'exec'), scope)
        result = {}
        for name, key in probes.items():
            try:
                prepared = scope['prepare_key'](None, key)
                assert prepared == (key.encode() if isinstance(key, str) else key)
                result[name] = 'accepted_unchanged'
            except InvalidKeyError:
                result[name] = 'rejected'
        outputs[arm] = result
        assert result['ed25519_public'] == ('rejected' if arm in {'ed25519_only', 'fixed'} else 'accepted_unchanged')
        assert result['ecdsa_public'] == ('rejected' if arm in {'ecdsa_only', 'fixed'} else 'accepted_unchanged')
        assert all(result[n] == 'rejected' for n in ['legacy_rsa_marker', 'legacy_pem_marker'])
        assert all(result[n] == 'accepted_unchanged' for n in ['ordinary_bytes_secret', 'ordinary_string_secret'])
    report = {'classification': 'DEVELOPMENTAL_NATURAL_COMPONENT_QUALIFICATION', 'case': case['cve_id'],
              'fixed_revision': FIX, 'vulnerable_revision': parent, 'dataset_sha256': DATA_SHA,
              'independent_task_count': 1, 'results': outputs,
              'fixtures': {k: v.decode() if isinstance(v, bytes) else v for k, v in probes.items()},
              'limitations': 'Single key-admission mechanism, two independent omitted format branches. Not two tasks, full JWT verification, full upstream tests, model outputs, or an admitted sixteen-task corpus.'}
    (a.out/'RESULT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='fixtures'}, indent=2))


if __name__ == '__main__':
    main()
