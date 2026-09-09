from copy import deepcopy

from validator_monoculture.prompts import parse_test_completion
from validator_monoculture.transport_v2 import verifier_prompt_v2


def test_public_metadata_removed_without_mutation_or_spec_source_leakage():
    task = dict(task_id='dev', cwe_id='CWE-20', entrypoint='f', signature='f(x)',
                public_spec='Return x.', vulnerable_source='SECRET_SOURCE',
                public_cases=[dict(args=[True], kwargs={}, expected=None, case_id='EXAMPLE_ID')])
    original = deepcopy(task)
    prompt = verifier_prompt_v2(task, None, requested_tests=2)
    assert task == original
    assert 'SECRET_SOURCE' not in prompt and 'EXAMPLE_ID' not in prompt
    assert '"args": [true]' in prompt and '"expected": null' in prompt
    assert 'exactly three keys' in prompt
    assert r'\u0000' in prompt and r'\\' in prompt


def test_documented_json_literals_parse_with_unchanged_parser():
    raw = r'{"tests":[{"args":[true,false,null,"\\","\u0000","\n"],"kwargs":{},"expected":null}]}'
    tests = parse_test_completion(raw, requested_tests=1)
    assert tests[0]['args'] == [True, False, None, '\\', '\x00', '\n']
