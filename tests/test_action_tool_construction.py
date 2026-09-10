import copy
import json
from prepare_action_tool_learning import construct


def example():
    return ({'id': 'test', 'question': [[{'role': 'user', 'content': 'Use 8.'}]],
             'function': [{'name': 'f', 'parameters': {'properties': {'n': {'type': 'integer'}}, 'required': ['n']}}]},
            {'ground_truth': [{'f': {'n': [8]}}]})


def test_exactly_one_correct_action_and_alias_equivalence():
    row, key = example()
    numeric_ranks = set()
    for i in range(32):
        row['id'] = f'test_{i}'
        value, error = construct(row, key)
        assert error is None
        actions = [json.loads(pair[0]) for pair in value['aliases']]
        assert sum(a['arguments']['n'] == 8 for a in actions) == 1
        assert actions[value['target']]['arguments']['n'] == 8
        assert all(json.loads(a) == json.loads(b) for a, b in value['aliases'])
        numeric_ranks.add(value['target_numeric_rank'])
    assert numeric_ranks == {0, 1, 2, 3}


def test_reject_function_name_mismatch_and_ambiguous_keys():
    row, key = example()
    bad = copy.deepcopy(key); bad['ground_truth'] = [{'other': {'n': [8]}}]
    assert construct(row, bad)[1] == 'function_name_mismatch'
    key['ground_truth'][0]['f']['n'] = [8, 9]
    assert construct(row, key)[1] == 'ambiguous_or_structured_required_value'


def test_do_not_ignore_explicit_parameter_constraints():
    row, key = example(); row['function'][0]['parameters']['properties']['n']['maximum'] = 8
    assert construct(row, key)[1] == 'no_unconstrained_integer'
