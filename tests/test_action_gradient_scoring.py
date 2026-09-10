from types import SimpleNamespace
import torch
from run_action_gradient_diagnostic import score_choices, contexts
import json


class PositionModel:
    def __call__(self, input_ids, attention_mask, use_cache):
        assert not use_cache
        positions = torch.arange(input_ids.shape[1], dtype=torch.float64)[None, :, None]
        vocabulary = torch.arange(7, dtype=torch.float64)[None, None, :]
        logits = (positions+1) * vocabulary / 13 + input_ids[:, :, None] * vocabulary / 17
        return SimpleNamespace(logits=logits)


def test_scoring_uses_causal_positions_and_excludes_padding():
    prefix, suffixes = [1, 3], [[2, 6], [4, 5, 6]]
    actual = score_choices(PositionModel(), prefix, suffixes, 0, 'cpu')
    expected = []
    for suffix in suffixes:
        ids = prefix+suffix
        total = 0.
        for target_position in range(len(prefix), len(ids)):
            position = target_position-1
            logits = torch.arange(7, dtype=torch.float64)*((position+1)/13 + ids[position]/17)
            total += logits.log_softmax(0)[ids[target_position]]
        expected.append(total)
    assert torch.allclose(actual.double(), torch.stack(expected), atol=1e-6)


def test_aliases_are_distinct_strings_with_identical_typed_actions():
    rows = contexts()
    assert {row['target'] for row in rows} == {0, 1, 2, 3}
    for row in rows:
        for first, second in row['aliases']:
            assert first != second and json.loads(first) == json.loads(second)
