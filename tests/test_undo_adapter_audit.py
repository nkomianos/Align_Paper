import pytest
import torch
from scripts.audit_undo_adapter_updates import compare


def test_numerical_change_and_no_change():
    a = {'a': torch.tensor([1., 2.])}
    assert compare(a, a)['changed_elements'] == 0
    result = compare(a, {'a': torch.tensor([4., 6.])})
    assert result['l2_change'] == 5 and result['changed_elements'] == 2


def test_bad_checkpoint_rejected():
    with pytest.raises(ValueError):
        compare({'a': torch.tensor([1.])}, {'a': torch.tensor([float('nan')])})
