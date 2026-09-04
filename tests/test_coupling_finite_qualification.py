from interaction_sprint.coupling_finite_qualification import qualify, check_independent_events
import pytest


def test_exact_marginals_and_segmentation_counterexample():
    rows = {x['case']: x for x in qualify()['cases']}
    for name in ('segmentation', 'silent_clock'):
        assert rows[name]['identical_rendered_laws']
        assert rows[name]['native_mismatch'] == '1/2'
        assert rows[name]['complete_byte_mismatch'] == '0'
        assert rows[name]['implementation_mismatch_witness_seeds']
    assert rows['explicit_eos']['native_mismatch'] == '0'
    assert rows['binding_cap']['identical_rendered_laws'] is False
    assert all(r['native_marginals_exact'] for r in rows.values())


def test_refuse_false_independence():
    with pytest.raises(AssertionError):
        check_independent_events({(0, (1, 2)), (0, (2, 3))})
