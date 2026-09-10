from check_monitor_recovery_candidates import identity_matches


def test_identical_text_hash_does_not_join_different_trials_or_models():
    expected = dict(trial_name='a', model='m', agent='h', trial_id='uuid-a')
    rows = [expected, {**expected, 'trial_name': 'b'}, {**expected, 'model': 'other'},
            {**expected, 'agent': 'other'}, {**expected, 'trial_id': 'uuid-b'}]
    assert identity_matches(rows, 'a', 'm', ['h'], 'uuid-a') == [expected]


def test_duplicate_source_rows_remain_ambiguous():
    row = dict(trial_name='a', model='m', agent='h', trial_id='uuid-a')
    assert len(identity_matches([row, dict(row)], 'a', 'm', ['h'], 'uuid-a')) == 2
