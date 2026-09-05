import pytest
import reconstruct_deepcanvassing_measured_responses_20260905 as r
from audit_deepcanvassing_measured_responses_20260905 import summarize, item_statistics, departure_grids


def fixture():
    bases, late = [], {}
    for arm in (0, 1):
        for j in (0, 1):
            key = f'{arm*2+j+1:024x}'
            row = {'prolific_pid': key, 'condition': 'treatment' if arm else 'control', 'Experimental_Condition': arm}
            later = {}
            for definitions in r.SCALES.values():
                for item, reverse in definitions:
                    for wave in ('baseline', 'immediate'):
                        value = 20 + arm*10
                        row[r.source_column(wave,item)] = 100-value if reverse else value
                    value = 30 + arm*15
                    later[r.source_column('recontact',item)] = 100-value if reverse else value
            bases.append(row)
            if j == 0:
                late[key] = later
    return bases, late


def test_exact_missing_wave_bounds_and_selected_description():
    bases, late = fixture()
    records, _ = r.reconstruct_rows(bases, late)
    result = summarize(records)
    for scale in r.SCALES:
        assert result[scale]['change_in_arm_contrast_bounds'] == pytest.approx((-52.5,47.5))
    chosen = result['prejudice']['joint_complete_responder_description']
    assert chosen['counts'] == {0:1,1:1}
    assert chosen['change'] == 5


def test_itemwise_departures_and_reverse_coding():
    bases, late = fixture()
    grids = departure_grids(item_statistics(bases,late))
    for scale in r.SCALES:
        assert grids[scale]['change_grid'][3][3] == pytest.approx(5)
        assert grids[scale]['change_grid'][4][2] == pytest.approx(-5)
        assert len(grids[scale]['change_grid']) == 7


def test_missing_entire_item_arm_skips_sensitivity_only():
    bases, late = fixture()
    cells = item_statistics(bases,late)
    cells[('prejudice',0,'recontact','living')] = {'observed_sum':0,'observed_count':0,'missing_count':2}
    assert departure_grids(cells)['prejudice']['skipped']
    assert not departure_grids(cells)['policy']['skipped']
