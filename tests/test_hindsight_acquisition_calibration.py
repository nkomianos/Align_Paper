from collections import Counter
from interaction_sprint.hindsight_acquisition_calibration import dataset


def test_split_schedule_and_no_label_drift():
    rows,schedule=dataset();lookup={r['id']:r for r in rows}
    assert len(rows)==96 and len(schedule)==96
    assert Counter(r['split'] for r in rows)=={'train':64,'eval':32}
    assert len({r['prompt'] for r in rows})==96
    assert all(len(b)==len(set(b))==8 for b in schedule)
    exposure=Counter(i for b in schedule for i in b)
    assert set(exposure)=={r['id'] for r in rows if r['split']=='train'}
    assert set(exposure.values())=={12}
    for base in {r['base_id'] for r in rows}:
        assert len({r['target'] for r in rows if r['base_id']==base})==1
