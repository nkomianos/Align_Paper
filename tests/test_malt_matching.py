import pytest
from audit_malt_matching import audit


def row(i,task,model,label):
    return dict(run_id=i,task_id=task,model=model,labels=[label],manually_reviewed=True,run_source='unprompted')


def test_overlap_and_joint_partition_are_distinct():
    rows=[row(1,'a/1','m','normal'),row(2,'a/1','m','bypass_constraints'),
          row(3,'b/1','m','normal'),row(4,'b/1','n','normal'),
          row(5,'c/1','z','normal')]
    r=audit(rows)
    assert r['task_model']['maximum_disjoint_pairs']==1
    assert sorted(c['runs'] for c in r['joint_disjoint_components'])==[1,4]
    assert sum(c['positives'] for c in r['joint_disjoint_components'])==1


def test_reject_duplicates_and_exclude_unreviewed():
    r=row(1,'a/1','m','normal')
    with pytest.raises(ValueError):audit([r,r])
    r['manually_reviewed']=False
    assert audit([r])['joint_disjoint_components']==[]
