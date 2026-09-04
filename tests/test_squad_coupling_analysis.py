from scripts.analyze_squad_coupling import moments,ratio_ci


def test_paired_variance_covariance_identity():
    a=[0,1,0,1];same=moments(a,a);opposite=moments(a,[1,0,1,0])
    assert same['variance_difference']==0 and same['covariance']>0
    assert opposite['variance_difference']==4/3 and opposite['covariance']<0


def test_undefined_ratio_not_reported_as_zero():
    r=ratio_ci([1,1],[0,0]);assert r['ratio'] is None and r['question_bootstrap_descriptive_95'] is None
    r=ratio_ci([.5,.5],[1,1]);assert r['ratio']==.5 and r['question_bootstrap_descriptive_95']==[.5,.5]
