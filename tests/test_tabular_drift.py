import copy
import numpy as np
import pytest
from research_pilots.tabular_drift import acquisition,project_joint,build_inputs,validate,Predictor,experiment
from research_pilots.budget import admission


def test_label_independent_change_is_drift_not_information():
    scores=acquisition([.5,.5],[[.5,.5]],[[[.99,.01]],[[.99,.01]]])
    assert scores['raw']>.6 and abs(scores['information'])<1e-12 and abs(scores['projected'])<1e-12
    assert scores['raw']==pytest.approx(scores['drift'])


def test_coherent_beta_bernoulli_has_no_drift():
    # Uniform Beta prior: observing failure/success yields next success probability 1/3 or 2/3.
    scores=acquisition([.5,.5],[[.5,.5]],[[[2/3,1/3]],[[1/3,2/3]]])
    assert abs(scores['drift'])<1e-12 and scores['raw']==pytest.approx(scores['information'])
    assert scores['projected']==pytest.approx(scores['information'])


def test_projection_respects_both_marginals_and_preserves_odds_ratio():
    r=np.array([[.3,.1],[.2,.4]])
    out=project_joint(r,np.array([.2,.8]),np.array([.7,.3]))
    assert np.allclose(out.sum(0),[.7,.3]) and np.allclose(out.sum(1),[.2,.8])
    assert out[0,0]*out[1,1]/out[0,1]/out[1,0]==pytest.approx(6.)


def test_unobserved_evaluation_labels_cannot_change_acquisition():
    c=build_inputs()['cases'][0]
    baseline=experiment(c,Predictor('logistic'),rounds=1)
    altered=copy.deepcopy(c)
    for i in c['evaluation']:altered['y'][i]=(altered['y'][i]+1)%3
    changed=experiment(altered,Predictor('logistic'),rounds=1)
    for method in baseline['arms']:
        assert baseline['arms'][method][0]['selected']==changed['arms'][method][0]['selected']


def test_budget_declines_new_job_without_prescribing_interruption():
    from datetime import datetime,timezone
    now=datetime(2026,9,7,tzinfo=timezone.utc)
    report=admission('tabular_drift',now.isoformat(),48,now=now)
    assert not report['admit'] and report['mid_run_timeout'] is False
    report=admission('tabular_drift',now.isoformat(),45,now=now)
    assert report['admit'] and report['required_hours_with_margin']==3.5


def test_split_integrity_rejects_rehashed_overlap():
    from research_pilots.common import digest
    data=build_inputs();c=data['cases'][0]
    c['target'][0]=c['initial'][0];data['case_hashes'][c['id']]=digest(c)
    with pytest.raises(ValueError,match='leakage'):validate(data)


@pytest.mark.parametrize('kind',['pilot','hindsight'])
def test_default_launchers_never_pass_an_experiment_timeout(tmp_path,monkeypatch,kind):
    import sys
    from datetime import datetime,timezone
    from types import SimpleNamespace
    import launch_research_pilot as pilot
    import launch_hindsight_calibration as hindsight
    module=pilot if kind=='pilot' else hindsight
    args=['launcher']+(['compensation'] if kind=='pilot' else ['--learning','unused.json'])
    args+=['--out',str(tmp_path/'run'),'--snapshot','unused','--allocation-start-utc',
           datetime.now(timezone.utc).isoformat(),'--previous-h200-hours','0']
    monkeypatch.setattr(sys,'argv',args)
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    observed=[]
    monkeypatch.setattr(module,'supervise',lambda command,seconds,env,log:observed.append(seconds) or 0)
    with pytest.raises(SystemExit) as result:module.main()
    assert result.value.code==0 and observed==[None]
