import json
from datetime import datetime,timezone,timedelta
from pathlib import Path
import pytest
from research_pilots.data import build,validate
from research_pilots.clara import worlds,probabilities,credible_set,certified
from research_pilots.common import seal,check_manifest,binary_threshold
from research_pilots.compensation import route


def test_frozen_data_truth_balance_and_overlap():
    data=validate(build())
    for split in ('train','dev','holdout'):
        rows=data[split]
        assert sum(r['target'] for r in rows)==len(rows)//2
        assert len({r['allowed'] for r in rows})==len(rows)//2
        assert len({r['base_id'] for r in rows})==len(rows)//2
    data['dev'][0]['target']^=1
    with pytest.raises(ValueError):validate(data)


def test_joint_set_sound_for_arbitrary_query_selection():
    ws=worlds(4)
    for correlation in (0.,.5,1.):
        p=probabilities(ws,(0,1,0,1),.05,correlation)
        joint=credible_set(ws,p,.1)
        assert sum(v for w,v in zip(ws,p) if w in joint)>=.9-1e-12
        for op in ('and','or'):
            assert certified((op,[0,1]),joint)==certified((op,[0,1,0,1]),joint)


def test_rehashed_manifest_cannot_hide_source_label_edit(tmp_path):
    (tmp_path/'INPUTS.json').write_text(json.dumps(build()))
    seal(tmp_path);check_manifest(tmp_path)
    data=json.loads((tmp_path/'INPUTS.json').read_text());data['train'][0]['target']^=1
    (tmp_path/'INPUTS.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):check_manifest(tmp_path)
    (tmp_path/'MANIFEST.json').unlink();seal(tmp_path)
    with pytest.raises(ValueError):validate(data)


def test_budget_counts_paid_idle_and_invalid_accounting():
    from launch_research_pilot import budget_seconds
    now=datetime(2026,9,6,tzinfo=timezone.utc)
    start=(now-timedelta(hours=1)).isoformat()
    assert budget_seconds('compensation',start,48.5,now)==1800
    for spent in (-1,float('nan'),50):
        with pytest.raises(ValueError):budget_seconds('reference',start,spent,now)


def test_invalid_recovery_never_becomes_scientific_negative():
    scores={k:{'accuracy':1.,'choice_mass':1.} for k in
            ('base','edit','recovered','withdrawn','d0','random_withdrawn','base_utility','recovered_utility')}
    assert route(scores)=='STOP_INVALID_ACQUISITION'
    scores['edit']['accuracy']=.5;scores['recovered']['accuracy']=.5
    assert route(scores)=='STOP_INVALID_RECOVERY'


def test_edit_reset_does_not_accumulate_roundoff_or_adapter_changes(tmp_path):
    import torch
    from research_pilots.neural import Backend
    from latent_contract.sender_update import LoRALinear
    b=Backend.__new__(Backend);b.torch=torch
    b.site=LoRALinear(torch.nn.Linear(6,4,bias=False),rank=2,alpha=4)
    b.base_weight=b.site.base.weight.detach().clone()
    b.initial={n:p.detach().clone() for n,p in b.site.named_parameters() if p.requires_grad}
    d=torch.tensor([1.,0.,0.,0.])
    b.set_edit(d,1.);edited=b.site.base.weight.detach().clone()
    for _ in range(10):b.set_edit(None,0);b.set_edit(d,1.)
    assert torch.equal(edited,b.site.base.weight)
    with torch.no_grad():b.site.b.add_(1.)
    b.reset();assert torch.count_nonzero(b.site.b)==0
    b.set_edit(None,0);assert torch.equal(b.base_weight,b.site.base.weight)


def test_monitor_rejects_unreviewed_and_family_leakage():
    from research_pilots.monitor import validate as vm
    rows=[]
    for split in ('calibration','dev','holdout'):
        for i in range(30):
            rows.append(dict(id=f'{split}{i}',family=split,model_lineage='fixture',split=split,
                             task=split,trace=str(i),label=int(i>=20),reviewed=True,prompted=False,source_sha256='0'*64))
    vm(rows)
    rows[0]['reviewed']=False
    with pytest.raises(ValueError):vm(rows)
    rows[0]['reviewed']=True;rows[0]['family']='dev'
    with pytest.raises(ValueError):vm(rows)


def test_calibration_threshold_ties_are_conservative():
    rows=[{'label':0,'probability':.3} for _ in range(20)]
    assert binary_threshold(rows)==.3


def test_verifier_rejects_rehashed_wrong_prompt(tmp_path):
    from research_pilots.common import write,score_rows
    from verify_research_pilot import verify
    data=build();write(tmp_path/'INPUTS.json',data)
    write(tmp_path/'PROVENANCE.json',{'pilot':'compensation'})
    records=[]
    for i,row in enumerate(data['dev']):
        raw={k:row[k] for k in ('id','target','prompt')}
        raw.update(choice_logits=[1.,0.],logsumexp=2.,gradient=False)
        name=f'forward_{i}.json';write(tmp_path/name,raw)
        records.append(dict(raw,forward_file=name))
    block={'rows':records,'summary':score_rows(records)}
    write(tmp_path/'base.json',block)
    write(tmp_path/'RESULT.json',{'decision':'STOP_INVALID_BASE_CAPABILITY','paper_green_light':False})
    seal(tmp_path);assert verify(tmp_path)['verified']
    records[0]['prompt']='Different task despite same id'
    (tmp_path/'base.json').write_text(json.dumps(block))
    (tmp_path/records[0]['forward_file']).write_text(json.dumps({k:v for k,v in records[0].items() if k!='forward_file'}))
    (tmp_path/'MANIFEST.json').unlink();seal(tmp_path)
    with pytest.raises(ValueError,match='source row'):verify(tmp_path)


def test_twelve_hour_allocation_admission(monkeypatch):
    from research_pilots.budget import admission
    monkeypatch.setenv('RESEARCH_ALLOCATION_HOURS','12')
    now=datetime.now(timezone.utc)
    from datetime import timedelta
    result=admission('compensation',(now-timedelta(hours=8)).isoformat(),0,now=now)
    assert result['remaining_hours']==4 and not result['admit']
    assert result['budget_target_hours']==12 and not result['mid_run_timeout']


def test_training_updates_only_adapter_and_resets_optimizer(tmp_path):
    import torch
    from research_pilots.neural import Backend
    from latent_contract.sender_update import LoRALinear
    b=Backend.__new__(Backend);b.torch=torch;b.out=tmp_path;b.ids=[0,1]
    torch.manual_seed(4)
    b.site=LoRALinear(torch.nn.Linear(3,2,bias=False),rank=2,alpha=4)
    base=b.site.base.weight.detach().clone()
    b.initial={n:p.detach().clone() for n,p in b.site.named_parameters() if p.requires_grad}
    b.forward=lambda row,grad=False:(b.site(torch.tensor([1.,2.,3.])),{})
    rows=[{'id':str(i),'target':0} for i in range(4)]
    b.train(rows,'first',steps=2)
    first=b.site.b.detach().clone()
    assert torch.count_nonzero(first)>0 and torch.equal(base,b.site.base.weight)
    b.reset();b.train(rows,'second',steps=2)
    assert torch.equal(first,b.site.b)


@pytest.mark.parametrize('failure',['execution','verifier_timeout'])
def test_suite_stops_after_failure_and_writes_receipt(tmp_path,monkeypatch,failure):
    import os
    import sys
    import subprocess
    from types import SimpleNamespace
    import launch_research_suite as suite
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    out=tmp_path/'suite'
    monkeypatch.setattr(suite,'os',SimpleNamespace(name='posix',environ=os.environ,path=os.path,pathsep=os.pathsep))
    monkeypatch.setattr(sys,'argv',['suite','--out',str(out),'--snapshot',str(snapshot),
        '--allocation-start-utc',datetime.now(timezone.utc).isoformat(),'--previous-h200-hours','0'])
    calls=[]
    def fake_run(command,**kwargs):
        calls.append(command)
        if 'verify_research_pilot.py' in command[1] and failure=='verifier_timeout':
            raise subprocess.TimeoutExpired(command,1)
        code=1 if 'launch_research_pilot.py' in command[1] and failure=='execution' else 0
        return SimpleNamespace(returncode=code,stdout='',stderr='')
    monkeypatch.setattr(suite.subprocess,'run',fake_run)
    with pytest.raises(SystemExit):suite.main()
    result=json.loads((out/'SUITE_RESULT.json').read_text())
    assert len(result['outcomes'])==1 and result['outcomes'][0]['status'].startswith('STOP_')
    assert not any('compensation' in c for c in calls)
