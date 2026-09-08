from datetime import datetime, timezone
import copy
import pytest
import torch
from interaction_sprint.hindsight_calibration import partition, summarize, qualified_teacher, route, ARMS
from interaction_sprint.hindsight_neural_anchor import reverse_kl_per_example
from scripts.launch_hindsight_calibration import remaining_seconds


def rows():
    return [dict(id=f'{b}-{i}',base_id=f'b{b}',label_rotation=i,old_target='ABCD'[i]) for b in range(630) for i in range(4)]


def test_partition_invariant_to_input_order_and_complete_rotations():
    a=partition(rows()); b=partition(list(reversed(rows())))
    assert a==b
    assert len(a['train'])==512 and len(a['holdout'])==128
    assert not {r['base_id'] for r in a['train']} & {r['base_id'] for r in a['holdout']}
    for start in range(0,512,16):
        assert sorted(r['old_target'] for r in a['train'][start:start+16])==sorted('ABCD'*4)


def test_bad_rotation_rejected():
    data=rows();data[0]['label_rotation']=1
    with pytest.raises(ValueError):partition(data)


def test_teacher_detached_and_reverse_kl_direction():
    s=torch.tensor([[1.,-1.]],requires_grad=True);t=torch.tensor([[-.2,.2]],requires_grad=True)
    loss=reverse_kl_per_example(s,t).sum()
    expected=(s.softmax(-1)*(s.log_softmax(-1)-t.log_softmax(-1).detach())).sum()
    assert torch.allclose(loss,expected)
    loss.backward();assert s.grad is not None and t.grad is None


@pytest.mark.parametrize('objective',['sft','reverse_kl'])
def test_microbatch_gradient_equals_full_batch(objective):
    torch.manual_seed(7);x=torch.randn(16,3,dtype=torch.float64);teacher=torch.randn(16,4)
    w=torch.randn(3,4,dtype=torch.float64,requires_grad=True);other=w.detach().clone().requires_grad_()
    def loss(logits,start=0):
        if objective=='sft':return torch.nn.functional.cross_entropy(logits,torch.arange(start,start+len(logits))%4)
        return reverse_kl_per_example(logits,teacher[start:start+len(logits)]).mean()
    loss(x@w).backward()
    for i in range(0,16,2):(loss(x[i:i+2]@other,i)/8).backward()
    assert torch.allclose(w.grad,other.grad,atol=1e-7,rtol=1e-6)


def metrics(nll=2.,prob=.2,conditional=.25):
    return {'nll':nll,'probability':prob,'conditional_probability':conditional,
            'mean_position_range':.1,'choice_mass':.9,'per_rotation_accuracy':dict.fromkeys('0123',1.)}


def test_routes_never_qualify_paper():
    base=metrics(); good=metrics(1.,.4,.5);bad=metrics(3.,.5,.6)
    teachers={a:metrics(1.,.8,.9) for a in ARMS}
    expected=['STOP_SUPERVISED_ACQUISITION_UNQUALIFIED','INVESTIGATE_DISTILLATION_OBJECTIVE_OR_TRANSFER',
              'INVESTIGATE_UPDATING_TEACHER','CALIBRATION_PASSED_NOVELTY_AND_EXTERNAL_TASK_STILL_REQUIRED']
    for k in range(4):
        final={a:good if i<k else bad for i,a in enumerate(ARMS)}
        result=route(base,final,teachers)
        assert result['decision']==expected[k] and result['paper_green_light'] is False


def test_teacher_gate_checks_every_rotation():
    m=metrics(1.,.8,.9);assert qualified_teacher(m)
    m['per_rotation_accuracy']['3']=.8749;assert not qualified_teacher(m)


def test_budget_includes_idle_and_prior_usage():
    now=datetime(2026,9,5,12,tzinfo=timezone.utc)
    assert remaining_seconds('2026-09-05T11:00:00Z',48.5,now)==1800
    assert remaining_seconds('2026-09-05T11:00:00Z',0,now)==49*3600
    for bad in (-1,50,float('nan')):
        with pytest.raises(ValueError):remaining_seconds('2026-09-05T11:00:00Z',bad,now)
    with pytest.raises(ValueError):remaining_seconds('2026-09-05T13:00:00Z',0,now)


def test_summary_rejects_pseudoreplicated_incomplete_base():
    scored=[dict(base_id='b',label_rotation=i,nll=1.,probability=.3,conditional_probability=.4,correct=1,choice_mass=.75) for i in range(4)]
    assert summarize(scored)['bases']==1
    with pytest.raises(ValueError):summarize(scored+scored[:1])


def test_adapter_reset_and_optimizer_reset():
    from latent_contract.sender_update import LoRALinear,adapter_state,load_adapter
    model=torch.nn.Sequential(LoRALinear(torch.nn.Linear(3,4),rank=2,alpha=2))
    initial=adapter_state(model);x=torch.ones(2,3);before=model(x).detach().clone()
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=.01)
    model(x).sum().backward();optimizer.step()
    assert not torch.equal(before,model(x))
    load_adapter(model,initial)
    assert torch.equal(before,model(x))
    assert not torch.optim.AdamW(params,lr=.01).state


def test_verifier_rejects_rehashed_wrong_context(tmp_path,monkeypatch):
    import json
    import scripts.verify_hindsight_calibration as verifier
    from interaction_sprint.hindsight_execution_integrity import seal_artifacts
    selected={'train':rows()[:4],'holdout':rows()[4:8]}
    monkeypatch.setattr(verifier,'prepare',lambda unused:selected)
    def write(name,obj):(tmp_path/name).write_text(json.dumps(obj))
    write('INPUTS.json',selected);write('RUNTIME.json',{'answer_ids':[0,1,2,3]})
    raw={'ids':[r['id'] for r in selected['holdout']], 'teacher_context':False,
         'student_gradient':False,'logits':torch.zeros(4,5)}
    torch.save(raw,tmp_path/'forward_000001.pt')
    scored=[]
    for i,r in enumerate(selected['holdout']):
        scored.append({**{k:r[k] for k in ('id','base_id','label_rotation')},'forward_file':'forward_000001.pt',
            'forward_row':i,'nll':float(torch.tensor(5.).log()),'probability':.2,
            'conditional_probability':.25,'correct':int(i==0),'choice_mass':.8})
    write('baseline_student.json',{'rows':scored,'summary':summarize(scored)})
    raw_teacher={**raw,'teacher_context':True}
    torch.save(raw_teacher,tmp_path/'forward_000002.pt')
    teachers=[{**r,'forward_file':'forward_000002.pt'} for r in scored]
    write('baseline_teacher.json',{'rows':teachers,'summary':summarize(teachers)})
    write('RESULT.json',{'decision':'STOP_INVALID_INITIAL_TEACHER','paper_green_light':False})
    seal_artifacts(tmp_path)
    assert verifier.verify(tmp_path,None)['saved_arithmetic_and_routing_verified']
    raw_teacher['teacher_context']=False;torch.save(raw_teacher,tmp_path/'forward_000002.pt')
    (tmp_path/'MANIFEST.json').unlink();seal_artifacts(tmp_path)
    with pytest.raises(ValueError,match='context'):verifier.verify(tmp_path,None)
