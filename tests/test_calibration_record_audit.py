import json
from types import SimpleNamespace
import pytest
import torch


@pytest.mark.parametrize('corruption',[None,'tokens','loss','teacher_context','optimizer'])
def test_saved_record_audit_detects_corruption(tmp_path,monkeypatch,corruption):
    import transformers
    import audit_calibration_forward_records as checker
    import interaction_sprint.hindsight_execution_integrity as integrity
    from interaction_sprint.hindsight_pahf_reduced import HINDSIGHT_BLOCK
    monkeypatch.setattr(integrity,'verify_manifest',lambda root:None)
    class Tokenizer:
        pad_token_id=0
        backend_tokenizer=SimpleNamespace(to_str=lambda:'backend')
        def encode(self,x,**kwargs):return ['ABCD'.index(x)]
        def apply_chat_template(self,messages,**kwargs):return messages[0]['content']
        def __call__(self,texts,**kwargs):
            ids=torch.tensor([[len(s),1] for s in texts])
            return {'input_ids':ids,'attention_mask':torch.ones_like(ids)}
    tokenizer=Tokenizer()
    monkeypatch.setattr(transformers.AutoTokenizer,'from_pretrained',lambda *a,**k:tokenizer)
    write=lambda name,value:(tmp_path/name).write_text(json.dumps(value))
    row={'id':'r0','prompt':'question','old_target':'A','delayed_expression_followup':'evidence'}
    write('CONFIG.json',{'steps':1,'effective_batch':1,'microbatch':1,'common_task_prefix':'prefix'})
    write('INPUTS.json',{'train':[row],'holdout':[]})
    write('RUNTIME.json',{'answer_ids':[0,1,2,3]})
    (tmp_path/'tokenizer.json').write_text('backend')
    target=torch.tensor([[1.,2.,3.,4.]])
    student=torch.tensor([[4.,3.,2.,1.]])
    def raw(teacher,gradient,logits):
        text='prefixquestion'+(HINDSIGHT_BLOCK.format(follow_up='evidence') if teacher else '')
        return {'ids':['r0'],'rendered':[text],'teacher_context':teacher,'student_gradient':gradient,
                'logits':logits,**tokenizer([text])}
    values=[raw(True,False,target),raw(False,True,student),raw(False,True,student),
            raw(True,False,target),raw(False,True,student)]
    if corruption=='tokens':values[1]['input_ids'][0,0]+=1
    if corruption=='teacher_context':values[4]['teacher_context']=True
    for i,value in enumerate(values,1):torch.save(value,tmp_path/f'forward_{i:06d}.pt')
    write('baseline_train_teacher.json',{'rows':[{'id':'r0','forward_file':'forward_000001.pt','forward_row':0}]})
    logp=student.log_softmax(-1);logq=target.log_softmax(-1)
    ce=float(torch.nn.functional.cross_entropy(student,torch.tensor([0])))
    kl=float((logp.exp()*(logp-logq)).sum())
    for arm,loss in zip(('supervised','frozen_teacher','current_teacher'),(ce,kl,kl)):
        write(f'{arm}_step_001.json',{'loss':loss+(1 if corruption=='loss' else 0)})
        torch.save({'state':{}},tmp_path/f'{arm}_initial_optimizer.pt')
        torch.save({'state':{0:{'step':2 if corruption=='optimizer' else 1}}},tmp_path/f'{arm}_optimizer.pt')
    write('RESULT.json',{'decision':'developmental'})
    if corruption:
        with pytest.raises(ValueError):checker.audit(tmp_path,tmp_path)
    else:
        result=checker.audit(tmp_path,tmp_path)
        assert result['gradient_batches']==3 and result['forward_records']==5
