import torch
from interaction_sprint.opdlm_onpolicy import task,score,choose,summarize,POLICIES

def test_fixed_tasks_and_strict_scoring():
    assert task('subtract',31,16)[1]==15
    assert task('multiply',23,7)[1]==161
    assert score('15',15)=={'strict_parse':True,'strict_correct':True,'last_number_correct':True}
    assert not score('The answer is 15.',15)['strict_correct']
    assert score('The answer is 15.',15)['last_number_correct']
    assert not score('15.2',15)['last_number_correct']
    assert not score('five5.',5)['strict_correct']
    assert not score('Wrong',3)['last_number_correct']

def test_selection_excludes_committed_seed():
    z=torch.tensor([[0.,9.,0.],[0.,0.,8.],[0.,1.,2.],[0.,2.,1.]])
    assert choose(z,torch.tensor([1,0,0,0]),0)==(1,2)

def test_summary_separates_task_metrics_from_audit():
    records=[{'policy':p,'score':score('3',3),'terminated':True,'events':[{'revised':False}],'calls':7 if p=='baseline' else 4} for p in POLICIES]
    s=summarize(records,[])
    assert s['n_tasks']==1 and s['paired_audits']==0
    assert s['policies']['baseline']['forward_calls']==7
