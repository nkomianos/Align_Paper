import ast
from collections import Counter
from run_unexplored_screens import action_rows, feedback_rows, jobs, summarize


def test_finite_oracle_matches_actual_trusted_functions():
    # Only locally generated, fixed source is executed here; no model code input.
    for row in feedback_rows():
        functions=[]
        for code in (row['good'],row['bad']):
            ast.parse(code)
            ns={}
            exec(code, {'int':int,'min':min,'max':max,'abs':abs}, ns)
            functions.append(ns['f'])
        for x,y,z in row['oracle']:
            assert functions[0](x)==y
            assert functions[1](x)==z
        x,y=row['public']
        assert functions[0](x)==functions[1](x)==y


def test_balanced_orders_and_no_oracle_in_prompt():
    work=list(jobs({'actions':action_rows(),'feedback':feedback_rows()}))
    assert len(work)==3456
    assert len({r['id'] for r in work})==len(work)
    for kind in ('action','feedback'):
        counts=Counter(r['target'] for r in work if r['kind']==kind)
        if kind=='feedback': assert counts[0]==counts[1]
    assert all('oracle' not in r for r in work)


def test_alias_mass_is_sum_not_max():
    r=dict(kind='action',mode='aliases',target=0,
           logits=[-1,-.8,-9,-9,-1,-9,-9,-9],logsumexp=1)
    result=summarize([r])
    assert result['action/aliases/best_string']['mean']==0
    assert result['action/aliases/marginal']['mean']==1
