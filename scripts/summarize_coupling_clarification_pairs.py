"""Display both mandatory pairs without pooling away configuration interaction."""
import argparse
import json
from pathlib import Path
from interaction_sprint.byte_clock_coupling import digest


def summarize(small,large,out):
    reports={name:json.loads(path.read_text()) for name,path in [('small',small),('large',large)]}
    expected={'small':{'qwen3_06b','smollm2_360m'},'large':{'qwen3_4b','smollm2_17b'}}
    ids=[];result={}
    for name,report in reports.items():
        assert report['runtime']['outputs']==10240
        cases={c['id'] for c in report['per_case']};assert len(cases)==40;ids.append(cases)
        assert set(report['policies']['independent']['by_model'])==expected[name]
        for policy in report['policies'].values():
            assert all(m['n']==1280 for m in policy['by_model'].values())
        candidate=report['policies']['byte_hierarchical']['f1']
        comparisons={}
        for baseline in ('independent','token_clock','byte_clock'):
            b=report['policies'][baseline]['f1']
            comparisons[baseline]={**report['comparisons']['f1']['hierarchical vs '+baseline],
                'twice_covariance_gain':2*(candidate['covariance']-b['covariance']),
                'marginal_variance_shift':(b['variance_a']+b['variance_b'])-(candidate['variance_a']+candidate['variance_b'])}
        result[name]={'manifest_sha':report['manifest_sha'],'comparisons_f1':comparisons,
                      'policies':report['policies']}
    assert ids[0]==ids[1]
    result['scope']='Both fixed model pairs mandatory. No pooled pass, no post-hoc pair selection. Interpret covariance and cost, retain previous DEV outcomes.'
    result['inputs']={'small_sha':digest(small),'large_sha':digest(large),'source_sha':digest(__file__)}
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k in ('scope','inputs')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('small',type=Path);p.add_argument('large',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();summarize(a.small,a.large,a.out)
