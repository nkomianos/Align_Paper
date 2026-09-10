"""CPU replay and conditional-bank variance audit; never emits a paper go."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import numpy as np
from run_reasoning_bank import answer
from run_unexplored_screens import sha,dump


def read_bank(root):
    for name,h in json.loads((root/'MANIFEST.json').read_text()).items():
        assert Path(name).name==name and sha(root/name)==h
    data=json.loads((root/'INPUTS.json').read_text())
    byid={r['id']:r for r in data}
    prefixes=json.loads((root/'PREFIXES.json').read_text())
    byprefix={(r['base'],r['prefix_index']):r for r in prefixes}
    rows=[json.loads(x) for x in (root/'ROLLOUTS.jsonl').read_bytes().split(b'\n') if x]
    expected={(r['id'],j,k) for r in data for j in (0,1) for k in range(8)}
    keys=[(r['base'],r['prefix_index'],r['sample']) for r in rows]
    assert len(keys)==len(set(keys)) and set(keys)==expected
    for r in rows:
        p=byprefix[(r['base'],r['prefix_index'])]
        target=byid[r['base']]['target']
        assert r['target']==target and r['split']==byid[r['base']]['split']
        assert r['length']==len(r['ids']) and 0<r['length']<=768
        assert answer(p['text']+r['completion'])==r['parsed_answer']
        assert int(r['parsed_answer']==target)==r['reward']
        assert len(r['score_projection'])==len(r['score_projection_128'])==10
        assert np.isfinite(r['score_projection']).all()
    return rows,byid,byprefix


def features(r,byid,prefixes):
    text=prefixes[(r['base'],r['prefix_index'])]['text']
    return [1.,len(byid[r['base']]['question'])/500,len(text)/200,
            len(re.findall(r'\d+',text))/10,text.count('\n')/5,text.count('=')/5]


def variance(g,p):
    return float((((1-p)/p)[:,None]*g*g).sum()/len(g)**2)


def analyze(root):
    rows,byid,prefixes=read_bank(root)
    # Already-complete prefixes cannot test selective continuation or rank utility.
    eligible=[r for r in rows if not r['prefix_ended'] and not r['prefix_has_answer']]
    cal=[r for r in eligible if r['split']=='calibration']
    dev=[r for r in eligible if r['split']=='dev']
    report={'integrity':'VERIFIED_SAVED_ROWS; sampling-logit derivatives require separate neural replay',
        'n_rows':len(rows),'n_eligible':len(eligible),'cal_questions':len({r['base'] for r in cal}),
        'dev_questions':len({r['base'] for r in dev}),
        'reward_mean':float(np.mean([r['reward'] for r in rows])),
        'answer_parse_rate':float(np.mean([r['parsed_answer'] is not None for r in rows])),
        'horizon_rate':float(np.mean([not r['eos'] for r in rows])),
        'mean_length':float(np.mean([r['length'] for r in rows])),
        'raw_sha256':sha(root/'ROLLOUTS.jsonl')}
    if len({r['base'] for r in cal})<4 or len({r['base'] for r in dev})<8:
        report['route']='INVALID_BANK_CAPACITY';return report
    xc=np.asarray([features(r,byid,prefixes) for r in cal])
    xd=np.asarray([features(r,byid,prefixes) for r in dev])
    gc=np.asarray([(r['reward']-.5)*np.array(r['score_projection']) for r in cal])
    gd=np.asarray([(r['reward']-.5)*np.array(r['score_projection']) for r in dev])
    cc=np.maximum(1,np.array([r['length']-128 for r in cal]))
    yd=np.log(1e-8+(gc*gc).sum(1)/cc)
    penalty=np.eye(xc.shape[1])*10;penalty[0,0]=1e-8
    beta=np.linalg.solve(xc.T@xc+penalty,xc.T@yd)
    # Fit once on calibration only; DEV rewards cannot affect continuation odds.
    score_cal=np.exp(np.clip(xc@beta/2,-10,10))
    score_dev=np.exp(np.clip(xd@beta/2,-10,10))
    scale=.25/max(1e-8,float(score_cal.mean()))
    probabilities=np.clip(scale*score_dev,.1,1)
    lengths=np.array([r['length'] for r in dev],dtype=float)
    short=lengths<=128
    inclusion=np.where(short,1,probabilities)
    initial=np.minimum(lengths,128)
    remaining=np.maximum(lengths-128,0)
    cost=float((initial+probabilities*remaining).sum())
    full_cost=float(lengths.sum())
    full_inclusion=np.full(len(dev),cost/full_cost)
    uniform_p=(cost-initial.sum())/max(1,float(remaining.sum()))
    uniform=np.where(short,1,uniform_p)
    v_adapt=variance(gd,inclusion);v_uniform=variance(gd,uniform);v_full=variance(gd,full_inclusion)
    target=gd.mean(0)
    gshort=np.asarray([((r['reward'] if r['length']<=128 else 0)-.5)*np.array(r['score_projection_128']) for r in dev])
    masked=gd*short[:,None]
    uncorrected=gd*inclusion[:,None]
    report['censoring']={'target_gradient_projection':target.tolist(),'fit_coefficients':beta.tolist(),
        'adaptive_mse':v_adapt,'uniform_continuation_mse':v_uniform,'fewer_full_rollouts_mse':v_full,
        'expected_continuation_tokens':cost,'full_continuation_tokens':full_cost,
        'calibration_tokens':sum(r['length'] for r in cal),
        'adaptive_probability_range':[float(probabilities.min()),float(probabilities.max())],
        'matched_uniform_probability':float(uniform_p),
        'hard_cutoff_bias_squared':float(((gshort.mean(0)-target)**2).sum()),
        'masking_bias_squared':float(((masked.mean(0)-target)**2).sum()),
        'adaptive_uncorrected_bias_squared':float(((uncorrected.mean(0)-target)**2).sum()),
        'scope':'Analytic Bernoulli design MSE conditional on this finite bank, 10 digit biases. Not optimizer learning or full-parameter gradients.',
        'cost_limits':'Expected generated-token matching; excludes prefill/hashing and does not equate token cost with GPU seconds. Calibration charged and reported separately.'}
    report['route']='DEV_VARIANCE_SIGNAL_REQUIRES_REPLICATION' if v_adapt<.8*min(v_uniform,v_full) else 'STOP_NO_DECISIVE_VARIANCE_ADVANTAGE'
    return report


def compare(a,b):
    ra,da,pa=read_bank(a);rb,db,pb=read_bank(b)
    assert da==db and pa==pb
    means=[]
    grouped=[]
    for rows in (ra,rb):
        d=defaultdict(list)
        for r in rows:d[(r['base'],r['prefix_index'])].append(r['reward'])
        grouped.append(d)
    for base,row in da.items():
        p0,p1=pa[(base,0)],pa[(base,1)]
        valid=not(p0['ended'] or p1['ended'] or answer(p0['text']) or answer(p1['text'])) and p0['text']!=p1['text']
        rates=[[float(np.mean(g[(base,j)])) for j in (0,1)] for g in grouped]
        delta=[v[1]-v[0] for v in rates]
        means.append({'base':base,'split':row['split'],'eligible':bool(valid),'rates':rates,
            'prefix_contrasts':delta,'observed_rank_reversal':bool(delta[0]*delta[1]<0),
            'large_reversal':bool(delta[0]*delta[1]<0 and min(abs(x) for x in delta)>=.5)})
    dev=[r for r in means if r['eligible'] and r['split']=='dev']
    return {'classification':'DEVELOPMENTAL; 8 draws per prefix is not a precise value estimate',
        'rows':means,'eligible_dev_questions':len(dev),'observed_reversals':sum(r['observed_rank_reversal'] for r in dev),
        'large_reversals':sum(r['large_reversal'] for r in dev),
        'route':'REPLICATE_WITH_FRESH_CONTINUATIONS' if sum(r['large_reversal'] for r in dev)>=4 else 'NO_DECISIVE_RANK_REVERSAL_SCREEN',
        'limitations':'Native interfaces and Qwen-origin prefixes may interact with family. No PRM trained or adapted.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--other',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=compare(a.root,a.other) if a.other else analyze(a.root)
    dump(a.out,result);print(json.dumps(result))
