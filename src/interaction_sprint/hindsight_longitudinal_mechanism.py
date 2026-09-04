"""Persistent-user coupling test with direct-report Brier-gradient learning.

Finite stochastic model, NOT language-model training or a causal estimator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

SPEC=dict(seed=90416001,seeds=16,users_per_seed=1024,rounds=64,
    initial_fidelities=[.5,.8],learning_rates=[.1,.5],influence_rates=[.2,.5,.8],
    regimes=['stationary','expression','persistent'],couplings=['closed','open','frozen'],
    checkpoints=[0,1,8,32,64],scope='Finite stochastic mechanism, no neural training',
    learner='p <- (1-eta)*p + eta*report; exact SGD on half squared probability error',
    familywise_arithmetic_check_alpha=.001,automatic_expansion=False,paper_green_light=False)


def configs():
    return [dict(regime=g,rho=r,q=q,eta=e) for q in SPEC['initial_fidelities']
        for e in SPEC['learning_rates'] for g in SPEC['regimes']
        for r in ([0.] if g=='stationary' else SPEC['influence_rates'])]


def exact(q,eta,rho,regime,coupling,rounds):
    """Means relative to each user's Z0; linear symmetric-transition model."""
    p,z=q,1.
    for _ in range(rounds):
        a=p if coupling=='closed' else q
        if regime=='persistent':z=(1-rho)*z+rho*a;report=z
        elif regime=='expression':report=(1-rho)+rho*a
        elif regime=='stationary':report=1.
        else:raise ValueError('Unknown regime')
        if coupling!='frozen':p=(1-eta)*p+eta*report
    return dict(baseline_fidelity=p,state_retention=z)


def simulate(z0,ua,uc,*,q,eta,rho,regime,coupling,checkpoints=(0,1,8,32,64)):
    if regime not in SPEC['regimes'] or coupling not in SPEC['couplings']:
        raise ValueError('Unknown regime/coupling')
    if not all(0<=v<=1 for v in (q,eta,rho)) or ua.shape!=uc.shape or ua.shape[1]!=len(z0):
        raise ValueError('Invalid inputs')
    p0=np.where(z0,q,1-q).astype(float);p=p0.copy();z=z0.copy();trace=[];first=None
    def measure(t,report=None,action=None):
        trace.append(dict(round=t,baseline_fidelity=float(np.where(z0,p,1-p).mean()),
            current_satisfaction=float(np.where(z,p,1-p).mean()),state_retention=float((z==z0).mean()),
            observed_agreement=None if report is None else float((report==action).mean())))
    if 0 in checkpoints:measure(0)
    for t in range(len(ua)):
        action=ua[t]<(p if coupling=='closed' else p0)
        if regime=='persistent':z=np.where(uc[t]<rho,action,z);report=z
        elif regime=='expression':report=np.where(uc[t]<rho,action,z0)
        else:report=z0
        if first is None:first=report.copy()
        if coupling!='frozen':p=(1-eta)*p+eta*report
        if t+1 in checkpoints:measure(t+1,report,action)
    return p,z,trace,first


def run(root):
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def write(name,value):
        with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    write('spec.json',SPEC);cells=configs();write('configs.json',cells)
    (root/'runner_source.py').write_bytes(Path(__file__).read_bytes())
    rng=np.random.default_rng(SPEC['seed']);n=SPEC['users_per_seed'];seeds=SPEC['seeds'];rounds=SPEC['rounds']
    zs=np.stack([rng.permutation(np.arange(n)%2).astype(bool) for _ in range(seeds)])
    uas=rng.random((seeds,rounds,n));ucs=rng.random((seeds,rounds,n))
    np.savez_compressed(root/'exogenous.npz',z0=zs,action_uniforms=uas,transition_uniforms=ucs)
    records=[];finalp=[];finalz=[];firsts={};contrasts=[]
    for ci,c in enumerate(cells):
        for si in range(seeds):
            outputs={}
            for coupling in SPEC['couplings']:
                p,z,trace,first=simulate(zs[si],uas[si],ucs[si],**c,coupling=coupling,checkpoints=SPEC['checkpoints'])
                outputs[coupling]=(p,z,trace)
                finalp.append(p);finalz.append(z)
                records.append(dict(config=ci,seed=si,coupling=coupling,trace=trace,
                    first_report_sha256=hashlib.sha256(first.tobytes()).hexdigest()))
                firsts[(c['regime'],c['rho'],c['q'],c['eta'],si,coupling)]=first
            assert np.array_equal(outputs['open'][1],outputs['frozen'][1])
            assert np.array_equal(outputs['frozen'][0],np.where(zs[si],c['q'],1-c['q']))
            contrasts.append(dict(config=ci,seed=si,
                fidelity_closed_minus_open=outputs['closed'][2][-1]['baseline_fidelity']-outputs['open'][2][-1]['baseline_fidelity'],
                retention_closed_minus_open=outputs['closed'][2][-1]['state_retention']-outputs['open'][2][-1]['state_retention']))
    for key,value in firsts.items():
        if key[0]=='expression':assert np.array_equal(value,firsts[('persistent',)+key[1:]])
    np.savez_compressed(root/'final_states.npz',p=np.stack(finalp),z=np.stack(finalz))
    write('records.json',records);write('paired_contrasts.json',contrasts)
    summaries=[];errors=[]
    for ci,c in enumerate(cells):
        arm={}
        for coupling in SPEC['couplings']:
            selected=[r for r in records if r['config']==ci and r['coupling']==coupling]
            analytic=exact(**c,coupling=coupling,rounds=rounds)
            empirical={k:float(np.mean([r['trace'][-1][k] for r in selected])) for k in
                ('baseline_fidelity','state_retention','current_satisfaction','observed_agreement')}
            errors.extend(abs(empirical[k]-analytic[k]) for k in analytic)
            arm[coupling]=dict(exact=analytic,empirical=empirical)
        diffs=[r['fidelity_closed_minus_open'] for r in contrasts if r['config']==ci]
        summaries.append(dict(config=ci,**c,arms=arm,
            exact_fidelity_contrast=arm['closed']['exact']['baseline_fidelity']-arm['open']['exact']['baseline_fidelity'],
            paired_fidelity_mean=float(np.mean(diffs)),paired_seed_se=float(np.std(diffs,ddof=1)/np.sqrt(seeds))))
    comparisons=len(cells)*3*2
    epsilon=float(np.sqrt(np.log(2*comparisons/SPEC['familywise_arithmetic_check_alpha'])/(2*seeds*n)))
    result=dict(status='COMPLETED_FINITE_MECHANISM_NOT_NEURAL_EVIDENCE',summaries=summaries,
        max_empirical_exact_error=max(errors),simultaneous_hoeffding_tolerance=epsilon,
        all_means_within_tolerance=max(errors)<=epsilon,
        minimum_exact_closed_open_fidelity_contrast=min(r['exact_fidelity_contrast'] for r in summaries),
        first_round_expression_persistent_matched=True,open_frozen_states_matched=True,
        elapsed=time.monotonic()-start,paper_green_light=False)
    write('RESULT.json',result)
    write('MANIFEST.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();run(a.root)
