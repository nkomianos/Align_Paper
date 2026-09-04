"""Hash/state/trajectory replay plus independent matrix-power mean check."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from interaction_sprint import hindsight_longitudinal_mechanism as runner


def matrix_mean(c,coupling,t):
    q,e,r=c['q'],c['eta'],c['rho']
    if coupling=='frozen':e=0.
    if c['regime']=='persistent':
        if coupling=='closed':
            m=np.array([[1-r,r,0],[e*(1-r),1-e+e*r,0],[0,0,1.]])
        else:
            m=np.array([[1-r,0,r*q],[e*(1-r),1-e,e*r*q],[0,0,1.]])
    else:
        influence=r if c['regime']=='expression' else 0.
        if coupling=='closed':coef=1-e+e*influence;offset=e*(1-influence)
        else:coef=1-e;offset=e*((1-influence)+influence*q)
        m=np.array([[1.,0,0],[0,coef,offset],[0,0,1.]])
    z,p,_=np.linalg.matrix_power(m,t)@np.array([1.,q,1.])
    return dict(baseline_fidelity=float(p),state_retention=float(z))


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert set(manifest)=={p.name for p in root.iterdir() if p.is_file()}-{'MANIFEST.json'}
    for name,digest in manifest.items():
        p=(root/name).resolve();assert p.parent==root.resolve()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    assert (root/'runner_source.py').read_bytes()==Path(runner.__file__).read_bytes()
    spec=json.loads((root/'spec.json').read_text());assert spec==runner.SPEC
    cells=json.loads((root/'configs.json').read_text());assert cells==runner.configs() and len(cells)==28
    # Materialize once: repeated NpzFile indexing re-decompresses a whole array.
    with np.load(root/'exogenous.npz',allow_pickle=False) as archive:
        draws={k:archive[k] for k in archive.files}
    with np.load(root/'final_states.npz',allow_pickle=False) as archive:
        state={k:archive[k] for k in archive.files}
    records=json.loads((root/'records.json').read_text());contrasts=json.loads((root/'paired_contrasts.json').read_text())
    assert len(records)==28*16*3 and len(contrasts)==28*16
    assert state['p'].shape==state['z'].shape==(len(records),1024)
    assert draws['action_uniforms'].shape==draws['transition_uniforms'].shape==(16,64,1024)
    assert draws['z0'].shape==(16,1024) and (draws['z0'].sum(1)==512).all()
    for key in ('action_uniforms','transition_uniforms'):
        assert ((draws[key]>=0)&(draws[key]<1)).all()
    replay={};firsts={}
    for i,rec in enumerate(records):
        ci,si,arm=rec['config'],rec['seed'],rec['coupling'];c=cells[ci]
        assert (ci,si,arm) not in replay
        p,z,trace,first=runner.simulate(draws['z0'][si],draws['action_uniforms'][si],draws['transition_uniforms'][si],
            **c,coupling=arm,checkpoints=spec['checkpoints'])
        assert np.array_equal(p,state['p'][i]) and np.array_equal(z,state['z'][i])
        assert trace==rec['trace'] and hashlib.sha256(first.tobytes()).hexdigest()==rec['first_report_sha256']
        replay[(ci,si,arm)]=(p,z,trace)
        firsts[(c['regime'],c['rho'],c['q'],c['eta'],si,arm)]=first
    for key,value in firsts.items():
        if key[0]=='expression':assert np.array_equal(value,firsts[('persistent',)+key[1:]])
    for r in contrasts:
        ci,si=r['config'],r['seed'];closed=replay[(ci,si,'closed')];opened=replay[(ci,si,'open')];frozen=replay[(ci,si,'frozen')]
        assert np.array_equal(opened[1],frozen[1])
        assert r['fidelity_closed_minus_open']==closed[2][-1]['baseline_fidelity']-opened[2][-1]['baseline_fidelity']
        assert r['retention_closed_minus_open']==closed[2][-1]['state_retention']-opened[2][-1]['state_retention']
    result=json.loads((root/'RESULT.json').read_text());errors=[];mean_deltas=[]
    for s in result['summaries']:
        ci=s['config'];c=cells[ci]
        assert all(c[k]==s[k] for k in c)
        for arm,values in s['arms'].items():
            analytic=matrix_mean(c,arm,64)
            for k,v in analytic.items():
                assert abs(v-values['exact'][k])<1e-12
                measured=float(np.mean([replay[(ci,si,arm)][2][-1][k] for si in range(16)]))
                assert measured==values['empirical'][k]
                errors.append(abs(measured-values['exact'][k]))
            for k in ('current_satisfaction','observed_agreement'):
                assert values['empirical'][k]==float(np.mean([replay[(ci,si,arm)][2][-1][k] for si in range(16)]))
        expected=s['arms']['closed']['exact']['baseline_fidelity']-s['arms']['open']['exact']['baseline_fidelity']
        assert s['exact_fidelity_contrast']==expected;mean_deltas.append(expected)
        diffs=[r['fidelity_closed_minus_open'] for r in contrasts if r['config']==ci]
        assert s['paired_fidelity_mean']==float(np.mean(diffs))
        assert s['paired_seed_se']==float(np.std(diffs,ddof=1)/4)
    assert len(result['summaries'])==28 and len(set(s['config'] for s in result['summaries']))==28
    tolerance=float(np.sqrt(np.log(2*28*3*2/.001)/(2*16*1024)))
    assert result['simultaneous_hoeffding_tolerance']==tolerance
    assert result['max_empirical_exact_error']==max(errors)
    assert result['all_means_within_tolerance']==(max(errors)<=tolerance)
    assert result['minimum_exact_closed_open_fidelity_contrast']==min(mean_deltas)
    return dict(status='FULL_SAVED_DRAW_STATE_TRAJECTORY_REPLAY_PLUS_MATRIX_MEAN_CHECK',
        scope='Finite-model replay using runner transition code; mean equations independently checked by matrix powers; not neural evidence',
        manifest_files=len(manifest),replayed_trajectories=len(records),
        max_empirical_exact_error=max(errors),simultaneous_tolerance=tolerance,
        exact_negative_contrasts=sum(d < -1e-12 for d in mean_deltas),numpy=np.__version__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert a.root.resolve() not in a.out.resolve().parents
    report=verify(a.root)
    with a.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(report,indent=2))
