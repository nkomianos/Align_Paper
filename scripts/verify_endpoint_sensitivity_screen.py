"""Independent tensor-to-metric replay for the developmental endpoint screen."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from run_unexplored_screens import sha,dump


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    for name,h in manifest.items():
        assert Path(name).name==name and sha(root/name)==h
    rows=json.loads((root/'ROWS.json').read_text())
    assert len(rows)==96
    keys={(r['seed'],r['k']) for r in rows}
    assert len(keys)==96 and keys=={(s,k) for s in range(2026091100,2026091124) for k in (4,12,20,28)}
    for i in range(24):
        t=torch.load(root/f'tensors_{i:02d}.pt',map_location='cpu',weights_only=True)
        for r in [r for r in rows if r['seed']==2026091100+i]:
            delta=t[f'altered_{r["k"]}'].numpy().astype('float64')-t['reference'].numpy().astype('float64')
            residual=t[f'residual_{r["k"]}'].numpy().astype('float64')
            assert np.isclose(np.mean(delta**2),r['endpoint_mse'],rtol=1e-5,atol=1e-9)
            assert np.isclose(np.mean(residual**2),r['local_residual_mse'],rtol=1e-5,atol=1e-9)
            assert r['split']==('calibration' if i<12 else 'dev')
    result=json.loads((root/'ANALYSIS.json').read_text())
    cal=[r for r in rows if r['split']=='calibration'];dev=[r for r in rows if r['split']=='dev']
    x=np.array([r['features'] for r in cal]);mean=x.mean(0);std=x.std(0)
    std[std<1e-8]=1;mean[0]=0;std[0]=1;z=(x-mean)/std
    penalty=10*np.eye(x.shape[1]);penalty[0,0]=1e-8
    y=np.log(np.array([r['endpoint_mse'] for r in cal])+1e-12)
    beta=np.linalg.solve(z.T@z+penalty,z.T@y)
    np.testing.assert_allclose(beta,result['calibration_coefficients'],rtol=1e-5,atol=1e-8)
    times={k:np.mean([r['endpoint_mse'] for r in cal if r['k']==k]) for k in (4,12,20,28)}
    damages={k:[] for k in ('learned','timestep','local_oracle','endpoint_oracle')}
    for seed in sorted({r['seed'] for r in dev}):
        group=[r for r in dev if r['seed']==seed]
        functions={'learned':lambda r:((np.array(r['features'])-mean)/std)@beta,
            'timestep':lambda r:times[r['k']], 'local_oracle':lambda r:r['local_residual_mse'],
            'endpoint_oracle':lambda r:r['endpoint_mse']}
        for name,fn in functions.items():damages[name].append(min(group,key=fn)['endpoint_mse'])
    means={k:float(np.mean(v)) for k,v in damages.items()}
    for k,v in means.items():assert np.isclose(v,result['mean_damage'][k],rtol=1e-5,atol=1e-10)
    wins={k:sum(a<b for a,b in zip(damages['learned'],damages[k])) for k in ('timestep','local_oracle')}
    assert wins==result['wins']
    passed=all(means['learned']<.8*means[k] and wins[k]>=8 for k in wins)
    route='DEV_SIGNAL_REQUIRES_REAL_CACHE_BASELINE' if passed else 'STOP_NO_DECISIVE_ENDPOINT_SELECTION_ADVANTAGE'
    assert route==result['route']
    return {'integrity':'VERIFIED_FROM_SAVED_IMAGE_TENSORS','route':route,'mean_damage':means,
        'wins':wins,'manifest_sha256':sha(root/'MANIFEST.json'),'source_sha256':sha(Path(__file__)),
        'scope':'Does not rerun the pretrained denoiser or validate image quality.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=verify(a.root);dump(a.out,r);print(json.dumps(r))
