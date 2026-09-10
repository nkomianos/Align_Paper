"""Frozen pretrained-DDIM endpoint-error screen; not a speedup/quality claim."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from run_unexplored_screens import sha,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('GPU occupied')
    import torch
    import diffusers
    from diffusers import UNet2DModel,DDIMScheduler
    torch.set_num_threads(4)
    model=UNet2DModel.from_pretrained(a.snapshot,local_files_only=True,use_safetensors=True).to('cuda').eval()
    model.requires_grad_(False)
    scheduler=DDIMScheduler.from_pretrained(a.snapshot,local_files_only=True)
    scheduler.set_timesteps(40,device='cuda')
    dump(a.out/'PROTOCOL.json',{'source_sha256':sha(Path(__file__)),'snapshot':str(a.snapshot),
        'torch':torch.__version__,'diffusers':diffusers.__version__,'dtype':'float32',
        'seeds':list(range(2026091100,2026091124)),'calibration_first':12,'steps':40,
        'injection_steps':[4,12,20,28],'scheduler':dict(scheduler.config),
        'scope':'one-step full-output cache reuse, deterministic DDIM eta=0, endpoint fidelity only',
        'gate':'>=20% mean DEV endpoint MSE reduction against timestep calibration AND oracle local residual norm; >=8 of 12 seeds vs each',
        'limitations':'Not rectified-flow transport, not ERTACache implementation, no FID or measured acceleration.'})
    records=[];started=time.monotonic()
    with torch.inference_mode():
        for i in range(24):
            seed=2026091100+i
            x=torch.randn((1,3,32,32),generator=torch.Generator(device='cuda').manual_seed(seed),device='cuda')
            states=[];eps=[]
            for t in scheduler.timesteps:
                states.append(x.clone())
                e=model(x,t).sample;eps.append(e.clone())
                x=scheduler.step(e,t,x,eta=0).prev_sample
            reference=x.clone()
            dump(a.out/f'base_{i:02d}.json',{'seed':seed,'reference_mean':float(x.mean()),'reference_std':float(x.std())})
            tensors={'initial':states[0].cpu(),'reference':reference.cpu()}
            for k in (4,12,20,28):
                residual=eps[k-1]-eps[k]
                local=float(residual.square().mean())
                # Available before a refresh: previous two outputs and present state.
                prior_change=eps[k-1]-eps[k-2]
                f=[1.,k/40,(k/40)**2,float(prior_change.square().mean()),
                   float(eps[k-1].square().mean()),float(states[k].square().mean()),
                   float((prior_change*states[k]).mean()),float((prior_change*eps[k-1]).mean())]
                altered=scheduler.step(eps[k-1],scheduler.timesteps[k],states[k],eta=0).prev_sample
                for t in scheduler.timesteps[k+1:]:
                    e=model(altered,t).sample
                    altered=scheduler.step(e,t,altered,eta=0).prev_sample
                damage=float((altered-reference).square().mean())
                if not np.isfinite(damage):raise ValueError('nonfinite endpoint')
                records.append({'seed':seed,'split':'calibration' if i<12 else 'dev','k':k,
                    'features':f,'local_residual_mse':local,'endpoint_mse':damage})
                tensors[f'altered_{k}']=altered.cpu();tensors[f'residual_{k}']=residual.cpu()
            torch.save(tensors,a.out/f'tensors_{i:02d}.pt')
            dump(a.out/'ROWS.json',records)
            print(json.dumps({'seeds':i+1,'seconds':time.monotonic()-started}),flush=True)
    cal=[r for r in records if r['split']=='calibration'];dev=[r for r in records if r['split']=='dev']
    x=np.array([r['features'] for r in cal]);y=np.log(np.array([r['endpoint_mse'] for r in cal])+1e-12)
    means=x.mean(0);std=x.std(0);std[std<1e-8]=1.;means[0]=0;std[0]=1
    z=(x-means)/std;penalty=np.eye(z.shape[1])*10.;penalty[0,0]=1e-8
    beta=np.linalg.solve(z.T@z+penalty,z.T@y)
    bytime={k:float(np.mean([r['endpoint_mse'] for r in cal if r['k']==k])) for k in (4,12,20,28)}
    selected=[]
    for seed in sorted({r['seed'] for r in dev}):
        group=[r for r in dev if r['seed']==seed]
        pred=lambda r:float(((np.array(r['features'])-means)/std)@beta)
        choices={'learned':min(group,key=pred),'timestep':min(group,key=lambda r:bytime[r['k']]),
                 'local_oracle':min(group,key=lambda r:r['local_residual_mse']),
                 'endpoint_oracle':min(group,key=lambda r:r['endpoint_mse'])}
        selected.append({'seed':seed,**{k:{'step':r['k'],'damage':r['endpoint_mse']} for k,r in choices.items()}})
    metrics={k:float(np.mean([r[k]['damage'] for r in selected])) for k in choices}
    wins={k:sum(r['learned']['damage']<r[k]['damage'] for r in selected) for k in ('timestep','local_oracle')}
    passed=all(metrics['learned']<.8*metrics[k] and wins[k]>=8 for k in wins)
    dump(a.out/'ANALYSIS.json',{'classification':'DEVELOPMENTAL_PRETRAINED_NEURAL_SCREEN',
        'route':'DEV_SIGNAL_REQUIRES_REAL_CACHE_BASELINE' if passed else 'STOP_NO_DECISIVE_ENDPOINT_SELECTION_ADVANTAGE',
        'mean_damage':metrics,'wins':wins,'selections':selected,'calibration_coefficients':beta.tolist(),
        'calibration_mean':means.tolist(),'calibration_std':std.tolist(),'seconds':time.monotonic()-started,
        'replication_unit':'initial noise seed; four candidate steps are repeated measures',
        'limitations':'Fidelity to full DDIM trajectory is not image quality; local residual is an expensive diagnostic oracle.'})
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()})
    print('ENDPOINT_SCREEN_COMPLETE',flush=True)


if __name__=='__main__':main()
