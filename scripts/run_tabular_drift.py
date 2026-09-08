"""Offline acquisition pilot. Logistic runs are developmental, never TFM evidence."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from research_pilots.common import write,seal
from research_pilots.tabular_drift import build_inputs,validate,Predictor,experiment,summarize

CHECKPOINT_SHA='bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=['logistic','tabicl'],default='logistic')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--checkpoint-sha256',default=CHECKPOINT_SHA)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=p.parse_args()
    data=validate(json.loads(a.inputs.read_text()) if a.inputs else build_inputs())
    if a.out.exists():raise FileExistsError('new output required')
    model_sha=None
    if a.backend=='tabicl' and not a.prepare_only:
        if not a.checkpoint or not a.checkpoint.is_file():raise ValueError('cached checkpoint missing')
        model_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        if model_sha!=a.checkpoint_sha256 or model_sha!=CHECKPOINT_SHA:raise ValueError('checkpoint not pinned v2')
        if a.device=='cuda' and os.environ.get('RESEARCH_PILOT_SUPERVISED')!='1':
            raise ValueError('use research launcher for GPU admission')
    a.out.mkdir(parents=True)
    write(a.out/'INPUTS.json',data)
    if a.prepare_only:return
    repo=Path(__file__).resolve().parents[1]
    source_paths=[Path(__file__),repo/'src/research_pilots/tabular_drift.py',repo/'src/research_pilots/common.py']
    provenance={'pilot':'tabular_drift','backend':a.backend,'device':a.device,'smoke':a.smoke,
        'checkpoint_sha256':model_sha,'sources':{str(s.relative_to(repo)):hashlib.sha256(s.read_bytes()).hexdigest() for s in source_paths},
        'versions':{s:importlib.metadata.version(s) for s in ('numpy','scikit-learn','scipy')}}
    if a.backend=='tabicl':
        import tabicl,torch
        torch.set_num_threads(4)
        base=Path(tabicl.__file__).parent
        provenance['tabicl_sources']={str(s.relative_to(base)):hashlib.sha256(s.read_bytes()).hexdigest() for s in base.rglob('*.py')}
        provenance['torch']=torch.__version__
    write(a.out/'PROVENANCE.json',provenance)
    predictor=Predictor(a.backend,a.checkpoint,a.device)
    cases=data['cases'][:1] if a.smoke else data['cases']
    outputs=[];started=time.monotonic()
    for i,c in enumerate(cases):
        result=experiment(c,predictor,rounds=1 if a.smoke else 4)
        write(a.out/f'case_{i:02d}.json',result);outputs.append(result)
        print(json.dumps({'completed_case':c['id'],'elapsed_seconds':time.monotonic()-started,
                          'predict_calls':predictor.calls}),flush=True)
    summary=summarize(cases,outputs,a.backend if not a.smoke else 'smoke')
    summary.update(wall_seconds=time.monotonic()-started,predict_calls=predictor.calls,
                   predictor_seconds=predictor.seconds)
    write(a.out/'RESULT.json',summary);seal(a.out)


if __name__=='__main__':main()
