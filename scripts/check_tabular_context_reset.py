"""CPU check of cached-weight adapter against independent official model loads."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from research_pilots.common import write
from research_pilots.tabular_drift import build_inputs,Predictor
from run_tabular_drift import CHECKPOINT_SHA

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:raise ValueError('wrong checkpoint')
torch.set_num_threads(2)
c=build_inputs()['cases'][0];x=np.array(c['x']);y=np.array(c['y']);ctx=c['initial'];target=c['target']
model=Predictor('tabicl',a.checkpoint,'cpu')
first=model.predict(x[ctx],y[ctx],x[target])
def weight_hash():
    h=hashlib.sha256()
    for name,param in model.model.model_.named_parameters():h.update(name.encode());h.update(param.detach().cpu().numpy().tobytes())
    return h.hexdigest()
before=weight_hash()
model.predict(x[ctx],(y[ctx]+1)%3,x[target])
repeat=model.predict(x[ctx],y[ctx],x[target])
fresh=Predictor('tabicl',a.checkpoint,'cpu').predict(x[ctx],y[ctx],x[target])
assert before==weight_hash() and np.allclose(first,repeat,atol=1e-7) and np.allclose(first,fresh,atol=1e-7)
report={'status':'CPU_CONTEXT_RESET_PASS','weights_unchanged':True,'fresh_model_agreement':True,
        'max_repeat_difference':float(np.max(abs(first-repeat))),'gpu_used':False}
write(a.out,report);print(json.dumps(report,indent=2))
