"""Execute pinned released loss methods with controlled two-token logits.

Not an end-to-end learner, tokenizer, or default top-20 vocabulary test.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from interaction_sprint.bayes_hindsight_objective_audit import gradients


def main():
    p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    raw=a.source.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
    assert digest=='6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d'
    tree=ast.parse(raw.decode()); names={'_full_distillation_loss','_renorm_topk'}
    methods=[n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
             if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods}==names
    cls=ast.ClassDef(name='Released',bases=[],keywords=[],body=methods,decorator_list=[])
    mod=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    ns=dict(torch=torch,F=F,Tuple=Tuple,Dict=Dict)
    exec(compile(mod,str(a.source),'exec'),ns)
    rows=[]
    for prob in [.1,.3,.5,.7,.9]:
        likelihood=np.array([[.9,.1],[.1,.9]])
        pi=np.array([1-prob,prob]); marginal=pi@likelihood
        posterior=pi[:,None]*likelihood/marginal
        direction=0.
        for obs in range(2):
            theta=torch.tensor(np.log(prob/(1-prob)),requires_grad=True,dtype=torch.float64)
            obj=ns['Released'](); obj.config=SimpleNamespace(distillation_topk=2,distillation_add_tail=False)
            def logits(text, ids, **kwargs):
                if text=='teacher':
                    values=torch.tensor(np.log(posterior[:,obs]),dtype=torch.float64).reshape(1,1,2)
                else:
                    values=torch.stack([theta*0,theta]).reshape(1,1,2)
                return F.log_softmax(values,dim=-1)[...,0],torch.ones((1,1)),values
            obj._compute_token_logprobs=logits
            loss,_=obj._full_distillation_loss('student','teacher',torch.zeros((1,1),dtype=torch.long))
            loss.backward(); direction-=marginal[obs]*theta.grad.item()
        expected=gradients(prob,likelihood)['reverse_kl']
        assert np.isclose(direction,expected,atol=1e-12,rtol=0)
        rows.append(dict(p=prob,released_direction=direction,analytic_direction=expected))
    result=dict(status='PASS_CONTROLLED_BINARY_LOSS_MAPPING',source_sha256=digest,
                torch_version=torch.__version__,rows=rows,
                limitations='Injected logits, two-token vocabulary, no tail, no training, no actual Bayesian LM teacher.')
    a.out.mkdir(parents=True,exist_ok=False)
    with (a.out/'probe.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
