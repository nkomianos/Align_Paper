"""CPU falsifiers for a candidate selection rule, not neural transfer evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize


def audit():
    cases=[]
    for cosine in np.linspace(-.9,.99,32):
        a=np.array([1.,0.]);b=np.array([cosine,np.sqrt(1-cosine*cosine)])
        pooled=(a+b)/np.linalg.norm(a+b)
        # Independent numerical optimization: maximize the weakest alignment.
        result=minimize(lambda z:-z[2],np.array([0.,0.,-.1]),method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda z:1-np.dot(z[:2],z[:2])},
                         {'type':'ineq','fun':lambda z:np.dot(a,z[:2])-z[2]},
                         {'type':'ineq','fun':lambda z:np.dot(b,z[:2])-z[2]}],
            options={'ftol':1e-12,'maxiter':500})
        optimum=np.sqrt((1+cosine)/2)
        cases.append({'cosine':float(cosine),'pooled_min_alignment':float(min(a@pooled,b@pooled)),
                      'numerical_minimax_alignment':float(result.x[2]),'direction_distance':float(np.linalg.norm(result.x[:2]-pooled)),
                      'optimizer_success':bool(result.success),'optimizer_message':str(result.message),
                      'objective_gap':float(abs(result.x[2]-optimum)),
                      'constraint_violation':float(max(0,np.dot(result.x[:2],result.x[:2])-1,result.x[2]-a@result.x[:2],result.x[2]-b@result.x[:2]))})
    # Two observationally identical held-out recipe worlds: h=x*v, logit=v.h.
    # For unit v, every unsteered logit equals x. Steering h+=alpha*d changes
    # the logit by alpha*v.d, which can have either sign without changing x.
    xs=[-2.,-1.,1.,2.];direction=np.array([1.,0.]);alpha=.5;worlds=[]
    for label,v in [('positive',np.array([1.,0.])),('negative',np.array([-1.,0.])),('orthogonal',np.array([0.,1.]))]:
        baseline=[float(v@(x*v)) for x in xs]
        intervention=[float(v@(x*v+alpha*direction)) for x in xs]
        assert baseline==xs
        worlds.append({'world':label,'baseline_logits':baseline,'steered_logits':intervention,
                       'mean_signed_shift':float(np.mean(np.array(intervention)-baseline))})
    return {'classification':'DEVELOPMENTAL_GEOMETRY_AND_IDENTIFICATION_AUDIT',
            'minimax_cases':cases,'counterexample_worlds':worlds,
            'decision':'TWO_DIRECTION_ALIGNMENT_MINIMAX_EQUALS_NORMALIZED_POOLING; BEHAVIOR_MATCHING_ALONE_DOES_NOT_IDENTIFY_UNSEEN_RECIPE_STEERING',
            'scope':'Restricted unit-direction linear objective and constructed observational counterexample. Does not disprove empirical cross-recipe transfer, nonlinear selection or a specified distributional assumption. No neural model, learned behavior or paper result.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=audit()
    with a.out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'classification':result['classification'],'cases':len(result['minimax_cases']),
                      'maximum_direction_distance':max(r['direction_distance'] for r in result['minimax_cases']),
                      'worlds':result['counterexample_worlds'],'decision':result['decision']},indent=2))
