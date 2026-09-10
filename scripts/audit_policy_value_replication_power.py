"""Prospective conditional Monte Carlo design audit, never model evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact,beta
from run_unexplored_screens import sha,dump


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rng=np.random.default_rng(2026091061);n=32;repeats=20000
    grid=np.asarray([[float(fisher_exact([[x,n-x],[y,n-y]],alternative='greater').pvalue)
                      for y in range(n+1)] for x in range(n+1)])
    results=[]
    for count in (4,16):
        for name,probabilities in [('null',[.5,.5,.5,.5]),('one_policy_only',[.8,.2,.5,.5]),
                                   ('gap_point3',[.65,.35,.35,.65]),('gap_point5',[.75,.25,.25,.75]),
                                   ('gap_point6',[.8,.2,.2,.8])]:
            draws=rng.binomial(n,probabilities,size=(repeats,count,4))
            pvalues=np.maximum(grid[draws[:,:,0],draws[:,:,1]],grid[draws[:,:,3],draws[:,:,2]])
            order=pvalues.argsort(axis=1,kind='stable');sorted_p=np.take_along_axis(pvalues,order,axis=1)
            sequential=np.logical_and.accumulate(sorted_p<=.05/np.arange(count,0,-1),axis=1)
            effects=(draws[:,:,0]-draws[:,:,1]>=8)&(draws[:,:,3]-draws[:,:,2]>=8)
            practical=np.take_along_axis(effects,order,axis=1)&sequential
            any_reject=sequential.any(axis=1);passes=practical.sum(axis=1)>=4
            rate=float(passes.mean())
            def interval(values):
                k=int(values.sum())
                return [0. if k==0 else float(beta.ppf(.025,k,repeats-k+1)),
                        1. if k==repeats else float(beta.ppf(.975,k+1,repeats-k))]
            results.append({'selected_cases':count,'scenario':name,'success_probabilities':probabilities,
                'replicates':repeats,'any_holm_rejection_rate':float(any_reject.mean()),
                'at_least4_practical_confirmation_rate':rate,
                'any_rejection_mc_interval95':interval(any_reject),'confirmation_mc_interval95':interval(passes)})
    dump(a.out,{'classification':'DEVELOPMENTAL_STATISTICAL_DESIGN_AUDIT','source_sha256':sha(Path(__file__)),
        'results':results,'assumptions':'Independent Bernoulli draws conditional on selected fixed prefixes; equal effects within each scenario.',
        'limitations':'Not empirical model accuracy, observed reversals, or power under arbitrary heterogeneous effects; no gate changed.'})
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
