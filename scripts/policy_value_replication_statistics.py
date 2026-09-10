"""Prospective casewise reversal inference; independent replication data only."""
from scipy.stats import fisher_exact


def reversal_pvalue(first,second,direction,n=32):
    """Intersection-union test: both policies must reverse in the frozen direction.

    first/second are successes for prefix0,prefix1. direction is +1 when the DEV
    first policy prefers prefix0; no direction may be selected from replication.
    This tests the union null that either claimed directional effect is absent.
    """
    if direction not in (-1,1):raise ValueError('frozen direction required')
    if any(not isinstance(v,int) or not 0<=v<=n for v in [*first,*second]):raise ValueError('invalid counts')
    preferred=0 if direction==1 else 1
    a,b=first[preferred],first[1-preferred]
    c,d=second[1-preferred],second[preferred]
    p1=float(fisher_exact([[a,n-a],[b,n-b]],alternative='greater').pvalue)
    p2=float(fisher_exact([[c,n-c],[d,n-d]],alternative='greater').pvalue)
    return {'first_p':p1,'second_p':p2,'joint_p':max(p1,p2),
            'first_gap':(a-b)/n,'second_gap':(c-d)/n}


def holm_rejections(pvalues,alpha=.05):
    """Familywise control over every selected case, including failed directions."""
    if any(not 0<=p<=1 for p in pvalues.values()):raise ValueError('invalid p value')
    ordered=sorted(pvalues,key=lambda k:(pvalues[k],k));rejected=[]
    for i,key in enumerate(ordered):
        if pvalues[key]>alpha/(len(ordered)-i):break
        rejected.append(key)
    return rejected


def summarize_replication(cases,n=32):
    stats={r['id']:reversal_pvalue(r['first'],r['second'],r['direction'],n) for r in cases}
    if len(stats)!=len(cases):raise ValueError('duplicate case')
    rejected=holm_rejections({key:s['joint_p'] for key,s in stats.items()})
    confirmed=[key for key in rejected if min(stats[key]['first_gap'],stats[key]['second_gap'])>=.25]
    return {'case_statistics':stats,'holm_rejected':rejected,'confirmed_practical_reversals':confirmed,
        'route':'REPLICATED_CASEWISE_REVERSALS_ONLY' if len(confirmed)>=4 else 'STOP_NO_DECISIVE_REPLICATION',
        'scope':'fixed selected prefix pairs; no population prevalence, model-family generalization or algorithm novelty claim'}
