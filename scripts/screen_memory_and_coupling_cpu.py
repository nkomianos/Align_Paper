"""Exact developmental falsification checks for two unrun proposals."""
import argparse
import itertools
import random
from pathlib import Path
from run_unexplored_screens import dump, sha


def terminal_values(events, edges):
    possible=set()
    for order in itertools.permutations(range(len(events))):
        positions={v:i for i,v in enumerate(order)}
        if all(positions[a]<positions[b] for a,b in edges):
            possible.add(events[order[-1]])
    if not possible:raise ValueError('cyclic graph')
    return possible


def maximal_values(events,edges):
    # Last element of a topological order is precisely a maximal element.
    nonmax={a for a,b in edges}
    return {v for i,v in enumerate(events) if i not in nonmax}


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    rng=random.Random(2026091003)
    rows=[]
    for i in range(240):
        threshold=rng.randrange(30,100)
        category=i%4
        values=[threshold-5,threshold+3,threshold+7]
        edges=[(0,1),(1,2)] if category==0 else [(0,1),(0,2)]
        if category==2:values[1]=threshold-3
        if category==3:
            # Historical question: restrict to events known by the queried cut.
            values=values[:2];edges=[(0,1)]
        possibilities=terminal_values(values,edges)
        fast=maximal_values(values,edges)
        assert fast==possibilities
        actions={int(v>=threshold) for v in possibilities}
        oracle=next(iter(actions)) if len(actions)==1 else 'clarify'
        physical=list(range(len(values)));rng.shuffle(physical)
        latest=int(values[physical[-1]]>=threshold)
        naive='clarify' if len(set(values))>1 else int(values[0]>=threshold)
        rows.append(dict(id=i,category=category,threshold=threshold,events=values,edges=edges,
            arrival_order=physical,possible_values=sorted(possibilities),oracle=oracle,
            certain_answer=next(iter({int(x>=threshold) for x in fast})) if len(actions)==1 else 'clarify',
            latest_arrival=latest,abstain_any_conflict=naive))
    metrics={method:{'exact_action_or_clarification_accuracy':sum(r[method]==r['oracle'] for r in rows)/len(rows),
                     'clarification_rate':sum(r[method]=='clarify' for r in rows)/len(rows)}
             for method in ('certain_answer','latest_arrival','abstain_any_conflict')}
    # Two binary outcomes with fixed margins. Exhaustive pairing demonstrates
    # joint variation, while the average treatment difference remains invariant.
    a=[0,0,0,1,1,1]; b=[0,0,1,1,1,1]
    joint=[]
    for order in set(itertools.permutations(b)):
        joint.append({'harm':sum(x==1 and y==0 for x,y in zip(a,order))/6,
                      'mean_difference':sum(y-x for x,y in zip(a,order))/6})
    report={'classification':'DEVELOPMENTAL_EXACT_CPU_ONLY','memory':metrics,
        'memory_decision':'NO_METHOD_HEADROOM_WITH_GOLD_GRAPH; neural extraction and natural ambiguity remain untested',
        'coupling':{'fixed_p_a':.5,'fixed_p_b':2/3,'harm_min':min(r['harm'] for r in joint),
            'harm_max':max(r['harm'] for r in joint),'mean_differences':sorted({r['mean_difference'] for r in joint}),
            'decision':'CLASSICAL_NONIDENTIFICATION; no new application or neural claim'},
        'limitations':['Synthetic parameterizations are not independent external tasks.',
                       'Arbitrary permutations are not admissible causal interventions when structural shared-noise assumptions are fixed.']}
    dump(out/'MEMORY_ROWS.json',rows);dump(out/'COUPLINGS.json',joint);dump(out/'REPORT.json',report)
    dump(out/'MANIFEST.json',{'source':sha(Path(__file__)),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
