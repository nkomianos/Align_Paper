"""Analytic null-reference control for the published PMI target, not model code."""
import hashlib
import json
import math
from pathlib import Path


def target(base, teacher, reference, clip=None):
    correction = [math.log(t/r) for t, r in zip(teacher, reference)]
    mean = sum(correction)/len(correction)
    correction = [x-mean for x in correction]
    if clip is not None:
        correction = [clip*math.tanh(x/clip) for x in correction]
    weights = [p*math.exp(c) for p, c in zip(base, correction)]
    return [w/sum(weights) for w in weights]


def cross_entropy(truth, pred):
    return -sum(p*math.log(q) for p, q in zip(truth, pred))


def run():
    rows = []
    for accuracy in [0.6, 0.8, 0.95]:
        # Q and R independent fair bits; V equals Q with stated probability.
        # All eight joint outcomes have positive probability. R supplies no
        # information about V, conditional on Q or otherwise.
        joint = {(q,r,v): 0.25*(accuracy if v == q else 1-accuracy)
                 for q in [0,1] for r in [0,1] for v in [0,1]}
        assert abs(sum(joint.values())-1) < 1e-12
        for q in [0,1]:
            for r in [0,1]:
                base = [sum(joint[q,rr,v] for rr in [0,1])/0.5 for v in [0,1]]
                teacher = [joint[q,r,v]/0.25 for v in [0,1]]
                reference = [sum(joint[qq,r,v] for qq in [0,1])/0.5 for v in [0,1]]
                assert base == teacher
                for clip in [None,10.0]:
                    purified = target(base, teacher, reference, clip)
                    # Question-only density-ratio control reproduces the target
                    # exactly here: there is no reference-specific information.
                    control = target(base, base, [0.5,0.5], clip)
                    assert all(abs(a-b)<1e-12 for a,b in zip(purified,control))
                    excess = cross_entropy(base,purified)-cross_entropy(base,base)
                    assert excess > 0
                    rows.append(dict(base_correct_probability=accuracy,q=q,r=r,clip=clip,
                                     target_correct_probability=purified[q],
                                     excess_expected_log_loss_nats=excess,
                                     matches_question_only_control=True))
    return dict(scope='Full-support probabilistic null model; not an empirical OPSD result',
                assumptions='Exact conditional distributions; independent uninformative reference; beta=1',
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows)


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); result=run()
    with args.out.open('x',encoding='utf-8') as out:
        json.dump(result,out,indent=2)
    print(json.dumps([r for r in result['rows'] if r['q']==0 and r['r']==0],indent=2))
