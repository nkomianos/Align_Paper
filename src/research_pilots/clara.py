"""Exact finite-scene reliability screen. No neural model or independence shortcut."""
import itertools
from .common import write, seal


def worlds(n):
    return list(itertools.product((0, 1), repeat=n))


def truth(query, world):
    op, indices = query
    vals = [world[i] for i in indices]
    return bool(all(vals) if op == 'and' else any(vals))


def probabilities(ws, observation, error, correlation):
    # Mixture likelihood: independent bit flips and a shared all-bit flip.
    # Uniform latent-world prior. Normalize explicitly.
    raw = []
    n = len(observation)
    for w in ws:
        d = sum(a != b for a, b in zip(w, observation))
        independent = error**d * (1-error)**(n-d)
        shared = (1-error) if d == 0 else error if d == n else 0.
        raw.append((1-correlation)*independent + correlation*shared)
    total = sum(raw)
    return [p/total for p in raw]


def credible_set(ws, probs, alpha):
    # Tie-complete set prevents arbitrary index order from selecting a world.
    levels = sorted(set(probs), reverse=True)
    for level in levels:
        keep = [i for i, p in enumerate(probs) if p >= level]
        if sum(probs[i] for i in keep) >= 1-alpha-1e-12:
            return [ws[i] for i in keep]
    return ws


def certified(query, scene_set):
    values = {truth(query, w) for w in scene_set}
    return int(next(iter(values))) if len(values) == 1 else None


def run(out, seed=2026090602):
    ws = worlds(6)
    queries = [(op, list(indices)) for op in ('and', 'or') for width in (1, 2, 4, 6)
               for indices in itertools.combinations(range(6), width)]
    rows = []
    for error, correlation in itertools.product((.01, .05, .15), (0., .5, 1.)):
        # Exhaustive observation x truth distribution, not Monte Carlo confidence claims.
        for obs in ws:
            probs = probabilities(ws, obs, error, correlation)
            joint = credible_set(ws, probs, .1)
            for qi, q in enumerate(queries):
                decision = certified(q, joint)
                posterior_true = sum(p for w, p in zip(ws, probs) if truth(q, w))
                # Oracle query-specific posterior is a bound, not a learned competitor.
                oracle = 1 if posterior_true >= .9 else 0 if posterior_true <= .1 else None
                rows.append({'error': error, 'correlation': correlation, 'observation': obs,
                             'query': q, 'query_id': qi, 'joint_decision': decision,
                             'oracle_decision': oracle, 'posterior_true': posterior_true,
                             'joint_mass': sum(p for w,p in zip(ws,probs) if w in joint)})
                assert certified((q[0], q[1]+q[1]), joint) == decision
    def summarize(key):
        accepted = [r for r in rows if r[key] is not None]
        risk = sum((1-r['posterior_true']) if r[key] else r['posterior_true'] for r in accepted)
        return {'coverage': len(accepted)/len(rows), 'accepted_posterior_error': risk/len(accepted) if accepted else None}
    write(out/'ROWS.json', rows)
    write(out/'RESULT.json', {'scope': 'exact Bayesian apparatus; not conformal or empirical perception evidence',
                            'decision': 'CPU_BASELINES_ONLY_NO_GPU_EXPANSION', 'paper_green_light': False,
                            'joint': summarize('joint_decision'), 'oracle_query': summarize('oracle_decision'),
                            'rows_are_not_independent_samples': True,
                            'min_joint_mass': min(r['joint_mass'] for r in rows)})
    seal(out)
