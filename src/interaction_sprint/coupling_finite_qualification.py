"""Exact finite-law qualification; no neural inference or Monte Carlo claims."""
from __future__ import annotations
import argparse
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import numpy as np
from .byte_coupling_native import NativeSampler, byte_event_clock
from .byte_clock_coupling import digest


def token(text, target=None, eos=False, silent=False):
    raw = text.encode()
    label = b'special:EOS' if eos else b'opaque:toy:0' if silent else b'raw:' + raw
    group = label if eos or silent else b'byte:' + raw[:1]
    return dict(raw=b'' if eos or silent else raw, label=label, group=group,
                target=target, eos=eos)


def fixtures():
    direct = {'s': [token('ab'), token('ac')]}
    split = {'s': [token('a', 'tail')], 'tail': [token('b'), token('c')]}
    silent = {'s': [token('', 'tail', silent=True)], 'tail': direct['s']}
    eos_a = {'s': [token('', eos=True), token('ab')]}
    eos_b = {'s': [token('', eos=True), token('a', 'tail')], 'tail': [token('b')]}
    return [('segmentation', direct, split, 2), ('silent_clock', direct, silent, 2),
            ('explicit_eos', eos_a, eos_b, 2), ('binding_cap', direct, split, 1)]


def native_law(graph, cap):
    """Enumerate every native path with rational probabilities, including cap."""
    law = defaultdict(F)
    paths = []
    def visit(state, output, probability, depth, history):
        for t in graph[state]:
            p = probability / len(graph[state])
            out = output + t['raw']
            path = history + [t['label'].decode()]
            reason = 'eos' if t['eos'] else 'cap' if depth + 1 == cap else 'terminal' if t['target'] is None else None
            if reason:
                law[out] += p
                paths.append(dict(tokens=path, output=out.hex(), probability=str(p), stop=reason))
            else:
                visit(t['target'], out, p, depth + 1, path)
    visit('s', b'', F(1), 0, [])
    assert sum(law.values()) == 1
    return dict(law), paths


def event(keys, clock):
    return (int(clock), tuple(sorted(int(k) for k in keys)))


def structural_events(graph, cap):
    """All reachable fair binary Gumbel-order events, no simulation."""
    events = set()
    def visit(state, offset, silent, depth):
        choices = graph[state]
        sampler = NativeSampler([t['label'] for t in choices], [t['group'] for t in choices], [len(t['raw']) for t in choices])
        clock = byte_event_clock(offset, silent)
        groups = sorted(set(sampler.group_ids))
        assert len(groups) <= 2
        counts = [sum(sampler.group_ids == g) for g in groups]
        if len(groups) == 2:
            assert counts[0] == counts[1], 'Only fair comparisons are qualified here'
            events.add(event(sampler.group_keys, clock))
        for g in groups:
            keys = sampler.keys[sampler.group_ids == g]
            assert len(keys) <= 2
            if len(keys) == 2:
                events.add(event(keys, clock))
        for t in choices:
            if not t['eos'] and t['target'] is not None and depth + 1 < cap:
                visit(t['target'], offset + len(t['raw']), 0 if t['raw'] else silent + 1, depth + 1)
    visit('s', 0, 0, 0)
    return events


def check_independent_events(events):
    """Independence only when distinct fair races share no underlying Gumbel."""
    used = set()
    for clock, keys in events:
        assert len(keys) == 2
        atoms = {(clock, key) for key in keys}
        assert not used.intersection(atoms), 'Overlapping races need a different exact integrator'
        used.update(atoms)


def select(keys, clock, assignment):
    if len(keys) == 1:
        return int(keys[0])
    e = event(keys, clock)
    return e[1][assignment[e]]


def hierarchical_path(graph, cap, assignment=None, seed=None):
    state, offset, silent, out = 's', 0, 0, b''
    for depth in range(cap):
        choices = graph[state]
        sampler = NativeSampler([t['label'] for t in choices], [t['group'] for t in choices], [len(t['raw']) for t in choices])
        clock = byte_event_clock(offset, silent)
        if seed is None:
            group_key = select(sampler.group_keys, clock, assignment)
            group = list(map(int, sampler.group_keys)).index(group_key)
            key = select(sampler.keys[sampler.group_ids == group], clock, assignment)
            idx = list(map(int, sampler.keys)).index(key)
        else:
            idx = sampler.draw(np.zeros(len(choices)), seed, clock, 'byte_hierarchical')
        t = choices[idx]
        out += t['raw']
        if t['eos'] or t['target'] is None:
            break
        state = t['target']
        offset += len(t['raw'])
        silent = 0 if t['raw'] else silent + 1
    return out


def hierarchical_joint(a, b, cap):
    events = sorted(structural_events(a, cap) | structural_events(b, cap))
    check_independent_events(events)
    joint = defaultdict(F)
    for bits in product((0, 1), repeat=len(events)):
        assignment = dict(zip(events, bits))
        joint[(hierarchical_path(a, cap, assignment), hierarchical_path(b, cap, assignment))] += F(1, 2**len(events))
    # Deterministic implementation replay, not statistical evidence for the law.
    mismatches = []
    for seed in range(128):
        assignment = {}
        for e in events:
            noise = NativeSampler.gumbel(np.array(e[1], dtype=np.uint64), seed, e[0])
            assignment[e] = int(noise.argmax())
        for graph in (a, b):
            assert hierarchical_path(graph, cap, assignment) == hierarchical_path(graph, cap, seed=seed)
        if hierarchical_path(a, cap, seed=seed) != hierarchical_path(b, cap, seed=seed):
            mismatches.append(seed)
    return dict(joint), len(events), mismatches[:3]


def byte_conditionals(law, prefix):
    mass = sum(p for s, p in law.items() if s.startswith(prefix))
    dist = defaultdict(F)
    for s, p in law.items():
        if s.startswith(prefix):
            dist[None if len(s) == len(prefix) else s[len(prefix)]] += p / mass
    return dict(dist)


def complete_byte_joint(a, b):
    """Exact shared categorical inverse-CDF CRN; fair binary equals shared race.

    These fixtures use only deterministic or fair binary conditionals. Both
    equal-law models therefore have identical next-byte distributions, and
    shared Gumbel CRN has exactly the joint computed here. Unequal laws are
    refused: this is not a general Gumbel-coupling integrator.
    """
    assert a == b
    joint = defaultdict(F)
    def visit(prefix, p):
        da, db = byte_conditionals(a, prefix), byte_conditionals(b, prefix)
        assert da == db and (len(da) == 1 or set(da.values()) == {F(1, 2)})
        for symbol, prob in da.items():
            if symbol is None:
                joint[(prefix, prefix)] += p * prob
            else:
                visit(prefix + bytes([symbol]), p * prob)
    visit(b'', F(1))
    return dict(joint)


def marginals(joint):
    a, b = defaultdict(F), defaultdict(F)
    for (x, y), p in joint.items():
        a[x] += p
        b[y] += p
    return dict(a), dict(b)


def serial_law(law):
    return {s.hex(): str(p) for s, p in sorted(law.items())}


def qualify():
    results = []
    for name, a, b, cap in fixtures():
        la, pa = native_law(a, cap)
        lb, pb = native_law(b, cap)
        joint, n_events, witnesses = hierarchical_joint(a, b, cap)
        assert marginals(joint) == (la, lb)
        byte_joint = complete_byte_joint(la, lb) if la == lb else None
        if byte_joint is not None:
            assert marginals(byte_joint) == (la, lb)
        results.append(dict(case=name, cap=cap, law_a=serial_law(la), law_b=serial_law(lb),
            identical_rendered_laws=la == lb, paths_a=pa, paths_b=pb,
            native_joint=[dict(a=x.hex(), b=y.hex(), p=str(p)) for (x,y),p in sorted(joint.items())],
            native_mismatch=str(sum(p for (x,y),p in joint.items() if x != y)),
            complete_byte_mismatch='0' if byte_joint is not None else None,
            independent_fair_races=n_events, implementation_mismatch_witness_seeds=witnesses,
            native_marginals_exact=True, complete_byte_marginals_exact=byte_joint is not None))
    return dict(scope='Finite exhaustive comparator qualification, not neural evidence or a new theorem',
        arithmetic='Rational path enumeration and exact independent fair Gumbel-order cells',
        iid_assumption='Ideal independent Gumbel atoms; hashed helper only checked by deterministic seed replay',
        cases=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = qualify()
    report['sources'] = {str(p): digest(p) for p in [Path(__file__), Path(__file__).with_name('byte_coupling_native.py'), Path(__file__).with_name('byte_clock_coupling.py')]}
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
