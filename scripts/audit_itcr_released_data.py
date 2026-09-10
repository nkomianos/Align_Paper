"""Freeze question groups, then replay fixed-score calibration on released JSON.

This audits calibration using author-provided labels, not their trained scorer.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import unicodedata

import networkx as nx
import numpy as np
import torch

from audit_itcr_calibration_contract import load_reviewed


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_data(root):
    raw = (root / 'gsm8k_data.json').read_bytes()
    tree = json.loads((root / 'TREE.json').read_text(encoding='utf-8-sig'))
    entry = next(x for x in tree['tree'] if x['path'] == 'ITCR-main/data/gsm8k_data.json')
    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == entry['sha']
    return json.loads(raw)['data'], digest(raw)


def freeze(root):
    rows, data_hash = read_data(root)
    keys = [digest(' '.join(unicodedata.normalize('NFKC', row['prompt']).casefold().split()).encode()) for row in rows]
    groups = sorted(set(keys), key=lambda k: digest(('itcr-contract-audit-v1:' + k).encode()))
    n = len(groups)
    partitions = {}
    for i, key in enumerate(groups):
        partitions[key] = 'development' if i < n // 5 else 'train' if i < 2*n // 5 else 'calibration' if i < 7*n // 10 else 'evaluation'
    plan = dict(data_sha256=data_hash, scope='Public release audit; author labels not independently verified',
                normalization='NFKC, casefold, whitespace collapse; semantic duplicate detection not established',
                scorer_plan=['constant_0.5_plus_0.25_size', 'running_max_1_minus_frequency_plus_0.25_size'],
                alpha=0.1, interpolation='lower',
                invalid_policy='Stop replay on invalid dimensions, labels, frequencies or cyclic graphs; no silent dropping',
                rows=[dict(index=i, question_hash=k, split=partitions[k]) for i, k in enumerate(keys)],
                unique_question_groups=n, row_count=len(rows),
                group_counts=dict(Counter(partitions.values())),
                row_counts=dict(Counter(partitions[k] for k in keys)))
    target = root / 'FROZEN_DATA_REPLAY_PLAN.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2)
    return {k: v for k, v in plan.items() if k != 'rows'}


def graph_from_row(row, require_probability=True):
    claims = row['claims']; adj = row['dep_graph']; n = len(claims)
    assert n > 0 and len(adj) == n and all(len(a) == n for a in adj)
    graph = nx.DiGraph()
    for i, claim in enumerate(claims):
        assert claim['annotation'] in {'Y', 'N'}
        score = float(claim['frequency-score'])
        assert math.isfinite(score)
        if require_probability:
            assert 0 <= score <= 1
        graph.add_node(i, annotation=claim['annotation'], frequency=score)
    for i, a in enumerate(adj):
        for j, value in enumerate(a):
            assert value in [0, 1]
            if value:
                graph.add_edge(i, j)
    assert nx.is_directed_acyclic_graph(graph)
    return graph


def lower_threshold(scores, alpha):
    k = math.floor(alpha * (len(scores) + 1))
    return sorted(scores)[k-1] if k else -math.inf


def upper_threshold(scores, alpha):
    k = math.ceil((1-alpha) * (len(scores) + 1))
    return sorted(scores)[k-1] if k <= len(scores) else math.inf


def score_sequence(seq, mode):
    risks = [0.5 if mode == 'constant' else max(1-g.nodes[v]['frequency'] for v in g) for g in seq[1:]]
    scores = [-math.inf] + [risk + 0.25*i for i, risk in enumerate(risks)]
    assert all(a <= b for a, b in zip(scores, scores[1:]))
    return np.array(scores)


def predict(seq, scores, threshold, strict):
    retained = []
    for graph, score in zip(seq[1:], scores[1:]):
        if score >= threshold if strict else score > threshold:
            break
        retained = list(graph)
    return retained


def replay(root, constant_only=False):
    rows, data_hash = read_data(root)
    plan_path = root / 'FROZEN_DATA_REPLAY_PLAN.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    assert plan['data_sha256'] == data_hash and len(plan['rows']) == len(rows)
    assert plan['alpha'] == 0.1 and plan['interpolation'] == 'lower'
    if constant_only:
        amendment = json.loads((root / 'CONSTANT_ONLY_AMENDMENT.json').read_text(encoding='utf-8'))
        assert amendment['original_plan_sha256'] == digest(plan_path.read_bytes())
        assert amendment['scorers'] == ['constant']
    graphs = [graph_from_row(row, require_probability=not constant_only) for row in rows]
    f, source_hash = load_reviewed(root)
    records = []; summaries = []
    cal_ids = [r['index'] for r in plan['rows'] if r['split'] == 'calibration']
    eval_ids = [r['index'] for r in plan['rows'] if r['split'] == 'evaluation']
    assert len({plan['rows'][i]['question_hash'] for i in eval_ids}) == len(eval_ids), 'Group clustered metrics required for duplicate evaluation questions'
    for mode in (['constant'] if constant_only else ['constant', 'frequency']):
        def replacement(seq, *args, **kwargs):
            scores = score_sequence(seq, mode)
            return np.zeros(len(scores)), scores
        # Replace only the scorer; the released calibration and selection routines
        # remain the reviewed function bodies. No annotation enters the scorer.
        f['risk_scores_for_sequence'] = replacement
        f['risk_scores_for_sequence_no_missing'] = replacement
        sequences = [f['generate_growth_subgraphs'](g) for g in graphs]
        scores_all = [score_sequence(seq, mode) for seq in sequences]
        bad_scores = []; last_true_scores = []
        for i in cal_ids:
            seq, scores = sequences[i], scores_all[i]
            bad = [t for t, g in enumerate(seq) if any(g.nodes[v]['annotation'] != 'Y' for v in g)]
            if bad:
                bad_scores.append(float(scores[bad[0]]))
            true = {v for v in graphs[i] if graphs[i].nodes[v]['annotation'] == 'Y'}
            t = next(t for t, g in enumerate(seq) if true <= set(g))
            last_true_scores.append(float(scores[t]))
        thresholds = {
            ('no_false', 'released'): float(f['calibrate_threshold']([graphs[i] for i in cal_ids], 0.1, None, interpolation='lower')),
            ('no_false', 'repaired'): lower_threshold(bad_scores, 0.1),
            ('no_miss', 'released'): float(f['calibrate_threshold_no_missing']('gsm8k', [graphs[i] for i in cal_ids], 0.1, None, interpolation='lower')),
            ('no_miss', 'repaired'): upper_threshold(last_true_scores, 0.1),
        }
        for (target, method), threshold in thresholds.items():
            assert not math.isnan(threshold)
            arm_records = []
            for i in eval_ids:
                graph = graphs[i]
                retained = predict(sequences[i], scores_all[i], threshold,
                                   strict=(target == 'no_false' and method == 'repaired'))
                if method == 'released':
                    fn = f['get_prediction_set_new'] if target == 'no_false' else f['get_prediction_set_no_missing']
                    result = fn(graph, threshold, None)
                    author_nodes = result[-1] if target == 'no_false' else result[1][-1]
                    assert retained == author_nodes
                true = {v for v in graph if graph.nodes[v]['annotation'] == 'Y'}
                covered = set(retained) <= true if target == 'no_false' else true <= set(retained)
                record = dict(index=i, mode=mode, target=target, method=method,
                              covered=covered, retained=len(retained), total=len(graph))
                arm_records.append(record); records.append(record)
            nonempty = [r for r in arm_records if r['retained']]
            summaries.append(dict(mode=mode, target=target, method=method,
                                  threshold=threshold if math.isfinite(threshold) else str(threshold),
                                  evaluated=len(arm_records), covered=sum(r['covered'] for r in arm_records),
                                  mean_retained_fraction=float(np.mean([r['retained']/r['total'] for r in arm_records])),
                                  nonempty=len(nonempty), covered_nonempty=sum(r['covered'] for r in nonempty),
                                  calibration_bad_graphs=len(bad_scores)))
    result = dict(scope='Controlled fixed-score replay, not trained ITCR replication; author labels only',
                  data_sha256=data_hash, source_sha256=source_hash, plan_sha256=digest(plan_path.read_bytes()),
                  runner_sha256=digest(Path(__file__).read_bytes()),
                  all_graphs_valid=True, graph_count=len(graphs),
                  summaries=summaries, rows=records)
    if constant_only:
        result['amendment_sha256'] = digest((root / 'CONSTANT_ONLY_AMENDMENT.json').read_bytes())
    output_name = 'CONSTANT_SCORE_DATA_REPLAY.json' if constant_only else 'FIXED_SCORE_DATA_REPLAY.json'
    with (root / output_name).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return {k: v for k, v in result.items() if k != 'rows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--mode', choices=['freeze', 'replay', 'constant-replay'], required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source_root) if args.mode == 'freeze' else replay(args.source_root, args.mode == 'constant-replay'), indent=2))
