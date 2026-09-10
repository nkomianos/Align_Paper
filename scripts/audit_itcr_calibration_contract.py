"""Exact CPU counterexamples for pinned, reviewed ITCR calibration functions.

No external module imports, serialized models, dataset claims, or neural inference.
The fixed constant scorer is admissible for the model-agnostic coverage claim.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import typing
import warnings

import networkx as nx
import numpy as np
import torch

REVISION = 'ccbfcbc80c150fa2f80dabbee4a75b59fd30c2b1'
FUNCTIONS = {
    'calculate_conformal_value', 'calculate_conformal_value_no_missing',
    'generate_growth_subgraphs', 'is_coherent_correct_subgraph',
    'risk_scores_for_sequence', 'risk_scores_for_sequence_no_missing',
    'calibrate_threshold', 'calibrate_threshold_no_missing',
    'get_prediction_set_new', 'get_prediction_set_no_missing',
}


class ConstantScorer:
    def predict_proba(self, x):
        return np.full((len(x), 2), 0.5)


def chain(labels):
    g = nx.DiGraph()
    for i, label in enumerate(labels):
        g.add_node(i, annotation=label)
        if i:
            g.add_edge(i - 1, i)
    return g


def load_reviewed(root):
    tree = json.loads((root / 'TREE.json').read_text(encoding='utf-8-sig'))
    assert json.loads((root / 'COMMIT.json').read_text(encoding='utf-8-sig'))['sha'] == REVISION
    rel = 'ITCR-main/conformal/conformal.py'
    raw = (root / rel).read_bytes()
    entry = next(e for e in tree['tree'] if e['path'] == rel)
    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == entry['sha']
    module = ast.parse(raw.decode())
    selected = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    assert {n.name for n in selected} == FUNCTIONS
    namespace = dict(np=np, nx=nx, torch=torch, math=math, warnings=warnings,
                     List=typing.List, Tuple=typing.Tuple, Any=typing.Any,
                     feature_fn=lambda graph: np.array([len(graph)], dtype=float))
    exec(compile(ast.Module(body=selected, type_ignores=[]), rel, 'exec'), namespace)
    return namespace, hashlib.sha256(raw).hexdigest()


def run(root):
    f, source_hash = load_reviewed(root)
    scorer = ConstantScorer()
    rows = []
    # Degenerate i.i.d. graph populations, with fixed monotone score at each size.
    # Repeated identical graphs are the actual population here, not independent
    # evidence of an empirical effect on language-model data.
    for interpolation in ['lower', 'linear', 'higher']:
        for name, labels in [('tied_first_bad', 'N'), ('pooled_bad_prefixes', 'N' * 20)]:
            graph = chain(labels)
            threshold = f['calibrate_threshold']([graph] * 20, 0.1, scorer,
                                                interpolation=interpolation)
            retained = f['get_prediction_set_new'](graph, threshold, scorer)[-1]
            # Separately evaluate changing only <= to <, isolating pooled-score
            # failure from equality-at-the-boundary failure.
            seq = f['generate_growth_subgraphs'](graph)
            _, scores = f['risk_scores_for_sequence'](seq, scorer, f['feature_fn'])
            strict_nodes = []
            for sg, score in zip(seq, scores):
                if score >= float(threshold):
                    break
                strict_nodes = list(sg.nodes())
            rows.append(dict(case=name, interpolation=interpolation,
                             threshold=float(threshold), retained_nodes=retained,
                             released_no_false_coverage=int(not retained),
                             strict_only_no_false_coverage=int(not strict_nodes),
                             first_bad_score=float(scores[1])))
        graph = chain('YNY')
        threshold = f['calibrate_threshold_no_missing']('gsm8k', [graph] * 20,
                                                       0.1, scorer, interpolation=interpolation)
        _, retained_sets = f['get_prediction_set_no_missing'](graph, threshold, scorer)
        retained = retained_sets[-1]
        rows.append(dict(case='no_miss_interleaved_truth', interpolation=interpolation,
                         threshold=float(threshold), retained_nodes=retained,
                         true_nodes=[0, 2], released_no_miss_coverage=int({0, 2} <= set(retained))))
    assert all(r.get('released_no_false_coverage', r.get('released_no_miss_coverage')) == 0 for r in rows)
    assert all(r['strict_only_no_false_coverage'] == 0 for r in rows if r['case'] == 'pooled_bad_prefixes')
    assert all(r['strict_only_no_false_coverage'] == 1 for r in rows if r['case'] == 'tied_first_bad')
    return dict(scope='Exact synthetic population counterexamples; not published benchmark replication',
                revision=REVISION, source_sha256=source_hash,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                packages=dict(torch=torch.__version__, numpy=np.__version__, networkx=nx.__version__),
                alpha=0.1, calibration_graphs=20, fixed_scorer_probabilities=[0.5, 0.5],
                rows=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    result = run(args.source_root)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
