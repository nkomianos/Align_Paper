"""Fixed-layer OOF descriptive analysis; no post-hoc layer selection."""
import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--verified', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    verified = json.loads(a.verified.read_text())
    assert verified['rows'] == 80 and verified['jointly_valid'] >= 75
    rows = [json.loads(s) for s in (a.run/'ROWS.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['answer_valid'] and r['decision_valid']]
    assert [r['feature_index'] for r in rows] == list(range(len(rows)))
    y = np.array([r['abstain'] for r in rows], dtype=int)
    assert min(y.sum(), len(y)-y.sum()) >= 10
    features = torch.load(a.run/'FEATURES.pt', map_location='cpu', weights_only=True)
    assert features.shape == (len(rows), 36, 4, 4096)
    assert torch.isfinite(features).all()
    sites = ['stem_end','options_end','before_answer','decision_readout']
    assert all(r['site_labels'] == sites for r in rows)
    x = features[:,17].numpy()
    folds = list(StratifiedKFold(5, shuffle=True, random_state=20260911).split(x, y))
    scores = np.full((len(rows),4),np.nan)
    fold_ids = np.full(len(rows),-1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for fold,(train,test) in enumerate(folds):
            assert not set(train)&set(test)
            fold_ids[test]=fold
            for site in range(4):
                model=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=2000))
                model.fit(x[train,site],y[train])
                scores[test,site]=model.predict_proba(x[test,site])[:,1]
    assert np.isfinite(scores).all() and (fold_ids>=0).all()
    aurocs={site:float(roc_auc_score(y,scores[:,i])) for i,site in enumerate(sites)}
    result=dict(classification='developmental_descriptive_oof_probe',layer_zero_based=17,
                n=len(rows),abstain=int(y.sum()),commit=int(len(y)-y.sum()),auroc=aurocs,
                options_minus_stem=aurocs['options_end']-aurocs['stem_end'],
                final_minus_options=aurocs['decision_readout']-aurocs['options_end'],
                seed=20260911,excluded=80-len(rows),
                scope='Single exposed domain/model; no equivalence test, neural causal identification or paper qualification.',
                predictions=[dict(id=r['id'],fold=int(fold_ids[i]),abstain=int(y[i]),
                                  scores=dict(zip(sites,map(float,scores[i])))) for i,r in enumerate(rows)])
    with a.out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='predictions'}))


if __name__ == '__main__':
    main()
