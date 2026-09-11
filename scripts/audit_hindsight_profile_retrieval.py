"""Exploratory equal-label retrieval baseline on exposed calibration inputs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def options(r):
    out=dict(re.findall(r'^([ABCD])\) (.*)$',r['prompt'],re.M))
    assert set(out)==set('ABCD')
    return out


def grouped(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['base_id']].append(r)
    result=[]
    for group in groups.values():
        assert {r['label_rotation'] for r in group}=={0,1,2,3}
        assert len({options(r)[r['old_target']] for r in group})==1
        assert len({tuple(sorted(options(r).values())) for r in group})==1
        result.append(min(group,key=lambda r:r['label_rotation']))
    return result


def name(r):
    return r['prompt'].split(':',1)[0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    manifest=json.loads((a.root/'MANIFEST.json').read_text())
    assert sha(a.root/'INPUTS.json')==manifest['INPUTS.json']
    inputs=json.loads((a.root/'INPUTS.json').read_text())
    train,held=grouped(inputs['train']),grouped(inputs['holdout'])
    assert len(train)==128 and len(held)==32
    assert not {r['base_id'] for r in train}&{r['base_id'] for r in held}
    # Vocabulary and profiles use training data only. No held-out label is used
    # until all three predictions for that question have been fixed.
    vocabulary=[text for r in train for text in options(r).values()]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),lowercase=True)
    vectorizer.fit(vocabulary)
    targets=[options(r)[r['old_target']] for r in train]
    target_vectors=vectorizer.transform(targets)
    names=sorted({name(r) for r in train})
    assert all(name(r) in names for r in held)
    matched={n:np.array([i for i,r in enumerate(train) if name(r)==n]) for n in names}
    # Fixed cyclic name mismatch; no outcome-informed control selection.
    next_name={n:names[(i+1)%len(names)] for i,n in enumerate(names)}
    scored=[]
    for r in held:
        opts=options(r)
        labels=sorted(opts)
        candidates=vectorizer.transform([opts[k] for k in labels])
        histories={'same_display_name':matched[name(r)],
                   'cyclic_other_name':matched[next_name[name(r)]],
                   'pooled_names':np.arange(len(train))}
        predictions={}
        for mode,indices in histories.items():
            profile=np.asarray(target_vectors[indices].mean(axis=0))
            scores=cosine_similarity(candidates,profile)[:,0]
            # Lexical ties are independent of answer rotation, unlike label ties.
            winner=min(range(4),key=lambda i:(-float(scores[i]),opts[labels[i]]))
            predictions[mode]={'selected_text':opts[labels[winner]],
                               'history_bases':len(indices),
                               'score_by_text':{opts[k]:float(s) for k,s in zip(labels,scores)}}
        gold=opts[r['old_target']]
        for prediction in predictions.values():
            prediction['correct']=prediction['selected_text']==gold
        scored.append({'base_id':r['base_id'],'predictions':predictions})
    counts={mode:sum(r['predictions'][mode]['correct'] for r in scored)
            for mode in histories}
    result={'classification':'posthoc_developmental_equal_label_baseline',
            'train_bases':128,'heldout_bases':32,'train_display_names':len(names),
            'heldout_display_names':len({name(r) for r in held}),
            'correct_counts':counts,'rows':scored,
            'input_sha256':sha(a.root/'INPUTS.json'),'source_sha256':sha(Path(__file__)),
            'scope':'Exposed calibration split; names not verified people; no neural calls or causal identification.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
