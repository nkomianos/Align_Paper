"""Posthoc nearest-task baseline after centroid retrieval diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from audit_hindsight_profile_retrieval import grouped, options, name, sha


def task(r):
    # Answer ordering and displayed name do not enter task-similarity matching.
    return r['prompt'].splitlines()[0].split(':',1)[1]+'\n'+'\n'.join(sorted(options(r).values()))


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
    names=sorted({name(r) for r in train})
    next_name={n:names[(i+1)%len(names)] for i,n in enumerate(names)}
    vocabulary=[task(r) for r in train]+[v for r in train for v in options(r).values()]
    vec=TfidfVectorizer(ngram_range=(1,2)).fit(vocabulary)
    tasks=vec.transform([task(r) for r in train])
    gold_train=[options(r)[r['old_target']] for r in train]
    targets=vec.transform(gold_train)
    records=[]
    for r in held:
        opts=options(r);labels=sorted(opts)
        scores=cosine_similarity(vec.transform([task(r)]),tasks)[0]
        histories={'same_display_name':[i for i,t in enumerate(train) if name(t)==name(r)],
                   'cyclic_other_name':[i for i,t in enumerate(train) if name(t)==next_name[name(r)]],
                   'pooled_names':list(range(len(train)))}
        preds={}
        for mode,indices in histories.items():
            selected=min(indices,key=lambda i:(-float(scores[i]),train[i]['base_id']))
            if gold_train[selected]=='Do not buy any of these options':
                answer='Do not buy any of these options'
            else:
                similarities=cosine_similarity(vec.transform([opts[k] for k in labels]),targets[selected])[:,0]
                index=min(range(4),key=lambda i:(-float(similarities[i]),opts[labels[i]]))
                answer=opts[labels[index]]
            preds[mode]={'selected_text':answer,'retrieved_base':train[selected]['base_id'],
                         'retrieval_similarity':float(scores[selected])}
        for pred in preds.values():pred['correct']=pred['selected_text']==opts[r['old_target']]
        records.append({'base_id':r['base_id'],'predictions':preds})
    result={'classification':'posthoc_developmental_nearest_task_baseline',
            'n':len(records),'correct_counts':{m:sum(r['predictions'][m]['correct'] for r in records) for m in histories},
            'none_counts':{m:sum(r['predictions'][m]['selected_text']=='Do not buy any of these options' for r in records) for m in histories},
            'input_sha256':sha(a.root/'INPUTS.json'),
            'source_sha256':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('audit_hindsight_profile_retrieval.py')]},
            'rows':records}
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
