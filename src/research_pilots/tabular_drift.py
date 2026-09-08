"""Exploratory acquisition audit. KL decomposition and matrix scaling are prior art."""
import time
import numpy as np
from .common import digest

METHODS=('raw','information','projected','entropy','random')


def probabilities(x):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all() or np.any(x<0) or not np.allclose(x.sum(-1),1,atol=1e-5):
        raise ValueError('invalid probabilities')
    x=np.maximum(x,1e-12)
    return x/x.sum(-1,keepdims=True)


def project_joint(r,p,q):
    r=np.asarray(r,dtype=float).copy()
    for _ in range(10000):
        r*=p[:,None]/r.sum(1)[:,None]
        r*=q[None,:]/r.sum(0)[None,:]
        if max(np.max(abs(r.sum(1)-p)),np.max(abs(r.sum(0)-q)))<1e-10:return r
    raise ValueError('marginal projection did not converge')


def acquisition(p,q,qa):
    p,q,qa=map(probabilities,(p,q,qa))
    # qa has axes: hypothetical candidate label, target row, target label.
    if qa.shape!=(len(p),len(q),q.shape[1]):raise ValueError('lookahead shape mismatch')
    m=np.einsum('a,azb->zb',p,qa)
    raw=np.einsum('a,azb->z',p,qa*np.log(qa/q[None]))
    info=np.einsum('a,azb->z',p,qa*np.log(qa/m[None]))
    drift=np.sum(m*np.log(m/q),axis=1)
    if not np.allclose(raw,info+drift,atol=1e-10):raise ValueError('KL identity failed')
    projected=[]
    for z in range(len(q)):
        r=project_joint(p[:,None]*qa[:,z,:],p,q[z])
        projected.append(np.sum(r*np.log(r/(p[:,None]*q[z][None,:]))))
    return dict(raw=float(raw.mean()),information=float(info.mean()),drift=float(drift.mean()),
                projected=float(np.mean(projected)),entropy=float(-np.sum(p*np.log(p))))


def build_inputs():
    from sklearn.datasets import load_iris,load_wine,load_breast_cancer,make_classification,make_moons
    from sklearn.preprocessing import StandardScaler
    datasets={name:loader(return_X_y=True) for name,loader in
              [('iris',load_iris),('wine',load_wine),('breast_cancer',load_breast_cancer)]}
    datasets['synthetic_linear']=make_classification(n_samples=240,n_features=8,n_informative=5,
                                                   n_redundant=0,random_state=701)
    datasets['synthetic_moons']=make_moons(n_samples=240,noise=.2,random_state=702)
    cases=[]
    for name,(x,y) in datasets.items():
        for seed in (17,29):
            rng=np.random.default_rng(seed)
            # Two labeled bootstrap examples per class; charged to initial context.
            initial=np.concatenate([rng.choice(np.flatnonzero(y==c),2,replace=False) for c in np.unique(y)])
            remaining=rng.permutation(np.setdiff1d(np.arange(len(y)),initial))
            initial=np.r_[initial,remaining[:12-len(initial)]]
            remaining=rng.permutation(np.setdiff1d(np.arange(len(y)),initial))
            target,evaluation,pool=np.split(remaining[:80],[16,48])
            scaled=StandardScaler().fit(x[initial]).transform(x)
            cases.append({'id':f'{name}_{seed}','dataset':name,'seed':seed,'x':scaled.tolist(),'y':y.tolist(),
                'initial':initial.tolist(),'target':target.tolist(),'evaluation':evaluation.tolist(),
                'shortlists':pool.reshape(4,8).tolist()})
    return {'cases':cases,'case_hashes':{c['id']:digest(c) for c in cases},
            'scope':'development only; initial-pool seeds are nested within datasets; no confirmation'}


def validate(data):
    if len({c['id'] for c in data['cases']})!=len(data['cases']):raise ValueError('duplicate cases')
    for c in data['cases']:
        if digest(c)!=data['case_hashes'][c['id']]:raise ValueError('input hash differs')
        x,y=np.array(c['x']),np.array(c['y'])
        groups=[c[k] for k in ('initial','target','evaluation')]+c['shortlists']
        ids=[i for group in groups for i in group]
        if len(ids)!=len(set(ids)) or min(ids)<0 or max(ids)>=len(y):raise ValueError('split leakage')
        if len(x)!=len(y) or not np.isfinite(x).all():raise ValueError('invalid features')
        if sorted(set(y))!=list(range(len(set(y)))) or set(y[c['initial']])!=set(y):
            raise ValueError('classes missing from context')
    return data


class Predictor:
    def __init__(self,backend,checkpoint=None,device='cuda'):
        self.backend=backend;self.calls=0;self.seconds=0.
        if backend=='logistic':
            from sklearn.linear_model import LogisticRegression
            self.model=LogisticRegression(C=1.,max_iter=1000,random_state=0)
        elif backend=='tabicl':
            from tabicl import TabICLClassifier
            # The official fit reloads immutable weights; reuse them, but refit context each call.
            class CachedWeights(TabICLClassifier):
                def _load_model(self):
                    if not hasattr(self,'model_'):super()._load_model()
            self.model=CachedWeights(model_path=checkpoint,allow_auto_download=False,device=device,
                n_estimators=4,random_state=0,kv_cache=False,use_fa3=False,n_jobs=1)
        else:raise ValueError('unknown backend')

    def predict(self,x,y,query):
        started=time.monotonic()
        self.model.fit(x,y)
        output=probabilities(self.model.predict_proba(query))
        self.calls+=1;self.seconds+=time.monotonic()-started
        return output


def experiment(case,predictor,rounds=4):
    x,y=np.array(case['x']),np.array(case['y'])
    target,evaluation=case['target'],case['evaluation']
    initial=case['initial']
    baseline=predictor.predict(x[initial],y[initial],x[evaluation])
    result={'id':case['id'],'dataset':case['dataset'],'baseline':baseline.tolist(),'arms':{}}
    for method in METHODS:
        context=list(initial);records=[]
        for step,shortlist in enumerate(case['shortlists'][:rounds]):
            start=time.monotonic();before=predictor.calls;scores=[];raw=None
            p=predictor.predict(x[context],y[context],x[shortlist])
            if method in ('raw','information','projected'):
                q=predictor.predict(x[context],y[context],x[target]);qa=[]
                for i in shortlist:
                    qa.append([predictor.predict(x[context+[i]],np.r_[y[context],a],x[target]).tolist()
                               for a in range(len(p[0]))])
                scores=[acquisition(p[j],q,qa[j]) for j in range(len(shortlist))]
                raw={'p':p.tolist(),'q':q.tolist(),'qa':qa}
                index=max(range(len(scores)),key=lambda j:(scores[j][method],-j))
            elif method=='entropy':index=int(np.argmax(-np.sum(p*np.log(p),axis=1)))
            else:index=int(np.random.default_rng(case['seed']+1000+step).integers(len(shortlist)))
            acquisition_seconds=time.monotonic()-start
            selected=shortlist[index];context.append(selected)
            # Label lookup occurs after selection; evaluation labels never enter acquisition.
            pred=predictor.predict(x[context],y[context],x[evaluation])
            records.append({'step':step,'shortlist':shortlist,'selected':selected,'context':list(context),
                'scores':scores,'lookahead':raw,'candidate_p':p.tolist(),'evaluation_p':pred.tolist(),
                'acquisition_seconds':acquisition_seconds,'predict_calls':predictor.calls-before})
        result['arms'][method]=records
    return result


def summarize(cases,outputs,backend):
    values={m:[] for m in METHODS};disagreements=[];drift=[]
    for c,o in zip(cases,outputs):
        labels=np.array(c['y'])[c['evaluation']]
        for method,records in o['arms'].items():
            # Mean held-out NLL over the four acquired-label checkpoints, not independent replicates.
            losses=[float(-np.log(probabilities(r['evaluation_p'])[np.arange(len(labels)),labels]).mean()) for r in records]
            values[method].append({'dataset':c['dataset'],'case':c['id'],'mean_nll':float(np.mean(losses))})
        first=o['arms']['raw'][0]['scores']
        disagreements.append(int(np.argmax([s['raw'] for s in first])!=np.argmax([s['information'] for s in first])))
        drift.extend(s['drift']/max(s['raw'],1e-12) for s in first)
    means={m:float(np.mean([r['mean_nll'] for r in rows])) for m,rows in values.items()}
    # Information is the prespecified primary correction; projection is secondary.
    improved=all(means['information']<=.98*means[b] for b in ('raw','entropy','random'))
    decision='DEV_SIGNAL_REQUIRES_SECOND_FAMILY' if improved and np.mean(disagreements)>=.2 else 'STOP_NO_DECISIVE_DEV_ADVANTAGE'
    if backend!='tabicl':decision='DEVELOPMENTAL_PIPELINE_ONLY'
    return {'decision':decision,'paper_green_light':False,'mean_nll':means,'case_nll':values,
            'initial_ranking_disagreement':float(np.mean(disagreements)),
            'mean_initial_drift_fraction':float(np.mean(drift)),
            'unit_of_replication':'dataset; initial pools nested, rounds and candidate rows not independent',
            'missing_confirmation':['second predictor family','independent datasets','EPIG/TabMGP baselines',
                                    'calibration and ensemble controls','equal-wall-clock comparison']}
