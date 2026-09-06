from .common import seal, write


def route(scores):
    baseline, edited, restored, withdrawn = (scores[k]['accuracy'] for k in ('base','edit','recovered','withdrawn'))
    damage = baseline-edited
    if baseline < .9 or scores['base']['choice_mass'] < .8 or damage < .15:
        return 'STOP_INVALID_ACQUISITION'
    if restored < baseline-.25*damage or scores['recovered_utility']['accuracy'] < scores['base_utility']['accuracy']-.03:
        return 'STOP_INVALID_RECOVERY'
    # Withdrawal shift beyond the independently trained no-edit control.
    shift = abs(withdrawn-scores['d0']['accuracy'])
    random_shift = abs(scores['random_withdrawn']['accuracy']-scores['d0']['accuracy'])
    return 'DEV_SIGNAL_REQUIRES_CAUSAL_FOLLOWUP' if shift-random_shift >= .10 else 'STOP_NO_SPECIFIC_WITHDRAWAL'


def run(backend,data,out):
    t=backend.torch
    train,dev,utility = data['train'],data['dev'],data['utility']
    scores={'base':backend.score(dev,'base'),'base_utility':backend.score(utility,'base_utility')}
    if scores['base']['accuracy'] < .9 or scores['base']['choice_mass'] < .8:
        write(out/'RESULT.json',{'decision':'STOP_INVALID_BASE_CAPABILITY','paper_green_light':False})
        seal(out);return
    h=backend.capture(train)
    positive=t.tensor([r['semantic']==1 for r in train],device=h.device)
    d=h[positive].mean(0)-h[~positive].mean(0)
    if float(d.norm())<1e-8:
        raise ValueError('zero contrast')
    d=d/d.norm()
    base_train=backend.score(train,'base_train')
    choices=[]
    for i,alpha in enumerate((-.5,-1.,-2.,.5,1.,2.)):
        backend.set_edit(d,alpha)
        s=backend.score(train,f'dose_{i}_train')
        if base_train['accuracy']-s['accuracy']>=.15 and s['choice_mass']>=.8:
            choices.append((abs(alpha),alpha))
    if not choices:
        write(out/'RESULT.json',{'decision':'STOP_INVALID_EDIT_FORMATION','paper_green_light':False})
        seal(out);return
    alpha=min(choices)[1]
    random=t.randn_like(d);random=random/random.norm()
    w=backend.base_weight.float()
    random_alpha=alpha*float((d@w).norm()/((random@w).norm()+1e-12))
    t.save({'direction':d.cpu(),'random_direction':random.cpu(),'alpha':alpha,'random_alpha':random_alpha},out/'EDIT.pt')
    write(out/'SELECTION.json',{'alpha':alpha,'random_alpha':random_alpha,'selection_split':'train'})
    backend.set_edit(d,alpha)
    scores['edit']=backend.score(dev,'edit')
    # D0 learns the same data from an exactly reset adapter with E absent.
    backend.set_edit(None,0);backend.reset();backend.train(train,'d0_adapter')
    scores['d0']=backend.score(dev,'d0')
    backend.set_edit(d,alpha);backend.score(dev,'d0_plus_edit')
    backend.reset();backend.train(train,'recovered_adapter')
    scores['recovered']=backend.score(dev,'recovered')
    scores['recovered_utility']=backend.score(utility,'recovered_utility')
    backend.set_edit(None,0)
    scores['withdrawn']=backend.score(dev,'withdrawn')
    backend.score(utility,'withdrawn_utility')
    # Exploratory dose response; a matched pre/post sweep is still needed for causal attribution.
    for i,multiplier in enumerate((-.5,.5,1.5)):
        backend.set_edit(d,alpha*multiplier);backend.score(dev,f'adapted_dose_{i}')
    backend.reset();backend.set_edit(random,random_alpha)
    backend.score(dev,'random_edit')
    backend.train(train,'random_adapter')
    backend.score(dev,'random_recovered')
    backend.set_edit(None,0)
    scores['random_withdrawn']=backend.score(dev,'random_withdrawn')
    decision=route(scores)
    # Holdout remains closed for failed development; no automated paper confirmation.
    if decision=='DEV_SIGNAL_REQUIRES_CAUSAL_FOLLOWUP':
        backend.load('recovered_adapter')
        backend.score(data['holdout'],'withdrawn_holdout')
        backend.set_edit(d,alpha);backend.score(data['holdout'],'recovered_holdout')
    write(out/'RESULT.json',{'decision':decision,'scores':scores,'paper_green_light':False,
        'scope':'single-site, single-seed synthetic policy pilot; not proof of mechanistic compensation',
        'missing_confirmation':['matched-effect control','causal module repair','second family','independent training seeds','external task']})
    seal(out)
