"""Replay sealed score arithmetic and routing; does not certify neural inference."""
import argparse
import json
import math
from pathlib import Path

from interaction_sprint.hindsight_calibration import ARMS, prepare, summarize, qualified_teacher, route
from interaction_sprint.hindsight_execution_integrity import verify_manifest


def close(actual,expected):
    if isinstance(actual,dict):
        if actual.keys()!=expected.keys(): raise ValueError('keys differ')
        for k in actual:close(actual[k],expected[k])
    elif isinstance(actual,list):
        if len(actual)!=len(expected): raise ValueError('length differs')
        for a,b in zip(actual,expected):close(a,b)
    elif isinstance(actual,float):
        if not math.isfinite(actual) or not math.isclose(actual,expected,abs_tol=1e-5,rel_tol=0):
            raise ValueError('recomputed float differs')
    elif actual!=expected: raise ValueError('identity differs')


def verify(root,learning):
    import torch
    verify_manifest(root)
    read=lambda name:json.loads((root/name).read_text(encoding='utf8'))
    selected=prepare(learning)
    close(read('INPUTS.json'),selected)
    lookup={r['id']:r for split in ('train','holdout') for r in selected[split]}
    answers=read('RUNTIME.json')['answer_ids']
    scored={}
    for path in root.glob('*.json'):
        payload=json.loads(path.read_text(encoding='utf8'))
        if not isinstance(payload,dict) or set(payload)!= {'rows','summary'}: continue
        rows=payload['rows']; expected_ids={r['id'] for r in selected['train' if '_train_' in path.stem else 'holdout']}
        if len(rows)!=len(expected_ids) or {r['id'] for r in rows}!=expected_ids: raise ValueError('score population differs')
        cache={}; replayed_rows=[]
        for r in rows:
            name=r['forward_file']
            if Path(name).name!=name or not name.startswith('forward_') or not name.endswith('.pt'): raise ValueError('unsafe forward path')
            if name not in cache:cache[name]=torch.load(root/name,map_location='cpu',weights_only=True)
            raw=cache[name]; i=r['forward_row']; source=lookup[r['id']]
            if raw['ids'][i]!=r['id'] or r['base_id']!=source['base_id'] or r['label_rotation']!=source['label_rotation']: raise ValueError('row binding differs')
            if raw['teacher_context']!=path.stem.endswith('_teacher') or raw['student_gradient']: raise ValueError('score context/gradient differs')
            # Accumulate the large-vocabulary reduction in float64. CPU float32
            # log_softmax can itself exceed the fixed tolerance to CUDA results.
            logits=raw['logits'][i].double(); choice=logits.log_softmax(-1)[answers]; cond=choice.log_softmax(-1)
            target='ABCD'.index(source['old_target'])
            recomputed={'nll':-float(choice[target]),'probability':float(choice[target].exp()),
                'conditional_probability':float(cond[target].exp()),'correct':int(int(choice.argmax())==target),
                'choice_mass':float(choice.exp().sum())}
            close(recomputed,{k:r[k] for k in recomputed})
            replayed_rows.append({**r,**recomputed})
        actual_summary=summarize(replayed_rows)
        close(actual_summary,payload['summary']); scored[path.stem]=actual_summary
    result=read('RESULT.json')
    if (not qualified_teacher(scored['baseline_teacher']) or
            ('baseline_train_teacher' in scored and not qualified_teacher(scored['baseline_train_teacher']))):
        if result['decision']!='STOP_INVALID_INITIAL_TEACHER':raise ValueError('interface route differs')
    else:
        if 'baseline_train_teacher' not in scored:raise ValueError('missing train teacher qualification')
        replay=route(scored['baseline_student'],{a:scored[a+'_final_student'] for a in ARMS},
            {a:scored[a+'_final_teacher'] for a in ARMS})
        close(replay,{k:result[k] for k in replay})
        inputs=[r['id'] for r in selected['train']]
        for arm in ARMS:
            steps=[read(f'{arm}_step_{i:03d}.json') for i in range(1,33)]
            if [r for step in steps for r in step['row_ids']]!=inputs:raise ValueError('training schedule differs')
            if [s['step'] for s in steps]!=list(range(1,33)):raise ValueError('update ordering differs')
    return {'saved_arithmetic_and_routing_verified':True,'neural_checkpoint_replay_performed':False,
            'decision':result['decision'],'paper_green_light':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--learning',type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.root,a.learning),indent=2))
