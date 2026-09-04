"""Post-hoc descriptive summaries of verified fixed-completion diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.verify_sdpo_frozen_forward import sha,require,read,rows


def describe(values):
    return dict(n=len(values),mean=float(np.mean(values)),median=float(np.median(values)),
                minimum=float(np.min(values)),maximum=float(np.max(values)))


def summarize(root, report_path):
    root=Path(root); verified=read(report_path)
    require(verified.get('verified') is True or
            (verified.get('verification_status')=='PARTIAL_ARITHMETIC_REPLAY_DIRECTIONAL_METRICS_UNRESOLVED'
             and verified.get('scalar_arithmetic_verified') is True),
            'requires scalar arithmetic replay; directional metrics are not used')
    require(verified['diagnostic_manifest_sha256']==sha(root/'MANIFEST.json'),'report root mismatch')
    require(verified['status']=='COMPLETE_FIXED_FORWARD_DIAGNOSTIC','requires complete diagnostic')
    cases=read(root/'cases.json'); records=rows(root/'records.jsonl')
    indexed={(r['checkpoint'],r['id'],r['context']):r for r in records}
    audits={(r['checkpoint'],r['id'],r['context'],r['offset']):r for r in verified['local_position_audits']}
    result=[]
    for checkpoint in ('initial_adapter.pt','adapter_step_16.pt','final_adapter.pt'):
        for correct in (True,False):
            group=[c for c in cases if c['original_joint']==correct]
            entry=dict(checkpoint=checkpoint,originally_correct=correct,n=len(group),contexts={})
            for context in ('base','empty','feedback','explicit'):
                subset=[indexed[checkpoint,c['id'],context] for c in group]
                first=[float(np.exp(r['completion_logps'][0])) for r in subset]
                entry['contexts'][context]=dict(original_first_token_probability=describe(first),
                    archived_forward_reported_completion_mean_logp=describe([float(np.mean(r['completion_logps'])) for r in subset]))
            for context in ('feedback','explicit'):
                deltas=[float(np.mean(indexed[checkpoint,c['id'],context]['completion_logps'])-
                              np.mean(indexed[checkpoint,c['id'],'empty']['completion_logps'])) for c in group]
                entry[context+'_minus_empty_archived_forward_reported_mean_logp']=describe(deltas)
                entry[context+'_minus_empty_archived_report_negative_cases']=sum(v<0 for v in deltas)
                selected_deltas=[audits[checkpoint,c['id'],context,0]['observed_advantage']-
                                 audits[checkpoint,c['id'],'empty',0]['observed_advantage'] for c in group]
                entry[context+'_minus_empty_replayed_first_token_logp']=describe(selected_deltas)
                entry[context+'_minus_empty_replayed_first_token_negative_cases']=sum(v<0 for v in selected_deltas)
            for slot in (0,1):
                per_context={}
                for context in ('empty','feedback','explicit'):
                    group_audits=[audits[checkpoint,c['id'],context,c['positions'][slot]] for c in group]
                    per_context[context]={key:describe([a[key] for a in group_audits]) for key in
                        ('full_reverse_kl','expected_gradient_norm','sampled_gradient_trace_variance',
                         'released_topk_tail_reverse_kl','topk_full_gradient_l2_difference')}
                entry['first_position' if slot==0 else 'punctuation_position']=per_context
            result.append(entry)
    return dict(verification_report_sha256=sha(report_path),source_manifest=verified['diagnostic_manifest_sha256'],
                verification_status=verified.get('verification_status','VERIFIED'),
                groups=result,scope='Post-hoc frozen-completion comparisons only. Only two selected positions have independently replayed logsoftmax; whole-completion mean logps are archived-forward-reported, not independently replayed. Original-token probability is not correct-answer probability; wrong and correct alternative responses are not enumerated. No causal attribution to LR, optimizer, objective, or feedback selection. Raw cosine ratios intentionally omitted.')


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('root');p.add_argument('verified');p.add_argument('output');args=p.parse_args()
    require(not Path(args.output).resolve().is_relative_to(Path(args.root).resolve()),'output must be outside evidence')
    result=summarize(args.root,args.verified)
    with open(args.output,'x',encoding='utf-8') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
