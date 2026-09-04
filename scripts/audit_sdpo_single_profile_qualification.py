"""Summarize separately supplied calibration judgments; never approve training."""
import argparse
import json
from pathlib import Path
from scripts.verify_sdpo_frozen_forward import sha,read,require


def audit(judgments,verified):
    require(verified.get('verified_receipts') is True and verified.get('cases')==16,'complete verified calibration required')
    require(judgments['calibration_manifest_sha256']==verified['calibration_manifest_sha256'],'judgments bound to different calibration')
    require(judgments['reviewer_kind'] in {'assistant_model_audit','human_audit'},'reviewer kind must be explicit')
    require(bool(judgments.get('reviewer_identity')) and bool(judgments.get('rubric')),'reviewer/rubric missing')
    records=judgments['records']; require(len(records)==16 and {r['id'] for r in records}=={r['id'] for r in verified['records']},'judgment IDs differ')
    positive=[]; negative=[]; ambiguity=[]; errors=[]
    for row in records:
        for condition in ('ordinary','explicit','teacher'):
            item=row[condition]
            require(item['style_satisfactory'] in (True,False,None) and item['material_content_error'] in (True,False,None),'invalid judgment labels')
            require(isinstance(item['reason'],str) and item['reason'].strip(),'judgment reason missing')
            if item['material_content_error'] is True:errors.append((row['id'],condition))
            if item['style_satisfactory'] is None or item['material_content_error'] is None:ambiguity.append((row['id'],condition))
        original=row['ordinary']
        if original['material_content_error'] is False:
            if original['style_satisfactory'] is True:positive.append(row)
            elif original['style_satisfactory'] is False:negative.append(row)
    def outcome(group):
        successes=sum(r['teacher']['style_satisfactory'] is True and r['teacher']['material_content_error'] is False for r in group)
        return dict(n=len(group),successes=successes,rate=successes/len(group) if group else None)
    preservation,recovery=outcome(positive),outcome(negative)
    informative=len(positive)>=4 and len(negative)>=4
    thresholds=informative and not ambiguity and not errors and preservation['rate']>=.8 and recovery['rate']>=.75
    return dict(calibration_manifest_sha256=verified['calibration_manifest_sha256'],reviewer_kind=judgments['reviewer_kind'],
        original_satisfactory=preservation,original_unsatisfactory=recovery,ambiguous_judgments=ambiguity,material_errors=errors,
        informative_strata=informative,descriptive_engineering_thresholds_met=thresholds,
        approval=False,scientific_decision=None,
        scope='Supplied reviewer labels are not independently established ground truth. Ambiguities remain explicit and block automatic threshold satisfaction. Pairwise judge consistency and root review still required; no training approval or paper decision.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('judgments');p.add_argument('verified');p.add_argument('report');args=p.parse_args()
    result=audit(read(args.judgments),read(args.verified));result['judgments_sha256']=sha(args.judgments)
    with open(args.report,'x',encoding='utf-8') as handle:json.dump(result,handle,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
