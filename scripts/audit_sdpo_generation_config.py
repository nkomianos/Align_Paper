"""Forensic merge audit: passed configs are NOT effective greedy configs.

No model inference; replay the visited 4.57.6 default-merging branch against
the separately archived exact installed-method CPU probe. No RNG replay.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from packaging.version import parse

METHOD_SHA = 'fbd277c3bc70bcc944dee222b653e0702234bdeebb8826d19e9613997db9f6ef'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def require(value, message):
    if not value: raise ValueError(message)

def replay(passed, model, defaults, use_model_defaults=None, **kwargs):
    """Only the supplied-config branch; preserved passed config, then kwargs."""
    result=copy.deepcopy(passed)
    if use_model_defaults is True or (use_model_defaults is None and parse(model['transformers_version'])>=parse('4.50.0')):
        for key,value in model.items():
            if key.startswith('_') or key=='transformers_version': continue
            if key=='cache_implementation' and value=='hybrid': continue
            if result.get(key)==defaults.get(key) and value!=defaults.get(key): result[key]=value
        if result['temperature']==0.: result['do_sample']=False
    else:
        for key in ['bos_token_id','eos_token_id','pad_token_id','decoder_start_token_id']:
            if result.get(key) is None: result[key]=model.get(key)
    for key,value in kwargs.items():
        if key in result: result[key]=value
    return result

def audit(root, receipt_path, verification_path):
    root=Path(root); receipt=read(receipt_path); verification=read(verification_path)
    require(verification.get('verified_receipts') is True, 'unverified experiment receipts')
    require(verification['calibration_manifest_sha256']==sha(root/'MANIFEST.json'), 'manifest differs')
    require(verification['cases']==16 and verification['updates']==0, 'not complete zero-update calibration')
    require(receipt['transformers_version']=='4.57.6', 'wrong installed version')
    require(hashlib.sha256(receipt['merge_method_source'].encode()).hexdigest()==METHOD_SHA==receipt['merge_method_sha256'], 'method differs')
    require(not receipt['incomplete_last_line'], 'partial probe')
    hashes=read(root/'model_hashes.json')
    for role,model in receipt['models'].items():
        require(model['generation_config_sha256']==hashes[role]['generation_config.json'], 'model config receipt differs')
    expected=[]
    records=[json.loads(line) for line in (root/'calibration_records.jsonl').read_text().splitlines()]
    for record in records:
        for condition,role in [('ordinary','policy'),('explicit','policy'),('teacher','policy'),('feedback','simulator')]:
            native=record['simulator_native'] if role=='simulator' else record[condition]['native']
            passed=native['generation_config']
            key=role+'/'+hashlib.sha256(json.dumps(passed,sort_keys=True).encode()).hexdigest()
            require(receipt['unique_config_receipts'][key]['passed']==passed,'passed config mismatch')
            expected.append(dict(id=record['id'],condition=condition,config_receipt=key))
    require(receipt['observed_calls']==expected and len(expected)==64,'probe call coverage differs')
    findings={}
    for key,row in receipt['unique_config_receipts'].items():
        require(row['passed_object_unmutated'] and row['method_kwargs_generation_overrides']=={} and row['method_use_model_defaults'] is None,'unexpected call assumptions')
        effective=replay(row['passed'],receipt['models'][row['role']]['resolved_config'],receipt['global_defaults'])
        require(effective==row['effective'],'independent merge differs from exact installed probe')
        require(effective['do_sample'] is True and effective['num_beams']==1 and row['generation_mode']=='GenerationMode.SAMPLE','mode discrepancy')
        findings[row['role']]={k:effective[k] for k in ['do_sample','temperature','top_k','top_p','bos_token_id','eos_token_id','max_new_tokens']}
    return dict(status='VERIFIED_AS_EXECUTED_SAMPLED_CALIBRATION_PROTOCOL_DEVIATION',
        calibration_manifest_sha256=sha(root/'MANIFEST.json'),probe_sha256=sha(receipt_path),
        passed_receipt_verification_sha256=sha(verification_path),auditor_sha256=sha(__file__),
        cases=16,generation_calls=64,updates=0,effective_configs=findings,
        intended_greedy_policy=False,training_approval=False,scientific_decision=None,
        scope='All saved passed configs matched. Exact installed-method CPU probe and independent visited-branch merge agree. Effective sampling inferred from pinned call sites (no direct generation overrides), not runtime effective-config capture or RNG/neural replay. This is an external forensic attestation; the original manifest did not contain this receipt. Pairwise scores and outcomes do not retroactively qualify the intended greedy protocol.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('receipt');p.add_argument('verification');p.add_argument('report');a=p.parse_args()
    require(not Path(a.report).resolve().is_relative_to(Path(a.root).resolve()),'report inside evidence')
    result=audit(a.root,a.receipt,a.verification)
    with open(a.report,'x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
