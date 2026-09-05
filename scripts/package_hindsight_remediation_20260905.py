"""Explicit allowlist package; never redistributes public participant exports."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT/'artifacts/hindsight_reduced_remediation_20260905'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visually-reviewed-pdf-sha256',required=True)
    args=parser.parse_args()
    if args.visually_reviewed_pdf_sha256!=sha(ROOT/'output/pdf/main.pdf'):
        raise ValueError('reviewed PDF differs from current output')
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code','--quiet'],check=True)
    subprocess.run(['git','-C',str(ROOT),'diff','--cached','--exit-code','--quiet'],check=True)
    now=datetime.now(timezone.utc)
    start=datetime(2026,9,5,5,46,tzinfo=timezone.utc)
    roots=ROOT/'artifacts/remote_reduced_evidence_20260905'
    v1=json.loads((roots/'reduced_dev_v1/RESULT.json').read_text())
    v2=json.loads((roots/'reduced_dev_v2/RESULT.json').read_text())
    diagnostic=json.loads((PHASE/'checkpoint_diagnostic_v1.json').read_text())
    executions=[{'id':'interface_v1','wall_seconds':v1['wall_seconds'],'updates':0},
                {'id':'acquisition_v2','wall_seconds':v2['wall_seconds'],'updates':70},
                {'id':'posthoc_checkpoint_diagnostic','wall_seconds':diagnostic['elapsed_seconds'],'updates':0}]
    ledger={'as_of_utc':now.isoformat(),'allocation_start_utc':start.isoformat(),
            'allocation_hours_including_setup_io_idle':(now-start).total_seconds()/3600,
            'maximum_gh200_hours':100,'executions':executions,
            'execution_wall_gh200_hours':sum(e['wall_seconds'] for e in executions)/3600,
            'reserved_confirmation_opened':False,'further_training_queued':False,
            'last_host_status_sha256':sha(PHASE/'final_host_status.txt'),
            'provider_instance_terminated':False,
            'note':'Execution time is not billing. Conservative allocation accounting includes idle host time and continues while retained.'}
    (PHASE/'COMPUTE_LEDGER.json').write_text(json.dumps(ledger,indent=2),encoding='utf-8')
    qa=json.loads((ROOT/'artifacts/paper_render_20260905/qa.json').read_text())
    visual={'pdf_sha256':sha(ROOT/'output/pdf/main.pdf'),'pages_reviewed':list(range(1,qa['pages']+1)),
            'visual_inspection_performed':True,'clipping_or_overlap_observed':False,
            'main_text_within_nine_pages':True,'submission_qualified':False}
    (PHASE/'VISUAL_REVIEW.json').write_text(json.dumps(visual,indent=2),encoding='utf-8')
    relative_files=[
        'output/pdf/main.pdf','paper/generated/EVIDENCE.json',
        'artifacts/paper_render_20260905/qa.json',
        'artifacts/hindsight_reduced_remediation_20260905/joint_rule_frozen.json',
        'artifacts/hindsight_reduced_remediation_20260905/v2_prelaunch_checks.json',
        'artifacts/hindsight_reduced_remediation_20260905/tests_before_launch.txt',
        'artifacts/hindsight_reduced_remediation_20260905/model_setup.json',
        'artifacts/hindsight_reduced_remediation_20260905/environment_freeze.txt',
        'artifacts/hindsight_reduced_remediation_20260905/reduced_dev_v1_verification.json',
        'artifacts/hindsight_reduced_remediation_20260905/reduced_dev_v2_partial_acquisition_verification.json',
        'artifacts/hindsight_reduced_remediation_20260905/local_v2_partial_replay.log',
        'artifacts/hindsight_reduced_remediation_20260905/local_v2_portable_replay.json',
        'artifacts/hindsight_reduced_remediation_20260905/checkpoint_diagnostic_v1.json',
        'artifacts/hindsight_reduced_remediation_20260905/failed_replay_verifier_first.py',
        'artifacts/hindsight_reduced_remediation_20260905/COMPUTE_LEDGER.json',
        'artifacts/hindsight_reduced_remediation_20260905/VISUAL_REVIEW.json',
        'artifacts/hindsight_reduced_remediation_20260905/final_host_status.txt',
        'artifacts/public_response_reconstruction_20260905_v1/RECONSTRUCTION_RECEIPT.json',
        'artifacts/public_response_statistical_audit_20260905_v1/AGGREGATE_RESULTS.json',
        'artifacts/independent_audit_20260905/theory/PUBLIC_RESPONSE_PROTOCOL_SYNTHETIC_RECEIPT.json',
        'artifacts/independent_audit_20260905/data/execution_integrity_validation_20260905.json',
        'artifacts/deployment/reduced_dev_inputs_20260905.json',
    ]
    assert not any('persistence_sources' in n or 'LOCAL_ONLY' in n or n.endswith(('.csv','.rds')) for n in relative_files)
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    destination=ROOT/'artifacts/deployment'
    bundle=destination/f'hindsight_remediation_{head[:7]}.bundle'
    archive=destination/f'hindsight_remediation_receipts_{head[:7]}.zip'
    manifest_path=destination/f'hindsight_remediation_{head[:7]}.json'
    if any(p.exists() for p in (bundle,archive,manifest_path)):
        raise FileExistsError('preserve commit-named packages')
    subprocess.run(['git','-C',str(ROOT),'bundle','create',str(bundle),'HEAD'],check=True)
    subprocess.run(['git','-C',str(ROOT),'bundle','verify',str(bundle)],check=True)
    files={}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in relative_files:
            path=ROOT/name
            assert path.resolve().is_relative_to(ROOT.resolve())
            files[name]=sha(path)
            z.write(path,name)
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist())==set(files)
        assert all(hashlib.sha256(z.read(n)).hexdigest()==digest for n,digest in files.items())
    components=[bundle,archive,destination/'reduced_dev_inputs_20260905.zip',
                destination/'reduced_dev_v1_evidence.tar.gz',destination/'reduced_dev_v2_evidence.tar.gz',
                destination/'execution_logs_20260905.tar.gz',
                destination/'independent_audit_df34437.bundle',destination/'independent_audit_evidence_df34437.zip']
    report={'source_commit':head,'created_utc':now.isoformat(),'receipt_archive_files':files,
            'components':{str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in components},
            'public_participant_exports_packaged':False,'participant_numeric_tables_packaged':False,
            'model_weights_packaged':False,'neural_raw_logits_and_checkpoints_packaged':True,
            'original_v2_full_verification':False,'neural_acquisition_evidence_verified':True,
            'submission_qualified':False,'prior_audit_commit':'df3443713016321aacaf8cc7050f92aec0f1f3a1'}
    manifest_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for path in components:
        side=path.with_name(path.name+'.sha256')
        if not side.exists():side.write_text(sha(path)+'  '+path.name+'\n',encoding='utf-8')
    print(json.dumps({'manifest':str(manifest_path),'receipts':len(files),'ledger':ledger},indent=2))


if __name__=='__main__':main()
