"""Create a commit-bound source bundle, learning-only input and optional wheel ZIPs."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from interaction_sprint.hindsight_calibration import prepare,digest
from interaction_sprint.hindsight_execution_integrity import capture_execution_provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learning',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args])
    if git('diff','--name-only') or git('diff','--cached','--name-only'):raise RuntimeError('commit tracked edits first')
    cfg=json.loads((repo/'configs/hindsight_calibration_v1.json').read_text())
    roots=['scripts/run_hindsight_calibration.py','scripts/launch_hindsight_calibration.py',
        'scripts/verify_hindsight_calibration.py','scripts/package_hindsight_calibration.py',
        'scripts/prepare_calibration_wheels.py','scripts/verify_calibration_wheels.py','scripts/check_calibration_host.py',
        'scripts/test_calibration_supervisor_posix.py','scripts/preflight_calibration_tokens.py','src/interaction_sprint/__init__.py',
        'src/latent_contract/__init__.py','tests/test_hindsight_calibration.py',
        'configs/hindsight_calibration_v1.json','docs/HINDSIGHT_CALIBRATION_PROTOCOL_20260905.md',
        'docs/ICLR_50H_RESEARCH_SHORTLIST_20260905.md','docs/CALIBRATION_FAST_DEPLOY_20260905.md']
    provenance=capture_execution_provenance(repo,roots,[],cfg)
    exported={}; eol_only=[]
    for name,sha in provenance['source_sha256'].items():
        blob=git('show','HEAD:'+name);exported[name]=digest(blob)
        if digest(blob)!=sha:
            if (repo/name).read_bytes().replace(b'\r\n',b'\n')!=blob.replace(b'\r\n',b'\n'):
                raise ValueError('source differs from committed content: '+name)
            eol_only.append(name)
    provenance['preparation_worktree_source_sha256']=provenance['source_sha256']
    provenance['source_sha256']=exported
    provenance['preparation_to_git_line_ending_only_differences']=eol_only
    provenance['export_scope']='source_sha256 binds Git blobs in bundle; preparation hashes retained separately'
    selected=prepare(a.learning)
    a.out.mkdir(parents=True,exist_ok=False)
    head=git('rev-parse','HEAD').decode().strip()
    bundle=a.out/'source.bundle'
    subprocess.run(['git','-C',str(repo),'bundle','create',str(bundle.resolve()),'HEAD'],check=True)
    subprocess.run(['git','-C',str(repo),'bundle','verify',str(bundle.resolve())],check=True,capture_output=True)
    with zipfile.ZipFile(a.out/'learning_only.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('learning.json',a.learning.read_bytes())
    with zipfile.ZipFile(a.out/'learning_only.zip') as z:
        if z.namelist()!=['learning.json'] or digest(z.read('learning.json'))!=selected['source_sha256']:
            raise ValueError('data allowlist/hash differs')
    wheels=repo/'artifacts/calibration_readiness_20260905/wheels'
    for directory in sorted(wheels.iterdir()):
        manifest=json.loads((directory/'WHEELS.json').read_text())
        with zipfile.ZipFile(a.out/(directory.name+'_wheels.zip'),'x',compression=zipfile.ZIP_STORED) as z:
            for name,sha in manifest['files'].items():
                data=(directory/name).read_bytes()
                if digest(data)!=sha:raise ValueError('wheel hash differs')
                z.writestr(name,data)
            z.writestr('WHEELS.json',(directory/'WHEELS.json').read_bytes())
    (a.out/'SOURCE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    receipts=repo/'artifacts/calibration_readiness_20260905'
    with zipfile.ZipFile(a.out/'cpu_receipts.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('CPU_TESTS.txt','POSIX_SUPERVISOR.json','TOKEN_PREFLIGHT.json'):
            z.writestr(name,(receipts/name).read_bytes())
    (a.out/'SPLIT_RECEIPT.json').write_text(json.dumps({
        'source_sha256':selected['source_sha256'],
        'train_base_ids':sorted({r['base_id'] for r in selected['train']}),
        'holdout_base_ids':sorted({r['base_id'] for r in selected['holdout']}),
        'unused_bases':len(selected['unused_base_ids']),
        'independent_confirmation':False,'old_dev_packaged':False,'confirmation_packaged':False},indent=2),encoding='utf8')
    files={f.name:{'sha256':digest(f.read_bytes()),'bytes':f.stat().st_size} for f in a.out.iterdir() if f.is_file()}
    report={'status':'READY_FOR_HOST_PREFLIGHT_NOT_GPU_EXECUTED','commit':head,
        'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,'budget_h200_hours':50,
        'first_run_hard_cap_hours':1,'gpu_contacted':False,'model_weights_included':False,
        'wheel_overlay_is_complete_cuda_environment':False,'paper_green_light':False}
    (a.out/'PACKAGE.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
