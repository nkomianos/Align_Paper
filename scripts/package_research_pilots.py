"""Seal committed source and CPU receipts for offline exploratory deployment."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile
from research_pilots.common import write,digest
from research_pilots.data import build,validate
from verify_research_pilot import verify


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--receipts',type=Path,required=True)
    p.add_argument('--calibration-package',type=Path,required=True)
    p.add_argument('--tabular-root',type=Path)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    git=lambda *args:subprocess.check_output(['git','-C',str(repo),*args])
    if git('diff','--name-only') or git('diff','--cached','--name-only'):
        raise ValueError('commit tracked edits first')
    # Unrelated user analysis is intentionally excluded; new executable code must be committed.
    for name in git('ls-files','--others','--exclude-standard','scripts','src','tests','docs').decode().splitlines():
        raise ValueError('uncommitted source: '+name)
    frozen=validate(json.loads((a.receipts/'frozen_inputs/INPUTS.json').read_text()))
    if digest(frozen)!=digest(build()):raise ValueError('packaged inputs differ from executable defaults')
    clara=verify(a.receipts/'clara_cpu_final')
    previous=json.loads((a.calibration_package/'PACKAGE.json').read_text())
    inherited=['learning_only.zip']+[n for n in previous['files'] if n.endswith('_wheels.zip')]
    for name in inherited:
        if sha(a.calibration_package/name)!=previous['files'][name]['sha256']:
            raise ValueError('existing package component hash differs: '+name)
    a.out.mkdir(parents=True,exist_ok=False)
    head=git('rev-parse','HEAD').decode().strip()
    subprocess.run(['git','-C',str(repo),'bundle','create',str((a.out/'source.bundle').resolve()),'HEAD'],check=True)
    subprocess.run(['git','-C',str(repo),'bundle','verify',str((a.out/'source.bundle').resolve())],check=True,capture_output=True)
    subprocess.run(['git','-C',str(repo),'archive','--format=zip','--output='+str((a.out/'source.zip').resolve()),'HEAD'],check=True)
    for name in inherited:shutil.copyfile(a.calibration_package/name,a.out/name)
    with zipfile.ZipFile(a.out/'cpu_receipts.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
        paths=[a.receipts/'CPU_TESTS.txt',a.receipts/'POSIX_SUPERVISOR.json']
        paths+=list((a.receipts/'frozen_inputs').glob('*.json'))
        paths+=list((a.receipts/'clara_cpu_final').glob('*.json'))
        for path in paths:archive.write(path,str(path.relative_to(a.receipts)))
    write(a.out/'pilot_inputs.json',frozen)
    shutil.copyfile(repo/'docs/RESEARCH_PILOTS_IMPLEMENTATION_20260906.md',a.out/'RUNBOOK.md')
    if a.tabular_root:
        from verify_tabular_drift import verify as verify_tabular
        from research_pilots.tabular_drift import validate as validate_tabular
        verify_tabular(a.tabular_root/'tabicl_cpu_smoke')
        validate_tabular(json.loads((a.tabular_root/'frozen/INPUTS.json').read_text()))
        assets=json.loads((a.tabular_root/'assets/ASSETS.json').read_text())
        for name,entry in assets['files'].items():
            if sha(a.tabular_root/'assets'/name)!=entry['sha256']:raise ValueError('tabular asset hash differs')
            shutil.copyfile(a.tabular_root/'assets'/name,a.out/name)
        shutil.copyfile(a.tabular_root/'frozen/INPUTS.json',a.out/'tabular_inputs.json')
        shutil.copyfile(repo/'docs/IDEA_TRIAGE_AND_RUNTIME_POLICY_20260907.md',a.out/'CURRENT_POLICY.md')
        shutil.copyfile(repo/'docs/IDEA_TRIAGE_AND_RUNTIME_POLICY_20260907.md',a.out/'IDEA_TRIAGE_AND_RUNTIME_POLICY_20260907.md')
        for directory in (a.tabular_root/'wheels').iterdir():
            manifest=json.loads((directory/'WHEELS.json').read_text())
            with zipfile.ZipFile(a.out/(directory.name+'_wheels.zip'),'w',compression=zipfile.ZIP_STORED) as archive:
                for name,expected in manifest['files'].items():
                    if sha(directory/name)!=expected:raise ValueError('tabular wheel hash differs')
                    archive.write(directory/name,name)
                archive.write(directory/'WHEELS.json','WHEELS.json')
        with zipfile.ZipFile(a.out/'tabular_cpu_receipts.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            paths=list((a.tabular_root/'tabicl_cpu_smoke').glob('*.json'))
            paths += [a.tabular_root/'CONTEXT_RESET.json',a.tabular_root/'CPU_TESTS.txt',a.tabular_root/'COMPLETION_POLICY.json']
            for path in paths:archive.write(path,str(path.relative_to(a.tabular_root)))
    report={'status':'CPU_VERIFIED_GPU_QUALIFICATION_PENDING','commit':head,
        'created_utc':datetime.now(timezone.utc).isoformat(),'gpu_contacted':False,'paper_green_light':False,
        'model_weights_included':bool(a.tabular_root),'qwen_weights_included':False,
        'monitor_data_included':False,'budget_target_allocated_h200_hours':50,'mid_run_timeout':False,
        'clara_verification':clara,'reused_calibration_commit':previous['commit'],
        'files':{f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(a.out.iterdir()) if f.is_file()}}
    write(a.out/'PACKAGE.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
