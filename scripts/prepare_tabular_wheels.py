"""Extend existing offline overlays without replacing their CUDA/PyTorch base."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from verify_calibration_wheels import missing_dependencies

PACKAGES=['scikit-learn==1.7.2','scipy==1.15.3','einops==0.8.2','joblib==1.5.3','threadpoolctl==3.6.0']


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    def prepare(pair):
        arch,py=pair;name=f'{arch}-cp{py}';dest=a.out/name;dest.mkdir(parents=True,exist_ok=False)
        old=a.old_root/name;manifest=json.loads((old/'WHEELS.json').read_text())
        for filename,sha in manifest['files'].items():
            if hashlib.sha256((old/filename).read_bytes()).hexdigest()!=sha:raise ValueError('old wheel hash differs')
            shutil.copyfile(old/filename,dest/filename)
        command=[sys.executable,'-m','pip','download','--disable-pip-version-check','--only-binary=:all:','--no-deps',
            '--platform','manylinux_2_28_'+arch,'--platform','manylinux2014_'+arch,'--python-version',py,
            '--implementation','cp','--abi','cp'+py,'--abi','abi3','--abi','none','--dest',str(dest),*PACKAGES]
        subprocess.run(command,check=True,capture_output=True,text=True)
        missing=missing_dependencies(dest,arch,py)
        if missing:raise ValueError(str(missing))
        report={'architecture':arch,'python':py,'tabular_packages':PACKAGES,'torch_supplied_by_host':True,
                'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dest.glob('*.whl'))}}
        (dest/'WHEELS.json').write_text(json.dumps(report,indent=2))
        return {'target':name,'wheels':len(report['files']),'dependency_closure_verified':True}
    with ThreadPoolExecutor(max_workers=4) as executor:
        for result in executor.map(prepare,[(arch,py) for arch in ('x86_64','aarch64') for py in ('310','312')]):
            print(json.dumps(result),flush=True)


if __name__=='__main__':main()
