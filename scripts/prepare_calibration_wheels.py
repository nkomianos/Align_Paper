"""Download pinned application wheels locally; never installs or connects to GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from verify_calibration_wheels import missing_dependencies

PACKAGES=['transformers==5.15.0','tokenizers==0.22.2','numpy==1.26.4',
          'accelerate==1.14.0','huggingface-hub==1.27.0','safetensors==0.8.0',
          'pillow==12.3.0','Jinja2==3.1.6']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arch',choices=['x86_64','aarch64'],required=True)
    p.add_argument('--python-version',choices=['310','312'],required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,'-m','pip','download','--disable-pip-version-check','--only-binary=:all:',
        '--platform','manylinux_2_28_'+a.arch,'--platform','manylinux2014_'+a.arch,
        '--python-version',a.python_version,'--implementation','cp','--abi','cp'+a.python_version,
        '--abi','abi3','--abi','none','--dest',str(a.out)]
    # Resolve application dependencies without pulling a multi-GB replacement
    # CUDA/PyTorch stack. Accelerate's non-torch requirements are explicit here.
    subprocess.run([*command,*[x for x in PACKAGES if not x.startswith('accelerate')],
                    'packaging>=20.0','psutil','pyyaml'],check=True)
    subprocess.run([*command,'--no-deps','accelerate==1.14.0'],check=True)
    for attempt in range(5):
        missing=missing_dependencies(a.out,a.arch,a.python_version)
        if not missing:break
        subprocess.run([*command,'--no-deps',*missing],check=True)
    else:raise RuntimeError('target dependency closure unresolved')
    result={'packages':PACKAGES,'architecture':a.arch,'python':a.python_version,
        'scope':'resolved application overlay; CUDA torch and its base dependencies supplied by host image',
        'target_dependency_markers_checked':True,
        'files':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.out.glob('*.whl'))}}
    (a.out/'WHEELS.json').write_text(json.dumps(result,indent=2),encoding='utf8')


if __name__=='__main__':main()
