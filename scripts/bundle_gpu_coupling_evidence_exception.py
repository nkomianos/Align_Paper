"""After process exit, package immutable evidence and small verification inputs.

Run remotely with root and a fresh destination directory. Large model weights
are rehashed but not copied. Does not modify source/evidence/checkpoints.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from coupling_manifest_exception import validate


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def bundle(root,destination,source_root):
    root=root.resolve();source_root=source_root.resolve()
    exit_file=Path(str(root)+'.exit')
    assert exit_file.is_file() and exit_file.read_text().strip()=='0', 'Run not successfully exited'
    pid_file=Path(str(root)+'.pid')
    if pid_file.exists():
        pid=int(pid_file.read_text().strip())
        assert not Path('/proc',str(pid)).exists(), 'Recorded process still exists'
    manifest=json.loads((root/'MANIFEST.json').read_text())
    assert all(digest(root/name)==sha for name,sha in manifest.items())
    frozen=json.loads((root/'FROZEN.json').read_text());files={};maps={};receipt={}
    def add(path,member):
        path=Path(path);assert path.is_file()
        assert not member.startswith('/') and '..' not in Path(member).parts
        if member in files:assert files[member]==path
        files[member]=path
    for p in root.iterdir():
        assert p.is_file(), 'Unexpected nested evidence directory'
        add(p,'evidence/'+p.name)
    for suffix in ('.log','.exit','.pid'):
        path=Path(str(root)+suffix)
        if path.exists():add(path,'sidecars/'+path.name)
    for original,sha in frozen['sources'].items():
        path=Path(original)
        if not path.is_absolute():path=source_root/path
        assert digest(path)==sha
        relative=path.resolve().relative_to(source_root)
        member='source/'+relative.as_posix();add(path,member);maps[original]=member
    prepared=Path(frozen['config']['prepared'])
    assert digest(prepared/'MANIFEST.json')==frozen['prepared_manifest_sha']
    prepared_manifest=json.loads((prepared/'MANIFEST.json').read_text())
    exception=validate(prepared,frozen['prepared_manifest_sha'],public_only=True)
    for name,sha in prepared_manifest.items():
        if name=='MANIFEST.json':continue  # Exact approved exception validated above.
        path=prepared/name
        # The generator may have received public inputs only. The matching local
        # private answer key can be supplied separately if absent remotely.
        if not path.exists():
            assert name=='answer_key.json';continue
        assert digest(path)==sha;add(path,'prepared/'+name)
    add(prepared/'MANIFEST.json','prepared/MANIFEST.json');maps[str(prepared)]='prepared'
    for model in frozen['config']['models']:
        path=Path(model['path']);maps[str(path)]='models/'+model['name']
        for name,sha in frozen['model_files'][model['name']].items():
            assert digest(path/name)==sha, 'Pinned model changed'
            if name.endswith('.safetensors'):receipt[str(path/name)]=sha
            else:add(path/name,'models/'+model['name']+'/'+name)
        for p in path.iterdir():
            if p.is_file() and p.suffix in ('.txt','.model'):
                add(p,'models/'+model['name']+'/'+p.name)
    # Confirm evidence did not change while hashing potentially large weights.
    assert all(digest(root/name)==sha for name,sha in manifest.items())
    destination.mkdir(parents=True,exist_ok=False)
    for name,obj in [('preparation_manifest_exception.json',exception),('mapping.relative.json',maps),('weight_receipt.json',receipt),
                     ('bundle_files.sha256.json',{name:digest(p) for name,p in files.items()})]:
        with (destination/name).open('x') as f:json.dump(obj,f,indent=2)
    archive=destination/'evidence_bundle.tar.gz'
    with tarfile.open(archive,'x:gz',dereference=True) as tar:
        for member,path in sorted(files.items()):tar.add(path,arcname=member,recursive=False)
    result={'archive':str(archive),'archive_sha256':digest(archive),
            'files':len(files),'weights_rehashed_not_copied':len(receipt),
            'metadata':{p.name:digest(p) for p in destination.glob('*.json')}}
    with (destination/'BUNDLE.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    p.add_argument('destination',type=Path);p.add_argument('source_root',type=Path)
    a=p.parse_args();bundle(a.root,a.destination,a.source_root)
