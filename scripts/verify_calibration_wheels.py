"""Check wheel dependency markers for the target, not the preparation computer."""
import argparse
import email
import json
from pathlib import Path
import zipfile
from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name


def missing_dependencies(directory,arch,python_version):
    version=python_version[0]+'.'+python_version[1:]
    environment={**default_environment(),'python_version':version,'python_full_version':version+'.0',
        'sys_platform':'linux','os_name':'posix','platform_system':'Linux','platform_machine':arch,'extra':''}
    packages={}
    for path in Path(directory).glob('*.whl'):
        with zipfile.ZipFile(path) as z:
            metadata=email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('/METADATA'))))
        name=canonicalize_name(metadata['Name'])
        if name in packages:raise ValueError('duplicate wheel version: '+name)
        if metadata['Requires-Python'] and not SpecifierSet(metadata['Requires-Python']).contains(version+'.0'):
            raise ValueError('wheel Python requirement mismatch: '+name)
        packages[name]=metadata
    missing=set()
    for metadata in packages.values():
        for raw in metadata.get_all('Requires-Dist',[]):
            requirement=Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate(environment):continue
            name=canonicalize_name(requirement.name)
            if name=='torch':continue
            if name not in packages:missing.add(requirement.name+str(requirement.specifier))
            elif not requirement.specifier.contains(packages[name]['Version']):
                raise ValueError('incompatible dependency: '+raw)
    return sorted(missing)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--arch',required=True);p.add_argument('--python-version',required=True)
    a=p.parse_args();missing=missing_dependencies(a.directory,a.arch,a.python_version)
    print(json.dumps({'missing':missing,'torch_supplied_by_image':True}));raise SystemExit(bool(missing))
