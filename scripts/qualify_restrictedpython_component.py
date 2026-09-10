"""Pure attribute-guard component check; no untrusted code or exploit execution."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from run_unexplored_screens import dump,sha


def function(path):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='safer_getattr')


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.source/'DOWNLOAD.json').read_text())
    for row in receipt['files']:assert sha(a.source/row['file'])==row['sha256']
    old=function(a.source/'vulnerable_Guards.py');fixed=function(a.source/'fixed_Guards.py')
    results={};variants={}
    for arm in ['vulnerable','instance_map_only','class_format_only','fixed']:
        node=copy.deepcopy(old if arm=='vulnerable' else fixed)
        first=next(n for n in node.body if isinstance(n,ast.If))
        if arm=='instance_map_only':
            first.test=ast.parse("isinstance(object, str) and name in ('format', 'format_map')",mode='eval').body
        elif arm=='class_format_only':
            first.test=ast.parse("name == 'format' and (isinstance(object, str) or (isinstance(object, type) and issubclass(object, str)))",mode='eval').body
        module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
        variants[arm]=ast.unparse(module);namespace={};exec(compile(module,'pinned_guard_component','exec'),namespace)
        guard=namespace['safer_getattr'];checks={}
        for label,obj,attr,exception in [
            ('instance_format_map','{value}','format_map',NotImplementedError),
            ('class_format',str,'format',NotImplementedError),
            ('legacy_instance_format','{value}','format',NotImplementedError),
            ('private_attribute',SimpleNamespace(value=2),'__class__',AttributeError)]:
            try:guard(obj,attr)
            except exception:checks[label]=True
            else:checks[label]=False
        checks['public_attribute']=guard(SimpleNamespace(value=2),'value')==2
        checks['missing_default']=guard(SimpleNamespace(),'missing',17)==17
        checks['benign_string_method']=guard('abc','upper')()=='ABC'
        assert all(checks[k] for k in ['legacy_instance_format','private_attribute','public_attribute','missing_default','benign_string_method'])
        assert checks['instance_format_map']==(arm in ['instance_map_only','fixed'])
        assert checks['class_format']==(arm in ['class_format_only','fixed'])
        results[arm]=checks
    dump(a.out,{'classification':'DEVELOPMENTAL_COMPONENT_QUALIFICATION','cve':'CVE-2023-41039',
        'source':receipt,'results':results,'variants':variants,'independent_task_count':1,
        'limitations':['Two controlled incomplete repairs, not model-produced patches.',
            'Pure safer_getattr component; string.Formatter delegation and full restricted compiler not exercised.',
            'No secret read, dynamic exploit, container test, or neural experiment.',
            'Two defect probes are one task, not independent study replicates.']})
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
