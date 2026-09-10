"""Read/write guard ablation on synthetic local objects; no exploit payloads."""
import argparse
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from qualify_pyjwt_component import fetch, digest, normalized, DATA_SHA

FIX='6ff0831ad285fff937cafd2a853f20cc9ae92021'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data=a.dataset.read_bytes();assert digest(data)==DATA_SHA
    case=next(x for x in json.loads(data) if x['cve_id']=='CVE-2023-26145')
    a.out.mkdir(parents=True,exist_ok=False)
    url=f'https://api.github.com/repos/dgilland/pydash/commits/{FIX}'
    raw=fetch(url);commit=json.loads(raw);assert commit['sha']==FIX and len(commit['parents'])==1
    parent=commit['parents'][0]['sha'];(a.out/'commit.json').write_bytes(raw)
    receipt={'commit.json':{'url':url,'sha256':digest(raw)}};trees={}
    for label,revision in [('vulnerable',parent),('fixed',FIX)]:
        url=f'https://raw.githubusercontent.com/dgilland/pydash/{revision}/src/pydash/helpers.py'
        raw=fetch(url);(a.out/f'{label}.py').write_bytes(raw)
        receipt[f'{label}.py']={'url':url,'sha256':digest(raw)};trees[label]=ast.parse(raw)
    (a.out/'DOWNLOAD.json').write_text(json.dumps(receipt,indent=2))
    functions={label:{n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)} for label,tree in trees.items()}
    for label,field in [('vulnerable','vul_func'),('fixed','fix_func')]:
        for snippet in case[field][:2]:
            node=ast.parse(snippet['snippet']).body[0]
            assert normalized(node)==normalized(functions[label][node.name])
    outputs={}
    for arm in ['vulnerable','read_only','write_only','fixed']:
        names=['_base_get_item','_base_get_object','base_set','_raise_if_restricted_key']
        selected=[]
        for name in names:
            source='fixed' if (name=='_raise_if_restricted_key' or arm=='fixed' or (name=='_base_get_object' and arm=='read_only') or (name=='base_set' and arm=='write_only')) else 'vulnerable'
            selected.append(copy.deepcopy(functions[source][name]))
        ns={'UNSET':object()}
        exec(compile(ast.Module(body=selected,type_ignores=[]),'pinned_pydash_helpers','exec'),ns)
        obj=SimpleNamespace(public='unchanged',__audit_marker__='synthetic')
        def blocked(call):
            try:call();return False
            except KeyError:return True
        results={'restricted_read_blocked':blocked(lambda:ns['_base_get_object'](obj,'__audit_marker__')),
                 'restricted_write_blocked':blocked(lambda:ns['base_set'](obj,'__audit_marker__','new'))}
        assert results['restricted_read_blocked']==(arm in {'read_only','fixed'})
        assert results['restricted_write_blocked']==(arm in {'write_only','fixed'})
        assert ns['_base_get_object'](obj,'public')=='unchanged'
        assert ns['_base_get_object'](obj,'missing',default='fallback')=='fallback'
        ns['base_set'](obj,'public','updated');assert obj.public=='updated'
        ns['base_set'](obj,'public','ignored',allow_override=False);assert obj.public=='updated'
        mapping={};ns['base_set'](mapping,'ordinary',1);assert mapping=={'ordinary':1}
        sequence=[1];ns['base_set'](sequence,2,3);assert sequence==[1,None,3]
        results['benign_regressions_passed']=6
        outputs[arm]=results
    report={'classification':'DEVELOPMENTAL_NATURAL_COMPONENT_QUALIFICATION','case':case['cve_id'],
            'fixed_revision':FIX,'vulnerable_revision':parent,'dataset_sha256':DATA_SHA,'results':outputs,
            'independent_task_count':1,
            'scope':'Synthetic attributes only; actual pinned helper functions, two separately repairable read/write omissions. No traversal exploit, globals access, full package regression suite or model outputs. Attribute-guard category overlaps RestrictedPython; not a new fourth category.'}
    (a.out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
