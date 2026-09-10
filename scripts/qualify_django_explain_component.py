"""Pinned option-name validation replay with a recording compiler, no database."""
import argparse
import ast
import copy
import json
from pathlib import Path
import re
import textwrap
from types import SimpleNamespace
from qualify_pyjwt_component import fetch, digest, normalized, DATA_SHA

FIX='00b0fc50e1738c7174c495464a5ef069408a4402'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.dataset.read_bytes();assert digest(raw)==DATA_SHA
    case=next(x for x in json.loads(raw) if x['cve_id']=='CVE-2022-28347')
    a.out.mkdir(parents=True,exist_ok=False)
    url=f'https://api.github.com/repos/django/django/commits/{FIX}'
    raw=fetch(url);commit=json.loads(raw);assert commit['sha']==FIX and len(commit['parents'])==1
    parent=commit['parents'][0]['sha'];(a.out/'commit.json').write_bytes(raw)
    receipt={'commit.json':{'url':url,'sha256':digest(raw)}};methods={};pattern=None
    for label,rev,field in [('vulnerable',parent,'vul_func'),('fixed',FIX,'fix_func')]:
        url=f'https://raw.githubusercontent.com/django/django/{rev}/django/db/models/sql/query.py'
        raw=fetch(url);(a.out/f'{label}.py').write_bytes(raw);receipt[f'{label}.py']={'url':url,'sha256':digest(raw)}
        tree=ast.parse(raw)
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Query')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='explain')
        snippet=next(x['snippet'] for x in case[field] if 'def explain(self,' in x['snippet'])
        assert normalized(method)==normalized(ast.parse(textwrap.dedent(snippet)).body[0])
        methods[label]=method
        if label=='fixed':
            pattern=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPLAIN_OPTIONS_PATTERN' for t in n.targets))
    (a.out/'DOWNLOAD.json').write_text(json.dumps(receipt,indent=2))
    results={}
    for arm in ['vulnerable','character_only','comment_only','fixed']:
        method=copy.deepcopy(methods['vulnerable' if arm=='vulnerable' else 'fixed'])
        if arm.endswith('_only'):
            loop=next(n for n in method.body if isinstance(n,ast.For))
            guard=loop.body[0];assert isinstance(guard,ast.If) and isinstance(guard.test,ast.BoolOp) and len(guard.test.values)==2
            guard.test=guard.test.values[0 if arm=='character_only' else 1]
        ns={'_lazy_re_compile':re.compile,'ExplainInfo':lambda format,options:SimpleNamespace(format=format,options=options)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(pattern),method],type_ignores=[])),'pinned_query_explain','exec'),ns)
        outputs={}
        for name,options in [('ordinary',{'ANALYZE':True}),('empty',{}),('hyphen',{'custom-option':False}),('space',{'invalid option':True}),('comment',{'invalid--option':True})]:
            calls=[]
            class QueryStub:
                def clone(self): return self
                def get_compiler(self,using):
                    calls.append({'using':using,'options':self.explain_info.options,'format':self.explain_info.format})
                    return SimpleNamespace(explain_query=lambda:['SYNTHETIC PLAN'])
            try:
                value=ns['explain'](QueryStub(),'recording',format='TEXT',**options)
                assert value=='SYNTHETIC PLAN' and calls==[{'using':'recording','options':options,'format':'TEXT'}]
                status='compiler_reached'
            except ValueError:
                assert not calls
                status='validation_rejected'
            outputs[name]=status
        assert all(outputs[n]=='compiler_reached' for n in ['ordinary','empty','hyphen'])
        assert outputs['space']==('validation_rejected' if arm in {'character_only','fixed'} else 'compiler_reached')
        assert outputs['comment']==('validation_rejected' if arm in {'comment_only','fixed'} else 'compiler_reached')
        results[arm]=outputs
    report={'classification':'DEVELOPMENTAL_NATURAL_COMPONENT_QUALIFICATION','case':case['cve_id'],
            'fixed_revision':FIX,'vulnerable_revision':parent,'dataset_sha256':DATA_SHA,'results':results,
            'independent_task_count':1,
            'scope':'Actual Query.explain and option regex; recording compiler, no SQL execution. Separately repairable character/comment guards in one lexical-validation component. Does not attest PostgreSQL backend behavior, query cloning, complete CVE repair, full regressions or model outputs.'}
    (a.out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
