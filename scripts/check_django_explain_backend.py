"""Check whether fixed downstream validation masks the proposed guard ablations."""
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from qualify_pyjwt_component import fetch,digest
from qualify_django_explain_component import FIX


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);receipt={};classes=[]
    for label,path,clsname in [('base','django/db/backends/base/operations.py','BaseDatabaseOperations'),('postgresql','django/db/backends/postgresql/operations.py','DatabaseOperations')]:
        url=f'https://raw.githubusercontent.com/django/django/{FIX}/{path}'
        raw=fetch(url);(a.out/f'{label}.py').write_bytes(raw);receipt[label]={'url':url,'sha256':digest(raw)}
        tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==clsname)
        body=[n for n in cls.body if (isinstance(n,ast.FunctionDef) and n.name=='explain_query_prefix') or
              (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'explain_options','explain_prefix'} for t in n.targets))]
        classes.append(ast.ClassDef(name=clsname,bases=[] if label=='base' else [ast.Name(id='BaseDatabaseOperations',ctx=ast.Load())],keywords=[],body=body,decorator_list=[]))
    ns={'NotSupportedError':RuntimeError}
    exec(compile(ast.fix_missing_locations(ast.Module(body=classes,type_ignores=[])),'pinned_backend_methods','exec'),ns)
    backend=ns['DatabaseOperations']();backend.connection=SimpleNamespace(features=SimpleNamespace(supports_explaining_query_execution=True,supported_explain_formats={'TEXT'}))
    results={}
    for name,options in [('ordinary',{'ANALYZE':True}),('empty',{}),('hyphen',{'custom-option':False}),('space',{'invalid option':True}),('comment',{'invalid--option':True})]:
        try:results[name]={'status':'prefix_built','value':backend.explain_query_prefix(format='TEXT',**options)}
        except ValueError as e:results[name]={'status':'backend_rejected','message':str(e)}
    assert all(results[n]['status']=='backend_rejected' for n in ['hyphen','space','comment'])
    assert all(results[n]['status']=='prefix_built' for n in ['ordinary','empty'])
    report={'classification':'DEVELOPMENTAL_DOWNSTREAM_MASKING_CHECK','fixed_revision':FIX,'results':results,
            'decision':'DO_NOT_COUNT_FRONTEND_ABLATIONS_AS_RETAINED_POSTGRESQL_SECURITY_DEFECTS',
            'scope':'Actual pinned base/PostgreSQL explain-prefix methods with static feature flags; no database. Both malformed probes are rejected even without frontend validation. Frontend component separation is insufficient to qualify incomplete repository repairs.'}
    (a.out/'DOWNLOAD.json').write_text(json.dumps(receipt,indent=2));(a.out/'RESULT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
