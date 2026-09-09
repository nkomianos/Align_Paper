"""Static label prerequisites on paired DEV source; never executes solutions."""
import argparse,ast,hashlib,json,re,zipfile
from collections import Counter,defaultdict
from pathlib import Path


def assertion_conflicts(code):
    try:tree=ast.parse(code)
    except SyntaxError:return {'parseable':False,'conflicting_literal_assertion_inputs':None}
    pairs=defaultdict(list)
    for node in ast.walk(tree):
        if not isinstance(node,ast.Assert):continue
        c=node.test
        if not (isinstance(c,ast.Compare) and len(c.ops)==1 and isinstance(c.ops[0],ast.Eq)
                and isinstance(c.left,ast.Call)):continue
        try:value=ast.literal_eval(c.comparators[0])
        except (ValueError,TypeError,SyntaxError,MemoryError,RecursionError):continue
        values=pairs[ast.dump(c.left,include_attributes=False)]
        # Match Python equality in the assertion: 1, 1.0 and True do not
        # constitute conflicting expected outputs merely because repr differs.
        if not any(value==prior for prior in values):values.append(value)
    return {'parseable':True,'conflicting_literal_assertion_inputs':sum(len(v)>1 for v in pairs.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('fresh report required')
    reports=[]
    with zipfile.ZipFile(a.archive) as z:
        for name in z.namelist():
            if '/training_time/' not in name or '/train_paired_' not in name or not name.endswith('.jsonl'):continue
            rows=[];groups=defaultdict(list)
            with z.open(name) as f:
                for index,line in enumerate(f):
                    r=json.loads(line);gt=r['reward_model']['ground_truth']
                    gt=json.loads(gt) if isinstance(gt,str) else gt
                    blocks=re.findall(r'```python\n(.*?)```',r['response'].split('</think>')[1] if '</think>' in r['response'] else r['response'],re.S)
                    calls=False;code_parsed=False
                    if blocks:
                        try:
                            tree=ast.parse(blocks[-1]);code_parsed=True
                            calls=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                                and isinstance(n.func.value,ast.Name) and n.func.value.id=='unittest'
                                and n.func.attr=='main' for n in ast.walk(tree))
                        except SyntaxError:pass
                    ph=hashlib.sha256(r['prompt'].encode()).hexdigest()
                    gh=hashlib.sha256(json.dumps(gt,sort_keys=True).encode()).hexdigest()
                    groups[ph].append(gh)
                    rows.append({'row':index,'label':r['hacking_label'],'type':r['hacking_type'],
                        'prompt_sha256':ph,'ground_truth_sha256':gh,
                        'ground_truth_fields':sorted(gt),'reward':r['reward'],
                        'code_parseable':code_parsed,'has_unittest_main_call':calls,
                        'assertion_audit':assertion_conflicts(gt['test_code']) if 'test_code' in gt else None})
            reports.append({'file':name,'rows':rows,'same_prompt_multiple_ground_truths':sum(len(set(v))>1 for v in groups.values()),
                'label_reward_counts':dict(Counter(f"{r['label']}:{r['reward']}" for r in rows)),
                'unittest_main_by_label':{str(label):sum(r['has_unittest_main_call'] for r in rows if r['label']==label) for label in (0,1)}})
    with a.archive.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    report={'archive_sha256':sha,'files':reports,'code_executed':False,
        'limitations':['AST call presence does not prove execution or lack of reassignment.',
            'Syntactically conflicting literal assertions assume a deterministic call; no solver/evaluator replay.',
            'Absence of this narrow assertion pattern does not prove satisfiable ground truth.',
            'Current source revision is not yet bound to historical collection.']}
    with a.out.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in reports],indent=2))


if __name__=='__main__':main()
