"""Verify post-result same-case reasoning control and realized token use."""
import argparse
import json
from pathlib import Path
from run_benign_memory_extraction import inputs,parse,solve
from run_unexplored_screens import dump,sha


def read(root):
    for name,h in json.loads((root/'MANIFEST.json').read_text()).items():assert Path(name).name==name and sha(root/name)==h
    assert json.loads((root/'INPUTS.json').read_text())==inputs()
    return [json.loads(x) for x in (root/'OUTPUTS.jsonl').read_text().splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--original',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    rows=read(a.run);original=read(a.original);gold={r['id']:r for r in inputs()}
    assert len(original)==96 and {(r['id'],r['arm']) for r in original}=={(identifier,arm) for identifier in gold for arm in ['direct','extract']}
    assert len(rows)==len({r['id'] for r in rows})==48 and {r['id'] for r in rows}==set(gold)
    assert json.loads((a.run/'MODEL.json').read_text())['files']==json.loads((a.original/'MODEL.json').read_text())['files']
    records=[]
    for r in rows:
        v=parse(r['text']);assert v==r['parsed']
        valid=isinstance(v,dict) and set(v)=={'reasoning','answer'} and isinstance(v['reasoning'],str) and v['answer'] in ['YES','NO','CLARIFY']
        records.append({'id':r['id'],'mechanism':gold[r['id']]['mechanism'],'valid':valid,'eos':r['eos'],'answer':v['answer'] if valid else None,'correct':bool(valid and v['answer']==gold[r['id']]['gold']),'tokens':len(r['output_ids'])})
    metrics={k:sum(r[k] for r in records)/48 for k in ['valid','eos','correct','tokens']}
    qualified=metrics['valid']>=.95 and metrics['eos']>=.95
    extraction_correct=0
    for r in original:
        if r['arm']!='extract':continue
        v=parse(r['text']);assert v==r['parsed']
        try:
            if not isinstance(v,dict) or set(v)!={'values','edges'}:continue
            extraction_correct+=solve(v['values'],v['edges'],gold[r['id']]['threshold'])==gold[r['id']]['gold']
        except (ValueError,TypeError,IndexError):pass
    extraction_accuracy=extraction_correct/48
    original_tokens={arm:sum(len(r['output_ids']) for r in original if r['arm']==arm)/48 for arm in ['direct','extract']}
    report={'classification':'POST_RESULT_REASONING_BUDGET_DIAGNOSTIC','metrics':metrics,'records':records,
            'original_mean_output_tokens':original_tokens,'original_extraction_accuracy':extraction_accuracy,
            'route':'INVALID_CONTROL_INTERFACE' if not qualified else 'ADVANTAGE_SURVIVES_CONTROL_REQUIRES_FRESH_DATA' if extraction_accuracy-metrics['correct']>=.10 else 'STOP_ADVANTAGE_CLOSED_BY_REASONING_CONTROL',
            'scope':'Same synthetic inputs after original result; not fresh confirmation or matched realized compute. Original extraction answers recomputed from saved outputs. No population or paper claim.',
            'manifest_sha256':sha(a.run/'MANIFEST.json')}
    dump(a.out,report);print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
