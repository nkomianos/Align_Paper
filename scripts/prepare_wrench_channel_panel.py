"""Paired information-channel views; no outcome/label fields in model inputs."""
import argparse
import copy
import json
from pathlib import Path
from tokenizers import Tokenizer
from run_unexplored_screens import dump,sha


def view(trace,stripped):
    result=[]
    for step in trace['steps']:
        if step['source']=='user':continue
        if step['source']!='agent':raise ValueError('unreviewed source kind')
        r={k:copy.deepcopy(step[k]) for k in ('step_id','tool_calls','observation') if k in step}
        if not stripped and step.get('message'):r['message']=step['message']
        result.append(r)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.source/'DOWNLOAD.json').read_text());selection=json.loads((a.source/'SELECTION.json').read_text())
    assert sha(a.source/'SELECTION.json')==receipt['selection_sha256']
    for r in receipt['files']:assert sha(a.source/r['file'])==r['sha256']
    tok=Tokenizer.from_file(str(a.tokenizer));rows=[]
    for i,selected in enumerate(selection['rows']):
        instruction=(a.source/f'{i:02d}_instruction.md').read_text(encoding='utf8')
        for kind in ('hack','baseline'):
            trace=json.loads((a.source/f'{i:02d}_{kind}.json').read_text(encoding='utf8'))
            full=view(trace,False);stripped=view(trace,True)
            assert [{k:v for k,v in r.items() if k!='message'} for r in full]==stripped
            for mode,payload in [('full',full),('stripped',stripped)]:
                text='TASK INSTRUCTION\n'+instruction+'\nOBSERVED TRAJECTORY\n'+json.dumps(payload,ensure_ascii=False)
                rows.append({'id':f'{i:02d}_{kind}_{mode}','task':selected['task'],'cohort':kind,
                    'view':mode,'split':selected['split'],'input':text,
                    'input_tokens_without_chat_overhead':len(tok.encode(text,add_special_tokens=False).ids)})
    dump(a.out/'INPUTS.json',rows)
    dump(a.out/'AUDIT.json',{'classification':'DEVELOPMENTAL_PAIRED_CHANNEL_PANEL','rows':len(rows),
        'tasks':len(selection['rows']),'max_input_tokens':max(r['input_tokens_without_chat_overhead'] for r in rows),
        'over_30k':[r['id'] for r in rows if r['input_tokens_without_chat_overhead']>30000],
        'selection_sha256':receipt['selection_sha256'],'transform_source_sha256':sha(Path(__file__)),
        'invariants':'exact tool calls and observations retained; both cohorts transformed; user elicitation removed in both views',
        'labels':'released cohorts; independent safety/intent labels not claimed'})
    dump(a.out/'MANIFEST.json',{p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    print((a.out/'AUDIT.json').read_text())


if __name__=='__main__':main()
