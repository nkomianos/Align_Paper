"""Run a CPU pilot or a supervised exploratory neural pilot; no remote access."""
import argparse
import json
import os
from pathlib import Path
import sys
import hashlib
from research_pilots.common import write
from research_pilots.data import build,validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot',choices=['compensation','clara','reference','monitor'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--compensation-repair',action='store_true')
    a=p.parse_args()
    if a.compensation_repair and a.pilot!='compensation':raise ValueError('repair applies only to compensation')
    if a.out.exists():raise FileExistsError('no overwrite/resume')
    if a.pilot=='monitor':
        from research_pilots.monitor import validate as validate_monitor
        if not a.data:raise ValueError('reviewed normalized MALT JSON required; no synthetic replacement')
        data=validate_monitor(json.loads(a.data.read_text()))
    else:
        data=validate(json.loads(a.data.read_text())) if a.data else build()
    if not a.prepare_only and a.pilot!='clara' and os.environ.get('RESEARCH_PILOT_SUPERVISED')!='1':
        raise RuntimeError('use launch_research_pilot.py')
    if a.snapshot and a.out.resolve().is_relative_to(a.snapshot.resolve()):
        raise ValueError('output overlaps model cache')
    a.out.mkdir(parents=True)
    repo=Path(__file__).resolve().parents[1]
    sources=list((repo/'src/research_pilots').glob('*.py'))+[Path(__file__),repo/'scripts/launch_research_pilot.py',
        repo/'src/latent_contract/sender_update.py',repo/'src/interaction_sprint/hindsight_execution_integrity.py',
        repo/'scripts/launch_hindsight_calibration.py']
    write(a.out/'PROVENANCE.json',{'pilot':a.pilot,'argv':sys.argv,'scope':'prospective exploratory pilot',
        'sources':{str(s.relative_to(repo)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}})
    write(a.out/'INPUTS.json',data)
    if a.prepare_only:
        print(json.dumps({'status':'INPUTS_PREPARED','gpu_used':False,'pilot':a.pilot}));return
    if a.pilot=='clara':
        from research_pilots.clara import run
        run(a.out);return
    if not a.snapshot:raise ValueError('offline pinned snapshot required')
    from transformers import AutoTokenizer
    from research_pilots.neural import REVISION
    if a.snapshot.name!=REVISION:raise ValueError('wrong model revision')
    tokenizer=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    if a.pilot=='monitor':
        from research_pilots.monitor import prompt
        texts=[prompt(r,m,s)['prompt'] for r in data if r['split']!='holdout'
               for m in ('task_only','trace_only','full_context','specification_grounded') for s in (0,1)]
    else:
        from research_pilots.reference import WITHHOLD
        texts=[prefix+r['prompt'] for split in ('train','dev','utility') for r in data[split]
               for prefix in (('',WITHHOLD) if a.pilot=='reference' else ('',))]
    lengths=[]
    for text in texts:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                              add_generation_prompt=True,enable_thinking=False)
        lengths.append(len(tokenizer.encode(rendered,add_special_tokens=False)))
    if max(lengths)>4096:raise ValueError('input exceeds 4096 tokens; rejected before GPU model load')
    write(a.out/'TOKEN_PREFLIGHT.json',{'max_tokens':max(lengths),'prompts_checked':len(lengths),'truncation':False})
    from research_pilots.neural import Backend
    backend=Backend(a.snapshot,a.out,2026090601)
    if a.pilot=='compensation':
        from research_pilots.compensation import run
    elif a.pilot=='reference':
        from research_pilots.reference import run
    else:
        from research_pilots.monitor import run
    if a.compensation_repair:
        from research_pilots.compensation import REPAIR_DOSES
        run(backend,data,a.out,doses=REPAIR_DOSES)
    else:
        run(backend,data,a.out)


if __name__=='__main__':main()
