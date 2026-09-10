"""Outcome-blind task selection for a developmental paired monitoring panel."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from fetch_action_tool_source import fetch


def rank(text): return hashlib.sha256(('wrench-channel-panel-v1:'+text).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.source/'DOWNLOAD.json').read_text());commit=receipt['commit']
    records={r['path']:r for r in receipt['files']}
    path=a.source/'index/tasks.json';assert hashlib.sha256(path.read_bytes()).hexdigest()==records['index/tasks.json']['sha256']
    entries=json.loads(path.read_text());by_task={}
    for e in entries:
        hacks=[r for r in e['trajectories'] if r['classification']=='rewarded_serious_exploit' and r['reward']==1]
        baselines=[r for r in e.get('baselines',[]) if r['reward']==1]
        if hacks and baselines:by_task.setdefault(e['task_id'],[]).append((e,hacks,baselines))
    selected=[];jobs=[]
    for i,task in enumerate(sorted(by_task,key=rank)[:24]):
        e,hacks,baselines=min(by_task[task],key=lambda x:rank(task+':'+x[0]['model']))
        model=e['model'];hack=min(hacks,key=lambda r:rank(task+':hack:'+r['trajectory_label']))
        baseline=min(baselines,key=lambda r:rank(task+':baseline:'+r['label']))
        assert all(re.fullmatch(r'[A-Za-z0-9_.-]+',v) for v in (task,model,hack['trajectory_label'],baseline['label']))
        stem=f'tasks/{task}/{model}'
        paths={'instruction':stem+'/original_task/instruction.md',
               'hack':stem+'/hack_trajectories/'+hack['trajectory_label']+'/trial/agent/trajectory.json',
               'baseline':stem+'/baseline_trajectories/'+baseline['label']+'/trial/agent/trajectory.json'}
        selected.append({'task':task,'model':model,'source_datasets':e.get('source_datasets'),
            'split':'calibration' if i<4 else 'dev','paths':paths,
            'hack_source_metadata':hack,'baseline_source_metadata':baseline})
        jobs.extend((i,kind,path) for kind,path in paths.items())
    (a.out/'SELECTION.json').write_text(json.dumps({'commit':commit,'selection':'first24 salted task hashes; one hashed model/hack/baseline; no length or monitor outcome selection',
        'source_index_sha256':hashlib.sha256(path.read_bytes()).hexdigest() if isinstance(path,Path) else records['index/tasks.json']['sha256'],
        'rows':selected},indent=2))
    def download(job):
        i,kind,path=job;url=f'https://raw.githubusercontent.com/few-sh/terminal-wrench/{commit}/{path}'
        content=fetch(url);name=f'{i:02d}_{kind}'+('.md' if kind=='instruction' else '.json')
        (a.out/name).write_bytes(content)
        return {'panel_index':i,'kind':kind,'source_path':path,'url':url,'file':name,'bytes':len(content),
                'sha256':hashlib.sha256(content).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(download,jobs))
    (a.out/'DOWNLOAD.json').write_text(json.dumps({'commit':commit,'files':files,
        'selection_sha256':hashlib.sha256((a.out/'SELECTION.json').read_bytes()).hexdigest()},indent=2))
    print(json.dumps({'tasks':len(selected),'files':len(files),'bytes':sum(r['bytes'] for r in files)}))


if __name__=='__main__':main()
