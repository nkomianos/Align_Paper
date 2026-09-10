"""Retrieve pinned monitoring code/index as data; never execute upstream scripts."""
import argparse
import hashlib
import json
from pathlib import Path
from fetch_action_tool_source import fetch


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((a.index/'DOWNLOAD.json').read_text());commit=receipt['commit']
    assert hashlib.sha256((a.index/'TREE.json').read_bytes()).hexdigest()==receipt['tree_sha256']
    tree={r['path']:r for r in json.loads((a.index/'TREE.json').read_text())['tree']};files=[]
    paths=('index/tasks.json','index/trajectories.json','monitoring/monitor.py','monitoring/strip.py',
           'monitoring/analyze.py','prompts/monitor_judge_v1.txt','prompts/hack_prompt_v5.md',
           'monitoring/analysis_stripped_vs_legitimate_gpt54_v1.json')
    for name in paths:
        url=f'https://raw.githubusercontent.com/few-sh/terminal-wrench/{commit}/{name}';content=fetch(url)
        blob=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest();assert blob==tree[name]['sha']
        dest=a.out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        files.append({'path':name,'git_blob':blob,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'url':url})
    (a.out/'DOWNLOAD.json').write_text(json.dumps({'commit':commit,'files':files},indent=2))
    print(json.dumps({'commit':commit,'files':len(files),'bytes':sum(r['bytes'] for r in files)}))


if __name__=='__main__':main()
