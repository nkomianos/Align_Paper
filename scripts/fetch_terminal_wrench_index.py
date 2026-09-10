"""Public immutable source index only; never runs released exploits."""
import argparse
import hashlib
import json
from pathlib import Path
from fetch_action_tool_source import fetch


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    repo='few-sh/terminal-wrench'
    commit=json.loads(fetch(f'https://api.github.com/repos/{repo}/commits/main'))['sha']
    tree=json.loads(fetch(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))
    (a.out/'TREE.json').write_text(json.dumps(tree))
    entries={r['path']:r for r in tree['tree']};files=[]
    for name in ('README.md','LICENSE','dataset_manifest.json','task_source_datasets.json'):
        url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}';content=fetch(url)
        actual=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        assert actual==entries[name]['sha']
        (a.out/name).write_bytes(content)
        files.append({'path':name,'url':url,'git_blob':actual,'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)})
    (a.out/'DOWNLOAD.json').write_text(json.dumps({'commit':commit,'files':files,'tree_truncated':tree.get('truncated'),
        'tree_sha256':hashlib.sha256((a.out/'TREE.json').read_bytes()).hexdigest()},indent=2))
    print(json.dumps({'commit':commit,'tree_entries':len(entries),'tree_truncated':tree.get('truncated'),'files':len(files)}))


if __name__=='__main__':main()
