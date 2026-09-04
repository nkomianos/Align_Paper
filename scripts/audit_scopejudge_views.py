import argparse
import hashlib
import json
from pathlib import Path
from interaction_sprint.scopejudge_views import preexecution_view

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();raw=(a.data/'train.jsonl').read_bytes()
manifest=json.loads((a.data/'dataset-manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest()==manifest['sha256']
n=total=maximum=0
for record in map(json.loads,raw.decode('utf8').splitlines()):
    for label in record['extra']['scopejudge']['labels']:
        view=preexecution_view(record,label['step_id'],label['tool_call_id'])
        size=len(json.dumps(view));n+=1;total+=size;maximum=max(maximum,size)
report=dict(views=n,total_serialized_characters=total,maximum_view_characters=maximum,
            manifest_matches=True,scope='Structural replay only, not model performance or online safety')
with a.out.open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
