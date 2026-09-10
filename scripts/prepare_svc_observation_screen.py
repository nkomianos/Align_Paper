"""Download an outcome-blind external video qualification sample and pinned VLM."""
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download,snapshot_download
from run_unexplored_screens import sha,dump

root=Path('/home/ubuntu/align_research_20260910')
assets=root/'video_assets';assets.mkdir(exist_ok=True)
official=json.loads((root/'code/SVC_HF.json').read_text())
revision='4c9bd87ef3b0f269ca8b4503c081f5f38bc1fc9a';assert official['sha']==revision
modelinfo=json.loads((root/'code/QWEN_VL_OFFICIAL.json').read_text())
mrev='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b';assert modelinfo['sha']==mrev
rows=[json.loads(x) for x in (root/'code/vcbench_data.jsonl').read_text().splitlines() if x]
files={x['rfilename']:x for x in official['siblings']}
selected=[];used=set()
for category in ('O1-Snap','E1-Action'):
    candidates=sorted([r for r in rows if r['counting_subtype']==category],
        key=lambda r:hashlib.sha256(('svc-observation-v1:'+r['id']).encode()).hexdigest())
    for r in candidates:
        matches=[n for n in files if n.startswith('videos/') and Path(n).name==Path(r['video_path']).name]
        if len(matches)!=1:continue
        name=matches[0]
        if name in used:continue
        # Feasibility restriction frozen without any model results or answer filtering.
        if files[name].get('size',10**20)>180_000_000:continue
        times=r['query_points']['time'];counts=r['query_points']['count']
        available=[i for i,t in enumerate(times) if 2<=t<=120]
        if not available:continue
        index=available[-1]
        local=Path(hf_hub_download('buaaplay/SVCBench',name,repo_type='dataset',revision=revision,
            token=False,cache_dir=root/'hf_cache'))
        h=sha(local);assert h==files[name]['lfs']['sha256']
        selected.append({'id':r['id'],'category':category,'question':r['question'],
            'query_time':times[index],'target':counts[index],'source_dataset':r['source_dataset'],
            'source_video':name,'local_video':str(local),'sha256':h,
            'scope':'size/time-limited DEV sample, not full-benchmark performance'})
        used.add(name);dump(assets/'INPUTS.partial.json',selected)
        print(json.dumps({'videos':len(selected),'id':r['id']}),flush=True)
        if sum(s['category']==category for s in selected)==12:break
assert len(selected)==24, 'Insufficient qualified sources; no invented fallback videos'
dump(assets/'INPUTS.json',selected)
snapshot=Path(snapshot_download('Qwen/Qwen3-VL-8B-Instruct',revision=mrev,token=False,
    cache_dir=root/'hf_cache',allow_patterns=['*.json','*.safetensors','*.jinja','*.model','*.txt'],max_workers=4))
hashes={}
for row in modelinfo['siblings']:
    if row['rfilename'].endswith('.safetensors'):
        h=sha(snapshot/row['rfilename']);assert h==row['lfs']['sha256'];hashes[row['rfilename']]=h
dump(root/'VIDEO_ASSETS_READY.json',{'snapshot':str(snapshot),'model_revision':mrev,'model_hashes':hashes,
    'input_sha256':sha(assets/'INPUTS.json'),'source_revision':revision,
    'source_annotations_sha256':sha(root/'code/vcbench_data.jsonl'),'n_videos':24})
print('VIDEO_ASSETS_READY',flush=True)
