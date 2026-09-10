"""Public pinned weights, isolated cache; no GPU and no credential required."""
from pathlib import Path
import json
from huggingface_hub import snapshot_download
from run_unexplored_screens import sha,dump

root=Path('/home/ubuntu/align_research_20260910')
official=json.loads((root/'code/Qwen3-32B_official.json').read_text())
revision='9216db5781bf21249d130ec9da846c4624c16137'
assert official['sha']==revision
snapshot=Path(snapshot_download('Qwen/Qwen3-32B',revision=revision,token=False,
    cache_dir=root/'hf_cache',allow_patterns=['*.json','*.safetensors','*.jinja','*.model','*.txt'],max_workers=4))
checks={}
for row in official['siblings']:
    if row['rfilename'].endswith('.safetensors'):
        f=snapshot/row['rfilename'];h=sha(f)
        assert h==row['lfs']['sha256'],row['rfilename']
        checks[row['rfilename']]=h
dump(root/'QWEN32_DOWNLOAD.json',{'snapshot':str(snapshot),'revision':revision,'verified_shards':checks})
print('DOWNLOAD_AND_HASH_VERIFICATION_COMPLETE',flush=True)
