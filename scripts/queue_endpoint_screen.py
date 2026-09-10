"""Download pinned small neural model, then run after stronger-language-model job."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from huggingface_hub import snapshot_download
from run_unexplored_screens import sha,dump

root=Path('/home/ubuntu/align_research_20260910')
revision='267b167dc01f0e4e61923ea244e8b988f84deb80'
official=json.loads((root/'code/DDPM_OFFICIAL.json').read_text());assert official['sha']==revision
snapshot=Path(snapshot_download('google/ddpm-cifar10-32',revision=revision,token=False,
    cache_dir=root/'hf_cache',allow_patterns=['*.json','*.safetensors']))
for row in official['siblings']:
    if row['rfilename'].endswith('.safetensors'):assert sha(snapshot/row['rfilename'])==row['lfs']['sha256']
dump(root/'DDPM_DOWNLOAD.json',{'snapshot':str(snapshot),'revision':revision})
while True:
    if datetime.now(timezone.utc)>=datetime(2026,9,10,15,55,tzinfo=timezone.utc):
        print('DEFER_INSUFFICIENT_TIME',flush=True);break
    if not (root/'screens_qwen32_v1_verified.json').exists():time.sleep(30);continue
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        time.sleep(30);continue
    env=os.environ.copy();env['PYTHONPATH']=str(root/'diffusion_overlay')+':'+str(root/'code')
    with (root/'endpoint_v1.log').open('x') as log:
        subprocess.run([sys.executable,str(root/'code/run_endpoint_sensitivity_screen.py'),
            '--snapshot',str(snapshot),'--out',str(root/'endpoint_v1')],env=env,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    print('ENDPOINT_QUEUE_COMPLETE',flush=True);break
