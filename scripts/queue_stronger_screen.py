"""One stronger-backbone developmental qualification after the bank queue."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from verify_unexplored_screens import verify
from run_unexplored_screens import dump

root=Path('/home/ubuntu/align_research_20260910')
deadline=datetime(2026,9,10,16,55,tzinfo=timezone.utc)
while True:
    if (deadline-datetime.now(timezone.utc)).total_seconds()<3600:
        print('DEFER_INSUFFICIENT_TIME',flush=True);break
    if not (root/'QWEN32_DOWNLOAD.json').exists() or not (root/'process_value_cross_family.json').exists():
        time.sleep(30);continue
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:time.sleep(30);continue
    snapshot=json.loads((root/'QWEN32_DOWNLOAD.json').read_text())['snapshot']
    out=root/'screens_qwen32_v1'
    if out.exists():raise RuntimeError('Existing attempt: inspect without restarting')
    with (root/'screens_qwen32_v1.log').open('x') as log:
        subprocess.run([sys.executable,str(root/'code/run_unexplored_screens.py'),'--out',str(out),
                        '--snapshot',snapshot,'--batch-size','4'],stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(root/'screens_qwen32_v1_verified.json',verify(out))
    print('STRONGER_MODEL_SCREEN_COMPLETE',flush=True);break
