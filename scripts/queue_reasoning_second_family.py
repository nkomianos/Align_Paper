"""Wait for verified predecessor, admit one cached second-family bank, then audit."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from analyze_reasoning_bank import read_bank,analyze,compare
from run_unexplored_screens import dump

root=Path('/home/ubuntu/align_research_20260910')
deadline=datetime(2026,9,10,16,55,tzinfo=timezone.utc)
while True:
    if (deadline-datetime.now(timezone.utc)).total_seconds()<5400:
        print('DEFER_INSUFFICIENT_ESTIMATED_TIME',flush=True);break
    if not (root/'bank_qwen_v1/MANIFEST.json').exists():
        time.sleep(30);continue
    read_bank(root/'bank_qwen_v1')
    dump(root/'bank_qwen_v1_audit.json',analyze(root/'bank_qwen_v1'))
    occupied=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if occupied:time.sleep(30);continue
    out=root/'bank_nemo_v1'
    if out.exists():raise RuntimeError('Existing second-family attempt; inspect, do not restart')
    command=[sys.executable,str(root/'code/run_reasoning_bank.py'),'--out',str(out),
        '--source',str(root/'code/gsm8k_train.jsonl'),'--prefixes',str(root/'bank_qwen_v1/PREFIXES.json'),
        '--snapshot','/home/ubuntu/OSH_Reopen/.hf_cache/hub/models--mistralai--Mistral-Nemo-Instruct-2407/snapshots/04d8a90549d23fc6bd7f642064003592df51e9b3']
    print('LAUNCH_SECOND_FAMILY',datetime.now(timezone.utc).isoformat(),flush=True)
    with (root/'bank_nemo_v1.log').open('x') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    dump(root/'bank_nemo_v1_audit.json',analyze(out))
    dump(root/'process_value_cross_family.json',compare(root/'bank_qwen_v1',out))
    print('SECOND_FAMILY_AND_CPU_AUDITS_COMPLETE',flush=True);break
