"""Sequential capability-gated MATH continuation banks within the paid window."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from audit_math_policy_value_bank import audit,compare
from run_unexplored_screens import dump

root=Path('/home/ubuntu/align_research_20260910')
deadline=datetime(2026,9,10,16,55,tzinfo=timezone.utc)
models=[Path('/home/ubuntu/OSH_Reopen/.hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218'),
        Path(json.loads((root/'QWEN32_DOWNLOAD.json').read_text())['snapshot'])]
for index,model in enumerate(models):
    if (deadline-datetime.now(timezone.utc)).total_seconds() < (3600 if index==0 else 6000):
        raise RuntimeError('Insufficient admission margin; no launch')
    if index:
        result=audit(root/'math_bank8_v2');dump(root/'math_bank8_v2_audit.json',result)
        if not result['qualified']:
            print('STOP_FIRST_BANK_QUALIFICATION',flush=True);break
    name='math_bank8_v2' if index==0 else 'math_bank32_v2'
    command=[sys.executable,str(root/'code/run_math_policy_value_bank.py'),'--source',str(root/'code/math500_test.jsonl'),
             '--snapshot',str(model),'--out',str(root/name),'--parser','legacy-v1']
    if index:command.extend(['--prefixes',str(root/'math_bank8_v2/PREFIXES.json')])
    with (root/(name+'.log')).open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
else:
    result=compare(root/'math_bank8_v2',root/'math_bank32_v2');dump(root/'math_policy_values_v2_audit.json',result)
    print(result['route'],flush=True)
print('MATH_QUEUE_TERMINAL',flush=True)
