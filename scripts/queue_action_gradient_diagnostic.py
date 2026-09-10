"""One admitted apparatus run after the existing MATH queue; no parallel GPU use."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

root = Path('/home/ubuntu/align_research_20260910')
# Exclusive-create records prevent accidental duplicate supervisors.
with (root/'action_gradient_diagnostic_queue.started').open('x') as receipt:
    receipt.write(datetime.now(timezone.utc).isoformat())
while Path('/proc/18395/cmdline').exists():
    command = Path('/proc/18395/cmdline').read_bytes()
    if b'queue_math_policy_values.py' not in command:
        raise RuntimeError('PID identity changed; inspect manually')
    time.sleep(15)
if (datetime(2026, 9, 10, 16, 55, tzinfo=timezone.utc)-datetime.now(timezone.utc)).total_seconds() < 3600:
    raise RuntimeError('Insufficient admission margin; not launched')
if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
    raise RuntimeError('GPU occupied after MATH terminal; not launched')
output = root/'action_gradient_diagnostic_v1'
with (root/'action_gradient_diagnostic_v1.log').open('x') as log:
    subprocess.run([sys.executable, str(root/'code/run_action_gradient_diagnostic.py'),
        '--snapshot', '/home/ubuntu/OSH_Reopen/.hf_cache/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218',
        '--out', str(output)], stdout=log, stderr=subprocess.STDOUT, check=True)
subprocess.run([sys.executable, str(root/'code/verify_action_gradient_diagnostic.py'),
    str(output), '--out', str(root/'action_gradient_diagnostic_v1_verified.json')], check=True)
print('ACTION_GRADIENT_QUEUE_TERMINAL', flush=True)
