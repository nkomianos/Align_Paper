"""Launch one bounded frozen DEV run on the authorized host; never retries."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-commit', required=True)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    workspace = Path('/home/ubuntu/align_iclr_2027')
    allocation_end = '2026-09-09T09:46:00Z'
    allocation_start = datetime(2026, 9, 5, 5, 46, tzinfo=timezone.utc).timestamp()
    elapsed = time.time() - allocation_start
    if not 0 <= elapsed < (100 * 3600 - 14400):
        raise RuntimeError('initial four-hour run does not fit remaining allocation budget')
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if head != args.expected_commit:
        raise RuntimeError('transport commit differs')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain']):
        raise RuntimeError('remote source tree must be clean')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if inventory != 'GPU-5fe87fda-e3c2-ef27-9559-53a2d4229c97':
        raise RuntimeError('device differs from authorized inspected single-GH200 host')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if processes:
        raise RuntimeError('another GPU process is active; do not overlap budgets')
    inputs = workspace / 'inputs/reduced_dev_v1'
    if {x.name for x in inputs.iterdir()} != {'MANIFEST.json', 'learning.json', 'development.json'}:
        raise RuntimeError('transfer input directory must contain exactly three authorized files')
    setup = json.loads((workspace / 'logs/model_setup.json').read_text())
    output = workspace / 'runs/reduced_dev_v1'
    ledger = workspace / 'logs/reduced_dev_v1_launch.json'
    if output.exists() or ledger.exists():
        raise FileExistsError('this run was already attempted; preserve evidence and do not retry automatically')
    output.parent.mkdir(exist_ok=True)
    command = [sys.executable, str(repo / 'scripts/run_hindsight_pahf_reduced_dev.py'),
               '--input-root', str(inputs), '--model-snapshot', setup['snapshot'],
               '--root', str(output), '--max-runtime-seconds', '14400',
               '--allocation-deadline-utc', allocation_end]
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join([str(repo / 'src'), str(repo), str(repo / 'scripts')]),
           'CUDA_VISIBLE_DEVICES': '0', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
    with (workspace / 'logs/reduced_dev_v1.log').open('xb') as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = {'pid': process.pid, 'launch_utc': datetime.now(timezone.utc).isoformat(),
               'command': command, 'source_commit': head,
               'launcher_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'allocated_hours_before_launch': elapsed/3600, 'maximum_run_hours': 4,
               'allocation_deadline_utc': allocation_end, 'gpu_uuid': inventory,
               'confirmation_transferred': False, 'automatic_retry': False}
    with ledger.open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
