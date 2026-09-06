"""Fail-closed research queue: inspect readiness or run the qualified calibration.

Research-only entries are visible but never silently treated as executable.
No SSH, downloads, retries, model substitution, or provider billing controls.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ENTRIES = [
    {'id': 'hindsight_calibration', 'state': 'CPU_TESTED_HOST_CHECK_REQUIRED',
     'cap_hours': 1, 'blockers': ['compatible single GPU', 'pinned cached model', 'learning input']},
    {'id': 'compensating_update', 'state': 'RESEARCH_BLOCKED', 'cap_hours': 3,
     'blockers': ['closest-source comparison', 'frozen task data', 'GPU runner', 'independent verifier', 'CPU rehearsal']},
    {'id': 'clara_rule_edit', 'state': 'RESEARCH_BLOCKED', 'cap_hours': 3,
     'blockers': ['exact CPU baseline comparison', 'natural evaluation dataset', 'method advantage']},
    {'id': 'unreliable_reference', 'state': 'RESEARCH_BLOCKED', 'cap_hours': 2,
     'blockers': ['reference acquisition protocol', 'model organism qualification', 'runner and verifier']},
    {'id': 'specification_monitor', 'state': 'RESEARCH_BLOCKED', 'cap_hours': 2,
     'blockers': ['MALT label/split audit', 'existing monitor comparison', 'runner and verifier']},
]


def launch_command(entry, learning, snapshot, out, start, previous):
    if entry != 'hindsight_calibration':
        raise ValueError('Research-only entry is not runnable; see readiness blockers')
    return [sys.executable, str(REPO / 'scripts/launch_hindsight_calibration.py'),
            '--learning', str(learning), '--snapshot', str(snapshot), '--out', str(out),
            '--allocation-start-utc', start, '--previous-h200-hours', str(previous)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', choices=[e['id'] for e in ENTRIES])
    p.add_argument('--learning', type=Path)
    p.add_argument('--snapshot', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--allocation-start-utc')
    p.add_argument('--previous-h200-hours', type=float)
    a = p.parse_args()
    if not a.run:
        print(json.dumps({'entries': ENTRIES, 'total_allocation_ceiling_hours': 50,
                          'automatic_expansion': False, 'all_candidates_ready': False}, indent=2))
        return
    if a.run != 'hindsight_calibration':
        p.error('This candidate is research-blocked, not GPU-ready.')
    for field in ['learning', 'snapshot', 'out', 'allocation_start_utc', 'previous_h200_hours']:
        if getattr(a, field) is None:
            p.error('Missing --' + field.replace('_', '-'))
    if os.name != 'posix':
        p.error('GPU launching requires Linux/POSIX supervision')
    if not a.learning.is_file() or not a.snapshot.is_dir():
        p.error('Local input/snapshot missing; online downloads are prohibited')
    if a.out.exists() or a.out.with_suffix('.queue.json').exists():
        p.error('Output already exists; no resume or overwrite')
    from launch_hindsight_calibration import remaining_seconds
    if remaining_seconds(a.allocation_start_utc, a.previous_h200_hours) < 120:
        p.error('Allocation budget exhausted')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=os.pathsep.join([str(REPO / 'src'), str(REPO)]))
    checks = [
        [sys.executable, str(REPO / 'scripts/check_calibration_host.py'), '--snapshot', str(a.snapshot)],
        [sys.executable, str(REPO / 'scripts/run_hindsight_calibration.py'), '--learning', str(a.learning),
         '--snapshot', str(a.snapshot), '--out', str(a.out), '--preflight-only'],
    ]
    # Preflight imports/tokenizer only; no CUDA weights. Bounded to avoid paid idle stalls.
    for command in checks:
        subprocess.run(command, cwd=REPO, env=env, check=True, timeout=60)
    command = launch_command(a.run, a.learning, a.snapshot, a.out,
                             a.allocation_start_utc, a.previous_h200_hours)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.with_suffix('.queue.json').open('x') as f:
        json.dump({'entry': a.run, 'started_utc': datetime.now(timezone.utc).isoformat(),
                   'argv': command, 'no_auto_followup': True,
                   'remaining_entries': ENTRIES[1:], 'provider_billing_stop': False}, f, indent=2)
    completed = subprocess.run(command, cwd=REPO, env=env)
    # Deliberately stop even on success; scientific decisions require output verification.
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
