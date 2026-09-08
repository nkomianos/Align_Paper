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
     'estimated_hours': 1, 'blockers': ['compatible single GPU', 'pinned cached model', 'learning input']},
    {'id': 'tabular_drift', 'state': 'REAL_MODEL_CPU_SMOKE_PASSED', 'estimated_hours': 2,
     'blockers': ['use launch_research_suite.py with pinned checkpoint, source and frozen inputs']},
    {'id': 'compensating_update', 'state': 'EXPLORATORY_RUNNER_CPU_TESTED', 'estimated_hours': 3,
     'blockers': ['GPU qualification pending; use launch_research_suite.py']},
    {'id': 'clara_rule_edit', 'state': 'CPU_BASELINES_RUN_AND_VERIFIED', 'estimated_hours': .05,
     'blockers': ['natural evaluation and method advantage required before GPU expansion']},
    {'id': 'unreliable_reference', 'state': 'EXPLORATORY_RUNNER_CPU_TESTED', 'estimated_hours': 2,
     'blockers': ['prompted organism only; GPU qualification pending; use launch_research_suite.py']},
    {'id': 'specification_monitor', 'state': 'RUNNER_DATA_BLOCKED', 'estimated_hours': 2,
     'blockers': ['reviewed local MALT data; remote endpoint returned HTTP401']},
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
        p.error('Use launch_research_suite.py for the new exploratory runners; see RESEARCH_PILOTS_IMPLEMENTATION_20260906.md.')
    for field in ['learning', 'snapshot', 'out', 'allocation_start_utc', 'previous_h200_hours']:
        if getattr(a, field) is None:
            p.error('Missing --' + field.replace('_', '-'))
    if os.name != 'posix':
        p.error('GPU launching requires Linux/POSIX supervision')
    if not a.learning.is_file() or not a.snapshot.is_dir():
        p.error('Local input/snapshot missing; online downloads are prohibited')
    if a.out.exists() or a.out.with_suffix('.queue.json').exists():
        p.error('Output already exists; no resume or overwrite')
    from research_pilots.budget import admission
    if not admission(a.run,a.allocation_start_utc,a.previous_h200_hours)['admit']:
        p.error('Estimated runtime plus margin does not fit remaining budget')
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
