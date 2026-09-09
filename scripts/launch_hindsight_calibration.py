"""Local POSIX supervisor. Never connects to a host or starts another study."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def remaining_seconds(allocation_started, already_spent_hours, now=None):
    start=datetime.fromisoformat(allocation_started.replace('Z','+00:00'))
    if start.tzinfo is None: raise ValueError('allocation start must be timezone-aware')
    current=datetime.now(timezone.utc) if now is None else now
    elapsed=(current-start).total_seconds()
    if elapsed<0: raise ValueError('allocation start cannot be in future')
    if not math.isfinite(already_spent_hours) or not 0<=already_spent_hours<50:
        raise ValueError('prior H200 allocation hours must be in [0,50)')
    return (50-already_spent_hours)*3600-elapsed


def supervise(command, seconds, env, log_path):
    if os.name!='posix': raise RuntimeError('supervisor requires POSIX process groups')
    with Path(log_path).open('x') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        def terminate(signum=None, frame=None):
            if process.poll() is None:
                try: os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        prior={s:signal.signal(s,terminate) for s in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
        grace=10. if seconds is None else min(10.,seconds*.1)
        try:
            try: return process.wait(timeout=None if seconds is None else max(.01,seconds-grace))
            except subprocess.TimeoutExpired:
                terminate()
                try: process.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL); process.wait()
                return 124
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGKILL); process.wait()
            for s,handler in prior.items(): signal.signal(s,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learning',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allocation-start-utc',required=True)
    p.add_argument('--previous-h200-hours',type=float,required=True)
    p.add_argument('--estimated-hours',type=float)
    a=p.parse_args()
    from research_pilots.budget import admission
    budget=admission('hindsight_calibration',a.allocation_start_utc,a.previous_h200_hours,a.estimated_hours)
    if not budget['admit']: raise ValueError('estimated run plus margin does not fit remaining budget')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if a.out.exists(): raise FileExistsError('no resume or overwrite')
    if shutil.disk_usage(a.out.parent).free<30*1024**3: raise ValueError('need30GiB free for full evidence')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             CALIBRATION_SUPERVISED_LAUNCH='1',PYTHONHASHSEED='0')
    repo=Path(__file__).resolve().parents[1]
    env['PYTHONPATH']=os.pathsep.join([str(repo/'src'),str(repo)])
    command=[sys.executable,str(repo/'scripts/run_hindsight_calibration.py'),'--learning',str(a.learning),
             '--snapshot',str(a.snapshot),'--out',str(a.out)]
    receipt=a.out.with_suffix('.launch.json')
    with receipt.open('x') as f: json.dump({'argv':command,'budget_admission':budget,'hard_cap_seconds':None,
        'allocation_start_utc':a.allocation_start_utc,'previous_h200_hours':a.previous_h200_hours,
        'budget_allocation_hours':budget['budget_target_hours'],'no_auto_followup':True},f,indent=2)
    start=time.monotonic()
    code=supervise(command,None,env,a.out.with_suffix('.log'))
    with a.out.with_suffix('.exit.json').open('x') as f:
        json.dump({'exit_code':code,'wall_seconds':time.monotonic()-start,
                   'complete':(a.out/'MANIFEST.json').exists(), 'provider_instance_terminated':False},f,indent=2)
    raise SystemExit(code)


if __name__=='__main__':main()
