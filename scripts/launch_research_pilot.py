"""POSIX hard-cap supervisor for exploratory pilots; no auto-expansion."""
import argparse
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import shutil
import sys
import time
from launch_hindsight_calibration import supervise

CAPS={'compensation':3.,'reference':2.,'monitor':2.,'clara':.05}


def budget_seconds(pilot,start,spent,now=None):
    moment=datetime.fromisoformat(start.replace('Z','+00:00'))
    now=now or datetime.now(timezone.utc)
    if moment.tzinfo is None or moment>now:raise ValueError('invalid allocation start')
    if not math.isfinite(spent) or not 0<=spent<50:raise ValueError('invalid previous hours')
    return min(CAPS[pilot]*3600,(50-spent)*3600-(now-moment).total_seconds())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot',choices=CAPS)
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allocation-start-utc',required=True)
    p.add_argument('--previous-h200-hours',type=float,required=True)
    a=p.parse_args()
    seconds=budget_seconds(a.pilot,a.allocation_start_utc,a.previous_h200_hours)
    if seconds<120:raise ValueError('insufficient remaining budget')
    if a.out.exists():raise FileExistsError('no resume')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out.parent).free<5*1024**3:raise ValueError('need 5GiB free')
    repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,RESEARCH_PILOT_SUPERVISED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false',PYTHONPATH=os.pathsep.join([str(repo/'src'),str(repo)]))
    command=[sys.executable,str(repo/'scripts/run_research_pilot.py'),a.pilot,'--out',str(a.out)]
    if a.snapshot:command+=['--snapshot',str(a.snapshot)]
    if a.data:command+=['--data',str(a.data)]
    from research_pilots.common import write
    write(a.out.with_suffix('.launch.json'),{'argv':command,'cap_seconds':seconds,
        'allocation_start_utc':a.allocation_start_utc,'previous_h200_hours':a.previous_h200_hours,
        'provider_billing_stopped':False})
    start=time.monotonic()
    code=supervise(command,seconds,env,a.out.with_suffix('.log'))
    write(a.out.with_suffix('.exit.json'),{'exit_code':code,'wall_seconds':time.monotonic()-start,
        'complete':(a.out/'MANIFEST.json').is_file(),'provider_billing_stopped':False})
    raise SystemExit(code)


if __name__=='__main__':main()
