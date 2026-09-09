"""Estimate before admission, then let each exploratory experiment finish."""
import argparse
from datetime import datetime,timezone
import math
import os
from pathlib import Path
import shutil
import sys
import time
from launch_hindsight_calibration import supervise

from research_pilots.budget import admission,remaining_hours,ESTIMATED_HOURS
PILOTS={k:v for k,v in ESTIMATED_HOURS.items() if k!='hindsight_calibration'}


def budget_seconds(pilot,start,spent,now=None):
    return remaining_hours(start,spent,now)*3600


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('pilot',choices=PILOTS)
    p.add_argument('--snapshot',type=Path)
    p.add_argument('--data',type=Path)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allocation-start-utc',required=True)
    p.add_argument('--previous-h200-hours',type=float,required=True)
    p.add_argument('--estimated-hours',type=float)
    p.add_argument('--compensation-repair',action='store_true')
    p.add_argument('--tabular-checkpoint',type=Path)
    p.add_argument('--tabular-checkpoint-sha256')
    a=p.parse_args()
    if a.compensation_repair and a.pilot!='compensation':raise ValueError('repair applies only to compensation')
    budget=admission(a.pilot,a.allocation_start_utc,a.previous_h200_hours,a.estimated_hours)
    if not budget['admit']:raise ValueError('estimated run plus margin does not fit remaining budget')
    if a.out.exists():raise FileExistsError('no resume')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out.parent).free<5*1024**3:raise ValueError('need 5GiB free')
    repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,RESEARCH_PILOT_SUPERVISED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
             TOKENIZERS_PARALLELISM='false',PYTHONPATH=os.pathsep.join([str(repo/'src'),str(repo),os.environ.get('PYTHONPATH','')]))
    command=[sys.executable,str(repo/'scripts/run_research_pilot.py'),a.pilot,'--out',str(a.out)]
    if a.compensation_repair:command.append('--compensation-repair')
    if a.pilot=='tabular_drift':
        command=[sys.executable,str(repo/'scripts/run_tabular_drift.py'),'--backend','tabicl','--out',str(a.out),
                 '--checkpoint',str(a.tabular_checkpoint),'--checkpoint-sha256',str(a.tabular_checkpoint_sha256)]
    if a.snapshot and a.pilot!='tabular_drift':command+=['--snapshot',str(a.snapshot)]
    if a.data and a.pilot!='tabular_drift':command+=['--data',str(a.data)]
    if a.data and a.pilot=='tabular_drift':command+=['--inputs',str(a.data)]
    from research_pilots.common import write
    write(a.out.with_suffix('.launch.json'),{'argv':command,'budget_admission':budget,'cap_seconds':None,
        'allocation_start_utc':a.allocation_start_utc,'previous_h200_hours':a.previous_h200_hours,
        'provider_billing_stopped':False})
    start=time.monotonic()
    code=supervise(command,None,env,a.out.with_suffix('.log'))
    write(a.out.with_suffix('.exit.json'),{'exit_code':code,'wall_seconds':time.monotonic()-start,
        'complete':(a.out/'MANIFEST.json').is_file(),'provider_billing_stopped':False})
    raise SystemExit(code)


if __name__=='__main__':main()
