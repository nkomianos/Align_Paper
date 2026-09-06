"""Sequential independent pilot screens, maximum 50 allocated hours in total.

Scientific negatives may continue to the next independent pilot. Execution,
integrity, or verifier failures stop the suite. No replication/confirmation is
automatically scheduled. This command does not stop provider billing.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from research_pilots.common import write
from launch_research_pilot import budget_seconds


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--learning',type=Path)
    p.add_argument('--monitor-data',type=Path)
    p.add_argument('--allocation-start-utc',required=True)
    p.add_argument('--previous-h200-hours',type=float,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    stages=[]
    if a.learning:stages.append('hindsight_calibration')
    stages+=['clara','compensation','reference']
    if a.monitor_data:stages.append('monitor')
    plan={'stages':stages,'monitor_blocked':not bool(a.monitor_data),
          'calibration_omitted':not bool(a.learning),'total_ceiling_hours':50,
          'maximum_stage_hours_sum':sum({'hindsight_calibration':1,'clara':.05,'compensation':3,'reference':2,'monitor':2}[s] for s in stages),
          'paper_confirmation_queued':False,'provider_billing_stopped':False}
    if a.dry_run:print(json.dumps(plan,indent=2));return
    if os.name!='posix':raise RuntimeError('run on Linux/POSIX')
    if a.out.exists():raise FileExistsError('fresh suite output required')
    if not a.snapshot.is_dir():raise ValueError('offline snapshot missing')
    # Validate all supplied data before loading any model; no wasted GPU launch.
    if a.monitor_data:
        from research_pilots.monitor import validate
        validate(json.loads(a.monitor_data.read_text()))
    if a.learning:
        from interaction_sprint.hindsight_calibration import prepare
        prepare(a.learning)
    a.out.mkdir(parents=True)
    write(a.out/'PLAN.json',plan)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(repo/'src'),str(repo)]))
    outcomes=[]
    try:
        remaining=budget_seconds('compensation',a.allocation_start_utc,a.previous_h200_hours)
        if remaining<120:raise ValueError('insufficient allocation budget')
        subprocess.run([sys.executable,str(repo/'scripts/check_calibration_host.py'),'--snapshot',str(a.snapshot)],
                       cwd=repo,env=env,check=True,timeout=min(60.,remaining))
    except (subprocess.SubprocessError,ValueError,OSError) as exc:
        write(a.out/'SUITE_RESULT.json',{'outcomes':[{'stage':'host_check','status':'STOP_PREFLIGHT_FAILURE',
            'error':str(exc)}],'paper_green_light':False,'provider_billing_stopped':False})
        raise SystemExit(1)
    for stage in stages:
        if budget_seconds('compensation',a.allocation_start_utc,a.previous_h200_hours)<120:
            outcomes.append({'stage':stage,'status':'STOP_ALLOCATION_CAP'});break
        dest=a.out/stage
        if stage=='hindsight_calibration':
            command=[sys.executable,str(repo/'scripts/launch_hindsight_calibration.py'),'--learning',str(a.learning)]
            verifier=[sys.executable,str(repo/'scripts/verify_hindsight_calibration.py'),'--root',str(dest),'--learning',str(a.learning)]
        else:
            command=[sys.executable,str(repo/'scripts/launch_research_pilot.py'),stage]
            if stage=='monitor':command+=['--data',str(a.monitor_data)]
            verifier=[sys.executable,str(repo/'scripts/verify_research_pilot.py'),'--root',str(dest)]
        command+=['--snapshot',str(a.snapshot),'--out',str(dest),'--allocation-start-utc',a.allocation_start_utc,
                  '--previous-h200-hours',str(a.previous_h200_hours)]
        code=subprocess.run(command,cwd=repo,env=env).returncode
        outcome={'stage':stage,'exit_code':code,'completed_utc':datetime.now(timezone.utc).isoformat()}
        if code:
            outcome['status']='STOP_EXECUTION_FAILURE';outcomes.append(outcome);break
        verification_seconds=min(60.,budget_seconds('compensation',a.allocation_start_utc,a.previous_h200_hours))
        if verification_seconds<1:
            outcome['status']='STOP_ALLOCATION_CAP';outcomes.append(outcome);break
        try:
            checked=subprocess.run(verifier,cwd=repo,env=env,capture_output=True,text=True,timeout=verification_seconds)
        except (subprocess.SubprocessError,OSError) as exc:
            outcome.update(status='STOP_VERIFIER_FAILURE',error=str(exc));outcomes.append(outcome);break
        (a.out/(stage+'.verify.log')).write_text(checked.stdout+checked.stderr)
        outcome['verifier_exit_code']=checked.returncode
        outcome['status']='VERIFIED_EXPLORATORY' if checked.returncode==0 else 'STOP_VERIFIER_FAILURE'
        outcomes.append(outcome)
        if checked.returncode:break
    write(a.out/'SUITE_RESULT.json',{'outcomes':outcomes,'paper_green_light':False,'provider_billing_stopped':False})
    if any(r['status'].startswith('STOP') for r in outcomes):raise SystemExit(1)


if __name__=='__main__':main()
