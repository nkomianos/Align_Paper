"""Sequential pilot screens planned against 50 allocated hours; no mid-run timeout.

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
from research_pilots.budget import admission,ESTIMATED_HOURS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--learning',type=Path)
    p.add_argument('--monitor-data',type=Path)
    p.add_argument('--tabular-checkpoint',type=Path)
    p.add_argument('--tabular-inputs',type=Path)
    p.add_argument('--runtime-estimates',type=Path,help='JSON stage-to-hours overrides from host benchmarks')
    p.add_argument('--allocation-start-utc',required=True)
    p.add_argument('--previous-h200-hours',type=float,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    stages=[]
    if a.tabular_checkpoint:stages.append('tabular_drift')
    if a.learning:stages.append('hindsight_calibration')
    stages+=['clara','compensation','reference']
    if a.monitor_data:stages.append('monitor')
    estimates=dict(ESTIMATED_HOURS)
    if a.runtime_estimates:
        overrides=json.loads(a.runtime_estimates.read_text())
        if set(overrides)-set(estimates):raise ValueError('unknown runtime estimate stage')
        estimates.update(overrides)
    for stage in stages:admission(stage,a.allocation_start_utc,a.previous_h200_hours,estimates[stage])
    plan={'stages':stages,'monitor_blocked':not bool(a.monitor_data),
          'tabular_omitted':not bool(a.tabular_checkpoint),'mid_run_timeout':False,
          'calibration_omitted':not bool(a.learning),'total_ceiling_hours':50,
          'estimated_stage_hours':{s:estimates[s] for s in stages},
          'estimated_hours_sum':sum(estimates[s] for s in stages),
          'paper_confirmation_queued':False,'provider_billing_stopped':False}
    if a.dry_run:print(json.dumps(plan,indent=2));return
    if os.name!='posix':raise RuntimeError('run on Linux/POSIX')
    if a.out.exists():raise FileExistsError('fresh suite output required')
    if not a.snapshot.is_dir():raise ValueError('offline snapshot missing')
    if a.tabular_checkpoint:
        from tabicl import TabICLClassifier  # qualify the optional dependency before any GPU work
        import hashlib
        from run_tabular_drift import CHECKPOINT_SHA
        from research_pilots.tabular_drift import validate
        if hashlib.sha256(a.tabular_checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA:
            raise ValueError('tabular checkpoint hash differs')
        if not a.tabular_inputs:raise ValueError('frozen tabular inputs required')
        validate(json.loads(a.tabular_inputs.read_text()))
    # Validate all supplied data before loading any model; no wasted GPU launch.
    if a.monitor_data:
        from research_pilots.monitor import validate
        validate(json.loads(a.monitor_data.read_text()))
    if a.learning:
        from interaction_sprint.hindsight_calibration import prepare
        prepare(a.learning)
    a.out.mkdir(parents=True)
    write(a.out/'PLAN.json',plan)
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(repo/'src'),str(repo),os.environ.get('PYTHONPATH','')]))
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
        budget=admission(stage,a.allocation_start_utc,a.previous_h200_hours,estimates[stage])
        if not budget['admit']:
            outcomes.append({'stage':stage,'status':'STOP_BUDGET_ADMISSION','budget':budget});break
        dest=a.out/stage
        if stage=='hindsight_calibration':
            command=[sys.executable,str(repo/'scripts/launch_hindsight_calibration.py'),'--learning',str(a.learning)]
            verifier=[sys.executable,str(repo/'scripts/verify_hindsight_calibration.py'),'--root',str(dest),'--learning',str(a.learning)]
        else:
            command=[sys.executable,str(repo/'scripts/launch_research_pilot.py'),stage]
            if stage=='monitor':command+=['--data',str(a.monitor_data)]
            verifier=[sys.executable,str(repo/'scripts/verify_research_pilot.py'),'--root',str(dest)]
            if stage=='tabular_drift':
                command+=['--tabular-checkpoint',str(a.tabular_checkpoint),'--tabular-checkpoint-sha256',CHECKPOINT_SHA,
                          '--data',str(a.tabular_inputs)]
                verifier=[sys.executable,str(repo/'scripts/verify_tabular_drift.py'),'--root',str(dest)]
        command+=['--snapshot',str(a.snapshot),'--out',str(dest),'--allocation-start-utc',a.allocation_start_utc,
                  '--previous-h200-hours',str(a.previous_h200_hours),'--estimated-hours',str(estimates[stage])]
        code=subprocess.run(command,cwd=repo,env=env).returncode
        outcome={'stage':stage,'exit_code':code,'completed_utc':datetime.now(timezone.utc).isoformat()}
        if code:
            outcome['status']='STOP_EXECUTION_FAILURE';outcomes.append(outcome);break
        try:
            checked=subprocess.run(verifier,cwd=repo,env=env,capture_output=True,text=True)
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
