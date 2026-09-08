"""Prove the unbounded path allows an admitted job to finish; no GPU/network."""
import json,os,sys,tempfile,time
from pathlib import Path
from launch_hindsight_calibration import supervise

with tempfile.TemporaryDirectory() as directory:
    start=time.monotonic()
    code=supervise([sys.executable,'-c','import time; time.sleep(1.2); print("finished")'],
                   None,dict(os.environ),Path(directory)/'completion.log')
    elapsed=time.monotonic()-start
    assert code==0 and elapsed>=1.2
    print(json.dumps({'exit_code':code,'elapsed_seconds':elapsed,'mid_run_timeout':False,'gpu_used':False}))
