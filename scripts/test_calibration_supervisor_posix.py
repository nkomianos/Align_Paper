"""Exercise real POSIX timeout and success paths locally; no GPU or network."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from launch_hindsight_calibration import supervise

with tempfile.TemporaryDirectory(prefix='calibration-supervisor-') as directory:
    started=time.monotonic()
    code=supervise([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'],.5,dict(os.environ),Path(directory)/'hard.log')
    elapsed=time.monotonic()-started
    assert code==124 and .45<=elapsed<2
    success=supervise([sys.executable,'-c','print("complete")'],20,dict(os.environ),Path(directory)/'success.log')
    assert success==0
    print(json.dumps({'hard_timeout_code':code,'hard_timeout_elapsed_seconds':elapsed,'success_code':success,
        'gpu_contacted':False,'scope':'real Linux subprocess-group termination and completion'}))
