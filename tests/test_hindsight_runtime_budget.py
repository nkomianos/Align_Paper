"""Budget arithmetic and an actual POSIX hung-process hard-stop check."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from run_hindsight_pahf_reduced_dev import runtime_deadlines


def test_allocation_deadline_clamps_run():
    result = runtime_deadlines(14400, '1970-01-01T01:00:00Z', wall_now=1800, monotonic_now=10)
    assert result['effective_runtime_seconds'] == 1800
    assert result['hard_monotonic'] == 1810
    assert result['soft_monotonic'] == 1690


@pytest.mark.parametrize('seconds', [float('nan'), float('inf'), 120, 360001])
def test_bad_budget_rejected(seconds):
    with pytest.raises(ValueError):
        runtime_deadlines(seconds, '2099-01-01T00:00:00Z')


def test_expired_allocation_rejected():
    with pytest.raises(ValueError):
        runtime_deadlines(14400, '1970-01-01T00:00:00Z')


@pytest.mark.skipif(os.name != 'posix', reason='requires real POSIX sessions/signals')
def test_watchdog_kills_uncooperative_isolated_parent(tmp_path):
    runner = Path(__file__).resolve().parents[1] / 'scripts/run_hindsight_pahf_reduced_dev.py'
    child = r'''
import os,signal,subprocess,sys,time
signal.signal(signal.SIGTERM,signal.SIG_IGN)
now=time.monotonic()
subprocess.Popen([sys.executable,sys.argv[1],'--_budget-watchdog',str(os.getpid()),str(now+2),str(now+3),sys.argv[2]],start_new_session=True)
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, '-c', child, str(runner), str(tmp_path)],
                              start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        _, errors = parent.communicate(timeout=12)
        assert parent.returncode == -signal.SIGKILL, errors.decode()
        receipt_path = tmp_path / 'BUDGET_HARD_STOP.json'
        deadline = time.monotonic() + 2
        while not receipt_path.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        receipt = json.loads(receipt_path.read_text())
        assert receipt['parent_pid'] == parent.pid
        assert receipt['decision'] == 'REDUCED_HARD_BUDGET_STOP'
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait()
