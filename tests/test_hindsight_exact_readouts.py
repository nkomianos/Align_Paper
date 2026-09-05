import json
import os
from pathlib import Path
import subprocess
import sys


def test_exact_profile_is_hash_seed_independent():
    repo = Path(__file__).resolve().parents[1]
    program = r'''
import json,hashlib,pathlib
from interaction_sprint.hindsight_execution_integrity import load_learning_dev,canonical_json
from interaction_sprint.hindsight_pahf_reduced import build_schedules
from interaction_sprint.hindsight_exact_readouts import cheap_readouts
cfg=json.loads(pathlib.Path('configs/hindsight_pahf_reduced_dev_v1.json').read_text())
learning,dev,_=load_learning_dev(pathlib.Path('artifacts/hindsight_endo_pahf_external_20260904_v3'))
panels=build_schedules(learning,cfg)['pooled_base_ids']
output=cheap_readouts(learning,dev,panels,cfg)
print(hashlib.sha256(canonical_json(output).encode()).hexdigest())
'''
    hashes = [subprocess.check_output([sys.executable, '-c', program], cwd=repo, text=True,
              env={**os.environ, 'PYTHONHASHSEED': str(seed), 'PYTHONPATH': str(repo/'src')}).strip()
              for seed in (1, 2, 3, 17, 43)]
    assert len(set(hashes)) == 1
