import argparse
import json
from pathlib import Path
from interaction_sprint.belief_reconstruction_metrics import measurement_null

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
result=measurement_null()
with a.out.open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
