"""Fast read-only CPU/import/cache readiness; no model weights loaded onto GPU."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--snapshot',type=Path,required=True)
    a=p.parse_args()
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('need exactly one CUDA GPU')
    if torch.cuda.get_device_properties(0).total_memory<70*1024**3:raise RuntimeError('need at least 70GiB; actual workload fit must be qualified')
    if not (a.snapshot/'config.json').is_file() or not list(a.snapshot.glob('*.safetensors')):raise RuntimeError('cached snapshot incomplete')
    tokenizer=AutoTokenizer.from_pretrained(a.snapshot,local_files_only=True)
    print(json.dumps({'status':'IMPORT_AND_CACHE_CHECK_PASS','python':sys.version,'architecture':platform.machine(),
        'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'transformers':importlib.metadata.version('transformers'),
        'free_disk_bytes':shutil.disk_usage(a.snapshot).free,'model_inference_performed':False},indent=2))


if __name__=='__main__':main()
