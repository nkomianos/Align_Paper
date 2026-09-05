"""Download the protocol-pinned public model; no inference or participant data."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from huggingface_hub import snapshot_download


def main():
    root = Path('/home/ubuntu/align_iclr_2027')
    started = time.time()
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    snapshot = snapshot_download(
        repo_id='Qwen/Qwen3.5-9B',
        revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a',
        cache_dir=root / 'model_cache',
        allow_patterns=['*.json', '*.safetensors', '*.jinja', 'vocab.txt', 'merges.txt', 'tokenizer.model'],
    )
    versions = {n: importlib.metadata.version(n) for n in
                ['torch', 'transformers', 'tokenizers', 'numpy', 'accelerate', 'huggingface-hub', 'safetensors']}
    receipt = {'snapshot': snapshot, 'start_unix': started, 'end_unix': time.time(),
               'architecture': platform.machine(), 'versions': versions,
               'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'inference_run': False, 'confirmation_opened': False}
    (root / 'logs/model_setup.json').write_text(json.dumps(receipt, indent=2))
    freeze = subprocess.check_output([str(root / 'venv/bin/python'), '-m', 'pip', 'freeze']).decode()
    (root / 'logs/environment_freeze.txt').write_text(freeze)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
