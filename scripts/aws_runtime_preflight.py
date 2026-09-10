"""Benign infrastructure check; creates no scientific experiment result."""
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import peft
import accelerate


def main():
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.manual_seed(4150)
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = (x @ x.T).float().square().mean()
    y.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(y) and torch.isfinite(x.grad).all()
    record = {
        "classification": "infrastructure only; no model or scientific endpoint",
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "cuda": torch.version.cuda,
        "bf16_forward_backward_finite": True,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
    }
    path = Path(__file__).resolve().parent / "RUNTIME_PREFLIGHT.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "packages"}, indent=2))


if __name__ == "__main__":
    main()
