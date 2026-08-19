#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
chmod +x scripts/*.sh
mkdir -p artifacts/logs
LOG_PATH="artifacts/logs/bootstrap_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "$LOG_PATH") 2>&1

if [[ "$(uname -m)" != "x86_64" ]]; then
  echo "Expected an x86_64 H100 instance; found $(uname -m)." >&2
  exit 2
fi

nvidia-smi
python3 - <<'PY'
import sys
import shutil
import torch
if not (3, 10) <= sys.version_info < (3, 14):
    raise SystemExit(f"Python >=3.10,<3.14 is required; found {sys.version}")
if not torch.cuda.is_available():
    raise SystemExit("Preinstalled PyTorch cannot see CUDA; stop the paid instance instead of repairing drivers in place.")
version_core = torch.__version__.split("+", 1)[0].split(".")
try:
    version = (int(version_core[0]), int(version_core[1]))
except (IndexError, ValueError) as exc:
    raise SystemExit(f"Cannot parse preinstalled PyTorch version {torch.__version__!r}") from exc
if not (2, 5) <= version < (3, 0):
    raise SystemExit(
        f"The pinned Qwen3.5 runtime requires Lambda Stack PyTorch >=2.5,<3; found {torch.__version__}."
    )
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if "H100" not in name.upper():
    raise SystemExit(f"The paid contract requires an H100; found {name!r}.")
if capability[0] != 9 or memory_gib < 75:
    raise SystemExit(
        f"Expected an H100-class compute capability 9.x device with at least 75 GiB; "
        f"found capability {capability} and {memory_gib:.1f} GiB."
    )
free_disk_gib = shutil.disk_usage(".").free / (1024**3)
if free_disk_gib < 150:
    raise SystemExit(
        f"At least 150 GiB of free project-volume storage is required; found {free_disk_gib:.1f} GiB."
    )
print({
    "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": name,
    "compute_capability": capability, "memory_gib": round(memory_gib, 1),
    "free_disk_gib": round(free_disk_gib, 1),
})
PY

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: --requirement requirements/h100-cu12x.lock
python -m pip install --no-deps --editable .
python -m pip check

python - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

from packaging.requirements import Requirement
from transformers import Qwen3_5ForCausalLM

for raw in Path("requirements/h100-cu12x.lock").read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith(("#", "--")):
        continue
    requirement = Requirement(line)
    installed = version(requirement.name)
    if installed not in requirement.specifier:
        raise SystemExit(
            f"Pinned dependency mismatch for {requirement.name}: {installed} not in {requirement.specifier}"
        )
if Qwen3_5ForCausalLM.__name__ != "Qwen3_5ForCausalLM":
    raise SystemExit("The pinned Transformers release does not expose Qwen3_5ForCausalLM")
optional_delta_net_packages = {
    name: find_spec(module) is not None
    for name, module in {
        "causal-conv1d": "causal_conv1d",
        "fla-core": "fla",
        "kernels": "kernels",
    }.items()
}
present_optional_backends = sorted(
    name for name, present in optional_delta_net_packages.items() if present
)
if present_optional_backends:
    raise SystemExit(
        "The frozen torch_fallback_required contract forbids optional DeltaNet "
        f"backends, but these are importable: {present_optional_backends}"
    )
print({
    "qwen_text_loader": Qwen3_5ForCausalLM.__name__,
    "transformers": version("transformers"),
    "peft": version("peft"),
    "optional_delta_net_packages_present": optional_delta_net_packages,
    "delta_net_kernel_policy": "torch_fallback_required",
})
PY

export PYTHONPATH="$PROJECT_ROOT/src"
python -m pytest tests -q

mkdir -p .hf_cache .torch_cache .triton_cache
echo "Bootstrap complete. The released Qwen3.5 software contract is available; no model was downloaded and no experiment was launched."
