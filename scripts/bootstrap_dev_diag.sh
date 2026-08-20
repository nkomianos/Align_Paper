#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: bootstrap_dev_diag.sh PROJECT_ROOT BUNDLE_ROOT RUNTIME_ROOT ATTESTATION_PATH" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$1" && pwd)"
BUNDLE_ROOT="$(cd "$2" && pwd)"
RUNTIME_ROOT="$3"
ATTESTATION_PATH="$4"
LOCK_PATH="$PROJECT_ROOT/requirements/h100-cu12x.lock"

if [[ ! -f "$LOCK_PATH" ]]; then
  echo "Missing pinned GPU dependency lock: $LOCK_PATH" >&2
  exit 2
fi
if [[ -e "$RUNTIME_ROOT/.venv" ]]; then
  echo "Refusing to reuse an existing diagnostic virtual environment." >&2
  exit 2
fi
mkdir -p "$RUNTIME_ROOT" "$(dirname "$ATTESTATION_PATH")"

export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PROJECT_ROOT/src"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Expected the frozen aarch64 GH200 runtime; found $(uname -m)." >&2
  exit 2
fi

nvidia-smi
python3 - "$RUNTIME_ROOT" <<'PY'
import shutil
import sys

import torch

if not (3, 10) <= sys.version_info < (3, 14):
    raise SystemExit(f"Python >=3.10,<3.14 is required; found {sys.version}")
if not torch.cuda.is_available():
    raise SystemExit("Preinstalled PyTorch cannot see CUDA; stop instead of repairing drivers")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"The paid contract requires exactly one CUDA device; found {torch.cuda.device_count()}")
version_core = torch.__version__.split("+", 1)[0].split(".")
try:
    version = (int(version_core[0]), int(version_core[1]))
except (IndexError, ValueError) as exc:
    raise SystemExit(f"Cannot parse provider PyTorch version {torch.__version__!r}") from exc
if not (2, 5) <= version < (3, 0):
    raise SystemExit(f"Provider PyTorch must be >=2.5,<3; found {torch.__version__}")
name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
if name != "NVIDIA GH200 480GB":
    raise SystemExit(
        "The frozen diagnostic contract requires device name "
        f"'NVIDIA GH200 480GB'; found {name!r}"
    )
if capability[0] != 9 or memory_gib < 90:
    raise SystemExit(
        "Expected one GH200 compute-capability-9.x GPU with at least 90 GiB; "
        f"found capability {capability}, memory {memory_gib:.1f} GiB"
    )
free_gib = shutil.disk_usage(sys.argv[1]).free / (1024**3)
if free_gib < 40:
    raise SystemExit(f"At least 40 GiB runtime storage is required; found {free_gib:.1f} GiB")
print({
    "provider_torch": torch.__version__,
    "provider_cuda": torch.version.cuda,
    "gpu": name,
    "compute_capability": capability,
    "memory_gib": round(memory_gib, 1),
    "runtime_free_disk_gib": round(free_gib, 1),
})
PY

python3 -m venv --system-site-packages "$RUNTIME_ROOT/.venv"
# shellcheck disable=SC1091
source "$RUNTIME_ROOT/.venv/bin/activate"

# Force every explicitly locked experiment root into the venv even when the
# provider image exposes a matching global package.  This pre-install probe is
# deliberately standard-library-only: importing dev_diag_bootstrap here would
# require ``packaging`` before the lock has installed its dependency closure.
python - "$LOCK_PATH" <<'PY'
import sys
from pathlib import Path

roots = []
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line and not line.startswith(("#", "--")):
        roots.append(line)
if not roots or any(line.lower().split("[", 1)[0].startswith("torch==") for line in roots):
    raise SystemExit("Diagnostic lock must contain non-Torch experiment roots")
print({"locked_non_torch_experiment_roots": roots})
PY
python -m pip install --ignore-installed --no-deps --only-binary=:all: \
  --requirement "$LOCK_PATH"
python -m pip install --only-binary=:all: --requirement "$LOCK_PATH"

# Pip can still treat an unpinned transitive dependency from the provider image
# as satisfied. Compute the complete non-Torch experiment closure, then overlay
# every such external distribution into the venv at its exact resolved version.
OVERLAY_PATH="$RUNTIME_ROOT/experiment_dependency_overlay.lock"
python - "$LOCK_PATH" "$OVERLAY_PATH" <<'PY'
from pathlib import Path
import sys
from under_extinction.dev_diag_bootstrap import plan_experiment_dependency_overlay

requirements = plan_experiment_dependency_overlay(sys.argv[1], Path(sys.prefix))
Path(sys.argv[2]).write_text(
    "".join(f"{requirement}\n" for requirement in requirements),
    encoding="utf-8",
    newline="\n",
)
print({"forced_experiment_dependency_overlay": requirements})
PY
if [[ -s "$OVERLAY_PATH" ]]; then
  python -m pip install --ignore-installed --no-deps --only-binary=:all: \
    --requirement "$OVERLAY_PATH"
fi

# A global pip check may report unrelated provider desktop/Jupyter packages.
# Record it, but use the complete reachable-closure check below as the
# authoritative experiment check, matching bootstrap_lambda.sh.
if ! python -m pip check; then
  echo "Global pip check found host-image packages outside the experiment closure; running scoped validation."
fi

python - "$LOCK_PATH" "$PROJECT_ROOT" "$BUNDLE_ROOT" "$ATTESTATION_PATH" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from importlib.util import find_spec
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys

lock_path = Path(sys.argv[1]).resolve()
project_root = Path(sys.argv[2]).resolve()
bundle_root = Path(sys.argv[3]).resolve()
attestation_path = Path(sys.argv[4]).resolve()
venv_root = Path(sys.prefix).resolve()

import attrs
import datasets
import fsspec
import idna
import jinja2
import markupsafe
import mpmath
import networkx
import numpy
import pandas
import peft
import PIL
import psutil
import pyarrow
import scipy
import sklearn
import sympy
import threadpoolctl
import torch
import torch.nn.functional as functional
import typing_extensions
import urllib3
from transformers import Qwen3_5ForCausalLM
import under_extinction
from under_extinction.dev_diag import load_dev_diag_spec
from under_extinction.dev_diag_bootstrap import (
    audit_dependency_closures,
    parse_lock_requirements,
)
from under_extinction.dev_diag_deployment import (
    verify_dev_diag_bootstrap_attestation,
    verify_dev_diag_bundle,
)
from under_extinction.dev_diag_evaluation import evaluate_dev_diagnostic


locked_roots = parse_lock_requirements(lock_path)
bundle_manifest = verify_dev_diag_bundle(bundle_root)
try:
    dependency_closure = audit_dependency_closures(lock_path, venv_root)
except ValueError as exc:
    raise SystemExit(str(exc)) from exc
for requirement in locked_roots:
    installed_version = version(requirement.name)
    if requirement.specifier and installed_version not in requirement.specifier:
        raise SystemExit(
            f"Pinned dependency mismatch for {requirement.name}: "
            f"{installed_version} not in {requirement.specifier}"
        )
if Qwen3_5ForCausalLM.__name__ != "Qwen3_5ForCausalLM":
    raise SystemExit("Pinned Transformers does not expose Qwen3_5ForCausalLM")
package_path = Path(under_extinction.__file__).resolve()
if not package_path.is_relative_to(project_root / "src"):
    raise SystemExit(f"DID-v1 source imported outside the immutable project: {package_path}")
if (
    load_dev_diag_spec.__name__ != "load_dev_diag_spec"
    or verify_dev_diag_bundle.__name__ != "verify_dev_diag_bundle"
    or evaluate_dev_diagnostic.__name__ != "evaluate_dev_diagnostic"
):
    raise SystemExit("DID-v1 source import probe differs")

isolated_modules = (
    attrs,
    datasets,
    fsspec,
    idna,
    jinja2,
    markupsafe,
    mpmath,
    networkx,
    numpy,
    pandas,
    peft,
    PIL,
    psutil,
    pyarrow,
    scipy,
    sklearn,
    sympy,
    threadpoolctl,
    typing_extensions,
    urllib3,
)
module_origins: dict[str, str] = {}
for module in isolated_modules:
    module_path = Path(module.__file__).resolve()
    module_origins[module.__name__] = str(module_path)
    if not module_path.is_relative_to(venv_root):
        raise SystemExit(
            f"Experiment dependency {module.__name__} leaked from outside the venv: "
            f"{module_path}"
        )
torch_path = Path(torch.__file__).resolve()
if torch_path.is_relative_to(venv_root):
    raise SystemExit(f"Expected provider PyTorch outside the venv; found {torch_path}")

# Exact provider-Torch <-> pinned-NumPy CPU ABI roundtrip.
cpu_probe = torch.tensor([1.25, -0.5], dtype=torch.float32, device="cpu")
probe_array = cpu_probe.numpy()
probe_roundtrip = torch.from_numpy(numpy.asarray(probe_array))
if not torch.equal(cpu_probe, probe_roundtrip):
    raise SystemExit("Provider PyTorch and pinned NumPy failed the exact CPU ABI roundtrip")

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

if platform.machine() != "aarch64" or torch.cuda.device_count() != 1:
    raise SystemExit("Hardware identity changed between bootstrap checks")
properties = torch.cuda.get_device_properties(0)
device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
memory_gib = properties.total_memory / (1024**3)
if device_name != "NVIDIA GH200 480GB" or capability[0] != 9 or memory_gib < 90:
    raise SystemExit("GH200 identity changed between bootstrap checks")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The frozen BF16 diagnostic contract is unsupported")

# Exercise provider CUDA, BF16 matmul, and PyTorch SDPA. These probes neither
# download nor instantiate the research model.
cuda_probe = torch.tensor([1.25, -0.5], dtype=torch.float32, device="cuda")
cuda_result = (cuda_probe * 2.0).cpu()
if not torch.equal(cuda_result, torch.tensor([2.5, -1.0])):
    raise SystemExit("Provider CUDA failed the exact tensor probe")
left = torch.eye(4, dtype=torch.bfloat16, device="cuda")
if not torch.equal((left @ left).float().cpu(), torch.eye(4)):
    raise SystemExit("Provider CUDA BF16 matmul probe failed")
query = torch.arange(32, dtype=torch.float32, device="cuda").reshape(1, 1, 4, 8)
query = query.to(torch.bfloat16)
sdpa = functional.scaled_dot_product_attention(query, query, query)
if sdpa.shape != query.shape or not torch.isfinite(sdpa.float()).all():
    raise SystemExit("Provider PyTorch SDPA probe failed")
torch.cuda.synchronize()

nvidia_query = subprocess.run(
    [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ],
    check=True,
    capture_output=True,
    text=True,
)
nvidia_rows = [row.strip() for row in nvidia_query.stdout.splitlines() if row.strip()]
if len(nvidia_rows) != 1 or "NVIDIA GH200 480GB" not in nvidia_rows[0]:
    raise SystemExit(f"nvidia-smi identity differs: {nvidia_rows}")

source_hashes = {}
for path in sorted((project_root / "src").rglob("*.py")):
    relative = path.relative_to(project_root).as_posix()
    source_hashes[f"project/{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
source_inventory_sha256 = hashlib.sha256(
    json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
project_hashes = {
    row["path"]: row["sha256"]
    for row in bundle_manifest["inventory"]
    if row["path"].startswith("project/")
}
for relative, expected_sha256 in project_hashes.items():
    observed_sha256 = hashlib.sha256((bundle_root / relative).read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise SystemExit(f"Executing project payload drifted after bundle verification: {relative}")
project_inventory_sha256 = hashlib.sha256(
    json.dumps(project_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
manifest_sha256 = hashlib.sha256(
    (bundle_root / "DEV_DIAG_BUNDLE_MANIFEST.json").read_bytes()
).hexdigest()
source_identity_sha256 = hashlib.sha256(
    json.dumps(
        bundle_manifest["source_identity"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

attestation = {
    "schema_version": "1.0",
    "kind": "did_v1_remote_bootstrap_attestation",
    "passed": True,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "bundle": {
        "manifest_sha256": manifest_sha256,
        "inventory_sha256": bundle_manifest["inventory_sha256"],
        "source_identity_sha256": source_identity_sha256,
        "git": bundle_manifest["source_identity"]["git"],
        "project_inventory_sha256": bundle_manifest["source_identity"][
            "project_inventory_sha256"
        ],
        "executing_project_payload_sha256": project_hashes,
        "executing_project_payload_inventory_sha256": project_inventory_sha256,
    },
    "hardware": {
        "architecture": platform.machine(),
        "hostname": socket.gethostname(),
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "compute_capability": list(capability),
        "total_memory_bytes": properties.total_memory,
        "total_memory_gib": memory_gib,
        "nvidia_smi_query": nvidia_rows,
    },
    "runtime": {
        "python": sys.version,
        "python_executable": sys.executable,
        "venv_root": str(venv_root),
        "torch": torch.__version__,
        "torch_path": str(torch_path),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": numpy.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "qwen_text_loader": Qwen3_5ForCausalLM.__name__,
        "module_origins": module_origins,
    },
    "dependency_closure": {
        "lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        **dependency_closure,
    },
    "kernel_contract": {
        "delta_net_policy": "torch_fallback_required",
        "optional_delta_net_packages_present": optional_delta_net_packages,
        "torch_numpy_cpu_abi_roundtrip": True,
        "cuda_tensor_probe": True,
        "cuda_bfloat16_matmul_probe": True,
        "cuda_sdpa_probe": True,
    },
    "source": {
        "package_path": str(package_path),
        "diagnostic_import_probe": True,
        "python_file_count": len(source_hashes),
        "python_file_sha256": source_hashes,
        "python_inventory_sha256": source_inventory_sha256,
    },
}
attestation_path.write_text(
    json.dumps(attestation, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)
verified_binding = verify_dev_diag_bootstrap_attestation(attestation_path, bundle_root)
print({
    "bootstrap_attestation": str(attestation_path),
    "verified_bootstrap_binding": verified_binding,
    "experiment_dependency_closure_distributions": len(
        dependency_closure["experiment_closure"]
    ),
    "provider_torch_closure_distributions": len(
        dependency_closure["provider_torch_closure"]
    ),
    "source_inventory_sha256": source_inventory_sha256,
    "hardware": attestation["hardware"],
    "kernel_contract": attestation["kernel_contract"],
})
PY

echo "DID-v1 diagnostic bootstrap passed; no model was downloaded or loaded."
