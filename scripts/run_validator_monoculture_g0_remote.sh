#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Frozen orchestration contract for the validator-monoculture G0.
#
# Required environment:
#   VALIDATOR_MONOCULTURE_RUN_ROOT       fresh absolute evidence root
#   VALIDATOR_MONOCULTURE_PINNED_COMMIT  exact 40-hex Git commit to execute
#
# Optional existing resources:
#   VALIDATOR_MONOCULTURE_VENV     defaults to REPO_ROOT/.venv
#   VALIDATOR_MONOCULTURE_HF_HOME  defaults to REPO_ROOT/.hf_cache
#   VALIDATOR_MONOCULTURE_LOCAL_FILES_ONLY defaults to 1. Set it to 0 only
#   when an authenticated model download during the paid run is intentional.
#   VALIDATOR_MONOCULTURE_RESUME defaults to 0. Set it to 1 only to resume the
#   exact existing incomplete root after an interruption.
#
# Each phase writes exclusively below RUN_ROOT/evidence/phases. Specification-
# only generation opens the public JSONL only. The private oracle is opened by
# CPU classification and offline scoring, never by a model-generation process.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RUN_ROOT="${VALIDATOR_MONOCULTURE_RUN_ROOT:?set a fresh absolute evidence root}"
PINNED_COMMIT="${VALIDATOR_MONOCULTURE_PINNED_COMMIT:?set the exact committed revision}"
CONFIG="$REPO_ROOT/configs/validator_monoculture_g0.yaml"

PINNED_CONFIG_SHA256="6c265fd0d8e0d7a3fba23f45410e2961089408449ca424544bb0fa23e73c7cf1"
PINNED_CORPUS_SHA256="b97a59829940395b5bb4d588402b9d7ff43bf18f3cdea113ae77aef47ac709e5"
PINNED_CODE_SHA256="c5aca267a6930843285fc3590d39fb1e7fe02466b3b28ea8b763cc21e7b82188"

VENV_ROOT="${VALIDATOR_MONOCULTURE_VENV:-$REPO_ROOT/.venv}"
HF_CACHE="${VALIDATOR_MONOCULTURE_HF_HOME:-$REPO_ROOT/.hf_cache}"
PYTHON="$VENV_ROOT/bin/python"
LOCAL_FILES_ONLY="${VALIDATOR_MONOCULTURE_LOCAL_FILES_ONLY:-1}"
RESUME_MODE="${VALIDATOR_MONOCULTURE_RESUME:-0}"

fail() {
  echo "validator-monoculture orchestration: $*" >&2
  exit 2
}

[[ "$RUN_ROOT" = /* ]] || fail "RUN_ROOT must be absolute"
[[ "$RUN_ROOT" != "/" ]] || fail "RUN_ROOT may not be filesystem root"
[[ "$RESUME_MODE" == "0" || "$RESUME_MODE" == "1" ]] || \
  fail "VALIDATOR_MONOCULTURE_RESUME must be 0 or 1"
if [[ "$RESUME_MODE" == "0" ]]; then
  [[ ! -e "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || fail "refusing to overwrite $RUN_ROOT"
else
  [[ -d "$RUN_ROOT" && ! -L "$RUN_ROOT" ]] || fail "resume root is not a regular directory"
fi
[[ "$PINNED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || fail "PINNED_COMMIT must be 40 lowercase hex characters"
[[ -f "$CONFIG" && ! -L "$CONFIG" ]] || fail "missing regular frozen config: $CONFIG"
[[ -x "$PYTHON" ]] || fail "existing virtualenv Python not found: $PYTHON"
[[ -d "$HF_CACHE" && ! -L "$HF_CACHE" ]] || fail "existing HF cache not found: $HF_CACHE"
[[ "$LOCAL_FILES_ONLY" == "0" || "$LOCAL_FILES_ONLY" == "1" ]] || \
  fail "VALIDATOR_MONOCULTURE_LOCAL_FILES_ONLY must be 0 or 1"

RUN_PARENT="$(dirname -- "$RUN_ROOT")"
RUN_BASENAME="$(basename -- "$RUN_ROOT")"
[[ -d "$RUN_PARENT" && ! -L "$RUN_PARENT" ]] || fail "RUN_ROOT parent must be an existing regular directory"
command -v flock >/dev/null 2>&1 || fail "flock is required for an exclusive run lease"
exec 9>"$RUN_PARENT/.${RUN_BASENAME}.validator-monoculture.lock"
flock -n 9 || fail "another process already holds the run-root lease"
printf 'pid=%s\nhost=%s\nstarted_utc=%s\nrun_root=%s\n' \
  "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUN_ROOT" >&9

if [[ "$RESUME_MODE" == "1" && -f "$RUN_ROOT/COMPLETION_MANIFEST.json" ]]; then
  "$PYTHON" - "$RUN_ROOT" "$RUN_PARENT" "$RUN_BASENAME" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
parent = Path(sys.argv[2]).resolve()
basename = sys.argv[3]
manifest_path = root / "COMPLETION_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    manifest.get("kind") != "validator_monoculture_g0_orchestration"
    or manifest.get("status") != "generation_complete__offline_analysis_pending"
):
    raise SystemExit("existing completion manifest is not a valid terminal record")
listed = manifest.get("artifacts_sha256")
if not isinstance(listed, dict) or not listed:
    raise SystemExit("existing completion manifest has no artifact inventory")
for relative, expected in listed.items():
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"completed artifact is missing or non-regular: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"completed artifact checksum differs: {relative}")
temporary = sorted(root.glob(".COMPLETION_MANIFEST.*"))
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
    and path.name != "COMPLETION_MANIFEST.json"
    and path not in temporary
}
if actual != set(listed):
    raise SystemExit("completed root has unlisted files beyond recoverable terminal temporaries")
if temporary:
    recovery = parent / f".{basename}.terminal-recovery"
    recovery.mkdir(mode=0o700, exist_ok=True)
    for path in temporary:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = recovery / f"{path.name.lstrip('.')}.{digest}.bin"
        if destination.exists():
            raise SystemExit("terminal recovery destination already exists")
        os.rename(path, destination)
    for directory_path in (recovery, root, parent):
        descriptor = os.open(directory_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
PY
  echo "validator-monoculture generation was already complete; terminal state verified"
  exit 0
fi

ACTUAL_COMMIT="$(git -C "$REPO_ROOT" rev-parse --verify HEAD)"
[[ "$ACTUAL_COMMIT" == "$PINNED_COMMIT" ]] || \
  fail "Git revision mismatch: expected $PINNED_COMMIT, observed $ACTUAL_COMMIT"
git -C "$REPO_ROOT" diff --quiet || fail "tracked repository files are modified"
git -C "$REPO_ROOT" diff --cached --quiet || fail "staged files differ from HEAD"
UNTRACKED_RELEVANT="$(git -C "$REPO_ROOT" ls-files --others --exclude-standard)"
[[ -z "$UNTRACKED_RELEVANT" ]] || fail "untracked repository files would bypass the commit pin"

ACTUAL_CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
[[ "$ACTUAL_CONFIG_SHA256" == "$PINNED_CONFIG_SHA256" ]] || \
  fail "config hash mismatch: expected $PINNED_CONFIG_SHA256, observed $ACTUAL_CONFIG_SHA256"

export PYTHONPATH="$REPO_ROOT/src"
export PYTHONNOUSERSITE=1
export PYTHONSAFEPATH=1
export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export TRANSFORMERS_CACHE="$HF_CACHE/hub"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0

"$PYTHON" - <<'PY'
import platform
import sys
if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
    raise SystemExit("the frozen generation runtime requires Linux Python 3.12")
PY

ACTUAL_CODE_SHA256="$($PYTHON - "$REPO_ROOT/src" <<'PY'
from pathlib import Path
import sys
from validator_monoculture.prepare import _code_inventory
print(_code_inventory(Path(sys.argv[1]).resolve())[1])
PY
)"
[[ "$ACTUAL_CODE_SHA256" == "$PINNED_CODE_SHA256" ]] || \
  fail "code-tree hash mismatch: expected $PINNED_CODE_SHA256, observed $ACTUAL_CODE_SHA256"

if [[ "$RESUME_MODE" == "0" ]]; then
  mkdir -- "$RUN_ROOT"
  mkdir -- "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints"
else
  [[ -d "$RUN_ROOT/logs" && -d "$RUN_ROOT/checkpoints" ]] || \
    fail "resume root lacks logs/checkpoints directories"
fi
RUN_ROOT_CREATED=1
CURRENT_STAGE="preparation"

RUN_BINDING_SHA256="$($PYTHON - "$RUN_ROOT" "$RESUME_MODE" "$ACTUAL_COMMIT" \
  "$ACTUAL_CONFIG_SHA256" "$PINNED_CORPUS_SHA256" "$ACTUAL_CODE_SHA256" <<'PY'
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sys
import tempfile

root = Path(sys.argv[1]).resolve()
resume = sys.argv[2] == "1"
static_binding = {
    "kind": "validator_monoculture_g0_run_binding",
    "git_commit": sys.argv[3],
    "config_sha256": sys.argv[4],
    "corpus_sha256": sys.argv[5],
    "code_tree_sha256": sys.argv[6],
    "pythonhashseed": "0",
    "models": {
        "qwen3_5": ["Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"],
        "gemma4": ["google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"],
    },
    "generation_package_versions": {
        "accelerate": "1.14.0",
        "huggingface-hub": "1.27.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "transformers": "5.15.0",
    },
    "generation_environment": {
        "platform_system": "Linux",
        "python_major_minor": "3.12",
        "torch_version_prefix": "2.7.1",
        "cuda_version": "12.8",
        "minimum_device_memory_bytes": 85899345920,
    },
}
target = root / "RUN_BINDING.json"
if resume:
    try:
        observed = target.read_bytes()
    except OSError as exc:
        raise SystemExit("resume root lacks a readable RUN_BINDING.json") from exc
    try:
        prior = json.loads(observed)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit("resume RUN_BINDING.json is malformed") from exc
    run_id = prior.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{64}", run_id):
        raise SystemExit("resume RUN_BINDING.json lacks a valid unique run id")
    binding = {
        **static_binding,
        "run_id": run_id,
        "remote_run_root": str(root),
    }
    payload = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if observed != payload:
        raise SystemExit("resume inputs differ from immutable RUN_BINDING.json")
else:
    binding = {
        **static_binding,
        "run_id": secrets.token_hex(32),
        "remote_run_root": str(root),
    }
    payload = (json.dumps(binding, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(prefix=".RUN_BINDING.", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        os.unlink(temporary)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
print(hashlib.sha256(payload).hexdigest())
PY
)"
export VALIDATOR_MONOCULTURE_RUN_BINDING_SHA256="$RUN_BINDING_SHA256"

write_failure_manifest() {
  local exit_code="$1"
  trap - ERR
  set +e
  if [[ "${RUN_ROOT_CREATED:-0}" == "1" && ! -e "$RUN_ROOT/FAILURE_MANIFEST.json" ]]; then
    "$PYTHON" - "$RUN_ROOT" "$CURRENT_STAGE" "$exit_code" "$ACTUAL_COMMIT" \
      "$ACTUAL_CONFIG_SHA256" "$PINNED_CORPUS_SHA256" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
excluded = {"FAILURE_MANIFEST.json", "COMPLETION_MANIFEST.json"}
files = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name not in excluded
}
manifest = {
    "kind": "validator_monoculture_g0_orchestration",
    "status": "failed",
    "failed_stage": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "git_commit": sys.argv[4],
    "config_sha256": sys.argv[5],
    "corpus_sha256": sys.argv[6],
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "partial_artifacts_sha256": files,
}
with (root / "FAILURE_MANIFEST.json").open("x", encoding="utf-8", newline="\n") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  fi
  exit "$exit_code"
}
trap 'write_failure_manifest $?' ERR
trap 'write_failure_manifest 130' INT
trap 'write_failure_manifest 143' TERM

preserve_stage_log() {
  local stage="$1"
  local log="$RUN_ROOT/logs/$stage.log"
  if [[ -f "$log" ]]; then
    local recovery="$RUN_ROOT/recovery/logs"
    local digest stamp
    mkdir -p -- "$recovery"
    digest="$(sha256sum -- "$log" | awk '{print $1}')"
    stamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
    mv -- "$log" "$recovery/${stage}.${stamp}.${digest}.log"
  fi
}

run_stage() {
  local stage="$1"
  shift
  if [[ -f "$RUN_ROOT/checkpoints/$stage.complete" ]]; then
    case "$stage" in
      00_prepare_corpus)
        validate_preparation
        ;;
      01a_oracle_preflight)
        "$PYTHON" - "$RUN_ROOT/logs/$stage.log" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("kind") != "validator_monoculture_oracle_preflight"
    or value.get("status") != "PASS"
    or value.get("task_count") != 32
    or any(row.get("plausible_mutant_count", 0) < 1 for row in value.get("tasks", []))
):
    raise SystemExit("checkpointed oracle preflight is invalid")
PY
        ;;
      *) fail "no checkpoint validator is defined for $stage" ;;
    esac
    return 0
  fi
  CURRENT_STAGE="$stage"
  printf '%s\n' "$stage" > "$RUN_ROOT/CURRENT_STAGE"
  preserve_stage_log "$stage"
  "$@" > "$RUN_ROOT/logs/$stage.log" 2>&1
  printf 'stage=%s\nstatus=complete\n' "$stage" > "$RUN_ROOT/checkpoints/$stage.complete"
}

validate_preparation() {
  "$PYTHON" - "$RUN_ROOT/corpus/PREPARATION_MANIFEST.json" \
    "$PINNED_CONFIG_SHA256" "$PINNED_CORPUS_SHA256" "$PINNED_CODE_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

manifest_path = Path(sys.argv[1])
value = json.loads(manifest_path.read_text(encoding="utf-8"))
if (
    value.get("kind") != "validator_monoculture_g0_preparation"
    or value.get("config_sha256") != sys.argv[2]
    or value.get("corpus_sha256") != sys.argv[3]
    or value.get("code_tree_sha256") != sys.argv[4]
    or value.get("task_count") != 32
):
    raise SystemExit("prepared corpus does not match frozen commitments")
root = manifest_path.parent
inputs = value.get("input_sha256")
if not isinstance(inputs, dict):
    raise SystemExit("preparation manifest lacks input commitments")
for relative, expected in inputs.items():
    path = root / relative
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"prepared input checksum mismatch: {relative}")
PY
}

validate_completed_phase() {
  "$PYTHON" - "$1" "$RUN_BINDING_SHA256" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

phase = Path(sys.argv[1])
manifest = json.loads((phase / "MANIFEST.json").read_text(encoding="utf-8"))
if manifest.get("state") != "COMPLETE" or not (phase / "COMPLETE").is_file():
    raise SystemExit("phase is not complete")
if (phase / "COMPLETE").stat().st_size != 0 or (phase / "RUNNING.json").exists():
    raise SystemExit("phase has inconsistent terminal markers")
if manifest.get("run_binding_sha256") != sys.argv[2]:
    raise SystemExit("phase is not bound to this immutable run")
files = manifest.get("files")
if not isinstance(files, dict) or not files:
    raise SystemExit("phase manifest lacks a file inventory")
observed = {path.name for path in phase.iterdir() if path.is_file()}
expected_layout = set(files) | {"MANIFEST.json", "COMPLETE"}
if observed != expected_layout:
    raise SystemExit(f"phase layout is not closed: {sorted(observed ^ expected_layout)}")
for name, expected in files.items():
    if not isinstance(expected, dict) or set(expected) != {"sha256", "bytes"}:
        raise SystemExit(f"malformed phase commitment: {name}")
    path = phase / name
    if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
        raise SystemExit(f"phase file size mismatch: {name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
        raise SystemExit(f"phase file hash mismatch: {name}")
PY
}

run_runner_stage() {
  local stage="$1"
  local phase_name="$2"
  shift 2
  local phase="$RUN_ROOT/evidence/phases/$phase_name"
  if [[ -f "$RUN_ROOT/checkpoints/$stage.complete" ]]; then
    validate_completed_phase "$phase"
    return 0
  fi
  if [[ -e "$phase/COMPLETE" ]]; then
    validate_completed_phase "$phase"
    printf 'stage=%s\nstatus=complete\n' "$stage" \
      > "$RUN_ROOT/checkpoints/$stage.complete"
    return 0
  fi
  local resume_args=()
  if [[ -e "$phase" ]]; then
    [[ "$RESUME_MODE" == "1" ]] || fail "incomplete phase exists outside resume mode: $phase"
    resume_args=(--resume)
  fi
  run_stage "$stage" "$@" "${resume_args[@]}"
}

if [[ "$RESUME_MODE" == "1" && -d "$RUN_ROOT/corpus" && \
      ! -f "$RUN_ROOT/corpus/PREPARATION_MANIFEST.json" ]]; then
  "$PYTHON" - "$RUN_ROOT" <<'PY'
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
root = Path(sys.argv[1])
source = root / "corpus"
recovery = root / "recovery"
recovery.mkdir(exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
target = recovery / f"incomplete-corpus-{stamp}"
os.rename(source, target)
PY
fi

if [[ ! -f "$RUN_ROOT/checkpoints/00_prepare_corpus.complete" && \
      -f "$RUN_ROOT/corpus/PREPARATION_MANIFEST.json" ]]; then
  "$PYTHON" - "$RUN_ROOT/corpus/PREPARATION_MANIFEST.json" \
    "$PINNED_CONFIG_SHA256" "$PINNED_CORPUS_SHA256" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if (
    value.get("kind") != "validator_monoculture_g0_preparation"
    or value.get("config_sha256") != sys.argv[2]
    or value.get("corpus_sha256") != sys.argv[3]
):
    raise SystemExit("existing prepared corpus does not match the frozen commitments")
PY
  printf 'stage=00_prepare_corpus\nstatus=complete\n' \
    > "$RUN_ROOT/checkpoints/00_prepare_corpus.complete"
fi

run_stage "00_prepare_corpus" \
  "$PYTHON" -m validator_monoculture.prepare \
  --destination "$RUN_ROOT/corpus" \
  --config "$CONFIG" \
  --expected-config-sha256 "$PINNED_CONFIG_SHA256" \
  --expected-corpus-sha256 "$PINNED_CORPUS_SHA256" \
  --expected-code-sha256 "$PINNED_CODE_SHA256"

if [[ ! -f "$RUN_ROOT/checkpoints/01_verify_preparation.complete" ]]; then
CURRENT_STAGE="01_verify_preparation"
preserve_stage_log "01_verify_preparation"
"$PYTHON" - "$RUN_ROOT/corpus/PREPARATION_MANIFEST.json" \
  "$PINNED_CONFIG_SHA256" "$PINNED_CORPUS_SHA256" \
  > "$RUN_ROOT/logs/01_verify_preparation.log" 2>&1 <<'PY'
import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("kind") != "validator_monoculture_g0_preparation":
    raise SystemExit("wrong preparation-manifest kind")
if manifest.get("config_sha256") != sys.argv[2]:
    raise SystemExit("prepared config hash does not match frozen hash")
if manifest.get("corpus_sha256") != sys.argv[3]:
    raise SystemExit("prepared corpus hash does not match frozen hash")
if not isinstance(manifest.get("code_tree_sha256"), str):
    raise SystemExit("preparation manifest is missing the code-tree hash")
print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
PY
printf 'stage=01_verify_preparation\nstatus=complete\n' \
  > "$RUN_ROOT/checkpoints/01_verify_preparation.complete"
fi
validate_preparation

run_stage "01a_oracle_preflight" \
  "$PYTHON" -m validator_monoculture.oracle_preflight \
  --public-corpus "$RUN_ROOT/corpus/public/tasks.jsonl" \
  --private-oracles "$RUN_ROOT/corpus/private/oracles.jsonl" \
  --timeout-seconds 2.0

if [[ ! -f "$RUN_ROOT/checkpoints/01b_model_cache_preflight.complete" ]]; then
CURRENT_STAGE="01b_model_cache_preflight"
preserve_stage_log "01b_model_cache_preflight"
"$PYTHON" - "$LOCAL_FILES_ONLY" > "$RUN_ROOT/logs/01b_model_cache_preflight.log" 2>&1 <<'PY'
from __future__ import annotations

import json
import importlib.metadata
import gc
import hashlib
import platform
import sys

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import Gemma4UnifiedForConditionalGeneration, Qwen3_5ForCausalLM
from validator_monoculture.runtime import TextRuntime

if platform.system() != "Linux" or sys.version_info[:2] != (3, 12):
    raise SystemExit("the frozen gate requires Linux Python 3.12")
if not torch.__version__.startswith("2.7.1"):
    raise SystemExit(f"the frozen gate requires PyTorch 2.7.1, observed {torch.__version__}")
if str(torch.version.cuda) != "12.8":
    raise SystemExit(f"the frozen gate requires CUDA 12.8, observed {torch.version.cuda}")

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("the frozen gate requires exactly one CUDA-visible GPU")
properties = torch.cuda.get_device_properties(0)
if properties.total_memory < 80 * 1024**3:
    raise SystemExit(f"insufficient CUDA memory: {properties.total_memory} bytes")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("the CUDA device must support bfloat16")

expected_packages = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.27.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.0",
}
observed_packages = {
    name: importlib.metadata.version(name) for name in expected_packages
}
if observed_packages != expected_packages:
    raise SystemExit(
        f"frozen package versions differ: expected={expected_packages}, observed={observed_packages}"
    )

local_only = sys.argv[1] == "1"
models = [
    ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
    ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
]
snapshots = {
    model_id: snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_files_only=local_only,
    )
    for model_id, revision in models
}
smokes = {}
for index, (model_id, revision) in enumerate(models):
    runtime = TextRuntime(model_id, revision, local_files_only=True)
    completion = runtime.generate(
        "Reply with the single word READY.",
        seed=20260830 + index,
        max_new_tokens=32,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
    )
    if not isinstance(completion, str) or not completion:
        raise SystemExit(f"empty exact-runtime smoke output from {model_id}")
    smokes[model_id] = {
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
        "runtime_provenance": runtime.provenance(),
    }
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
print(json.dumps({
    "device": properties.name,
    "device_memory_bytes": properties.total_memory,
    "compute_capability": list(torch.cuda.get_device_capability(0)),
    "transformers_version": transformers.__version__,
    "torch_version": torch.__version__,
    "cuda_version": str(torch.version.cuda),
    "python_version": sys.version,
    "package_versions": observed_packages,
    "loader_classes": [
        Qwen3_5ForCausalLM.__name__, Gemma4UnifiedForConditionalGeneration.__name__
    ],
    "snapshots": snapshots,
    "exact_runtime_smokes": smokes,
}, sort_keys=True))
PY
printf 'stage=01b_model_cache_preflight\nstatus=complete\n' \
  > "$RUN_ROOT/checkpoints/01b_model_cache_preflight.complete"
fi

"$PYTHON" - "$RUN_ROOT/logs/01b_model_cache_preflight.log" <<'PY'
import json
from pathlib import Path
import sys
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_packages = {
    "accelerate": "1.14.0",
    "huggingface-hub": "1.27.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "transformers": "5.15.0",
}
if value.get("package_versions") != expected_packages:
    raise SystemExit("checkpointed model preflight has package drift")
if (
    not str(value.get("python_version", "")).startswith("3.12.")
    or not str(value.get("torch_version", "")).startswith("2.7.1")
    or value.get("cuda_version") != "12.8"
    or int(value.get("device_memory_bytes", 0)) < 85899345920
):
    raise SystemExit("checkpointed model preflight has generation-environment drift")
smokes = value.get("exact_runtime_smokes", {})
if len(smokes) != 2:
    raise SystemExit("checkpointed model preflight lacks both runtime smokes")
expected_revisions = {
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "google/gemma-4-12B-it": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
}
if set(smokes) != set(expected_revisions):
    raise SystemExit("checkpointed model preflight has the wrong model set")
for model_id, smoke in smokes.items():
    provenance = smoke.get("runtime_provenance", {})
    if (
        not provenance.get("chat_template_sha256")
        or provenance.get("model_id") != model_id
        or provenance.get("model_revision") != expected_revisions[model_id]
    ):
        raise SystemExit("checkpointed model preflight lacks exact runtime provenance")
PY

PUBLIC_CORPUS="$RUN_ROOT/corpus/public/tasks.jsonl"
PRIVATE_ORACLES="$RUN_ROOT/corpus/private/oracles.jsonl"
EVIDENCE_ROOT="$RUN_ROOT/evidence"
LOCAL_FLAG=()
if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  LOCAL_FLAG=(--local-files-only)
fi

run_runner_stage "02_patch_qwen3_5" "patches_qwen3_5" \
  "$PYTHON" -m validator_monoculture.runner patch-run \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --family qwen3_5 --output-root "$EVIDENCE_ROOT" "${LOCAL_FLAG[@]}"

run_runner_stage "03_patch_gemma4" "patches_gemma4" \
  "$PYTHON" -m validator_monoculture.runner patch-run \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --family gemma4 --output-root "$EVIDENCE_ROOT" "${LOCAL_FLAG[@]}"

run_runner_stage "04_classify_patches" "classifications" \
  "$PYTHON" -m validator_monoculture.runner classify \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --private-oracles "$PRIVATE_ORACLES" --output-root "$EVIDENCE_ROOT"

run_runner_stage "05_specification_only_qwen3_5" "tests_qwen3_5_spec_only" \
  "$PYTHON" -m validator_monoculture.runner test-run \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --prompt-mode spec_only --family qwen3_5 --output-root "$EVIDENCE_ROOT" \
  "${LOCAL_FLAG[@]}"

run_runner_stage "06_specification_only_gemma4" "tests_gemma4_spec_only" \
  "$PYTHON" -m validator_monoculture.runner test-run \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --prompt-mode spec_only --family gemma4 --output-root "$EVIDENCE_ROOT" \
  "${LOCAL_FLAG[@]}"

run_runner_stage "07_patch_aware_qwen3_5" "tests_qwen3_5_patch_aware" \
  "$PYTHON" -m validator_monoculture.runner test-run \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --prompt-mode patch_aware --family qwen3_5 --output-root "$EVIDENCE_ROOT" \
  "${LOCAL_FLAG[@]}"

run_runner_stage "08_patch_aware_gemma4" "tests_gemma4_patch_aware" \
  "$PYTHON" -m validator_monoculture.runner test-run \
  --config "$CONFIG" --public-corpus "$PUBLIC_CORPUS" \
  --prompt-mode patch_aware --family gemma4 --output-root "$EVIDENCE_ROOT" \
  "${LOCAL_FLAG[@]}"

CURRENT_STAGE="complete"
printf 'complete\n' > "$RUN_ROOT/CURRENT_STAGE"
for phase_name in \
  patches_qwen3_5 patches_gemma4 classifications \
  tests_qwen3_5_spec_only tests_gemma4_spec_only \
  tests_qwen3_5_patch_aware tests_gemma4_patch_aware; do
  validate_completed_phase "$RUN_ROOT/evidence/phases/$phase_name"
done
"$PYTHON" - "$RUN_ROOT" "$ACTUAL_COMMIT" "$ACTUAL_CONFIG_SHA256" \
  "$PINNED_CORPUS_SHA256" "$PINNED_CODE_SHA256" "$RUN_BINDING_SHA256" <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1])
preparation = json.loads((root / "corpus" / "PREPARATION_MANIFEST.json").read_text(encoding="utf-8"))
if preparation.get("code_tree_sha256") != sys.argv[5]:
    raise SystemExit("preparation code-tree hash differs from final pin")
binding_path = root / "RUN_BINDING.json"
if hashlib.sha256(binding_path.read_bytes()).hexdigest() != sys.argv[6]:
    raise SystemExit("immutable run binding hash changed")
recovery = root / "recovery" / "terminal-temporaries"
for stale in sorted(root.glob(".COMPLETION_MANIFEST.*")):
    recovery.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(stale.read_bytes()).hexdigest()
    os.rename(stale, recovery / f"{stale.name.lstrip('.')}.{digest}.bin")
excluded = {"COMPLETION_MANIFEST.json"}
files = {
    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*"))
    if path.is_file() and path.name not in excluded
}
evidence_root = root / "evidence"
evidence_files = {
    path.relative_to(evidence_root).as_posix(): {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    for path in sorted(evidence_root.rglob("*"))
    if path.is_file()
}
evidence_payload = (
    json.dumps(evidence_files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    + "\n"
).encode("utf-8")
stages = [
    "00_prepare_corpus",
    "01_verify_preparation",
    "01a_oracle_preflight",
    "01b_model_cache_preflight",
    "02_patch_qwen3_5",
    "03_patch_gemma4",
    "04_classify_patches",
    "05_specification_only_qwen3_5",
    "06_specification_only_gemma4",
    "07_patch_aware_qwen3_5",
    "08_patch_aware_gemma4",
]
for stage in stages:
    if not (root / "checkpoints" / f"{stage}.complete").is_file():
        raise SystemExit(f"missing completion checkpoint for {stage}")
expected_phases = {
    "patches_qwen3_5", "patches_gemma4", "classifications",
    "tests_qwen3_5_spec_only", "tests_gemma4_spec_only",
    "tests_qwen3_5_patch_aware", "tests_gemma4_patch_aware",
}
phase_root = evidence_root / "phases"
if {path.name for path in phase_root.iterdir() if path.is_dir()} != expected_phases:
    raise SystemExit("terminal phase set differs from the frozen seven phases")
for name in expected_phases:
    phase = phase_root / name
    if (phase / "RUNNING.json").exists() or not (phase / "COMPLETE").is_file():
        raise SystemExit(f"phase has inconsistent terminal state: {name}")
    phase_manifest = json.loads((phase / "MANIFEST.json").read_text(encoding="utf-8"))
    if phase_manifest.get("run_binding_sha256") != sys.argv[6]:
        raise SystemExit(f"phase run binding differs: {name}")
manifest = {
    "kind": "validator_monoculture_g0_orchestration",
    "status": "generation_complete__offline_analysis_pending",
    "git_commit": sys.argv[2],
    "config_sha256": sys.argv[3],
    "corpus_sha256": sys.argv[4],
    "code_tree_sha256": sys.argv[5],
    "run_binding_sha256": sys.argv[6],
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "completed_stages": stages,
    "artifacts_sha256": files,
    "evidence_root_sha256": hashlib.sha256(evidence_payload).hexdigest(),
}
payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
target = root / "COMPLETION_MANIFEST.json"
descriptor, temporary = tempfile.mkstemp(prefix=".COMPLETION_MANIFEST.", dir=root)
try:
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, target)
    os.unlink(temporary)
    directory = os.open(root, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY

trap - ERR
echo "validator-monoculture generation complete; offline verification is still required"
