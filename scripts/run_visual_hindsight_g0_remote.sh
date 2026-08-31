#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${VISUAL_HINDSIGHT_RUN_ROOT:?set a fresh absolute output path}"
PINNED_COMMIT="${VISUAL_HINDSIGHT_PINNED_COMMIT:?set the audited 40-hex git commit}"
PINNED_CODE_SHA256="7dd8ba4ad80559e1a060b15776ea2f08871fa6e1508a7437912f478b48415dce"
PINNED_CONFIG_SHA256="c7a236ac5b57320c294035f758ad2db488e2facab0b38888bcf3b6e0b6899842"
CONFIG="$ROOT/configs/visual_hindsight_g0.yaml"

[[ "$RUN_ROOT" = /* ]] || { echo "RUN_ROOT must be absolute" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing to overwrite $RUN_ROOT" >&2; exit 2; }
mkdir "${RUN_ROOT}.lease" || { echo "Run-root lease already exists" >&2; exit 2; }
printf 'pid=%s\nhost=%s\nstarted_utc=%s\n' "$$" "$(hostname)" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${RUN_ROOT}.lease/OWNER"
mkdir "$RUN_ROOT"

cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0

[[ "$(git rev-parse HEAD)" == "$PINNED_COMMIT" ]] || { echo "Git commit mismatch" >&2; exit 3; }
if [[ -n "$(git status --porcelain -- src/visual_hindsight_g0 configs/visual_hindsight_g0.yaml scripts/run_visual_hindsight_g0_remote.sh)" ]]; then
  echo "Relevant visual-hindsight files are dirty or untracked" >&2
  exit 3
fi

CONFIG_SHA256="$(sha256sum "$CONFIG" | awk '{print $1}')"
[[ "$CONFIG_SHA256" == "$PINNED_CONFIG_SHA256" ]] || { echo "Config digest mismatch" >&2; exit 3; }
CODE_SHA256="$(python - "$ROOT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "src" / "visual_hindsight_g0").glob("*.py")):
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
print(hashlib.sha256(payload.encode("utf-8")).hexdigest())
PY
)"
[[ "$CODE_SHA256" == "$PINNED_CODE_SHA256" ]] || { echo "Visual source digest mismatch" >&2; exit 3; }

python - <<'PY'
import sys
import torch
import transformers
from PIL import __version__ as pillow_version
from transformers import Qwen3VLForConditionalGeneration

assert sys.version_info[:2] == (3, 12), sys.version
assert transformers.__version__ == "5.15.0", transformers.__version__
assert pillow_version == "12.3.0", pillow_version
assert torch.cuda.is_available(), "CUDA is unavailable"
assert Qwen3VLForConditionalGeneration.__name__ == "Qwen3VLForConditionalGeneration"
PY

python -m visual_hindsight_g0.prepare \
  --root "$RUN_ROOT/corpus" --pairs 48 --width 384 --height 288 \
  --prefix-frames 8 --suffix-frames 4 --seed 20260830

python -m visual_hindsight_g0.runner \
  --inputs "$RUN_ROOT/corpus/frozen_inputs.jsonl" \
  --frame-root "$RUN_ROOT/corpus" \
  --output "$RUN_ROOT/qwen3_vl_native_video" \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --presentation-mode native_video \
  --config-sha256 "$CONFIG_SHA256" \
  --code-sha256 "$CODE_SHA256" \
  --git-commit "$PINNED_COMMIT" \
  --max-new-tokens 8

ROOT_ARGS=(--root "$RUN_ROOT/qwen3_vl_native_video")
if [[ "${VISUAL_HINDSIGHT_RUN_MULTI_IMAGE:-0}" == "1" ]]; then
  python -m visual_hindsight_g0.runner \
    --inputs "$RUN_ROOT/corpus/frozen_inputs.jsonl" \
    --frame-root "$RUN_ROOT/corpus" \
    --output "$RUN_ROOT/qwen3_vl_multi_image" \
    --model-id Qwen/Qwen3-VL-8B-Instruct \
    --revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --presentation-mode multi_image \
    --config-sha256 "$CONFIG_SHA256" \
    --code-sha256 "$CODE_SHA256" \
    --git-commit "$PINNED_COMMIT" \
    --max-new-tokens 8
  ROOT_ARGS+=(--root "$RUN_ROOT/qwen3_vl_multi_image")
fi

if [[ "${VISUAL_HINDSIGHT_RUN_GEMMA4:-0}" == "1" ]]; then
  python - <<'PY'
from transformers import Gemma4UnifiedForConditionalGeneration
assert Gemma4UnifiedForConditionalGeneration.__name__ == "Gemma4UnifiedForConditionalGeneration"
PY
  python -m visual_hindsight_g0.runner \
    --inputs "$RUN_ROOT/corpus/frozen_inputs.jsonl" \
    --frame-root "$RUN_ROOT/corpus" \
    --output "$RUN_ROOT/gemma4_multi_image" \
    --model-id google/gemma-4-12B-it \
    --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 \
    --presentation-mode multi_image \
    --config-sha256 "$CONFIG_SHA256" \
    --code-sha256 "$CODE_SHA256" \
    --git-commit "$PINNED_COMMIT" \
    --max-new-tokens 8
  ROOT_ARGS+=(--root "$RUN_ROOT/gemma4_multi_image")
fi

python -m visual_hindsight_g0.verify \
  --config "$CONFIG" "${ROOT_ARGS[@]}" \
  --destination "$RUN_ROOT/GATE_REPORT.json" \
  --expected-config-sha256 "$CONFIG_SHA256" \
  --expected-code-sha256 "$CODE_SHA256" \
  --expected-git-commit "$PINNED_COMMIT"

python - "$RUN_ROOT" "$PINNED_COMMIT" "$CONFIG_SHA256" "$CODE_SHA256" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

root = Path(sys.argv[1])
destination = root / "COMPLETION_MANIFEST.json"
if destination.exists():
    raise SystemExit("completion manifest already exists")
rows = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path == destination:
        continue
    rows.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    })
payload = {
    "kind": "visual_hindsight_g0_v2_completion",
    "git_commit": sys.argv[2],
    "config_sha256": sys.argv[3],
    "code_sha256": sys.argv[4],
    "files": rows,
    "tree_sha256": hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest(),
}
descriptor, temporary = tempfile.mkstemp(prefix=".COMPLETION_MANIFEST.", dir=root, text=True)
with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, destination)
PY

echo "Visual-hindsight G0 v2 completed with an integrity manifest at $RUN_ROOT"
