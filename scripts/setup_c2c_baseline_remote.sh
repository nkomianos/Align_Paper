#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/Align_Paper
ENV_ROOT=/home/ubuntu/Align_Paper/.venv-c2c-baseline-v1
EVIDENCE=/home/ubuntu/c2c_baseline_setup_20260904_v1
UV=/home/ubuntu/Align_Paper/.venv-bootstrap-vision/bin/uv
test ! -e "$ENV_ROOT"
test ! -e "$EVIDENCE"
mkdir "$EVIDENCE"
cp "$0" "$EVIDENCE/setup.sh"
# CUDA torch 2.7.1 is the already-tested ARM64 wheel, a declared deviation from
# upstream's torch 2.6.0. Keep the cache API-sensitive Transformers version exact.
"$UV" venv --python 3.12 "$ENV_ROOT"
"$UV" pip install --python "$ENV_ROOT/bin/python" --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1'
"$UV" pip install --python "$ENV_ROOT/bin/python" 'transformers==4.52.4' 'huggingface-hub==0.36.0' 'tokenizers==0.21.4' 'accelerate==1.10.1' 'numpy==2.4.2' 'Jinja2==3.1.6'
"$ENV_ROOT/bin/python" - <<'PY'
import json, sys, torch, transformers
from transformers import Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache
assert transformers.__version__ == '4.52.4'
assert torch.__version__ == '2.7.1+cu128'
assert hasattr(DynamicCache(), 'key_cache')
# No model or CUDA tensor allocated by this dependency check.
print(json.dumps({'python': sys.version, 'torch': torch.__version__,
                  'transformers': transformers.__version__, 'cache_api_pass': True,
                  'scope': 'DEPENDENCY_CHECK_ONLY', 'upstream_torch_match': False}))
PY
"$UV" pip freeze --python "$ENV_ROOT/bin/python" > "$EVIDENCE/requirements.txt"
"$UV" pip check --python "$ENV_ROOT/bin/python"
touch "$EVIDENCE/DEPENDENCIES_READY"
