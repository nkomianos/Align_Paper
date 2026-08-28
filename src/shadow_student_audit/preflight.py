"""Public-artifact preflight for SENTRY; never opens the sealed answer key."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any, Mapping

import yaml

from under_extinction.io import canonical_json, write_json


KIND = "sentry_shadow_student_g0"
EXPECTED_MODEL = "Qwen/Qwen3.5-9B"


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_public_config(path: str | Path) -> dict[str, Any]:
    """Load only non-secret G0 configuration and reject test-key leakage."""

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("SENTRY config must be a mapping")
    config = dict(value)
    if config.get("kind") != KIND or config.get("schema_version") != "1.0":
        raise ValueError("unexpected SENTRY contract")
    model, training, sources, sealed = (
        config.get("model"), config.get("training"), config.get("public_sources"), config.get("sealed_test")
    )
    if not all(isinstance(item, Mapping) for item in (model, training, sealed)) or not isinstance(sources, list):
        raise ValueError("missing SENTRY sections")
    if model.get("id") != EXPECTED_MODEL or not isinstance(model.get("revision"), str) or len(model["revision"]) != 40:
        raise ValueError("SENTRY requires an immutable Qwen3.5-9B revision")
    if sealed.get("answer_key_path") is not None:
        raise ValueError("public preflight must not open or name the sealed answer key")
    if int(training.get("full_lora_rank", 0)) <= int(training.get("shadow_lora_rank", 0)):
        raise ValueError("full LoRA rank must exceed shadow rank")
    if int(training.get("shadow_token_budget", 0)) * 8 != int(training.get("full_token_budget", 0)):
        raise ValueError("shadow budget must be exactly one eighth of full budget")
    if len(training.get("full_seeds", [])) != 2 or len(training.get("shadow_seeds", [])) != 4:
        raise ValueError("SENTRY requires two full and four shadow seeds")
    required_source_keys = {"id", "revision", "url", "purpose"}
    if len(sources) < 3 or any(not isinstance(source, Mapping) or set(source) != required_source_keys for source in sources):
        raise ValueError("every public source needs id, revision, url, and purpose")
    if len({source["id"] for source in sources}) != len(sources):
        raise ValueError("public source IDs must be unique")
    return config


def public_preflight(config_path: str | Path, destination: str | Path, *, require_cuda: bool = True) -> dict[str, Any]:
    """Attest the public contract without mutating artifacts or seeing test data."""

    output = Path(destination)
    if output.exists():
        raise FileExistsError("refusing to overwrite SENTRY public preflight")
    config = load_public_config(config_path)
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - GPU runner environment only
        raise RuntimeError("SENTRY preflight requires torch") from exc
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("SENTRY G0 requires a CUDA device")
    result = {
        "kind": "sentry_g0_public_preflight",
        "config_sha256": _sha(config),
        "model": dict(config["model"]),
        "public_sources": list(config["public_sources"]),
        "training": dict(config["training"]),
        "sealed_answer_key_opened": False,
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    write_json(output, json.loads(canonical_json(result)))
    return result
