"""Fail-closed runtime preflight for the semantic-ancestry RAG G0 gate.

This check intentionally imports model classes but never downloads model weights.
It creates a host-specific attestation which a later serving-family run must bind
into its immutable artifact root.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from under_extinction.io import canonical_json, sha256_file, write_json

from .gate import Conditions, Thresholds


KIND = "semantic_ancestry_rag_runtime_preflight"
QWEN_ID = "Qwen/Qwen3.5-9B"
QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MISTRAL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"


def load_contract(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "semantic_ancestry_rag_g0":
        raise ValueError("not a semantic-ancestry RAG G0 config")
    models = value.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("G0 config needs a model contract")
    expected = {
        "ancestor": (QWEN_ID, QWEN_REVISION, "Qwen3_5ForCausalLM"),
        "rewriter": (MISTRAL_ID, MISTRAL_REVISION, "AutoModelForCausalLM"),
        "qwen3_5": (QWEN_ID, QWEN_REVISION, "Qwen3_5ForCausalLM"),
        "mistral": (MISTRAL_ID, MISTRAL_REVISION, "AutoModelForCausalLM"),
    }
    actual = {"ancestor": models.get("ancestor"), "rewriter": models.get("rewriter")}
    serving = models.get("serving_families")
    if not isinstance(serving, Mapping):
        raise ValueError("G0 config needs two serving-model families")
    actual.update({"qwen3_5": serving.get("qwen3_5"), "mistral": serving.get("mistral")})
    for name, (model_id, revision, loader) in expected.items():
        contract = actual[name]
        if not isinstance(contract, Mapping):
            raise ValueError(f"missing model contract for {name}")
        if (contract.get("id"), contract.get("revision"), contract.get("loader_class")) != (model_id, revision, loader):
            raise ValueError(f"unexpected frozen model contract for {name}")
    for name in ("ancestor", "qwen3_5"):
        contract = actual[name]
        if contract.get("text_only") is not True or contract.get("enable_thinking") is not False:
            raise ValueError(f"Qwen contract {name} must be text-only with thinking disabled")
    if tuple(value.get("conditions", ())) != Conditions.ALL:
        raise ValueError("G0 config conditions differ from the executable gate")
    if int(value.get("question_count", 0)) < 30 or int(value.get("completions_per_cell", 0)) < 1:
        raise ValueError("G0 config has an invalid experiment size")
    Thresholds(**dict(value.get("thresholds") or {}))
    return value


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime dependency is missing: {distribution}") from exc


def inspect_runtime() -> dict[str, Any]:
    """Validate hardware and native classes without touching the Hub cache."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("semantic-ancestry G0 requires a CUDA host")
    try:
        from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("Transformers lacks a required native G0 model loader") from exc
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "qwen_text_loader": Qwen3_5ForCausalLM.__name__,
        "mistral_loader": AutoModelForCausalLM.__name__,
        "cuda_device_index": device,
        "cuda_device_name": properties.name,
        "cuda_total_memory_bytes": int(properties.total_memory),
    }


def validate_bound_preflight(*, config: str | Path, runtime_preflight: str | Path) -> dict[str, Any]:
    """Return the frozen contract only when a preflight is bound to it exactly."""

    contract = load_contract(config)
    source = Path(runtime_preflight)
    preflight = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(preflight, Mapping) or preflight.get("kind") != KIND:
        raise ValueError("not a semantic-ancestry runtime preflight")
    if preflight.get("config_sha256") != sha256_file(config):
        raise ValueError("runtime preflight is not bound to the frozen G0 config")
    if preflight.get("model_contract") != contract["models"]:
        raise ValueError("runtime preflight model contract differs from G0 config")
    return contract


def preflight(*, config: str | Path, destination: str | Path) -> dict[str, Any]:
    target = Path(destination)
    if target.exists():
        raise FileExistsError("refusing to overwrite a semantic-ancestry runtime preflight")
    contract = load_contract(config)
    result = {
        "kind": KIND,
        "config_path": str(Path(config).resolve()),
        "config_sha256": sha256_file(config),
        "model_contract": contract["models"],
        "runtime": inspect_runtime(),
    }
    write_json(target, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a semantic-ancestry RAG G0 host without downloading weights")
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(preflight(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
