"""Runtime and role preflight for the corrected semantic-ancestry G0b gate."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from under_extinction.io import canonical_json, sha256_file, write_json

from .g0b import G0BCell, validate_role_plan
from .gate import Conditions, Thresholds


KIND = "semantic_ancestry_rag_g0b_runtime_preflight"


def load_contract(config_path: str | Path) -> dict[str, Any]:
    """Load a role-complete G0b config; reject accidental G0 reuse."""

    path = Path(config_path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("kind") != "semantic_ancestry_rag_g0b":
        raise ValueError("not a semantic-ancestry RAG G0b config")
    if tuple(value.get("conditions", ())) != Conditions.ALL:
        raise ValueError("G0b config conditions differ from the executable gate")
    if int(value.get("question_count", 0)) < 30 or int(value.get("completions_per_cell", 0)) < 1:
        raise ValueError("G0b config has an invalid experiment size")
    Thresholds(**dict(value.get("thresholds") or {}))
    models = value.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("G0b config needs model contracts")
    serving = models.get("serving_families")
    external = models.get("external_models")
    if not isinstance(serving, Mapping) or not isinstance(external, Mapping):
        raise ValueError("G0b needs serving and external model contracts")
    if len(serving) < 2 or len(external) < 2:
        raise ValueError("G0b needs at least two serving and two external models")
    for namespace, entries in (("serving", serving), ("external", external)):
        for name, spec in entries.items():
            if not isinstance(name, str) or not isinstance(spec, Mapping):
                raise ValueError(f"malformed {namespace} model specification")
            if not all(isinstance(spec.get(key), str) and spec[key] for key in ("id", "revision", "loader_class")):
                raise ValueError(f"{namespace} model {name} is not pinned")
            if spec.get("trust_remote_code") is True:
                raise ValueError("G0b forbids trust_remote_code")
    qwen = serving.get("qwen3_5")
    if not isinstance(qwen, Mapping) or qwen.get("loader_class") != "Qwen3_5ForCausalLM":
        raise ValueError("G0b must preserve the native Qwen text-only loader")
    if qwen.get("text_only") is not True or qwen.get("enable_thinking") is not False:
        raise ValueError("G0b Qwen requires text-only loading with thinking disabled")
    cells_raw = value.get("role_cells")
    if not isinstance(cells_raw, list):
        raise ValueError("G0b config needs role_cells")
    cells = tuple(G0BCell(**row) for row in cells_raw if isinstance(row, Mapping))
    if len(cells) != len(cells_raw):
        raise ValueError("G0b role cells must be mappings")
    pairs = tuple((str(cell["rewriter_model"]), str(cell["shadow_answer_model"])) for cell in cells_raw)
    validate_role_plan(cells, serving_models=serving.keys(), external_pairs=pairs)
    return value


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(f"required runtime dependency is missing: {distribution}") from exc


def inspect_runtime() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("semantic-ancestry G0b requires a CUDA host")
    try:
        from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM
    except ImportError as exc:
        raise RuntimeError("Transformers lacks a required native G0b model loader") from exc
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    return {
        "torch_version": _package_version("torch"),
        "transformers_version": _package_version("transformers"),
        "generic_loader": AutoModelForCausalLM.__name__,
        "qwen_text_loader": Qwen3_5ForCausalLM.__name__,
        "cuda_device_index": device,
        "cuda_device_name": properties.name,
        "cuda_total_memory_bytes": int(properties.total_memory),
    }


def validate_bound_preflight(*, config: str | Path, runtime_preflight: str | Path) -> dict[str, Any]:
    import json

    contract = load_contract(config)
    preflight = json.loads(Path(runtime_preflight).read_text(encoding="utf-8"))
    if not isinstance(preflight, Mapping) or preflight.get("kind") != KIND:
        raise ValueError("not a semantic-ancestry G0b runtime preflight")
    if preflight.get("config_sha256") != sha256_file(config):
        raise ValueError("runtime preflight is not bound to the frozen G0b config")
    if preflight.get("model_contract") != contract["models"]:
        raise ValueError("runtime preflight model contract differs from G0b config")
    return contract


def preflight(*, config: str | Path, destination: str | Path) -> dict[str, Any]:
    target = Path(destination)
    if target.exists():
        raise FileExistsError("refusing to overwrite a semantic-ancestry G0b runtime preflight")
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
    parser = argparse.ArgumentParser(description="Validate a semantic-ancestry RAG G0b host without downloading weights")
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    print(canonical_json(preflight(**vars(args))))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
