"""Offline execution receipts and DEV-only input boundaries for the reduced screen.

No model/network calls occur at import or in these helpers. A checksum receipt
attests to saved bytes and arithmetic, not to neural inference from a checkpoint.
This module intentionally provides no confirmation reader or unlock flag.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
from typing import Mapping, Sequence

INPUT_MANIFEST_SHA256 = "2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2"
DEV_INPUT_NAMES = ("learning.json", "development.json")
FORWARD_GENERATION_RECEIPT = {"mode": "forward_logits_only", "generate_called": False}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RESERVED_PART = re.compile(r"(?:^|[_\-.])(confirmation|confirm|locked|reserved|test)(?:$|[_\-.])", re.I)
RESERVED_ROOT_PART = re.compile(r"(?:^|[_\-.])(confirmation|confirm|locked|reserved)(?:$|[_\-.])", re.I)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    # Callers choose fixed, checked paths before reaching this boundary.
    return json.loads(path.read_text(encoding="utf8"))


def _safe_relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("artifact member must be a relative POSIX path")
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(x in {".", ".."} for x in p.parts):
        raise ValueError("unsafe artifact member")
    if str(p) != name:
        raise ValueError("noncanonical artifact member")
    return p


def checked_file(root: Path, name: str, *, forbid_reserved: bool = True) -> Path:
    """Reject traversal, symlink/junction escape, and reserved split members."""
    relative = _safe_relative(name)
    if forbid_reserved and any(RESERVED_PART.search(part) for part in relative.parts):
        raise ValueError("reserved split access is unavailable in the DEV layer")
    root = Path(root).absolute()
    resolved_root = root.resolve(strict=True)
    candidate = root.joinpath(*relative.parts)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("artifact path escapes its root")
    if resolved != resolved_root.joinpath(*relative.parts):
        raise ValueError("symlink/junction artifact members are not allowed")
    if not resolved.is_file():
        raise ValueError("artifact member is not a file")
    return resolved


def checked_source_file(repository: Path, name: str) -> Path:
    """A source receipt cannot turn into an arbitrary or reserved-data reader."""
    relative=_safe_relative(name)
    allowed={"src":{ ".py"},"scripts":{ ".py", ".sh"},"tests":{ ".py"},
             "configs":{ ".json"},"docs":{ ".md"}}
    if relative.parts[0] not in allowed or relative.suffix not in allowed[relative.parts[0]]:
        raise ValueError("source receipt member is outside allowed code/config/protocol paths")
    return checked_file(repository,name,forbid_reserved=False)


def row_binding(row: Mapping[str, object]) -> dict[str, object]:
    """Bind a released row to its base/rotation and prompt display-name cohort.

    The release has no source user-ID field. `source_user` is explicitly the
    display name preceding the prompt's first colon, not a new identity claim.
    """
    rotation = row.get("label_rotation")
    if type(rotation) is not int or rotation not in range(4):
        raise ValueError("rotation must be one of the four integer positions")
    prompt = row.get("prompt")
    if not isinstance(prompt, str) or ":" not in prompt.splitlines()[0]:
        raise ValueError("missing source display-name prefix")
    user = prompt.split(":", 1)[0].strip()
    if not user or "\n" in user:
        raise ValueError("invalid source display-name prefix")
    row_id, base_id = row.get("id"), row.get("base_id")
    if not isinstance(row_id, str) or not isinstance(base_id, str):
        raise ValueError("row and base IDs must be strings")
    if row_id != f"{base_id}-rotation-{rotation}":
        raise ValueError("row ID does not bind its base and rotation")
    return {"id": row_id, "base_id": base_id, "label_rotation": rotation, "source_user": user}


def _validate_grid(rows: Sequence[Mapping[str, object]], bases: int, partition: str) -> list[dict[str, object]]:
    bindings = [row_binding(row) for row in rows]
    if len(rows) != bases * 4 or len({x["id"] for x in bindings}) != len(rows):
        raise ValueError("incorrect or duplicated row grid")
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("partition") != partition:
            raise ValueError("wrong split partition")
        if row.get("old_target") not in set("ABCD") or row.get("new_target") not in set("ABCD"):
            raise ValueError("invalid target")
        if row["old_target"] == row["new_target"]:
            raise ValueError("targets must differ in this fixed expression construction")
        if row.get("immediate_followup") != row.get("delayed_transition_followup"):
            raise ValueError("transition identity failed")
        grouped.setdefault(str(row["base_id"]), []).append(row)
    if len(grouped) != bases:
        raise ValueError("incorrect base grid")
    for variants in grouped.values():
        if {r["label_rotation"] for r in variants} != set(range(4)):
            raise ValueError("missing or duplicated rotation")
        if {r["old_target"] for r in variants} != set("ABCD") or {r["new_target"] for r in variants} != set("ABCD"):
            raise ValueError("targets are not counterbalanced within base")
        if len({row_binding(r)["source_user"] for r in variants}) != 1:
            raise ValueError("rotations disagree on source display name")
    return sorted(bindings, key=lambda r: str(r["id"]))


def load_learning_dev(input_root: Path, *, expected_manifest_sha256: str = INPUT_MANIFEST_SHA256,
                      expected_learning_bases: int = 630, expected_dev_bases: int = 96):
    """Open exactly MANIFEST.json, learning.json, development.json; never glob.

    The manifest may mention reserved files. Those entries are metadata only:
    their targets are neither opened nor hashed. There is no split override.
    """
    root = Path(input_root)
    root_parts = [*root.absolute().parts, *root.resolve(strict=True).parts]
    if any(RESERVED_ROOT_PART.search(part) or part.lower() == "test" for part in root_parts):
        raise ValueError("a reserved split cannot be a DEV input root")
    manifest_path = checked_file(root, "MANIFEST.json")
    if not HEX64.fullmatch(expected_manifest_sha256) or sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("pinned DEV input manifest mismatch")
    manifest = _read_json(manifest_path)
    rows = {}
    hashes = {}
    for name in DEV_INPUT_NAMES:
        path = checked_file(root, name)
        actual = sha256_file(path)
        if manifest.get(name) != actual:
            raise ValueError(f"input checksum mismatch: {name}")
        rows[name] = _read_json(path)
        hashes[name] = actual
    learning, development = rows["learning.json"], rows["development.json"]
    train_bindings = _validate_grid(learning, expected_learning_bases, "learning")
    dev_bindings = _validate_grid(development, expected_dev_bases, "evaluation")
    if {r["base_id"] for r in train_bindings} & {r["base_id"] for r in dev_bindings}:
        raise ValueError("learning/DEV base overlap")
    for field in ("surface_sha256", "prompt"):
        if {r[field] for r in learning} & {r[field] for r in development}:
            raise ValueError(f"learning/DEV {field} overlap")
    receipt = {"input_manifest_sha256": expected_manifest_sha256, "input_sha256": hashes,
               "learning_rows": len(learning), "development_rows": len(development),
               "learning_binding_sha256": canonical_sha256(train_bindings),
               "development_binding_sha256": canonical_sha256(dev_bindings),
               "development_bindings": dev_bindings,
               "source_user_definition": "prompt display name before first colon; no separate user ID supplied",
               "opened_members": ["MANIFEST.json", *DEV_INPUT_NAMES],
               "confirmation_opened": False}
    return learning, development, receipt


def validate_prediction_bindings(source_rows: Sequence[Mapping[str, object]], predictions: Sequence[Mapping[str, object]]) -> None:
    expected = {r["id"]: row_binding(r) for r in source_rows}
    if len(expected) != len(source_rows) or len(predictions) != len(expected):
        raise ValueError("prediction row count mismatch")
    actual = [p.get("id") for p in predictions]
    if len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("prediction IDs do not exactly match expected DEV IDs")
    for p in predictions:
        if any(p.get(k) != v for k, v in expected[p["id"]].items()):
            raise ValueError(f"prediction metadata binding mismatch: {p['id']}")


def tensor_tree_sha256(value: object) -> str:
    """Canonical typed digest, independent of torch archive serialization."""
    import torch
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu().contiguous()
            digest.update(canonical_json(["tensor", str(t.dtype), list(t.shape)]).encode())
            digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, Mapping):
            digest.update(b"mapping[")
            for key in sorted(x, key=lambda k: canonical_json([type(k).__name__, k])):
                visit([type(key).__name__, key]);visit(x[key])
            digest.update(b"]")
        elif isinstance(x, (list, tuple)):
            digest.update(b"sequence[")
            for item in x:visit(item)
            digest.update(b"]")
        elif x is None or type(x) in (str, int, float, bool):
            digest.update(canonical_json([type(x).__name__, x]).encode())
        else:
            raise ValueError(f"unsupported checkpoint value: {type(x).__name__}")
    visit(value)
    return digest.hexdigest()


def effective_adamw_settings(optimizer, frozen_settings: Mapping[str, object]) -> dict:
    """Reject undeclared defaults; account for PyTorch's derived AdamW flag."""
    import torch
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise ValueError("the protocol requires one AdamW parameter group")
    actual = {k:v for k,v in optimizer.param_groups[0].items() if k != "params"}
    expected = dict(frozen_settings)
    # Added to AdamW serialization in newer PyTorch; inherent in AdamW, not a
    # tunable experiment choice. All other new defaults fail closed.
    if "decoupled_weight_decay" in actual and "decoupled_weight_decay" not in expected:
        expected["decoupled_weight_decay"] = True
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("effective AdamW settings differ from frozen settings")
    return json.loads(canonical_json(actual))


def inspect_adapter_checkpoint(path: Path) -> tuple[dict, dict]:
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("adapter checkpoint must contain tensors")
    for name, t in state.items():
        if not isinstance(name, str) or not isinstance(t, torch.Tensor) or t.ndim != 2:
            raise ValueError("invalid adapter schema")
        if not t.is_floating_point() or not bool(torch.isfinite(t).all()) or min(t.shape) < 1:
            raise ValueError("nonfinite or empty adapter tensor")
    return state, {"file_sha256": sha256_file(path), "tensor_sha256": tensor_tree_sha256(state),
                   "parameters": {name: {"shape": list(t.shape), "dtype": str(t.dtype)} for name,t in state.items()},
                   "load_mode": "torch.load(weights_only=True,map_location='cpu')"}


def inspect_optimizer_checkpoint(path: Path, adapter_state: Mapping, parameter_names: Sequence[str], *,
                                 expected_steps: int, optimizer_settings: Mapping[str, object]) -> dict:
    import torch
    if type(expected_steps) is not int or expected_steps < 0:
        raise ValueError("invalid optimizer step count")
    if len(set(parameter_names)) != len(parameter_names) or set(parameter_names) != set(adapter_state):
        raise ValueError("optimizer parameter order does not cover adapter")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"} or len(state["param_groups"]) != 1:
        raise ValueError("expected one explicit AdamW parameter group")
    group = state["param_groups"][0]
    settings = {k:v for k,v in group.items() if k != "params"}
    if canonical_json(settings) != canonical_json(dict(optimizer_settings)):
        raise ValueError("optimizer settings differ from effective frozen configuration")
    ids = group["params"]
    if ids != list(range(len(parameter_names))):
        raise ValueError("optimizer parameter IDs/order changed")
    expected_ids = set(ids) if expected_steps else set()
    if set(state["state"]) != expected_ids:
        raise ValueError("optimizer state is not fresh or misses trained parameters")
    for index, item in state["state"].items():
        required = {"step", "exp_avg", "exp_avg_sq"}
        if settings.get("amsgrad"):
            required.add("max_exp_avg_sq")
        if set(item) != required:
            raise ValueError("unexpected AdamW state fields")
        step = item["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or not bool(torch.isfinite(step).all()) or float(step) != expected_steps:
            raise ValueError("optimizer step tensor does not match executed ledger")
        expected = adapter_state[parameter_names[index]]
        for field in required - {"step"}:
            t = item[field]
            if not isinstance(t, torch.Tensor) or t.shape != expected.shape or t.dtype != expected.dtype or not bool(torch.isfinite(t).all()):
                raise ValueError("invalid AdamW moment shape, dtype or values")
            if field.endswith("sq") and bool((t < 0).any()):
                raise ValueError("negative AdamW second moment")
    return {"file_sha256": sha256_file(path), "tensor_sha256": tensor_tree_sha256(state),
            "state_entries": len(state["state"]), "executed_steps": expected_steps,
            "optimizer_class": "torch.optim.AdamW", "optimizer_settings": json.loads(canonical_json(settings)),
            "parameter_names": list(parameter_names), "load_mode": "torch.load(weights_only=True,map_location='cpu')"}


def make_arm_execution_receipt(root: Path, arm: str, parameter_names: Sequence[str],
                               optimizer_settings: Mapping[str, object], executed_steps: int) -> dict:
    initial, global_receipt = inspect_adapter_checkpoint(checked_file(root, "initial_adapter.pt"))
    start, start_receipt = inspect_adapter_checkpoint(checked_file(root, f"{arm}_initial_adapter.pt"))
    final, final_receipt = inspect_adapter_checkpoint(checked_file(root, f"{arm}_adapter.pt"))
    if start_receipt["tensor_sha256"] != global_receipt["tensor_sha256"]:
        raise ValueError("arm did not record the common initialization")
    if start_receipt["parameters"] != final_receipt["parameters"]:
        raise ValueError("adapter parameter schema changed")
    initial_optimizer = inspect_optimizer_checkpoint(checked_file(root, f"{arm}_initial_optimizer.pt"), start, parameter_names,
                                                     expected_steps=0, optimizer_settings=optimizer_settings)
    final_optimizer = inspect_optimizer_checkpoint(checked_file(root, f"{arm}_optimizer.pt"), final, parameter_names,
                                                   expected_steps=executed_steps, optimizer_settings=optimizer_settings)
    return {"arm": arm, "common_initial_adapter_tensor_sha256": global_receipt["tensor_sha256"],
            "initial_adapter": start_receipt, "final_adapter": final_receipt,
            "initial_optimizer": initial_optimizer, "final_optimizer": final_optimizer,
            "executed_steps": executed_steps,
            "scope": "Saved reset/state/step integrity; not proof of training execution or neural recomputation"}


def capture_execution_provenance(repository: Path, source_paths: Sequence[Path | str], argv: Sequence[str], effective_config: Mapping) -> dict:
    """Record effective CLI/config and a transitive local-Python source closure.

    Git diff contents are hashed rather than copied. Untracked files are listed,
    not opened unless explicitly required as a source dependency. No secrets or
    arbitrary environment variables are copied into the receipt.
    """
    repo = Path(repository).resolve(strict=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])
    status = git("status", "--porcelain=v1", "-z", "--untracked-files=all").decode("utf8").split("\0")
    pending = [Path(p) if Path(p).is_absolute() else repo/Path(p) for p in source_paths]
    files: dict[str, str] = {}; unresolved = set()
    while pending:
        path = pending.pop().resolve(strict=True)
        if not path.is_relative_to(repo):
            raise ValueError("source dependency escapes repository")
        rel = path.relative_to(repo).as_posix()
        if any(part == "analysis" or RESERVED_PART.search(part) for part in path.relative_to(repo).parts[:-1]):
            raise ValueError("source dependencies may not read user analysis or reserved data")
        if rel in files:continue
        files[rel] = sha256_file(checked_source_file(repo,rel))
        if path.suffix != ".py":continue
        tree = ast.parse(path.read_text(encoding="utf8"), filename=rel)
        candidates = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    parent = path.parent
                    for _ in range(node.level - 1):parent = parent.parent
                    target = parent.joinpath(*(node.module or "").split("."))
                    for candidate in [target.with_suffix(".py"), target/"__init__.py"]:
                        if candidate.is_file():pending.append(candidate)
                    if not node.module:
                        for alias in node.names:
                            candidate = parent/(alias.name+".py")
                            if candidate.is_file():pending.append(candidate)
                elif node.module:candidates.append(node.module)
        for module in candidates:
            found = False
            for base in (repo/"src", repo):
                target = base.joinpath(*module.split("."))
                for candidate in [target.with_suffix(".py"), target/"__init__.py"]:
                    if candidate.is_file():pending.append(candidate);found = True
            if not found:unresolved.add(module.split(".")[0])
    versions = {}
    for distribution in ("torch", "transformers", "tokenizers", "numpy", "safetensors", "huggingface-hub", "scipy", "scikit-learn"):
        try:versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:versions[distribution] = None
    return {"schema_version": 1, "argv": list(argv), "effective_config": json.loads(canonical_json(effective_config)),
            "effective_config_sha256": canonical_sha256(effective_config),
            "git": {"head": git("rev-parse", "HEAD").decode().strip(),
                    "branch": git("branch", "--show-current").decode().strip(),
                    "status_porcelain_v1_z_entries": [x for x in status if x],
                    "index_entries_sha256": hashlib.sha256(git("ls-files", "--stage", "-z")).hexdigest(),
                    "unstaged_binary_diff_sha256": hashlib.sha256(git("diff", "--binary", "--no-ext-diff")).hexdigest(),
                    "staged_binary_diff_sha256": hashlib.sha256(git("diff", "--cached", "--binary", "--no-ext-diff")).hexdigest()},
            "source_sha256": dict(sorted(files.items())), "external_import_roots": sorted(unresolved),
            "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                            "packages": versions, "variables": {k:os.environ.get(k) for k in
                            ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM")}},
            "generation": dict(FORWARD_GENERATION_RECEIPT),
            "scope": "Git index/diffs/status plus effective local source closure; untracked user data not opened"}


def capture_model_provenance(snapshot_path: Path, model_id: str, resolved_revision: str, *, tokenizer=None, model=None) -> dict:
    """Hash an already present local immutable HF snapshot; never resolve online."""
    snapshot = Path(snapshot_path).absolute()
    if not HEX40.fullmatch(resolved_revision) or snapshot.name != resolved_revision or snapshot.parent.name != "snapshots":
        raise ValueError("require a local snapshots/<immutable 40-character commit> directory")
    model_cache = snapshot.parent.parent.resolve(strict=True)
    expected_cache_name = "models--" + model_id.replace("/", "--")
    if model_cache.name != expected_cache_name:
        raise ValueError("local snapshot does not bind the declared model ID")
    entries = {}
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():continue
        target = path.resolve(strict=True)
        if not target.is_relative_to(model_cache):
            raise ValueError("model snapshot link escapes its model cache")
        rel = path.relative_to(snapshot).as_posix()
        entries[rel] = {"sha256": sha256_file(target), "bytes": target.stat().st_size}
    if "config.json" not in entries or "tokenizer_config.json" not in entries:
        raise ValueError("model/tokenizer configuration receipt is incomplete")
    weight_files = [name for name in entries if name.endswith((".safetensors", ".bin"))]
    if not weight_files:
        raise ValueError("model snapshot has no local weights")
    for name in entries:
        if name.endswith(".index.json"):
            index = _read_json(snapshot/name)
            if "weight_map" in index and not set(index["weight_map"].values()) <= set(weight_files):
                raise ValueError("weight index references absent shards")
    result = {"model_id": model_id, "resolved_revision": resolved_revision, "tokenizer_revision": resolved_revision,
              "snapshot_directory_name": snapshot.name, "files": entries, "weights": sorted(weight_files),
              "snapshot_tree_sha256": canonical_sha256(entries), "local_files_only": True}
    if model is not None:
        commit = getattr(model.config, "_commit_hash", None)
        if commit is not None and commit != resolved_revision:raise ValueError("effective model revision mismatch")
        result["effective_model_config"] = model.config.to_dict()
        result["model_class"] = type(model).__module__+"."+type(model).__name__
        result["parameter_dtype"] = str(next(model.parameters()).dtype)
    if tokenizer is not None:
        commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        if commit is not None and commit != resolved_revision:raise ValueError("effective tokenizer revision mismatch")
        result["effective_tokenizer"] = {"class": type(tokenizer).__module__+"."+type(tokenizer).__name__,
            "padding_side": tokenizer.padding_side, "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.bos_token_id,
            "special_tokens_map": tokenizer.special_tokens_map, "chat_template": tokenizer.chat_template,
            "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    return result


def capture_effective_generation_config(generation_config, overrides: Mapping[str, object], expected: Mapping[str, object]) -> dict:
    """Validate what generate will actually receive, including inherited defaults.

    Callers must pass the returned effective settings to generate. This is not
    used to suggest that generation settings affect forward-only scoring.
    """
    from copy import deepcopy
    effective = deepcopy(generation_config)
    unused = effective.update(**dict(overrides))
    if unused:
        raise ValueError(f"unrecognized generation overrides: {sorted(unused)}")
    effective.validate()
    settings = effective.to_dict()
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"effective generation setting mismatch: {key}")
    if settings.get("do_sample") is not False and expected.get("do_sample") is False:
        raise ValueError("requested greedy generation would sample")
    return {"mode": "generate", "generate_called": True, "overrides": dict(overrides),
            "effective_config": settings, "effective_config_sha256": canonical_sha256(settings)}


def seal_artifacts(root: Path) -> dict[str, str]:
    root = Path(root)
    if (root/"MANIFEST.json").exists():raise FileExistsError("evidence manifest already exists")
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():continue
        name = path.relative_to(root).as_posix()
        manifest[name] = sha256_file(checked_file(root, name))
    with (root/"MANIFEST.json").open("x", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
    return manifest


def verify_manifest(root: Path) -> dict[str, str]:
    manifest = _read_json(checked_file(root, "MANIFEST.json"))
    if not isinstance(manifest, dict) or "MANIFEST.json" in manifest:
        raise ValueError("invalid manifest schema")
    root = Path(root)
    actual_names = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root/"MANIFEST.json"}
    if actual_names != set(manifest):raise ValueError("manifest omits files or names absent files")
    for name, expected in manifest.items():
        if not isinstance(expected, str) or not HEX64.fullmatch(expected) or sha256_file(checked_file(root, name)) != expected:
            raise ValueError(f"manifest checksum mismatch: {name}")
    return manifest


def verify_saved_logits(root: Path, index: Sequence[Mapping], predictions_by_arm: Mapping[str, Sequence[Mapping]],
                        *, answer_token_ids: Sequence[int], max_tokens: int,
                        rendered_texts: Mapping[str, Mapping[str, str]] | None = None,
                        vocab_size: int | None = None) -> dict:
    """Recompute probability arithmetic from saved float32 full-vocabulary logits.

    This deliberately does not load a neural model or claim the logits were
    produced by the stored adapter. Such a claim requires separate inference.
    """
    import numpy as np
    if len(answer_token_ids) != 4 or len(set(answer_token_ids)) != 4 or any(type(x) is not int or x < 0 for x in answer_token_ids):
        raise ValueError("four distinct native answer token IDs required")
    expected = {}
    for arm, predictions in predictions_by_arm.items():
        by_id = {p.get("job_id", p["id"]): p for p in predictions}
        if len(by_id) != len(predictions):raise ValueError("duplicate logit prediction ID")
        expected[arm] = by_id
    seen = {arm:set() for arm in expected};paths=set();max_error=0.;token_batches=0
    for batch in index:
        arm = batch["arm"]
        if arm not in expected:raise ValueError("unknown logit arm")
        ids = batch["row_ids"]
        if not isinstance(ids,list) or not ids or len(set(ids)) != len(ids) or set(ids) & seen[arm] or not set(ids) <= set(expected[arm]):
            raise ValueError("logit row IDs duplicated, missing or foreign")
        if batch["path"] in paths:raise ValueError("logit file reused across batches")
        paths.add(batch["path"]);seen[arm].update(ids)
        if batch.get("answer_token_ids") != list(answer_token_ids):raise ValueError("logit token mapping mismatch")
        logits = np.load(checked_file(root,batch["path"]),allow_pickle=False)
        if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[0] != len(ids) or logits.shape[1] <= max(answer_token_ids) or not np.isfinite(logits).all():
            raise ValueError("invalid full vocabulary logit array")
        if vocab_size is not None and logits.shape[1] != vocab_size:
            raise ValueError("logit vocabulary dimension differs from model configuration")
        values=logits.astype(np.float64);normalizer=np.logaddexp.reduce(values,axis=1)
        choices=values[:,answer_token_ids];full=choices-normalizer[:,None]
        conditional=choices-np.logaddexp.reduce(choices,axis=1)[:,None]
        mass=np.exp(full).sum(axis=1)
        for j,id in enumerate(ids):
            p=expected[arm][id]
            checks={"choice_logits":choices[j],"log_normalizer":normalizer[j],
                    "full_vocab_choice_log_probabilities":full[j],
                    "normalized_choice_log_probabilities":conditional[j],"full_vocabulary_choice_mass":mass[j]}
            for field,value in checks.items():
                actual=np.asarray(p[field],dtype=float)
                if actual.shape != np.asarray(value).shape or not np.isfinite(actual).all() or not np.allclose(actual,value,rtol=1e-6,atol=5e-5):
                    raise ValueError(f"saved logit arithmetic mismatch: {arm}/{id}/{field}")
                max_error=max(max_error,float(np.max(np.abs(actual-value))))
        if rendered_texts is not None:
            target=[hashlib.sha256(rendered_texts[arm][id].encode("utf8")).hexdigest() for id in ids]
            if batch.get("rendered_text_sha256") != target:raise ValueError("rendered prompt binding differs")
        if "input_ids_path" in batch or "attention_mask_path" in batch:
            inputs=np.load(checked_file(root,batch["input_ids_path"]),allow_pickle=False)
            mask=np.load(checked_file(root,batch["attention_mask_path"]),allow_pickle=False)
            if inputs.dtype.kind not in "iu" or inputs.ndim != 2 or inputs.shape != mask.shape or inputs.shape[0] != len(ids) or inputs.shape[1]>max_tokens:
                raise ValueError("tokenization shape or context ceiling mismatch")
            if mask.dtype.kind not in "biu" or not np.isin(mask,[0,1]).all() or not np.all(mask.sum(axis=1)>0) or np.any(inputs<0) or np.any(inputs>=logits.shape[1]) or np.any(np.diff(mask.astype(int),axis=1)<0):
                raise ValueError("invalid saved tokenization")
            token_batches+=1
        elif rendered_texts is not None:
            raise ValueError("saved tokenization is required with rendered prompt audit")
    if any(seen[arm] != set(rows) for arm,rows in expected.items()):raise ValueError("incomplete logit grid")
    return {"batches":len(index),"rows":sum(len(x) for x in seen.values()),"maximum_absolute_arithmetic_error":max_error,
            "tokenization_batches":token_batches,"offline_arithmetic_recomputed":True,"neural_inference_recomputed":False}


def declared_vocabulary_size(model_receipt: Mapping) -> int:
    config = model_receipt["effective_model_config"]
    value = config.get("vocab_size", config.get("text_config", {}).get("vocab_size"))
    if type(value) is not int or value < 4:raise ValueError("effective model vocabulary size missing")
    return value


def verify_tokenization_backend(root: Path, index: Sequence[Mapping], source_texts: Mapping[str, Mapping[str, str]],
                                model_receipt: Mapping, answer_token_ids: Sequence[int], max_tokens: int) -> dict:
    """Bind source prompts -> captured chat template -> native input IDs offline.

    Only the saved tokenizer JSON and a sandboxed template renderer are loaded;
    no model, weight, network, or reserved split is accessed. Model-family
    tokenizer wrappers are not trusted to describe the actual saved IDs.
    """
    import numpy as np
    from tokenizers import Tokenizer
    from transformers.utils.chat_template_utils import render_jinja_template
    tokenizer_receipt=model_receipt["effective_tokenizer"]
    backend_path=checked_file(root,"tokenizer_backend.json")
    if sha256_file(backend_path)!=tokenizer_receipt["backend_sha256"]:
        raise ValueError("saved tokenizer backend does not match loaded-tokenizer receipt")
    backend=Tokenizer.from_str(backend_path.read_text(encoding="utf8"))
    backend.no_padding();backend.no_truncation()
    if [backend.encode(letter,add_special_tokens=False).ids for letter in "ABCD"] != [[x] for x in answer_token_ids]:
        raise ValueError("backend native answer token IDs differ")
    if tokenizer_receipt["padding_side"]!="left":raise ValueError("protocol requires left padding")
    pad=tokenizer_receipt["pad_token_id"]
    if type(pad) is not int or not 0<=pad<declared_vocabulary_size(model_receipt):raise ValueError("invalid padding token ID")
    template=tokenizer_receipt["chat_template"]
    if not isinstance(template,str) or not template:raise ValueError("explicit text chat template required")
    rendered_saved=_read_json(checked_file(root,"rendered_texts.json"))
    seen={};batches=0;rows=0
    for batch in index:
        arm=batch["arm"];ids=batch["row_ids"]
        if arm not in source_texts or any(id not in source_texts[arm] for id in ids):raise ValueError("tokenization source ID is foreign")
        seen.setdefault(arm,set())
        if set(ids)&seen[arm] or len(set(ids))!=len(ids):raise ValueError("duplicate tokenization row")
        seen[arm].update(ids)
        texts=[source_texts[arm][id] for id in ids]
        rendered,_=render_jinja_template([[{"role":"user","content":text}] for text in texts],chat_template=template,
                                         add_generation_prompt=True,enable_thinking=False,
                                         **tokenizer_receipt["special_tokens_map"])
        if [rendered_saved.get(arm,{}).get(id) for id in ids]!=rendered:raise ValueError("rendered prompt differs from source/template")
        hashes=[hashlib.sha256(text.encode("utf8")).hexdigest() for text in rendered]
        if batch.get("rendered_text_sha256")!=hashes:raise ValueError("rendered prompt hash differs")
        encoded=[backend.encode(text,add_special_tokens=False).ids for text in rendered]
        if not encoded or any(not x or len(x)>max_tokens for x in encoded):raise ValueError("tokenizer output exceeds context ceiling")
        width=max(map(len,encoded));expected_ids=np.asarray([[pad]*(width-len(x))+x for x in encoded],dtype=np.int64)
        expected_mask=np.asarray([[0]*(width-len(x))+[1]*len(x) for x in encoded],dtype=np.int64)
        for key,expected in [("input_ids_path",expected_ids),("attention_mask_path",expected_mask)]:
            actual=np.load(checked_file(root,batch[key]),allow_pickle=False)
            if actual.dtype.kind not in "biu" or not np.array_equal(actual,expected):raise ValueError("saved input token/mask array differs from rendered text encoding")
        raw=np.load(checked_file(root,batch["path"]),allow_pickle=False,mmap_mode="r")
        if raw.ndim!=2 or raw.shape!=(len(ids),declared_vocabulary_size(model_receipt)):
            raise ValueError("logit vocabulary dimension differs from model configuration")
        batches+=1;rows+=len(ids)
    if set(rendered_saved)!=set(seen) or any(set(rendered_saved[arm])!=ids for arm,ids in seen.items()):
        raise ValueError("rendered text grid differs from actual saved evaluation batches")
    return {"tokenization_recomputed":True,"source_template_rendering_recomputed":True,
            "batches":batches,"rows":rows,"backend_sha256":tokenizer_receipt["backend_sha256"],
            "model_or_weights_loaded":False,"neural_inference_recomputed":False}
