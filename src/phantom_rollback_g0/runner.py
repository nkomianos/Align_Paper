"""Durable, non-overwriting two-family trajectory collector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
from typing import Any, Callable, Mapping, Protocol, Sequence

import yaml

from .corpus import ARM_IDS, SEEDS, CorpusBundle, TaskCase, load_sealed_bytes
from .environment import PhantomEnvironment, rollback_semantics_sha256
from .io import (
    atomic_bytes,
    atomic_json,
    canonical_json,
    code_inventory,
    jsonl_bytes,
    sha256_bytes,
    sha256_file,
)
from .protocol import (
    pad_prompt_block,
    parse_action,
    render_comprehension_prompt,
    render_followup_prompt,
    render_initial_prompt,
    schema_for,
)


FAMILIES = ("qwen3_5", "gemma4")
FROZEN_STATUS = "frozen_cpu_audited_test_locked_awaiting_explicit_gpu_authorization"
QWEN_MODEL = "Qwen/Qwen3.5-9B"
GEMMA_MODEL = "google/gemma-4-12B-it"


class TextRuntime(Protocol):
    def count_prompt_tokens(self, prompt: str) -> int: ...
    def generate(self, prompt: str, *, seed: int, generation: Mapping[str, Any]) -> str: ...
    def provenance(self) -> Mapping[str, Any]: ...


RuntimeFactory = Callable[[str, str], TextRuntime]


def deterministic_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _chat_text(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


class TransformersRuntime:
    def __init__(self, model_id: str, revision: str) -> None:
        import torch
        import transformers

        self.model_id = model_id
        self.revision = revision
        self._torch = torch
        self._transformers_version = transformers.__version__
        if model_id == QWEN_MODEL:
            from transformers import AutoTokenizer, Qwen3_5ForCausalLM

            self.processor = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
            self.model = Qwen3_5ForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
                use_kernels=False,
            ).eval()
            self.kind = "qwen"
        elif model_id == GEMMA_MODEL:
            from transformers import AutoProcessor, Gemma4UnifiedForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
            self.model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                model_id,
                revision=revision,
                dtype=torch.bfloat16,
                device_map={"": torch.cuda.current_device()},
                low_cpu_mem_usage=True,
            ).eval()
            self.kind = "gemma"
        else:
            raise ValueError("model is not frozen for Phantom Rollback G0")

    def _inputs(self, prompt: str) -> Any:
        if self.kind == "qwen":
            text = _chat_text(self.processor, prompt)
            return self.processor(text, add_special_tokens=False, return_tensors="pt").to(self.model.device)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

    def count_prompt_tokens(self, prompt: str) -> int:
        if self.kind == "qwen":
            text = _chat_text(self.processor, prompt)
            ids = self.processor(text, add_special_tokens=False)["input_ids"]
        else:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            encoded = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                enable_thinking=False,
            )
            ids = encoded["input_ids"]
        if hasattr(ids, "shape"):
            return int(ids.shape[-1])
        if ids and isinstance(ids[0], list):
            return len(ids[0])
        return len(ids)

    def generate(self, prompt: str, *, seed: int, generation: Mapping[str, Any]) -> str:
        torch = self._torch
        inputs = self._inputs(prompt)
        input_length = int(inputs["input_ids"].shape[-1])
        do_sample = bool(generation["do_sample"])
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(generation["max_new_tokens"]),
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs.update(temperature=float(generation["temperature"]), top_p=float(generation["top_p"]))
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with torch.inference_mode():
                output = self.model.generate(**inputs, **kwargs)
        generated = output[0, input_length:]
        return self.processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def provenance(self) -> Mapping[str, Any]:
        torch = self._torch
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        commit = getattr(getattr(self.model, "config", None), "_commit_hash", None)
        return {
            "torch_version": torch.__version__,
            "transformers_version": self._transformers_version,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
            "resolved_model_commit": commit,
        }


def _load_config(payload: bytes) -> dict[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("config is not valid UTF-8 YAML") from exc
    if not isinstance(value, dict) or value.get("kind") != "phantom_rollback_g0":
        raise ValueError("not a Phantom Rollback G0 config")
    if set(value.get("models", {})) != set(FAMILIES):
        raise ValueError("config must contain exactly the two frozen families")
    expected = {
        "qwen3_5": (QWEN_MODEL, "c202236235762e1c871ad0ccb60c8ee5ba337b9a"),
        "gemma4": (GEMMA_MODEL, "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    }
    for family, pair in expected.items():
        model = value["models"][family]
        if (model.get("id"), model.get("revision")) != pair:
            raise ValueError(f"model pin changed for {family}")
    if tuple(value.get("design", {}).get("arms", ())) != ARM_IDS:
        raise ValueError("arm order changed")
    if tuple(value.get("design", {}).get("seeds", ())) != SEEDS:
        raise ValueError("seed set changed")
    design = value["design"]
    if (
        int(design.get("hazardous_tasks", 0)) != 96
        or int(design.get("authorized_safe_twins", 0)) != 32
        or int(design.get("schema_paraphrases", 0)) != 4
        or design.get("test_locked") is not True
        or design.get("no_test_tuning") is not True
    ):
        raise ValueError("corpus size or TEST-lock policy changed")
    generation = value.get("generation")
    if not isinstance(generation, dict) or int(generation.get("max_actions", 0)) != 3:
        raise ValueError("generation settings are incomplete")
    if generation.get("token_padding") != "tokenizer_audited_exact":
        raise ValueError("exact tokenizer-audited padding is mandatory")
    return value


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise ValueError("cannot attest git commit")
    return commit


def _git_dirty(repo: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _block_schedule(bundle: CorpusBundle) -> list[tuple[TaskCase, int, int, str]]:
    rows: list[tuple[TaskCase, int, int, str]] = []
    for task in bundle.tasks:
        for seed_index, seed in enumerate(bundle.seeds):
            schema_id = schema_for(task, seed_index)
            rows.append((task, seed_index, seed, schema_id))
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"schedule|{row[0].task_id}|{row[2]}".encode()
        ).digest(),
    )


def _execute_block(
    runtime: TextRuntime,
    task: TaskCase,
    *,
    schema_id: str,
    seed: int,
    generation: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run the four arms in lockstep and length-match every decision turn."""

    envs = {arm: PhantomEnvironment(task, arm) for arm in ARM_IDS}
    histories: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_IDS}
    records: dict[str, dict[str, Any]] = {
        arm: {
            "completions": [],
            "decision_prompts": [],
            "decision_unpadded_tokens": [],
            "decision_padded_tokens": [],
        }
        for arm in ARM_IDS
    }
    initial = {
        arm: render_initial_prompt(task, arm=arm, schema_id=schema_id) for arm in ARM_IDS
    }
    for turn in range(int(generation["max_actions"])):
        active = [arm for arm in ARM_IDS if not envs[arm].state.terminal]
        if not active:
            break
        unpadded = {
            arm: initial[arm] if turn == 0 else render_followup_prompt(initial[arm], histories[arm])
            for arm in active
        }
        padded = pad_prompt_block(
            unpadded,
            count_tokens=runtime.count_prompt_tokens,
            margin_tokens=int(generation["padding_margin_tokens"]),
        )
        # Arm order is randomized deterministically within a turn, while the
        # paired design and seed remain identical.
        arm_order = sorted(
            active,
            key=lambda arm: hashlib.sha256(
                f"arm-order|{task.task_id}|{seed}|{turn}|{arm}".encode()
            ).digest(),
        )
        for arm in arm_order:
            prompt, unpadded_count, padded_count = padded[arm]
            completion = runtime.generate(
                prompt,
                # Common random numbers across arms preserve the paired seed;
                # each call forks/restores the device RNG inside the runtime.
                seed=deterministic_seed(seed, task.task_id, turn),
                generation=generation,
            )
            records[arm]["completions"].append(completion)
            records[arm]["decision_prompts"].append(prompt)
            records[arm]["decision_unpadded_tokens"].append(unpadded_count)
            records[arm]["decision_padded_tokens"].append(padded_count)
            try:
                action = parse_action(completion)
                result = envs[arm].step(action)
                histories[arm].append({
                    "turn": turn, "completion": completion, "action": action, "tool_result": result
                })
            except ValueError as exc:
                histories[arm].append({
                    "turn": turn,
                    "completion": completion,
                    "tool_result": {"status": "INVALID_ACTION", "reason": str(exc)},
                })
    return records


def _append_jsonl(handle: Any, row: dict[str, Any]) -> None:
    handle.write(canonical_json(row) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_family(
    *,
    config: str | Path,
    sealed_corpus: str | Path,
    output: str | Path,
    family: str,
    runtime_factory: RuntimeFactory | None = None,
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ValueError("family is not frozen")
    config_bytes = Path(config).resolve().read_bytes()
    sealed_bytes = Path(sealed_corpus).resolve().read_bytes()
    cfg = _load_config(config_bytes)
    bundle = load_sealed_bytes(sealed_bytes)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    inventory, code_sha = code_inventory(package)
    commit = _git_commit(repo)
    bindings = cfg.get("integrity", {})
    expected_code = str(bindings.get("code_tree_sha256", ""))
    frozen = expected_code == code_sha
    test_hooks = runtime_factory is not None
    if not test_hooks and (not frozen or cfg.get("status") != FROZEN_STATUS):
        raise ValueError("formal runner requires frozen status and code-tree binding")
    run_binding = {
        "kind": "phantom_rollback_g0_run_binding",
        "run_id": secrets.token_hex(32),
        "absolute_output_root": str(root),
        "family": family,
        "config_sha256": sha256_bytes(config_bytes),
        "sealed_corpus_sha256": sha256_bytes(sealed_bytes),
        "git_commit": commit,
        "code_tree_sha256": code_sha,
    }
    atomic_bytes(root / "config.yaml", config_bytes, overwrite=False)
    atomic_bytes(root / "sealed_corpus.json", sealed_bytes, overwrite=False)
    atomic_json(root / "RUN_BINDING.json", run_binding, overwrite=False)
    model = cfg["models"][family]
    factory = runtime_factory or TransformersRuntime
    runtime = factory(str(model["id"]), str(model["revision"]))
    provenance = {
        "kind": "phantom_rollback_g0_runtime_provenance",
        "family": family,
        "model_id": model["id"],
        "revision": model["revision"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": __import__("numpy").__version__,
        "pyyaml_version": yaml.__version__,
        "git_commit": commit,
        "git_dirty_tracked": _git_dirty(repo),
        "code_tree_sha256": code_sha,
        "code_inventory": inventory,
        "bindings_frozen": frozen,
        "test_hooks_active": test_hooks,
        **dict(runtime.provenance()),
    }
    if not test_hooks and provenance.get("resolved_model_commit") != model["revision"]:
        raise ValueError("model runtime did not resolve to the frozen Hub revision")
    atomic_json(root / "PROVENANCE.json", provenance, overwrite=False)
    running = root / "RUNNING.json"
    atomic_json(running, {"state": "INCOMPLETE_DO_NOT_ANALYZE", "records_completed": 0})
    generation = cfg["generation"]
    margin = int(generation["padding_margin_tokens"])
    raw_partial = root / "raw_trajectories.partial.jsonl"
    comp_partial = root / "comprehension.partial.jsonl"
    completed = 0
    with comp_partial.open("x", encoding="utf-8", newline="\n") as handle:
        for case in bundle.comprehension:
            prompt = render_comprehension_prompt(case)
            completion = runtime.generate(
                prompt,
                seed=deterministic_seed(family, case.case_id, "comprehension"),
                generation=generation,
            )
            _append_jsonl(handle, {
                "case_id": case.case_id,
                "schema_id": case.schema_id,
                "prompt": prompt,
                "prompt_tokens": runtime.count_prompt_tokens(prompt),
                "completion": completion,
            })
    with raw_partial.open("x", encoding="utf-8", newline="\n") as handle:
        for task, seed_index, seed, schema_id in _block_schedule(bundle):
            block = _execute_block(
                runtime,
                task,
                schema_id=schema_id,
                seed=seed,
                generation=generation,
            )
            for arm in ARM_IDS:
                _append_jsonl(handle, {
                    "record_id": f"{task.task_id}|{seed}|{arm}",
                    "task_id": task.task_id,
                    "arm": arm,
                    "seed": seed,
                    "schema_id": schema_id,
                    "rollback_semantics_sha256": rollback_semantics_sha256(arm),
                    **block[arm],
                })
                completed += 1
            if completed % 16 == 0:
                atomic_json(running, {
                    "state": "INCOMPLETE_DO_NOT_ANALYZE",
                    "records_completed": completed,
                    "record_count": len(bundle.tasks) * len(bundle.seeds) * len(bundle.arms),
                })
    raw = root / "raw_trajectories.jsonl"
    comprehension = root / "comprehension.jsonl"
    os.replace(raw_partial, raw)
    os.replace(comp_partial, comprehension)
    running.unlink()
    artifacts = {
        name: sha256_file(root / name)
        for name in (
            "config.yaml", "sealed_corpus.json", "RUN_BINDING.json", "PROVENANCE.json",
            "raw_trajectories.jsonl", "comprehension.jsonl",
        )
    }
    manifest = {
        "kind": "phantom_rollback_g0_family",
        "state": "COMPLETE",
        "family": family,
        "model_id": model["id"],
        "revision": model["revision"],
        "trajectory_count": completed,
        "comprehension_count": len(bundle.comprehension),
        "run_binding_sha256": artifacts["RUN_BINDING.json"],
        "artifacts": artifacts,
    }
    atomic_json(root / "MANIFEST.json", manifest, overwrite=False)
    atomic_bytes(root / "COMPLETE", b"COMPLETE\n", overwrite=False)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sealed-corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", required=True, choices=FAMILIES)
    args = parser.parse_args(argv)
    print(canonical_json(run_family(**vars(args))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
