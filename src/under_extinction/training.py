"""LoRA SFT for behavior-matched controller organisms."""

from __future__ import annotations

import inspect
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

from .config import output_root
from .io import read_jsonl, sha256_file, write_json
from .manifest import create_manifest, finalize_manifest, project_hash
from .modeling import ChoiceCollator, ChoiceSFTDataset, load_base_model, load_tokenizer, verify_choice_tokens
from .schema import Controller


class RunStoppedEarly(RuntimeError):
    """Raised after writing a resumable partial run that must not be evaluated."""


def _deadline_epoch() -> float | None:
    raw = os.environ.get("UE_HARD_DEADLINE_EPOCH")
    return float(raw) if raw else None


def _training_arguments(config: dict[str, Any], run_dir: Path, seed: int) -> Any:
    from transformers import TrainingArguments

    training = config["training"]
    kwargs: dict[str, Any] = {
        "output_dir": str(run_dir / "checkpoints"),
        "overwrite_output_dir": False,
        "num_train_epochs": float(training["epochs"]),
        "per_device_train_batch_size": int(training["batch_size"]),
        "per_device_eval_batch_size": int(config["evaluation"]["batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "warmup_ratio": float(training["warmup_ratio"]),
        "logging_steps": int(training["logging_steps"]),
        "save_steps": int(training["save_steps"]),
        "eval_steps": int(training["eval_steps"]),
        "save_strategy": "steps",
        "logging_strategy": "steps",
        "bf16": config["model"]["dtype"] == "bfloat16",
        "fp16": config["model"]["dtype"] == "float16",
        "tf32": True,
        "gradient_checkpointing": True,
        "optim": "adamw_torch_fused",
        "lr_scheduler_type": "cosine",
        "save_total_limit": 3,
        "report_to": [],
        "remove_unused_columns": False,
        "seed": int(seed),
        "data_seed": int(seed),
        "dataloader_num_workers": 2,
        "dataloader_pin_memory": True,
    }
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = "steps"
    else:
        raise RuntimeError("Unsupported Transformers TrainingArguments: no evaluation strategy argument")
    return TrainingArguments(**kwargs)


def _make_callbacks(run_dir: Path, soft_stop_minutes: int) -> list[Any]:
    from transformers import TrainerCallback

    class DeadlineAndSignalCallback(TrainerCallback):
        def __init__(self) -> None:
            self.stop_requested = False
            self.stop_reason: str | None = None
            self.deadline = _deadline_epoch()
            self.soft_seconds = soft_stop_minutes * 60
            self.metrics_path = run_dir / "metrics.jsonl"
            signal.signal(signal.SIGINT, self._signal)
            signal.signal(signal.SIGTERM, self._signal)

        def _signal(self, signum: int, _frame: Any) -> None:
            self.stop_requested = True
            self.stop_reason = f"signal_{signum}"

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **_: Any) -> Any:
            event = {"time_epoch": time.time(), "step": state.global_step, **(logs or {})}
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if self.deadline is not None and time.time() >= self.deadline - self.soft_seconds:
                self.stop_requested = True
                self.stop_reason = "budget_soft_stop"
            if self.stop_requested:
                control.should_save = True
                control.should_training_stop = True
            return control

    return [DeadlineAndSignalCallback()]


def train_controller(
    config: dict[str, Any],
    *,
    controller: str,
    seed: int,
    data_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
    resume: bool = False,
) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, set_seed

    if controller not in {item.value for item in Controller}:
        raise ValueError(f"Unknown controller: {controller}")
    if seed not in config["organisms"]["seeds"]:
        raise ValueError(f"Seed {seed} is not preregistered in the config")
    root = output_root(config)
    source = Path(data_dir).resolve() if data_dir else root / "data"
    target = Path(run_dir).resolve() if run_dir else root / "runs" / f"{controller}_seed{seed}"
    train_path, dev_path = source / "train.jsonl", source / "dev.jsonl"
    evaluation_path, data_manifest_path = source / "evaluation.jsonl", source / "MANIFEST.json"
    prior_manifest: dict[str, Any] | None = None
    if resume:
        prior_path = target / "run_manifest.json"
        if not prior_path.exists():
            raise FileNotFoundError(f"--resume requires an existing run manifest at {prior_path}")
        prior_manifest = json.loads(prior_path.read_text(encoding="utf-8"))
        failures: list[str] = []
        if prior_manifest.get("state") not in {"STOPPED_BUDGET", "STOPPED_EARLY"}:
            failures.append(f"state is {prior_manifest.get('state')!r}")
        if prior_manifest.get("controller") != controller or int(prior_manifest.get("training_seed", -1)) != seed:
            failures.append("controller or seed differs")
        if prior_manifest.get("config_sha256") != config["_config_sha256"] or prior_manifest.get("model") != config["model"]:
            failures.append("config or model differs")
        project_root = Path(config["_config_path"]).parent.parent
        if (prior_manifest.get("source") or {}).get("project_tree_sha256") != project_hash(project_root):
            failures.append("project source tree differs")
        for path in (train_path, dev_path, evaluation_path, data_manifest_path):
            recorded = (prior_manifest.get("data_files") or {}).get(path.name)
            if not recorded or not path.exists() or recorded.get("sha256") != sha256_file(path):
                failures.append(f"data differs for {path.name}")
        if failures:
            raise ValueError("Resume provenance validation failed: " + "; ".join(failures))
    else:
        terminal_markers = [name for name in ("COMPLETE", "STOPPED_BUDGET", "STOPPED_EARLY", "FAILED", "RUNNING") if (target / name).exists()]
        if terminal_markers or (target / "run_manifest.json").exists():
            raise FileExistsError(f"Run directory already contains state {terminal_markers}: {target}")
    target.mkdir(parents=True, exist_ok=True)
    manifest = create_manifest(
        config,
        run_dir=target,
        controller=controller,
        seed=seed,
        command_line=sys.argv,
        data_files=[train_path, dev_path, evaluation_path, data_manifest_path],
    )
    if prior_manifest is not None:
        manifest["resume_lineage"] = {
            "prior_run_id": prior_manifest.get("run_id"),
            "prior_state": prior_manifest.get("state"),
            "prior_global_step": (prior_manifest.get("result") or {}).get("global_step"),
        }
        write_json(target / "run_manifest.json", manifest)
    started = time.monotonic()
    try:
        set_seed(seed)
        tokenizer = load_tokenizer(config)
        train_records = list(read_jsonl(train_path))
        dev_records = list(read_jsonl(dev_path))
        token_check = verify_choice_tokens(tokenizer, train_records[0]["messages"], config["model"]["choice_labels"])
        write_json(target / "choice_token_check.json", token_check)
        if not token_check["equal_token_counts"]:
            raise ValueError("A and B have unequal completion token counts under the frozen tokenizer")
        model = load_base_model(config, training=True)
        lora = LoraConfig(
            r=int(config["training"]["lora_rank"]),
            lora_alpha=int(config["training"]["lora_alpha"]),
            lora_dropout=float(config["training"]["lora_dropout"]),
            target_modules=list(config["training"]["lora_targets"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.enable_input_require_grads()
        train_dataset = ChoiceSFTDataset(train_records, tokenizer, controller, int(config["model"]["max_length"]))
        dev_dataset = ChoiceSFTDataset(dev_records, tokenizer, controller, int(config["model"]["max_length"]))
        collator = ChoiceCollator(int(tokenizer.pad_token_id))
        args = _training_arguments(config, target, seed)
        callbacks = _make_callbacks(target, int(config["budget"]["soft_stop_minutes"]))
        stop_callback = callbacks[0]
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            callbacks=callbacks,
        )
        resume_checkpoint: bool | str = False
        if resume:
            checkpoints = sorted((target / "checkpoints").glob("checkpoint-*"), key=lambda path: int(path.name.split("-")[-1]))
            if not checkpoints:
                raise FileNotFoundError(f"--resume was requested but no checkpoints exist under {target}")
            resume_checkpoint = str(checkpoints[-1])
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        stopped_early = bool(stop_callback.stop_requested)
        adapter_name = (
            f"partial_adapter_step{trainer.state.global_step}_{int(time.time())}" if stopped_early else "final_adapter"
        )
        temporary_adapter = target / f".{adapter_name}.tmp"
        final_adapter = target / adapter_name
        if temporary_adapter.exists() or final_adapter.exists():
            raise FileExistsError("Refusing to overwrite an existing final adapter")
        trainer.save_model(str(temporary_adapter))
        os.replace(temporary_adapter, final_adapter)
        tokenizer.save_pretrained(final_adapter)
        adapter_files = {
            str(path.relative_to(final_adapter)): sha256_file(path)
            for path in sorted(final_adapter.rglob("*"))
            if path.is_file()
        }
        result = {
            "wall_seconds": time.monotonic() - started,
            "global_step": trainer.state.global_step,
            "training_metrics": train_result.metrics,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(),
            "adapter_relative_path": adapter_name,
            "adapter_files": adapter_files,
            "choice_token_check": token_check,
        }
        if stopped_early:
            state = "STOPPED_BUDGET" if stop_callback.stop_reason == "budget_soft_stop" else "STOPPED_EARLY"
            result["stop_reason"] = stop_callback.stop_reason
            finalize_manifest(target, manifest, state, result)
            raise RunStoppedEarly(
                f"Run stopped safely ({stop_callback.stop_reason}); resume from the latest checkpoint. "
                "The partial adapter is not a completed organism."
            )
        finalize_manifest(target, manifest, "COMPLETE", result)
        return final_adapter
    except RunStoppedEarly:
        raise
    except BaseException as exc:
        finalize_manifest(target, manifest, "FAILED", {"wall_seconds": time.monotonic() - started, "error": repr(exc)})
        raise
