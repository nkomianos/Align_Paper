"""Launch ONLY the prospectively superseding reduced EndoPAHF DEV protocol.

This file is an unexecuted launcher. Preparation/testing imports no model. An
actual invocation requires a separately authorized single-GPU local snapshot.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from interaction_sprint.hindsight_execution_integrity import (
    canonical_sha256, capture_execution_provenance, capture_model_provenance,
    effective_adamw_settings, load_learning_dev, make_arm_execution_receipt, row_binding, seal_artifacts, sha256_file,
    tensor_tree_sha256,
)
from interaction_sprint.hindsight_pahf_reduced import (
    HINDSIGHT_BLOCK, TRAINED_ARMS, build_interface_jobs, build_schedules,
    cheap_readouts, method_decision, qualify_acquisition, qualify_interface,
    validate_learning_dev,
)


class BudgetStop(RuntimeError):
    """A bounded incomplete run is invalid, never a negative scientific result."""


def runtime_deadlines(max_seconds, allocation_deadline_utc, *, wall_now=None, monotonic_now=None, grace_seconds=120.0):
    """Execution budget only; no scientific threshold or schedule changes."""
    if not math.isfinite(max_seconds) or not 120 < max_seconds <= 100 * 3600:
        raise ValueError("runtime cap must exceed120s and be at most100h")
    deadline = datetime.fromisoformat(allocation_deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("allocation deadline requires explicit UTC offset")
    wall = time.time() if wall_now is None else wall_now
    mono = time.monotonic() if monotonic_now is None else monotonic_now
    remaining = min(float(max_seconds), deadline.timestamp() - wall)
    if remaining <= grace_seconds + 1:
        raise ValueError("insufficient allocation time for graceful stop reserve")
    return {"max_runtime_seconds": float(max_seconds), "effective_runtime_seconds": remaining,
            "grace_seconds": grace_seconds, "soft_monotonic": mono + remaining - grace_seconds,
            "hard_monotonic": mono + remaining,
            "soft_deadline_utc": datetime.fromtimestamp(wall + remaining - grace_seconds, timezone.utc).isoformat(),
            "hard_deadline_utc": datetime.fromtimestamp(wall + remaining, timezone.utc).isoformat(),
            "allocation_deadline_utc": deadline.astimezone(timezone.utc).isoformat()}


def _budget_watchdog(arguments):
    """Separate POSIX session, outside the watched group; no model imports."""
    parent, soft, hard, directory = int(arguments[0]), float(arguments[1]), float(arguments[2]), Path(arguments[3])
    if os.name != "posix" or os.getppid() != parent or os.getpgid(parent) != parent or os.getsid(parent) != parent:
        raise RuntimeError("watchdog refuses a shared/unbound process group")
    requested_stop = [None]
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda signum, frame: requested_stop.__setitem__(0, time.monotonic()))
    ready_pending = directory / "budget_watchdog_ready.pending"
    ready_pending.write_text(json.dumps({
        "pid": os.getpid(), "parent_pid": parent, "watched_process_group": parent,
        "watchdog_session": os.getsid(0), "soft_monotonic": soft, "hard_monotonic": hard,
        "survives_ssh_disconnect": True}), encoding="utf-8")
    # Publish a complete acknowledgement: existence must imply parseable JSON.
    ready_pending.replace(directory / "budget_watchdog_ready.json")
    soft_sent = False
    while True:
        # The actual-parent check prevents PID reuse from targeting another job.
        if os.getppid() != parent:
            return
        if os.getpgid(parent) != parent or os.getsid(parent) != parent:
            raise RuntimeError("watched process lost isolated session binding")
        now = time.monotonic()
        if requested_stop[0] is not None:
            hard = min(hard, requested_stop[0] + 120.0)
        if now >= hard:
            # Stop compute before receipt I/O, which can stall on a full disk.
            os.killpg(parent, signal.SIGKILL)
            (directory / "BUDGET_HARD_STOP.json").write_text(json.dumps({
                "decision": "REDUCED_HARD_BUDGET_STOP", "parent_pid": parent,
                "watchdog_pid": os.getpid(), "utc": datetime.now(timezone.utc).isoformat(),
                "reason": "parent did not exit by bounded hard deadline", "paper_green_light": False}), encoding="utf-8")
            return
        if now >= soft and not soft_sent:
            os.killpg(parent, signal.SIGTERM)
            soft_sent = True
        time.sleep(min(.2, max(.01, hard - now)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/hindsight_pahf_reduced_dev_v1.json"))
    parser.add_argument("--max-runtime-seconds", type=float, default=14400.0)
    parser.add_argument("--allocation-deadline-utc", required=True,
                        help="Actual allocation-wide deadline; transport owns cumulative GPU-hour accounting")
    args = parser.parse_args()
    deadlines = runtime_deadlines(args.max_runtime_seconds, args.allocation_deadline_utc)
    repository = Path(__file__).resolve().parents[1]
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    frozen = json.loads((repository / "configs/hindsight_pahf_reduced_dev_v1.json").read_text(encoding="utf-8"))
    if cfg != frozen or tuple(cfg["trained_arms"]) != TRAINED_ARMS:
        raise ValueError("config differs from prospectively versioned protocol")
    if args.root.resolve() == args.input_root.resolve() or args.root.resolve().is_relative_to(args.input_root.resolve()):
        raise ValueError("output root must not be inside frozen input root")
    if args.model_snapshot.name != cfg["model_revision"] or args.model_snapshot.parent.name != "snapshots":
        raise ValueError("model snapshot must use the immutable snapshots/<revision> layout")
    if args.root.resolve().is_relative_to(args.model_snapshot.parent.parent.resolve()):
        raise ValueError("output root must not mutate the model cache/snapshot")
    learning, development, input_receipt = load_learning_dev(args.input_root, expected_manifest_sha256=cfg["input_manifest_sha256"])
    validate_learning_dev(learning, development, cfg)
    schedules = build_schedules(learning, cfg)
    interface_jobs = build_interface_jobs(learning, cfg, schedules["pooled_base_ids"])
    readouts = cheap_readouts(learning, development, schedules["pooled_base_ids"], cfg)
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def write(name, value):
        path = args.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    write("spec.json", cfg)
    write("input_receipt.json", input_receipt)
    write("panels.json", {k: schedules[k] for k in ("original_panels", "pooled_base_ids")})
    write("schedules.json", schedules)
    write("interface_jobs.json", interface_jobs)
    write("cheap_controls.json", readouts)
    sources = [Path(__file__), args.config,
               repository / "src/interaction_sprint/hindsight_pahf_reduced.py",
               repository / "src/interaction_sprint/hindsight_execution_integrity.py",
               repository / "src/interaction_sprint/hindsight_neural_anchor.py",
               repository / "src/latent_contract/sender_update.py",
               repository / "scripts/verify_hindsight_pahf_reduced_dev.py"]
    try:
        provenance = capture_execution_provenance(repository, sources, sys.argv, cfg)
    except Exception as exc:
        failure = {"decision": "REDUCED_EXECUTION_INVALID", "classification": "Invalid assay/capability",
                   "stage": "source_provenance", "error": type(exc).__name__, "message": str(exc),
                   "qualified": False, "completed_arms": [], "confirmation_opened": False, "paper_green_light": False}
        write("FAILED.json", failure)
        write("RESULT.json", failure)
        seal_artifacts(args.root)
        raise
    write("provenance.json", provenance)
    write("source_hashes.json", provenance["source_sha256"])
    write("generation_receipt.json", {"mode": "forward_logits_only", "generate_called": False})
    counters = defaultdict(lambda: defaultdict(float))
    logits_index = []
    rendered_texts = defaultdict(dict)
    completed = []
    predictions = {}
    active_arm = None
    active_optimizer = None
    active_steps = []
    model = None
    phase = "setup"
    finalized = False
    watchdog = None
    stop_reason = [None]

    def check_budget():
        if stop_reason[0] is not None:
            raise BudgetStop(stop_reason[0])
        if time.monotonic() >= deadlines["soft_monotonic"]:
            raise BudgetStop("soft runtime/allocation deadline reached; preserving partial evidence")
        if watchdog is not None and watchdog.poll() is not None:
            raise BudgetStop("independent budget watchdog exited unexpectedly")

    def verify_bound_sources():
        for relative, expected in provenance["source_sha256"].items():
            if sha256_file(repository / relative) != expected:
                raise RuntimeError("bound source changed during run: " + relative)

    def finish(result):
        nonlocal finalized
        check_budget()
        verify_bound_sources()
        write("RESULT.json", {**result, "wall_seconds": time.monotonic() - started,
                              "completed_arms": list(completed), "confirmation_opened": False})
        write("compute.json", {"by_phase": {k: dict(v) for k, v in counters.items()},
                               "optimizer_updates": sum(v.get("optimizer_updates", 0) for v in counters.values()),
                               "global_unique_immediate_learning_bases": 630,
                               "sparse_unique_delayed_learning_bases": 64,
                               "oracle_unique_delayed_learning_bases": 630 if "oracle_delayed" in completed else 0,
                               "wall_seconds": time.monotonic() - started,
                               "equality_scope": "equal sparse label access; not equal compute"})
        write("logits_index.json", logits_index)
        write("rendered_texts.json", dict(rendered_texts))
        write("COMPLETE.json", {"status": "complete_with_recorded_decision", "completed_arms": completed,
                                "paper_green_light": False, "confirmation_opened": False})
        seal_artifacts(args.root)
        finalized = True

    try:
        phase = "budget_supervision"
        if os.name != "posix":
            raise RuntimeError("GPU launcher requires POSIX process-group budget supervision")
        if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
            try:
                os.setsid()
            except OSError as exc:
                raise RuntimeError("launch runner through setsid in its own session") from exc
        if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
            raise RuntimeError("runner process group is not isolated")
        watchdog = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--_budget-watchdog",
            str(os.getpid()), str(deadlines["soft_monotonic"]), str(deadlines["hard_monotonic"]), str(args.root.resolve())],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, start_new_session=True)

        def request_stop(signum, frame):
            stop_reason[0] = "external/deadline signal " + str(signum)
            if watchdog is not None and watchdog.poll() is None:
                try:
                    os.kill(watchdog.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass

        for stop_signal in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(stop_signal, request_stop)
        ready_deadline = time.monotonic() + 10
        while not (args.root / "budget_watchdog_ready.json").exists():
            check_budget()
            if time.monotonic() > ready_deadline:
                raise BudgetStop("watchdog did not acknowledge readiness")
            time.sleep(.05)
        ready = json.loads((args.root / "budget_watchdog_ready.json").read_text())
        if ready["pid"] != watchdog.pid or ready["parent_pid"] != os.getpid():
            raise BudgetStop("watchdog readiness binding differs")
        write("budget.json", {**deadlines, "runner_pid": os.getpid(), "runner_session": os.getsid(0),
            "runner_process_group": os.getpgrp(), "watchdog_pid": watchdog.pid,
            "watchdog_session": ready["watchdog_session"], "default_runtime_hours": 4,
            "scope": "per-run wall cap AND caller-supplied allocation deadline; cumulative billing owned by transport",
            "resume_supported": False})
        check_budget()
        phase = "model_setup"
        import numpy as np
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM
        from interaction_sprint.hindsight_neural_anchor import install_qwen35_lora, reverse_kl_per_example
        from latent_contract.sender_update import adapter_state, load_adapter

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU required; no CPU model fallback")
        # This hashes the already-cached immutable snapshot and performs no download.
        model_receipt = capture_model_provenance(args.model_snapshot, cfg["model_id"], cfg["model_revision"])
        check_budget()
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        torch.set_float32_matmul_precision("high")
        tokenizer = AutoTokenizer.from_pretrained(args.model_snapshot, local_files_only=True, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = Qwen3_5ForCausalLM.from_pretrained(args.model_snapshot, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation=cfg["attention"], device_map={"": torch.cuda.current_device()},
            low_cpu_mem_usage=True, use_kernels=cfg["use_kernels"]).eval()
        model.config.use_cache = cfg["use_cache"]
        effective_receipt = capture_model_provenance(args.model_snapshot, cfg["model_id"], cfg["model_revision"], tokenizer=tokenizer, model=model)
        check_budget()
        if model_receipt["snapshot_tree_sha256"] != effective_receipt["snapshot_tree_sha256"]:
            raise ValueError("local model files changed during loading")
        verify_bound_sources()
        write("model_provenance.json", effective_receipt)
        with (args.root / "tokenizer_backend.json").open("x", encoding="utf-8", newline="") as handle:
            handle.write(tokenizer.backend_tokenizer.to_str())
        vocab_size = int(getattr(getattr(model.config, "text_config", model.config), "vocab_size"))
        answer_sequences = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"]
        if any(len(ids) != 1 for ids in answer_sequences) or len({ids[0] for ids in answer_sequences}) != 4:
            raise ValueError("A/B/C/D must be distinct native single tokens")
        answer_ids = [ids[0] for ids in answer_sequences]
        modules = install_qwen35_lora(model, rank=cfg["lora_rank"], alpha=cfg["lora_alpha"])
        named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        parameter_names = [n for n, _ in named_parameters]
        parameters = [p for _, p in named_parameters]
        initial = adapter_state(model)
        torch.save(initial, args.root / "initial_adapter.pt")
        write("runtime_setup.json", {"lora_modules": modules, "parameter_names": parameter_names,
            "trainable_parameters": sum(p.numel() for p in parameters), "answer_token_ids": answer_ids,
            "vocab_size": vocab_size,
            "initial_adapter_tensor_sha256": tensor_tree_sha256(initial), "model_mode": "eval_with_student_gradients",
            "cuda_device": torch.cuda.get_device_name(0), "effective_optimizer_config": cfg["optimizer"],
            "cuda_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda_allocated_after_setup_bytes": torch.cuda.memory_allocated(0),
            "cuda_reserved_after_setup_bytes": torch.cuda.memory_reserved(0),
            "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(), "cuda_device_count": torch.cuda.device_count(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": "high", "sdpo_scope": "first-native-answer-token reverse KL(student||detached current-adapter hindsight teacher); not released multi-token SDPO"})

        def render(text):
            return tokenizer.apply_chat_template([{"role": "user", "content": str(text)}], tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)

        def logits(texts, *, grad=False, role="evaluation", return_inputs=False):
            check_budget()
            rendered = [render(text) for text in texts]
            encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
            lengths = encoded["attention_mask"].sum(-1)
            if not texts or int(lengths.max()) > cfg["max_tokens"]:
                raise ValueError("context exceeds 1024 tokens; truncation is forbidden")
            count = counters[phase]
            count["forward_calls_attempted"] += 1
            with torch.set_grad_enabled(grad):
                output = model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
            if output.shape[-1] != vocab_size:
                raise ValueError("effective logits vocabulary differs from model config")
            count["forward_calls"] += 1
            count["forward_rows"] += len(texts)
            count["forward_tokens_unpadded"] += int(lengths.sum())
            count["forward_tokens_padded"] += encoded["input_ids"].numel()
            count[role + "_forward_calls"] += 1
            count[role + "_forward_rows"] += len(texts)
            count[role + "_forward_tokens_unpadded"] += int(lengths.sum())
            check_budget()
            if return_inputs:
                return output, {"input_ids": encoded["input_ids"].detach().cpu().numpy(),
                                "attention_mask": encoded["attention_mask"].detach().cpu().numpy(),
                                "rendered_texts": rendered,
                                "rendered_text_sha256": [hashlib.sha256(t.encode()).hexdigest() for t in rendered]}
            return output

        def score_batch(rows, texts, name, batch_index, kind):
            output, inputs = logits(texts, return_inputs=True)
            output = output.float()
            full_log = output.log_softmax(-1)
            choice = full_log[:, answer_ids]
            normalized = choice.log_softmax(-1)
            path = f"logits/{name}/batch_{batch_index:03d}.npy"
            (args.root / path).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.root / path, output.detach().cpu().numpy(), allow_pickle=False)
            input_path = f"logits/{name}/batch_{batch_index:03d}_input_ids.npy"
            mask_path = f"logits/{name}/batch_{batch_index:03d}_attention_mask.npy"
            np.save(args.root / input_path, inputs["input_ids"], allow_pickle=False)
            np.save(args.root / mask_path, inputs["attention_mask"], allow_pickle=False)
            for row, text in zip(rows, inputs["rendered_texts"]):
                rendered_texts[name][row.get("job_id", row["id"])] = text
            logits_index.append({"path": path, "arm": name, "row_ids": [r.get("job_id", r["id"]) for r in rows],
                                 "answer_token_ids": answer_ids, "kind": kind,
                                 "input_ids_path": input_path, "attention_mask_path": mask_path,
                                 "rendered_text_sha256": inputs["rendered_text_sha256"]})
            result = []
            for i, row in enumerate(rows):
                b = {k: row[k] for k in ("id", "base_id", "label_rotation", "source_user")} if kind == "interface" else row_binding(row)
                if kind == "interface":
                    b.update({k: row[k] for k in ("job_id", "context", "target")})
                result.append({**b, "full_vocab_choice_log_probabilities": choice[i].tolist(),
                    "normalized_choice_log_probabilities": normalized[i].tolist(),
                    "full_vocabulary_choice_mass": float(choice[i].logsumexp(0).exp()),
                    "choice_logits": output[i, answer_ids].tolist(), "log_normalizer": float(output[i].logsumexp(0)),
                    "full_vocab_argmax_token_id": int(output[i].argmax()), "answer_token_ids": answer_ids})
            return result

        phase = "interface"
        interface_predictions = []
        for batch_index, start in enumerate(range(0, len(interface_jobs), cfg["eval_batch"])):
            batch = interface_jobs[start:start + cfg["eval_batch"]]
            interface_predictions.extend(score_batch(batch, [r["text"] for r in batch], "interface", batch_index, "interface"))
        write("interface_predictions.json", interface_predictions)
        interface_result = qualify_interface(interface_jobs, interface_predictions, cfg)
        write("interface_RESULT.json", interface_result)
        if not interface_result["qualified"]:
            finish(interface_result)
            return

        def evaluate(name):
            nonlocal phase
            phase = name + "_eval"
            result = []
            for batch_index, start in enumerate(range(0, len(development), cfg["eval_batch"])):
                batch = development[start:start + cfg["eval_batch"]]
                result.extend(score_batch(batch, [r["prompt"] for r in batch], name, batch_index, "dev"))
            write(name + "_dev_predictions.json", result)
            predictions[name] = result

        evaluate("baseline")
        # Student functions receive whitelisted immediate records; sparse target
        # lookup is physically restricted to the same 64 anchor bases.
        public = {r["id"]: {"id": r["id"], "base_id": r["base_id"], "prompt": r["prompt"],
                           "immediate_followup": r["immediate_followup"]} for r in learning}
        anchors = {r["id"]: {**public[r["id"]], "old_target": r["old_target"],
                             "delayed_expression_followup": r["delayed_expression_followup"]}
                   for r in learning if r["base_id"] in set(schedules["pooled_base_ids"])}
        write("label_access.json", {"raw_immediate": {"immediate_bases": 630, "delayed_bases": 0},
            "oracle_delayed": {"immediate_bases": 630, "delayed_bases": 630, "fair_comparator": False},
            **{name: {"immediate_bases_accessible": 630, "delayed_bases": 64, "base_ids": schedules["pooled_base_ids"]}
               for name in ("pooled_sft", "pooled_sdpo", "residual", "mixture")}})

        def teacher(rows, field):
            return logits([r["prompt"] + HINDSIGHT_BLOCK.format(follow_up=str(r[field]).strip()) for r in rows], role="teacher")

        def student(rows):
            return logits([r["prompt"] for r in rows], grad=True, role="student")

        def run_arm(name):
            nonlocal active_arm, active_optimizer, active_steps, phase
            active_arm = name
            active_steps = []
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            load_adapter(model, initial)
            start_state = adapter_state(model)
            if tensor_tree_sha256(start_state) != tensor_tree_sha256(initial):
                raise ValueError("adapter reset differs")
            torch.save(start_state, args.root / (name + "_initial_adapter.pt"))
            settings = dict(cfg["optimizer"])
            settings["betas"] = tuple(settings["betas"])
            optimizer = torch.optim.AdamW(parameters, **settings)
            active_optimizer = optimizer
            if optimizer.state:
                raise ValueError("fresh optimizer is not empty")
            torch.save(optimizer.state_dict(), args.root / (name + "_initial_optimizer.pt"))
            effective_settings = effective_adamw_settings(optimizer, cfg["optimizer"])
            oracle = ({r["id"]: {**public[r["id"]], "delayed_expression_followup": r["delayed_expression_followup"]}
                       for r in learning} if name == "oracle_delayed" else None)
            phase = name + "_train"
            for i, population_ids in enumerate(schedules["population"]):
                check_budget()
                batch = [public[id] for id in population_ids]
                anchor_ids = schedules["anchors"][i] if name in ("pooled_sft", "pooled_sdpo", "residual", "mixture") else []
                anchor_batch = [anchors[id] for id in anchor_ids]
                torch.cuda.synchronize()
                step_start = time.monotonic()
                before = dict(counters[phase])
                optimizer.zero_grad(set_to_none=True)
                detail = {}
                if name in ("raw_immediate", "oracle_delayed"):
                    target_rows = batch if oracle is None else [oracle[id] for id in population_ids]
                    field = "immediate_followup" if oracle is None else "delayed_expression_followup"
                    t = teacher(target_rows, field)
                    loss = reverse_kl_per_example(student(batch), t).mean()
                    detail["population_mean"] = float(loss.detach())
                elif name == "pooled_sft":
                    output = student(anchor_batch).float().log_softmax(-1)
                    targets = torch.tensor([answer_ids["ABCD".index(r["old_target"])] for r in anchor_batch], device=model.device)
                    loss = -output.gather(1, targets[:, None]).mean()
                elif name == "pooled_sdpo":
                    t = teacher(anchor_batch, "delayed_expression_followup")
                    loss = reverse_kl_per_example(student(anchor_batch), t).mean()
                else:
                    immediate_teacher = teacher(batch, "immediate_followup")
                    immediate_mean = reverse_kl_per_example(student(batch), immediate_teacher).mean()
                    anchor_student = student(anchor_batch)
                    delayed_teacher = teacher(anchor_batch, "delayed_expression_followup")
                    delayed_each = reverse_kl_per_example(anchor_student, delayed_teacher)
                    if name == "mixture":
                        loss = cfg["mixture_weight"] * immediate_mean + (1 - cfg["mixture_weight"]) * delayed_each.mean()
                    else:
                        anchor_immediate = teacher(anchor_batch, "immediate_followup")
                        residual = (delayed_each - reverse_kl_per_example(anchor_student, anchor_immediate)).mean()
                        loss = immediate_mean + residual
                        detail["anchor_residual_mean"] = float(residual.detach())
                    detail.update(population_immediate_mean=float(immediate_mean.detach()), anchor_delayed_mean=float(delayed_each.mean().detach()))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite objective")
                check_budget()
                counters[phase]["backward_calls_attempted"] += 1
                loss.backward()
                counters[phase]["backward_calls"] += 1
                counters[phase]["backward_student_tokens_unpadded"] += counters[phase]["student_forward_tokens_unpadded"] - before.get("student_forward_tokens_unpadded", 0)
                check_budget()
                norm = torch.nn.utils.clip_grad_norm_(parameters, cfg["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                counters[phase]["optimizer_updates"] += 1
                counters[phase]["seconds"] += time.monotonic() - step_start
                step = {"step": i + 1, "population_batch_ids": population_ids if name not in ("pooled_sft", "pooled_sdpo") else [],
                        "anchor_ids": anchor_ids, "loss": float(loss.detach()), "gradient_norm": float(norm),
                        "learning_rate": optimizer.param_groups[0]["lr"], "details": detail,
                        "cuda_memory": {"allocated_bytes": torch.cuda.memory_allocated(0),
                                        "reserved_bytes": torch.cuda.memory_reserved(0),
                                        "cumulative_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                                        "cumulative_peak_reserved_bytes": torch.cuda.max_memory_reserved(0)},
                        "compute_delta": {k: v - before.get(k, 0) for k, v in counters[phase].items()}}
                active_steps.append(step)
                with (args.root / (name + "_progress.jsonl")).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step, allow_nan=False) + "\n")
            torch.save(adapter_state(model), args.root / (name + "_adapter.pt"))
            torch.save(optimizer.state_dict(), args.root / (name + "_optimizer.pt"))
            write(name + "_steps.json", active_steps)
            write(name + "_execution.json", make_arm_execution_receipt(args.root, name, parameter_names, effective_settings, cfg["steps"]))
            completed.append(name)
            active_arm = None
            active_optimizer = None
            evaluate(name)

        run_arm("raw_immediate")
        run_arm("oracle_delayed")
        acquisition = qualify_acquisition(development, predictions, cfg)
        write("acquisition_RESULT.json", acquisition)
        if not acquisition["qualified"]:
            finish(acquisition)
            return
        for name in ("pooled_sft", "pooled_sdpo", "residual", "mixture"):
            run_arm(name)
        finish(method_decision(development, predictions, readouts, cfg))
    except Exception as exc:
        if finalized:
            raise
        if active_arm is not None and model is not None:
            import torch
            from latent_contract.sender_update import adapter_state
            torch.save(adapter_state(model), args.root / (active_arm + "_failed_adapter.pt"))
            if active_optimizer is not None:
                torch.save(active_optimizer.state_dict(), args.root / (active_arm + "_failed_optimizer.pt"))
            write(active_arm + "_failed_steps.json", active_steps)
        write("FAILED.json", {"stage": phase, "active_arm": active_arm, "executed_steps": len(active_steps),
                              "error": type(exc).__name__, "message": str(exc), "completed_arms": completed,
                              "confirmation_opened": False, "paper_green_light": False})
        if (args.root / "RESULT.json").exists():
            (args.root / "RESULT.json").rename(args.root / "UNSEALED_RESULT.json")
        if (args.root / "COMPLETE.json").exists():
            (args.root / "COMPLETE.json").rename(args.root / "UNSEALED_COMPLETE.json")
        write("RESULT.json", {"decision": "REDUCED_BUDGET_STOP" if isinstance(exc, BudgetStop) else "REDUCED_EXECUTION_INVALID", "classification": "Invalid assay/capability",
                              "qualified": False, "completed_arms": completed, "active_arm": active_arm,
                              "error": type(exc).__name__, "message": str(exc), "confirmation_opened": False,
                              "paper_green_light": False})
        if not (args.root / "logits_index.json").exists():
            write("logits_index.json", logits_index)
        if not (args.root / "rendered_texts.json").exists():
            write("rendered_texts.json", dict(rendered_texts))
        if not (args.root / "compute.json").exists():
            write("compute.json", {"by_phase": {k: dict(v) for k, v in counters.items()}, "wall_seconds": time.monotonic() - started})
        if not (args.root / "MANIFEST.json").exists():
            seal_artifacts(args.root)
        raise


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_budget-watchdog":
        _budget_watchdog(sys.argv[2:])
    else:
        main()
