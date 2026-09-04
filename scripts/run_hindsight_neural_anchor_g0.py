"""GPU runner for the frozen single-token neural delayed-anchor SDPO gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np

from interaction_sprint.hindsight_neural_anchor import (
    ANCHORS_PER_BATCH,
    HINDSIGHT_BLOCK,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_SDPO_COMMIT,
    OFFICIAL_SDPO_REPOSITORY,
    SEED,
    TRAIN_BATCH,
    TRAIN_STEPS,
    augmented_reverse_kl,
    build_records,
    feedback_for_row,
    hindsight_user_text,
    install_qwen35_lora,
    record_invariants,
    reverse_kl_per_example,
    semantic_letter_index,
)
from latent_contract.sender_update import adapter_state, load_adapter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    rows, evaluation, schedule = build_records()
    by_id = {str(row["id"]): row for row in rows}
    spec = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "full-vocabulary reverse KL(student(.|x) || stopgrad teacher(.|x,o)) at first answer token",
        "seed": SEED,
        "steps": TRAIN_STEPS,
        "batch": TRAIN_BATCH,
        "anchors_per_batch": ANCHORS_PER_BATCH,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "paper_green_light": False,
    }
    write("spec.json", spec)
    write("cases.json", {"train": rows, "evaluation": evaluation})
    write("schedule.json", schedule)
    write("invariants.json", record_invariants(rows))
    (args.root / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    dependencies = [
        Path(__file__).parents[1] / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        Path(__file__).parents[1] / "src" / "latent_contract" / "sender_update.py",
    ]
    write("dependency_hashes.json", {str(path.relative_to(Path(__file__).parents[1])).replace("\\", "/"): sha256(path)
                                      for path in dependencies})

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.hf_home,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        use_kernels=False,
    ).eval()
    model.config.use_cache = False
    answer_sequences = [tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")]
    if any(len(ids) != 1 for ids in answer_sequences):
        raise RuntimeError("A and B must each be a single native token")
    answer_id_values = [ids[0] for ids in answer_sequences]
    answer_ids = torch.tensor(answer_id_values, device=model.device)

    def render(user_text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def logits(texts: list[str], *, grad: bool) -> torch.Tensor:
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
        if not lengths or max(lengths) > 512:
            raise RuntimeError("empty batch or context exceeds frozen 512-token ceiling")
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(model.device)
        if grad:
            return model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        with torch.no_grad():
            return model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]

    plain = {str(row["id"]): render(str(row["prompt"])) for row in rows}
    plain_eval = {str(row["id"]): render(str(row["prompt"])) for row in evaluation}
    feedback_fields = (
        "immediate_feedback", "delayed_expression_feedback",
        "delayed_transition_feedback", "truthful_feedback",
    )
    feedback_prompts = {
        (str(row["id"]), field): render(hindsight_user_text(str(row["prompt"]), str(row[field])))
        for row in rows for field in feedback_fields
    }

    # Frozen semantic interface qualification before any adapter is installed.
    qualification_rows = []
    for start in range(0, len(evaluation), TRAIN_BATCH):
        batch = evaluation[start:start + TRAIN_BATCH]
        base = logits([plain_eval[str(row["id"])] for row in batch], grad=False)
        for semantic in (0, 1):
            teacher = logits([
                render(hindsight_user_text(str(row["prompt"]), feedback_for_row(row, semantic)))
                for row in batch
            ], grad=False)
            for index, row in enumerate(batch):
                target_index = semantic_letter_index(semantic, int(row["swap"]))
                target_id = answer_id_values[target_index]
                teacher_log = teacher[index].float().log_softmax(-1)
                base_log = base[index].float().log_softmax(-1)
                normalized = teacher_log[answer_ids].softmax(0)
                qualification_rows.append({
                    "id": row["id"],
                    "semantic": semantic,
                    "target_letter": "AB"[target_index],
                    "normalized_target_probability": float(normalized[target_index]),
                    "normalized_correct": int(normalized.argmax()) == target_index,
                    "ab_mass": float(teacher_log[answer_ids].logsumexp(0).exp()),
                    "target_log_probability_shift": float(teacher_log[target_id] - base_log[target_id]),
                })
    qualification_metrics = {
        "n": len(qualification_rows),
        "correct": sum(int(row["normalized_correct"]) for row in qualification_rows),
        "mean_normalized_target_probability": float(np.mean([
            row["normalized_target_probability"] for row in qualification_rows
        ])),
        "min_normalized_target_probability": min(row["normalized_target_probability"] for row in qualification_rows),
        "min_ab_mass": min(row["ab_mass"] for row in qualification_rows),
        "mean_target_log_probability_shift": float(np.mean([
            row["target_log_probability_shift"] for row in qualification_rows
        ])),
    }
    qualification_gates = {
        "all_semantic_choices_correct": qualification_metrics["correct"] == qualification_metrics["n"],
        "mean_normalized_target_probability_at_least_point_90":
            qualification_metrics["mean_normalized_target_probability"] >= .90,
        "min_normalized_target_probability_at_least_point_70":
            qualification_metrics["min_normalized_target_probability"] >= .70,
        "min_full_vocabulary_ab_mass_at_least_point_20": qualification_metrics["min_ab_mass"] >= .20,
    }
    write("qualification.json", {
        "qualified": all(qualification_gates.values()),
        "metrics": qualification_metrics,
        "gates": qualification_gates,
        "rows": qualification_rows,
    })

    def finish(decision: str, payload: dict[str, object]) -> None:
        write("RESULT.json", {
            "decision": decision,
            **payload,
            "runtime": {
                "wall_seconds": time.time() - started,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
            "paper_green_light": False,
        })
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
        print(json.dumps({"decision": decision, **payload}, indent=2), flush=True)

    if not all(qualification_gates.values()):
        finish("STOP_MODEL_INTERFACE_UNQUALIFIED", {"qualification": qualification_metrics})
        return

    lora_modules = install_qwen35_lora(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial = adapter_state(model)
    torch.save(initial, args.root / "initial_adapter.pt")
    write("runtime_setup.json", {
        "lora_modules": lora_modules,
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "answer_token_ids": answer_ids.tolist(),
    })

    def evaluate(name: str) -> dict[str, object]:
        result_rows = []
        with torch.no_grad():
            for start in range(0, len(evaluation), TRAIN_BATCH):
                batch = evaluation[start:start + TRAIN_BATCH]
                output = logits([plain_eval[str(row["id"])] for row in batch], grad=False)
                logp = output.float().log_softmax(-1)
                choice = logp[:, answer_ids].softmax(-1)
                mass = logp[:, answer_ids].logsumexp(-1).exp()
                for index, row in enumerate(batch):
                    semantic_probabilities = [0., 0.]
                    for semantic in (0, 1):
                        letter = semantic_letter_index(semantic, int(row["swap"]))
                        semantic_probabilities[semantic] = float(choice[index, letter])
                    result_rows.append({
                        "id": row["id"],
                        "swap": row["swap"],
                        "semantic_probabilities": semantic_probabilities,
                        "predicted_semantic": int(np.argmax(semantic_probabilities)),
                        "ab_mass": float(mass[index]),
                    })
        write(f"{name}_eval.json", result_rows)
        metrics = {}
        for semantic in (0, 1):
            probabilities = np.asarray([row["semantic_probabilities"][semantic] for row in result_rows])
            metrics[str(semantic)] = {
                "accuracy": float(np.mean([row["predicted_semantic"] == semantic for row in result_rows])),
                "mean_probability": float(np.mean(probabilities)),
                "mean_nll": float(np.mean(-np.log(np.maximum(probabilities, 1e-12)))),
                "min_ab_mass": min(row["ab_mass"] for row in result_rows),
            }
        return metrics

    baseline = evaluate("baseline")
    write("baseline_metrics.json", baseline)
    arm_results: dict[str, object] = {}
    completed: list[str] = []
    max_transition_identity_error = 0.
    active_arm: str | None = None
    active_steps: list[dict[str, object]] = []
    active_optimizer: torch.optim.Optimizer | None = None

    def run_arm(
        name: str,
        mode: str,
        feedback_field: str,
        semantic_field: str | None = None,
        delayed_feedback_field: str | None = None,
    ) -> dict[str, object]:
        nonlocal max_transition_identity_error, active_arm, active_steps, active_optimizer
        load_adapter(model, initial)
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=0.)
        active_arm = name
        active_steps = []
        active_optimizer = optimizer
        steps = []
        for step_index, batch_ids in enumerate(schedule, 1):
            batch = [by_id[row_id] for row_id in batch_ids]
            anchor_mask = torch.tensor([bool(row["anchor"]) for row in batch], device=model.device)
            optimizer.zero_grad(set_to_none=True)
            immediate_teacher = None
            delayed_teacher = None
            if mode in {"raw", "augmented"}:
                immediate_teacher = logits([
                    feedback_prompts[(str(row["id"]), feedback_field)] for row in batch
                ], grad=False)
            if mode in {"anchor_sdpo", "augmented"}:
                teacher_field = delayed_feedback_field or feedback_field
                delayed_teacher = logits([
                    feedback_prompts[(str(row["id"]), teacher_field)]
                    for row in batch if row["anchor"]
                ], grad=False)
            student = logits([plain[str(row["id"])] for row in batch], grad=True)
            detail: dict[str, float] = {}
            if mode == "raw":
                assert immediate_teacher is not None
                immediate_loss = reverse_kl_per_example(student, immediate_teacher)
                loss = immediate_loss.mean()
                detail["immediate_mean"] = float(loss.detach())
                # Transition B=O. Audit the exact augmented/raw identity using
                # the same tensors at every raw-training batch.
                identity = augmented_reverse_kl(
                    student, immediate_teacher, immediate_teacher[anchor_mask], anchor_mask
                )
                error = float(abs(identity.detach() - loss.detach()))
                max_transition_identity_error = max(max_transition_identity_error, error)
                detail["transition_augmented_identity_error"] = error
            elif mode == "anchor_sdpo":
                assert delayed_teacher is not None
                loss = reverse_kl_per_example(student[anchor_mask], delayed_teacher).mean()
                detail["anchor_mean"] = float(loss.detach())
            elif mode == "augmented":
                assert immediate_teacher is not None and delayed_teacher is not None
                immediate_each = reverse_kl_per_example(student, immediate_teacher)
                delayed_each = reverse_kl_per_example(student[anchor_mask], delayed_teacher)
                residual = (delayed_each - immediate_each[anchor_mask]).mean()
                loss = immediate_each.mean() + residual
                detail.update({
                    "immediate_mean": float(immediate_each.mean().detach()),
                    "anchor_residual_mean": float(residual.detach()),
                })
            elif mode == "anchor_sft":
                if semantic_field is None:
                    raise RuntimeError("anchor_sft requires a semantic field")
                anchor_rows = [row for row in batch if row["anchor"]]
                targets = torch.tensor([
                    answer_id_values[semantic_letter_index(int(row[semantic_field]), int(row["swap"]))]
                    for row in anchor_rows
                ], device=model.device)
                loss = -student[anchor_mask].float().log_softmax(-1).gather(1, targets[:, None]).mean()
                detail["anchor_sft_mean"] = float(loss.detach())
            else:
                raise RuntimeError(f"unknown arm mode: {mode}")
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in {name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            steps.append({
                "step": step_index,
                "batch_ids": batch_ids,
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **detail,
            })
            active_steps = steps
            write(f"{name}_steps.partial.json", steps)
        write(f"{name}_steps.json", steps)
        torch.save(adapter_state(model), args.root / f"{name}_adapter.pt")
        torch.save(optimizer.state_dict(), args.root / f"{name}_optimizer.pt")
        metrics = evaluate(name)
        completed.append(name)
        active_arm = None
        active_steps = []
        active_optimizer = None
        print(json.dumps({"arm": name, "metrics": metrics, "wall_seconds": time.time() - started}), flush=True)
        return metrics

    try:
        # Validate acquisition before spending on the causal comparison.
        arm_results["truthful_raw"] = run_arm("truthful_raw", "raw", "truthful_feedback")
        truthful = arm_results["truthful_raw"]["1"]  # type: ignore[index]
        positive_control = (
            truthful["accuracy"] >= .80
            and truthful["mean_probability"] - baseline["1"]["mean_probability"] >= .10
        )
        if not positive_control:
            finish("STOP_TRUTHFUL_SDPO_ACQUISITION_UNQUALIFIED", {
                "baseline": baseline, "arms": arm_results, "completed_arms": completed,
            })
            return

        arm_results["raw_shared"] = run_arm("raw_shared", "raw", "immediate_feedback")
        arm_results["expression_anchor_sdpo"] = run_arm(
            "expression_anchor_sdpo", "anchor_sdpo", "delayed_expression_feedback"
        )
        arm_results["expression_anchor_sft"] = run_arm(
            "expression_anchor_sft", "anchor_sft", "delayed_expression_feedback",
            "delayed_expression_semantic",
        )
        arm_results["expression_augmented"] = run_arm(
            "expression_augmented", "augmented", "immediate_feedback",
            delayed_feedback_field="delayed_expression_feedback",
        )
        arm_results["transition_anchor_sdpo"] = run_arm(
            "transition_anchor_sdpo", "anchor_sdpo", "delayed_transition_feedback"
        )
        arm_results["transition_anchor_sft"] = run_arm(
            "transition_anchor_sft", "anchor_sft", "delayed_transition_feedback",
            "delayed_transition_semantic",
        )

        raw_zero = arm_results["raw_shared"]["0"]  # type: ignore[index]
        expression_augmented = arm_results["expression_augmented"]["1"]  # type: ignore[index]
        expression_baselines = [
            arm_results["expression_anchor_sdpo"]["1"],  # type: ignore[index]
            arm_results["expression_anchor_sft"]["1"],  # type: ignore[index]
        ]
        best_baseline_probability = max(item["mean_probability"] for item in expression_baselines)
        best_baseline_accuracy = max(item["accuracy"] for item in expression_baselines)
        gates = {
            "teacher_interface_qualified": all(qualification_gates.values()),
            "truthful_sdpo_positive_control": positive_control,
            "raw_selects_transition_action": raw_zero["accuracy"] >= .80 and raw_zero["mean_probability"] >= .65,
            "expression_augmented_selects_expression_action": expression_augmented["accuracy"] >= .80,
            "expression_augmented_beats_equal_anchor_baselines": (
                expression_augmented["mean_probability"] >= best_baseline_probability + .05
                and expression_augmented["accuracy"] >= best_baseline_accuracy
            ),
            "transition_augmented_raw_identity": max_transition_identity_error <= 1e-6,
        }
        finish("NEURAL_ANCHOR_G0_QUALIFIED" if all(gates.values()) else "NEURAL_ANCHOR_G0_NOT_QUALIFIED", {
            "baseline": baseline,
            "qualification": qualification_metrics,
            "arms": arm_results,
            "gates": gates,
            "max_transition_augmented_raw_identity_error": max_transition_identity_error,
            "completed_arms": completed,
            "scope": "Single-token synthetic neural SDPO DEV; not human evidence, free-form replication, or paper greenlight.",
        })
    except Exception as exc:
        write("FAILED.json", {
            "error": type(exc).__name__, "message": str(exc), "completed_arms": completed,
            "wall_seconds": time.time() - started,
        })
        if parameters:
            torch.save(adapter_state(model), args.root / "failure_adapter.pt")
        if active_optimizer is not None:
            torch.save(active_optimizer.state_dict(), args.root / "failure_optimizer.pt")
        if active_arm is not None:
            write("failure_progress.json", {"active_arm": active_arm, "steps": active_steps})
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})
        raise


if __name__ == "__main__":
    main()
