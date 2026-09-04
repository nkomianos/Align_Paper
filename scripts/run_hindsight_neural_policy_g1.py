"""GPU runner for the prospectively frozen Hindsight neural policy G1."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from interaction_sprint.hindsight_neural_anchor import (
    HINDSIGHT_BLOCK,
    LORA_ALPHA,
    LORA_RANK,
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_SDPO_COMMIT,
    OFFICIAL_SDPO_REPOSITORY,
    SEED,
    TRAIN_BATCH,
    build_records,
    feedback_for_row,
    hindsight_user_text,
    install_qwen35_lora,
    reverse_kl_per_example,
    semantic_letter_index,
)
from interaction_sprint.hindsight_neural_policy_g1 import (
    POLICY_ANCHORS_PER_ACTION,
    POLICY_ANCHORS_PER_PANEL,
    POLICY_BATCH,
    POLICY_LEARNING_RATE,
    POLICY_PANEL_COUNT,
    POLICY_SEED,
    POLICY_STEPS,
    build_disjoint_policy_panels,
    build_policy_schedules,
    expected_policy_arm_names,
    summarize_evaluation_rows,
    summarize_policy_controls,
    summarize_policy_endpoints,
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
    parser.add_argument("--gradient-root", type=Path, required=True)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8",
        )

    def write_manifest() -> None:
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})

    # Fail closed on the exact verified prerequisite before loading a model.
    gradient_verifier = repository / "scripts" / "verify_hindsight_neural_gradient_g0_v2.py"
    verification = subprocess.run(
        [sys.executable, str(gradient_verifier), "--root", str(args.gradient_root)],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if verification.returncode != 0:
        write("FAILED.json", {
            "stage": "gradient_prerequisite_verification",
            "returncode": verification.returncode,
            "stdout": verification.stdout,
            "stderr": verification.stderr,
        })
        write_manifest()
        raise RuntimeError("gradient prerequisite failed committed verification")
    verifier_receipt = json.loads(verification.stdout)
    gradient_result_path = args.gradient_root / "RESULT.json"
    gradient_manifest_path = args.gradient_root / "MANIFEST.json"
    gradient_result = json.loads(gradient_result_path.read_text(encoding="utf-8"))
    if (
        verifier_receipt.get("verified") is not True
        or verifier_receipt.get("decision") != "NEURAL_GRADIENT_G0_V2_QUALIFIED"
        or gradient_result.get("decision") != "NEURAL_GRADIENT_G0_V2_QUALIFIED"
    ):
        write("FAILED.json", {
            "stage": "gradient_prerequisite_decision",
            "verifier_receipt": verifier_receipt,
            "gradient_decision": gradient_result.get("decision"),
        })
        write_manifest()
        raise RuntimeError("G1 requires a qualified neural-gradient G0 v2")
    prerequisite = {
        "gradient_root_at_launch": str(args.gradient_root.resolve()),
        "manifest_sha256": sha256(gradient_manifest_path),
        "result_sha256": sha256(gradient_result_path),
        "decision": gradient_result["decision"],
        "committed_verifier_receipt": verifier_receipt,
    }
    write("gradient_prerequisite.json", prerequisite)

    rows, evaluation, _ = build_records()
    panels = build_disjoint_policy_panels(rows)
    schedules = build_policy_schedules(rows, panels)
    by_id = {str(row["id"]): row for row in rows}
    spec = {
        "version": "policy-learning-g1-v3-oracle-distance",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "full-vocabulary reverse KL(student(.|x) || stopgrad teacher(.|x,o)) at first answer token",
        "data_seed": SEED,
        "policy_seed": POLICY_SEED,
        "steps_per_arm": POLICY_STEPS,
        "batch": POLICY_BATCH,
        "panel_count": POLICY_PANEL_COUNT,
        "anchors_per_panel": POLICY_ANCHORS_PER_PANEL,
        "anchors_per_action": POLICY_ANCHORS_PER_ACTION,
        "learning_rate": POLICY_LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "trained_arms": 2 + 3 * POLICY_PANEL_COUNT,
        "required_gradient_manifest_sha256": prerequisite["manifest_sha256"],
        "paper_green_light": False,
    }
    write("spec.json", spec)
    write("cases.json", {"train": rows, "evaluation": evaluation})
    write("panels.json", panels)
    write("schedules.json", schedules)
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient_v2.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_policy_g1.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        gradient_verifier,
        Path(__file__),
        repository / "scripts" / "verify_hindsight_neural_policy_g1.py",
        repository / "scripts" / "run_hindsight_neural_policy_g1_remote.sh",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    })

    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required")
    torch.manual_seed(POLICY_SEED)
    torch.cuda.manual_seed_all(POLICY_SEED)
    torch.set_float32_matmul_precision("high")
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
    feedback_prompts = {
        (str(row["id"]), field): render(hindsight_user_text(str(row["prompt"]), str(row[field])))
        for row in rows
        for field in ("immediate_feedback", "delayed_expression_feedback")
    }

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
            "gradient_prerequisite": prerequisite,
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
        write_manifest()
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

    def evaluate(name: str) -> dict[str, float]:
        result_rows = []
        with torch.no_grad():
            for start in range(0, len(evaluation), POLICY_BATCH):
                batch = evaluation[start:start + POLICY_BATCH]
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
        return summarize_evaluation_rows(result_rows)

    endpoint_metrics: dict[str, dict[str, float]] = {"baseline": evaluate("baseline")}
    completed: list[str] = []
    active_arm: str | None = None
    active_steps: list[dict[str, object]] = []
    active_optimizer: torch.optim.Optimizer | None = None

    def run_arm(
        name: str,
        mode: str,
        schedule: list[list[str]],
        anchor_ids: list[str] | None = None,
    ) -> dict[str, float]:
        nonlocal active_arm, active_steps, active_optimizer
        load_adapter(model, initial)
        optimizer = torch.optim.AdamW(parameters, lr=POLICY_LEARNING_RATE, weight_decay=0.)
        active_arm = name
        active_steps = []
        active_optimizer = optimizer
        anchor_rows = [by_id[row_id] for row_id in (anchor_ids or [])]
        if mode in {"anchor_sdpo", "anchor_sft", "augmented"}:
            if len(anchor_rows) != POLICY_ANCHORS_PER_PANEL:
                raise RuntimeError(f"invalid anchor count in {name}")
        steps = []
        for step_index, batch_ids in enumerate(schedule, 1):
            batch = [by_id[row_id] for row_id in batch_ids]
            optimizer.zero_grad(set_to_none=True)
            detail: dict[str, float] = {}
            if mode == "raw":
                teacher = logits([
                    feedback_prompts[(str(row["id"]), "immediate_feedback")] for row in batch
                ], grad=False)
                student = logits([plain[str(row["id"])] for row in batch], grad=True)
                loss = reverse_kl_per_example(student, teacher).mean()
                detail["raw_mean"] = float(loss.detach())
            elif mode == "oracle":
                teacher = logits([
                    feedback_prompts[(str(row["id"]), "delayed_expression_feedback")] for row in batch
                ], grad=False)
                student = logits([plain[str(row["id"])] for row in batch], grad=True)
                loss = reverse_kl_per_example(student, teacher).mean()
                detail["oracle_mean"] = float(loss.detach())
            elif mode == "anchor_sdpo":
                teacher = logits([
                    feedback_prompts[(str(row["id"]), "delayed_expression_feedback")]
                    for row in anchor_rows
                ], grad=False)
                student = logits([plain[str(row["id"])] for row in anchor_rows], grad=True)
                loss = reverse_kl_per_example(student, teacher).mean()
                detail["anchor_sdpo_mean"] = float(loss.detach())
            elif mode == "anchor_sft":
                student = logits([plain[str(row["id"])] for row in anchor_rows], grad=True)
                targets = torch.tensor([
                    answer_id_values[semantic_letter_index(
                        int(row["delayed_expression_semantic"]), int(row["swap"]),
                    )]
                    for row in anchor_rows
                ], device=model.device)
                loss = -student.float().log_softmax(-1).gather(1, targets[:, None]).mean()
                detail["anchor_sft_mean"] = float(loss.detach())
            elif mode == "augmented":
                immediate_teacher = logits([
                    feedback_prompts[(str(row["id"]), "immediate_feedback")] for row in batch
                ], grad=False)
                delayed_teacher = logits([
                    feedback_prompts[(str(row["id"]), "delayed_expression_feedback")]
                    for row in anchor_rows
                ], grad=False)
                anchor_immediate_teacher = logits([
                    feedback_prompts[(str(row["id"]), "immediate_feedback")]
                    for row in anchor_rows
                ], grad=False)
                population_student = logits([plain[str(row["id"])] for row in batch], grad=True)
                anchor_student = logits([plain[str(row["id"])] for row in anchor_rows], grad=True)
                population_immediate_each = reverse_kl_per_example(
                    population_student, immediate_teacher,
                )
                anchor_delayed_each = reverse_kl_per_example(anchor_student, delayed_teacher)
                anchor_immediate_each = reverse_kl_per_example(
                    anchor_student, anchor_immediate_teacher,
                )
                residual = (anchor_delayed_each - anchor_immediate_each).mean()
                loss = population_immediate_each.mean() + residual
                detail.update({
                    "population_immediate_mean": float(population_immediate_each.mean().detach()),
                    "anchor_residual_mean": float(residual.detach()),
                })
            else:
                raise RuntimeError(f"unknown arm mode: {mode}")
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in {name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            steps.append({
                "step": step_index,
                "population_batch_ids": batch_ids,
                "anchor_ids": list(anchor_ids or []),
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
        write("training_progress.json", {"completed": completed})
        active_arm = None
        active_steps = []
        active_optimizer = None
        print(json.dumps({"arm": name, "metrics": metrics, "wall_seconds": time.time() - started}), flush=True)
        return metrics

    try:
        global_schedule = schedules["global"]
        endpoint_metrics["raw_immediate"] = run_arm(
            "raw_immediate", "raw", global_schedule,
        )
        endpoint_metrics["oracle_delayed"] = run_arm(
            "oracle_delayed", "oracle", global_schedule,
        )
        controls = summarize_policy_controls(endpoint_metrics)
        if controls["decision"] != "POLICY_ACQUISITION_QUALIFIED":
            finish(controls["decision"], {
                "qualification": qualification_metrics,
                "endpoint_metrics": endpoint_metrics,
                "completed_arms": completed,
                "gates": controls["gates"],
                "aggregate": controls["aggregate"],
                "scope": (
                    "Raw/full-oracle policy acquisition control stopped before sparse-panel "
                    "training; this is an assay stop, not a correction result."
                ),
            })
            return
        for panel_index in range(POLICY_PANEL_COUNT):
            panel_schedule = schedules["panels"][str(panel_index)]
            anchor_ids = panel_schedule["anchor_ids"]
            for suffix, mode in (
                ("anchor_sdpo", "anchor_sdpo"),
                ("anchor_sft", "anchor_sft"),
                ("augmented", "augmented"),
            ):
                name = f"panel_{panel_index:02d}_{suffix}"
                endpoint_metrics[name] = run_arm(
                    name, mode, global_schedule, anchor_ids,
                )
        if list(endpoint_metrics) != expected_policy_arm_names():
            raise RuntimeError("arm completion order differs from frozen design")
        summary = summarize_policy_endpoints(endpoint_metrics)
        finish(summary["decision"], {
            "qualification": qualification_metrics,
            "endpoint_metrics": endpoint_metrics,
            "completed_arms": completed,
            "gates": summary["gates"],
            "aggregate": summary["aggregate"],
            "panels": summary["panels"],
            "scope": summary["scope"],
        })
    except Exception as exc:
        write("FAILED.json", {
            "error": type(exc).__name__,
            "message": str(exc),
            "completed_arms": completed,
            "wall_seconds": time.time() - started,
        })
        if parameters:
            torch.save(adapter_state(model), args.root / "failure_adapter.pt")
        if active_optimizer is not None:
            torch.save(active_optimizer.state_dict(), args.root / "failure_optimizer.pt")
        if active_arm is not None:
            write("failure_progress.json", {"active_arm": active_arm, "steps": active_steps})
        write_manifest()
        raise


if __name__ == "__main__":
    main()
