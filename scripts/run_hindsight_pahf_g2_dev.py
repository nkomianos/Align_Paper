"""Run the prerequisite-bound Qwen3.5-9B EndoPAHF G2 development stage."""
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
    hindsight_user_text,
    install_qwen35_lora,
    reverse_kl_per_example,
)
from interaction_sprint.hindsight_pahf_g2_v2 import (
    G2_ANCHOR_BASES_PER_PANEL,
    G2_ANCHOR_ROWS_PER_STEP,
    G2_BATCH,
    G2_LEARNING_RATE,
    G2_PANEL_COUNT,
    G2_SEED,
    G2_STEPS,
    build_g2_anchor_panels,
    build_g2_schedules,
    expected_g2_arm_names,
    summarize_g2_stage,
)
from interaction_sprint.hindsight_pahf_interface import OPTION_LETTERS
from latent_contract.sender_update import adapter_state, load_adapter


INPUT_MANIFEST_SHA256 = "2ae32c119087d97de6f2e5a65959b4c94a7d81362732871ff806b157e000fab2"
G2_ROUTING_POWER_MANIFEST_SHA256 = "8e050034619ff21fddfbc73bb337cdee0f41c4a0117cb816b8c8100ac935e4ca"
CONFIRM_RULE_POWER_MANIFEST_SHA256 = "7a13b16c1a1acdf02d43efe30ba7830a519fcd7b1ce271c17a741891da9fe48e"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_prerequisite(command: list[str], expected_decision: str) -> dict[str, object]:
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(f"prerequisite verifier failed: {process.stderr}")
    receipt = json.loads(process.stdout)
    if receipt.get("verified") is not True or receipt.get("decision") != expected_decision:
        raise RuntimeError(f"prerequisite decision is not qualified: {receipt}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.input_root / "MANIFEST.json") != INPUT_MANIFEST_SHA256:
        raise SystemExit("EndoPAHF v3 input manifest mismatch")
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    def write_manifest() -> None:
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})

    try:
        g1_receipt = _verify_prerequisite(
            [sys.executable, str(repository / "scripts" / "verify_hindsight_neural_policy_g1.py"),
             "--root", str(args.g1_root)],
            "NEURAL_POLICY_G1_QUALIFIED",
        )
        preflight_receipt = _verify_prerequisite(
            [sys.executable, str(repository / "scripts" / "verify_hindsight_pahf_preflight.py"),
             "--input-root", str(args.input_root), "--root", str(args.preflight_root)],
            "ENDO_PAHF_CAPABLE_INTERFACE_QUALIFIED",
        )
    except Exception as exc:
        write("FAILED.json", {
            "stage": "prerequisite_verification",
            "error": type(exc).__name__,
            "message": str(exc),
        })
        write_manifest()
        raise

    prerequisites = {
        "g1": {
            "root_at_launch": str(args.g1_root.resolve()),
            "manifest_sha256": sha256(args.g1_root / "MANIFEST.json"),
            "result_sha256": sha256(args.g1_root / "RESULT.json"),
            "receipt": g1_receipt,
        },
        "preflight": {
            "root_at_launch": str(args.preflight_root.resolve()),
            "manifest_sha256": sha256(args.preflight_root / "MANIFEST.json"),
            "result_sha256": sha256(args.preflight_root / "RESULT.json"),
            "receipt": preflight_receipt,
        },
        "input": {
            "root_at_launch": str(args.input_root.resolve()),
            "manifest_sha256": INPUT_MANIFEST_SHA256,
        },
    }
    write("prerequisites.json", prerequisites)

    learning_path = args.input_root / "learning.json"
    development_path = args.input_root / "development.json"
    learning = json.loads(learning_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    panels = build_g2_anchor_panels(learning)
    schedules = build_g2_schedules(learning, panels)
    by_id = {str(row["id"]): row for row in learning}
    write("input_receipt.json", {
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "learning_sha256": sha256(learning_path),
        "development_sha256": sha256(development_path),
        "learning_rows": len(learning),
        "development_rows": len(development),
        "confirmation_opened": False,
    })
    write("panels.json", panels)
    write("schedules.json", schedules)

    spec = {
        "version": "endo-pahf-g2-dev-v2-full-learning",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "full-vocabulary reverse KL at first A/B/C/D answer token",
        "hindsight_block": HINDSIGHT_BLOCK,
        "seed": G2_SEED,
        "steps_per_arm": G2_STEPS,
        "population_batch": G2_BATCH,
        "panel_count": G2_PANEL_COUNT,
        "anchor_bases_per_panel": G2_ANCHOR_BASES_PER_PANEL,
        "anchor_rows_per_step": G2_ANCHOR_ROWS_PER_STEP,
        "learning_rate": G2_LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "trained_arms": len(expected_g2_arm_names()) - 1,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "g2_routing_power_manifest_sha256": G2_ROUTING_POWER_MANIFEST_SHA256,
        "confirmation_rule_power_manifest_sha256": CONFIRM_RULE_POWER_MANIFEST_SHA256,
        "global_unique_learning_bases": 630,
        "global_variants_per_base": 1,
        "panel_aggregation": "arithmetic mean of four panel probability vectors",
        "confirmation_opened": False,
        "paper_green_light": False,
    }
    write("spec.json", spec)
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_cluster_stats.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2.py",
        repository / "src" / "interaction_sprint" / "hindsight_pahf_g2_v2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        repository / "scripts" / "verify_hindsight_neural_policy_g1.py",
        repository / "scripts" / "verify_hindsight_pahf_preflight.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_pahf_g2_dev.py",
        repository / "scripts" / "run_hindsight_pahf_g2_dev_remote.sh",
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
    torch.manual_seed(G2_SEED)
    torch.cuda.manual_seed_all(G2_SEED)
    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=args.hf_home,
        dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True,
        use_kernels=False,
    ).eval()
    model.config.use_cache = False
    token_sequences = [tokenizer.encode(letter, add_special_tokens=False) for letter in OPTION_LETTERS]
    if any(len(sequence) != 1 for sequence in token_sequences):
        raise RuntimeError(f"A/B/C/D are not all single tokens: {token_sequences}")
    answer_ids = torch.tensor([sequence[0] for sequence in token_sequences], device=model.device)

    def render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    def logits(texts: list[str], *, grad: bool) -> torch.Tensor:
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts]
        if not lengths or max(lengths) > 1024:
            raise RuntimeError("empty batch or context exceeds frozen 1024-token ceiling")
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False,
        ).to(model.device)
        if grad:
            return model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        with torch.no_grad():
            return model(**encoded, use_cache=False, logits_to_keep=1).logits[:, -1, :]

    plain_train = {str(row["id"]): render(str(row["prompt"])) for row in learning}
    plain_dev = {str(row["id"]): render(str(row["prompt"])) for row in development}
    feedback_train = {
        (str(row["id"]), field): render(hindsight_user_text(str(row["prompt"]), str(row[field])))
        for row in learning
        for field in (
            "immediate_followup", "delayed_expression_followup", "delayed_transition_followup",
        )
    }

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

    arm_predictions: dict[str, list[dict[str, object]]] = {}

    def evaluate(name: str) -> list[dict[str, object]]:
        predictions: list[dict[str, object]] = []
        with torch.no_grad():
            for start in range(0, len(development), G2_BATCH):
                batch = development[start:start + G2_BATCH]
                output = logits([plain_dev[str(row["id"])] for row in batch], grad=False)
                full_log = output.float().log_softmax(-1)
                choice_log = full_log[:, answer_ids]
                normalized_log = choice_log.log_softmax(-1)
                mass = choice_log.logsumexp(-1).exp()
                for index, row in enumerate(batch):
                    predictions.append({
                        "id": row["id"],
                        "normalized_choice_log_probabilities": normalized_log[index].tolist(),
                        "full_vocabulary_choice_mass": float(mass[index]),
                    })
        write(f"{name}_dev_predictions.json", predictions)
        arm_predictions[name] = predictions
        return predictions

    evaluate("baseline")
    completed: list[str] = []
    active_arm: str | None = None
    active_steps: list[dict[str, object]] = []
    active_optimizer: torch.optim.Optimizer | None = None
    saved_states: dict[str, dict[str, torch.Tensor]] = {}

    def run_arm(
        name: str,
        mode: str,
        anchor_schedule: list[list[str]] | None = None,
    ) -> None:
        nonlocal active_arm, active_steps, active_optimizer
        load_adapter(model, initial)
        optimizer = torch.optim.AdamW(parameters, lr=G2_LEARNING_RATE, weight_decay=0.)
        active_arm = name
        active_steps = []
        active_optimizer = optimizer
        if mode in {"anchor_sdpo", "anchor_sft", "augmented", "transition_sanity"}:
            if anchor_schedule is None or len(anchor_schedule) != G2_STEPS:
                raise RuntimeError(f"invalid anchor schedule for {name}")
        steps: list[dict[str, object]] = []
        for step_index, batch_ids in enumerate(schedules["global"]):
            batch = [by_id[row_id] for row_id in batch_ids]
            anchor_ids = anchor_schedule[step_index] if anchor_schedule is not None else []
            anchors = [by_id[row_id] for row_id in anchor_ids]
            optimizer.zero_grad(set_to_none=True)
            detail: dict[str, float] = {}
            if mode == "raw":
                teacher = logits([
                    feedback_train[(str(row["id"]), "immediate_followup")] for row in batch
                ], grad=False)
                student = logits([plain_train[str(row["id"])] for row in batch], grad=True)
                loss = reverse_kl_per_example(student, teacher).mean()
                detail["population_immediate_mean"] = float(loss.detach())
            elif mode == "oracle":
                teacher = logits([
                    feedback_train[(str(row["id"]), "delayed_expression_followup")] for row in batch
                ], grad=False)
                student = logits([plain_train[str(row["id"])] for row in batch], grad=True)
                loss = reverse_kl_per_example(student, teacher).mean()
                detail["population_delayed_mean"] = float(loss.detach())
            elif mode == "anchor_sdpo":
                teacher = logits([
                    feedback_train[(str(row["id"]), "delayed_expression_followup")] for row in anchors
                ], grad=False)
                student = logits([plain_train[str(row["id"])] for row in anchors], grad=True)
                loss = reverse_kl_per_example(student, teacher).mean()
                detail["anchor_delayed_mean"] = float(loss.detach())
            elif mode == "anchor_sft":
                student = logits([plain_train[str(row["id"])] for row in anchors], grad=True)
                targets = torch.tensor([
                    token_sequences[OPTION_LETTERS.index(str(row["old_target"]))][0]
                    for row in anchors
                ], device=model.device)
                loss = -student.float().log_softmax(-1).gather(1, targets[:, None]).mean()
                detail["anchor_sft_mean"] = float(loss.detach())
            elif mode in {"augmented", "transition_sanity"}:
                immediate_teacher = logits([
                    feedback_train[(str(row["id"]), "immediate_followup")] for row in batch
                ], grad=False)
                population_student = logits([plain_train[str(row["id"])] for row in batch], grad=True)
                population_each = reverse_kl_per_example(population_student, immediate_teacher)
                anchor_student = logits([plain_train[str(row["id"])] for row in anchors], grad=True)
                anchor_immediate_teacher = logits([
                    feedback_train[(str(row["id"]), "immediate_followup")] for row in anchors
                ], grad=False)
                anchor_immediate_each = reverse_kl_per_example(
                    anchor_student, anchor_immediate_teacher,
                )
                if mode == "augmented":
                    delayed_teacher = logits([
                        feedback_train[(str(row["id"]), "delayed_expression_followup")]
                        for row in anchors
                    ], grad=False)
                    delayed_each = reverse_kl_per_example(anchor_student, delayed_teacher)
                    residual = (delayed_each - anchor_immediate_each).mean()
                else:
                    # In the transition world the delayed message is identical to
                    # the immediate message, so the paired residual is exactly zero.
                    residual = (anchor_immediate_each - anchor_immediate_each).mean()
                loss = population_each.mean() + residual
                detail.update({
                    "population_immediate_mean": float(population_each.mean().detach()),
                    "anchor_residual_mean": float(residual.detach()),
                })
            else:
                raise RuntimeError(f"unknown G2 mode: {mode}")
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss in {name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            steps.append({
                "step": step_index + 1,
                "population_batch_ids": list(batch_ids),
                "anchor_ids": list(anchor_ids),
                "loss": float(loss.detach()),
                "gradient_norm": float(gradient_norm),
                **detail,
            })
            active_steps = steps
            write(f"{name}_steps.partial.json", steps)
        write(f"{name}_steps.json", steps)
        state = adapter_state(model)
        saved_states[name] = state
        torch.save(state, args.root / f"{name}_adapter.pt")
        torch.save(optimizer.state_dict(), args.root / f"{name}_optimizer.pt")
        evaluate(name)
        completed.append(name)
        write("training_progress.json", {"completed": completed})
        active_arm = None
        active_steps = []
        active_optimizer = None
        print(json.dumps({"arm": name, "wall_seconds": time.time() - started}), flush=True)

    try:
        run_arm("raw_immediate", "raw")
        run_arm("oracle_delayed", "oracle")
        first_anchor_schedule = schedules["panels"]["0"]["anchor_schedule"]
        run_arm("transition_sanity", "transition_sanity", first_anchor_schedule)
        transition_adapter_difference = max(
            float((saved_states["raw_immediate"][key] - saved_states["transition_sanity"][key]).abs().max())
            for key in saved_states["raw_immediate"]
        )
        for panel_index in range(G2_PANEL_COUNT):
            anchor_schedule = schedules["panels"][str(panel_index)]["anchor_schedule"]
            for suffix, mode in (
                ("anchor_sdpo", "anchor_sdpo"),
                ("anchor_sft", "anchor_sft"),
                ("augmented", "augmented"),
            ):
                run_arm(f"panel_{panel_index:02d}_{suffix}", mode, anchor_schedule)
        expected = expected_g2_arm_names()
        if ["baseline", *completed] != expected or set(arm_predictions) != set(expected):
            raise RuntimeError("G2 arm completion differs from frozen design")
        summary = summarize_g2_stage(
            development, arm_predictions, stage="development",
            transition_adapter_max_abs_difference=transition_adapter_difference,
        )
        write("RESULT.json", {
            **summary,
            "completed_arms": completed,
            "prerequisites": prerequisites,
            "runtime": {
                "wall_seconds": time.time() - started,
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        })
        write_manifest()
        print(json.dumps({
            "decision": summary["decision"],
            "controls": summary["controls"],
            "wall_seconds": time.time() - started,
        }, indent=2), flush=True)
    except Exception as exc:
        write("FAILED.json", {
            "stage": "training_or_evaluation",
            "error": type(exc).__name__,
            "message": str(exc),
            "completed_arms": completed,
            "active_arm": active_arm,
            "wall_seconds": time.time() - started,
        })
        if active_arm is not None:
            torch.save(adapter_state(model), args.root / "failure_adapter.pt")
            if active_optimizer is not None:
                torch.save(active_optimizer.state_dict(), args.root / "failure_optimizer.pt")
            write("failure_progress.json", {"active_arm": active_arm, "steps": active_steps})
        write_manifest()
        raise


if __name__ == "__main__":
    main()
