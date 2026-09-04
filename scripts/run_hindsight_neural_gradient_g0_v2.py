"""GPU runner for the prospectively corrected nested-budget gradient G0."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
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
from interaction_sprint.hindsight_neural_gradient_v2 import (
    ANCHOR_BUDGETS,
    build_nested_anchor_panels,
    summarize_nested_gradients,
)
from latent_contract.sender_update import adapter_state


POWER_AUDIT_MANIFEST_SHA256 = "45c6910723b7a816e0e2c183ac5802b48c848af404c7eb53dc7e9512142a3a2b"


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
    repository = Path(__file__).parents[1]

    def write(name: str, value: object) -> None:
        (args.root / name).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8",
        )

    def write_manifest() -> None:
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})

    rows, evaluation, _ = build_records()
    panels = build_nested_anchor_panels(rows)
    by_id = {str(row["id"]): row for row in rows}
    spec = {
        "version": "nested-budget-v2",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "objective": "initial-adapter nested-budget full-vocabulary first-token reverse-KL gradient estimation",
        "seed": SEED,
        "batch": TRAIN_BATCH,
        "panel_count": len(next(iter(panels.values()))),
        "anchor_budgets": list(ANCHOR_BUDGETS),
        "anchors_per_action": {str(budget): budget // 2 for budget in ANCHOR_BUDGETS},
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "gradient_storage_dtype": "bfloat16",
        "power_audit_manifest_sha256": POWER_AUDIT_MANIFEST_SHA256,
        "fixed_absolute_cosine_gain_is_diagnostic_only": True,
        "paper_green_light": False,
    }
    write("spec.json", spec)
    write("cases.json", {"train": rows, "evaluation": evaluation})
    write("panels.json", {str(budget): value for budget, value in panels.items()})
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient_v2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_neural_gradient_g0_v2.py",
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
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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

    plain_eval = {str(row["id"]): render(str(row["prompt"])) for row in evaluation}
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
        write_manifest()
        print(json.dumps({"decision": decision, **payload}, indent=2), flush=True)

    if not all(qualification_gates.values()):
        finish("STOP_MODEL_INTERFACE_UNQUALIFIED", {"qualification": qualification_metrics})
        return

    lora_modules = install_qwen35_lora(model)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named_parameters]
    torch.save(adapter_state(model), args.root / "initial_adapter.pt")
    write("parameter_layout.json", {
        "names": [name for name, _ in named_parameters],
        "shapes": [list(parameter.shape) for _, parameter in named_parameters],
        "numels": [parameter.numel() for parameter in parameters],
        "total": sum(parameter.numel() for parameter in parameters),
        "lora_modules": lora_modules,
        "answer_token_ids": answer_id_values,
    })

    plain = {str(row["id"]): render(str(row["prompt"])) for row in rows}
    feedback_prompts = {
        (str(row["id"]), field): render(hindsight_user_text(str(row["prompt"]), str(row[field])))
        for row in rows for field in ("immediate_feedback", "delayed_expression_feedback")
    }
    completed_vectors: list[str] = []

    def gradient_vector(name: str, selected: list[dict[str, object]], feedback_field: str) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        total = len(selected)
        for start in range(0, total, TRAIN_BATCH):
            batch = selected[start:start + TRAIN_BATCH]
            teacher = logits([
                feedback_prompts[(str(row["id"]), feedback_field)] for row in batch
            ], grad=False)
            student = logits([plain[str(row["id"])] for row in batch], grad=True)
            loss = reverse_kl_per_example(student, teacher).sum() / total
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss while computing {name}")
            loss.backward()
        pieces = []
        for parameter in parameters:
            gradient = parameter.grad
            pieces.append(
                torch.zeros(parameter.numel(), dtype=torch.float32)
                if gradient is None else gradient.detach().float().cpu().reshape(-1)
            )
        vector = torch.cat(pieces).to(torch.bfloat16)
        if not bool(torch.isfinite(vector.float()).all()):
            raise RuntimeError(f"nonfinite gradient vector: {name}")
        torch.save(vector, args.root / f"gradient_{name}.pt")
        completed_vectors.append(name)
        write("gradient_progress.json", {"completed": completed_vectors})
        print(json.dumps({
            "gradient": name,
            "norm": float(torch.linalg.vector_norm(vector.float())),
            "wall_seconds": time.time() - started,
        }), flush=True)
        return vector

    try:
        vectors = {
            "oracle_delayed": gradient_vector("oracle_delayed", rows, "delayed_expression_feedback"),
            "raw_immediate": gradient_vector("raw_immediate", rows, "immediate_feedback"),
        }
        for budget in ANCHOR_BUDGETS:
            for panel_index, panel in enumerate(panels[budget]):
                panel_rows = [by_id[row_id] for row_id in panel]
                for field, feedback_field in (
                    ("delayed", "delayed_expression_feedback"),
                    ("immediate", "immediate_feedback"),
                ):
                    name = f"budget_{budget:02d}_panel_{panel_index:02d}_{field}"
                    vectors[name] = gradient_vector(name, panel_rows, feedback_field)
        summary = summarize_nested_gradients(vectors)
        finish(summary["decision"], {
            "qualification": qualification_metrics,
            "gates": summary["gates"],
            "budgets": summary["budgets"],
            "completed_vectors": completed_vectors,
            "scope": summary["scope"],
        })
    except Exception as exc:
        write("FAILED.json", {
            "error": type(exc).__name__,
            "message": str(exc),
            "completed_vectors": completed_vectors,
            "wall_seconds": time.time() - started,
        })
        write_manifest()
        raise


if __name__ == "__main__":
    main()
