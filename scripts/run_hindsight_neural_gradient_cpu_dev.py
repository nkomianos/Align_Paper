"""Local CPU rehearsal of the nested-budget Hindsight gradient apparatus."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from interaction_sprint.hindsight_neural_anchor import (
    HINDSIGHT_BLOCK,
    LORA_ALPHA,
    LORA_RANK,
    OFFICIAL_SDPO_COMMIT,
    OFFICIAL_SDPO_REPOSITORY,
    SEED,
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


MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BATCH = 8
THREADS = 4


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
        (args.root / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

    def manifest() -> None:
        files = [path for path in args.root.iterdir() if path.is_file() and path.name != "MANIFEST.json"]
        write("MANIFEST.json", {path.name: sha256(path) for path in sorted(files)})

    rows, evaluation, _ = build_records()
    panels = build_nested_anchor_panels(rows)
    by_id = {str(row["id"]): row for row in rows}
    write("spec.json", {
        "scope": "CPU_REHEARSAL_ONLY_NOT_NEURAL_G0",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "official_sdpo_repository": OFFICIAL_SDPO_REPOSITORY,
        "official_sdpo_commit": OFFICIAL_SDPO_COMMIT,
        "seed": SEED,
        "batch": BATCH,
        "threads": THREADS,
        "anchor_budgets": list(ANCHOR_BUDGETS),
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "hindsight_block": HINDSIGHT_BLOCK,
        "gradient_storage_dtype": "bfloat16",
        "paper_green_light": False,
    })
    write("cases.json", {"train": rows, "evaluation": evaluation})
    write("panels.json", {str(budget): value for budget, value in panels.items()})
    sources = [
        repository / "src" / "interaction_sprint" / "hindsight_neural_anchor.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient.py",
        repository / "src" / "interaction_sprint" / "hindsight_neural_gradient_v2.py",
        repository / "src" / "latent_contract" / "sender_update.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_neural_gradient_cpu_dev.py",
    ]
    write("source_hashes.json", {
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path) for path in sources
    })

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.hf_home,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=args.hf_home,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).eval()
    model.config.use_cache = False
    answer_sequences = [tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")]
    if any(len(ids) != 1 for ids in answer_sequences):
        raise RuntimeError("A and B must each be a single native token")
    answer_values = [ids[0] for ids in answer_sequences]
    answer_ids = torch.tensor(answer_values)

    def render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def logits(texts: list[str], *, grad: bool) -> torch.Tensor:
        encoded = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        if encoded["input_ids"].shape[1] > 512:
            raise RuntimeError("context exceeds rehearsal ceiling")
        if grad:
            return model(**encoded, use_cache=False).logits[:, -1, :]
        with torch.no_grad():
            return model(**encoded, use_cache=False).logits[:, -1, :]

    qualification_rows = []
    for start in range(0, len(evaluation), BATCH):
        batch = evaluation[start:start + BATCH]
        base = logits([render(str(row["prompt"])) for row in batch], grad=False)
        for semantic in (0, 1):
            teacher = logits([
                render(hindsight_user_text(str(row["prompt"]), feedback_for_row(row, semantic)))
                for row in batch
            ], grad=False)
            for index, row in enumerate(batch):
                target_index = semantic_letter_index(semantic, int(row["swap"]))
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
                    "target_log_probability_shift": float(
                        teacher_log[answer_values[target_index]] - base_log[answer_values[target_index]]
                    ),
                })
    qmetrics = {
        "n": len(qualification_rows),
        "correct": sum(int(row["normalized_correct"]) for row in qualification_rows),
        "mean_normalized_target_probability": float(np.mean([
            row["normalized_target_probability"] for row in qualification_rows
        ])),
        "min_normalized_target_probability": min(row["normalized_target_probability"] for row in qualification_rows),
        "min_ab_mass": min(row["ab_mass"] for row in qualification_rows),
    }
    qgates = {
        "all_semantic_choices_correct": qmetrics["correct"] == qmetrics["n"],
        "mean_normalized_target_probability_at_least_point_90":
            qmetrics["mean_normalized_target_probability"] >= .90,
        "min_normalized_target_probability_at_least_point_70":
            qmetrics["min_normalized_target_probability"] >= .70,
        "min_full_vocabulary_ab_mass_at_least_point_20": qmetrics["min_ab_mass"] >= .20,
    }
    write("qualification.json", {
        "qualified": all(qgates.values()), "metrics": qmetrics, "gates": qgates,
        "rows": qualification_rows,
    })
    if not all(qgates.values()):
        write("RESULT.json", {
            "decision": "CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED",
            "qualification": qmetrics,
            "wall_seconds": time.time() - started,
            "paper_green_light": False,
        })
        manifest()
        print(json.dumps({"decision": "CPU_REHEARSAL_ONLY_INTERFACE_UNQUALIFIED", "qualification": qmetrics}, indent=2))
        return

    modules = install_qwen35_lora(model)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameters = [parameter for _, parameter in named]
    torch.save(adapter_state(model), args.root / "initial_adapter.pt")
    write("parameter_layout.json", {
        "names": [name for name, _ in named],
        "shapes": [list(parameter.shape) for parameter in parameters],
        "numels": [parameter.numel() for parameter in parameters],
        "total": sum(parameter.numel() for parameter in parameters),
        "lora_modules": modules,
        "answer_token_ids": answer_values,
    })
    plain = {str(row["id"]): render(str(row["prompt"])) for row in rows}
    feedback = {
        (str(row["id"]), field): render(hindsight_user_text(str(row["prompt"]), str(row[field])))
        for row in rows for field in ("immediate_feedback", "delayed_expression_feedback")
    }
    completed: list[str] = []

    def gradient(name: str, selected: list[dict[str, object]], field: str) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        for start in range(0, len(selected), BATCH):
            batch = selected[start:start + BATCH]
            teacher = logits([feedback[(str(row["id"]), field)] for row in batch], grad=False)
            student = logits([plain[str(row["id"])] for row in batch], grad=True)
            (reverse_kl_per_example(student, teacher).sum() / len(selected)).backward()
        vector = torch.cat([
            torch.zeros(parameter.numel()) if parameter.grad is None
            else parameter.grad.detach().float().reshape(-1)
            for parameter in parameters
        ]).to(torch.bfloat16)
        torch.save(vector, args.root / f"gradient_{name}.pt")
        completed.append(name)
        write("gradient_progress.json", {"completed": completed})
        print(json.dumps({
            "gradient": name,
            "norm": float(torch.linalg.vector_norm(vector.float())),
            "wall_seconds": time.time() - started,
        }), flush=True)
        return vector

    vectors = {
        "oracle_delayed": gradient("oracle_delayed", rows, "delayed_expression_feedback"),
        "raw_immediate": gradient("raw_immediate", rows, "immediate_feedback"),
    }
    for budget in ANCHOR_BUDGETS:
        for panel_index, panel in enumerate(panels[budget]):
            selected = [by_id[row_id] for row_id in panel]
            for short, field in (
                ("delayed", "delayed_expression_feedback"),
                ("immediate", "immediate_feedback"),
            ):
                name = f"budget_{budget:02d}_panel_{panel_index:02d}_{short}"
                vectors[name] = gradient(name, selected, field)
    summary = summarize_nested_gradients(vectors)
    write("RESULT.json", {
        "decision": f"CPU_REHEARSAL_ONLY_{summary['decision']}",
        "qualification": qmetrics,
        "gates": summary["gates"],
        "budgets": summary["budgets"],
        "completed_vectors": completed,
        "scope": summary["scope"],
        "runtime": {
            "wall_seconds": time.time() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "model_class": model.__class__.__name__,
        },
        "paper_green_light": False,
    })
    manifest()
    print(json.dumps({
        "decision": f"CPU_REHEARSAL_ONLY_{summary['decision']}",
        "gates": summary["gates"], "budgets": summary["budgets"],
    }, indent=2))


if __name__ == "__main__":
    main()
