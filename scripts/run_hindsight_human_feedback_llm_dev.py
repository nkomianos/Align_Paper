"""GPU runner for the frozen Qwen3.5 PUPPET DEV reader audit.

Raw PUPPET text stays only in memory and the private input file.  Evidence stores
hashes and numeric ratings, never prompts, completions, or participant IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
from pathlib import Path
import time

import numpy as np

from interaction_sprint.hindsight_human_feedback import (
    ARMS,
    DATA_SHA256,
    cluster_bootstrap_gain,
    metrics,
    records_from_rows,
)
from interaction_sprint.hindsight_human_feedback_llm import (
    MAX_CONTEXT_TOKENS,
    MAX_NEW_TOKENS,
    MODEL_ID,
    MODEL_REVISION,
    build_rating_messages,
    completion_receipt,
    qualification_cases,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(hf_home: Path):
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=hf_home, use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=hf_home,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": torch.cuda.current_device()},
        low_cpu_mem_usage=True,
        use_kernels=False,
    ).eval()
    model.config.use_cache = True
    return torch, tokenizer, model


def render(tokenizer, messages) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate_batches(torch, tokenizer, model, prompts: list[str], batch_size: int):
    receipts = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in batch]
        if max(lengths) + MAX_NEW_TOKENS > MAX_CONTEXT_TOKENS:
            raise ValueError("context ceiling exceeded; no truncation allowed")
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        for row in output:
            completion = tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            receipts.append(completion_receipt(completion))
    return receipts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("positive batch size required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError("dataset checksum mismatch")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    records, splits = records_from_rows(rows)
    dev = [record for record in records if record.query_sha256 in splits["dev"]]
    if len(dev) != 72 or len({record.query_sha256 for record in dev}) != 7:
        raise ValueError("frozen DEV cohort changed")

    repository = Path(__file__).parents[1]
    spec = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dataset_sha256": DATA_SHA256,
        "cohort": "attention-pass strict-alternation non-personalized C3/C4/C6 with six USER turns",
        "split": "frozen seven-query DEV only",
        "depths": [3, 6],
        "arms": list(ARMS),
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "thinking": False,
        "training": False,
        "raw_text_persisted": False,
        "paper_green_light": False,
    }
    (args.output / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    source_paths = [
        repository / "src" / "interaction_sprint" / "hindsight_human_feedback.py",
        repository / "src" / "interaction_sprint" / "hindsight_human_feedback_llm.py",
        Path(__file__),
        repository / "scripts" / "verify_hindsight_human_feedback_llm_dev.py",
    ]
    (args.output / "source_hashes.json").write_text(json.dumps({
        str(path.relative_to(repository)).replace("\\", "/"): sha256(path)
        for path in source_paths
    }, indent=2), encoding="utf-8")

    torch, tokenizer, model = load_runtime(args.hf_home)
    qualification = qualification_cases()
    q_prompts = [render(tokenizer, build_rating_messages(case["pre_belief"], case["evidence"]))
                 for case in qualification]
    q_receipts = generate_batches(torch, tokenizer, model, q_prompts, args.batch_size)
    q_rows = [{"case_id": case["case_id"], "expected": case["expected"], **receipt}
              for case, receipt in zip(qualification, q_receipts)]
    qualified = all(row["strict_parse"] and abs(row["post_rating"] - row["expected"]) <= 5 for row in q_rows)
    (args.output / "qualification.json").write_text(json.dumps({
        "qualified": qualified,
        "criterion": "6/6 strict JSON and within 5 rating points of explicit terminal ratings",
        "rows": q_rows,
    }, indent=2), encoding="utf-8")
    if not qualified:
        report_path = args.output / "report.json"
        report_path.write_text(json.dumps({
            "decision": "MODEL_INTERFACE_QUALIFICATION_FAILED",
            "human_prompts_run": 0,
            "paper_green_light": False,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "wall_seconds": time.time() - started,
                "batch_size": args.batch_size,
            },
        }, indent=2), encoding="utf-8")
        manifest_paths = [
            args.output / "qualification.json", report_path,
            args.output / "spec.json", args.output / "source_hashes.json",
        ]
        (args.output / "MANIFEST.json").write_text(
            json.dumps({path.name: sha256(path) for path in manifest_paths}, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"decision": "MODEL_INTERFACE_QUALIFICATION_FAILED"}, indent=2))
        return

    jobs = []
    for record in dev:
        for depth in (3, 6):
            for arm in ARMS:
                jobs.append((record, depth, arm, render(
                    tokenizer, build_rating_messages(record.pre_belief, record.texts[arm][depth])
                )))
    receipts = generate_batches(torch, tokenizer, model, [job[3] for job in jobs], args.batch_size)
    prediction_path = args.output / "predictions.jsonl"
    parsed = {}
    with prediction_path.open("x", encoding="utf-8") as handle:
        for (record, depth, arm, _), receipt in zip(jobs, receipts):
            key = (record.record_sha256, depth, arm)
            parsed[key] = receipt["post_rating"]
            handle.write(json.dumps({
                "record_sha256": record.record_sha256,
                "query_sha256": record.query_sha256,
                "condition": record.condition,
                "depth": depth,
                "arm": arm,
                **receipt,
            }, sort_keys=True) + "\n")
    all_parseable = all(value is not None for value in parsed.values())
    target = np.array([record.belief_delta for record in dev], dtype=float)
    groups = [record.query_sha256 for record in dev]
    result_metrics = {}
    comparisons = {}
    gates = {"all_human_outputs_strictly_parseable": all_parseable}
    if all_parseable:
        predictions = {
            depth: {
                arm: np.array([parsed[(record.record_sha256, depth, arm)] - record.pre_belief for record in dev])
                for arm in ARMS
            }
            for depth in (3, 6)
        }
        result_metrics = {
            depth: {arm: metrics(target, predictions[depth][arm]) for arm in ARMS}
            for depth in (3, 6)
        }
        comparisons = {
            "early_user_over_query": cluster_bootstrap_gain(
                target, predictions[3]["query_only"], predictions[3]["user_only"], groups),
            "late_user_over_query": cluster_bootstrap_gain(
                target, predictions[6]["query_only"], predictions[6]["user_only"], groups),
            "early_full_over_assistant": cluster_bootstrap_gain(
                target, predictions[3]["assistant_only"], predictions[3]["full"], groups),
            "late_full_over_assistant": cluster_bootstrap_gain(
                target, predictions[6]["assistant_only"], predictions[6]["full"], groups),
        }
        gates.update({
            "late_full_spearman_at_least_point_30": result_metrics[6]["full"]["spearman"] >= 0.30,
            "late_user_adds_query_cluster_robust_signal":
                comparisons["late_user_over_query"]["cluster_bootstrap_ci95_low"] > 0,
            "late_user_relative_mse_gain_at_least_5pct":
                comparisons["late_user_over_query"]["relative_mse_gain"] >= 0.05,
            "late_full_adds_beyond_assistant_cluster_robust_signal":
                comparisons["late_full_over_assistant"]["cluster_bootstrap_ci95_low"] > 0,
            "late_full_relative_mse_gain_at_least_5pct":
                comparisons["late_full_over_assistant"]["relative_mse_gain"] >= 0.05,
        })
    decision = "LLM_DEV_SIGNAL_QUALIFIED_CONFIRMATION_LOCKED" if all(gates.values()) else "LLM_DEV_SIGNAL_NOT_QUALIFIED"
    report = {
        "decision": decision,
        "scope": "Zero-shot DEV predictive audit only; not causal mediation, learning, welfare, or a paper gate.",
        "dataset_sha256": DATA_SHA256,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "class": model.__class__.__name__,
                  "dtype": str(next(model.parameters()).dtype), "thinking": False},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "wall_seconds": time.time() - started, "batch_size": args.batch_size},
        "cohort": {"rows": len(dev), "query_groups": len(set(groups)), "jobs": len(jobs)},
        "metrics": result_metrics,
        "comparisons": comparisons,
        "gates": gates,
        "interpretation_limits": [
            "Predictive signal does not identify the causal effect of any message.",
            "Six-turn inclusion conditions on conversation length.",
            "Model predictions may track linguistic style rather than latent belief.",
            "Confirmation outcomes remain unopened by this runner.",
        ],
        "paper_green_light": False,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_paths = [
        args.output / "qualification.json", prediction_path, report_path,
        args.output / "spec.json", args.output / "source_hashes.json",
    ]
    (args.output / "MANIFEST.json").write_text(
        json.dumps({path.name: sha256(path) for path in manifest_paths}, indent=2), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates, "metrics": result_metrics,
                      "comparisons": comparisons, "wall_seconds": report["runtime"]["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
