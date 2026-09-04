"""Stage A: fixed ordinary sender update and sender-only qualification evidence.

Never reads final_eval, calibration data or private answer keys. No bridge calls,
checkpoint selection, repair or automatic next stage. Requires confirmed GPU access.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from scripts.run_c2c_baseline_dev import SPEC as BASE_SPEC

SPEC = {"scope": "NATURAL_SENDER_UPDATE_STAGE_A_NOT_INTERFACE_RESULT",
        "model": BASE_SPEC["sender"], "revision": BASE_SPEC["sender_revision"],
        "upstream_commit": BASE_SPEC["upstream_commit"],
        "torch": BASE_SPEC["torch"], "transformers": BASE_SPEC["transformers"],
        "seeds": [202609041, 202609042], "rank": 16, "alpha": 32,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj"], "epochs": 1,
        "micro_batch": 2, "accumulation": 4, "lr": 0.00005,
        "weight_decay": 0.0, "betas": [0.9, 0.999], "epsilon": 1e-8,
        "max_input_tokens": 2048, "max_new_tokens": 64,
        "train_sha256": "07ee7fc5a02472c720d86a06ac9f139085dfa2a0c024499e5dce80464397ad90",
        "qualification_sha256": "3d0d4ec4857f317f678f3ba953a7df933bbf2610b18c844cbb73caf9ca929d93",
        "train_count": 1024, "qualification_count": 128, "optimizer_steps": 128,
        "qualification_accuracy_retention_pp": 3.125,
        "qualification_choice_ce_relative_reduction": 0.05,
        "qualification_parse_rate": 0.95}
SOURCE_FILES = ["scripts/run_c2c_sender_update.py", "src/latent_contract/sender_update.py",
                "scripts/run_c2c_baseline_dev.py"]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for blob in iter(lambda: stream.read(1024*1024), b""):
            digest.update(blob)
    return digest.hexdigest()


def pinned_prompt(upstream):
    source = subprocess.check_output(["git", "-C", str(upstream), "show",
                                      SPEC["upstream_commit"] + ":rosetta/utils/evaluate.py"])
    if hashlib.sha256(source).hexdigest() != "b341338aecf1cf07a5cb61664b8dac5f2ff96b32622e54eb37b32604deeddf16":
        raise ValueError("wrong prompt source")
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "build_prompt")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "pinned_prompt", "exec"), namespace)
    return namespace["build_prompt"]


def load_inputs(train_path, qualification_path):
    # Only two explicitly supplied public files are opened; never scan a private root.
    values = []
    for path, digest, count, labeled in (
            (train_path, SPEC["train_sha256"], SPEC["train_count"], True),
            (qualification_path, SPEC["qualification_sha256"], SPEC["qualification_count"], False)):
        if sha256(path) != digest:
            raise ValueError("data digest mismatch")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if len(rows) != count or len({r["case_id"] for r in rows}) != count:
            raise ValueError("data count mismatch")
        if any(("answer" in row) != labeled for row in rows):
            raise ValueError("label visibility mismatch")
        values.append(rows)
    if {r["content_sha256"] for r in values[0]} & {r["content_sha256"] for r in values[1]}:
        raise ValueError("train/qualification overlap")
    return values


def run(train_path, qualification_path, upstream, seed, output):
    if seed not in SPEC["seeds"]:
        raise ValueError("seed not fixed in protocol")
    train, qualification = load_inputs(train_path, qualification_path)
    build_prompt = pinned_prompt(upstream)
    if shutil.disk_usage(output.parent).free < 30 * 1024**3:
        raise ValueError("less than 30GiB free for preserved checkpoints")
    output.mkdir(parents=True, exist_ok=False)
    try:
        write(output / "spec.json", {**SPEC, "seed": seed})
        source_root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        frozen_sources = {}
        for relative in SOURCE_FILES:
            local = source_root / relative
            pinned = subprocess.check_output(["git", "-C", str(source_root), "show", commit + ":" + relative])
            if local.read_bytes() != pinned:
                raise ValueError("runner/import source differs from committed bytes")
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pinned)
            frozen_sources[relative] = hashlib.sha256(pinned).hexdigest()
        write(output / "source.json", {"commit": commit, "files": frozen_sources})
        (output / "update_train.json").write_bytes(train_path.read_bytes())
        (output / "qualification.json").write_bytes(qualification_path.read_bytes())
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from latent_contract.sender_update import (adapter_state, install_lora, load_adapter, merge_lora,
            supervised_example, collate, completion_log_probs, train_epoch)
        if torch.__version__ != SPEC["torch"] or transformers.__version__ != SPEC["transformers"]:
            raise ValueError("dependency mismatch; use isolated C2C environment")
        if not torch.cuda.is_available():
            raise ValueError("CUDA unavailable")
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(SPEC["model"], revision=SPEC["revision"], local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(SPEC["model"], revision=SPEC["revision"], local_files_only=True,
                    trust_remote_code=False, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
        if model.config._commit_hash != SPEC["revision"] or model.config.num_hidden_layers != 36:
            raise ValueError("unexpected sender")
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        completion_ids = {label: tokenizer.encode("The correct answer is " + label, add_special_tokens=False) + [tokenizer.eos_token_id]
                          for label in "ABCD"}
        def ids_for(row):
            choices = "".join(f"{chr(65+i)}. {text}\n" for i, text in enumerate(row["choices"]))
            prompt = build_prompt("mmlu-redux", "", row["question"], choices, False, True)
            return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                   add_generation_prompt=True, enable_thinking=False)
        train_examples = [supervised_example(ids_for(row), completion_ids[row["answer"]], SPEC["max_input_tokens"]) for row in train]
        qual_ids = {r["case_id"]: ids_for(r) for r in qualification}
        for ids in qual_ids.values():
            for answer in completion_ids.values():
                supervised_example(ids, answer, SPEC["max_input_tokens"])
        write(output / "encoded_train.json", train_examples)
        write(output / "qualification_prompt_ids.json", qual_ids)
        write(output / "answer_token_ids.json", completion_ids)
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(0), "revision": model.config._commit_hash,
              "training": "explicit standard LoRA, float32 adapters, BF16 base, no PEFT dependency"})
        def evaluate(stage, rows):
            model.eval()
            model.config.use_cache = True
            results = []
            with torch.inference_mode(), (output / (stage + ".jsonl")).open("x", encoding="utf-8") as stream:
                for row in rows:
                    ids = qual_ids[row["case_id"]]
                    batch = collate([supervised_example(ids, completion_ids[a]) for a in "ABCD"], pad, "cuda:0")
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
                    scores = completion_log_probs(logits, batch["labels"]).tolist()
                    if not all(math.isfinite(s) for s in scores):
                        raise ValueError("nonfinite qualification score")
                    del logits
                    prompt_tensor = torch.tensor([ids], device="cuda:0")
                    generated = model.generate(prompt_tensor, attention_mask=torch.ones_like(prompt_tensor),
                              do_sample=False, max_new_tokens=SPEC["max_new_tokens"], pad_token_id=pad)[0, len(ids):]
                    torch.cuda.synchronize()
                    item = {"case_id": row["case_id"], "dataset": row["dataset"], "stage": stage,
                            "input_ids": ids, "choice_sequence_logps": dict(zip("ABCD", scores)),
                            "generated_ids": generated.tolist(), "completion": tokenizer.decode(generated, skip_special_tokens=True),
                            "hit_token_limit": len(generated) == SPEC["max_new_tokens"], "seconds": time.monotonic()-start}
                    stream.write(json.dumps(item) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    results.append(item)
            return results
        base = evaluate("base", qualification)
        targets = install_lora(model, SPEC["rank"], SPEC["alpha"])
        if len(targets) != 144:
            raise ValueError("expected four projections in each of 36 layers")
        write(output / "adapter_targets.json", targets)
        torch.save(adapter_state(model), output / "adapter_initial.pt")
        load_adapter(model, torch.load(output / "adapter_initial.pt", weights_only=True, map_location="cpu"))
        noop = evaluate("noop", qualification[:8])
        for old, new in zip(base[:8], noop):
            if old["generated_ids"] != new["generated_ids"] or old["choice_sequence_logps"] != new["choice_sequence_logps"]:
                raise ValueError("zero adapter save/load changed qualification behavior")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=SPEC["lr"],
                 betas=tuple(SPEC["betas"]), eps=SPEC["epsilon"], weight_decay=SPEC["weight_decay"])
        start = time.monotonic()
        with (output / "training.jsonl").open("x", encoding="utf-8") as stream:
            def callback(row):
                row["elapsed_seconds"] = time.monotonic()-start
                stream.write(json.dumps(row)+"\n")
                stream.flush()
                os.fsync(stream.fileno())
                if row["step"] == 64:
                    torch.save(adapter_state(model), output / "adapter_step_0064.pt")
            steps = train_epoch(model, train_examples, optimizer, seed, SPEC["micro_batch"], SPEC["accumulation"], pad, callback)
        if steps != SPEC["optimizer_steps"]:
            raise ValueError("optimizer step count mismatch")
        torch.save(adapter_state(model), output / "adapter_final.pt")
        torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(), "steps": steps}, output / "optimizer_final.pt")
        del optimizer
        model.gradient_checkpointing_disable()
        wrapped = evaluate("wrapped_final", qualification[:8])
        merge_lora(model)
        model.save_pretrained(output / "merged_sender", safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(output / "merged_sender")
        updated = evaluate("updated", qualification)
        write(output / "merge_audit.json", {"n": 8,
              "exact_generation_agreement": sum(a["generated_ids"] == b["generated_ids"] for a, b in zip(wrapped, updated)),
              "max_choice_logp_difference": max(abs(a["choice_sequence_logps"][k]-b["choice_sequence_logps"][k])
                                                for a, b in zip(wrapped, updated) for k in "ABCD"),
              "interpretation": "BF16 merge can round differently; qualification is on the merged sender used downstream"})
        write(output / "COMPLETE.json", {"seed": seed, "steps": steps, "qualification_cases": 128,
                                         "bridge_calls": 0, "final_eval_accessed": False})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.relative_to(output).as_posix(): sha256(p)
              for p in output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train", "qualification", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=SPEC["seeds"])
    args = parser.parse_args()
    run(args.train, args.qualification, args.upstream, args.seed, args.output)
