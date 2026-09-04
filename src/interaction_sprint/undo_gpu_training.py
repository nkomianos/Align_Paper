"""Frozen one-action UNDO training development; no experiment at import.

Three token-exposure-matched arms. Distillation is next-token full-vocabulary
forward KL, NOT an on-policy sequence-level CCOPD reproduction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from latent_contract.sender_update import install_lora, adapter_state, load_adapter

ARMS = ("terminal_sft", "canonical_distillation", "local_relation")
CONTROLS = ("canonical", "padded", "counterfactual")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, value):
    with open(path, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def loading_metadata_json(value):
    """Normalize Transformers container metadata without changing its contents."""
    if isinstance(value, dict):
        return {key: loading_metadata_json(item) for key, item in value.items()}
    if isinstance(value, set):
        return [loading_metadata_json(item) for item in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [loading_metadata_json(item) for item in value]
    return value


def finish(root, status):
    write_json(root / "status.json", status)
    write_json(root / "MANIFEST.json", {str(p.relative_to(root)): digest(p) for p in root.rglob("*") if p.is_file()})


def qualification(rows):
    controls = {}
    for condition in CONTROLS:
        rs = [r for r in rows if r["condition"] == condition]
        controls[condition] = dict(n=len(rs), accuracy=sum(r["prediction"] == r["target"] for r in rs) / len(rs) if rs else 0.)
    mass = sum(r["choice_mass"] for r in rows) / len(rows) if rows else 0.
    return dict(qualified=bool(rows) and all(r["n"] and r["accuracy"] >= .9 for r in controls.values()) and mass >= .5,
                controls=controls, mean_choice_mass=mass, rule="each control >=.9; all-dev mean mass >=.5")


def synchronize(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def validate_data(train, dev, evaluation):
    ids = []
    for split, rows in (("train", train), ("dev", dev), ("eval", evaluation)):
        if not rows:
            raise ValueError("empty split")
        for row in rows:
            ids.append(row["id"])
            if row["target"] not in "ABCD" or len(row["target"]) != 1:
                raise ValueError("target must be exactly one ABCD character")
            for field in (["prompt", "canonical_prompt", "local_prompt"] if split == "train" else ["prompt"]):
                if not row[field]:
                    raise ValueError("empty prompt")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate or overlapping split IDs")


def tokenize(tokenizer, prompt, cap):
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, return_dict=False)
    if not ids or len(ids) > cap:
        raise ValueError(f"empty/overlength prompt: {len(ids)}, cap {cap}; no truncation")
    return list(ids)


def collate(prompts, pad, device):
    width = max(map(len, prompts))
    ids, masks = [], []
    for p in prompts:
        ids.append([pad] * (width - len(p)) + p)
        masks.append([0] * (width - len(p)) + [1] * len(p))
    mask = torch.tensor(masks, device=device)
    return dict(input_ids=torch.tensor(ids, device=device), attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp_min(0))


def next_logits(model, prompts, pad, device):
    return model(**collate(prompts, pad, device), use_cache=False, logits_to_keep=1).logits[:, -1].float()


def loss_for(logits, labels=None, teacher=None):
    if (labels is None) == (teacher is None):
        raise ValueError("specify exactly one target")
    if teacher is None:
        return torch.nn.functional.cross_entropy(logits.float(), labels)
    if teacher.shape != logits.shape or not torch.isfinite(teacher).all():
        raise ValueError("bad teacher")
    # Stored full-vocabulary distribution, detached from student parameters.
    p = teacher.float().detach()
    if not torch.allclose(p.sum(-1), torch.ones(len(p), device=p.device), atol=2e-5):
        raise ValueError("teacher not normalized")
    return (p * (p.clamp_min(1e-30).log() - logits.float().log_softmax(-1))).sum(-1).mean()


def schedule(n, epochs, batch, seed):
    result = []
    rng = random.Random(seed)
    for epoch in range(epochs):
        order = list(range(n)); rng.shuffle(order)
        result.extend([order[i:i + batch] for i in range(0, n, batch)])
    return result


def evaluate(model, rows, prompts, choices, pad, device, batch, dest):
    model.eval()
    correct = 0
    with open(dest, "x", encoding="utf-8") as f, torch.no_grad():
        for start in range(0, len(rows), batch):
            logits = next_logits(model, prompts[start:start + batch], pad, device)
            lp = logits.log_softmax(-1)
            cp = lp[:, choices].exp()
            pred = cp.argmax(-1).tolist()
            for k, (r, probs) in enumerate(zip(rows[start:start + batch], cp.tolist())):
                answer = "ABCD"[pred[k]]
                correct += answer == r["target"]
                output = dict(r, prediction=answer, choice_probabilities=probs,
                              choice_mass=sum(probs), native_argmax_token=int(logits[k].argmax()),
                              prompt_tokens=len(prompts[start + k]))
                f.write(json.dumps(output) + "\n"); f.flush()
    return dict(correct=correct, total=len(rows), accuracy=correct / len(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=9047701)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.eval_batch_size < 1 or args.threads < 1:
        raise ValueError("invalid budget")
    if args.model_path.name != args.revision:
        raise ValueError("snapshot path must equal declared revision")
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    data = {s: json.loads((args.prepared / f"{s}.json").read_text(encoding="utf-8")) for s in ("train", "dev", "eval")}
    validate_data(data["train"], data["dev"], data["eval"])
    source = Path(__file__)
    (args.root / "runner_source.py").write_bytes(source.read_bytes())
    lora_source = Path(__import__("latent_contract.sender_update", fromlist=["x"]).__file__)
    (args.root / "lora_source.py").write_bytes(lora_source.read_bytes())
    write_json(args.root / "freeze.json", dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256=digest(source), lora_source_sha256=digest(lora_source),
        input_sha256={s: digest(args.prepared / f"{s}.json") for s in data},
        objective="one-action full-vocabulary forward KL; not sequence-level CCOPD", arms=ARMS,
        selection="fixed final update; no eval checkpoint selection", torch=torch.__version__))
    for split, rows in data.items():
        (args.root / f"{split}.json").write_bytes((args.prepared / f"{split}.json").read_bytes())
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    choices = [tokenizer.encode(c, add_special_tokens=False) for c in "ABCD"]
    if any(len(c) != 1 for c in choices):
        raise ValueError("choices must be single tokens")
    choices = [c[0] for c in choices]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompts = {s: [tokenize(tokenizer, r["prompt"], args.max_tokens) for r in rows] for s, rows in data.items()}
    teachers = {a: [tokenize(tokenizer, r[field], args.max_tokens) for r in data["train"]]
                for a, field in (("canonical_distillation", "canonical_prompt"), ("local_relation", "local_prompt"))}
    plan = schedule(len(data["train"]), args.epochs, args.batch_size, args.seed)
    write_json(args.root / "tokenized.json", dict(student=prompts, teacher=teachers, choices=choices, schedule=plan))
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model_files = {str(p.relative_to(args.model_path)): digest(p) for p in args.model_path.rglob("*")
                   if p.is_file() and (p.suffix in (".json", ".safetensors", ".model", ".txt") or p.name == "tokenizer_config.json")}
    if not any(p.endswith(".safetensors") for p in model_files):
        raise ValueError("no native model weights found")
    write_json(args.root / "model.json", dict(revision=args.revision, sha256=model_files, transformers=transformers.__version__))
    model, loading_info = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
                    torch_dtype=dtype, attn_implementation="sdpa", output_loading_info=True)
    write_json(args.root / "loading_info.json", loading_metadata_json(loading_info))
    if any(loading_info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        finish(args.root, dict(status="INVALID_MODEL_LOADING", scientific_decision=None))
        return
    model = model.to(args.device)
    model.eval()
    baseline = {"dev": evaluate(model, data["dev"], prompts["dev"], choices, pad, args.device, args.eval_batch_size,
                                args.root / "baseline_dev.jsonl")}
    q = qualification([json.loads(line) for line in (args.root / "baseline_dev.jsonl").read_text().splitlines()])
    write_json(args.root / "qualification.json", q)
    if not q["qualified"]:
        finish(args.root, dict(status="UNQUALIFIED_ASSAY", scientific_decision=None, reason="development control/mass prerequisites",
                               elapsed_seconds=time.time() - started))
        print(json.dumps(q), flush=True)
        return
    baseline["eval"] = evaluate(model, data["eval"], prompts["eval"], choices, pad, args.device,
                                args.eval_batch_size, args.root / "baseline_eval.jsonl")
    teacher_arrays = {}
    teacher_seconds = {}
    for arm, ps in teachers.items():
        synchronize(args.device); teacher_start = time.time()
        values = []
        with torch.no_grad():
            for i in range(0, len(ps), args.batch_size):
                values.append(next_logits(model, ps[i:i + args.batch_size], pad, args.device).softmax(-1).cpu())
        teacher_arrays[arm] = torch.cat(values)
        torch.save(teacher_arrays[arm], args.root / f"{arm}_teacher.pt")
        synchronize(args.device); teacher_seconds[arm] = time.time() - teacher_start
    write_json(args.root / "teacher_timing.json", teacher_seconds)
    with torch.no_grad():
        insertion_reference = next_logits(model, prompts["train"][:2], pad, args.device).cpu()
    install_lora(model, rank=8, alpha=16)
    with torch.no_grad():
        insertion_error = float((next_logits(model, prompts["train"][:2], pad, args.device).cpu() - insertion_reference).abs().max())
    write_json(args.root / "adapter_insertion_check.json", dict(max_abs_logit_error=insertion_error, tolerance=1e-5))
    if insertion_error > 1e-5:
        raise ValueError("nonzero initial adapter changed model outputs")
    initial = adapter_state(model)
    torch.save(initial, args.root / "initial_adapter.pt")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [p for p in model.parameters() if p.requires_grad]
    results = {"baseline": baseline}
    labels = torch.tensor([choices["ABCD".index(r["target"])] for r in data["train"]], device=args.device)
    for arm in ARMS:
        armroot = args.root / arm; armroot.mkdir()
        load_adapter(model, initial); torch.manual_seed(args.seed)
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0)
        with open(armroot / "training.jsonl", "x", encoding="utf-8") as log:
            for step, indices in enumerate(plan):
                model.train(); optimizer.zero_grad(set_to_none=True)
                synchronize(args.device); begin = time.time()
                logits = next_logits(model, [prompts["train"][i] for i in indices], pad, args.device)
                loss = loss_for(logits, labels=labels[indices]) if arm == "terminal_sft" else loss_for(
                    logits, teacher=teacher_arrays[arm][indices].to(args.device))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                synchronize(args.device)
                log.write(json.dumps(dict(step=step, indices=indices, loss=float(loss.detach()),
                    grad_norm=float(norm), student_tokens=sum(len(prompts["train"][i]) for i in indices),
                    elapsed_seconds=time.time() - begin)) + "\n"); log.flush()
                if (step + 1) % 32 == 0:
                    torch.save(adapter_state(model), armroot / f"adapter_step_{step + 1}.pt")
                    print(json.dumps(dict(arm=arm, step=step + 1, total_steps=len(plan))), flush=True)
        torch.save(adapter_state(model), armroot / "final_adapter.pt")
        torch.save(optimizer.state_dict(), armroot / "final_optimizer.pt")
        results[arm] = {s: evaluate(model, data[s], prompts[s], choices, pad, args.device,
                   args.eval_batch_size, armroot / f"{s}.jsonl") for s in ("dev", "eval")}
    write_json(args.root / "summary.json", dict(results=results, elapsed_seconds=time.time() - started,
        transformers=transformers.__version__, training_steps_per_arm=len(plan),
        student_tokens_per_arm=sum(sum(len(prompts["train"][i]) for i in ix) for ix in plan),
        teacher_tokens={a: sum(map(len, ps)) for a, ps in teachers.items()},
        teacher_seconds=teacher_seconds,
        scientific_scope="single-seed development; no acceptance or theorem claim"))
    finish(args.root, dict(status="COMPLETE", scientific_decision=None))
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
