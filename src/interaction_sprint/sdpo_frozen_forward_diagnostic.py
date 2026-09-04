"""Post-hoc fixed-feedback neural diagnostic. No sampling, optimizer or training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

HINDSIGHT = "\n\n=== HINDSIGHT CONTEXT ===\n[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
CONTEXTS = ("base", "empty", "feedback", "explicit")
CHECKPOINTS = ("initial_adapter.pt", "adapter_step_16.pt", "final_adapter.pt")
SOURCE_MANIFEST_SHA = "3765136f5452a3831c9a5620938a8b20bd28d0ad18c59465179b5f2bebc0fd41"
PUNCTUATION = frozenset(":=;|{}[]*-\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(x):
    if isinstance(x, dict):
        return {str(k): normalize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [normalize(v) for v in (sorted(x, key=repr) if isinstance(x, set) else x)]
    return x


def dump(path, x):
    Path(path).write_text(json.dumps(normalize(x), indent=2, allow_nan=False), encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def select_cases(rows):
    if len(rows) != 32 or len({r["id"] for r in rows}) != 32:
        raise ValueError("expected 32 distinct calibration originals")
    good = sorted((r for r in rows if r["score"]["joint"]), key=lambda r: r["id"])
    if len(good) != 8:
        raise ValueError("expected exactly eight originally correct cases")
    selected = list(good)
    for style, count in (("json", 3), ("table", 3), ("bullets", 2)):
        pool = sorted((r for r in rows if not r["score"]["joint"] and r["preference"] == style), key=lambda r: r["id"])
        if len(pool) < count:
            raise ValueError("insufficient predetermined mismatch stratum")
        selected.extend(pool[:count])
    assert len(selected) == len({r["id"] for r in selected}) == 16
    return selected


def positions(tokenizer, ids):
    """First emitted token and first later token containing a layout delimiter.

    Decode one native token at a time, excluding special IDs. Search offsets>=1
    to keep two positions distinct. If absent, stop: never choose using logits.
    """
    for i, token in enumerate(ids):
        if i and token not in tokenizer.all_special_ids:
            text = tokenizer.decode([token], skip_special_tokens=False)
            if any(c in PUNCTUATION for c in text):
                return [0, i]
    raise ValueError("no noninitial punctuation token in frozen completion")


def fixed_feedback_moments(student_logits, teacher_logits, sampled_id):
    """Categorical local score moments for FROZEN feedback and prefix, not RL replay."""
    lp = student_logits.double().log_softmax(-1)
    lq = teacher_logits.double().log_softmax(-1)
    p = lp.exp()
    advantage = (lq - lp).detach()
    mean_a = (p * advantage).sum()
    gradient = p * (advantage - mean_a)
    second = (p * advantage.square() * (1 - 2*p + p.square().sum())).sum()
    variance = (second - gradient.square().sum()).clamp_min(0)
    sample_gradient = -p.clone()
    sample_gradient[sampled_id] += 1
    sample_gradient *= advantage[sampled_id]
    # Released top20 + tail reverse-KL scalar, differentiating student only.
    z = student_logits.double().detach().clone().requires_grad_()
    slp = z.log_softmax(-1)
    ix = slp.detach().topk(20).indices
    def tail(v):
        total = v.logsumexp(-1).clamp(max=-1e-7)
        return torch.cat((v, torch.log(-torch.expm1(total)).reshape(1)))
    st, te = tail(slp[ix]), tail(lq[ix].detach())
    loss = (st.exp() * (st - te)).sum()
    descent = -torch.autograd.grad(loss, z)[0]
    def cosine(a, b):
        denominator = a.norm() * b.norm()
        return float((a @ b) / denominator) if denominator > 0 else None
    return dict(student_selected_logp=float(lp[sampled_id]), teacher_selected_logp=float(lq[sampled_id]),
        selected_advantage=float(advantage[sampled_id]), fixed_feedback_expected_ascent_norm=float(gradient.norm()),
        fixed_feedback_variance_trace=float(variance), sampled_ascent_norm=float(sample_gradient.norm()),
        sampled_vs_expected_cosine=cosine(sample_gradient, gradient),
        top20_tail_reverse_kl=float(loss.detach()), top20_tail_descent_norm=float(descent.norm()),
        top20_tail_vs_expected_cosine=cosine(descent, gradient))


def forward(model, context, completion, offsets, device):
    ids = torch.tensor([context + completion[:-1]], device=device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(completion)).logits[0].float()
        targets = torch.tensor(completion, device=device)
        logps = logits.log_softmax(-1).gather(-1, targets[:, None])[:, 0]
        return logits[offsets].cpu().contiguous(), logps.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def finish(status, **extra):
        dump(args.root / "status.json", dict(status=status, elapsed_seconds=time.monotonic()-started, **extra))
        dump(args.root / "MANIFEST.json", {str(f.relative_to(args.root)): sha(f) for f in args.root.rglob("*") if f.is_file() and f.name != "MANIFEST.json"})
    from latent_contract import sender_update as lora
    needed = ["MANIFEST.json", "calibration_original.jsonl", "calibration_explicit.jsonl", "calibration_teacher.jsonl",
              "PREPARATION_MANIFEST.json", "model.json", "lora_source.py", "upstream_loss_source.py", *CHECKPOINTS]
    if sha(args.evidence / "MANIFEST.json") != SOURCE_MANIFEST_SHA:
        raise ValueError("not the frozen verified v3 evidence root")
    manifest = json.loads((args.evidence / "MANIFEST.json").read_text())
    for name in needed:
        if name != "MANIFEST.json" and sha(args.evidence / name) != manifest[name]:
            raise ValueError("source evidence checksum mismatch: " + name)
    if sha(Path(lora.__file__)) != sha(args.evidence / "lora_source.py"):
        raise ValueError("LoRA helper differs from archived run")
    prepared_path = args.prepared / "calibration.json"
    pm = json.loads((args.evidence / "PREPARATION_MANIFEST.json").read_text())
    if sha(prepared_path) != pm["calibration.json"]["sha256"]:
        raise ValueError("prepared calibration mismatch")
    raw = {r["id"]: r for r in json.loads(prepared_path.read_text())}
    originals = read_rows(args.evidence / "calibration_original.jsonl")
    cases = select_cases(originals)
    views = {kind: {r["id"]: r for r in read_rows(args.evidence / filename)} for kind, filename in
             (("explicit", "calibration_explicit.jsonl"), ("feedback", "calibration_teacher.jsonl"))}
    sources = {"runner_source.py": Path(__file__), "lora_source.py": Path(lora.__file__)}
    for name, src in sources.items():
        (args.root / name).write_bytes(src.read_bytes())
    (args.root / "upstream_loss_source.py").write_bytes((args.evidence / "upstream_loss_source.py").read_bytes())
    input_hashes = {n: sha(args.evidence / n) for n in needed}
    dump(args.root / "freeze.json", dict(input_sha256=input_hashes,
        arguments={k: str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        prepared_calibration_sha256=sha(prepared_path), sources={n: sha(p) for n, p in sources.items()},
        checkpoints=list(CHECKPOINTS), contexts=list(CONTEXTS), selected_ids=[r["id"] for r in cases],
        layout_delimiters=sorted(PUNCTUATION), calls=192, time_cap_seconds=2700,
        scope="Post-hoc fixed feedback/prefix logit moments only; no semantic mass, global unbiasedness, optimizer replay or paper claim"))
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dump(args.root / "environment.json", dict(torch=torch.__version__, transformers=transformers.__version__,
        threads=args.threads, device=args.device, seed=9047801))
    torch.set_num_threads(args.threads)
    torch.manual_seed(9047801)
    model_hashes = {str(f.relative_to(args.model_path)): sha(f) for f in args.model_path.rglob("*") if f.is_file()}
    old_model = json.loads((args.evidence / "model.json").read_text())
    if any(model_hashes.get(n) != h for n, h in old_model["sha256"].items()):
        raise ValueError("model differs from source study")
    dump(args.root / "model.json", dict(revision=old_model["revision"], sha256=model_hashes))
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    frozen = []
    for case in cases:
        item = copy.deepcopy(raw[case["id"]]["prompt"])
        for msg in reversed(item):
            if msg["role"] == "user":
                msg["content"] += HINDSIGHT
                break
        empty = tok.apply_chat_template(item, tokenize=True, add_generation_prompt=True,
                                       enable_thinking=False, return_dict=False)
        if not isinstance(empty, list) or not all(isinstance(x,int) for x in empty):
            raise ValueError("native tokenizer did not return a flat token-ID list")
        contexts = {"base": case["prompt_ids"], "empty": empty,
                    **{k: views[k][case["id"]]["prompt_ids"] for k in views}}
        if any(len(v)+len(case["completion_ids"]) > 1024 for v in contexts.values()):
            raise ValueError("overlength frozen prefix")
        frozen.append(dict(id=case["id"], preference=case["preference"], original_joint=case["score"]["joint"],
            contexts=contexts, completion_ids=case["completion_ids"], positions=positions(tok, case["completion_ids"]),
            original_logps=case["token_logprobs"], original_text=case["text"]))
    dump(args.root / "cases.json", frozen)
    if time.monotonic()-started > 1800:
        finish("TIME_CAP_PRELOAD"); return
    model, info = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa", output_loading_info=True)
    dump(args.root / "loading_info.json", info)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        finish("INVALID_MODEL_LOADING"); return
    model = model.to(args.device).eval()
    inserted = lora.install_lora(model, rank=16, alpha=32)
    dump(args.root / "adapter_insertion.json", inserted)
    count, forward_times = 0, []
    loop_started = time.monotonic()
    (args.root / "logits").mkdir()
    with (args.root / "records.jsonl").open("x", encoding="utf-8") as out:
        for checkpoint in CHECKPOINTS:
            state = torch.load(args.evidence / checkpoint, map_location="cpu", weights_only=True)
            lora.load_adapter(model, state)
            # Verify loaded float32 adapters exactly against archived checkpoint.
            loaded = lora.adapter_state(model)
            if loaded.keys() != state.keys() or any(not torch.equal(loaded[k], state[k]) for k in state):
                raise ValueError("loaded adapter mismatch")
            dump(args.root / (checkpoint + ".loaded.json"), dict(checkpoint_sha256=input_hashes[checkpoint],
                tensor_sha256={k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in loaded.items()},
                tensor_shapes={k:list(v.shape) for k,v in loaded.items()}, exact_equal=True))
            for case_index, case in enumerate(frozen):
                base_logits = None
                for context in CONTEXTS:
                    if time.monotonic()-started > 2640:
                        finish("TIME_CAP_PARTIAL", forwards=count); return
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t = time.monotonic()
                    logits, logps = forward(model, case["contexts"][context], case["completion_ids"], case["positions"], args.device)
                    if args.device.startswith("cuda"):
                        torch.cuda.synchronize()
                    elapsed = time.monotonic()-t
                    if not torch.isfinite(logits).all() or not all(torch.isfinite(torch.tensor(logps))):
                        raise ValueError("nonfinite forward")
                    count += 1; forward_times.append(elapsed)
                    name = f"logits/{count:03d}.pt"
                    torch.save(logits, args.root / name)
                    if context == "base":
                        base_logits = logits
                    metrics = [fixed_feedback_moments(base_logits[j], logits[j], case["completion_ids"][p]) for j,p in enumerate(case["positions"])]
                    row = dict(call=count, checkpoint=checkpoint, checkpoint_sha256=input_hashes[checkpoint],
                        checkpoint_loaded_exact=True, case_index=case_index, id=case["id"], context=context,
                        positions=case["positions"], logits_path=name, logits_sha256=sha(args.root/name),
                        logits_shape=list(logits.shape), logits_dtype=str(logits.dtype), completion_logps=logps,
                        elapsed_seconds=elapsed, fixed_feedback_local_moments=metrics)
                    if checkpoint == CHECKPOINTS[0] and context == "base":
                        row["original_base_max_logprob_gap"] = max(abs(a-b) for a,b in zip(logps,case["original_logps"]))
                    out.write(json.dumps(row, allow_nan=False)+"\n"); out.flush()
                    if count == 12:
                        estimate = time.monotonic()-started + (time.monotonic()-loop_started)/12*180
                        dump(args.root / "early_timing.json", dict(forwards=12, elapsed_seconds=time.monotonic()-started,
                            projected_total_seconds=estimate, rule="startup+observed elapsed+180*first12 walltime/call; includes serialization and CPU metrics"))
                        if estimate > 2700:
                            finish("TIME_PROJECTION_STOP", forwards=count); return
    finish("COMPLETE_FIXED_FORWARD_DIAGNOSTIC", forwards=count)


if __name__ == "__main__":
    main()
