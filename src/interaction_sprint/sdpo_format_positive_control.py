"""Native full-response SDPO positive control; no endogenous-feedback arm.

Uses the actual pinned released simple-signal loss method with a native-token
forward bridge. All user preferences are withheld from ordinary policy inputs.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from latent_contract.sender_update import install_lora, adapter_state
from .undo_gpu_training import digest, write_json, loading_metadata_json, synchronize

UPSTREAM_COMMIT = "3b17d2a67bd2565b9fbda495fd16a485406aa954"
UPSTREAM_SHA = "6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d"
HINDSIGHT = "\n\n=== HINDSIGHT CONTEXT ===\n[The following is a future user message. Use this to guide your answer to the user prompt.]\n{follow_up}"


def released_loss_class(source):
    source = source.replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA:
        raise ValueError("released loss checksum mismatch")
    original = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "OnlineSDPOUpdater")
    methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_simple_signal_loss"]
    if len(methods) != 1:
        raise ValueError("missing loss")
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="ReleasedLoss", bases=[], keywords=[],
        body=methods, decorator_list=[])], type_ignores=[]))
    namespace = dict(torch=torch, Tuple=Tuple, Dict=Dict)
    exec(compile(tree, "pinned_upstream_simple_signal", "exec"), namespace)
    return namespace["ReleasedLoss"]


def released_reference_check(loss_cls):
    """Check exact released loss/gradient against its stopped-advantage formula."""
    z = torch.tensor([[.1, -.2, .3], [.6, .2, -.1], [-.3, .4, .1]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.4, .1, -.1], [-.4, .5, .2], [.1, -.6, .3]], dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([[0, 1, 2]])
    stub = SimpleNamespace(config=SimpleNamespace(signal_clip=0), _log_token_table=lambda *a: None)
    def forward(name, completion, need_grad=False, **unused):
        logits = z if name == "base" else q.detach()
        lp = logits.log_softmax(-1).gather(-1, ids[0].unsqueeze(-1)).squeeze(-1).unsqueeze(0)
        return lp, torch.ones_like(lp, dtype=torch.long), None
    stub._compute_token_logprobs = forward
    loss, _ = loss_cls._simple_signal_loss(stub, "base", "teacher", ids)
    base = forward("base", ids)[0]
    expected = -((forward("teacher", ids)[0] - base).detach() * base).mean()
    gradient = torch.autograd.grad(loss, z, retain_graph=True)[0]
    reference = torch.autograd.grad(expected, z)[0]
    error = float((gradient - reference).abs().max())
    result = dict(loss_error=float((loss - expected).detach().abs()), gradient_max_error=error,
                  teacher_gradient_is_absent=torch.autograd.grad(loss, q, allow_unused=True)[0] is None)
    if error > 1e-12 or result["loss_error"] > 1e-12 or not result["teacher_gradient_is_absent"]:
        raise ValueError("released signal reference mismatch")
    return result


def hindsight_messages(messages, feedback):
    result = copy.deepcopy(messages)
    for r in reversed(result):
        if r["role"] == "user":
            r["content"] += HINDSIGHT.format(follow_up=feedback.strip())
            return result
    raise ValueError("no user message")


def completion_logps(model, context, completion, device, need_grad):
    """Predictions for actual emitted IDs, with no retokenization/appended EOS."""
    if not context or not completion:
        raise ValueError("empty context/completion")
    ids = torch.tensor([context + completion[:-1]], device=device)
    cm = torch.enable_grad() if need_grad else torch.no_grad()
    with cm:
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                       logits_to_keep=len(completion)).logits.float()
        targets = torch.tensor([completion], device=device)
        lp = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return lp


class NativeBridge:
    def __init__(self, model, device, base, teacher):
        self.model, self.device = model, device
        self.contexts = {"base": base, "teacher": teacher}
        self.config = SimpleNamespace(signal_clip=0.)
        self.saved = {}

    def _compute_token_logprobs(self, name, completion_ids, need_grad=False, need_logits=False):
        lp = completion_logps(self.model, self.contexts[name], completion_ids[0].tolist(), self.device, need_grad)
        self.saved[name] = lp.detach().cpu().tolist()[0]
        return lp, torch.ones_like(lp, dtype=torch.long), None

    def _log_token_table(self, *args):
        pass


def tokenize(tokenizer, messages, cap):
    ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                       add_generation_prompt=True, enable_thinking=False)
    if not ids or len(ids) > cap:
        raise ValueError("overlength/empty input; no truncation")
    return list(ids)


def generation_config(tokenizer, sample, max_tokens):
    from transformers import GenerationConfig
    # Fresh configuration: never inherit checkpoint-specific logits processors.
    return GenerationConfig(max_new_tokens=max_tokens, do_sample=sample,
        temperature=1., top_p=1., top_k=0, typical_p=1., repetition_penalty=1.,
        encoder_repetition_penalty=1., num_beams=1, min_length=0,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
        use_cache=True, return_dict_in_generate=True, output_scores=True)


def generate(model, tokenizer, prompt_ids, device, seed, sample, max_tokens):
    model.eval(); torch.manual_seed(seed)
    ids = torch.tensor([prompt_ids], device=device)
    synchronize(device); start = time.time()
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                generation_config=generation_config(tokenizer, sample, max_tokens))
    y = output.sequences[0, len(prompt_ids):].tolist()
    selected = [float(z[0].float().log_softmax(-1)[token]) for z, token in zip(output.scores, y)]
    synchronize(device)
    return dict(prompt_ids=prompt_ids, completion_ids=y, token_logprobs=selected,
        text=tokenizer.decode(y, skip_special_tokens=True), raw_decoded=tokenizer.decode(y, skip_special_tokens=False),
        terminated=bool(y and y[-1] == tokenizer.eos_token_id), sampling=sample, seed=seed,
        elapsed_seconds=time.time() - start)


def summary(rows):
    return {k: sum(bool(r["score"][k]) for r in rows) / len(rows)
            for k in ("joint", "format_valid", "content_valid")}


def calibration_decision(explicit, teacher, original):
    a, b = summary(explicit), summary(teacher)
    mismatch = [i for i, r in enumerate(original) if not r["score"]["joint"]]
    recovery = sum(teacher[i]["score"]["joint"] for i in mismatch) / len(mismatch) if mismatch else None
    return dict(qualified=a["joint"] >= .9 and b["content_valid"] >= .9 and
        (recovery is None or recovery >= .75), explicit=a, hindsight=b,
        mismatches=len(mismatch), mismatch_recovery=recovery,
        note="engineering qualification, not hypothesis decision")


def positive_decision(before, after):
    b, a = summary(before), summary(after)
    return dict(positive_control=a["joint"] - b["joint"] >= .10 and
                a["content_valid"] - b["content_valid"] >= -.05,
                before=b, after=a, joint_gain=a["joint"] - b["joint"],
                scope="single-seed fixed-budget development; not a paper gate")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prepared", type=Path); p.add_argument("root", type=Path)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--revision", default="1cfa9a7208912126459214e8b04321603b3df60c")
    p.add_argument("--upstream-file", type=Path, required=True)
    p.add_argument("--device", default="cuda"); p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--seed", type=int, default=9047801)
    args = p.parse_args()
    if args.model_path.name != args.revision or args.threads < 1 or args.lr <= 0:
        raise ValueError("invalid config/snapshot")
    loss_cls = released_loss_class(args.upstream_file.read_bytes())
    loss_reference = released_reference_check(loss_cls)
    from . import sdpo_format_control_data as apparatus
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write_json(args.root / "released_loss_reference.json", loss_reference)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    data = {s: json.loads((args.prepared / f"{s}.json").read_text()) for s in ("train", "eval", "calibration")}
    preparation_manifest = json.loads((args.prepared / "MANIFEST.json").read_text())
    if preparation_manifest["source_sha256"] != digest(Path(apparatus.__file__)):
        raise ValueError("prepared apparatus source mismatch")
    for split in data:
        item = preparation_manifest[f"{split}.json"]
        if item["sha256"] != digest(args.prepared / f"{split}.json") or item["count"] != len(data[split]):
            raise ValueError("prepared input checksum/count mismatch")
    (args.root / "PREPARATION_MANIFEST.json").write_bytes((args.prepared / "MANIFEST.json").read_bytes())
    if len(data["train"]) != 64 or len(data["calibration"]) != 32 or len(data["eval"]) != 64:
        raise ValueError("frozen split counts required")
    all_ids = [r["id"] for rs in data.values() for r in rs]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split IDs overlap")
    schedule = sorted(range(64), key=lambda i: hashlib.sha256(f"{args.seed}/{data['train'][i]['id']}".encode()).hexdigest())
    write_json(args.root / "training_schedule.json", schedule)
    for s in data:
        (args.root / f"{s}.json").write_bytes((args.prepared / f"{s}.json").read_bytes())
    sources = {"runner_source.py": Path(__file__), "apparatus_source.py": Path(apparatus.__file__),
               "helper_source.py": Path(__import__("interaction_sprint.undo_gpu_training", fromlist=["x"]).__file__),
               "lora_source.py": Path(__import__("latent_contract.sender_update", fromlist=["x"]).__file__)}
    for name, path in sources.items():
        (args.root / name).write_bytes(path.read_bytes())
    (args.root / "upstream_loss_source.py").write_bytes(args.upstream_file.read_bytes().replace(b"\r\n", b"\n"))
    write_json(args.root / "freeze.json", dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        inputs={s: digest(args.prepared / f"{s}.json") for s in data}, sources={n: digest(v) for n, v in sources.items()},
        upstream_commit=UPSTREAM_COMMIT, upstream_sha=UPSTREAM_SHA, max_new_tokens=64, max_context_tokens=1024,
        native_ID_handling="retain actual generated IDs, including EOS only if emitted; no string retokenization",
        loss="exact released simple_signal, stopgradient teacher refreshed every update, no clipping, no token exclusions",
        lora=dict(rank=16, alpha=32, targets="attention q/k/v/o", dropout=0),
        optimizer=dict(name="AdamW", lr=args.lr, eps=1e-6, weight_decay=0, clip=1., steps=64),
        sampling_recompute_max_logprob_gap=0.25,
        torch=torch.__version__, transformers=transformers.__version__, endogeneity_arm=False))

    def finish(status, **extra):
        write_json(args.root / "status.json", dict(status=status, elapsed_seconds=time.time() - started, **extra))
        write_json(args.root / "MANIFEST.json", {str(f.relative_to(args.root)): digest(f) for f in args.root.rglob("*") if f.is_file()})

    files = {str(f.relative_to(args.model_path)): digest(f) for f in args.model_path.rglob("*")
             if f.is_file() and f.suffix in (".json", ".safetensors", ".model", ".txt")}
    write_json(args.root / "model.json", dict(revision=args.revision, sha256=files))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    write_json(args.root / "generation_config_used.json", {str(sample): generation_config(tokenizer, sample, 64).to_dict()
                                                          for sample in (False, True)})
    model, loading = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa", output_loading_info=True)
    write_json(args.root / "loading_info.json", loading_metadata_json(loading))
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        finish("INVALID_MODEL_LOADING"); return
    model = model.to(args.device).eval()
    write_json(args.root / "generation_config_checkpoint_unused.json", model.generation_config.to_dict())

    def run_eval(records, filename, field="prompt", offset=0):
        result = []
        with open(args.root / filename, "x", encoding="utf-8") as f:
            for i, record in enumerate(records):
                response = generate(model, tokenizer, tokenize(tokenizer, record[field], 1024), args.device,
                                    args.seed + offset + i, False, 64)
                row = dict(id=record["id"], user_id=record["user_id"], preference=record["preference"],
                           score=apparatus.score(response["text"], record), **response)
                result.append(row); f.write(json.dumps(row) + "\n"); f.flush()
        return result

    original = run_eval(data["calibration"], "calibration_original.jsonl")
    explicit = run_eval(data["calibration"], "calibration_explicit.jsonl", "calibration_prompt")
    calibration_teacher = []
    for record, response in zip(data["calibration"], original):
        teacher = dict(record)
        teacher["prompt"] = hindsight_messages(record["prompt"], apparatus.feedback(response["text"], record))
        calibration_teacher.append(teacher)
    teacher_rows = run_eval(calibration_teacher, "calibration_teacher.jsonl")
    qualified = calibration_decision(explicit, teacher_rows, original)
    write_json(args.root / "qualification.json", qualified)
    if not qualified["qualified"]:
        finish("UNQUALIFIED_GENERATIVE_APPARATUS"); return
    if summary(original)["joint"] > .85:
        finish("INSUFFICIENT_ADAPTATION_HEADROOM", scientific_decision=None); return
    before = run_eval(data["eval"], "baseline_eval.jsonl", offset=10000)
    with torch.no_grad():
        test_context = tokenize(tokenizer, data["train"][0]["prompt"], 1024)
        ref = completion_logps(model, test_context, [tokenizer.eos_token_id], args.device, False).cpu()
    install_lora(model, rank=16, alpha=32)
    initial = adapter_state(model); torch.save(initial, args.root / "initial_adapter.pt")
    diff = float((completion_logps(model, test_context, [tokenizer.eos_token_id], args.device, False).cpu() - ref).abs().max())
    write_json(args.root / "adapter_insertion.json", dict(logprob_error=diff, tolerance=1e-5))
    if diff > 1e-5:
        finish("INVALID_ADAPTER_INSERTION"); return
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    parameters = [x for x in model.parameters() if x.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0, eps=1e-6)
    with open(args.root / "training.jsonl", "x", encoding="utf-8") as log:
        for step, index in enumerate(schedule):
            record = data["train"][index]
            base = tokenize(tokenizer, record["prompt"], 1024)
            response = generate(model, tokenizer, base, args.device, args.seed + 20000 + step, True, 64)
            feedback = apparatus.feedback(response["text"], record)
            teacher = tokenize(tokenizer, hindsight_messages(record["prompt"], feedback), 1024)
            bridge = NativeBridge(model, args.device, base, teacher)
            optimizer.zero_grad(set_to_none=True)
            # Qwen3 dropout is zero; eval disables stochastic layers while retaining gradients.
            model.eval(); synchronize(args.device); update_start = time.time()
            loss, metrics = loss_cls._simple_signal_loss(bridge, "base", "teacher",
                                     torch.tensor([response["completion_ids"]], device=args.device))
            sampling_gap = max(abs(a - b) for a, b in zip(response["token_logprobs"], bridge.saved["base"]))
            if len(response["token_logprobs"]) != len(bridge.saved["base"]) or sampling_gap > .25:
                failure = dict(step=step, id=record["id"], response=response, feedback=feedback,
                    teacher_prompt_ids=teacher, training_logprobs=bridge.saved, max_abs_logprob_gap=sampling_gap)
                write_json(args.root / "sampling_mismatch.json", failure)
                finish("INVALID_SAMPLING_RECOMPUTE", tolerance=.25); return
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step(); synchronize(args.device)
            row = dict(step=step, id=record["id"], user_id=record["user_id"], score=apparatus.score(response["text"], record),
                response=response, feedback=feedback, teacher_prompt_ids=teacher, training_logprobs=bridge.saved,
                metrics={k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in metrics.items()},
                metric_note="undefined one-token sample std stored as null, not zero",
                sampling_recompute_max_logprob_gap=sampling_gap,
                loss=float(loss.detach()), grad_norm=float(norm), update_seconds=time.time() - update_start)
            log.write(json.dumps(row, allow_nan=False) + "\n"); log.flush()
            if (step + 1) % 16 == 0:
                torch.save(adapter_state(model), args.root / f"adapter_step_{step + 1}.pt")
                print(json.dumps(dict(step=step + 1, total=64)), flush=True)
    torch.save(adapter_state(model), args.root / "final_adapter.pt")
    torch.save(optimizer.state_dict(), args.root / "final_optimizer.pt")
    after = run_eval(data["eval"], "final_eval.jsonl", offset=10000)
    write_json(args.root / "positive_control.json", positive_decision(before, after))
    finish("COMPLETE_POSITIVE_CONTROL", automatic_expansion=False)


if __name__ == "__main__":
    main()
