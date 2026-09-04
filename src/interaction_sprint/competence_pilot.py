"""CPU capability-controlled feedback learning; synthetic, not full SDPO."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from .parameter_probe import MODEL, REVISION
from .feedback_gradient_audit import gradients
from .factorial_probe import feedback_channel
from latent_contract.sender_update import install_lora, adapter_state, load_adapter

SPEC = dict(schema="hindsight-competence-pilot-v1", model=MODEL, revision=REVISION,
            seed=9046401, threads=4, device="cpu", dtype="float32", max_tokens=256,
            rank=4, alpha=8, warm_epochs=2, warm_lr=0.0001,
            adaptation_steps=8, batch_size=8, adaptation_lr=0.01, gradient_clip=1.,
            copy_strengths=[0., .9], reference="truth90",
            methods=["own", "full", "projected_full", "anchor_only"],
            anchor_per_batch=2, teacher="competent checkpoint, frozen during adaptation",
            qualification_accuracy=.9, qualification_domain_accuracy=.75,
            qualification_min_AB_mass=.95, paper_green_light=False, automatic_expansion=False)


def dataset():
    rng = random.Random(SPEC["seed"])
    result = []
    for domain, item, unit in [("price", "item costing", "dollars"),
                               ("duration", "job lasting", "minutes"),
                               ("capacity", "drive holding", "gigabytes"),
                               ("distance", "destination", "kilometers away")]:
        pairs = rng.sample([(a, b) for a in range(10, 75) for b in range(a+5, 100)], 24)
        offset = 0
        for split, count in [("warm", 8), ("qualify", 4), ("adapt", 4), ("eval", 8)]:
            for i, (low, high) in enumerate(pairs[offset:offset+count]):
                higher = i % 2 == 1
                right, wrong = (high, low) if higher else (low, high)
                for swap in (0, 1):
                    values = [wrong, right] if swap else [right, wrong]
                    options = [f"the {item} {v} {unit}" for v in values]
                    direction = "higher" if higher else "lower"
                    need = f"My only preference is the {direction} {domain}; ignore other considerations."
                    # The final evaluation uses a distinct rendering, not distinct task semantics.
                    prompt = (f"{need}\nChoose exactly one option.\nA: {options[0]}\nB: {options[1]}\nReturn only A or B."
                              if split != "eval" else
                              f"Select according to my preference: {direction} {domain}.\nThe alternatives are:\nA) {options[0]}\nB) {options[1]}\nReply with just A or B.")
                    result.append(dict(id=f"{split}-{domain}-{i}-{swap}", split=split, domain=domain,
                        pair=[low, high], higher=higher, answer=swap, options=options, prompt=prompt))
            offset += count
    return result


def schedule(data, steps):
    """Eight distinct contexts per step; same fixed order and anchors in every arm."""
    rng = random.Random(SPEC["seed"]+1)
    result = []
    while len(result) < steps:
        order = list(data); rng.shuffle(order)
        result.extend([order[i:i+8] for i in range(0, len(order), 8)])
    return result[:steps]


def project_ascent(direction, anchor):
    """One standard half-space projection, not a proposed novel algorithm."""
    a, g = anchor.double(), direction.double()
    dot = torch.dot(a, g)
    if dot < 0 and torch.dot(a, a) > 1e-20:
        g -= dot / torch.dot(a, a) * a
    return g.to(direction.dtype)


def summarize(rows):
    def accuracy(rs):
        return float(np.mean([np.argmax(r["probabilities"]) == r["answer"] for r in rs]))
    return dict(n=len(rows), accuracy=accuracy(rows),
        nll=float(np.mean([-np.log(r["probabilities"][r["answer"]]) for r in rows])),
        min_AB_mass=min(r["AB_mass"] for r in rows),
        domains={d: accuracy([r for r in rows if r["domain"] == d]) for d in sorted({r["domain"] for r in rows})})


def qualifies(rows):
    s = summarize(rows)
    return (s["accuracy"] >= SPEC["qualification_accuracy"] and
            min(s["domains"].values()) >= SPEC["qualification_domain_accuracy"] and
            s["min_AB_mass"] >= SPEC["qualification_min_AB_mass"])


def run(root):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (root/name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    started = time.time()
    counts = dict(forwards=0, backwards=0, warm_updates=0, feedback_updates=0)
    data = dataset()
    write("spec.json", SPEC); write("cases.json", data)
    (root/"runner_source.py").write_bytes(Path(__file__).read_bytes())
    write("sources.json", {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [Path(__file__), Path(__file__).with_name("feedback_gradient_audit.py"),
         Path(__file__).with_name("factorial_probe.py"), Path(__file__).parents[1]/"latent_contract"/"sender_update.py"]})
    def finish(decision, results):
        write("RESULT.json", dict(decision=decision, paper_green_light=False, counts=counts,
             elapsed_seconds=time.time()-started, **results))
        write("COMPLETE.json", counts)
        write("MANIFEST.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps({"decision": decision, "counts": counts, "elapsed_seconds": time.time()-started}), flush=True)
    try:
        torch.set_num_threads(SPEC["threads"]); torch.manual_seed(SPEC["seed"])
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                    dtype=torch.float32, attn_implementation="eager").eval()
        abids = [tok.encode(x, add_special_tokens=False) for x in ("A", "B")]
        if any(len(x) != 1 for x in abids) or abids[0] == abids[1]:
            raise ValueError("invalid answer tokens")
        abids = torch.tensor([x[0] for x in abids])
        encoded, prompts = {}, []
        for c in data:
            texts = {"base": c["prompt"]}
            if c["split"] == "adapt":
                for o in (0, 1):
                    texts[str(o)] = (c["prompt"] + "\n\n=== HINDSIGHT CONTEXT ===\n"
                        "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
                        f"Thanks, I prefer {c['options'][o]}.")
            for kind, text in texts.items():
                ids = tok.apply_chat_template([dict(role="user", content=text)], tokenize=True,
                        return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if len(ids) > SPEC["max_tokens"]:
                    raise ValueError("overlength; never truncate")
                encoded[c["id"], kind] = torch.tensor([ids])
                prompts.append(dict(id=c["id"], kind=kind, text=text, token_ids=ids))
        write("prompts.json", prompts)
        def forward(c, kind="base"):
            counts["forwards"] += 1
            h = model.model(input_ids=encoded[c["id"], kind], use_cache=False).last_hidden_state[:, -1]
            z = model.get_output_embeddings()(h)[0]
            ab = z[abids]
            # float64 softmax avoids saturating probabilities before exact expectation calculations.
            return ab, dict(id=c["id"], domain=c["domain"], answer=c["answer"],
                probabilities=ab.detach().double().softmax(-1).tolist(),
                AB_mass=float(torch.exp(torch.logsumexp(ab.detach().double(), 0)-torch.logsumexp(z.detach().double(), 0))))
        def evaluate(rows):
            with torch.no_grad():
                return [forward(c)[1] for c in rows]
        with torch.no_grad():
            before = forward(data[0])[0]
        modules = install_lora(model, rank=SPEC["rank"], alpha=SPEC["alpha"])
        with torch.no_grad():
            after = forward(data[0])[0]
        if not torch.equal(before, after):
            raise ValueError("zero adapter failed")
        params = [p for p in model.parameters() if p.requires_grad]
        write("runtime.json", dict(torch=torch.__version__, transformers=transformers.__version__,
            platform=platform.platform(), threads=torch.get_num_threads(), modules=modules,
            trainable_parameters=sum(p.numel() for p in params), zero_adapter_exact=True))
        torch.save(adapter_state(model), root/"initial_adapter.pt")
        splits = {s: [c for c in data if c["split"] == s] for s in ("warm", "qualify", "adapt", "eval")}
        write("base_qualification.json", evaluate(splits["qualify"]))
        opt = torch.optim.AdamW(params, lr=SPEC["warm_lr"], weight_decay=0.)
        warm_order = schedule(splits["warm"], len(splits["warm"])*SPEC["warm_epochs"]//8)
        adaptation_order = schedule(splits["adapt"], SPEC["adaptation_steps"])
        write("schedule.json", dict(warm=[[c["id"] for c in batch] for batch in warm_order],
              adapt=[[c["id"] for c in batch] for batch in adaptation_order], anchor_positions=[0, 1]))
        warm_logs = []
        for step, batch in enumerate(warm_order):
            opt.zero_grad(set_to_none=True); losses = []
            for c in batch:
                ab, _ = forward(c)
                loss = -ab.log_softmax(0)[c["answer"]]/len(batch)
                loss.backward(); counts["backwards"] += 1
                losses.append(float(loss.detach())*len(batch))
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            opt.step(); counts["warm_updates"] += 1
            warm_logs.append(dict(step=step+1, loss=float(np.mean(losses)), gradient_norm=float(norm)))
        competent = adapter_state(model)
        torch.save(competent, root/"competent_adapter.pt")
        torch.save(opt.state_dict(), root/"warm_optimizer.pt")
        write("warm_steps.json", warm_logs)
        qualification = evaluate(splits["qualify"])
        write("qualification.json", qualification)
        if not qualifies(qualification):
            finish("INVALID_COMPETENCE_NO_FEEDBACK_TRAINING", {"qualification": summarize(qualification)})
            return
        baseline = evaluate(splits["eval"])
        write("competent_eval.json", baseline)
        teachers = {}
        with torch.no_grad():
            for c in splits["adapt"]:
                teachers[c["id"]] = [forward(c, str(o))[1] for o in (0, 1)]
        write("teachers.json", teachers)
        results = {"competent": summarize(baseline)}
        for rho in SPEC["copy_strengths"]:
            for method in SPEC["methods"]:
                load_adapter(model, competent)
                arm = f"rho{rho}_{method}"
                logs = []
                for step, batch in enumerate(adaptation_order):
                    g = torch.zeros(sum(p.numel() for p in params)); anchor = torch.zeros_like(g)
                    detail = []
                    for i, c in enumerate(batch):
                        if method == "anchor_only" and i >= SPEC["anchor_per_batch"]:
                            continue
                        ab, row = forward(c)
                        js = torch.autograd.grad(ab[1]-ab[0], params)
                        counts["backwards"] += 1
                        j = torch.cat([v.detach().reshape(-1) for v in js])
                        p = row["probabilities"][1]
                        if i < SPEC["anchor_per_batch"]:
                            anchor += j*((c["answer"]-p)/SPEC["anchor_per_batch"])
                        if method != "anchor_only":
                            q = np.array([r["probabilities"] for r in teachers[c["id"]]])
                            direction = gradients(p, feedback_channel(c["answer"], rho, SPEC["reference"]), q)
                            coefficient = direction["sampled_response_ascent" if method == "own" else "full_reverse_kl_ascent"]
                            g += j*(coefficient/len(batch))
                            row["ascent_coefficient"] = coefficient
                        row["anchor_available"] = i < SPEC["anchor_per_batch"]
                        detail.append(row)
                    original_norm = float(g.double().norm())
                    dot = float(torch.dot(g.double(), anchor.double()))
                    if method == "projected_full":
                        g = project_ascent(g, anchor)
                    elif method == "anchor_only":
                        g = anchor
                    norm = float(g.double().norm())
                    if not np.isfinite(norm):
                        raise ValueError("nonfinite gradient")
                    scale = SPEC["adaptation_lr"] * min(1., SPEC["gradient_clip"]/max(norm, 1e-30))
                    offset = 0
                    with torch.no_grad():
                        for p in params:
                            n = p.numel(); p.add_(g[offset:offset+n].reshape_as(p), alpha=scale); offset += n
                    counts["feedback_updates"] += 1
                    logs.append(dict(step=step+1, original_gradient_norm=original_norm, anchor_dot_before=dot,
                         applied_gradient_norm=norm, scale=scale, anchor_dot_after=float(torch.dot(g.double(), anchor.double())), rows=detail))
                torch.save(adapter_state(model), root/f"{arm}_adapter.pt")
                write(f"{arm}_steps.json", logs)
                rows = evaluate(splits["eval"]); write(f"{arm}_eval.json", rows)
                results[arm] = summarize(rows)
        finish("DEVELOPMENT_LEARNING_COMPLETE_REQUIRES_ANALYSIS", {"qualification": summarize(qualification), "metrics": results})
    except Exception as exc:
        write("FAILED.json", dict(exception=type(exc).__name__, message=str(exc), counts=counts))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
