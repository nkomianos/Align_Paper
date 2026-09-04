"""CPU pretrained-model, restricted-action, one-step feedback-gradient probe.

Not full SDPO training, a population study, or an effective correction method.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from .feedback_gradient_audit import gradients
from latent_contract.sender_update import install_lora, adapter_state, load_adapter

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SPEC = {"schema": "hindsight-parameter-probe-v1", "model": MODEL, "revision": REVISION,
        "seed": 9046201, "threads": 4, "device": "cpu", "dtype": "float32",
        "lora_rank": 4, "lora_alpha": 8, "maximum_prompt_tokens": 256,
        "copy_strengths": [0., .5, .9], "one_step_copy_strength": .5,
        "one_step_parameter_L2": .1, "teacher": "current policy with fixed future message",
        "teacher_refresh": "only before the one step", "null_truthful_feedback_probability": .9,
        "objective": "restricted A/B categorical policy, exact action-feedback expectation",
        "scope": "PRETRAINED_CPU_ONE_STEP_DIAGNOSTIC_NOT_FULL_SDPO_TRAINING",
        "new_generations": 0, "automatic_expansion": False}


def cases():
    tasks = {
        "train": [("storage", "The stored data must survive a power outage.", "disk storage", "volatile RAM"),
                  ("trip", "I want a quiet rural location, not city nightlife.", "a rural cottage", "a city nightclub district"),
                  ("appointment", "I am available only before noon.", "a 9 AM appointment", "a 6 PM appointment"),
                  ("purchase", "Choose the less expensive item; price is my only criterion.", "the $20 item", "the $80 item")],
        "heldout": [("file", "Keep the file available after the computer restarts.", "save to disk", "keep only in RAM"),
                    ("hotel", "I want to avoid a noisy neighborhood.", "a quiet countryside hotel", "a hotel above a nightclub"),
                    ("call", "Schedule the call strictly after 5 PM.", "a 7 PM call", "a 10 AM call"),
                    ("ticket", "Minimize the price of my ticket; nothing else matters.", "a $15 ticket", "a $60 ticket")],
    }
    result = []
    for split, definitions in tasks.items():
        for name, need, right, wrong in definitions:
            for swap in (False, True):
                options = [wrong, right] if swap else [right, wrong]
                result.append({"id": f"{split}-{name}-{int(swap)}", "split": split,
                               "prompt": f"{need}\nChoose exactly one option.\nA: {options[0]}\nB: {options[1]}\nReturn only A or B.",
                               "options": options, "answer": int(swap)})
    return result


def flat_gradient(scalar, parameters):
    values = torch.autograd.grad(scalar, parameters)
    return torch.cat([g.detach().reshape(-1) for g in values])


def apply_direction(parameters, direction, length):
    if not torch.isfinite(direction).all() or direction.norm() <= 1e-12:
        raise ValueError("invalid or zero direction")
    if sum(p.numel() for p in parameters) != direction.numel():
        raise ValueError("parameter dimension mismatch")
    delta = direction * (length / direction.norm())
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            n = parameter.numel()
            parameter.add_(delta[offset:offset+n].reshape_as(parameter))
            offset += n
    return float(delta.norm())


def cosine(a, b):
    denom = float(a.norm()*b.norm())
    return None if denom <= 1e-20 else float(torch.dot(a, b)/denom)


def channel(answer, copy_strength):
    truthful = np.array([.1, .1]); truthful[answer] = .9
    return (1-copy_strength)*np.tile(truthful, (2, 1)) + copy_strength*np.eye(2)


def run(root):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root.mkdir(parents=True, exist_ok=False)
    def write(name, value):
        with (root/name).open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
    write("spec.json", SPEC)
    write("cases.json", cases())
    source = Path(__file__).read_bytes()
    (root/"probe_source.py").write_bytes(source)
    write("source_digests.json", {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(__file__).with_name("feedback_gradient_audit.py"),
                                  Path(__file__).parents[1]/"latent_contract"/"sender_update.py")})
    started = time.time()
    try:
        torch.set_num_threads(SPEC["threads"])
        torch.manual_seed(SPEC["seed"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                                                    dtype=torch.float32, attn_implementation="eager")
        model.eval()
        model.config.use_cache = False
        ids = [tokenizer.encode(s, add_special_tokens=False) for s in ("A", "B")]
        if any(len(x) != 1 for x in ids) or ids[0] == ids[1]:
            raise ValueError("A/B must be distinct single tokens")
        choice_ids = torch.tensor([x[0] for x in ids])
        # Compute an exact pre-install no-op check before training any adapter.
        rows, encoded = [], {}
        for c in cases():
            texts = {"base": c["prompt"]}
            if c["split"] == "train":
                for o in (0, 1):
                    texts[f"teacher{o}"] = (c["prompt"] + "\n\n=== HINDSIGHT CONTEXT ===\n"
                        "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
                        f"Thanks, I prefer {c['options'][o]}.")
            for arm, text in texts.items():
                tokens = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                         tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if len(tokens) > SPEC["maximum_prompt_tokens"]:
                    raise ValueError("overlength prompt; no truncation")
                encoded[c["id"], arm] = torch.tensor([tokens])
                rows.append({"id": c["id"], "arm": arm, "text": text, "token_ids": tokens})
        write("prompts.json", rows)
        counters = {"forwards": 0, "backwards": 0}

        def logits(c, arm="base"):
            counters["forwards"] += 1
            hidden = model.model(input_ids=encoded[c["id"], arm], use_cache=False).last_hidden_state[:, -1]
            return model.get_output_embeddings()(hidden)[0]

        def probabilities(c, arm="base"):
            z = logits(c, arm)
            ab = z[choice_ids]
            p = ab.softmax(-1)
            mass = torch.exp(torch.logsumexp(ab, 0)-torch.logsumexp(z, 0))
            return ab, p, float(mass.detach())

        with torch.no_grad():
            no_adapter = probabilities(cases()[0])[1].clone()
        layers = install_lora(model, rank=SPEC["lora_rank"], alpha=SPEC["lora_alpha"])
        with torch.no_grad():
            zero_adapter = probabilities(cases()[0])[1].clone()
        if not torch.equal(no_adapter, zero_adapter):
            raise ValueError("zero adapter changed output")
        initial = adapter_state(model)
        torch.save(initial, root/"initial_adapter.pt")
        parameters = [p for p in model.parameters() if p.requires_grad]
        write("runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
                              "platform": platform.platform(), "threads": torch.get_num_threads(),
                              "trainable_parameters": sum(p.numel() for p in parameters),
                              "adapter_modules": layers, "zero_adapter_exact": True})
        print("Model loaded; zero-adapter control passed. Starting parameter probe.", flush=True)
        directions = {(r, name): torch.zeros(sum(p.numel() for p in parameters))
                      for r in SPEC["copy_strengths"] for name in ("own", "full")}
        task_gradient = torch.zeros_like(next(iter(directions.values())))
        baseline, details = [], []
        n_train = sum(c["split"] == "train" for c in cases())
        for c in cases():
            if c["split"] == "train":
                with torch.no_grad():
                    teacher = np.stack([probabilities(c, f"teacher{o}")[1].numpy() for o in (0, 1)])
                ab, p_tensor, mass = probabilities(c)
                p = p_tensor.detach().numpy().astype(float); p /= p.sum()
                jacobian = flat_gradient(ab[1]-ab[0], parameters)
                counters["backwards"] += 1
                for rho in SPEC["copy_strengths"]:
                    g = gradients(float(p[1]), channel(c["answer"], rho), teacher)
                    directions[rho, "own"] += jacobian*(g["sampled_response_ascent"]/n_train)
                    directions[rho, "full"] += jacobian*(g["full_reverse_kl_ascent"]/n_train)
                    details.append({"id": c["id"], "rho": rho, "teacher": teacher.tolist(), **g})
                task_gradient += jacobian*((c["answer"]-float(p[1]))/n_train)
            else:
                with torch.no_grad():
                    _, p_tensor, mass = probabilities(c)
                p = p_tensor.numpy().astype(float); p /= p.sum()
            baseline.append({"id": c["id"], "split": c["split"], "answer": c["answer"],
                             "probabilities": p.tolist(), "AB_mass": mass})
        write("baseline.json", baseline)
        write("per_context_gradients.json", details)
        null_error = float((directions[0., "own"]-directions[0., "full"]).norm())
        if null_error > 1e-6*(1+float(directions[0., "full"].norm())):
            raise ValueError("independent-feedback parameter null failed")
        write("parameter_geometry.json", [{"rho": r, "cosine_own_full": cosine(directions[r, "own"], directions[r, "full"]),
                      "own_norm": float(directions[r, "own"].norm()), "full_norm": float(directions[r, "full"].norm()),
                      "cosine_own_original_task": cosine(directions[r, "own"], task_gradient),
                      "cosine_full_original_task": cosine(directions[r, "full"], task_gradient)} for r in SPEC["copy_strengths"]])
        torch.save({f"rho{r}_{name}": g for (r, name), g in directions.items()}, root/"directions.pt")
        evaluations = {}
        for name in ("own", "full"):
            load_adapter(model, initial)
            delta_norm = apply_direction(parameters, directions[SPEC["one_step_copy_strength"], name], SPEC["one_step_parameter_L2"])
            torch.save(adapter_state(model), root/f"{name}_one_step_adapter.pt")
            updated = []
            for c in cases():
                with torch.no_grad():
                    _, p, mass = probabilities(c)
                updated.append({"id": c["id"], "split": c["split"], "answer": c["answer"],
                                "probabilities": p.tolist(), "AB_mass": mass})
            write(f"{name}_one_step.json", {"delta_norm": delta_norm, "rows": updated})
            evaluations[name] = updated
        load_adapter(model, initial)
        def metrics(rs, split):
            subset = [r for r in rs if r["split"] == split]
            return {"n": len(subset), "accuracy": float(np.mean([np.argmax(r["probabilities"]) == r["answer"] for r in subset])),
                    "original_request_NLL": float(np.mean([-np.log(r["probabilities"][r["answer"]]) for r in subset])),
                    "minimum_AB_mass": min(r["AB_mass"] for r in subset)}
        result = {"scope": SPEC["scope"], "paper_green_light": False,
                  "metrics": {name: {s: metrics(rs, s) for s in ("train", "heldout")}
                              for name, rs in {"baseline": baseline, **evaluations}.items()},
                  "counts": counters, "elapsed_seconds": time.time()-started,
                  "warning": "One norm-matched step; designed feedback; tiny handwritten split; not SDPO training or human preference change."}
        write("RESULT.json", result)
        write("COMPLETE.json", counters)
        write("MANIFEST.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()})
        print(json.dumps(result), flush=True)
    except Exception as exc:
        write("FAILED.json", {"exception": type(exc).__name__, "message": str(exc), "elapsed_seconds": time.time()-started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
