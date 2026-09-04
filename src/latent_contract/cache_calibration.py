"""Collect matched public calibration caches and fit declared repair baselines."""
import hashlib
import json
from pathlib import Path
import time

import torch

from latent_contract.cache_repair import (METHODS, sample_positions, fit_head_map,
                                        apply_head_map, retune_projector)


def cache_pair(cache, index):
    # Production is pinned to 4.52.4; the second branch permits current local
    # tiny-model tests without relaxing that runner dependency check.
    if hasattr(cache, "key_cache"):
        return cache.key_cache[index], cache.value_cache[index]
    if hasattr(cache, "layers"):
        return cache.layers[index].keys, cache.layers[index].values
    raise TypeError("unsupported cache representation")


def calibration_split(cases, fit_per_dataset=48):
    groups = {}
    for row in cases:
        if "answer" in row:
            raise ValueError("calibration must be unlabeled")
        groups.setdefault(row["dataset"], []).append(row["case_id"])
    if len({r["case_id"] for r in cases}) != len(cases):
        raise ValueError("duplicate calibration case")
    chosen = set()
    for ids in groups.values():
        if len(ids) <= fit_per_dataset:
            raise ValueError("need held-out calibration checks in each dataset")
        ordered = sorted(ids, key=lambda cid: hashlib.sha256(("repair-fit-v1:"+cid).encode()).hexdigest())
        chosen.update(ordered[:fit_per_dataset])
    return chosen


def collect_bank(models, tokenizers, cases, build_prompt, mapping, output,
                 fit_per_dataset=48, maximum_tokens=32):
    """Save selected B,H,N,D cache rows, never full answers or final DEV.

Use prompt[:-1], matching the first frozen-C2C prefill segment. Direct calls to
Qwen3Model omit its language-model head, which does not affect the cached K/V.
"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if set(models) != {"old", "new", "receiver"} or set(tokenizers) != set(models):
        raise ValueError("expected old/new/receiver models and tokenizers")
    fit_ids = calibration_split(cases, fit_per_dataset)
    buffers = {target: {part: {role+"_"+kind: [] for role in models for kind in ("key", "value")}
                        for part in ("fit", "check")} for target, source in mapping}
    records = []
    with torch.inference_mode():
        for case in cases:
            choices = "".join(f"{chr(65+i)}. {t}\n" for i, t in enumerate(case["choices"]))
            prompt = build_prompt("mmlu-redux", "", case["question"], choices, False, True)
            turns = [{"role": "user", "content": prompt}]
            ids = {role: tok.apply_chat_template(turns, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                   for role, tok in tokenizers.items()}
            if any(value != ids["old"] for value in ids.values()) or not 1 < len(ids["old"]) <= 2048:
                raise ValueError("calibration tokenizer contract mismatch")
            positions = sample_positions(len(ids["old"])-1, maximum_tokens)
            part = "fit" if case["case_id"] in fit_ids else "check"
            records.append({"case_id": case["case_id"], "dataset": case["dataset"], "split": part,
                            "prompt_ids": ids["old"], "selected_prefix_positions": positions})
            for role, model in models.items():
                model.eval()
                device = next(model.parameters()).device
                prefix = torch.tensor([ids[role][:-1]], device=device)
                result = model.model(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
                indices = torch.tensor(positions, device=device)
                for target, source in mapping:
                    layer = target if role == "receiver" else source
                    key, value = cache_pair(result.past_key_values, layer)
                    for kind, tensor in (("key", key), ("value", value)):
                        selected = tensor.index_select(2, indices).detach().cpu().clone()
                        if not torch.isfinite(selected).all():
                            raise ValueError("nonfinite cache")
                        buffers[target][part][role+"_"+kind].append(selected)
                del result
    for target, source in mapping:
        layer = {part: {name: torch.cat(chunks, dim=2) for name, chunks in groups.items()}
                 for part, groups in buffers[target].items()}
        torch.save(layer, output / f"layer_{target:02d}.pt")
    metadata = {"cases": records, "mapping": [list(x) for x in mapping],
                "fit_cases": len(fit_ids), "check_cases": len(cases)-len(fit_ids),
                "maximum_positions_per_case": maximum_tokens, "cache_space": "post-RoPE B,H,N,D",
                "generation_calls": 0, "backbone_forwards": 3*len(cases)}
    with (output / "bank.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return metadata


def normalized_mse(actual, expected):
    return float((actual.float()-expected.float()).square().mean() / expected.float().square().mean().clamp_min(1e-6))


def fit_layer(projector, bank, output, seed, device="cpu", steps=100,
              ridge=1e-3, retune_lr=1e-4, batch_tokens=64):
    """All repairs fit only `fit`; `check` errors are reported without selection."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = {part: {name: tensor.to(device) for name, tensor in group.items()} for part, group in bank.items()}
    errors, method_seconds = {}, {}
    for method in METHODS:
        start = time.monotonic()
        maps = {kind: fit_head_map(data["fit"]["new_"+kind], data["fit"]["old_"+kind], method, ridge=ridge)
                for kind in ("key", "value")}
        torch.save({kind: {name: tensor.cpu() for name, tensor in values.items()} for kind, values in maps.items()}, output / (method+".pt"))
        errors[method] = {kind: normalized_mse(apply_head_map(data["check"]["new_"+kind], **maps[kind]), data["check"]["old_"+kind])
                          for kind in ("key", "value")}
        method_seconds[method] = time.monotonic()-start
    def pair(part, role):
        return tuple(data[part][role+"_"+kind] for kind in ("key", "value"))
    start = time.monotonic()
    student, curve = retune_projector(projector.to(device), pair("fit", "new"), pair("fit", "old"), pair("fit", "receiver"),
                                     seed=seed, steps=steps, lr=retune_lr, batch_tokens=batch_tokens)
    torch.save({name: tensor.detach().cpu().clone() for name, tensor in student.state_dict().items()}, output / "retuned_float32.pt")
    # Compare float32 teacher/student on separate calibration checks, not final answers.
    import copy
    teacher = copy.deepcopy(projector).float().eval()
    with torch.no_grad():
        old, new, target = [tuple(x.float() for x in pair("check", role)) for role in ("old", "new", "receiver")]
        desired, original_output, updated_output = teacher(old, target), teacher(new, target), student(new, target)
    retune_check = {kind: {"before": normalized_mse(before, desired_one), "after": normalized_mse(after, desired_one)}
                   for kind, before, after, desired_one in zip(("key", "value"), original_output, updated_output, desired)}
    student.bfloat16()
    torch.save({name: tensor.detach().cpu().clone() for name, tensor in student.state_dict().items()}, output / "retuned_bfloat16.pt")
    method_seconds["retuned"] = time.monotonic()-start
    report = {"maps": errors, "retune_output_check": retune_check, "curve": curve, "fit_check_save_seconds": method_seconds,
              "selection": "none; all declared methods retained; ridge is the primary repair",
              "interpretation": "Held-out cache/output reconstruction only, not language accuracy or repair success"}
    with (output / "fit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report
