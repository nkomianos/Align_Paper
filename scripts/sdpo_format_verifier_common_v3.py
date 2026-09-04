"""Offline SDPO evidence verifier; no neural forwards or experiment mutation."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch

APPARATUS_SHA = "140e93dd3b6b04d98b64723735ee20886bd554b2c54d8784e714cfeb26bab297"
UPSTREAM_SHA = "6d9b92726fffa857e02d3a3773a309d8418433c50f33f91252e760b17fec6b8d"
UPSTREAM_COMMIT = "3b17d2a67bd2565b9fbda495fd16a485406aa954"
HINDSIGHT = "\n\n=== HINDSIGHT CONTEXT ===\n[The following is a future user message. Use this to guide your answer to the user prompt.]\n{follow_up}"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def near(a, b, name, atol=2e-5):
    if a is None or b is None:
        if a is not None or b is not None:
            raise ValueError(f"undefined metric mismatch: {name}")
    elif not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, rtol=2e-5, atol=atol):
        raise ValueError(f"numeric mismatch: {name}: {a} != {b}")


def hindsight(messages, feedback):
    result = copy.deepcopy(messages)
    for message in reversed(result):
        if message["role"] == "user":
            message["content"] += HINDSIGHT.format(follow_up=feedback.strip())
            return result
    raise ValueError("missing user message")


def stats(rows):
    return {key: float(np.mean([r["score"][key] for r in rows])) for key in ("joint", "format_valid", "content_valid")}


def qualification(original, explicit, teacher):
    a, b = stats(explicit), stats(teacher)
    failures = [i for i, row in enumerate(original) if not row["score"]["joint"]]
    recovery = float(np.mean([teacher[i]["score"]["joint"] for i in failures])) if failures else None
    return {"qualified": a["joint"] >= .9 and b["content_valid"] >= .9 and (recovery is None or recovery >= .75),
            "explicit": a, "hindsight": b, "mismatches": len(failures), "mismatch_recovery": recovery}


def released_loss(source):
    if hashlib.sha256(source).hexdigest() != UPSTREAM_SHA:
        raise ValueError("upstream loss not pinned source")
    parent = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "OnlineSDPOUpdater")
    method = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "_simple_signal_loss"]
    if len(method) != 1:
        raise ValueError("loss method missing")
    tree = ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name="Loss", bases=[], keywords=[], body=method, decorator_list=[])], type_ignores=[]))
    namespace = {"torch": torch, "Tuple": Tuple, "Dict": Dict}
    exec(compile(tree, "verified_pinned_loss", "exec"), namespace)
    return namespace["Loss"]


def verify_loss(row, loss_class):
    n = len(row["response"]["completion_ids"])
    base = torch.tensor([row["training_logprobs"]["base"]], dtype=torch.float32, requires_grad=True)
    teacher = torch.tensor([row["training_logprobs"]["teacher"]], dtype=torch.float32)
    if base.shape != (1, n) or teacher.shape != (1, n) or not torch.isfinite(base).all() or not torch.isfinite(teacher).all() or (base > 1e-6).any() or (teacher > 1e-6).any():
        raise ValueError("invalid saved completion logprobs")
    stub = SimpleNamespace(config=SimpleNamespace(signal_clip=0), _log_token_table=lambda *a: None)
    stub._compute_token_logprobs = lambda name, *a, **kw: (base if name == "base" else teacher, torch.ones_like(base, dtype=torch.long), None)
    loss, metrics = loss_class._simple_signal_loss(stub, "base", "teacher", torch.tensor([row["response"]["completion_ids"]]))
    near(row["loss"], float(loss.detach()), "released loss")
    if set(row["metrics"]) != set(metrics):
        raise ValueError("released metric keys differ")
    for key, value in metrics.items():
        value = value if np.isfinite(value) else None
        near(row["metrics"][key], value, key)
    expected_gradient = -(teacher - base.detach()) / n
    if not torch.allclose(torch.autograd.grad(loss, base)[0], expected_gradient, atol=1e-7, rtol=1e-6):
        raise ValueError("stopped-advantage gradient mismatch")
    if metrics["completion_tokens"] != n:
        raise ValueError("masked/excluded completion tokens")
    if not np.isfinite(row["grad_norm"]) or row["grad_norm"] < 0 or not np.isfinite(row["update_seconds"]) or row["update_seconds"] < 0:
        raise ValueError("invalid training metrics")


def paired_users(before, after):
    index = {r["id"]: r for r in after}
    if len(index) != len(after) or {r["id"] for r in before} != set(index):
        raise ValueError("unpaired final evaluation")
    groups = {}
    for row in before:
        groups.setdefault(row["user_id"], []).append(int(index[row["id"]]["score"]["joint"]) - int(row["score"]["joint"]))
    means = np.array([np.mean(groups[uid]) for uid in sorted(groups)])
    rng = np.random.default_rng(9048316)
    draws = means[rng.integers(0, len(means), (10000, len(means)))].mean(axis=1)
    return {"users": len(groups), "cases": len(before), "joint_gain_pp": 100 * float(means.mean()),
            "user_level_gains_pp": {uid: 100 * float(np.mean(values)) for uid, values in groups.items()},
            "descriptive_user_cluster_bootstrap_95_pp": [float(v) for v in 100 * np.quantile(draws, [.025, .975])]}


def validate_native_generation(response, tokenizer, expected_prompt, vocab_size, sample, expected_seed):
    prompt, ids, logps = response["prompt_ids"], response["completion_ids"], response["token_logprobs"]
    if prompt != expected_prompt or not 0 < len(prompt) <= 1024 or not 0 < len(ids) <= 64 or len(logps) != len(ids):
        raise ValueError("native prompt/generation length mismatch")
    if any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
        raise ValueError("invalid emitted token ID")
    if not np.isfinite(logps).all() or any(p > 1e-6 for p in logps):
        raise ValueError("invalid token logprobs")
    eos = tokenizer.eos_token_id
    if eos in ids[:-1] or response["terminated"] != (ids[-1] == eos) or (not response["terminated"] and len(ids) != 64):
        raise ValueError("EOS/termination mismatch")
    if response["text"] != tokenizer.decode(ids, skip_special_tokens=True) or response["raw_decoded"] != tokenizer.decode(ids, skip_special_tokens=False):
        raise ValueError("decoded text mismatch")
    if response["sampling"] != sample or response["seed"] != expected_seed or not np.isfinite(response["elapsed_seconds"]) or response["elapsed_seconds"] < 0:
        raise ValueError("generation configuration mismatch")


def checkpoints(root, config):
    initial = torch.load(root / "initial_adapter.pt", map_location="cpu", weights_only=True)
    final = torch.load(root / "final_adapter.pt", map_location="cpu", weights_only=True)
    optimizer = torch.load(root / "final_optimizer.pt", map_location="cpu", weights_only=True)
    if not initial or set(initial) != set(final):
        raise ValueError("adapter parameter keys differ")
    delta = 0.0
    for key, a in initial.items():
        b = final[key]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor) or a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("adapter tensors invalid")
        delta += float((a.float() - b.float()).square().sum())
    group_ids = []
    for group in optimizer["param_groups"]:
        group_ids.extend(group["params"])
        for key in ("lr", "eps", "weight_decay"):
            near(group[key], config[key], f"optimizer {key}")
    if len(group_ids) != len(set(group_ids)) or len(group_ids) != len(initial) or set(group_ids) != set(optimizer["state"]):
        raise ValueError("optimizer state does not cover adapter parameters")
    for value in optimizer["state"].values():
        if float(value["step"]) != 64:
            raise ValueError("optimizer update count mismatch")
        for field in ("exp_avg", "exp_avg_sq"):
            if not torch.isfinite(value[field]).all():
                raise ValueError("nonfinite optimizer state")
    return {"adapter_tensors": len(initial), "squared_parameter_change": delta,
            "optimizer_steps_verified": 64, "optimizer_update_replay": False}


def load_apparatus(root, apparatus_sha=APPARATUS_SHA, checker_sha=None):
    if digest(root / "apparatus_source.py") != apparatus_sha:
        raise ValueError("apparatus source differs from pinned version")
    namespace = {"__name__": "frozen_apparatus", "__file__": str(root / "apparatus_source.py")}
    source = (root / "apparatus_source.py").read_bytes()
    if checker_sha is None:
        exec(compile(source, "frozen_apparatus", "exec"), namespace)
    else:
        if digest(root / "checker_source.py") != checker_sha:
            raise ValueError("strict checker dependency differs")
        checker = {"__name__": "frozen_checker", "__file__": str(root / "checker_source.py")}
        exec(compile((root / "checker_source.py").read_bytes(), "frozen_checker", "exec"), checker)
        tree = ast.parse(source)
        imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
                   and len(node.names) == 1 and node.names[0].name == "sdpo_format_control_data" and node.names[0].asname == "CHECKER"]
        if len(imports) != 1:
            raise ValueError("unexpected strict checker import")
        tree.body.remove(imports[0])
        namespace["CHECKER"] = SimpleNamespace(**checker)
        exec(compile(ast.fix_missing_locations(tree), "frozen_apparatus_with_verified_checker", "exec"), namespace)
    return namespace


def verify(root, tokenizer_path, model_path=None, *, apparatus_sha=APPARATUS_SHA, checker_sha=None):
    root, tokenizer_path = Path(root), Path(tokenizer_path)
    manifest = read(root / "MANIFEST.json")
    required = {"freeze.json", "status.json", "model.json", "loading_info.json", "PREPARATION_MANIFEST.json", "training_schedule.json",
                "runner_source.py", "apparatus_source.py", "lora_source.py", "upstream_loss_source.py", "train.json", "eval.json", "calibration.json", "released_loss_reference.json"}
    if not required.issubset(manifest):
        raise ValueError("missing frozen evidence")
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"manifest mismatch: {name}")
    freeze, state, preparation, model = (read(root / name) for name in ("freeze.json", "status.json", "PREPARATION_MANIFEST.json", "model.json"))
    reference = read(root / "released_loss_reference.json")
    if not np.isfinite([reference["loss_error"], reference["gradient_max_error"]]).all() or reference["loss_error"] > 1e-12 or reference["gradient_max_error"] > 1e-12 or reference["teacher_gradient_is_absent"] is not True:
        raise ValueError("released loss reference check failed")
    if preparation["source_sha256"] != apparatus_sha:
        raise ValueError("apparatus differs from frozen version")
    if checker_sha is not None and ("checker_source.py" not in manifest or preparation.get("checker_sha256") != checker_sha):
        raise ValueError("checker dependency not frozen in preparation")
    for name, expected in freeze["sources"].items():
        if name not in manifest or digest(root / name) != expected:
            raise ValueError("source freeze mismatch")
    if freeze["upstream_commit"] != UPSTREAM_COMMIT or freeze["upstream_sha"] != UPSTREAM_SHA or digest(root / "upstream_loss_source.py") != UPSTREAM_SHA:
        raise ValueError("upstream freeze mismatch")
    apparatus = load_apparatus(root, apparatus_sha, checker_sha)
    generated = apparatus["build"]()
    data = {s: read(root / f"{s}.json") for s in ("train", "eval", "calibration")}
    if data != generated:
        raise ValueError("prepared data differs from frozen generator")
    for split, rows in data.items():
        if digest(root / f"{split}.json") != freeze["inputs"][split] or preparation[f"{split}.json"] != {"sha256": digest(root / f"{split}.json"), "count": len(rows)}:
            raise ValueError("input freeze mismatch")
    seed = freeze["arguments"]["seed"]
    schedule = sorted(range(64), key=lambda i: hashlib.sha256(f"{seed}/{data['train'][i]['id']}".encode()).hexdigest())
    if read(root / "training_schedule.json") != schedule:
        raise ValueError("training schedule mismatch")
    if freeze["optimizer"] != {"name": "AdamW", "lr": freeze["arguments"]["lr"], "eps": 1e-6, "weight_decay": 0, "clip": 1., "steps": 64} or freeze["max_new_tokens"] != 64 or freeze["max_context_tokens"] != 1024:
        raise ValueError("optimization/generation budget differs")
    if model["revision"] != freeze["arguments"]["revision"] or not any(n.endswith(".safetensors") for n in model["sha256"]):
        raise ValueError("model metadata invalid")
    checked_model_files = []
    for name, expected in model["sha256"].items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("bad model hash")
        candidate = Path(model_path) / name if model_path else tokenizer_path / name
        if model_path or (not name.endswith(".safetensors") and candidate.exists()):
            if not candidate.resolve().is_relative_to((Path(model_path) if model_path else tokenizer_path).resolve()) or digest(candidate) != expected:
                raise ValueError(f"model/tokenizer file mismatch {name}")
            checked_model_files.append(name)
    result = {"status": state["status"], "verified_files": len(manifest), "manifest_sha256": digest(root / "MANIFEST.json"),
              "model_weights_verified_locally": bool(model_path), "checked_model_files": checked_model_files,
              "scope": "Hashes, native decoding, prompts, truthful feedback, loss arithmetic and saved records only; no neural-forward, optimizer-update or RNG-generation replay. Apparatus result, not a paper gate."}
    loading = read(root / "loading_info.json")
    bad_loading = any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    if state["status"] == "INVALID_MODEL_LOADING":
        if not bad_loading:
            raise ValueError("unsupported loading failure")
        return result
    if bad_loading:
        raise ValueError("loaded model not qualified")
    from transformers import AutoTokenizer
    if not {"config.json", "tokenizer_config.json", "tokenizer.json"}.issubset(checked_model_files):
        raise ValueError("native tokenizer/config metadata not verified")
    for path in tokenizer_path.glob("*.jinja"):
        if path.name not in checked_model_files:
            raise ValueError("unverified native chat template")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    vocab_size = read(tokenizer_path / "config.json")["vocab_size"]
    if "generation_config_used.json" not in manifest:
        raise ValueError("generation configuration evidence missing")
    configs = read(root / "generation_config_used.json")
    fixed = dict(max_new_tokens=64, temperature=1., top_p=1., top_k=0, typical_p=1., repetition_penalty=1., encoder_repetition_penalty=1.,
                 num_beams=1, min_length=0, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id,
                 pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                 use_cache=True, return_dict_in_generate=True, output_scores=True)
    for sample in (False, True):
        if any(configs[str(sample)].get(k) != v for k, v in dict(fixed, do_sample=sample).items()):
            raise ValueError("native sampling configuration mismatch")
        for field in ("bad_words_ids", "force_words_ids", "suppress_tokens", "begin_suppress_tokens", "forced_bos_token_id", "forced_eos_token_id", "sequence_bias", "constraints"):
            if configs[str(sample)].get(field) is not None:
                raise ValueError("unexpected generation constraint")
    def tokens(messages):
        return list(tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False))
    def generation(response, record, messages, sample, expected_seed):
        validate_native_generation(response, tokenizer, tokens(messages), vocab_size, sample, expected_seed)
        return apparatus["score"](response["text"], record)
    def evaluated(name, records, prompts, offset=0):
        if name not in manifest:
            raise ValueError(f"unmanifested output: {name}")
        rows = jsonl(root / name)
        if len(rows) != len(records) or [r["id"] for r in rows] != [r["id"] for r in records]:
            raise ValueError("evaluation IDs missing/reordered")
        for i, (row, record, messages) in enumerate(zip(rows, records, prompts)):
            if row["user_id"] != record["user_id"] or row["preference"] != record["preference"] or row["score"] != generation(row, record, messages, False, seed + offset + i):
                raise ValueError("evaluation metadata/scoring mismatch")
        return rows
    calibration = data["calibration"]
    original = evaluated("calibration_original.jsonl", calibration, [r["prompt"] for r in calibration])
    explicit = evaluated("calibration_explicit.jsonl", calibration, [r["calibration_prompt"] for r in calibration])
    teacher = evaluated("calibration_teacher.jsonl", calibration, [hindsight(r["prompt"], apparatus["feedback"](o["text"], r)) for r, o in zip(calibration, original)])
    qualified = qualification(original, explicit, teacher)
    if "qualification.json" not in manifest:
        raise ValueError("unmanifested qualification")
    stored = read(root / "qualification.json")
    if any(stored[k] != v for k, v in qualified.items()):
        raise ValueError("qualification mismatch")
    result["qualification"] = qualified
    if state["status"] == "UNQUALIFIED_GENERATIVE_APPARATUS":
        if qualified["qualified"] or any(n in manifest for n in ("training.jsonl", "final_adapter.pt", "baseline_eval.jsonl")):
            raise ValueError("unsupported qualification stop")
        result["scientific_decision"] = None
        return result
    if not qualified["qualified"]:
        raise ValueError("unqualified continuation")
    if state["status"] == "INSUFFICIENT_ADAPTATION_HEADROOM":
        if stats(original)["joint"] <= .85 or "training.jsonl" in manifest or "baseline_eval.jsonl" in manifest:
            raise ValueError("unsupported headroom stop")
        return result
    if stats(original)["joint"] > .85:
        raise ValueError("headroom prerequisite not met")
    before = evaluated("baseline_eval.jsonl", data["eval"], [r["prompt"] for r in data["eval"]], offset=10000)
    if "adapter_insertion.json" not in manifest:
        raise ValueError("unmanifested insertion check")
    insertion = read(root / "adapter_insertion.json")
    if insertion["tolerance"] != 1e-5 or not np.isfinite(insertion["logprob_error"]):
        raise ValueError("invalid insertion check")
    if state["status"] == "INVALID_ADAPTER_INSERTION":
        if insertion["logprob_error"] <= insertion["tolerance"]:
            raise ValueError("unsupported insertion failure")
        return result
    if state["status"] not in ("COMPLETE_POSITIVE_CONTROL", "INVALID_SAMPLING_RECOMPUTE") or insertion["logprob_error"] > insertion["tolerance"]:
        raise ValueError("unexpected terminal state")
    checkpoint_required = ["initial_adapter.pt", "training.jsonl", "adapter_insertion.json"]
    if state["status"] == "COMPLETE_POSITIVE_CONTROL":
        checkpoint_required += ["final_adapter.pt", "final_optimizer.pt", "positive_control.json"]
    for name in checkpoint_required:
        if name not in manifest or ((root / name).stat().st_size == 0 and name != "training.jsonl"):
            raise ValueError("training/checkpoint evidence missing")
    losses = released_loss((root / "upstream_loss_source.py").read_bytes())
    training = jsonl(root / "training.jsonl")
    if state["status"] == "COMPLETE_POSITIVE_CONTROL" and len(training) != 64:
        raise ValueError("incomplete fixed training budget")
    for step, (row, index) in enumerate(zip(training, schedule)):
        record = data["train"][index]
        if row["step"] != step or row["id"] != record["id"] or row["user_id"] != record["user_id"]:
            raise ValueError("training ID/schedule mismatch")
        computed = generation(row["response"], record, record["prompt"], True, seed + 20000 + step)
        feedback = apparatus["feedback"](row["response"]["text"], record)
        if computed != row["score"] or feedback != row["feedback"] or tokens(hindsight(record["prompt"], feedback)) != row["teacher_prompt_ids"]:
            raise ValueError("training feedback/teacher leakage mismatch")
        verify_loss(row, losses)
        gap = max(abs(a - b) for a, b in zip(row["response"]["token_logprobs"], row["training_logprobs"]["base"]))
        near(gap, row["sampling_recompute_max_logprob_gap"], "sampling recompute gap")
        if gap > .25:
            raise ValueError("generation and training logprobs differ")
    if state["status"] == "INVALID_SAMPLING_RECOMPUTE":
        if "sampling_mismatch.json" not in manifest or len(training) >= 64:
            raise ValueError("sampling failure evidence missing")
        failure = read(root / "sampling_mismatch.json")
        step = len(training)
        record = data["train"][schedule[step]]
        if failure["step"] != step or failure["id"] != record["id"]:
            raise ValueError("sampling failure step mismatch")
        generation(failure["response"], record, record["prompt"], True, seed + 20000 + step)
        feedback = apparatus["feedback"](failure["response"]["text"], record)
        if failure["feedback"] != feedback or failure["teacher_prompt_ids"] != tokens(hindsight(record["prompt"], feedback)):
            raise ValueError("sampling failure feedback mismatch")
        a, b = failure["response"]["token_logprobs"], failure["training_logprobs"]["base"]
        gap = max(abs(x - y) for x, y in zip(a, b))
        near(gap, failure["max_abs_logprob_gap"], "failure gap")
        if len(a) == len(b) and gap <= .25:
            raise ValueError("sampling failure unsupported")
        result.update(training_updates=len(training), scientific_decision=None, sampling_gap=gap)
        return result
    after = evaluated("final_eval.jsonl", data["eval"], [r["prompt"] for r in data["eval"]], offset=10000)
    b, a = stats(before), stats(after)
    decision = {"positive_control": a["joint"] - b["joint"] >= .10 and a["content_valid"] - b["content_valid"] >= -.05,
                "before": b, "after": a, "joint_gain": a["joint"] - b["joint"]}
    if any(read(root / "positive_control.json")[k] != v for k, v in decision.items()):
        raise ValueError("positive control decision mismatch")
    result.update(positive_control=decision, paired_evaluation=paired_users(before, after), training_updates=64,
                  checkpoint_verification=checkpoints(root, freeze["optimizer"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root"); parser.add_argument("report")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    result = verify(args.root, args.tokenizer_path, args.model_path)
    with open(args.report, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
