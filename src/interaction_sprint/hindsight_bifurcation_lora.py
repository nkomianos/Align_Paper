"""Finite-step shared-LoRA follow-up to the Hindsight bifurcation screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from .hindsight_bifurcation_g0 import fixed_points
from .parameter_probe import MODEL, REVISION
from latent_contract.sender_update import adapter_state, install_lora, load_adapter


SEED = 90418001
ARMS = ("dynamic_minus", "dynamic_plus", "fixed_minus", "fixed_plus")
STEPS = 64
BATCH = 8


def logit(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def initial_probability(middle: float, sign: int) -> float:
    epsilon = min(0.01, middle / 2.0, (1.0 - middle) / 2.0)
    return middle + sign * epsilon


def binary_reverse_kl(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    p = student.clamp(1e-12, 1.0 - 1e-12)
    q = teacher.clamp(1e-12, 1.0 - 1e-12)
    return p * (p.log() - q.log()) + (1.0 - p) * ((1.0 - p).log() - (1.0 - q).log())


def schedule(context_ids: list[str]) -> list[list[str]]:
    if len(context_ids) != 16 or len(set(context_ids)) != 16:
        raise ValueError("Need the 16 development contexts")
    rng = random.Random(SEED)
    result = []
    while len(result) < STEPS:
        order = context_ids.copy()
        rng.shuffle(order)
        result.extend(order[index : index + BATCH] for index in range(0, len(order), BATCH))
    return result[:STEPS]


def prepare_contexts(evidence: Path) -> list[dict]:
    recorded = json.loads((evidence / "RESULT.json").read_text(encoding="utf-8"))
    if recorded["status"] != "BIFURCATION_SCREEN_POSITIVE":
        raise ValueError("The prerequisite screen is not positive")
    rows = json.loads((evidence / "rows.json").read_text(encoding="utf-8"))
    base = {row["base_id"]: row for row in rows if row["kind"] == "base"}
    hindsight: dict[str, dict[int, dict]] = {}
    for row in rows:
        if row["kind"] == "hindsight":
            hindsight.setdefault(row["base_id"], {})[row["target"]] = row
    contexts = []
    for item in recorded["contexts"]:
        roots = item["roots"]
        middle = roots[1]["probability"] if item["bistable"] else None
        contexts.append(
            dict(
                base_id=item["base_id"],
                split=item["split"],
                domain=item["domain"],
                options=base[item["base_id"]]["options"],
                prompt=base[item["base_id"]]["text"],
                prompt_tokens=base[item["base_id"]]["prompt_tokens"],
                candidate_tokens=base[item["base_id"]]["candidate_tokens"],
                artifact_base_probability_one=base[item["base_id"]]["probabilities"][1],
                teacher_zero_probability_one=hindsight[item["base_id"]][0]["probabilities"][1],
                teacher_one_probability_one=hindsight[item["base_id"]][1]["probabilities"][1],
                bistable=item["bistable"],
                middle=middle,
            )
        )
    if len(contexts) != 32:
        raise ValueError("Incomplete prerequisite contexts")
    return contexts


def sequence_scores(model, tokenizer, contexts: list[dict]) -> torch.Tensor:
    """Return differentiable summed continuation log likelihoods, shape [N,2]."""
    records = []
    for context_index, context in enumerate(contexts):
        for candidate_index, candidate in enumerate(context["candidate_tokens"]):
            inputs = context["prompt_tokens"] + candidate[:-1]
            records.append((context_index, candidate_index, inputs, candidate, len(context["prompt_tokens"]) - 1))
    width = max(len(record[2]) for record in records)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((len(records), width), pad, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for index, record in enumerate(records):
        input_ids[index, : len(record[2])] = torch.tensor(record[2])
        attention[index, : len(record[2])] = 1
    hidden = model.model(input_ids=input_ids, attention_mask=attention, use_cache=False).last_hidden_state
    selected = []
    targets = []
    owners = []
    for record_index, (_, _, _, candidate, first_position) in enumerate(records):
        for offset, token in enumerate(candidate):
            selected.append(hidden[record_index, first_position + offset])
            targets.append(token)
            owners.append(record_index)
    logits = model.get_output_embeddings()(torch.stack(selected)).double()
    target = torch.tensor(targets, dtype=torch.long)
    token_logp = logits[torch.arange(len(target)), target] - logits.logsumexp(-1)
    record_scores = torch.zeros(len(records), dtype=torch.float64)
    record_scores = record_scores.index_add(0, torch.tensor(owners), token_logp)
    return record_scores.reshape(len(contexts), 2)


def run(root: Path, evidence: Path) -> None:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    evidence = evidence.resolve()
    manifest = json.loads((evidence / "MANIFEST.json").read_text(encoding="utf-8"))
    for name, digest in manifest.items():
        path = (evidence / name).resolve()
        if path.parent != evidence or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Prerequisite evidence mismatch")
    contexts = prepare_contexts(evidence)
    development_ids = [context["base_id"] for context in contexts if context["split"] == "development"]
    batches = schedule(development_ids)
    lookup = {context["base_id"]: context for context in contexts}

    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def write(name: str, value: object) -> None:
        with (root / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    write(
        "spec.json",
        dict(
            model=MODEL,
            revision=REVISION,
            seed=SEED,
            arms=ARMS,
            steps=STEPS,
            batch=BATCH,
            rank=8,
            alpha=16,
            learning_rate=0.0003,
            training_split="development only",
            evaluation_split="all contexts; confirmation is primary",
            loss="candidate-normalized binary reverse KL; report mixture weights detached",
            initialization="context offset places policy 0.01 below/above measured unstable point",
            primary_gate="confirmation bistable median dynamic separation >=.25, fixed <=.10, dynamic sign consistency >=.75",
            prerequisite_manifest_sha256=hashlib.sha256((evidence / "MANIFEST.json").read_bytes()).hexdigest(),
            automatic_expansion=False,
            paper_green_light=False,
        ),
    )
    write("contexts.json", contexts)
    write("schedule.json", batches)
    (root / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    install_lora(model, rank=8, alpha=16)
    initial = adapter_state(model)
    torch.save(initial, root / "initial_adapter.pt")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]

    with torch.no_grad():
        initial_scores = sequence_scores(model, tokenizer, contexts)
        initial_probabilities = (initial_scores[:, 1] - initial_scores[:, 0]).sigmoid()
    maximum_replay_error = max(
        abs(float(initial_probabilities[index]) - context["artifact_base_probability_one"])
        for index, context in enumerate(contexts)
    )
    if maximum_replay_error > 5e-5:
        raise ValueError("Zero-adapter replay differs from prerequisite probabilities")

    offsets = {}
    fixed_marginals = {}
    for sign_name, sign in (("minus", -1), ("plus", 1)):
        offsets[sign_name] = {}
        fixed_marginals[sign_name] = {}
        for index, context in enumerate(contexts):
            if not context["bistable"]:
                continue
            desired = initial_probability(context["middle"], sign)
            offsets[sign_name][context["base_id"]] = logit(desired) - logit(float(initial_probabilities[index]))
            fixed_marginals[sign_name][context["base_id"]] = desired
    write("offsets.json", offsets)
    write("fixed_marginals.json", fixed_marginals)
    write(
        "runtime.json",
        dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            maximum_zero_adapter_probability_replay_error=maximum_replay_error,
        ),
    )

    def evaluate(arm: str) -> list[dict]:
        sign_name = "minus" if arm.endswith("minus") else "plus"
        with torch.no_grad():
            scores = sequence_scores(model, tokenizer, contexts)
            raw_logits = scores[:, 1] - scores[:, 0]
        output = []
        for index, context in enumerate(contexts):
            if context["bistable"]:
                probability = float((raw_logits[index] + offsets[sign_name][context["base_id"]]).sigmoid())
            else:
                probability = float(raw_logits[index].sigmoid())
            output.append(dict(base_id=context["base_id"], split=context["split"], domain=context["domain"], bistable=context["bistable"], probability_one=probability))
        return output

    completed = []
    counts = dict(forward_batches=1, backwards=0, updates=0)
    optimizer = None
    try:
        for arm in ARMS:
            load_adapter(model, initial)
            optimizer = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.0)
            sign_name = "minus" if arm.endswith("minus") else "plus"
            arm_rows = [context for context in contexts if context["split"] == "development" and context["bistable"]]
            checkpoints = {0: evaluate(arm)}
            step_rows = []
            for step, batch_ids in enumerate(batches, 1):
                batch = [lookup[base_id] for base_id in batch_ids if lookup[base_id]["bistable"]]
                if not batch:
                    continue
                optimizer.zero_grad(set_to_none=True)
                scores = sequence_scores(model, tokenizer, batch)
                counts["forward_batches"] += 1
                raw_logits = scores[:, 1] - scores[:, 0]
                bias = torch.tensor([offsets[sign_name][context["base_id"]] for context in batch], dtype=torch.float64)
                student = (raw_logits + bias).sigmoid()
                if arm.startswith("dynamic"):
                    report_one = student.detach()
                else:
                    report_one = torch.tensor([fixed_marginals[sign_name][context["base_id"]] for context in batch], dtype=torch.float64)
                teacher_zero = torch.tensor([context["teacher_zero_probability_one"] for context in batch], dtype=torch.float64)
                teacher_one = torch.tensor([context["teacher_one_probability_one"] for context in batch], dtype=torch.float64)
                loss = ((1.0 - report_one) * binary_reverse_kl(student, teacher_zero) + report_one * binary_reverse_kl(student, teacher_one)).mean()
                loss.backward()
                counts["backwards"] += 1
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                counts["updates"] += 1
                step_rows.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), batch=[context["base_id"] for context in batch], mean_probability=float(student.detach().mean())))
                if step in (32, 64):
                    checkpoints[step] = evaluate(arm)
                    print(json.dumps(dict(arm=arm, step=step, elapsed=time.monotonic() - start)), flush=True)
            torch.save(adapter_state(model), root / f"{arm}_adapter.pt")
            torch.save(optimizer.state_dict(), root / f"{arm}_optimizer.pt")
            write(f"{arm}_steps.json", step_rows)
            write(f"{arm}_checkpoints.json", checkpoints)
            completed.append(arm)

        finals = {arm: json.loads((root / f"{arm}_checkpoints.json").read_text())[str(STEPS)] for arm in ARMS}
        rows_by_arm = {arm: {row["base_id"]: row for row in values} for arm, values in finals.items()}
        primary_ids = [context["base_id"] for context in contexts if context["split"] == "confirmation" and context["bistable"]]
        dynamic_differences = [rows_by_arm["dynamic_plus"][base_id]["probability_one"] - rows_by_arm["dynamic_minus"][base_id]["probability_one"] for base_id in primary_ids]
        fixed_differences = [rows_by_arm["fixed_plus"][base_id]["probability_one"] - rows_by_arm["fixed_minus"][base_id]["probability_one"] for base_id in primary_ids]
        dynamic_sorted = sorted(dynamic_differences)
        fixed_sorted = sorted(fixed_differences)
        metrics = dict(
            confirmation_bistable=len(primary_ids),
            dynamic_median_separation=dynamic_sorted[len(dynamic_sorted) // 2],
            fixed_median_separation=fixed_sorted[len(fixed_sorted) // 2],
            fixed_maximum_absolute_separation=max(abs(value) for value in fixed_differences),
            dynamic_positive_fraction=sum(value > 0 for value in dynamic_differences) / len(dynamic_differences),
            dynamic_differences=dynamic_differences,
            fixed_differences=fixed_differences,
        )
        passed = metrics["dynamic_median_separation"] >= 0.25 and metrics["fixed_maximum_absolute_separation"] <= 0.10 and metrics["dynamic_positive_fraction"] >= 0.75
        result = dict(status="LORA_BIFURCATION_SCREEN_POSITIVE" if passed else "LORA_BIFURCATION_SCREEN_NEGATIVE", metrics=metrics, counts=counts, elapsed=time.monotonic() - start, paper_green_light=False)
        write("RESULT.json", result)
        write("MANIFEST.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.is_file()})
        print(json.dumps(result), flush=True)
    except Exception as error:
        write("FAILED.json", dict(error=type(error).__name__, message=str(error), completed=completed, counts=counts))
        torch.save(adapter_state(model), root / "failure_adapter.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), root / "failure_optimizer.pt")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.evidence)
