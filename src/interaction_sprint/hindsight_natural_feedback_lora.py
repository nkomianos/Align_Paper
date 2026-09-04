"""Natural-initialization dynamic versus fixed Hindsight LoRA diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .hindsight_bifurcation_lora import BATCH, SEED, STEPS, binary_reverse_kl, prepare_contexts, schedule, sequence_scores
from .parameter_probe import MODEL, REVISION
from latent_contract.sender_update import adapter_state, install_lora, load_adapter


ARMS = ("dynamic_natural", "fixed_natural")


def basin_side(initial: float, middle: float) -> int:
    if initial == middle:
        raise ValueError("Natural policy lies exactly on the unstable point")
    return 1 if initial > middle else -1


def natural_metrics(contexts: list[dict], initial_rows: list[dict], dynamic_rows: list[dict], fixed_rows: list[dict]) -> dict:
    initial = {row["base_id"]: row["probability_one"] for row in initial_rows}
    dynamic = {row["base_id"]: row["probability_one"] for row in dynamic_rows}
    fixed = {row["base_id"]: row["probability_one"] for row in fixed_rows}
    selected = [context for context in contexts if context["split"] == "confirmation" and context["bistable"]]
    records = []
    for context in selected:
        key = context["base_id"]
        side = basin_side(initial[key], context["middle"])
        records.append(
            dict(
                base_id=key,
                side=side,
                initial=initial[key],
                middle=context["middle"],
                dynamic=dynamic[key],
                fixed=fixed[key],
                signed_dynamic_advantage=side * (dynamic[key] - fixed[key]),
                dynamic_basin_consistent=side * (dynamic[key] - context["middle"]) > 0,
            )
        )
    advantages = sorted(row["signed_dynamic_advantage"] for row in records)
    return dict(
        n=len(records),
        upper_median_signed_dynamic_advantage=advantages[len(advantages) // 2],
        positive_advantage_fraction=sum(row["signed_dynamic_advantage"] > 0 for row in records) / len(records),
        dynamic_basin_consistency=sum(row["dynamic_basin_consistent"] for row in records) / len(records),
        records=records,
    )


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
            initialization="Unmodified natural zero-adapter policy; no context offsets",
            training="All 16 development contexts",
            primary="Bistable confirmation contexts",
            gate="upper-median signed dynamic-minus-fixed basin movement >=.25; positive and basin-consistent fractions >=.75",
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
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    install_lora(model, rank=8, alpha=16)
    initial_adapter = adapter_state(model)
    torch.save(initial_adapter, root / "initial_adapter.pt")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def evaluate() -> list[dict]:
        with torch.no_grad():
            scores = sequence_scores(model, tokenizer, contexts)
            probabilities = (scores[:, 1] - scores[:, 0]).sigmoid()
        return [dict(base_id=context["base_id"], split=context["split"], domain=context["domain"], bistable=context["bistable"], probability_one=float(probabilities[index])) for index, context in enumerate(contexts)]

    initial_rows = evaluate()
    maximum_replay_error = max(abs(row["probability_one"] - lookup[row["base_id"]]["artifact_base_probability_one"]) for row in initial_rows)
    if maximum_replay_error > 5e-5:
        raise ValueError("Zero-adapter replay differs from prerequisite probabilities")
    fixed_marginals = {row["base_id"]: row["probability_one"] for row in initial_rows}
    write("initial_probabilities.json", initial_rows)
    write("fixed_marginals.json", fixed_marginals)
    write("runtime.json", dict(torch=torch.__version__, transformers=transformers.__version__, maximum_zero_adapter_probability_replay_error=maximum_replay_error))

    completed = []
    counts = dict(forward_batches=1, backwards=0, updates=0)
    optimizer = None
    try:
        for arm in ARMS:
            load_adapter(model, initial_adapter)
            optimizer = torch.optim.AdamW(params, lr=0.0003, weight_decay=0.0)
            checkpoints = {0: evaluate()}
            steps = []
            for step, batch_ids in enumerate(batches, 1):
                batch = [lookup[key] for key in batch_ids]
                optimizer.zero_grad(set_to_none=True)
                scores = sequence_scores(model, tokenizer, batch)
                counts["forward_batches"] += 1
                student = (scores[:, 1] - scores[:, 0]).sigmoid()
                report_one = student.detach() if arm == "dynamic_natural" else torch.tensor([fixed_marginals[context["base_id"]] for context in batch], dtype=torch.float64)
                teacher_zero = torch.tensor([context["teacher_zero_probability_one"] for context in batch], dtype=torch.float64)
                teacher_one = torch.tensor([context["teacher_one_probability_one"] for context in batch], dtype=torch.float64)
                loss = ((1.0 - report_one) * binary_reverse_kl(student, teacher_zero) + report_one * binary_reverse_kl(student, teacher_one)).mean()
                loss.backward()
                counts["backwards"] += 1
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                optimizer.step()
                counts["updates"] += 1
                steps.append(dict(step=step, loss=float(loss.detach()), gradient_norm=float(norm), batch=batch_ids, mean_probability=float(student.detach().mean())))
                if step in (32, 64):
                    checkpoints[step] = evaluate()
                    print(json.dumps(dict(arm=arm, step=step, elapsed=time.monotonic() - start)), flush=True)
            torch.save(adapter_state(model), root / f"{arm}_adapter.pt")
            torch.save(optimizer.state_dict(), root / f"{arm}_optimizer.pt")
            write(f"{arm}_steps.json", steps)
            write(f"{arm}_checkpoints.json", checkpoints)
            completed.append(arm)

        dynamic = json.loads((root / "dynamic_natural_checkpoints.json").read_text())[str(STEPS)]
        fixed = json.loads((root / "fixed_natural_checkpoints.json").read_text())[str(STEPS)]
        metrics = natural_metrics(contexts, initial_rows, dynamic, fixed)
        passed = metrics["upper_median_signed_dynamic_advantage"] >= 0.25 and metrics["positive_advantage_fraction"] >= 0.75 and metrics["dynamic_basin_consistency"] >= 0.75
        result = dict(status="NATURAL_LORA_FEEDBACK_SCREEN_POSITIVE" if passed else "NATURAL_LORA_FEEDBACK_SCREEN_NEGATIVE", metrics=metrics, counts=counts, elapsed=time.monotonic() - start, paper_green_light=False)
        write("RESULT.json", result)
        write("MANIFEST.json", {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.is_file()})
        print(json.dumps({key: value for key, value in result.items() if key != "metrics"} | {"metrics_summary": {key: value for key, value in metrics.items() if key != "records"}}), flush=True)
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
