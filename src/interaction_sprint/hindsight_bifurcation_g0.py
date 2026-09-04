"""Prospective screen for self-confirming full-KL Hindsight equilibria.

This is a constrained two-action teacher assay plus exact finite-state dynamics.
It is not evidence about human preference change or free-form generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from .hindsight_revelation_probe import published_template
from .parameter_probe import MODEL, REVISION


SEED = 90417001
GRID_POINTS = 20_001


def dataset() -> list[dict]:
    groups = {
        "development": [
            ("afternoon beverage", "tea", "coffee"),
            ("casual footwear", "sneakers", "sandals"),
            ("music genre", "jazz", "rock"),
            ("household pet", "cat", "dog"),
        ],
        "confirmation": [
            ("screen brightness", "dim", "bright"),
            ("soup temperature", "hot", "cold"),
            ("public transport", "bus", "train"),
            ("notebook cover", "red", "blue"),
        ],
    }
    rows = []
    for split, domains in groups.items():
        for index, (domain, a, b) in enumerate(domains):
            for order in (0, 1):
                options = [a, b] if order == 0 else [b, a]
                for wording in (0, 1):
                    prompt = (
                        f"Choose my {domain}: {options[0]} or {options[1]}. "
                        "Respond with the chosen option."
                        if wording == 0
                        else f"For my {domain}, the options are {options[0]} and "
                        f"{options[1]}. Which option should I use?"
                    )
                    base_id = f"{split}-{index}-{order}-{wording}"
                    for preferred in (0, 1):
                        rows.append(
                            dict(
                                id=f"{base_id}-{preferred}",
                                base_id=base_id,
                                split=split,
                                domain=domain,
                                order=order,
                                wording=wording,
                                options=options,
                                preferred=(a, b)[preferred],
                                target=preferred ^ order,
                                prompt=prompt,
                            )
                        )
    return rows


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -700.0))
    return z / (1.0 + z)


def _logit(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def distribution(scores: list[float]) -> list[float]:
    if len(scores) != 2 or not all(math.isfinite(value) for value in scores):
        raise ValueError("Need two finite sequence log probabilities")
    maximum = max(scores)
    weights = [math.exp(value - maximum) for value in scores]
    total = sum(weights)
    return [value / total for value in weights]


def reverse_kl_map(p: float, report_zero_logit: float, report_one_logit: float) -> float:
    """Exact reverse-KL barycenter under a copying report with P(report=1)=p."""
    return _sigmoid((1.0 - p) * report_zero_logit + p * report_one_logit)


def fixed_points(report_zero_logit: float, report_one_logit: float) -> list[dict]:
    def residual(p: float) -> float:
        return reverse_kl_map(p, report_zero_logit, report_one_logit) - p

    roots: list[float] = []
    left = 0.0
    left_value = residual(left)
    for index in range(1, GRID_POINTS):
        right = index / (GRID_POINTS - 1)
        right_value = residual(right)
        if right_value == 0.0 or left_value * right_value < 0.0:
            lo, hi = left, right
            for _ in range(64):
                mid = (lo + hi) / 2.0
                if residual(lo) * residual(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            root = (lo + hi) / 2.0
            if not roots or abs(root - roots[-1]) > 1e-7:
                roots.append(root)
        left, left_value = right, right_value
    gap = report_one_logit - report_zero_logit
    return [
        dict(
            probability=root,
            derivative=root * (1.0 - root) * gap,
            stable=abs(root * (1.0 - root) * gap) < 1.0,
        )
        for root in roots
    ]


def endpoint_pair(report_zero_logit: float, report_one_logit: float, middle: float) -> dict:
    epsilon = min(0.01, middle / 2.0, (1.0 - middle) / 2.0)
    initial = [middle - epsilon, middle + epsilon]
    dynamic = []
    fixed = []
    for p0 in initial:
        p = p0
        for _ in range(128):
            p = reverse_kl_map(p, report_zero_logit, report_one_logit)
        dynamic.append(p)
        fixed.append(reverse_kl_map(p0, report_zero_logit, report_one_logit))
    return dict(
        epsilon=epsilon,
        initial=initial,
        dynamic_terminal=dynamic,
        fixed_marginal_terminal=fixed,
        dynamic_separation=dynamic[1] - dynamic[0],
        fixed_marginal_separation=fixed[1] - fixed[0],
    )


def analyze(rows: list[dict]) -> dict:
    grouped: dict[str, dict[int, dict]] = {}
    for row in rows:
        if row["kind"] == "hindsight":
            grouped.setdefault(row["base_id"], {})[row["target"]] = row
    if len(grouped) != 32 or any(set(pair) != {0, 1} for pair in grouped.values()):
        raise ValueError("Incomplete crossed teacher pairs")

    contexts = []
    for base_id, pair in sorted(grouped.items()):
        report_zero_logit = _logit(pair[0]["probabilities"][1])
        report_one_logit = _logit(pair[1]["probabilities"][1])
        roots = fixed_points(report_zero_logit, report_one_logit)
        bistable = (
            len(roots) == 3
            and roots[0]["stable"]
            and not roots[1]["stable"]
            and roots[2]["stable"]
        )
        endpoints = endpoint_pair(report_zero_logit, report_one_logit, roots[1]["probability"]) if bistable else None
        contexts.append(
            dict(
                base_id=base_id,
                split=pair[0]["split"],
                domain=pair[0]["domain"],
                order=pair[0]["order"],
                wording=pair[0]["wording"],
                report_zero_probability_one=pair[0]["probabilities"][1],
                report_one_probability_one=pair[1]["probabilities"][1],
                report_zero_logit=report_zero_logit,
                report_one_logit=report_one_logit,
                roots=roots,
                bistable=bistable,
                endpoints=endpoints,
            )
        )

    teacher = {}
    split_summary = {}
    for split in ("development", "confirmation"):
        teacher_rows = [row for row in rows if row["kind"] == "hindsight" and row["split"] == split]
        teacher[split] = dict(
            n=len(teacher_rows),
            correct=sum(row["prediction"] == row["target"] for row in teacher_rows),
            mean_target_probability=sum(row["probabilities"][row["target"]] for row in teacher_rows) / len(teacher_rows),
        )
        selected = [context for context in contexts if context["split"] == split]
        bistable_selected = [context for context in selected if context["bistable"]]
        split_summary[split] = dict(
            n=len(selected),
            bistable=len(bistable_selected),
            by_domain={
                domain: sum(context["bistable"] for context in selected if context["domain"] == domain)
                for domain in sorted({context["domain"] for context in selected})
            },
            median_dynamic_separation=(
                sorted(context["endpoints"]["dynamic_separation"] for context in bistable_selected)[len(bistable_selected) // 2]
                if bistable_selected
                else 0.0
            ),
            maximum_fixed_marginal_separation=max(
                (context["endpoints"]["fixed_marginal_separation"] for context in bistable_selected), default=0.0
            ),
        )

    teacher_qualified = all(value["correct"] / value["n"] >= 0.9 for value in teacher.values())
    confirmation = split_summary["confirmation"]
    mechanism_supported = (
        teacher_qualified
        and confirmation["bistable"] >= 8
        and all(value >= 1 for value in confirmation["by_domain"].values())
        and confirmation["median_dynamic_separation"] >= 0.5
        and confirmation["maximum_fixed_marginal_separation"] <= 0.1
    )
    return dict(
        status="BIFURCATION_SCREEN_POSITIVE" if mechanism_supported else "BIFURCATION_SCREEN_NEGATIVE",
        teacher=teacher,
        splits=split_summary,
        contexts=contexts,
        criteria=dict(
            teacher_each_split_accuracy_at_least=0.9,
            confirmation_bistable_contexts_at_least=8,
            confirmation_each_domain_bistable_at_least=1,
            confirmation_median_dynamic_separation_at_least=0.5,
            confirmation_maximum_fixed_separation_at_most=0.1,
        ),
        parameter_updates=0,
        human_preference_transition_tested=False,
        paper_green_light=False,
    )


def run(root: Path, config: Path) -> None:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    template = published_template(config)
    root.mkdir(parents=True, exist_ok=False)

    def write(name: str, value: object) -> None:
        with (root / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)

    cases = dataset()
    write("cases.json", cases)
    write(
        "spec.json",
        dict(
            model=MODEL,
            revision=REVISION,
            seed=SEED,
            scope="Fresh constrained-choice teacher plus exact full-reverse-KL copying-feedback dynamics",
            estimand="Existence of initialization-selected policy fixed points, not user welfare or preference change",
            dynamics="Exact reverse-KL barycenter; dynamic report marginal m_t=p_t versus m frozen at p_0",
            iterations=128,
            template=template,
            template_source_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
            automatic_expansion=False,
            parameter_updates=0,
        ),
    )
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
    write("runtime.json", dict(torch=torch.__version__, transformers=transformers.__version__))

    rows = []
    seen = set()
    forwards = 0
    start = time.monotonic()
    try:
        for case in cases:
            candidates = [tokenizer.encode(option, add_special_tokens=False) for option in case["options"]]
            if any(not value for value in candidates) or any(
                candidates[index] == candidates[1 - index][: len(candidates[index])] for index in (0, 1)
            ):
                raise ValueError("Candidate token strings must be nonempty and prefix-free")
            kinds = ["hindsight"]
            if case["base_id"] not in seen:
                kinds = ["base", "hindsight"]
                seen.add(case["base_id"])
            for kind in kinds:
                feedback = f'My actual preference is {case["preferred"]}. Please use that preference.'
                text = case["prompt"] if kind == "base" else case["prompt"] + template.format(follow_up=feedback)
                prompt = tokenizer.apply_chat_template(
                    [dict(role="user", content=text)],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                scores = []
                token_logprobabilities = []
                for candidate in candidates:
                    inputs = prompt + candidate[:-1]
                    if not prompt or len(inputs) > 512:
                        raise ValueError("No truncation")
                    with torch.no_grad():
                        hidden = model.model(input_ids=torch.tensor([inputs]), use_cache=False).last_hidden_state[
                            0, len(prompt) - 1 :
                        ]
                        logits = model.get_output_embeddings()(hidden).double()
                        logp = logits.log_softmax(-1)[torch.arange(len(candidate)), torch.tensor(candidate)]
                    forwards += 1
                    scores.append(float(logp.sum()))
                    token_logprobabilities.append(logp.tolist())
                probabilities = distribution(scores)
                rows.append(
                    dict(
                        id=case["id"],
                        base_id=case["base_id"],
                        domain=case["domain"],
                        split=case["split"],
                        order=case["order"],
                        wording=case["wording"],
                        kind=kind,
                        target=None if kind == "base" else case["target"],
                        options=case["options"],
                        text=text,
                        prompt_tokens=prompt,
                        candidate_tokens=candidates,
                        token_logprobabilities=token_logprobabilities,
                        logprobabilities=scores,
                        probabilities=probabilities,
                        prediction=0 if probabilities[0] >= probabilities[1] else 1,
                    )
                )
            if len(rows) % 24 == 0:
                print(json.dumps(dict(scored_prompts=len(rows), sequence_forwards=forwards, elapsed=time.monotonic() - start)), flush=True)
        result = analyze(rows)
        result.update(scored_prompts=len(rows), sequence_forwards=forwards, elapsed=time.monotonic() - start)
        write("rows.json", rows)
        write("RESULT.json", result)
        write(
            "MANIFEST.json",
            {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir() if path.is_file()},
        )
        print(json.dumps({key: value for key, value in result.items() if key != "contexts"}), flush=True)
    except Exception as error:
        write("partial_rows.json", rows)
        write("FAILED.json", dict(error=type(error).__name__, message=str(error), sequence_forwards=forwards))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--upstream-config", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.upstream_config)
