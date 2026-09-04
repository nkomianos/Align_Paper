"""Frozen data and loss primitives for a neural delayed-anchor SDPO gate."""
from __future__ import annotations

from collections import Counter
import hashlib
import random
from typing import Iterable

import torch
from torch import nn

from latent_contract.sender_update import LoRALinear


MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OFFICIAL_SDPO_REPOSITORY = "https://github.com/lasgroup/user_interactions"
OFFICIAL_SDPO_COMMIT = "3b17d2a67bd2565b9fbda495fd16a485406aa954"
SEED = 2026090407
TRAIN_STEPS = 64
LEARNING_RATE = 1e-4
LORA_RANK = 8
LORA_ALPHA = 16
TRAIN_BATCH = 16
ANCHORS_PER_BATCH = 4
HINDSIGHT_BLOCK = (
    "\n\n=== HINDSIGHT CONTEXT ===\n"
    "[The following is a future user message. Use this to guide your answer to the user prompt.]\n"
    "{follow_up}"
)


SURFACES = (
    ("travel plan", "a brief direct itinerary", "a detailed step-by-step itinerary"),
    ("code explanation", "a concise explanation", "a detailed step-by-step explanation"),
    ("product comparison", "a short comparison", "a comprehensive comparison"),
    ("study plan", "a compact checklist", "a detailed staged plan"),
    ("professional email", "a brief email", "a detailed explanatory email"),
    ("recipe", "a minimal recipe", "a detailed recipe with explanations"),
    ("financial summary", "a concise summary", "a detailed breakdown"),
    ("health information", "a brief plain-language answer", "a detailed structured answer"),
    ("book recommendation", "a short recommendation", "a detailed recommendation with rationale"),
    ("debugging response", "a minimal fix", "a detailed diagnosis and fix"),
    ("meeting notes", "a compact action list", "a detailed annotated summary"),
    ("historical explanation", "a brief overview", "a detailed chronological explanation"),
    ("home project", "a concise instruction list", "a detailed step-by-step guide"),
    ("data analysis", "a short result summary", "a detailed methodological explanation"),
    ("language lesson", "a compact example", "a detailed lesson with examples"),
    ("research overview", "a concise abstract-style answer", "a detailed structured review"),
)


def semantic_letter_index(semantic: int, swap: int) -> int:
    if semantic not in (0, 1) or swap not in (0, 1):
        raise ValueError("semantic and swap must be binary")
    return semantic ^ swap


def _prompt(domain: str, options: tuple[str, str], swap: int, *, evaluation: bool) -> str:
    ordered = options if not swap else (options[1], options[0])
    prefix = (
        "This is an independent evaluation case. " if evaluation else
        "This is one interaction from a population study. "
    )
    return (
        f"{prefix}Choose the response format that would best suit an anonymous client's {domain}. "
        "The client's preference has not been disclosed. "
        f"A) {ordered[0]}\nB) {ordered[1]}\n"
        "Reply with exactly A or B."
    )


def _feedback(options: tuple[str, str], semantic: int) -> str:
    return f"My preference is {options[semantic]}."


def feedback_for_row(row: dict[str, object], semantic: int) -> str:
    if semantic not in (0, 1):
        raise ValueError("semantic must be binary")
    options = tuple(str(value) for value in row["options"])
    if len(options) != 2:
        raise ValueError("row must contain two options")
    return _feedback(options, semantic)


def build_records() -> tuple[list[dict[str, object]], list[dict[str, object]], list[list[str]]]:
    """Return main training interactions, evaluation prompts, and fixed batches.

    The main population has P(Z0=1)=.75.  Logged actions are balanced within
    each initial state.  c0=1 and c1=0, so the expression and transition worlds
    share immediate reports but prefer opposite population actions.
    """
    strata = [(1, 0)] * 48 + [(1, 1)] * 48 + [(0, 0)] * 16 + [(0, 1)] * 16
    rng = random.Random(SEED)
    rng.shuffle(strata)
    rows: list[dict[str, object]] = []
    for index, (z0, logged_action) in enumerate(strata):
        surface_index = index % len(SURFACES)
        repetition = index // len(SURFACES)
        domain, concise, detailed = SURFACES[surface_index]
        options = (concise, detailed)
        swap = repetition % 2
        immediate = 0 if logged_action == 0 else z0
        row_id = f"train-{index:03d}"
        rows.append({
            "id": row_id,
            "domain": domain,
            "options": list(options),
            "swap": swap,
            "z0": z0,
            "logged_action": logged_action,
            "logged_letter": "AB"[semantic_letter_index(logged_action, swap)],
            "immediate_semantic": immediate,
            "delayed_expression_semantic": z0,
            "delayed_transition_semantic": immediate,
            "truthful_semantic": z0,
            "prompt": _prompt(domain, options, swap, evaluation=False),
            "immediate_feedback": _feedback(options, immediate),
            "delayed_expression_feedback": _feedback(options, z0),
            "delayed_transition_feedback": _feedback(options, immediate),
            "truthful_feedback": _feedback(options, z0),
            "anchor": False,
        })

    # Outcome-independent within-stratum hash ranks preserve the population
    # proportions exactly: 12/48,12/48,4/16,4/16.
    targets = {(1, 0): 12, (1, 1): 12, (0, 0): 4, (0, 1): 4}
    for stratum, count in targets.items():
        candidates = [row for row in rows if (row["z0"], row["logged_action"]) == stratum]
        candidates.sort(key=lambda row: hashlib.sha256((str(row["id"]) + "|anchor").encode()).hexdigest())
        for row in candidates[:count]:
            row["anchor"] = True

    evaluation: list[dict[str, object]] = []
    for surface_index, (domain, concise, detailed) in enumerate(SURFACES):
        options = (concise, detailed)
        for swap in (0, 1):
            evaluation.append({
                "id": f"eval-{surface_index:02d}-{swap}",
                "domain": domain,
                "options": list(options),
                "swap": swap,
                "prompt": _prompt(domain, options, swap, evaluation=True),
            })

    anchors = [str(row["id"]) for row in rows if row["anchor"]]
    ordinary = [str(row["id"]) for row in rows if not row["anchor"]]
    batches: list[list[str]] = []
    for _ in range(TRAIN_STEPS // 8):
        rng.shuffle(anchors)
        rng.shuffle(ordinary)
        for offset in range(0, len(anchors), ANCHORS_PER_BATCH):
            batch = anchors[offset:offset + ANCHORS_PER_BATCH]
            ordinary_offset = (offset // ANCHORS_PER_BATCH) * (TRAIN_BATCH - ANCHORS_PER_BATCH)
            batch += ordinary[ordinary_offset:ordinary_offset + TRAIN_BATCH - ANCHORS_PER_BATCH]
            rng.shuffle(batch)
            batches.append(batch)
    if len(batches) != TRAIN_STEPS or any(len(batch) != TRAIN_BATCH for batch in batches):
        raise AssertionError("invalid fixed schedule")
    return rows, evaluation, batches


def record_invariants(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(rows)
    anchors = [row for row in rows if row["anchor"]]
    return {
        "n": len(rows),
        "z0": dict(Counter(int(row["z0"]) for row in rows)),
        "logged_action": dict(Counter(int(row["logged_action"]) for row in rows)),
        "strata": {f"z{z}-a{a}": sum(row["z0"] == z and row["logged_action"] == a for row in rows)
                    for z in (0, 1) for a in (0, 1)},
        "anchors": len(anchors),
        "anchor_strata": {f"z{z}-a{a}": sum(row["z0"] == z and row["logged_action"] == a for row in anchors)
                           for z in (0, 1) for a in (0, 1)},
        "immediate_world_mismatches": sum(
            row["immediate_semantic"] != row["delayed_transition_semantic"] for row in rows
        ),
    }


def reverse_kl_per_example(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 2:
        raise ValueError("student and teacher logits must have matching [batch,vocab] shape")
    student_log = student_logits.float().log_softmax(-1)
    teacher_log = teacher_logits.float().log_softmax(-1).detach()
    return (student_log.exp() * (student_log - teacher_log)).sum(-1)


def augmented_reverse_kl(
    student_logits: torch.Tensor,
    immediate_teacher_logits: torch.Tensor,
    delayed_teacher_logits: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    if anchor_mask.dtype != torch.bool or anchor_mask.ndim != 1 or anchor_mask.shape[0] != student_logits.shape[0]:
        raise ValueError("anchor_mask must be a batch-length boolean vector")
    if not bool(anchor_mask.any()):
        raise ValueError("augmented batches require an anchor")
    immediate = reverse_kl_per_example(student_logits, immediate_teacher_logits)
    delayed = reverse_kl_per_example(student_logits[anchor_mask], delayed_teacher_logits)
    return immediate.mean() + (delayed - immediate[anchor_mask]).mean()


def install_qwen35_lora(model: nn.Module, rank: int = LORA_RANK, alpha: int = LORA_ALPHA) -> list[str]:
    """Install adapters in both full-attention and DeltaNet projections."""
    leaves = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
    }
    model.requires_grad_(False)
    replacements = [
        (name, module) for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in leaves and isinstance(module, nn.Linear)
    ]
    if not replacements:
        raise ValueError("no Qwen3.5 attention/DeltaNet projections found")
    for name, module in replacements:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, rank=rank, alpha=alpha))
    return [name for name, _ in replacements]


def hindsight_user_text(prompt: str, follow_up: str) -> str:
    return prompt + HINDSIGHT_BLOCK.format(follow_up=follow_up.strip())
