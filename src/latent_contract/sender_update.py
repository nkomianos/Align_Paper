"""Small, explicit LoRA training primitives; no GPU/network access at import."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen linear plus alpha/r * B A; float32 adapters, zero initial B."""
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a linear and positive rank")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        delta = F.linear(F.linear(x.float(), self.a), self.b) * self.scale
        return original + delta.to(original.dtype)

    def merged_weight(self):
        return (self.base.weight.float() + self.scale * (self.b @ self.a)).to(self.base.weight.dtype)


def install_lora(model, rank=16, alpha=32):
    model.requires_grad_(False)
    replacements = [(n, m) for n, m in model.named_modules()
                    if n.rsplit(".", 1)[-1] in {"q_proj", "k_proj", "v_proj", "o_proj"} and isinstance(m, nn.Linear)]
    if not replacements:
        raise ValueError("no attention projections")
    for name, module in replacements:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(module, rank, alpha))
    return [n for n, _ in replacements]


def adapter_state(model):
    return {name + suffix: getattr(module, field).detach().cpu().clone()
            for name, module in model.named_modules() if isinstance(module, LoRALinear)
            for suffix, field in ((".a", "a"), (".b", "b"))}


def load_adapter(model, state):
    expected = adapter_state(model)
    if state.keys() != expected.keys():
        raise ValueError("adapter keys differ")
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or not torch.isfinite(tensor).all():
            raise ValueError("invalid adapter tensor")
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.a.copy_(state[name + ".a"])
                module.b.copy_(state[name + ".b"])


def merge_lora(model):
    """In-place export into ordinary linear layers, preserving the base model API."""
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                module.base.weight.copy_(module.merged_weight())
            parent_name, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, leaf, module.base)
    return model


def supervised_example(prompt_ids, answer_ids, max_length=2048):
    if not prompt_ids or not answer_ids or len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError("empty or overlength supervised example; never truncate")
    return {"input_ids": list(prompt_ids) + list(answer_ids),
            "labels": [-100] * len(prompt_ids) + list(answer_ids)}


def collate(examples, pad_id, device):
    width = max(len(x["input_ids"]) for x in examples)
    result = {k: [] for k in ("input_ids", "attention_mask", "labels")}
    for row in examples:
        n = len(row["input_ids"])
        result["input_ids"].append(row["input_ids"] + [pad_id] * (width-n))
        result["labels"].append(row["labels"] + [-100] * (width-n))
        result["attention_mask"].append([1] * n + [0] * (width-n))
    return {k: torch.tensor(v, device=device, dtype=torch.long) for k, v in result.items()}


def completion_log_probs(logits, labels):
    """Sum teacher-forced completion log probabilities, excluding prompt/padding."""
    shifted = labels[:, 1:]
    mask = shifted != -100
    tokens = shifted.masked_fill(~mask, 0)
    logp = logits[:, :-1].float().log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    return (logp * mask).sum(-1)


def train_epoch(model, examples, optimizer, seed, micro_batch=2, accumulation=4, pad_id=0, callback=None):
    """One shuffled epoch; equal-example mean losses, including a short last group."""
    if micro_batch < 1 or accumulation < 1 or not examples:
        raise ValueError("invalid batching")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(examples), generator=generator).tolist()
    effective = micro_batch * accumulation
    model.train()
    device = next(model.parameters()).device
    step = 0
    for start in range(0, len(order), effective):
        group = order[start:start+effective]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for offset in range(0, len(group), micro_batch):
            ids = group[offset:offset+micro_batch]
            batch = collate([examples[i] for i in ids], pad_id, device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits
            counts = (batch["labels"][:, 1:] != -100).sum(-1)
            loss = (-completion_log_probs(logits, batch["labels"]) / counts).sum() / len(group)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            total += float(loss.detach())
        grad_norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if callback:
            callback({"step": step, "example_indices": group, "mean_completion_token_nll": total,
                      "gradient_norm_before_clip": float(grad_norm)})
    return step
