"""Independent BERT port of seed-KV override; not COVER decoder evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from types import MethodType

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import transformers

ARMS = ["native", "plain"] + [f"{candidate}_{mode}" for candidate in ("gold", "wrong")
                               for mode in ("uncorrected", "diagonal", "first", "last")]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


class Instrument:
    def __init__(self, model):
        self.modules = [layer.attention.self for layer in model.bert.encoder.layer]
        self.original = [m.forward for m in self.modules]
        self.mode, self.position, self.cache = "plain", 0, None
        self.saved = []

    def __enter__(self):
        for index, module in enumerate(self.modules):
            def run(module, hidden_states, attention_mask=None, past_key_values=None, _index=index, **kwargs):
                assert past_key_values is None
                assert not module.training
                b, t, _ = hidden_states.shape
                shape = (b, t, -1, module.attention_head_size)
                q = module.query(hidden_states).view(shape).transpose(1, 2)
                k = module.key(hidden_states).view(shape).transpose(1, 2)
                v = module.value(hidden_states).view(shape).transpose(1, 2)
                if self.mode == "collect":
                    self.saved.append((k.detach().clone(), v.detach().clone()))
                override = (self.mode in ("uncorrected", "diagonal") or
                            (self.mode == "first" and _index == 0) or
                            (self.mode == "last" and _index == len(self.modules)-1))
                current_k, current_v = k[:, :, self.position].clone(), v[:, :, self.position].clone()
                if override:
                    k, v = k.clone(), v.clone()
                    k[:, :, self.position] = self.cache[_index][0][:, :, self.position]
                    v[:, :, self.position] = self.cache[_index][1][:, :, self.position]
                scores = q @ k.transpose(-1, -2) * module.scaling
                if attention_mask is not None:
                    scores = scores + attention_mask
                weights = scores.softmax(-1)
                output = weights @ v
                if override and self.mode != "uncorrected":
                    # Direct row recomputation avoids relying on the post-hoc
                    # algebra. Other queries still see the overridden column.
                    row = scores[:, :, self.position:self.position+1, :].clone()
                    own_score = (q[:, :, self.position] * current_k).sum(-1) * module.scaling
                    if attention_mask is not None:
                        own_score += attention_mask[..., self.position].reshape(b, -1)[:, :1]
                    row[:, :, 0, self.position] = own_score
                    rw = row.softmax(-1)
                    fixed = rw @ v
                    fixed += rw[..., self.position:self.position+1] * (current_v - v[:, :, self.position]).unsqueeze(-2)
                    output[:, :, self.position:self.position+1] = fixed
                    weights[:, :, self.position:self.position+1] = rw
                output = output.transpose(1, 2).reshape(b, t, -1).contiguous()
                return output, weights
            module.forward = MethodType(run, module)
        return self

    def __exit__(self, *_):
        for module, original in zip(self.modules, self.original):
            module.forward = original


def tokenize_case(tokenizer, case):
    encoded = tokenizer(case["text"], return_tensors="pt")
    positions = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError("Exactly one mask required")
    labels = {name: tokenizer.encode(case[name], add_special_tokens=False) for name in ("gold", "wrong")}
    if any(len(ids) != 1 for ids in labels.values()) or labels["gold"] == labels["wrong"]:
        raise ValueError("Labels must be distinct single tokens")
    return encoded, int(positions[0]), {name: ids[0] for name, ids in labels.items()}


def summarize(logits, rows, config):
    assert logits.shape[:2] == (len(config["cases"]), len(ARMS))
    assert len(rows) == len(config["cases"])
    assert [r["id"] for r in rows] == [c["id"] for c in config["cases"]]
    assert np.isfinite(logits).all()
    gold = np.array([r["gold_id"] for r in rows])
    wrong = np.array([r["wrong_id"] for r in rows])
    pred = logits.argmax(-1)
    qualified = pred[:, 0] == gold
    result = {"n": len(rows), "clean_correct": int(qualified.sum()), "arms": {}}
    for index, arm in enumerate(ARMS):
        z = logits[:, index].astype(np.float64)
        p = np.exp(z-z.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        result["arms"][arm] = {
            "correct": int((pred[:, index] == gold).sum()),
            "wrong_retained": int((pred[:, index] == wrong).sum()),
            "competent_correct": int(((pred[:, index] == gold) & qualified).sum()),
            "competent_wrong_retained": int(((pred[:, index] == wrong) & qualified).sum()),
            "mean_wrong_probability": float(p[np.arange(len(rows)), wrong].mean()),
        }
    result["native_plain_max_logit_error"] = float(np.abs(logits[:, 0]-logits[:, 1]).max())
    result["last_only_max_logit_error"] = float(max(np.abs(logits[:, ARMS.index(c+"_last")]-logits[:, 0]).max()
                                                       for c in ("gold", "wrong")))
    result["diagonal_candidate_top1_changes"] = int((pred[:, ARMS.index("gold_diagonal")] != pred[:, ARMS.index("wrong_diagonal")]).sum())
    tol = config["equivalence_tolerance"]
    assert result["native_plain_max_logit_error"] < tol
    assert result["last_only_max_logit_error"] < tol
    assert np.array_equal(pred[:, 0], pred[:, 1])
    result["decision"] = ("INTERPRETABLE_DEVELOPMENTAL_AUDIT_NO_DIFFUSION_OR_PAPER_GO"
                          if qualified.sum() >= config["minimum_clean_correct_for_interpretation"]
                          else "INSUFFICIENT_CLEAN_CAPABILITY_FOR_PRACTICAL_INTERPRETATION")
    return result


def run(config_path, model_path, root):
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert model_path.name == config["revision"]
    (root / "config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    torch.set_num_threads(2)
    torch.manual_seed(7401)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForMaskedLM.from_pretrained(model_path, local_files_only=True, attn_implementation="eager").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    assert model.config.model_type == "bert"
    encoded_cases = [tokenize_case(tokenizer, case) for case in config["cases"]]
    output, metadata = [], []
    start = time.perf_counter()
    calls = 0
    with torch.inference_mode():
        for case, (encoded, pos, labels) in zip(config["cases"], encoded_cases):
            native = model(**encoded).logits[0, pos].float().numpy().copy(); calls += 1
            item = [native]
            with Instrument(model) as inst:
                inst.position = pos
                item.append(model(**encoded).logits[0, pos].float().numpy().copy()); calls += 1
                for candidate in ("gold", "wrong"):
                    draft = {k: v.clone() for k, v in encoded.items()}
                    draft["input_ids"][0, pos] = labels[candidate]
                    inst.mode, inst.saved = "collect", []
                    model(**draft); calls += 1
                    inst.cache = inst.saved
                    assert len(inst.cache) == model.config.num_hidden_layers
                    for mode in ("uncorrected", "diagonal", "first", "last"):
                        inst.mode = mode
                        item.append(model(**encoded).logits[0, pos].float().numpy().copy()); calls += 1
            output.append(np.stack(item))
            row = {"id": case["id"], "position": pos, "input_ids": encoded["input_ids"][0].tolist(),
                   "gold_id": labels["gold"], "wrong_id": labels["wrong"],
                   "top_tokens": {arm: tokenizer.decode([int(logit.argmax())]) for arm, logit in zip(ARMS, item)}}
            metadata.append(row)
            with (root / "cases.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(row)+"\n")
    logits = np.stack(output)
    np.save(root / "logits.npy", logits, allow_pickle=False)
    summary = summarize(logits, metadata, config)
    runtime = {"forward_calls": calls, "seconds": time.perf_counter()-start, "device": "cpu",
               "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
               "model": config["model"], "revision": config["revision"], "model_path": str(model_path.resolve()),
               "model_files": {p.name: sha(p) for p in sorted(model_path.iterdir()) if p.is_file()},
               "source_sha256": sha(__file__), "arms": ARMS}
    for name, value in (("summary.json", summary), ("runtime.json", runtime)):
        (root/name).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    manifest = {p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file()}
    (root/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return summary


def verify(root):
    manifest = json.loads((root/"MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"config.json", "cases.jsonl", "logits.npy", "summary.json", "runtime.json"}
    assert all(sha(root/name) == value for name, value in manifest.items())
    runtime = json.loads((root/"runtime.json").read_text(encoding="utf-8"))
    assert runtime["source_sha256"] == sha(__file__)
    assert runtime["arms"] == ARMS and runtime["device"] == "cpu"
    model_path = Path(runtime["model_path"])
    assert all(sha(model_path/name) == value for name, value in runtime["model_files"].items())
    config = json.loads((root/"config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (root/"cases.jsonl").read_text(encoding="utf-8").splitlines()]
    assert runtime["forward_calls"] == len(rows)*12
    summary = summarize(np.load(root/"logits.npy", allow_pickle=False), rows, config)
    assert summary == json.loads((root/"summary.json").read_text(encoding="utf-8"))
    return {"verified": True, "scope": "byte_integrity_and_metric_recomputation_not_full_model_replay",
            "manifest_sha256": sha(root/"MANIFEST.json"), "summary": summary}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["run", "verify"])
    p.add_argument("root", type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/cache_verification_cloze_v1.json"))
    p.add_argument("--model-path", type=Path)
    args = p.parse_args()
    print(json.dumps(run(args.config, args.model_path, args.root) if args.command == "run" else verify(args.root), indent=2))
