"""Fresh, sealed, next-token assays. No weight updates or automatic expansion."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import os

from .fixtures import BUILDERS, MODEL, REVISION, settings


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def seal(root):
    write(root / "MANIFEST.json", {"files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()}})


def validate(root):
    expected = json.loads((root / "MANIFEST.json").read_text())["files"]
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file() and p != root / "MANIFEST.json"}
    if actual != expected:
        raise ValueError("evidence inventory/checksum mismatch")


def prepare(study, root):
    root.mkdir(parents=True, exist_ok=False)
    cases, answers = BUILDERS[study]()
    write(root / "cases.json", cases)
    write(root / "private_answer_key.json", answers)
    write(root / "settings.json", settings(study))
    seal(root)
    return {"study": study, "full_forwards": len(cases), "smoke_forwards": sum(c["smoke"] for c in cases)}


def inputs(root):
    validate(root)
    spec = json.loads((root / "settings.json").read_text())
    study = spec["study"]
    cases, key = BUILDERS[study]()
    # JSON round trip normalizes tuples to lists before checking exact frozen data.
    expected = json.loads(json.dumps({"cases": cases, "key": key}))
    if (spec != settings(study) or json.loads((root / "cases.json").read_text()) != expected["cases"]
            or json.loads((root / "private_answer_key.json").read_text()) != expected["key"]):
        raise ValueError("prepared inputs differ from deterministic source")
    return spec, cases, key


def select(cases, mode):
    if mode not in ("smoke", "full"):
        raise ValueError("invalid mode")
    return [c for c in cases if mode == "full" or c["smoke"]]


def run(prepared, output, mode):
    spec, cases, _ = inputs(prepared)
    cases = select(cases, mode)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(prepared, output / "inputs")
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    write(output / "plan.json", {"mode": mode, "case_ids": [c["case_id"] for c in cases], "settings": spec})
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import HfApi
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU tests do not run a scientific model")
        if HfApi().model_info(MODEL, revision=REVISION).sha != REVISION:
            raise ValueError("model revision mismatch")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in "ABCD"}
        if any(len(value) != 1 for value in ids.values()):
            raise ValueError("choice interface is not single-token")
        encoded, audit = {}, []
        for case in cases:
            text = tokenizer.apply_chat_template(case["messages"], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            batch = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            length = batch.input_ids.numel()
            if length > spec["max_context_tokens"]:
                raise ValueError("context exceeds frozen budget; no truncation allowed")
            encoded[case["case_id"]] = batch
            audit.append({"case_id": case["case_id"], "input_tokens": length,
                "token_ids_sha256": hashlib.sha256(json.dumps(batch.input_ids.tolist()).encode()).hexdigest()})
        write(output / "BUDGET_AUDIT.json", audit)
        # Reject invalid interfaces or oversized prompts before allocating model weights.
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True).eval()
        device = model.get_input_embeddings().weight.device
        write(output / "runtime.json", {"model": MODEL, "revision": REVISION,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "choice_ids": ids, "weight_updates": 0})
        with torch.inference_mode(), (output / "raw.jsonl").open("x", encoding="utf-8") as f:
            for case in cases:
                batch = encoded[case["case_id"]].to(device)
                torch.cuda.synchronize()
                start = time.monotonic()
                logits = model(**batch, use_cache=False).logits[0, -1].float()
                logp = torch.log_softmax(logits, dim=-1)
                choices = torch.tensor([ids[a][0] for a in case["choices"]], device=device)
                cp = torch.softmax(logp[choices], dim=-1)
                torch.cuda.synchronize()
                row = {"case_id": case["case_id"], "log_prob": logp[choices].cpu().tolist(),
                    "choice_probability": cp.cpu().tolist(), "choice_mass": float(logp[choices].exp().sum()),
                    "predicted": case["choices"][int(cp.argmax())],
                    "unrestricted_greedy_token": tokenizer.decode([int(logp.argmax())]),
                    "input_tokens": batch.input_ids.numel(), "seconds": time.monotonic()-start}
                f.write(json.dumps(row, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        inputs(output / "inputs")
        write(output / "COMPLETE.json", {"records": len(cases), "weight_updates": 0})
    except Exception as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        seal(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("prepare", "run"))
    p.add_argument("--study", choices=tuple(BUILDERS))
    p.add_argument("--prepared", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = p.parse_args()
    if args.command == "prepare":
        if args.study is None:
            p.error("prepare requires --study")
        print(json.dumps(prepare(args.study, args.output)))
    else:
        if args.prepared is None:
            p.error("run requires --prepared")
        run(args.prepared, args.output, args.mode)
