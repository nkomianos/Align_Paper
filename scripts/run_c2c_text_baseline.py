"""Pinned text-transfer comparator for the preserved C2C DEV experiment.

No model update. Two serial generations per question, using the upstream
TwoStageInference prompt structure without its unpinned model loader.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.run_c2c_baseline_dev import SPEC as BASE_SPEC

SPEC = {**BASE_SPEC, "scope": "MATCHED_TEXT_TRANSFER_DEV_NO_UPDATES",
        "arms": ["background", "answer_with_background"],
        "background_prompt": "Briefly describe the most useful background to solve the problem:\n\n{question}",
        "source_methods": "TwoStageInference.get_background_context and answer_with_context",
        "calls": 256}


def messages(question, prompt, background=None):
    query = SPEC["background_prompt"].format(question=question)
    if background is None:
        return [{"role": "user", "content": query}]
    return [{"role": "user", "content": query}, {"role": "assistant", "content": background},
            {"role": "user", "content": prompt}]


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def run(upstream, cases_path, output):
    output.mkdir(parents=True, exist_ok=False)
    try:
        write(output / "spec.json", SPEC)
        (output / "runner.py").write_bytes(Path(__file__).read_bytes())
        commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True)
        if commit != SPEC["upstream_commit"] or dirty:
            raise ValueError("upstream not pinned and clean")
        raw = cases_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SPEC["cases_sha256"]:
            raise ValueError("case digest mismatch")
        cases = json.loads(raw)
        (output / "cases.json").write_bytes(raw)
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if torch.__version__ != SPEC["torch"] or transformers.__version__ != SPEC["transformers"]:
            raise ValueError("dependency mismatch")
        torch.manual_seed(20260904)
        sys.path.insert(0, str(upstream.resolve()))
        from rosetta.utils.evaluate import build_prompt
        models, tokenizers = {}, {}
        for role in ("sender", "receiver"):
            kwargs = {"revision": SPEC[role + "_revision"], "local_files_only": True, "trust_remote_code": False}
            tokenizers[role] = AutoTokenizer.from_pretrained(SPEC[role], **kwargs)
            models[role] = AutoModelForCausalLM.from_pretrained(SPEC[role], **kwargs, torch_dtype=torch.bfloat16,
                              device_map={"": "cuda:0"}, attn_implementation="sdpa").eval()
            if models[role].config._commit_hash != SPEC[role + "_revision"]:
                raise ValueError("model revision mismatch")
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
              "models": {r: m.config._commit_hash for r, m in models.items()}, "device": torch.cuda.get_device_name(0)})
        count = 0
        with torch.inference_mode(), (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
            for case in cases:
                choices = "".join(f"{chr(65+i)}. {text}\n" for i, text in enumerate(case["choices"]))
                prompt = build_prompt("mmlu-redux", "", case["question"], choices, False, True)
                background = None
                for role, arm in (("sender", "background"), ("receiver", "answer_with_background")):
                    turns = messages(case["question"], prompt, background)
                    ids = tokenizers[role].apply_chat_template(turns, tokenize=True, add_generation_prompt=True,
                               enable_thinking=False, return_tensors="pt").to("cuda:0")
                    if ids.shape[1] > SPEC["max_input_tokens"]:
                        raise ValueError("input budget exceeded")
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    generated = models[role].generate(ids, do_sample=False, max_new_tokens=SPEC["max_new_tokens"])[0, ids.shape[1]:]
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - start
                    completion = tokenizers[role].decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    row = {"case_id": case["case_id"], "dataset": case["dataset"], "arm": arm, "messages": turns,
                           "input_ids": ids[0].tolist(), "generated_ids": generated.tolist(), "completion": completion,
                           "seconds": elapsed, "hit_token_limit": len(generated) == SPEC["max_new_tokens"]}
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    count += 1
                    background = completion
        write(output / "COMPLETE.json", {"calls": count, "updates": 0})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.iterdir() if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("upstream", "cases", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.upstream, args.cases, args.output)
