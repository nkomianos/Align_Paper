"""One explicit LC0 run; no training, no remote actions, no automatic expansion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from .channel import (ARMS, MODEL, REVISION, SETTINGS, alignment, plan, prepare,
                      receiver_text, seal, sender_text, validate_inputs, write)


def run(prepared, output, mode="smoke"):
    validate_inputs(prepared)
    if json.loads((prepared / "settings.json").read_text()) != SETTINGS:
        raise ValueError("prepared settings mismatch")
    all_cases = json.loads((prepared / "cases.json").read_text())
    cases = plan(all_cases, mode)
    if len(cases) != (8 if mode == "smoke" else 128):
        raise ValueError("frozen workload mismatch")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(prepared, output / "inputs")
    shutil.copytree(Path(__file__).parent, output / "source", ignore=shutil.ignore_patterns("__pycache__"))
    write(output / "plan.json", {"mode": mode, "case_ids": [c["case_id"] for c in cases],
                                "arms": list(ARMS), "receiver_forwards": len(cases) * len(ARMS),
                                "sender_prefills": len(cases), "settings": SETTINGS})
    try:
        import torch
        import transformers
        from huggingface_hub import HfApi
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for real LC0 inference")
        if HfApi().model_info(MODEL, revision=REVISION).sha != REVISION:
            raise ValueError("immutable model revision mismatch")
        torch.manual_seed(SETTINGS["seed"])
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16,
            device_map={"": torch.cuda.current_device()}, low_cpu_mem_usage=True).eval()
        embedding = model.get_input_embeddings()
        device = embedding.weight.device
        write(output / "runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "model": MODEL, "revision": REVISION,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "state_location": "backbone_final_normalized_hidden_state",
            "sender_encoding": "teacher_forced_relation_no_target_query_no_generation",
            "bridge_reference": "https://arxiv.org/abs/2608.13317",
            "reproduction_claim": False})
        def ids(text):
            return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        def context(case):
            marker = "LC0_CHANNEL_SLOT_81ca2"
            chat = tokenizer.apply_chat_template([
                {"role": "system", "content": "Answer the relation query using supplied information. Output one answer letter only."},
                {"role": "user", "content": marker + receiver_text(case)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if chat.count(marker) != 1:
                raise ValueError("communication insertion marker mismatch")
            before, after = chat.split(marker)
            return ids(before), ids(after)
        contexts = {c["case_id"]: context(c) for c in cases}
        token_ids = {c["case_id"]: ids(sender_text(c)) for c in cases}
        audit = []
        for case in cases:
            cid = case["case_id"]
            other = cid.rsplit("/", 1)[0] + ("/counterfactual" if cid.endswith("/original") else "/original")
            if token_ids[cid].shape != token_ids[other].shape:
                raise ValueError("counterfactual prefix token budgets differ")
            if any(not torch.equal(a, b) for a, b in zip(contexts[cid], contexts[other])):
                raise ValueError("receiver context differs across counterfactual worlds")
            total = token_ids[cid].numel() + sum(t.numel() for t in contexts[cid])
            if total > SETTINGS["max_context"]:
                raise ValueError("context budget exceeded; no truncation allowed")
            audit.append({"case_id": cid, "message_tokens": token_ids[cid].numel(),
                          "input_tokens_with_message": total,
                          "input_tokens_without_message": total - token_ids[cid].numel()})
        write(output / "BUDGET_AUDIT.json", audit)
        (output / "prefixes").mkdir()
        # Reusable per-world payload; no hidden labels, target query, or receiver facts.
        payloads = {}
        with torch.inference_mode():
            for case in cases:
                cid = case["case_id"]
                start = time.monotonic()
                tokens = token_ids[cid]
                hidden = model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                     use_cache=False, return_dict=True).last_hidden_state[0]
                text = embedding(tokens)[0]
                latent = alignment(hidden, text, embedding.weight, SETTINGS["reg"], SETTINGS["snap"])
                norm_text = text.float() / text.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                norm_text = (norm_text * embedding.weight.float().norm(dim=-1).mean()).to(text.dtype)
                payloads[cid] = {"text": text.cpu(), "norm_text": norm_text.cpu(), "latent": latent.cpu()}
                torch.cuda.synchronize()
                target = output / "prefixes" / (cid.replace("/", "__") + ".pt")
                torch.save({"tokens": tokens.cpu(), "hidden": hidden.cpu(), **payloads[cid],
                            "encoding_and_alignment_seconds": time.monotonic() - start}, target)
            with (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
                for case in cases:
                    cid = case["case_id"]
                    other = cid.rsplit("/", 1)[0] + ("/counterfactual" if cid.endswith("/original") else "/original")
                    for arm in ARMS:
                        left, right = (embedding(tokens) for tokens in contexts[cid])
                        if arm == "no_message":
                            combined = torch.cat((left, right), dim=1)
                        else:
                            source = other if arm.startswith("counterfactual_") else cid
                            kind = arm.removeprefix("counterfactual_")
                            prefix = payloads[source][kind].to(device).unsqueeze(0)
                            combined = torch.cat((left, prefix, right), dim=1)
                        mask = torch.ones(combined.shape[:2], device=device, dtype=torch.long)
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        generated = model.generate(inputs_embeds=combined, attention_mask=mask,
                            max_new_tokens=SETTINGS["max_new_tokens"], do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
                        torch.cuda.synchronize()
                        # inputs_embeds generation returns newly generated IDs, not prompt IDs.
                        if generated.shape[1] > SETTINGS["max_new_tokens"]:
                            raise ValueError("unexpected inputs_embeds generation output contract")
                        record = {"case_id": cid, "arm": arm,
                            "completion": tokenizer.decode(generated[0], skip_special_tokens=True),
                            "generated_ids": generated[0].tolist(), "seconds": time.monotonic() - start,
                            "input_tokens": combined.shape[1]}
                        stream.write(json.dumps(record) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
        validate_inputs(output / "inputs")
        write(output / "COMPLETE.json", {"records": len(cases) * len(ARMS), "updates_trained": 0})
    except Exception as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        seal(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output)
    else:
        if args.prepared is None:
            parser.error("run requires --prepared")
        run(args.prepared, args.output, args.mode)
