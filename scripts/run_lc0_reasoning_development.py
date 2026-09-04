"""Fresh DEV communication test after identifying a reasoning-budget confound.

Not a repeat or replacement of LC0's original frozen gate. No model updates.
Uses eight preselected unseen nonce pairs and a fixed 512-token reasoning budget.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from latent_contract.channel import ARMS, MODEL, REVISION, alignment, receiver_text, sender_text

SPEC = {"schema": "lc0_reasoning_development_v1", "pair_ids": list(range(4, 12)),
        "arms": list(ARMS), "max_new_tokens": 512, "enable_thinking": True,
        "reg": .001, "snap": .3, "model": MODEL, "revision": REVISION,
        "prepared_manifest_sha256": "3903c8b19e349c70f0ebb67ad599c00900f8ca81c2de32069f1827d37d063fbb",
        "scope": "developmental_channel_prerequisite_no_updates_not_paper_gate"}


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def final_answer(text):
    if "</think>" not in text:
        return None
    candidate = text.rsplit("</think>", 1)[1].strip()
    return candidate if candidate in ("A", "B", "C", "D") else None


def run(prepared, output):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert hashlib.sha256((prepared / "MANIFEST.json").read_bytes()).hexdigest() == SPEC["prepared_manifest_sha256"]
    # Validate public inputs against their frozen manifest without reading labels.
    manifest = json.loads((prepared / "MANIFEST.json").read_text())
    assert hashlib.sha256((prepared / "cases.json").read_bytes()).hexdigest() == manifest["files"]["cases.json"]
    cases = [c for c in json.loads((prepared / "cases.json").read_text()) if c["pair_id"] in SPEC["pair_ids"]]
    assert len(cases) == 16
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / "runner.py")
    write(output / "spec.json", SPEC)
    write(output / "cases.json", cases)
    try:
        assert torch.cuda.is_available()
        torch.manual_seed(91027)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}).eval()
        embedding = model.get_input_embeddings()
        device = model.device
        write(output / "runtime.json", {"torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(), "resolved_commit": model.config._commit_hash})
        def ids(text):
            return tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        def context(case):
            marker = "LC0_CHANNEL_SLOT_81ca2"
            chat = tokenizer.apply_chat_template([
                {"role": "system", "content": "Answer the relation query using supplied information. Output one answer letter only."},
                {"role": "user", "content": marker + receiver_text(case)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            assert chat.count(marker) == 1
            before, after = chat.split(marker)
            return ids(before), ids(after)
        contexts = {c["case_id"]: context(c) for c in cases}
        tokens = {c["case_id"]: ids(sender_text(c)) for c in cases}
        def other(cid):
            return cid.rsplit("/", 1)[0] + ("/counterfactual" if cid.endswith("/original") else "/original")
        audit = []
        for case in cases:
            cid = case["case_id"]
            assert tokens[cid].shape == tokens[other(cid)].shape
            assert all(torch.equal(a, b) for a, b in zip(contexts[cid], contexts[other(cid)]))
            total = tokens[cid].numel() + sum(t.numel() for t in contexts[cid])
            assert total <= 2048
            audit.append({"case_id": cid, "with_message": total,
                          "without_message": total-tokens[cid].numel()})
        write(output / "BUDGET_AUDIT.json", audit)
        (output / "prefixes").mkdir()
        payloads = {}
        with torch.inference_mode():
            for case in cases:
                cid = case["case_id"]
                hidden = model.model(input_ids=tokens[cid], attention_mask=torch.ones_like(tokens[cid]),
                                     use_cache=False, return_dict=True).last_hidden_state[0]
                text = embedding(tokens[cid])[0]
                latent = alignment(hidden, text, embedding.weight, SPEC["reg"], SPEC["snap"])
                norm = text.float() / text.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
                norm = (norm * embedding.weight.float().norm(dim=-1).mean()).to(text.dtype)
                payloads[cid] = {"text": text.cpu(), "norm_text": norm.cpu(), "latent": latent.cpu()}
                torch.save({"tokens": tokens[cid].cpu(), "hidden": hidden.cpu(), **payloads[cid]},
                           output / "prefixes" / (cid.replace("/", "__") + ".pt"))
            with (output / "raw.jsonl").open("x", encoding="utf-8") as stream:
                for case in cases:
                    cid = case["case_id"]
                    for arm in ARMS:
                        left, right = (embedding(t) for t in contexts[cid])
                        if arm == "no_message":
                            combined = torch.cat((left, right), dim=1)
                        else:
                            source = other(cid) if arm.startswith("counterfactual_") else cid
                            kind = arm.removeprefix("counterfactual_")
                            combined = torch.cat((left, payloads[source][kind].to(device).unsqueeze(0), right), dim=1)
                        mask = torch.ones(combined.shape[:2], device=device, dtype=torch.long)
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        generated = model.generate(inputs_embeds=combined, attention_mask=mask,
                            do_sample=False, max_new_tokens=SPEC["max_new_tokens"], pad_token_id=tokenizer.eos_token_id)[0]
                        torch.cuda.synchronize()
                        assert len(generated) <= SPEC["max_new_tokens"]
                        completion = tokenizer.decode(generated, skip_special_tokens=True)
                        row = {"case_id": cid, "arm": arm, "completion": completion,
                               "final_answer": final_answer(completion), "generated_ids": generated.tolist(),
                               "input_tokens": combined.shape[1],
                               "hit_token_limit": len(generated) == SPEC["max_new_tokens"],
                               "seconds": time.monotonic()-start}
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
        write(output / "COMPLETE.json", {"records": 96, "updates_trained": 0})
    except BaseException as exc:
        write(output / "FAILED.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write(output / "MANIFEST.json", {"files": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob("*") if p.is_file()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prepared, args.output)
